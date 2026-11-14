"""Deterministic policy evaluation for continuous_drift_iwd.

Loads an rl_games checkpoint, rebuilds the actor (plain MLP, or GRU + MLP for
the recurrent policies trained with ``--rnn``) and runs the environment with
many parallel agents.  Reports drift tracking quality, episode outcomes and
failure reasons.  No rl_games player, so evaluation is fast.

    cd experiments/continuous_drift_iwd
    python eval.py ../../runs/continuous_drift_iwd_<exp>/nn/continuous_drift_iwd.pth \
        --car-preset fullsize_racecar_ev --ref-mode circle --n 256 --steps 600
"""
import argparse
import json
import os
import glob
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))
sys.path.insert(0, os.path.join(ROOT, "utils"))

from envs.continuous_drift_iwd import ContinuousDriftIWDEnv


class Policy:
    """Actor rebuilt from an rl_games checkpoint (MLP or GRU + MLP)."""

    def __init__(self, path, device):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model"] if "model" in ck else ck
        w = {k.replace("_orig_mod.", ""): v.float() for k, v in sd.items()}
        self.mean = w["running_mean_std.running_mean"].to(device)
        self.var = w["running_mean_std.running_var"].to(device)
        self.mirror_core = None
        if "a2c_network.core.gru.weight_ih_l0" in w:
            from utils.mirror_policy import SharedMirrorActor
            self.mirror_core = SharedMirrorActor().to(device)
            prefix = "a2c_network.core."
            self.mirror_core.load_state_dict({k[len(prefix):]: v for k, v in w.items()
                                             if k.startswith(prefix)}, strict=True)
            self.mirror_core.eval()
            self.action_std = self.mirror_core.logstd.detach().exp()
            self.gru = self.mirror_core.gru
            self.h = None
            self._mirror_beta = None
            self.observation_bypass = True
            return
        self.layers = []
        i = 0
        while f"a2c_network.actor_mlp.{i}.weight" in w:
            self.layers.append((w[f"a2c_network.actor_mlp.{i}.weight"].to(device),
                                w[f"a2c_network.actor_mlp.{i}.bias"].to(device)))
            i += 2
        self.mu = (w["a2c_network.mu.weight"].to(device), w["a2c_network.mu.bias"].to(device))
        self.action_std = w["a2c_network.sigma"].exp().to(device)
        rnn_keys = [k for k in w if k.startswith("a2c_network.rnn.rnn.weight_ih")]
        self.gru = None
        if rnn_keys:
            input_size = w["a2c_network.rnn.rnn.weight_ih_l0"].shape[1]
            hidden = w["a2c_network.rnn.rnn.weight_hh_l0"].shape[1]
            self.gru = torch.nn.GRU(input_size, hidden, num_layers=1).to(device)
            with torch.no_grad():
                self.gru.weight_ih_l0.copy_(w["a2c_network.rnn.rnn.weight_ih_l0"])
                self.gru.weight_hh_l0.copy_(w["a2c_network.rnn.rnn.weight_hh_l0"])
                self.gru.bias_ih_l0.copy_(w["a2c_network.rnn.rnn.bias_ih_l0"])
                self.gru.bias_hh_l0.copy_(w["a2c_network.rnn.rnn.bias_hh_l0"])
            self.gru.eval()
        self.h = None
        self.beta_head_weights = None
        if "a2c_network.beta_head.0.weight" in w:
            self.beta_head_weights = tuple(w[f"a2c_network.beta_head.{i}.{kind}"].to(device)
                                           for i in (0, 2) for kind in ("weight", "bias"))
        self.observation_bypass = (self.gru is not None and
            self.layers[0][0].shape[1] == self.gru.hidden_size + self.mean.numel())
        if self.gru is not None and self.layers[0][0].shape[1] not in (
                self.gru.hidden_size, self.gru.hidden_size + self.mean.numel()):
            raise ValueError("unsupported recurrent actor dimensions")

    def reset_hidden(self, mask):
        if self.gru is None:
            return
        if self.h is None:
            return
        if self.mirror_core is not None:
            self.h[:, :, mask, :] = 0.0
        else:
            self.h[:, mask, :] = 0.0

    @torch.no_grad()
    def estimate_beta(self):
        """Read-only auxiliary diagnostic after act(); never used by actions."""
        if self.mirror_core is not None:
            return self._mirror_beta
        if self.h is None or self.beta_head_weights is None:
            return None
        w0, b0, w1, b1 = self.beta_head_weights
        hidden = torch.nn.functional.elu(torch.nn.functional.linear(self.h[0], w0, b0))
        return torch.nn.functional.linear(hidden, w1, b1).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        if self.mirror_core is not None:
            mu, _, beta, self.h = self.mirror_core(
                obs.unsqueeze(0), self.mean, self.var, self.h)
            self._mirror_beta = beta[0, :, 0]
            return mu[0]
        x = (obs - self.mean) / torch.sqrt(self.var + 1e-5)
        x = torch.clamp(x, -5.0, 5.0)
        if self.gru is not None:
            current_obs = x
            if self.h is None:
                self.h = torch.zeros(1, x.shape[0], self.gru.hidden_size, device=x.device)
            out, self.h = self.gru(x.unsqueeze(0), self.h)
            x = out.squeeze(0)
            if self.observation_bypass:
                x = torch.cat((x, current_obs), dim=-1)
        for w, b in self.layers:
            x = torch.nn.functional.elu(torch.nn.functional.linear(x, w, b))
        return torch.nn.functional.linear(x, self.mu[0], self.mu[1])


def physical_action(raw, env):
    # rl_games clips the raw action to [-1, 1] and then rescales it affinely onto
    # the declared action space.  The evaluator used to feed raw values straight
    # in, so a raw steering of 0.2 became 0.2 rad here but 0.09 rad in training.
    raw = raw.clamp(-1.0, 1.0)
    low = torch.as_tensor(env.action_space.low, device=raw.device, dtype=raw.dtype)
    high = torch.as_tensor(env.action_space.high, device=raw.device, dtype=raw.dtype)
    return low + 0.5 * (raw + 1.0) * (high - low)


def explicit_cli_dests(parser, argv):
    """Return argparse destinations explicitly present in ``argv``.

    Comparing a parsed value with its default is insufficient: an explicit
    ``--spawn-mode equilibrium`` equals the default but must still override a
    legacy value loaded from env_config.json.
    """
    tokens = [str(x) for x in argv]
    explicit = set()
    for action in parser._actions:
        for option in action.option_strings:
            if any(token == option or token.startswith(option + "=") for token in tokens):
                explicit.add(action.dest)
                break
    return explicit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--car-preset", default="fullsize_racecar_ev")
    ap.add_argument("--control-mode", default="pedals")
    ap.add_argument("--ref-mode", default="hybrid")
    ap.add_argument("--speed-target-mode", type=str, default="signal")
    ap.add_argument("--speed-margin", type=float, default=0.9)
    ap.add_argument("--speed-cap", type=float, default=None)
    ap.add_argument("--speed-weight", type=float, default=0.0)
    ap.add_argument("--prog-weight", type=float, default=0.2)
    # the actuator is part of the trained MDP: evaluating a servo-trained
    # policy without it measures a different task
    ap.add_argument("--steer-actuator-profile", choices=["legacy", "responsive"], default="legacy")
    ap.add_argument("--steer-actuator-tau", type=float, default=0.0)
    ap.add_argument("--steer-rate-limit", type=float, default=0.0)
    ap.add_argument("--steer-deadband", type=float, default=0.0)
    ap.add_argument("--brake-margin", type=float, default=0.0)
    ap.add_argument("--actuator-dr", action="store_true")
    ap.add_argument("--pedal-actuator-tau", type=float, default=0.0)
    ap.add_argument("--action-delay-max", type=int, default=2)
    ap.add_argument("--sensor-dr", action="store_true")
    ap.add_argument("--sensor-delay-max", type=int, default=2)
    ap.add_argument("--sensor-dropout-prob", type=float, default=0.002)
    ap.add_argument("--alpha-peak", type=float, default=0.12)
    ap.add_argument("--beta-max", type=float, default=0.32)
    ap.add_argument("--ackermann-lo", type=float, default=0.8)
    ap.add_argument("--ackermann-hi", type=float, default=1.2)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cpu-threads", type=int, default=1,
                    help="CPU evaluation thread budget while training runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic-diagnostic", action="store_true",
                    help="sample training action noise; not a deployment acceptance test")
    ap.add_argument("--save-scenario", type=str, default=None)
    ap.add_argument("--load-scenario", type=str, default=None)
    ap.add_argument("--reference-path", type=str, default=None)
    ap.add_argument("--json-out", type=str, default=None,
                    help="explicit path for machine-readable metrics; no file is written by default")
    ap.add_argument("--beta-scale", type=float, default=1.0)
    ap.add_argument("--offpath-scale", type=float, default=1.0)
    ap.add_argument("--course-fail", type=float, default=1.0)
    ap.add_argument("--stall-speed", type=float, default=2.0)
    ap.add_argument("--alive-bonus", type=float, default=0.0)
    ap.add_argument("--curv-scale", type=float, default=1.0)
    ap.add_argument("--seg-scale", type=float, default=1.0)
    ap.add_argument("--save-path-bank", type=str, default=None)
    ap.add_argument("--load-path-bank", type=str, default=None)
    ap.add_argument("--mirror-policy", action="store_true")
    ap.add_argument("--critic-velocity-frame", choices=["world", "body"], default="world")
    ap.add_argument("--config-from", type=str, default=None,
                    help="load the supported evaluator defaults from env_config.json; "
                         "fields that cannot be reconstructed are reported")
    ap.add_argument("--renewal-eval", action="store_true",
                    help="allow repeated respawns (biased); one episode per env is the default")
    ap.add_argument("--curv-smooth", type=int, default=0)
    ap.add_argument("--preview-scale", type=float, default=1.0)
    ap.add_argument("--reference-segments", type=int, default=2)
    ap.add_argument("--reference-transition-min-m", type=float, default=None)
    ap.add_argument("--preview-time", type=str, default=None)
    ap.add_argument("--preview-distances-m", type=str, default=None)
    ap.add_argument("--flip-prob", type=float, default=None)
    ap.add_argument("--stall-window", type=int, default=None)
    ap.add_argument("--stall-move", type=float, default=1.0)
    ap.add_argument("--spawn-speed-min", type=float, default=None)
    ap.add_argument("--spawn-deficit-frac", type=float, default=0.0)
    ap.add_argument("--yaw-weight", type=float, default=0.0)
    ap.add_argument("--yaw-scale", type=float, default=0.5)
    ap.add_argument("--beta-weight", type=float, default=1.6)
    ap.add_argument("--dir-weight", type=float, default=0.5)
    ap.add_argument("--pos-weight", type=float, default=2.4)
    ap.add_argument("--speed-target", type=float, default=None)
    ap.add_argument("--speed-hinge", action="store_true")
    ap.add_argument("--spawn-mode", default="equilibrium")
    ap.add_argument("--equilibrium-reset-prob", type=float, default=None,
                    help="per-respawn equilibrium-state probability; default follows --spawn-mode")
    ap.add_argument("--spawn-speed", type=float, default=6.5)
    ap.add_argument("--spawn-beta-mag", type=float, default=None,
                    help="legacy reset beta range; loaded from env_config when omitted")
    ap.add_argument("--legacy-zero-prob", type=float, default=None,
                    help="force the legacy reset mixture (0 dynamic only, 1 true rest)")
    ap.add_argument("--grip-scale", type=float, default=1.0,
                    help="global friction multiplier (evaluate on different surfaces)")
    ap.add_argument("--disturbed", action="store_true",
                    help="apply the OU disturbance the policy was trained with")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--physics-dt", type=float, default=0.01)
    ap.add_argument("--plant", default="fast")
    ap.add_argument("--asymmetric", action="store_true")
    ap.add_argument("--obs-privileged", action="store_true",
                    help="replay a non-deployable 31-channel observability upper bound")
    ap.add_argument("--beta-gate-scale", type=float, default=0.0)
    ap.add_argument("--beta-ref-mode", default="physics")
    ap.add_argument("--drift-util-lo", type=float, default=0.55)
    ap.add_argument("--drift-util-hi", type=float, default=0.90)
    ap.add_argument("--drift-curve-lo", type=float, default=0.02)
    ap.add_argument("--drift-curve-hi", type=float, default=0.06)
    ap.add_argument("--drift-curve-fade-lo", type=float, default=0.10)
    ap.add_argument("--drift-curve-fade-hi", type=float, default=0.15)
    ap.add_argument("--drift-mu-lo", type=float, default=1.0623)
    ap.add_argument("--drift-mu-hi", type=float, default=2.1246)
    ap.add_argument("--recover-weight", type=float, default=0.0)
    ap.add_argument("--kappa-weight", type=float, default=0.15)
    ap.add_argument("--kappa-cap", type=float, default=4.0)
    ap.add_argument("--pos-cap", type=float, default=1.0)
    ap.add_argument("--pp-weight", type=float, default=0.0)
    ap.add_argument("--yaw-cap", type=float, default=4.0)
    ap.add_argument("--num-paths", type=int, default=256)
    ap.add_argument("--closed-random-radius", type=float, default=1.0)
    ap.add_argument("--closed-switch-fraction", type=float, default=0.0)
    ap.add_argument("--closed-curvature-cap", type=float, default=None)
    ap.add_argument("--grip-bias", type=float, default=1.0)
    ap.add_argument("--drift-mode", action="store_true")
    ap.add_argument("--drift-beta-target", type=float, default=0.40)
    ap.add_argument("--drift-kappa-target", type=float, default=0.55)
    ap.add_argument("--drift-kappa-weight", type=float, default=60.0)
    ap.add_argument("--drift-throttle-gate", type=float, default=0.30)
    ap.add_argument("--drift-throttle-weight", type=float, default=4.0)
    ap.add_argument("--drift-counter-weight", type=float, default=8.0)
    ap.add_argument("--drift-joint-weight", type=float, default=12.0)
    ap.add_argument("--drift-corridor-deadband", type=float, default=0.6)
    ap.add_argument("--throttle-weight", type=float, default=0.0)
    ap.add_argument("--throttle-gate", type=float, default=0.25)
    ap.add_argument("--smooth-weight", type=float, default=0.015)
    ap.add_argument("--terminal-cost", type=float, default=100.0)
    ap.add_argument("--spawn-beta-frac", type=float, default=1.0)
    ap.add_argument("--drift-spawn-spin-frac", type=float, default=1.0)
    ap.add_argument("--drift-spawn-trim-frac", type=float, default=1.0)
    ap.add_argument("--continuing-task", action="store_true")
    ap.add_argument("--reward-mode", choices=("legacy", "bounded_joint", "adaptive_envelope"), default="legacy")
    ap.add_argument("--adaptive-alive", type=float, default=0.10)
    ap.add_argument("--adaptive-tracking", type=float, default=1.0)
    ap.add_argument("--adaptive-performance", type=float, default=3.0)
    ap.add_argument("--adaptive-shape", choices=["legacy", "smooth_v1", "smooth_task_v2"], default="legacy")
    ap.add_argument("--adaptive-speed-penalty", choices=["target", "envelope", "none"], default="target")
    ap.add_argument("--adaptive-speed-coupling", choices=["precision", "corridor", "independent_position"], default="precision")
    ap.add_argument("--adaptive-path-scale-m", type=float, default=None)
    ap.add_argument("--adaptive-drift", type=float, default=1.0)
    ap.add_argument("--adaptive-smooth", type=float, default=0.10)
    ap.add_argument("--adaptive-guidance", type=float, default=0.0)
    # Strict per-step powered-drift contract.  These are evaluation thresholds,
    # not reward knobs; a mean can no longer hide a long non-drift interval.
    ap.add_argument("--metric-throttle-min", type=float, default=0.15)
    ap.add_argument("--metric-kappa-min", type=float, default=0.25)
    ap.add_argument("--metric-beta-min", type=float, default=0.20)
    ap.add_argument("--metric-wheel-ratio-min", type=float, default=1.15)
    ap.add_argument("--metric-util-min", type=float, default=0.70)
    ap.add_argument("--metric-util-max", type=float, default=1.10)
    ap.add_argument("--metric-epos-max", type=float, default=1.50)
    args = ap.parse_args()
    _explicit_cli = explicit_cli_dests(ap, sys.argv[1:])
    if str(args.device).startswith("cpu"):
        torch.set_num_threads(max(1, args.cpu_threads))

    # reconstruct the training environment from the run directory when asked:
    # every parameter left at its default takes the value used in training, so an
    # evaluation cannot silently disagree with the MDP the policy saw.
    if args.config_from:
        import json
        _cfg_dir = args.config_from
        _cfg_path = os.path.join(_cfg_dir, "env_config.json")
        if not os.path.exists(_cfg_path):
            _cfg_path = os.path.join(os.path.dirname(os.path.abspath(_cfg_dir)), "env_config.json")
        if os.path.exists(_cfg_path):
            with open(_cfg_path) as _f:
                _cfg = json.load(_f)
            _applied = {}
            for _k, _v in _cfg.items():
                if (hasattr(args, _k) and _k not in _explicit_cli and
                        getattr(args, _k) == ap.get_default(_k)):
                    setattr(args, _k, _v)
                    _applied[_k] = _v
            print("env_config loaded:", len(_applied), "parameters")
            # Training-only stochastic settings are deliberately fixed for the
            # nominal deterministic evaluation.  --disturbed is the explicit
            # opt-in for the time-varying tyre disturbance.
            _skip = {"seq_length", "gamma", "aux_reward_decay_steps",
                     "aux_reward_coef", "train", "ref_mode_cli",
                     "disturbance_param", "randomize_param", "random_seed",
                     "quiet", "obs_privileged"}
            _unsupported = sorted(k for k in _cfg
                                  if not hasattr(args, k) and k not in _skip)
            if _unsupported:
                print("WARNING: env_config fields not reconstructed:",
                      ", ".join(_unsupported))
            if _cfg.get("ref_mode_cli") and "ref_mode" not in _explicit_cli:
                args.ref_mode = _cfg["ref_mode_cli"]
                _applied["ref_mode"] = args.ref_mode
        else:
            print("WARNING: no env_config.json in", _cfg_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.reference_path:
        args.ref_mode = "closed_random"
    env = ContinuousDriftIWDEnv(args.car_preset, args.n, args.device, ref_mode=args.ref_mode,
                                dt=args.dt, physics_dt=args.physics_dt, plant=args.plant,
                                control_mode=args.control_mode, beta_scale=args.beta_scale,
                                obs_privileged=args.obs_privileged,
                                spawn_mode=args.spawn_mode, spawn_speed=args.spawn_speed,
                                equilibrium_reset_prob=(
                                    args.equilibrium_reset_prob
                                    if args.equilibrium_reset_prob is not None
                                    else (1.0 if args.spawn_mode == "equilibrium" else 0.0)),
                                spawn_beta_mag=(args.spawn_beta_mag
                                                if args.spawn_beta_mag is not None else 0.15),
                                legacy_zero_prob=(args.legacy_zero_prob
                                                  if args.legacy_zero_prob is not None else 0.0),
                                offpath_scale=args.offpath_scale, course_fail=args.course_fail,
                                stall_speed=args.stall_speed, alive_bonus=args.alive_bonus,
                                curv_scale=args.curv_scale, seg_scale=args.seg_scale,
                                grip_scale=args.grip_scale,
                                disturbance_param=((0.97, 0.03) if args.disturbed else None),
                                # matching the training DR; the env forces tyre_scale
                                # to 1.0 whenever the disturbance is on, so a fixed
                                # --grip-scale cannot be applied in that mode
                                randomize_param=(({"B": [4.8, 7.2], "C": [1.7, 2.1],
                                                   "D": [float(args.grip_scale), float(args.grip_scale)]})
                                                  if args.disturbed else None),
                                curv_smooth=args.curv_smooth, preview_scale=args.preview_scale,
                                reference_segments=args.reference_segments,
                                reference_transition_min_m=args.reference_transition_min_m,
                                preview_time=args.preview_time,
                                preview_distances_m=args.preview_distances_m,
                                flip_prob=args.flip_prob,
                                stall_window=(args.stall_window if args.stall_window is not None
                                              else max(int(round(0.2 * (2.65 / 0.35) ** 0.5 / 0.05)), 2)),
                                stall_move=args.stall_move,
                                spawn_speed_min=(args.spawn_speed_min if args.spawn_speed_min is not None else args.spawn_speed),
                                spawn_deficit_frac=args.spawn_deficit_frac,
                                dir_weight=args.dir_weight, pos_weight=args.pos_weight,
                                 pos_cap=args.pos_cap, beta_gate_scale=args.beta_gate_scale,
                                 recover_weight=args.recover_weight,
                                 kappa_weight=args.kappa_weight, kappa_cap=args.kappa_cap,
                                 pp_weight=args.pp_weight, yaw_cap=args.yaw_cap,
                                # loaded from env_config.json but previously dropped:
                                beta_weight=args.beta_weight,
                                yaw_weight=args.yaw_weight, yaw_scale=args.yaw_scale,
                                # these three come from env_config.json rather than
                                # argparse, so they are read with a default instead of
                                # attribute access (which crashed the evaluator)
                                spawn_beta_frac=getattr(args, "spawn_beta_frac", 1.0),
                                drift_spawn_spin_frac=args.drift_spawn_spin_frac,
                                drift_spawn_trim_frac=args.drift_spawn_trim_frac,
                                continuing_task=args.continuing_task,
                                critic_velocity_frame=args.critic_velocity_frame,
                                reward_mode=args.reward_mode,
                                 adaptive_alive=args.adaptive_alive,
                                 adaptive_tracking=args.adaptive_tracking,
                                 adaptive_performance=args.adaptive_performance,
                                 adaptive_speed_penalty=args.adaptive_speed_penalty,
                                 adaptive_speed_coupling=args.adaptive_speed_coupling,
                                 adaptive_path_scale_m=args.adaptive_path_scale_m,
                                 adaptive_shape=args.adaptive_shape,
                                 adaptive_drift=args.adaptive_drift,
                                 adaptive_smooth=args.adaptive_smooth,
                                 adaptive_guidance=args.adaptive_guidance,
                                terminal_cost=getattr(args, "terminal_cost", 100.0),
                                smooth_weight=getattr(args, "smooth_weight", 0.015),
                                speed_weight=args.speed_weight,
                                speed_target=(args.speed_target if args.speed_target is not None else args.spawn_speed),
                                speed_hinge=args.speed_hinge,
                                speed_target_mode=args.speed_target_mode,
                                speed_margin=args.speed_margin,
                                speed_cap=(args.speed_cap if args.speed_cap is not None else float("inf")),
                                prog_weight=args.prog_weight,
                                alpha_peak=args.alpha_peak,
                                beta_max=args.beta_max,
                                ackermann_lo=args.ackermann_lo,
                                 ackermann_hi=args.ackermann_hi,
                                 beta_ref_mode=args.beta_ref_mode,
                                 drift_util_lo=args.drift_util_lo,
                                 drift_util_hi=args.drift_util_hi,
                                 drift_curve_lo=args.drift_curve_lo,
                                 drift_curve_hi=args.drift_curve_hi,
                                 drift_curve_fade_lo=args.drift_curve_fade_lo,
                                 drift_curve_fade_hi=args.drift_curve_fade_hi,
                                 drift_mu_lo=args.drift_mu_lo,
                                 drift_mu_hi=args.drift_mu_hi,
                                 asymmetric=args.asymmetric, num_paths=args.num_paths,
                                 closed_random_radius=args.closed_random_radius,
                                 closed_switch_fraction=getattr(args, "closed_switch_fraction", 0.0),
                                 closed_curvature_cap=args.closed_curvature_cap,
                                 grip_bias=args.grip_bias,
                                 drift_mode=args.drift_mode,
                                 drift_beta_target=args.drift_beta_target,
                                 drift_kappa_target=args.drift_kappa_target,
                                 drift_kappa_weight=args.drift_kappa_weight,
                                 drift_throttle_gate=args.drift_throttle_gate,
                                 drift_throttle_weight=args.drift_throttle_weight,
                                 drift_counter_weight=args.drift_counter_weight,
                                 drift_joint_weight=args.drift_joint_weight,
                                 drift_corridor_deadband=args.drift_corridor_deadband,
                                 throttle_weight=args.throttle_weight,
                                 throttle_gate=args.throttle_gate,
                                 steer_actuator_profile=args.steer_actuator_profile,
                                 steer_actuator_tau=args.steer_actuator_tau,
                                 steer_rate_limit=args.steer_rate_limit,
                                 steer_deadband=args.steer_deadband,
                                 brake_margin=args.brake_margin,
                                 actuator_dr=args.actuator_dr,
                                 pedal_actuator_tau=args.pedal_actuator_tau,
                                 action_delay_max=args.action_delay_max,
                                 sensor_dr=args.sensor_dr,
                                 sensor_delay_max=args.sensor_delay_max,
                                 sensor_dropout_prob=args.sensor_dropout_prob,
                                 train=True)
    # optional immutable held-out road bank: never regenerated, never used in
    # training, so checkpoint comparisons are paired across runs
    if args.load_path_bank:
        _pb = torch.load(args.load_path_bank, map_location=args.device)
        _nb = _pb.shape[1]
        env.path_bank[env.current_generation, :_nb].copy_(_pb[0].to(env.device))
        print("held-out bank loaded:", tuple(_pb.shape))
    if args.save_path_bank:
        torch.save(env.path_bank[env.current_generation:env.current_generation + 1].detach().cpu(),
                   args.save_path_bank)
        print("bank saved:", args.save_path_bank)
    obs = env.reset()
    # the environment only spawns on respawn, so force the first episode here
    if getattr(env, "train", False):
        env.is_done[:] = 1
        env.reset_done_envs()
        # the spawn above invalidated the observation, which still carried the
        # pre-spawn tracking errors
        obs = env.obs()
    # Evaluation is one physical episode.  Leaving train=True makes step()
    # respawn terminal environments before metrics are read, contaminating the
    # terminal sample with a synthetic new spawn.
    env.train = False
    if args.reference_path:
        if args.load_scenario or args.save_scenario:
            raise ValueError('Use either reference-path or scenario replay, not both')
        from experiments.continuous_drift_iwd.reference_fixture import install_reference
        obs = install_reference(env, args.reference_path)
    if args.load_scenario:
        _sc = np.load(args.load_scenario)
        obs = env.apply_scenario(_sc)
    elif args.save_scenario:
        np.savez(args.save_scenario, **env.capture_scenario())
        print("saved scenario", args.save_scenario)
        obs = env.obs()
    # Capture actual sampled settings without advancing sensors or consuming RNG.
    # Bound diagnostic output for large parallel evaluations.
    diagnostic_rows = min(args.n, 16)
    sensor_configuration = {
        "enabled": bool(env.sensor_dr),
        "recorded_rows": diagnostic_rows,
        "total_rows": args.n,
        "delay_steps": env._sensor_delay_steps[:diagnostic_rows].detach().cpu().tolist(),
        "noise_scale": env._sensor_noise_scale[:diagnostic_rows].detach().cpu().tolist(),
        "bias_scaled_packet_units": env._sensor_bias[:diagnostic_rows].detach().cpu().tolist(),
        "control_dt_s": float(env.dt),
        "dropout_probability": float(env.sensor_dropout_prob),
    }
    rolling = getattr(env, "_rolling_reference", None)
    reference_generation = ({k: v for k, v in rolling.options.items() if k != "generator"}
                            if rolling is not None else None)
    initial_conditions = {
        "state_rows": env.s[:diagnostic_rows].detach().cpu().tolist(),
        "state_definition": "CG x,y,heading,world vx,world vy,yawrate,wFL,wFR,wRL,wRR; SI",
        "max_speed_mps": float(env.es[:, 6].abs().max()),
        "max_abs_rear_power_slip": float(env.extras["kappa_power"][:, 2:4].abs().max()),
        "max_abs_trim": float(env.u_trim.abs().max()),
        "equilibrium_reset_prob": env.equilibrium_reset_prob,
        "legacy_zero_prob": env.legacy_zero_prob,
        "drift_spawn_spin_frac": env.drift_spawn_spin_frac,
    }
    ckpt = args.checkpoint
    if os.path.isdir(ckpt) or ckpt.endswith("nn"):
        # pick the newest per-epoch snapshot instead of the best-reward one
        cands = glob.glob(os.path.join(ckpt, "last_*.pth"))
        if not cands:
            cands = [os.path.join(ckpt, "continuous_drift_iwd.pth")]
        ckpt = max(cands, key=os.path.getmtime)
    policy = Policy(ckpt, torch.device(args.device))
    args.checkpoint = ckpt
    policy.reset_hidden(torch.ones(args.n, dtype=torch.bool, device=args.device))

    dev = torch.device(args.device)
    fails = torch.zeros(args.n, device=dev)
    succ = torch.zeros(args.n, device=dev)
    reason = [0, 0, 0]        # no_progress, large course error, off path
    # per-environment accumulators: the reported means are averages over
    # EPISODES, not over timesteps, so one long episode cannot dominate
    beta_sum = torch.zeros(args.n, device=dev)
    # the same quantity after a first-order low pass: the OU disturbance injects
    # sideslip at its own bandwidth, so the raw error mostly measures how much
    # of the disturbance leaks through, while the filtered error measures whether
    # the car holds the reference drift angle
    beta_f_sum = torch.zeros(args.n, device=dev)
    beta_filt = torch.zeros(args.n, device=dev)
    v_sum = torch.zeros(args.n, device=dev)
    n_sum = torch.zeros(args.n, device=dev)
    alive_hist = []
    # one-episode mode: every environment contributes exactly one episode and is
    # frozen at its first terminal state, so the statistics are not a renewal
    # process nor biased by censored re-spawns.
    ended = torch.zeros(args.n, dtype=torch.bool, device=dev)
    end_step = torch.full((args.n,), -1, device=dev)
    end_succ = torch.zeros(args.n, dtype=torch.bool, device=dev)
    end_reason = torch.zeros((args.n, 3), dtype=torch.bool, device=dev)
    strict_sum = torch.zeros(args.n, device=dev)
    strict_n = torch.zeros(args.n, device=dev)
    throttle_sum = torch.zeros(args.n, device=dev)
    brake_sum = torch.zeros(args.n, device=dev)
    brake_active_sum = torch.zeros(args.n, device=dev)
    kappa_sum = torch.zeros(args.n, device=dev)
    beta_abs_sum = torch.zeros(args.n, device=dev)
    beta_signed_sum = torch.zeros(args.n, device=dev)
    beta_ref_signed_sum = torch.zeros(args.n, device=dev)
    beta_ref_abs_sum = torch.zeros(args.n, device=dev)
    beta_ref_demand_005_sum = torch.zeros(args.n, device=dev)
    beta_ref_demand_015_sum = torch.zeros(args.n, device=dev)
    beta_ref_demand_030_sum = torch.zeros(args.n, device=dev)
    beta_aligned_sum = torch.zeros(args.n, device=dev)
    beta_sign_match_sum = torch.zeros(args.n, device=dev)
    beta_sign_valid_sum = torch.zeros(args.n, device=dev)
    counter_sum = torch.zeros(args.n, device=dev)
    wheel_ratio_sum = torch.zeros(args.n, device=dev)
    util_sum = torch.zeros(args.n, device=dev)
    tracking_sum = torch.zeros(args.n, device=dev)
    performance_sum = torch.zeros(args.n, device=dev)
    drift_active_sum = torch.zeros(args.n, device=dev)
    drift_quality_sum = torch.zeros(args.n, device=dev)
    # Keep the actual error distribution as well as thresholded duty cycles.
    # A policy sitting just inside the tracking gate must not look identical to
    # one that follows the centreline tightly.  At the default 256 x 600 this is
    # only about 1.2 MB per scalar metric on the CPU.
    epos_samples = []
    eyaw_samples = []
    demand_speed_samples = []
    demand_curvature_samples = []
    demand_util_samples = []
    demand_throttle_samples = []
    demand_kappa_samples = []
    demand_beta_samples = []
    demand_beta_error_samples = []
    demand_wheel_ratio_samples = []
    demand_counter_samples = []
    demand_tracking_samples = []
    demand_strict_samples = []
    demand_epos_samples = []
    prev_steer = None
    steer_step = []
    for step in range(args.steps):
        actor_obs = obs["obs"] if isinstance(obs, dict) else obs
        raw = policy.act(actor_obs)
        if args.stochastic_diagnostic:
            raw = raw + torch.randn_like(raw) * policy.action_std
        a = physical_action(raw, env)
        done_prev = env.is_done.clone()
        obs, reward, done, info = env.step(a)
        # the environment respawns inside step(), so ``env.is_done`` is already
        # cleared; the terminal flags of this step are the returned ``done``
        newly = (done_prev == 0) & (done != 0)
        term = done.clone()
        policy.reset_hidden(newly)
        sample_mask = ~ended
        if not args.renewal_eval:
            first = newly & ~ended
            if first.any():
                end_step[first] = step + 1
                end_succ[first] = (term[first] == 2)
                end_reason[first] = torch.stack([env._term_reason[first] == k for k in range(3)], dim=1)
                ended |= first
            # data from already finished environments is ignored
            live = ~ended
        else:
            live = torch.ones(args.n, dtype=torch.bool, device=dev)
            sample_mask = live
        # Per-step joint criterion, sampled from the same post-physics state.
        # Positive kappa_power always means driven-wheel overspeed on both plants.
        kp = env.extras["kappa_power"][:, 2:4].mean(dim=1)
        v_now = env.es[:, 6]
        beta_signed = env.es[:, 7]
        beta_now = beta_signed.abs()
        beta_ref_now = env.beta_ref
        wheel_v = env.s[:, 6:10] * env.wheel_radius
        wheel_ratio = wheel_v[:, 2:4].abs().mean(dim=1) / wheel_v[:, :2].abs().mean(dim=1).clamp_min(0.5)
        mu_now = (env.plant.col("D") * env.plant.tyre_scale.squeeze(1)).clamp_min(0.05)
        util = v_now.square() * env.kappa_ref.abs() / (mu_now * 9.81)
        counter = (env.u_cmd[:, 0] * env.kappa_ref < 0.0) & (env.u_cmd[:, 0].abs() >= 0.03)
        tracking = ((env.signed_pos_diff.abs() <= args.metric_epos_max) &
                    (env.angle_diff.abs() <= 0.35))
        speed_target = env._reference_speed()
        speed_ratio = v_now / speed_target.clamp_min(0.5)
        performance = tracking & (speed_ratio >= 0.70) & (speed_ratio <= 1.20)
        drift_active = ((env.u_cmd[:, 1] >= args.metric_throttle_min)
                        & (kp >= args.metric_kappa_min)
                        & (beta_now >= args.metric_beta_min)
                        & (wheel_ratio >= args.metric_wheel_ratio_min))
        drift_quality = drift_active & counter & tracking & performance
        strict = ((env.u_cmd[:, 1] >= args.metric_throttle_min)
                  & (kp >= args.metric_kappa_min)
                  & (beta_now >= args.metric_beta_min)
                  & (wheel_ratio >= args.metric_wheel_ratio_min)
                  & counter
                  & (util >= args.metric_util_min) & (util <= args.metric_util_max)
                  & (env.signed_pos_diff.abs() <= args.metric_epos_max))
        strict_sum[sample_mask] += strict[sample_mask].float()
        strict_n[sample_mask] += 1.0
        throttle_sum[sample_mask] += env.u_cmd[sample_mask, 1]
        brake_sum[sample_mask] += env.u_cmd[sample_mask, 2]
        brake_active_sum[sample_mask] += (env.u_cmd[sample_mask, 2] > .05).float()
        kappa_sum[sample_mask] += kp[sample_mask]
        beta_abs_sum[sample_mask] += beta_now[sample_mask]
        beta_signed_sum[sample_mask] += beta_signed[sample_mask]
        beta_ref_signed_sum[sample_mask] += beta_ref_now[sample_mask]
        beta_ref_abs_sum[sample_mask] += beta_ref_now[sample_mask].abs()
        beta_ref_demand_005_sum[sample_mask] += (
            beta_ref_now[sample_mask].abs() >= 0.05).float()
        beta_ref_demand_015_sum[sample_mask] += (
            beta_ref_now[sample_mask].abs() >= 0.15).float()
        beta_ref_demand_030_sum[sample_mask] += (
            beta_ref_now[sample_mask].abs() >= 0.30).float()
        beta_aligned_sum[sample_mask] += (
            beta_signed * torch.sign(beta_ref_now))[sample_mask]
        beta_sign_valid = beta_ref_now.abs() >= 0.05
        beta_sign_match_sum[sample_mask] += (
            ((beta_signed * beta_ref_now) > 0.0) & beta_sign_valid
        )[sample_mask].float()
        beta_sign_valid_sum[sample_mask] += beta_sign_valid[sample_mask].float()
        counter_sum[sample_mask] += counter[sample_mask].float()
        wheel_ratio_sum[sample_mask] += wheel_ratio[sample_mask]
        util_sum[sample_mask] += util[sample_mask]
        tracking_sum[sample_mask] += tracking[sample_mask].float()
        performance_sum[sample_mask] += performance[sample_mask].float()
        drift_active_sum[sample_mask] += drift_active[sample_mask].float()
        drift_quality_sum[sample_mask] += drift_quality[sample_mask].float()
        epos_samples.append(env.signed_pos_diff[sample_mask].abs().detach().cpu())
        eyaw_samples.append(env.angle_diff[sample_mask].abs().detach().cpu())
        demand_mask = sample_mask & (beta_ref_now.abs() >= 0.15)
        if demand_mask.any():
            demand_speed_samples.append(v_now[demand_mask].detach().cpu())
            demand_curvature_samples.append(env.kappa_ref[demand_mask].abs().detach().cpu())
            demand_util_samples.append(util[demand_mask].detach().cpu())
            demand_throttle_samples.append(env.u_cmd[demand_mask, 1].detach().cpu())
            demand_kappa_samples.append(kp[demand_mask].detach().cpu())
            demand_beta_samples.append(beta_now[demand_mask].detach().cpu())
            demand_beta_error_samples.append(
                (beta_signed[demand_mask] - beta_ref_now[demand_mask]).abs().detach().cpu())
            demand_wheel_ratio_samples.append(wheel_ratio[demand_mask].detach().cpu())
            demand_counter_samples.append(counter[demand_mask].float().detach().cpu())
            demand_tracking_samples.append(tracking[demand_mask].float().detach().cpu())
            demand_strict_samples.append(strict[demand_mask].float().detach().cpu())
            demand_epos_samples.append(env.signed_pos_diff[demand_mask].abs().detach().cpu())
        steer_now = env.u_cmd[:, 0].detach()
        if prev_steer is not None:
            steer_step.append((steer_now - prev_steer).abs()[sample_mask].detach().cpu())
        prev_steer = steer_now.clone()
        if live.any():
            beta_sum[live] += (env.es[live, 7] - env.beta_ref[live]).abs()
            _a = float(np.exp(-env.dt / 0.5))
            beta_filt[live] = _a * beta_filt[live] + (1.0 - _a) * env.es[live, 7]
            beta_f_sum[live] += (beta_filt[live] - env.beta_ref[live]).abs()
            v_sum[live] += env.es[live, 6]
            n_sum[live] += 1.0
            alive_hist.append(float((done[live] == 0).float().mean()))
        fails[newly & (term == 1)] += 1
        succ[newly & (term == 2)] += 1
        if newly.any() and args.renewal_eval:
            reason[0] += int((newly & env._term_masks[:, 0]).sum())
            reason[1] += int((newly & env._term_masks[:, 1]).sum())
            reason[2] += int((newly & env._term_masks[:, 2]).sum())
        if (not args.renewal_eval) and bool(ended.all()):
            break

    if not args.renewal_eval:
        done_mask = end_step > 0
        # an environment censored by the evaluation cap is NOT a success, so the
        # denominator is every environment, not only the terminated ones
        censored = ~done_mask
        lengths = torch.where(
            done_mask, end_step, torch.full_like(end_step, args.steps)).float() * env.dt
        n_ep = args.n
        print(f"checkpoint {os.path.basename(args.checkpoint)}  preset={args.car_preset} "
              f"ref={args.ref_mode} mode={args.control_mode}")
        effective_mu = float((env.plant.col("D") *
                              env.plant.tyre_scale.squeeze(1)).mean())
        print(f"  road grip scale {args.grip_scale:.3f}  "
              f"effective tyre peak mu {effective_mu:.4f}")
        print(f"  one-episode mode, {args.n} envs,"
              f" {n_ep - int(censored.sum())} terminated, {int(censored.sum())} censored")
        if n_ep == 0:
            print("  no episode terminated")
            return
        q = torch.quantile(lengths, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=dev))
        # A continuing task deliberately never emits a success terminal.  Reaching
        # the evaluation cap is therefore a full-window pass (censored in survival
        # terminology), not a success flag.  Using end_succ here made every
        # continuing checkpoint print 0 % even when many vehicles ran all 30 s.
        full_rate = (float(censored.float().mean()) if env.continuing_task
                     else float(end_succ.float().mean()))
        # Wilson 95 % interval
        z = 1.96
        denom = 1 + z * z / n_ep
        centre = (full_rate + z * z / (2 * n_ep)) / denom
        half = z * ((full_rate * (1 - full_rate) / n_ep + z * z / (4 * n_ep * n_ep)) ** 0.5) / denom
        print(f"  mean {lengths.mean():.2f} s  median {q[2]:.2f} s  "
              f"p10/p25/p75/p90 {q[0]:.2f}/{q[1]:.2f}/{q[3]:.2f}/{q[4]:.2f}")
        print(f"  full length rate {full_rate * 100:.1f} % (95 % CI "
              f"{max(0.0, centre - half) * 100:.1f}-{min(1.0, centre + half) * 100:.1f})")
        exc = end_reason[done_mask]
        first_reason = torch.where(exc[:, 0], 0, torch.where(exc[:, 1], 1, torch.where(exc[:, 2], 2, 3)))
        names = ["no_progress", "course", "off_path", "other"]
        print("  exclusive first failure reasons "
              + "  ".join(f"{names[i]} {int((first_reason == i).sum())}" for i in range(4)))
        _n = n_sum.clamp(min=1.0)
        print(f"  per-episode |beta - beta_ref| {(beta_sum / _n).mean():.3f} rad  "
              f"filtered (tau 0.5 s) {(beta_f_sum / _n).mean():.3f} rad  "
              f"per-episode speed {(v_sum / _n).mean():.2f} m/s")
        _sn = strict_n.clamp_min(1.0)
        duty = strict_sum / _sn
        print(f"  STRICT powered-drift duty {duty.mean() * 100:.2f} %  "
              f"all-step pass {float((duty >= 1.0 - 1e-9).float().mean()) * 100:.1f} %")
        print(f"  adaptive duties: tracking {(tracking_sum/_sn).mean()*100:.2f} %  "
              f"near-envelope {(performance_sum/_sn).mean()*100:.2f} %  "
              f"powered-slide-active {(drift_active_sum/_sn).mean()*100:.2f} %  "
              f"tracked-quality-drift {(drift_quality_sum/_sn).mean()*100:.2f} %")
        ep = ey = None
        if epos_samples:
            ep = torch.cat(epos_samples)
            ey = torch.cat(eyaw_samples)
            print(f"  tracking errors: |e_pos| mean {ep.mean():.3f} m  "
                  f"p95 {torch.quantile(ep, 0.95):.3f} m  p99 {torch.quantile(ep, 0.99):.3f} m;  "
                  f"|e_yaw| mean {ey.mean():.3f} rad  "
                  f"p95 {torch.quantile(ey, 0.95):.3f} rad  p99 {torch.quantile(ey, 0.99):.3f} rad")
        print(f"  means: throttle {(throttle_sum/_sn).mean():.3f}  "
              f"kappa_rear {(kappa_sum/_sn).mean():.3f}  |beta| {(beta_abs_sum/_sn).mean():.3f}  "
              f"rear/front {(wheel_ratio_sum/_sn).mean():.3f}  friction_util {(util_sum/_sn).mean():.3f}")
        _bvn = beta_sign_valid_sum.clamp_min(1.0)
        print(f"  signed: beta {(beta_signed_sum/_sn).mean():.3f} rad  "
              f"beta_ref {(beta_ref_signed_sum/_sn).mean():.3f} rad  "
              f"beta*sign(beta_ref) {(beta_aligned_sum/_sn).mean():.3f} rad  "
              f"beta sign-match {(beta_sign_match_sum/_bvn).mean()*100:.2f} %  "
              f"countersteer {(counter_sum/_sn).mean()*100:.2f} %")
        print(f"  beta demand: |ref| {(beta_ref_abs_sum/_sn).mean():.3f} rad  "
              f">=0.05 {(beta_ref_demand_005_sum/_sn).mean()*100:.2f} %  "
              f">=0.15 {(beta_ref_demand_015_sum/_sn).mean()*100:.2f} %  "
              f">=0.30 {(beta_ref_demand_030_sum/_sn).mean()*100:.2f} %")
        demand_speed = demand_curvature = demand_util = None
        demand_throttle = demand_kappa = demand_beta = demand_beta_error = None
        demand_wheel_ratio = demand_counter = demand_tracking = demand_strict = None
        demand_epos = None
        if demand_speed_samples:
            demand_speed = torch.cat(demand_speed_samples)
            demand_curvature = torch.cat(demand_curvature_samples)
            demand_util = torch.cat(demand_util_samples)
            demand_throttle = torch.cat(demand_throttle_samples)
            demand_kappa = torch.cat(demand_kappa_samples)
            demand_beta = torch.cat(demand_beta_samples)
            demand_beta_error = torch.cat(demand_beta_error_samples)
            demand_wheel_ratio = torch.cat(demand_wheel_ratio_samples)
            demand_counter = torch.cat(demand_counter_samples)
            demand_tracking = torch.cat(demand_tracking_samples)
            demand_strict = torch.cat(demand_strict_samples)
            demand_epos = torch.cat(demand_epos_samples)
            qs = torch.tensor([0.10, 0.50, 0.90])
            qv = torch.quantile(demand_speed, qs)
            qk = torch.quantile(demand_curvature, qs)
            print(f"  beta>=0.15 cells: speed p10/p50/p90 "
                  f"{qv[0]:.2f}/{qv[1]:.2f}/{qv[2]:.2f} m/s  "
                  f"|kappa_path| {qk[0]:.3f}/{qk[1]:.3f}/{qk[2]:.3f}  "
                  f"util mean {demand_util.mean():.3f}")
            print(f"    behavior: throttle {demand_throttle.mean():.3f}  "
                  f"kappa_rear {demand_kappa.mean():.3f}  |beta| {demand_beta.mean():.3f}  "
                  f"|beta-beta_ref| {demand_beta_error.mean():.3f}  "
                  f"rear/front {demand_wheel_ratio.mean():.3f}")
            print(f"    quality: countersteer {demand_counter.mean()*100:.2f} %  "
                  f"tracking {demand_tracking.mean()*100:.2f} %  "
                  f"strict {demand_strict.mean()*100:.2f} %  "
                  f"|e_pos| {demand_epos.mean():.3f} m")
        ds = None
        if steer_step:
            ds = torch.cat(steer_step)
            print(f"  steering |step| mean {ds.mean():.5f} rad  p95 {torch.quantile(ds, 0.95):.5f} rad  "
                  f"max {ds.max():.5f} rad")

        if args.json_out:
            metrics = {
                "checkpoint": os.path.abspath(args.checkpoint),
                "stochastic_diagnostic": args.stochastic_diagnostic,
                "metric_throttle_min": args.metric_throttle_min,
                "initial_conditions": initial_conditions,
                "sensor_configuration": sensor_configuration,
                "reference_generation": reference_generation,
                "seed": args.seed,
                "preset": args.car_preset,
                "ref_mode": args.ref_mode,
                "control_mode": args.control_mode,
                "road_grip_scale": float(args.grip_scale),
                "effective_tyre_peak_mu": effective_mu,
                "num_envs": args.n,
                "steps": args.steps,
                "dt_s": env.dt,
                "window_s": args.steps * env.dt,
                "terminated": n_ep - int(censored.sum()),
                "censored": int(censored.sum()),
                "full_window_rate": full_rate,
                "survival_mean_s": float(lengths.mean()),
                "survival_median_s": float(q[2]),
                "failure_no_progress": int((first_reason == 0).sum()),
                "failure_course": int((first_reason == 1).sum()),
                "failure_off_path": int((first_reason == 2).sum()),
                "beta_reference_error_mean_rad": float((beta_sum / _n).mean()),
                "beta_reference_error_filtered_mean_rad": float((beta_f_sum / _n).mean()),
                "speed_mean_mps": float((v_sum / _n).mean()),
                "strict_powered_drift_duty": float(duty.mean()),
                "strict_all_step_rate": float((duty >= 1.0 - 1e-9).float().mean()),
                "tracking_duty": float((tracking_sum / _sn).mean()),
                "near_envelope_duty": float((performance_sum / _sn).mean()),
                "powered_slide_active_duty": float((drift_active_sum / _sn).mean()),
                "tracked_quality_drift_duty": float((drift_quality_sum / _sn).mean()),
                "throttle_mean": float((throttle_sum / _sn).mean()),
                "brake_mean": float((brake_sum / _sn).mean()),
                "brake_duty_gt_005": float((brake_active_sum / _sn).mean()),
                "rear_kappa_mean": float((kappa_sum / _sn).mean()),
                "beta_abs_mean_rad": float((beta_abs_sum / _sn).mean()),
                "beta_signed_mean_rad": float((beta_signed_sum / _sn).mean()),
                "beta_reference_signed_mean_rad": float((beta_ref_signed_sum / _sn).mean()),
                "beta_reference_abs_mean_rad": float((beta_ref_abs_sum / _sn).mean()),
                "beta_reference_demand_ge_005_duty": float((beta_ref_demand_005_sum / _sn).mean()),
                "beta_reference_demand_ge_015_duty": float((beta_ref_demand_015_sum / _sn).mean()),
                "beta_reference_demand_ge_030_duty": float((beta_ref_demand_030_sum / _sn).mean()),
                "beta_reference_alignment_mean_rad": float((beta_aligned_sum / _sn).mean()),
                "beta_reference_sign_match_duty": float((beta_sign_match_sum / _bvn).mean()),
                "countersteer_duty": float((counter_sum / _sn).mean()),
                "rear_front_wheel_speed_ratio_mean": float((wheel_ratio_sum / _sn).mean()),
                "friction_utilization_mean": float((util_sum / _sn).mean()),
            }
            if ep is not None:
                metrics.update({
                    "path_error_abs_mean_m": float(ep.mean()),
                    "path_error_abs_p95_m": float(torch.quantile(ep, 0.95)),
                    "yaw_error_abs_mean_rad": float(ey.mean()),
                    "yaw_error_abs_p95_rad": float(torch.quantile(ey, 0.95)),
                })
            if ds is not None:
                metrics.update({
                    "steering_step_abs_mean_rad": float(ds.mean()),
                    "steering_step_abs_p95_rad": float(torch.quantile(ds, 0.95)),
                    "steering_step_abs_max_rad": float(ds.max()),
                })
            if demand_speed is not None:
                metrics.update({
                    "beta_demand_ge_015_speed_mean_mps": float(demand_speed.mean()),
                    "beta_demand_ge_015_speed_p10_mps": float(torch.quantile(demand_speed, 0.10)),
                    "beta_demand_ge_015_speed_p50_mps": float(torch.quantile(demand_speed, 0.50)),
                    "beta_demand_ge_015_speed_p90_mps": float(torch.quantile(demand_speed, 0.90)),
                    "beta_demand_ge_015_curvature_abs_mean_inv_m": float(demand_curvature.mean()),
                    "beta_demand_ge_015_curvature_abs_p10_inv_m": float(torch.quantile(demand_curvature, 0.10)),
                    "beta_demand_ge_015_curvature_abs_p50_inv_m": float(torch.quantile(demand_curvature, 0.50)),
                    "beta_demand_ge_015_curvature_abs_p90_inv_m": float(torch.quantile(demand_curvature, 0.90)),
                    "beta_demand_ge_015_friction_util_mean": float(demand_util.mean()),
                    "beta_demand_ge_015_throttle_mean": float(demand_throttle.mean()),
                    "beta_demand_ge_015_rear_kappa_mean": float(demand_kappa.mean()),
                    "beta_demand_ge_015_beta_abs_mean_rad": float(demand_beta.mean()),
                    "beta_demand_ge_015_beta_error_abs_mean_rad": float(demand_beta_error.mean()),
                    "beta_demand_ge_015_rear_front_wheel_speed_ratio_mean": float(demand_wheel_ratio.mean()),
                    "beta_demand_ge_015_countersteer_duty": float(demand_counter.mean()),
                    "beta_demand_ge_015_tracking_duty": float(demand_tracking.mean()),
                    "beta_demand_ge_015_strict_duty": float(demand_strict.mean()),
                    "beta_demand_ge_015_path_error_abs_mean_m": float(demand_epos.mean()),
                })
            output_parent = os.path.dirname(os.path.abspath(args.json_out))
            os.makedirs(output_parent, exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as stream:
                json.dump(metrics, stream, indent=2)
            print("  metrics saved", os.path.abspath(args.json_out))

        return

    n_done = int((fails + succ).sum().item())
    print(f"checkpoint {os.path.basename(args.checkpoint)}  preset={args.car_preset} "
          f"ref={args.ref_mode} mode={args.control_mode}")
    print(f"  {args.n} envs x {args.steps} control steps ({args.steps * env.dt:.1f} s each)")
    print(f"  episodes finished {n_done}: failed {int(fails.sum())} succeeded {int(succ.sum())}")
    print(f"  mean episode length {args.steps * args.n / max(n_done, 1) * env.dt:.2f} s")
    print(f"  failure reasons: no_progress {reason[0]}  course {reason[1]}  off_path {reason[2]}")
    print(f"  mean |beta - beta_ref| {float((beta_sum / n_sum.clamp(min=1.0)).mean()):.3f} rad")
    if getattr(env, "rew_stats", None):
        print("  reward components", {k: round(v, 2) for k, v in env.rew_stats.items()})
    print(f"  mean speed {float((v_sum / n_sum.clamp(min=1.0)).mean()):.2f} m/s  "
          f"alive fraction {sum(alive_hist) / max(len(alive_hist), 1):.2f}")


if __name__ == "__main__":
    main()



