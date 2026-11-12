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
        self.layers = []
        i = 0
        while f"a2c_network.actor_mlp.{i}.weight" in w:
            self.layers.append((w[f"a2c_network.actor_mlp.{i}.weight"].to(device),
                                w[f"a2c_network.actor_mlp.{i}.bias"].to(device)))
            i += 2
        self.mu = (w["a2c_network.mu.weight"].to(device), w["a2c_network.mu.bias"].to(device))
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

    def reset_hidden(self, mask):
        if self.gru is None:
            return
        if self.h is None:
            return
        self.h[:, mask, :] = 0.0

    @torch.no_grad()
    def act(self, obs):
        x = (obs - self.mean) / torch.sqrt(self.var + 1e-5)
        x = torch.clamp(x, -5.0, 5.0)
        if self.gru is not None:
            if self.h is None:
                self.h = torch.zeros(1, x.shape[0], self.gru.hidden_size, device=x.device)
            out, self.h = self.gru(x.unsqueeze(0), self.h)
            x = out.squeeze(0)
        for w, b in self.layers:
            x = torch.nn.functional.elu(torch.nn.functional.linear(x, w, b))
        return torch.nn.functional.linear(x, self.mu[0], self.mu[1])


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
    ap.add_argument("--steer-actuator-tau", type=float, default=0.0)
    ap.add_argument("--steer-rate-limit", type=float, default=0.0)
    ap.add_argument("--steer-deadband", type=float, default=0.0)
    ap.add_argument("--brake-margin", type=float, default=0.0)
    ap.add_argument("--alpha-peak", type=float, default=0.12)
    ap.add_argument("--beta-max", type=float, default=0.32)
    ap.add_argument("--ackermann-lo", type=float, default=0.8)
    ap.add_argument("--ackermann-hi", type=float, default=1.2)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-scenario", type=str, default=None)
    ap.add_argument("--load-scenario", type=str, default=None)
    ap.add_argument("--beta-scale", type=float, default=1.0)
    ap.add_argument("--offpath-scale", type=float, default=1.0)
    ap.add_argument("--course-fail", type=float, default=1.0)
    ap.add_argument("--stall-speed", type=float, default=2.0)
    ap.add_argument("--alive-bonus", type=float, default=0.0)
    ap.add_argument("--curv-scale", type=float, default=1.0)
    ap.add_argument("--seg-scale", type=float, default=1.0)
    ap.add_argument("--save-path-bank", type=str, default=None)
    ap.add_argument("--load-path-bank", type=str, default=None)
    ap.add_argument("--config-from", type=str, default=None,
                    help="load the supported evaluator defaults from env_config.json; "
                         "fields that cannot be reconstructed are reported")
    ap.add_argument("--renewal-eval", action="store_true",
                    help="allow repeated respawns (biased); one episode per env is the default")
    ap.add_argument("--curv-smooth", type=int, default=0)
    ap.add_argument("--preview-scale", type=float, default=1.0)
    ap.add_argument("--preview-time", type=str, default=None)
    ap.add_argument("--flip-prob", type=float, default=None)
    ap.add_argument("--stall-window", type=int, default=None)
    ap.add_argument("--stall-move", type=float, default=1.0)
    ap.add_argument("--spawn-speed-min", type=float, default=None)
    ap.add_argument("--spawn-deficit-frac", type=float, default=0.0)
    ap.add_argument("--yaw-weight", type=float, default=0.0)
    ap.add_argument("--beta-weight", type=float, default=1.6)
    ap.add_argument("--dir-weight", type=float, default=0.5)
    ap.add_argument("--pos-weight", type=float, default=2.4)
    ap.add_argument("--speed-target", type=float, default=None)
    ap.add_argument("--speed-hinge", action="store_true")
    ap.add_argument("--spawn-mode", default="equilibrium")
    ap.add_argument("--spawn-speed", type=float, default=6.5)
    ap.add_argument("--grip-scale", type=float, default=1.0,
                    help="global friction multiplier (evaluate on different surfaces)")
    ap.add_argument("--disturbed", action="store_true",
                    help="apply the OU disturbance the policy was trained with")
    args = ap.parse_args()

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
                if hasattr(args, _k) and getattr(args, _k) == ap.get_default(_k):
                    setattr(args, _k, _v)
                    _applied[_k] = _v
            print("env_config loaded:", len(_applied), "parameters")
            _skip = {"seq_length", "gamma", "aux_reward_decay_steps",
                     "aux_reward_coef", "train", "ref_mode_cli"}
            _unsupported = sorted(k for k in _cfg
                                  if not hasattr(args, k) and k not in _skip)
            if _unsupported:
                print("WARNING: env_config fields not reconstructed:",
                      ", ".join(_unsupported))
            if _cfg.get("ref_mode_cli"):
                args.ref_mode = _cfg["ref_mode_cli"]
                _applied["ref_mode"] = args.ref_mode
        else:
            print("WARNING: no env_config.json in", _cfg_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = ContinuousDriftIWDEnv(args.car_preset, args.n, args.device, ref_mode=args.ref_mode,
                                control_mode=args.control_mode, beta_scale=args.beta_scale,
                                spawn_mode=args.spawn_mode, spawn_speed=args.spawn_speed,
                                offpath_scale=args.offpath_scale, course_fail=args.course_fail,
                                stall_speed=args.stall_speed, alive_bonus=args.alive_bonus,
                                curv_scale=args.curv_scale, seg_scale=args.seg_scale,
                                grip_scale=args.grip_scale,
                                disturbance_param=((0.97, 0.03) if args.disturbed else None),
                                # matching the training DR; the env forces tyre_scale
                                # to 1.0 whenever the disturbance is on, so a fixed
                                # --grip-scale cannot be applied in that mode
                                randomize_param=(({"B": [8.0, 12.0], "C": [1.7, 2.1],
                                                   "D": [float(args.grip_scale), float(args.grip_scale)]})
                                                  if args.disturbed else None),
                                curv_smooth=args.curv_smooth, preview_scale=args.preview_scale,
                                preview_time=args.preview_time,
                                flip_prob=args.flip_prob,
                                stall_window=(args.stall_window if args.stall_window is not None
                                              else max(int(round(0.2 * (2.65 / 0.35) ** 0.5 / 0.05)), 2)),
                                stall_move=args.stall_move,
                                spawn_speed_min=(args.spawn_speed_min if args.spawn_speed_min is not None else args.spawn_speed),
                                spawn_deficit_frac=args.spawn_deficit_frac,
                                dir_weight=args.dir_weight, pos_weight=args.pos_weight,
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
                                ackermann_hi=args.ackermann_hi, train=True)
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
    if args.load_scenario:
        _sc = np.load(args.load_scenario)
        obs = env.apply_scenario(_sc)
    elif args.save_scenario:
        np.savez(args.save_scenario, **env.capture_scenario())
        print("saved scenario", args.save_scenario)
        obs = env.obs()
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
    for step in range(args.steps):
        a = policy.act(obs)
        done_prev = env.is_done.clone()
        obs, reward, done, info = env.step(a)
        # the environment respawns inside step(), so ``env.is_done`` is already
        # cleared; the terminal flags of this step are the returned ``done``
        newly = (done_prev == 0) & (done != 0)
        term = done.clone()
        policy.reset_hidden(newly)
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
        print(f"  one-episode mode, {args.n} envs,"
              f" {n_ep - int(censored.sum())} terminated, {int(censored.sum())} censored")
        if n_ep == 0:
            print("  no episode terminated")
            return
        q = torch.quantile(lengths, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=dev))
        succ_rate = float(end_succ.float().mean())
        # Wilson 95 % interval
        z = 1.96
        denom = 1 + z * z / n_ep
        centre = (succ_rate + z * z / (2 * n_ep)) / denom
        half = z * ((succ_rate * (1 - succ_rate) / n_ep + z * z / (4 * n_ep * n_ep)) ** 0.5) / denom
        print(f"  mean {lengths.mean():.2f} s  median {q[2]:.2f} s  "
              f"p10/p25/p75/p90 {q[0]:.2f}/{q[1]:.2f}/{q[3]:.2f}/{q[4]:.2f}")
        print(f"  full length rate {succ_rate * 100:.1f} % (95 % CI "
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

