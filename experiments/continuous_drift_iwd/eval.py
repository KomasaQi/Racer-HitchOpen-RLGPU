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
    ap.add_argument("--ref-mode", default="circle")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta-scale", type=float, default=1.0)
    ap.add_argument("--offpath-scale", type=float, default=1.0)
    ap.add_argument("--course-fail", type=float, default=1.0)
    ap.add_argument("--stall-speed", type=float, default=2.0)
    ap.add_argument("--alive-bonus", type=float, default=0.0)
    ap.add_argument("--curv-scale", type=float, default=1.0)
    ap.add_argument("--seg-scale", type=float, default=1.0)
    ap.add_argument("--obs-privileged", action="store_true")
    ap.add_argument("--one-episode", action="store_true",
                    help="each env contributes exactly one episode (unbiased)")
    ap.add_argument("--curv-smooth", type=int, default=0)
    ap.add_argument("--preview-scale", type=float, default=1.0)
    ap.add_argument("--flip-prob", type=float, default=None)
    ap.add_argument("--stall-window", type=int, default=None)
    ap.add_argument("--stall-move", type=float, default=1.0)
    ap.add_argument("--yaw-weight", type=float, default=0.0)
    ap.add_argument("--beta-weight", type=float, default=1.6)
    ap.add_argument("--dir-weight", type=float, default=0.5)
    ap.add_argument("--pos-weight", type=float, default=2.4)
    ap.add_argument("--spawn-mode", default="equilibrium")
    ap.add_argument("--spawn-speed", type=float, default=6.5)
    ap.add_argument("--grip-scale", type=float, default=1.0,
                    help="global friction multiplier (evaluate on different surfaces)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env = ContinuousDriftIWDEnv(args.car_preset, args.n, args.device, ref_mode=args.ref_mode,
                                control_mode=args.control_mode, beta_scale=args.beta_scale,
                                spawn_mode=args.spawn_mode, spawn_speed=args.spawn_speed,
                                offpath_scale=args.offpath_scale, course_fail=args.course_fail,
                                stall_speed=args.stall_speed, alive_bonus=args.alive_bonus,
                                curv_scale=args.curv_scale, seg_scale=args.seg_scale, grip_scale=args.grip_scale,
                                obs_privileged=args.obs_privileged,
                                curv_smooth=args.curv_smooth, preview_scale=args.preview_scale,
                                flip_prob=args.flip_prob,
                                stall_window=args.stall_window, stall_move=args.stall_move,
                                yaw_weight=args.yaw_weight, beta_weight=args.beta_weight,
                                dir_weight=args.dir_weight, pos_weight=args.pos_weight, train=True)
    obs = env.reset()
    # the environment only spawns on respawn, so force the first episode here
    if getattr(env, "train", False):
        env.is_done[:] = 1
        env.reset_done_envs()
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
    beta_err, v_hist, alive_hist = [], [], []
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
        if args.one_episode:
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
            beta_err.append(float((env.es[live, 7] - env.beta_ref[live]).abs().mean()))
            v_hist.append(float(env.es[live, 6].mean()))
            alive_hist.append(float((done[live] == 0).float().mean()))
        fails[newly & (term == 1)] += 1
        succ[newly & (term == 2)] += 1
        if newly.any() and not args.one_episode:
            reason[0] += int((newly & env._term_masks[:, 0]).sum())
            reason[1] += int((newly & env._term_masks[:, 1]).sum())
            reason[2] += int((newly & env._term_masks[:, 2]).sum())
        if args.one_episode and bool(ended.all()):
            break

    if args.one_episode:
        done_mask = end_step > 0
        lengths = end_step[done_mask].float() * env.dt
        n_ep = int(done_mask.sum())
        print(f"checkpoint {os.path.basename(args.checkpoint)}  preset={args.car_preset} "
              f"ref={args.ref_mode} mode={args.control_mode}")
        print(f"  one-episode mode, {args.n} envs, {n_ep} terminated")
        if n_ep == 0:
            print("  no episode terminated")
            return
        q = torch.quantile(lengths, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=dev))
        succ_rate = float(end_succ[done_mask].float().mean())
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
        print(f"  mean |beta - beta_ref| {sum(beta_err) / max(len(beta_err), 1):.3f} rad  "
              f"mean speed {sum(v_hist) / max(len(v_hist), 1):.2f} m/s")
        return

    n_done = int((fails + succ).sum().item())
    print(f"checkpoint {os.path.basename(args.checkpoint)}  preset={args.car_preset} "
          f"ref={args.ref_mode} mode={args.control_mode}")
    print(f"  {args.n} envs x {args.steps} control steps ({args.steps * env.dt:.1f} s each)")
    print(f"  episodes finished {n_done}: failed {int(fails.sum())} succeeded {int(succ.sum())}")
    print(f"  mean episode length {args.steps * args.n / max(n_done, 1) * env.dt:.2f} s")
    print(f"  failure reasons: no_progress {reason[0]}  course {reason[1]}  off_path {reason[2]}")
    print(f"  mean |beta - beta_ref| {sum(beta_err) / len(beta_err):.3f} rad "
          f"(last 100 steps {sum(beta_err[-100:]) / min(100, len(beta_err)):.3f})")
    if getattr(env, "rew_stats", None):
        print("  reward components", {k: round(v, 2) for k, v in env.rew_stats.items()})
    print(f"  mean speed {sum(v_hist) / len(v_hist):.2f} m/s  "
          f"alive fraction {sum(alive_hist) / len(alive_hist):.2f}")


if __name__ == "__main__":
    main()

