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
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env = ContinuousDriftIWDEnv(args.car_preset, args.n, args.device, ref_mode=args.ref_mode,
                                control_mode=args.control_mode, train=True)
    obs = env.reset()
    policy = Policy(args.checkpoint, torch.device(args.device))
    policy.reset_hidden(torch.ones(args.n, dtype=torch.bool, device=args.device))

    dev = torch.device(args.device)
    fails = torch.zeros(args.n, device=dev)
    succ = torch.zeros(args.n, device=dev)
    reason = [0, 0, 0]        # no_progress, large course error, off path
    beta_err, v_hist, alive_hist = [], [], []
    for _ in range(args.steps):
        a = policy.act(obs)
        done_prev = env.is_done.clone()
        obs, reward, done, info = env.step(a)
        newly = (done_prev == 0) & (env.is_done != 0)
        policy.reset_hidden(newly)
        beta_err.append((env.es[:, 7] - env.beta_ref).abs().mean().item())
        v_hist.append(env.es[:, 6].mean().item())
        alive_hist.append(float((env.is_done == 0).float().mean()))
        fails[newly & (env.is_done == 1)] += 1
        succ[newly & (env.is_done == 2)] += 1
        if newly.any():
            npc = env.no_progress_count >= env.no_progress_limit
            ang = env.angle_diff.abs() > 1.0
            pos = env.signed_pos_diff.abs() > env.length_scale
            reason[0] += int((newly & npc).sum())
            reason[1] += int((newly & ang).sum())
            reason[2] += int((newly & pos).sum())

    n_done = int((fails + succ).sum().item())
    print(f"checkpoint {os.path.basename(args.checkpoint)}  preset={args.car_preset} "
          f"ref={args.ref_mode} mode={args.control_mode}")
    print(f"  {args.n} envs x {args.steps} control steps ({args.steps * env.dt:.1f} s each)")
    print(f"  episodes finished {n_done}: failed {int(fails.sum())} succeeded {int(succ.sum())}")
    print(f"  mean episode length {args.steps * args.n / max(n_done, 1) * env.dt:.2f} s")
    print(f"  failure reasons: no_progress {reason[0]}  course {reason[1]}  off_path {reason[2]}")
    print(f"  mean |beta - beta_ref| {sum(beta_err) / len(beta_err):.3f} rad "
          f"(last 100 steps {sum(beta_err[-100:]) / min(100, len(beta_err)):.3f})")
    print(f"  mean speed {sum(v_hist) / len(v_hist):.2f} m/s  "
          f"alive fraction {sum(alive_hist) / len(alive_hist):.2f}")


if __name__ == "__main__":
    main()
