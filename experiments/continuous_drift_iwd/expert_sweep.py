"""Vectorised privileged feedback sweep for finding a feasible S0 teacher."""
import argparse
import itertools
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.continuous_drift_iwd import ContinuousDriftIWDEnv  # noqa: E402
from expert import expert_action  # noqa: E402
from expert_gate import residual_for_desired  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--per-config", type=int, default=16)
    ap.add_argument("--grip-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=367000)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    configs = [dict(zip(("q0", "kv", "kff", "kb", "kr"), vals)) for vals in
               itertools.product((0.50, 0.70, 0.90), (0.0, 0.02),
                                 (0.6, 1.0), (0.5, 1.2), (-0.2, -0.5))]
    n = len(configs) * args.per_config
    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", n, "cuda:0", ref_mode="closed_random",
        plant="fast", control_mode="pedals", train=True, asymmetric=True,
        continuing_task=True, reward_mode="bounded_joint", drift_mode=True,
        drift_spawn_spin_frac=1.0, drift_spawn_trim_frac=1.0,
        spawn_beta_frac=1.0, speed_target_mode="limit", speed_margin=0.90,
        speed_cap=22.0, steer_actuator_tau=0.08, steer_rate_limit=1.5,
        preview_time="0.35,0.7,1.2", grip_scale=args.grip_scale)
    env.reset()
    def expand(key):
        return torch.tensor([c[key] for c in configs], device=env.device).repeat_interleave(args.per_config)
    gains = {key: expand(key) for key in configs[0]}
    active = torch.ones(n, dtype=torch.bool, device=env.device)
    duration = torch.full((n,), args.steps, dtype=torch.long, device=env.device)
    strict_steps = torch.zeros(n, dtype=torch.long, device=env.device)
    active_steps = torch.zeros(n, dtype=torch.long, device=env.device)
    sums = torch.zeros((n, 5), device=env.device)
    for step in range(args.steps):
        mu_t = (env.plant.col("D") * env.plant.tyre_scale.squeeze(1)).clamp(0.3, 3.0)
        kap_t = torch.maximum(env.kappa_ref.abs(), env.kappa_preview.abs().amax(1)).clamp_min(1e-2)
        v_t = (0.90 * torch.sqrt(mu_t * 9.81 / kap_t)).clamp(2.0, 22.0)
        desired = expert_action(env, v_target=v_t, k_ff=gains["kff"],
                                kb=gains["kb"], kr=gains["kr"], kp=0.10,
                                kd=-0.60, q0=gains["q0"], kv=gains["kv"])
        _, _, done, _ = env.step(residual_for_desired(desired, env.u_trim))
        now = env._now
        rear = now["kappa_power"][:, 2:4].mean(1)
        speed, beta = now["v"], now["beta"].abs()
        wheel_v = env.s[:, 6:10] * env.wheel_radius
        ratio = wheel_v[:, 2:4].abs().mean(1) / wheel_v[:, :2].abs().mean(1).clamp_min(0.5)
        mu = (env.plant.col("D") * env.plant.tyre_scale.squeeze(1)).clamp_min(0.05)
        util = speed.square() * env.kappa_ref.abs() / (mu * 9.81)
        counter = ((env.u_cmd[:, 0] * env.kappa_ref) < 0.0) & (env.u_cmd[:, 0].abs() >= 0.03)
        strict = ((env.u_cmd[:, 1] >= 0.40) & (rear >= 0.30) & (beta >= 0.20) &
                  (ratio >= 1.25) & counter & (util >= 0.55) & (util <= 1.35) &
                  (now["signed_pos_diff"].abs() <= 1.0))
        strict_steps += (strict & active).long()
        active_steps += active.long()
        sums += torch.stack((env.u_cmd[:, 1], rear, beta, ratio, util), 1) * active.unsqueeze(1)
        newly = active & done.bool()
        duration[newly] = step + 1
        active &= ~done.bool()
    rows = []
    for i, cfg in enumerate(configs):
        sl = slice(i * args.per_config, (i + 1) * args.per_config)
        denom = active_steps[sl].clamp_min(1).float()
        means = (sums[sl] / denom.unsqueeze(1)).mean(0).cpu().tolist()
        row = dict(cfg)
        row.update({
            "full_rate": float(active[sl].float().mean()),
            "mean_survival_s": float(duration[sl].float().mean() * env.dt),
            "strict_duty": float(strict_steps[sl].sum().float() / active_steps[sl].sum().clamp_min(1)),
            **dict(zip(("throttle", "rear_kappa", "abs_beta", "wheel_ratio", "util"), means)),
        })
        rows.append(row)
    rows.sort(key=lambda r: (r["full_rate"], r["strict_duty"], r["mean_survival_s"]), reverse=True)
    report = {"grip_scale": args.grip_scale, "steps": args.steps,
              "per_config": args.per_config, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(rows[:12], indent=2))


if __name__ == "__main__":
    main()
