"""One-episode baseline for the seeded drift curriculum.

This answers whether the installed equilibrium trim can keep the teleported
state alive without a learned correction.  All statistics stay on the GPU and
are synchronized once at the end, so it is safe to use between checkpoints.
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from envs.continuous_drift_iwd import ContinuousDriftIWDEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=365000)
    ap.add_argument("--grip-scale", type=float, default=1.0,
                    help="road multiplier applied to the calibrated tyre peak")
    ap.add_argument("--closed-random-radius", type=float, default=2.0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", args.n, "cuda:0", ref_mode="closed_random",
        plant="fast", control_mode="pedals", train=True, asymmetric=True,
        continuing_task=True, reward_mode="bounded_joint", drift_mode=True,
        drift_spawn_spin_frac=1.0, drift_spawn_trim_frac=1.0,
        spawn_beta_frac=1.0, speed_target_mode="limit", speed_margin=0.90,
        speed_cap=15.0, steer_actuator_tau=0.08, steer_rate_limit=1.5,
        closed_random_radius=args.closed_random_radius,
        beta_ref_mode="adaptive_drift", beta_max=0.32,
        drift_beta_target=0.40, drift_util_lo=0.55, drift_util_hi=0.90,
        drift_curve_lo=0.02, drift_curve_hi=0.06,
        drift_curve_fade_lo=0.10, drift_curve_fade_hi=0.15,
        drift_mu_lo=1.0623, drift_mu_hi=2.1246,
        grip_scale=args.grip_scale)
    env.reset()
    # Physical correction zero.  The environment adds the seeded trim; in the
    # final curriculum trim is zero and this is exactly zero physical command.
    action = torch.zeros((args.n, 3), device=env.device)
    active = torch.ones(args.n, dtype=torch.bool, device=env.device)
    duration_steps = torch.full((args.n,), args.steps, dtype=torch.long,
                                device=env.device)
    strict_steps = torch.zeros(args.n, dtype=torch.long, device=env.device)
    active_steps = torch.zeros(args.n, dtype=torch.long, device=env.device)
    signal_sum = torch.zeros((args.n, 5), device=env.device)

    for step in range(args.steps):
        _, _, done, _ = env.step(action)
        now = env._now
        rear_kappa = now["kappa_power"][:, 2:4].mean(dim=1)
        speed = now["v"]
        beta = now["beta"].abs()
        wheel_v = env.s[:, 6:10] * env.wheel_radius
        wheel_ratio = (wheel_v[:, 2:4].abs().mean(dim=1) /
                       wheel_v[:, :2].abs().mean(dim=1).clamp_min(0.5))
        mu = (env.plant.col("D") * env.plant.tyre_scale.squeeze(1)).clamp_min(0.05)
        util = speed.square() * env.kappa_ref.abs() / (mu * 9.81)
        counter = ((env.u_cmd[:, 0] * env.kappa_ref) < 0.0) & (env.u_cmd[:, 0].abs() >= 0.03)
        strict = ((env.u_cmd[:, 1] >= 0.40) & (rear_kappa >= 0.30) &
                  (beta >= 0.20) & (wheel_ratio >= 1.25) & counter &
                  (util >= 0.55) & (util <= 1.35) &
                  (now["signed_pos_diff"].abs() <= 1.0))
        strict_steps += (strict & active).long()
        active_steps += active.long()
        values = torch.stack((env.u_cmd[:, 1], rear_kappa, beta, wheel_ratio, util), dim=1)
        signal_sum += values * active.unsqueeze(1)
        newly_done = active & done.bool()
        duration_steps[newly_done] = step + 1
        active &= ~done.bool()

    denom = active_steps.clamp_min(1).float()
    means = (signal_sum / denom.unsqueeze(1)).mean(dim=0).cpu().tolist()
    report = {
        "n": args.n, "steps": args.steps, "seed": args.seed,
        "grip_scale": args.grip_scale,
        "closed_random_radius": args.closed_random_radius,
        "full_length_rate": float(active.float().mean()),
        "mean_survival_s": float(duration_steps.float().mean() * env.dt),
        "median_survival_s": float(duration_steps.float().median() * env.dt),
        "strict_duty": float(strict_steps.sum().float() / active_steps.sum().clamp_min(1)),
        "means": dict(zip(("throttle", "rear_kappa", "abs_beta", "wheel_ratio", "util"), means)),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
