"""Feasibility gate for the sensor-causal A0 teacher before distillation."""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.continuous_drift_iwd import ContinuousDriftIWDEnv  # noqa: E402
from observable_teacher import ObservableEnvelopeTeacher  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=373000)
    ap.add_argument("--speed-margin", type=float, default=0.45)
    ap.add_argument("--closed-random-radius", type=float, default=2.0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", args.n, "cuda:0", ref_mode="closed_random",
        closed_random_radius=args.closed_random_radius, plant="fast",
        control_mode="pedals", train=True, asymmetric=True,
        continuing_task=True, reward_mode="adaptive_envelope", drift_mode=True,
        spawn_beta_frac=0.0, drift_spawn_spin_frac=0.0,
        drift_spawn_trim_frac=0.0, speed_target_mode="limit",
        speed_margin=args.speed_margin, speed_cap=22.0,
        preview_time="0.35,0.7,1.2", steer_actuator_tau=0.08,
        steer_rate_limit=1.5, randomize_param={"D": [1.0623, 2.1246]},
        random_seed=args.seed)
    env.reset()
    teacher = ObservableEnvelopeTeacher(
        args.n, env.device, speed_margin=args.speed_margin, speed_cap=22.0)
    initial_mu = env.plant.col("D").detach().clone()
    active = torch.ones(args.n, dtype=torch.bool, device=env.device)
    duration = torch.full((args.n,), args.steps, dtype=torch.long, device=env.device)
    tracking = torch.zeros(args.n, device=env.device)
    speeds = torch.zeros(args.n, device=env.device)
    samples = torch.zeros(args.n, device=env.device)
    for step in range(args.steps):
        action, _ = teacher.command(env)
        _, _, done, _ = env.step(action)
        tracking += (active & (env.signed_pos_diff.abs() <= 1.5)
                     & (env.angle_diff.abs() <= 0.35)).float()
        front_speed = (env.s[:, 6:8] * env.wheel_radius[..., :2]).mean(1)
        speeds += active.float() * front_speed
        samples += active.float()
        newly = active & done.bool()
        duration[newly] = step + 1
        active &= ~done.bool()
        teacher.reset(newly)
    rows = []
    edges = torch.tensor([1.0623, 1.4164, 1.7705, 2.1247], device=env.device)
    for i in range(3):
        mask = (initial_mu >= edges[i]) & (initial_mu < edges[i + 1])
        denom = samples[mask].clamp_min(1.0)
        rows.append({
            "mu_lo": float(edges[i]), "mu_hi": float(edges[i + 1]),
            "n": int(mask.sum()),
            "full_rate": float(active[mask].float().mean()),
            "mean_survival_s": float(duration[mask].float().mean() * env.dt),
            "tracking_duty": float((tracking[mask] / denom).mean()),
            "mean_front_speed": float((speeds[mask] / denom).mean()),
            "mean_mu_hat": float(teacher.mu_hat[mask].mean()),
        })
    report = {"steps": args.steps, "n": args.n, "seed": args.seed,
              "speed_margin": args.speed_margin,
              "closed_random_radius": args.closed_random_radius,
              "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
