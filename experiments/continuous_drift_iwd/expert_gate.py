"""One-episode privileged-controller feasibility gate for FastPlant."""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.continuous_drift_iwd import ContinuousDriftIWDEnv  # noqa: E402
from expert import expert_action  # noqa: E402


def residual_for_desired(desired, trim):
    """Encode a desired physical command for the trim-residual action interface."""
    out = torch.zeros_like(desired)
    out[:, 0] = desired[:, 0] - trim[:, 0]
    desired_long = desired[:, 1] - desired[:, 2]
    trim_long = trim[:, 1] - trim[:, 2]
    correction = desired_long - trim_long
    out[:, 1] = correction.clamp(min=0.0)
    out[:, 2] = (-correction).clamp(min=0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=366000)
    ap.add_argument("--grip-scale", type=float, default=1.0)
    ap.add_argument("--controller", choices=("drift", "pp"), default="drift")
    ap.add_argument("--closed-random-radius", type=float, default=1.0)
    ap.add_argument("--speed-margin", type=float, default=0.90)
    ap.add_argument("--physical-entry", action="store_true")
    ap.add_argument("--q0", type=float, default=0.55)
    ap.add_argument("--kv", type=float, default=0.08)
    ap.add_argument("--kff", type=float, default=0.6)
    ap.add_argument("--kb", type=float, default=1.2)
    ap.add_argument("--kr", type=float, default=-0.5)
    ap.add_argument("--kp", type=float, default=0.10)
    ap.add_argument("--kd", type=float, default=-0.6)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", args.n, "cuda:0", ref_mode="closed_random",
        plant="fast", control_mode="pedals", train=True, asymmetric=True,
        continuing_task=True, reward_mode="bounded_joint", drift_mode=True,
        drift_spawn_spin_frac=0.0 if args.physical_entry else 1.0,
        drift_spawn_trim_frac=0.0 if args.physical_entry else 1.0,
        spawn_beta_frac=0.0 if args.physical_entry else 1.0,
        speed_target_mode="limit", speed_margin=args.speed_margin,
        speed_cap=22.0, steer_actuator_tau=0.08, steer_rate_limit=1.5,
        preview_time="0.35,0.7,1.2", grip_scale=args.grip_scale,
        closed_random_radius=args.closed_random_radius)
    env.reset()
    active = torch.ones(args.n, dtype=torch.bool, device=env.device)
    duration = torch.full((args.n,), args.steps, dtype=torch.long, device=env.device)
    strict_steps = torch.zeros(args.n, dtype=torch.long, device=env.device)
    active_steps = torch.zeros(args.n, dtype=torch.long, device=env.device)
    signal_sum = torch.zeros((args.n, 5), device=env.device)
    for step in range(args.steps):
        mu_target = (env.plant.col("D") * env.plant.tyre_scale.squeeze(1)).clamp(0.3, 3.0)
        kap_target = torch.maximum(env.kappa_ref.abs(),
                                   env.kappa_preview.abs().amax(dim=1)).clamp_min(1e-2)
        v_target = (args.speed_margin * torch.sqrt(
            mu_target * 9.81 / kap_target)).clamp(2.0, 22.0)
        if args.controller == "pp":
            long = (0.12 + 0.18 * (v_target - env.es[:, 6])).clamp(-1.0, 1.0)
            desired = torch.stack((env.delta_pp,
                                   long.clamp(min=0.0),
                                   (-long).clamp(min=0.0)), dim=1)
        else:
            desired = expert_action(
                env, v_target=v_target, k_ff=args.kff, kb=args.kb, kr=args.kr,
                kp=args.kp, kd=args.kd, q0=args.q0, kv=args.kv)
        action = residual_for_desired(desired, env.u_trim)
        _, _, done, _ = env.step(action)
        now = env._now
        rear_kappa = now["kappa_power"][:, 2:4].mean(dim=1)
        speed, beta = now["v"], now["beta"].abs()
        wheel_v = env.s[:, 6:10] * env.wheel_radius
        ratio = wheel_v[:, 2:4].abs().mean(1) / wheel_v[:, :2].abs().mean(1).clamp_min(0.5)
        mu = (env.plant.col("D") * env.plant.tyre_scale.squeeze(1)).clamp_min(0.05)
        util = speed.square() * env.kappa_ref.abs() / (mu * 9.81)
        counter = ((env.u_cmd[:, 0] * env.kappa_ref) < 0.0) & (env.u_cmd[:, 0].abs() >= 0.03)
        strict = ((env.u_cmd[:, 1] >= 0.40) & (rear_kappa >= 0.30) &
                  (beta >= 0.20) & (ratio >= 1.25) & counter &
                  (util >= 0.55) & (util <= 1.35) &
                  (now["signed_pos_diff"].abs() <= 1.0))
        strict_steps += (strict & active).long()
        active_steps += active.long()
        signal_sum += torch.stack((env.u_cmd[:, 1], rear_kappa, beta, ratio, util), 1) * active.unsqueeze(1)
        newly_done = active & done.bool()
        duration[newly_done] = step + 1
        active &= ~done.bool()
    means = (signal_sum / active_steps.clamp_min(1).unsqueeze(1)).mean(0).cpu().tolist()
    report = {
        "n": args.n, "steps": args.steps, "seed": args.seed,
        "grip_scale": args.grip_scale, "controller": args.controller,
        "closed_random_radius": args.closed_random_radius,
        "speed_margin": args.speed_margin, "physical_entry": args.physical_entry,
        "gains": {k: getattr(args, k) for k in ("q0", "kv", "kff", "kb", "kr", "kp", "kd")},
        "full_length_rate": float(active.float().mean()),
        "mean_survival_s": float(duration.float().mean() * env.dt),
        "median_survival_s": float(duration.float().median() * env.dt),
        "strict_duty": float(strict_steps.sum().float() / active_steps.sum().clamp_min(1)),
        "means": dict(zip(("throttle", "rear_kappa", "abs_beta", "wheel_ratio", "util"), means)),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
