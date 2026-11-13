"""Numerical/reward smoke gate before launching PPO.

Runs the continuing closed-random task with a zero residual policy.  This is
not a performance benchmark; it verifies finite observations/rewards, reward
component scale, physical failure/reset behavior and strict-drift occupancy at
the curriculum spawn.  It never creates or updates a checkpoint.
"""
import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from envs.continuous_drift_iwd import ContinuousDriftIWDEnv  # noqa: E402


EV_DR = {
    "bx": [26.8, 32.8], "cx": [1.78, 1.94], "ex": [0.94, 1.02],
    "by": [15.6, 19.0], "cy": [1.76, 1.92], "ey": [0.91, 1.00],
    "m": [1417.0, 1505.0], "Iz": [1558.0, 1829.0],
    "IwF": [1.55, 1.99], "IwR": [2.00, 2.54],
    "Tf": [1.700, 1.740], "Tr": [1.580, 1.620],
    "rF": [0.3290, 0.3350], "rR": [0.3435, 0.3505],
    "load_sensitivity": [0.12, 0.19], "combined_slip_blend": [0.35, 0.65],
    "tire_stiffness_grip_exponent_x": [0.10, 0.30],
    "tire_stiffness_grip_exponent_y": [0.30, 0.50],
    "lsd_torque_ref": [72.0, 108.0], "lsd_exponent": [0.40, 0.60],
    "lsd_tau_max": [240.0, 360.0],
    "drive_efficiency": [0.975, 0.995], "motor_efficiency": [0.90, 1.10],
    "max_power_w": [345600.0, 422400.0], "D": [0.95607, 2.33706],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--hardware-dr", action="store_true")
    ap.add_argument("--plant-dr", action="store_true")
    ap.add_argument("--spin-frac", type=float, default=1.0)
    ap.add_argument("--beta-frac", type=float, default=1.0)
    ap.add_argument("--trim-frac", type=float, default=1.0)
    args = ap.parse_args()

    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", args.n, "cuda:0", ref_mode="closed_random",
        plant="fast", control_mode="pedals", train=True, asymmetric=True,
        continuing_task=True, reward_mode="bounded_joint", drift_mode=True,
        drift_spawn_spin_frac=args.spin_frac, drift_spawn_trim_frac=args.trim_frac,
        spawn_beta_frac=args.beta_frac,
        randomize_param=EV_DR if args.plant_dr else {},
        sensor_dr=args.hardware_dr, actuator_dr=args.hardware_dr,
        sensor_delay_max=2, action_delay_max=2, sensor_dropout_prob=0.002,
        speed_target_mode="limit", speed_margin=0.90, speed_cap=22.0,
        steer_actuator_tau=0.08, steer_rate_limit=1.5,
    )
    obs = env.reset()
    action = torch.zeros((args.n, 3), device="cuda:0")
    reward_sum = 0.0
    failures = 0
    strict_count = 0
    live_count = 0
    criterion_counts = {k: 0 for k in ("beta", "slip", "throttle", "counter",
                                        "utilization", "path")}
    signal_sums = {k: 0.0 for k in ("abs_beta", "rear_kappa", "throttle",
                                     "abs_steer", "utilization", "speed")}
    obs_abs_max = 0.0
    stats = {}
    finite = True
    for _ in range(args.steps):
        obs, reward, done, _ = env.step(action)
        actor = obs["obs"]
        finite = finite and bool(torch.isfinite(actor).all()) and bool(torch.isfinite(reward).all())
        obs_abs_max = max(obs_abs_max, float(actor.abs().max()))
        reward_sum += float(reward.mean())
        failures += int((done != 0).sum())
        now = env._now
        rear_kappa = now["kappa_power"][:, 2:4].mean(dim=1)
        speed = now["v"]
        yaw = now["r"]
        beta = now["beta"]
        mu = env.plant.col("D") * env.plant.tyre_scale.squeeze(1)
        util = (speed * yaw).abs() / (mu * 9.81).clamp(min=1e-6)
        gates = {
            "beta": (beta.abs() >= 0.20) & (beta.abs() <= 0.80),
            "slip": (rear_kappa >= 0.30) & (rear_kappa <= 5.0),
            "throttle": env.u_cmd[:, 1] >= 0.40,
            "counter": (env.u_cmd[:, 0] * yaw) < 0.0,
            "utilization": (util >= 0.55) & (util <= 1.35),
            "path": now["signed_pos_diff"].abs() <= env.offpath_limit,
        }
        strict = torch.ones_like(beta, dtype=torch.bool)
        for key, gate in gates.items():
            criterion_counts[key] += int(gate.sum())
            strict &= gate
        signal_sums["abs_beta"] += float(beta.abs().mean())
        signal_sums["rear_kappa"] += float(rear_kappa.mean())
        signal_sums["throttle"] += float(env.u_cmd[:, 1].mean())
        signal_sums["abs_steer"] += float(env.u_cmd[:, 0].abs().mean())
        signal_sums["utilization"] += float(util.mean())
        signal_sums["speed"] += float(speed.mean())
        strict_count += int(strict.sum())
        live_count += args.n
        for key, value in env.rew_stats.items():
            stats[key] = stats.get(key, 0.0) + value

    report = {
        "n": args.n, "steps": args.steps, "hardware_dr": args.hardware_dr,
        "plant_dr": args.plant_dr, "spin_frac": args.spin_frac,
        "beta_frac": args.beta_frac, "trim_frac": args.trim_frac,
        "finite": finite, "actor_shape": list(obs["obs"].shape),
        "critic_shape": list(obs["states"].shape),
        "mean_reward_per_step": reward_sum / args.steps,
        "reward_component_means": {k: v / args.steps for k, v in stats.items()},
        "failure_fraction_per_env_step": failures / max(live_count, 1),
        "strict_fraction": strict_count / max(live_count, 1),
        "criterion_fractions": {k: v / max(live_count, 1)
                                for k, v in criterion_counts.items()},
        "signal_means": {k: v / args.steps for k, v in signal_sums.items()},
        "max_abs_normalized_actor_obs": obs_abs_max,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if not finite or report["actor_shape"][1] != 27 or report["critic_shape"][1] != 52:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
