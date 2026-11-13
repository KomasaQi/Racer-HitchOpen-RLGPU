"""Use privileged feedback only to locate unstable CarSim drift equilibria.

Beta/Kappa feedback in this file is an identification fixture, never an actor
input.  Once a trim is located, small command perturbations around it provide
combined-slip data for the training surrogate.
"""
import argparse
import csv
import itertools
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from envs.carsim_flat_env import CarSimFlatEnv, MAX_ROAD_WHEEL_RAD  # noqa: E402

SCENARIOS = (("flat_mu050", 0.50), ("flat_mu075", 0.75), ("flat_mu100", 1.00))
TIRE_MU_PEAK = 2.1246


def run_case(scenario, mu, beta_target, counter, beta_kp, throttle_base,
             kappa_kp, entry_brake, entry_time, flick_time, entry_start,
             duration, dt):
    env = CarSimFlatEnv(scenario, dt=dt)
    env.reset()
    rows, beta_prev = [], 0.0
    try:
        for k in range(int(round(duration / dt))):
            t = k * dt
            raw0 = env.raw()
            speed0 = math.hypot(raw0["Vx"], raw0["Vy"]) / 3.6
            beta = math.radians(raw0["Beta"]) if speed0 >= 0.5 else 0.0
            kappa = 0.5 * (raw0["Kappa_L2"] + raw0["Kappa_R2"])
            if t < entry_start:
                steer, throttle, brake = 0.0, 0.40, 0.0
            elif t < entry_start + flick_time:
                steer, throttle, brake = -MAX_ROAD_WHEEL_RAD, 0.10, 0.0
            elif t < entry_start + flick_time + entry_time:
                steer, throttle, brake = MAX_ROAD_WHEEL_RAD, 0.0, entry_brake
            else:
                # Positive error at beta=0 keeps steering into the corner; as
                # the desired negative sideslip develops the command crosses to
                # the countersteer trim.
                steer = counter + beta_kp * (beta - beta_target)
                steer = float(np.clip(steer, -MAX_ROAD_WHEEL_RAD, MAX_ROAD_WHEEL_RAD))
                kappa_ctrl = float(np.clip(kappa, -2.0, 3.0))
                throttle = float(np.clip(throttle_base + kappa_kp * (0.70 - kappa_ctrl),
                                         0.35, 1.0))
                brake = 0.0
            env.step([steer / MAX_ROAD_WHEEL_RAD, throttle, brake])
            raw = env.raw()
            speed = math.hypot(raw["Vx"], raw["Vy"]) / 3.6
            beta = math.radians(raw["Beta"]) if speed >= 0.5 else 0.0
            yaw = math.radians(raw["AVz"])
            krl, krr = raw["Kappa_L2"], raw["Kappa_R2"]
            util = abs(speed * yaw) / max(TIRE_MU_PEAK * mu * 9.81, 1e-6)
            rows.append([t + dt, steer, throttle, speed, beta, yaw, krl, krr, util, brake])
            beta_prev = beta
    finally:
        env.close()
    a = np.asarray(rows)
    tail = a[a[:, 0] >= duration - 3.0]
    kappa = 0.5 * (tail[:, 6] + tail[:, 7])
    beta_abs = np.abs(tail[:, 4])
    strict = ((beta_abs >= 0.20) & (beta_abs <= 0.80) &
              (kappa >= 0.30) & (kappa <= 5.0) & (tail[:, 2] >= 0.40) &
              ((tail[:, 1] * tail[:, 5]) < 0.0) &
              (tail[:, 8] >= 0.55) & (tail[:, 8] <= 1.35))
    occ = float(strict.mean())
    score = (10.0 * occ - 1.5 * abs(beta_abs.mean() - abs(beta_target))
             - 0.4 * abs(np.clip(kappa, -10, 10).mean() - 0.70)
             - float(tail[:, 4].std()) - 0.25 * float(tail[:, 5].std()))
    summary = {
        "scenario": scenario, "mu": mu, "beta_target": beta_target,
        "counter": counter, "beta_kp": beta_kp,
        "throttle_base": throttle_base, "kappa_kp": kappa_kp,
        "entry_brake": entry_brake, "entry_time": entry_time,
        "flick_time": flick_time, "entry_start": entry_start,
        "strict_occupancy_tail": occ, "score": float(score),
        "speed_mean": float(tail[:, 3].mean()),
        "beta_mean": float(tail[:, 4].mean()), "beta_std": float(tail[:, 4].std()),
        "yaw_mean": float(tail[:, 5].mean()), "yaw_std": float(tail[:, 5].std()),
        "kappa_rear_mean": float(np.clip(kappa, -10, 10).mean()),
        "steer_mean": float(tail[:, 1].mean()),
        "throttle_mean": float(tail[:, 2].mean()),
        "friction_util_mean": float(tail[:, 8].mean()),
    }
    return summary, rows


def vals(text):
    return [float(x) for x in text.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenarios", default="flat_mu050,flat_mu075,flat_mu100")
    ap.add_argument("--beta-targets", default="-0.25,-0.40")
    ap.add_argument("--counters", default="-0.10,-0.20")
    ap.add_argument("--beta-kps", default="0.5,1.0")
    ap.add_argument("--throttle-bases", default="0.55,0.80")
    ap.add_argument("--kappa-kps", default="0.10,0.30")
    ap.add_argument("--entry-brakes", default="0.0")
    ap.add_argument("--entry-times", default="0.0")
    ap.add_argument("--flick-times", default="0.0")
    ap.add_argument("--entry-starts", default="2.0")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    selected = set(args.scenarios.split(","))
    grid = list(itertools.product(vals(args.beta_targets), vals(args.counters),
                                  vals(args.beta_kps), vals(args.throttle_bases),
                                  vals(args.kappa_kps), vals(args.entry_brakes),
                                  vals(args.entry_times), vals(args.flick_times),
                                  vals(args.entry_starts)))
    summaries = []
    for scenario, mu in SCENARIOS:
        if scenario not in selected:
            continue
        best = None
        for i, p in enumerate(grid):
            print("%-12s %3d/%d %s" % (scenario, i + 1, len(grid), p), flush=True)
            summary, rows = run_case(scenario, mu, *p, args.duration, args.dt)
            summaries.append(summary)
            if best is None or summary["score"] > best[0]["score"]:
                best = summary, rows
        with open(os.path.join(args.out, scenario + "__best.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "steer", "throttle", "speed", "beta", "yaw_rate",
                        "kappa_RL", "kappa_RR", "friction_util", "brake"])
            w.writerows(best[1])
        with open(os.path.join(args.out, scenario + "__best.json"), "w") as f:
            json.dump(best[0], f, indent=2)
    summaries.sort(key=lambda x: (x["scenario"], -x["score"]))
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)


if __name__ == "__main__":
    main()
