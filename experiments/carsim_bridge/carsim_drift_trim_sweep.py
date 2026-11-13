"""Search open-loop entry/countersteer schedules for CarSim drift equilibria.

This is not a controller benchmark.  It locates physically realizable regions
around which combined-slip system-identification perturbations can be designed.
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


def command(t, throttle, counter, entry_steer, entry_time):
    if t < 2.0:
        return 0.0, 0.40, 0.0
    if t < 2.0 + entry_time:
        return entry_steer, throttle, 0.0
    return counter, throttle, 0.0


def run_case(scenario, mu, throttle, counter, entry_steer, entry_time,
             duration, dt):
    env = CarSimFlatEnv(scenario, dt=dt)
    env.reset()
    rows = []
    try:
        for k in range(int(round(duration / dt))):
            t = k * dt
            steer, q, brake = command(t, throttle, counter, entry_steer, entry_time)
            env.step([steer / MAX_ROAD_WHEEL_RAD, q, brake])
            raw = env.raw()
            speed = math.hypot(raw["Vx"], raw["Vy"]) / 3.6
            beta = math.radians(raw["Beta"]) if speed >= 0.5 else 0.0
            yaw = math.radians(raw["AVz"])
            krl, krr = raw["Kappa_L2"], raw["Kappa_R2"]
            util = abs(speed * yaw) / max(TIRE_MU_PEAK * mu * 9.81, 1e-6)
            rows.append([t + dt, steer, q, speed, beta, yaw, krl, krr, util])
    finally:
        env.close()
    a = np.asarray(rows, dtype=float)
    tail = a[a[:, 0] >= duration - 2.0]
    beta_abs = np.abs(tail[:, 4])
    kappa = 0.5 * (tail[:, 6] + tail[:, 7])
    counter_ok = (tail[:, 1] * tail[:, 5]) < 0.0
    finite = np.isfinite(tail).all(axis=1)
    strict = (finite & (beta_abs >= 0.25) & (beta_abs <= 0.80) &
              (kappa >= 0.35) & (kappa <= 5.0) & (tail[:, 2] >= 0.40) &
              counter_ok & (tail[:, 8] >= 0.65) & (tail[:, 8] <= 1.25))
    # Rank by simultaneous occupancy first, then proximity to a useful steady
    # operating point.  Large variance and spins are explicitly costly.
    occupancy = float(strict.mean())
    beta_mean = float(beta_abs.mean())
    kappa_mean = float(np.clip(kappa, -10.0, 10.0).mean())
    score = (8.0 * occupancy
             - abs(beta_mean - 0.40)
             - 0.4 * abs(kappa_mean - 0.70)
             - 0.5 * float(np.std(tail[:, 4]))
             - 0.2 * float(np.std(tail[:, 5])))
    summary = {
        "scenario": scenario, "mu": mu, "throttle": throttle,
        "countersteer": counter, "entry_steer": entry_steer,
        "entry_time": entry_time, "strict_occupancy_tail": occupancy,
        "score": score, "speed_mean": float(tail[:, 3].mean()),
        "beta_abs_mean": beta_mean, "beta_std": float(tail[:, 4].std()),
        "yaw_abs_mean": float(np.abs(tail[:, 5]).mean()),
        "kappa_rear_mean": kappa_mean,
        "friction_util_mean": float(tail[:, 8].mean()),
        "finite": bool(finite.all()),
    }
    return summary, rows


def parse_values(text):
    return [float(x) for x in text.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenarios", default="flat_mu050,flat_mu075,flat_mu100")
    ap.add_argument("--throttles", default="0.45,0.65,0.85")
    ap.add_argument("--countersteers", default="-0.06,-0.12,-0.18")
    ap.add_argument("--entry-steers", default="0.14,0.20")
    ap.add_argument("--entry-times", default="0.35,0.60")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--dt", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    selected = set(args.scenarios.split(","))
    grid = itertools.product(parse_values(args.throttles),
                             parse_values(args.countersteers),
                             parse_values(args.entry_steers),
                             parse_values(args.entry_times))
    grid = list(grid)
    summaries = []
    for scenario, mu in SCENARIOS:
        if scenario not in selected:
            continue
        best = None
        for i, (q, counter, entry, entry_time) in enumerate(grid):
            print("%-12s %3d/%d q=%.2f counter=%.2f entry=%.2f/%.2f" %
                  (scenario, i + 1, len(grid), q, counter, entry, entry_time), flush=True)
            summary, rows = run_case(scenario, mu, q, counter, entry, entry_time,
                                     args.duration, args.dt)
            summaries.append(summary)
            if best is None or summary["score"] > best[0]["score"]:
                best = (summary, rows)
        if best:
            with open(os.path.join(args.out, scenario + "__best.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "steer", "throttle", "speed", "beta", "yaw_rate",
                            "kappa_RL", "kappa_RR", "friction_util"])
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
