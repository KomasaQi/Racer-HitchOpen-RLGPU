"""Reproducible FastPlant/CarSim open-loop alignment report.

The comparison starts both plants from physical rest, applies the same zero-order
held pedal command at 20 Hz, and records the channels that matter to the actor and
the powered-drift contract.  It does not tune parameters; it produces evidence for
the next calibration step.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))

from envs.carsim_flat_env import CarSimFlatEnv, CARSIM_WHEEL_SUFFIXES  # noqa: E402
from fast_plant import FastRacerPlant, load_preset  # noqa: E402


def gpu_trace(mu, throttle, seconds, dt=0.05):
    plant = FastRacerPlant(load_preset("fullsize_racecar_ev"), 1, "cpu")
    plant.tyre_scale.fill_(mu)
    s = torch.zeros((1, 10))
    plant.init_wheels(s)
    cmd = torch.tensor([[0.0, throttle, 0.0]])
    rows = []
    for k in range(int(round(seconds / dt))):
        s, ex = plant.step(s, cmd, dt=dt, substeps=5, mode="pedals")
        v = float(torch.hypot(s[0, 3], s[0, 4]))
        wheel = (s[0, 6:10] * plant.geometry()[2][0]).numpy()
        rows.append(["fast", (k + 1) * dt, v, *wheel,
                     float(ex["kappa_power"][0, 2:4].mean()), float(ex["ax"][0])])
    return rows


def carsim_trace(scenario, throttle, seconds, dt=0.05):
    env = CarSimFlatEnv(scenario, dt=dt)
    env.reset()
    rows = []
    try:
        for k in range(int(round(seconds / dt))):
            env.step([0.0, throttle, 0.0])
            raw = env.raw()
            v = float(torch.hypot(env.s[0, 3], env.s[0, 4]))
            wheel = (env.s[0, 6:10] * 0.332).numpy()
            kap = np.mean([raw["Kappa_" + s] for s in CARSIM_WHEEL_SUFFIXES[2:4]])
            rows.append(["carsim", (k + 1) * dt, v, *wheel, float(kap),
                         float(raw["Ax"] * 9.81)])
    finally:
        env.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--throttles", default="0.1,0.3,0.5,0.8")
    ap.add_argument("--fast-mu-floor", type=float, default=0.0,
                    help="test affine FastPlant scale floor+(1-floor)*road_mu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    throttles = [float(x) for x in args.throttles.split(",")]
    summary = []
    for scenario, mu in (("flat_mu050", 0.50), ("flat_mu075", 0.75),
                         ("flat_mu100", 1.00)):
        for throttle in throttles:
            fast_mu = args.fast_mu_floor + (1.0 - args.fast_mu_floor) * mu
            g = gpu_trace(fast_mu, throttle, args.seconds)
            c = carsim_trace(scenario, throttle, args.seconds)
            path = os.path.join(args.out, "%s_thr%03d.csv" % (scenario, round(100 * throttle)))
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["plant", "t", "v", "wFL", "wFR", "wRL", "wRR",
                            "kappa_rear", "ax"])
                w.writerows(g + c)
            ga, ca = np.asarray([r[2:] for r in g], float), np.asarray([r[2:] for r in c], float)
            summary.append([scenario, mu, throttle, ga[-1, 0], ca[-1, 0],
                            float(np.sqrt(np.mean((ga[:, 0] - ca[:, 0]) ** 2))),
                            ga[-1, 5], ca[-1, 5],
                            float(np.sqrt(np.mean((ga[:, 5] - ca[:, 5]) ** 2)))])
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "mu", "throttle", "v_final_fast", "v_final_carsim",
                    "v_rmse", "kappa_final_fast", "kappa_final_carsim", "kappa_rmse"])
        w.writerows(summary)
    print("%-12s %4s %8s %8s %8s %8s %8s" %
          ("scenario", "thr", "V_fast", "V_cs", "V_rmse", "K_fast", "K_cs"))
    for r in summary:
        print("%-12s %4.1f %8.2f %8.2f %8.2f %8.3f %8.3f" %
              (r[0], r[2], r[3], r[4], r[5], r[6], r[7]))


if __name__ == "__main__":
    main()
