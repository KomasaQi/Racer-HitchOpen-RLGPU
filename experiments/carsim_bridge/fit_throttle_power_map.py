"""Fit a two-parameter pedal-to-demand-power map on 0.1/0.3/0.5 launches."""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))
from fast_plant import FastRacerPlant, load_preset, PARAM_INDEX  # noqa: E402

FIT = ("launch_10", "launch_30", "launch_50")


def load_batch(dataset):
    cases = []
    for path in sorted(glob.glob(os.path.join(dataset, "*.csv"))):
        x = pd.read_csv(path)
        maneuver = str(x.maneuver.iloc[0])
        if maneuver not in FIT:
            continue
        cs = x[x.plant == "carsim"].sort_values("t")
        q = float(cs.throttle_cmd.iloc[0])
        target = cs[["speed", "motor_speed", "surface_speed_RL", "surface_speed_RR"]].to_numpy()
        cases.append((float(cs.mu.iloc[0]), q, cs.t.to_numpy(), target))
    if len({len(c[2]) for c in cases}) != 1:
        raise ValueError("fit launches must have equal timelines")
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cases = load_batch(args.dataset)
    n = len(cases)
    params = load_preset("fullsize_racecar_ev")
    plant = FastRacerPlant(params, n, "cpu")
    tyre_mu = torch.tensor([c[0] for c in cases], dtype=plant.dtype).unsqueeze(1)
    throttle = torch.tensor([c[1] for c in cases], dtype=plant.dtype)
    timeline = cases[0][2]
    target = np.stack([c[3] for c in cases], axis=1)  # [time, env, signal]
    scales = np.array([10.0, 3000.0, 10.0, 10.0])
    cache = {}

    def objective(x):
        gain, rolloff = float(x[0]), float(x[1])
        if not (0.5 <= gain <= 2.0 and 0.0 <= rolloff <= 2.0):
            return 1e3 + (gain - 1.0) ** 2 + rolloff ** 2
        key = (round(gain, 7), round(rolloff, 7))
        if key in cache:
            return cache[key]
        plant.reset_randomization()
        plant.p[:, PARAM_INDEX["throttle_power_gain"]] = gain
        plant.p[:, PARAM_INDEX["throttle_power_rolloff"]] = rolloff
        plant.tyre_scale.copy_(tyre_mu)
        plant.reset_guess()
        state = torch.zeros((n, 10))
        plant.init_wheels(state)
        cmd = torch.stack((torch.zeros_like(throttle), throttle, torch.zeros_like(throttle)), 1)
        pred, last_t = [], 0.0
        for t in timeline:
            dt = float(t - last_t)
            state, ex = plant.step(state, cmd, dt=dt, substeps=max(round(dt / 0.01), 1))
            radius = plant.geometry()[2]
            surface = state[:, 6:10] * radius
            speed = torch.hypot(state[:, 3], state[:, 4])
            rpm = ex["motor_speed"] * 60.0 / (2.0 * math.pi)
            pred.append(torch.stack((speed, rpm, surface[:, 2], surface[:, 3]), 1).numpy())
            last_t = float(t)
        pred = np.stack(pred)
        mask = target[:, :, 0] >= 1.0
        err = (pred - target) / scales
        value = float(np.mean(err[mask] ** 2))
        cache[key] = value
        print("gain=%.5f rolloff=%.5f objective=%.7f" % (gain, rolloff, value), flush=True)
        return value

    fit = minimize(objective, np.array([1.15, 0.70]), method="Nelder-Mead",
                   options={"xatol": 2e-3, "fatol": 2e-5, "maxiter": 32})
    result = {"throttle_power_gain": float(fit.x[0]),
              "throttle_power_rolloff": float(fit.x[1]),
              "objective": float(fit.fun), "success": bool(fit.success),
              "cases": n, "fit_maneuvers": list(FIT), "evaluations": len(cache)}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
