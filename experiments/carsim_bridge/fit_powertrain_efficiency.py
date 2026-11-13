"""Fit one effective motor-efficiency scalar on longitudinal fit maneuvers."""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))

from experiments.carsim_bridge.system_id_dataset import command  # noqa: E402
from fast_plant import FastRacerPlant, load_preset  # noqa: E402

FIT = {"launch_10", "launch_30", "launch_50"}


def simulate(mu, maneuver, timeline, efficiency):
    params = load_preset("fullsize_racecar_ev")
    params["motor_efficiency"] = float(efficiency)
    plant = FastRacerPlant(params, 1, "cpu")
    plant.tyre_scale.fill_(mu)
    state = torch.zeros((1, 10))
    plant.init_wheels(state)
    radius = plant.geometry()[2][0]
    rows = []
    last_t = 0.0
    for t in timeline:
        dt = float(t - last_t)
        delta, throttle, brake = command(maneuver, last_t)
        cmd = torch.tensor([[delta, throttle, brake]], dtype=state.dtype)
        state, ex = plant.step(state, cmd, dt=dt, substeps=max(round(dt / 0.01), 1))
        speed = float(torch.hypot(state[0, 3], state[0, 4]))
        surface = state[0, 6:10] * radius
        rpm = float(ex["motor_speed"][0]) * 60.0 / (2.0 * math.pi)
        rows.append((speed, rpm, float(surface[2]), float(surface[3])))
        last_t = float(t)
    return np.asarray(rows)


def load_cases(dataset):
    cases = []
    for path in sorted(glob.glob(os.path.join(dataset, "*.csv"))):
        frame = pd.read_csv(path)
        maneuver = str(frame.maneuver.iloc[0])
        if maneuver not in FIT:
            continue
        cs = frame[frame.plant == "carsim"].sort_values("t")
        target = cs[["speed", "motor_speed", "surface_speed_RL", "surface_speed_RR"]].to_numpy()
        cases.append((str(cs.scenario.iloc[0]), float(cs.mu.iloc[0]), maneuver,
                      cs.t.to_numpy(), target))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cases = load_cases(args.dataset)
    cache = {}

    def objective(eff):
        key = round(float(eff), 8)
        if key in cache:
            return cache[key]
        total, count = 0.0, 0
        scales = np.array([10.0, 3000.0, 10.0, 10.0])
        for _, mu, maneuver, timeline, target in cases:
            pred = simulate(mu, maneuver, timeline, eff)
            mask = target[:, 0] >= 1.0
            err = (pred[mask] - target[mask]) / scales
            total += float(np.sum(err * err))
            count += int(err.size)
        value = total / max(count, 1)
        cache[key] = value
        print("motor_eff=%.5f objective=%.7f" % (eff, value), flush=True)
        return value

    fit = minimize_scalar(objective, bounds=(0.50, 0.98), method="bounded",
                          options={"xatol": 2e-3, "maxiter": 24})
    result = {"motor_efficiency": float(fit.x), "objective": float(fit.fun),
              "success": bool(fit.success), "cases": len(cases),
              "fit_maneuvers": sorted(FIT), "evaluations": cache}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "evaluations"}, indent=2))


if __name__ == "__main__":
    main()
