"""Fit the compact viscous LSD on a true CarSim drift-state replay.

All candidates start from the same CarSim state at ``start`` and receive the
same physical commands.  The short local window isolates differential/wheel
dynamics from drift initiation and path-controller differences.
"""
import argparse
import csv
import itertools
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))
from fast_plant import FastRacerPlant, PARAM_INDEX, load_preset  # noqa: E402

WHEELS = ("FL", "FR", "RL", "RR")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenario", default="flat_mu075")
    ap.add_argument("--start", type=float, default=6.0)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--dt", type=float, default=0.05)
    args = ap.parse_args()
    path = os.path.join(args.dataset, args.scenario + "__drift_trim_prbs.csv")
    with open(path, newline="") as f:
        carsim = [r for r in csv.DictReader(f) if r["plant"] == "carsim"]
    i0 = min(range(len(carsim)), key=lambda i: abs(float(carsim[i]["t"]) - args.start))
    i1 = min(len(carsim), i0 + int(round(args.duration / args.dt)) + 1)
    target = carsim[i0:i1]

    kd_grid = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
    cap_grid = (25.0, 50.0, 100.0, 200.0, 300.0, 600.0)
    candidates = list(itertools.product(kd_grid, cap_grid))
    plant = FastRacerPlant(load_preset("fullsize_racecar_ev"), len(candidates), "cpu")
    plant.tyre_scale.fill_(float(target[0]["mu"]))
    plant.p[:, PARAM_INDEX["lsd_kd"]] = torch.tensor([x[0] for x in candidates])
    plant.p[:, PARAM_INDEX["lsd_tau_max"]] = torch.tensor([x[1] for x in candidates])

    s = torch.zeros((len(candidates), 10))
    r0 = target[0]
    s[:, 3] = float(r0["vx_body"])
    s[:, 4] = float(r0["vy_body"])
    s[:, 5] = float(r0["yaw_rate"])
    for j, w in enumerate(WHEELS):
        s[:, 6 + j] = float(r0["omega_" + w])
    plant._ax_guess.fill_(float(r0["ax"]))
    plant._ay_guess.fill_(float(r0["ay"]))

    sq = torch.zeros(len(candidates))
    count = 0
    traces = [[] for _ in candidates]
    for row in target[1:]:
        cmd = torch.tensor([[float(row["steer_cmd"]), float(row["throttle_cmd"]),
                             float(row["brake_cmd"])]]).repeat(len(candidates), 1)
        s, ex = plant.step(s, cmd, dt=args.dt, substeps=5, mode="pedals")
        psi = s[:, 2]
        vxb = torch.cos(psi) * s[:, 3] + torch.sin(psi) * s[:, 4]
        vyb = -torch.sin(psi) * s[:, 3] + torch.cos(psi) * s[:, 4]
        beta = torch.atan2(vyb, vxb)
        k = ex["kappa_power"][:, 2:4]
        target_vec = torch.tensor([
            float(row["yaw_rate"]), float(row["beta"]),
            float(row["kappa_RL"]), float(row["kappa_RR"]),
            float(row["fy_RL"]), float(row["fy_RR"]),
        ])
        pred = torch.stack([s[:, 5], beta, k[:, 0], k[:, 1],
                            ex["fy_wheel"][:, 2], ex["fy_wheel"][:, 3]], dim=1)
        scales = torch.tensor([1.0, 0.4, 2.0, 2.0, 10000.0, 10000.0])
        err = (pred - target_vec) / scales
        sq += (err * err).mean(dim=1)
        count += 1
        for i in range(len(candidates)):
            traces[i].append([float(x) for x in pred[i]])

    score = torch.sqrt(sq / max(count, 1)).numpy()
    order = np.argsort(score)
    results = []
    for i in order:
        results.append({"lsd_kd": candidates[i][0], "lsd_tau_max": candidates[i][1],
                        "normalized_rmse": float(score[i]),
                        "final": traces[i][-1]})
    report = {"scenario": args.scenario, "start": args.start,
              "duration": args.duration, "signals": ["yaw", "beta", "kappa_RL",
              "kappa_RR", "fy_RL", "fy_RR"], "results": results}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"best": results[:12]}, indent=2))


if __name__ == "__main__":
    main()
