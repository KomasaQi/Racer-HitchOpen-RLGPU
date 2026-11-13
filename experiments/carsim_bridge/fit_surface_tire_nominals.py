"""Fit one compact analytic tyre nominal for each CarSim road surface.

The fit deliberately uses CarSim *outputs* from each flat_mu scenario, not the
source tyre carpet.  Longitudinal and lateral B/C/D/E are independent.  The
result is the evidence used to construct domain-randomisation bounds that cover
all three surface nominals plus uncertainty margin.
"""
import argparse
import csv
import glob
import json
import math
import os

import numpy as np
from scipy.optimize import least_squares


WHEELS = ("FL", "FR", "RL", "RR")
LONG_MANEUVERS = ("launch_", "brake_", "throttle_prbs", "coast")
LAT_MANEUVERS = ("steer_small", "steer_large", "steer_peak")


def mf(s, b, c, d, e):
    bs = b * s
    return d * np.sin(c * np.arctan(bs - e * (bs - np.arctan(bs))))


def fit_axis(samples):
    slip = np.asarray([s[0] for s in samples])
    fz = np.asarray([s[1] for s in samples])
    fz_ref = np.asarray([s[2] for s in samples])
    target = np.asarray([s[3] for s in samples])

    def residual(q):
        b, c, d, e, sens = q
        pred = mf(slip, b, c, d * np.clip(fz / fz_ref, .1, 4.) ** (-sens), e)
        return pred - target

    starts = ([10., 1.9, max(np.quantile(target, .98), .2), .9, .15],
              [25., 1.6, max(np.quantile(target, .98), .2), .5, .05],
              [5., 2.1, max(np.quantile(target, .98), .2), 0., .25])
    fits = [least_squares(residual, x,
                         bounds=([.1, .5, .05, -2., 0.],
                                 [100., 2.5, 4., 2., .8]),
                         loss="soft_l1", f_scale=.05, max_nfev=4000)
            for x in starts]
    fit = min(fits, key=lambda z: np.mean(residual(z.x) ** 2))
    err = residual(fit.x)
    return {"B": float(fit.x[0]), "C": float(fit.x[1]),
            "D": float(fit.x[2]), "E": float(fit.x[3]),
            "load_sensitivity": float(fit.x[4]),
            "count": int(len(samples)),
            "mu_rmse": float(np.sqrt(np.mean(err ** 2))),
            "mu_mae": float(np.mean(np.abs(err))),
            "success": bool(fit.success)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, nargs="+",
                    help="one or more paired system-ID dataset directories")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    scenarios = {}
    paths = []
    for dataset in args.dataset:
        paths.extend(glob.glob(os.path.join(dataset, "flat_mu*__*.csv")))
    for path in paths:
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["plant"] == "carsim"]
        if not rows:
            continue
        scenario, maneuver = rows[0]["scenario"], rows[0]["maneuver"]
        bucket = scenarios.setdefault(scenario, {"x": [], "y": [],
                                                  "road_scale": float(rows[0]["mu"])})
        for r in rows:
            if float(r["speed"]) < 2.0:
                continue
            for wheel in WHEELS:
                fz = float(r["fz_" + wheel])
                if fz < 500.0:
                    continue
                fz_ref = 3460.843 if wheel.startswith("F") else 5537.353
                kappa = abs(float(r["kappa_" + wheel]))
                alpha = abs(float(r["alpha_" + wheel]))
                if maneuver.startswith(LONG_MANEUVERS) and alpha < .02 and kappa > .002:
                    bucket["x"].append((kappa, fz, fz_ref,
                                        abs(float(r["fx_" + wheel])) / fz))
                if maneuver in LAT_MANEUVERS and kappa < .02 and alpha > .002:
                    bucket["y"].append((abs(math.tan(alpha)), fz, fz_ref,
                                        abs(float(r["fy_" + wheel])) / fz))

    report = {"model": "per-surface compact analytic Pacejka",
              "source": [os.path.abspath(x) for x in args.dataset], "scenarios": {}}
    for scenario, data in sorted(scenarios.items()):
        report["scenarios"][scenario] = {
            "road_scale": data["road_scale"],
            "longitudinal": fit_axis(data["x"]),
            "lateral": fit_axis(data["y"]),
        }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
