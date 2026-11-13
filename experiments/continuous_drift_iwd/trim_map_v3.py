"""Build a deduplicated exact trim map before creating any drift teacher."""
import argparse
import json
import os

import numpy as np

from trim_solve_v3 import TrimV3


def csv_floats(text):
    return [float(x) for x in text.split(",") if x.strip()]


def unique_roots(candidates, tol=2e-3):
    roots = []
    for err, cost, z in sorted(candidates, key=lambda x: (x[0], x[1])):
        if err >= 1e-3:
            continue
        if any(np.linalg.norm((z - old[2]) / np.array(
                [0.3, 0.3, 0.4, 0.2, 0.2, 0.5, 0.5])) < tol for old in roots):
            continue
        roots.append((err, cost, z.copy()))
    return roots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", default="1.0623,1.59345,2.1246")
    ap.add_argument("--curvature", default="0.05,0.075,0.10,0.125,0.15")
    ap.add_argument("--limit-frac", default="0.60,0.75,0.90")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for mu in csv_floats(args.mu):
        solver = TrimV3(mu)
        for curvature in csv_floats(args.curvature):
            for frac in csv_floats(args.limit_frac):
                speed = frac * np.sqrt(mu * 9.81 / abs(curvature))
                _, candidates = solver.solve(curvature, speed)
                roots = unique_roots(candidates)
                encoded = []
                for err, cost, z in roots:
                    powered = (abs(z[0]) >= 0.20 and z[2] >= 0.25
                               and np.mean(z[5:7]) >= 0.25)
                    encoded.append({
                        "max_abs_residual": float(err), "cost": float(cost),
                        "beta": float(z[0]), "delta": float(z[1]),
                        "throttle": float(z[2]),
                        "kappa": [float(x) for x in z[3:7]],
                        "drift_sign": bool(z[0] * curvature < 0.0),
                        "countersteer": bool(z[1] * curvature < 0.0),
                        "powered_slide": bool(powered),
                        "quality_branch": bool(powered and z[0] * curvature < 0.0
                                               and z[1] * curvature < 0.0),
                    })
                rows.append({
                    "effective_mu": mu, "curvature": curvature,
                    "limit_fraction": frac, "speed": float(speed),
                    "n_unique_roots": len(encoded), "roots": encoded,
                })
                print("mu %.5f k %.4f frac %.2f roots %d quality %d" %
                      (mu, curvature, frac, len(encoded),
                       sum(int(x["quality_branch"]) for x in encoded)), flush=True)
    report = {"schema": 1, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
