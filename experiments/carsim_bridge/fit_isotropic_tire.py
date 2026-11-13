"""Fit the best possible XCAR radial Pacejka curve to CarSim drift samples.

This is a structural gate, not merely another parameter fit.  One shared
``mu(hypot(kappa,tan(alpha)))`` curve is projected along the combined-slip
vector exactly as in XCAR.  If its best attainable force error remains much
larger than the CarSim transition model, a nominal FastPlant cannot honestly
be called CarSim-aligned while using a strictly isotropic tyre.
"""
import argparse
import csv
import json
import math

import numpy as np
from scipy.optimize import least_squares


def magic(s, b, c, d, e):
    bs = b * s
    return d * np.sin(c * np.arctan(bs - e * (bs - np.arctan(bs))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.samples, newline="") as f:
        rows = list(csv.DictReader(f))
    kappa = np.abs(np.asarray([float(r["kappa"]) for r in rows]))
    tan_alpha = np.abs(np.tan([float(r["alpha_rad"]) for r in rows]))
    fz = np.maximum(np.asarray([float(r["fz"]) for r in rows]), 1.0)
    fx = np.asarray([float(r["fx_actual"]) for r in rows])
    fy = np.asarray([float(r["fy_actual"]) for r in rows])
    peak_x = np.maximum(np.asarray([float(r["peak_x"]) for r in rows]), 1.0)
    peak_y = np.maximum(np.asarray([float(r["peak_y"]) for r in rows]), 1.0)
    road = np.asarray([0.50 if r["scenario"] == "flat_mu050" else 0.75
                       for r in rows])
    scenario = np.asarray([r["scenario"] for r in rows])
    slip = np.hypot(kappa, tan_alpha)
    ux = np.divide(kappa, slip, out=np.zeros_like(slip), where=slip > 1e-12)
    uy = np.divide(tan_alpha, slip, out=np.zeros_like(slip), where=slip > 1e-12)

    # Load sensitivity is retained because it is a separate, already calibrated
    # FastPlant module rather than longitudinal/lateral tyre anisotropy.  Use the
    # exact axle references in the production preset so the fitted parameters
    # can be replayed without changing the load-transfer contract.
    fz_ref = np.asarray([3460.843 if r["wheel"].startswith("F") else 5537.353
                         for r in rows])

    def predict(q):
        b, c, d, e, load_sens = q
        d_eff = d * np.clip(fz / fz_ref, 0.1, 4.0) ** (-load_sens)
        mu = magic(slip, b, c, d_eff, e)
        return road * fz * mu * ux, road * fz * mu * uy

    def residual(q):
        px, py = predict(q)
        return np.concatenate(((px - fx) / peak_x, (py - fy) / peak_y))

    starts = ([10.0, 1.9, 2.0, 0.97, 0.15],
              [25.0, 1.85, 2.1, 0.95, 0.15],
              [5.0, 1.4, 2.0, 0.0, 0.05])
    fits = [least_squares(residual, x, bounds=([0.1, 0.5, 0.1, -2.0, 0.0],
                                                [80.0, 2.5, 4.0, 2.0, 0.6]),
                         max_nfev=3000)
            for x in starts]
    fit = min(fits, key=lambda z: np.mean(residual(z.x) ** 2))
    px, py = predict(fit.x)

    def score(mask):
        err = np.concatenate(((px[mask] - fx[mask]) / peak_x[mask],
                              (py[mask] - fy[mask]) / peak_y[mask]))
        direction_actual = np.arctan2(fy[mask], fx[mask])
        direction_model = np.arctan2(py[mask], px[mask])
        return {"count": int(mask.sum()),
                "normalized_rmse": float(np.sqrt(np.mean(err ** 2))),
                "normalized_mae": float(np.mean(np.abs(err))),
                "force_direction_rmse_rad": float(np.sqrt(np.mean(
                    (direction_model - direction_actual) ** 2)))}

    report = {
        "model": "XCAR isotropic radial Pacejka",
        "parameters": dict(zip(("B", "C", "D", "E", "load_sensitivity"),
                               map(float, fit.x))),
        "fz_ref_front": 3460.843, "fz_ref_rear": 5537.353,
        "success": bool(fit.success),
        "all": score(np.ones(len(rows), dtype=bool)),
        "by_scenario": {s: score(scenario == s) for s in sorted(set(scenario))},
        "comparison_transition_rmse": 0.09859207612581752,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
