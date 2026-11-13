"""Fit compact Magic-Formula shape/load parameters to extracted CarSim carpets."""
import argparse
import json
import math

import numpy as np
from scipy.optimize import least_squares


def mf(x, b, c, e):
    bx = b * x
    return np.sin(c * np.arctan(bx - e * (bx - np.arctan(bx))))


def fit_one(table, kind, fixed_e=None):
    x = np.asarray(table["slip"], float)
    if kind == "fy":
        x = np.radians(x)
    fz = np.asarray(table["fz"], float)
    force = np.asarray(table["force"], float)
    peak = force.max(axis=0)
    norm = force / np.maximum(peak, 1e-9)
    xx = np.repeat(x[:, None], len(fz), axis=1).ravel()
    yy = norm.ravel()
    guess = (22.0, 1.9, 0.97) if kind == "fx" else (8.0, 1.9, 0.97)
    if fixed_e is None:
        fit = least_squares(lambda p: mf(xx, *p) - yy, guess,
                            bounds=([0.1, 1.0, 0.0], [100.0, 2.5, 1.5]))
        b, c, e = fit.x
    else:
        fit = least_squares(lambda p: mf(xx, p[0], p[1], fixed_e) - yy,
                            guess[:2], bounds=([0.1, 1.0], [100.0, 2.5]))
        b, c, e = fit.x[0], fit.x[1], float(fixed_e)
    pred = mf(xx, b, c, e)
    shape_rmse = float(np.sqrt(np.mean((pred - yy) ** 2)))

    mu_peak = peak / fz
    f_ref = float(np.exp(np.mean(np.log(fz))))
    # log(mu)=log(mu_ref)-s*log(Fz/Fref)
    a = np.column_stack([np.ones_like(fz), -np.log(fz / f_ref)])
    coef, _, _, _ = np.linalg.lstsq(a, np.log(mu_peak), rcond=None)
    mu_ref, sens = float(np.exp(coef[0])), float(coef[1])
    mu_pred = mu_ref * (fz / f_ref) ** (-sens)
    load_rmse = float(np.sqrt(np.mean((mu_pred - mu_peak) ** 2)))
    fine = np.linspace(0.0, max(x), 10001)
    peak_x = float(fine[np.argmax(mf(fine, b, c, e))])
    return {"B": float(b), "C": float(c), "E": float(e),
            "shape_rmse": shape_rmse, "peak_slip": peak_x,
            "fz_ref": f_ref, "mu_ref": mu_ref, "load_exponent": sens,
            "load_mu_rmse": load_rmse,
            "source_peak_mu": [float(v) for v in mu_peak]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixed-e", type=float, default=None,
                    help="fix Pacejka curvature E (the minimal model uses E=1)")
    args = ap.parse_args()
    with open(args.tables, encoding="utf-8") as f:
        data = json.load(f)
    result = {"source": args.tables, "model": "compact MF + power-law load sensitivity",
              "fits": {}}
    for name, table in sorted(data["tables"].items()):
        kind = name.split("_")[1]
        result["fits"][name] = fit_one(table, kind, args.fixed_e)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    for name, fit in result["fits"].items():
        unit = "deg" if name.endswith("fy") else "ratio"
        peak = math.degrees(fit["peak_slip"]) if unit == "deg" else fit["peak_slip"]
        print("%-10s B=%7.3f C=%5.3f E=%5.3f peak=%7.3f %-5s mu_ref=%5.3f "
              "load_exp=%5.3f shape_rmse=%6.4f" %
              (name, fit["B"], fit["C"], fit["E"], peak, unit,
               fit["mu_ref"], fit["load_exponent"], fit["shape_rmse"]))


if __name__ == "__main__":
    main()
