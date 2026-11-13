"""Compare CarSim drift forces with candidate combined-slip projections.

The pure Fx/Fy carpets are interpolated at each observed slip and wheel load.
Candidate p-norm projections then scale both pure forces onto a shared envelope.
This does not assume that CarSim uses a friction ellipse; it measures whether
that approximation is supported in the actual drift operating region.
"""
import argparse
import csv
import json
import math
import os

import numpy as np


WHEELS = ("FL", "FR", "RL", "RR")


def interp_table(table, slip, fz):
    slips = np.asarray(table["slip"], dtype=float)
    loads = np.asarray(table["fz"], dtype=float)
    force = np.asarray(table["force"], dtype=float)
    fz_c = float(np.clip(fz, loads[0], loads[-1]))
    at_load = np.asarray([np.interp(fz_c, loads, row) for row in force])
    # The exported carpets begin at a small nonzero slip.  Their physical
    # origin is exactly (0, 0); clamping to the first row would invent force at
    # zero slip and badly bias the lightly braked/rolling front wheels.
    slips_i = np.concatenate(([0.0], slips))
    forces_i = np.concatenate(([0.0], at_load))
    value = float(np.interp(np.clip(abs(slip), 0.0, slips[-1]), slips_i, forces_i))
    peak_i = int(np.argmax(at_load))
    return value, float(at_load[peak_i]), float(slips[peak_i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tire-tables", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, default=6.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(args.tire_tables, encoding="utf-8") as f:
        tables = json.load(f)["tables"]

    samples = []
    for name in sorted(os.listdir(args.dataset)):
        if not name.endswith("__drift_trim_prbs.csv"):
            continue
        path = os.path.join(args.dataset, name)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["plant"] != "carsim" or float(row["t"]) < args.start:
                    continue
                road = float(row["mu"])
                for wheel in WHEELS:
                    axle = "front" if wheel.startswith("F") else "rear"
                    fz = float(row["fz_" + wheel])
                    pure_x, peak_x, peak_slip_x = interp_table(
                        tables[axle + "_fx"], float(row["kappa_" + wheel]), fz)
                    pure_y, peak_y, peak_slip_y = interp_table(
                        tables[axle + "_fy"], math.degrees(float(row["alpha_" + wheel])), fz)
                    pure_x, peak_x = road * pure_x, road * peak_x
                    pure_y, peak_y = road * pure_y, road * peak_y
                    samples.append({
                        "scenario": row["scenario"], "t": float(row["t"]), "wheel": wheel,
                        "fz": fz, "kappa": float(row["kappa_" + wheel]),
                        "alpha_rad": float(row["alpha_" + wheel]),
                        "fx_actual": abs(float(row["fx_" + wheel])),
                        "fy_actual": abs(float(row["fy_" + wheel])),
                        "fx_pure": pure_x, "fy_pure": pure_y,
                        "peak_x": peak_x, "peak_y": peak_y,
                        "peak_slip_x": peak_slip_x, "peak_slip_y_deg": peak_slip_y,
                    })

    p_grid = np.linspace(1.0, 10.0, 181)
    scores = []
    for p in p_grid:
        errors = []
        for s in samples:
            nx = s["fx_pure"] / max(s["peak_x"], 1.0)
            ny = s["fy_pure"] / max(s["peak_y"], 1.0)
            q = (nx ** p + ny ** p) ** (1.0 / p)
            scale = min(1.0, 1.0 / max(q, 1e-9))
            errors.extend(((scale * s["fx_pure"] - s["fx_actual"]) / max(s["peak_x"], 1.0),
                           (scale * s["fy_pure"] - s["fy_actual"]) / max(s["peak_y"], 1.0)))
        scores.append((float(np.mean(np.square(errors))), float(p)))
    best_mse, best_p = min(scores)

    for s in samples:
        nx = s["fx_pure"] / max(s["peak_x"], 1.0)
        ny = s["fy_pure"] / max(s["peak_y"], 1.0)
        for label, p in (("ellipse", 2.0), ("best_p", best_p)):
            q = (nx ** p + ny ** p) ** (1.0 / p)
            scale = min(1.0, 1.0 / max(q, 1e-9))
            s[label + "_scale"] = scale
            s[label + "_fx"] = scale * s["fx_pure"]
            s[label + "_fy"] = scale * s["fy_pure"]
        s["actual_fx_over_pure"] = s["fx_actual"] / max(s["fx_pure"], 1.0)
        s["actual_fy_over_pure"] = s["fy_actual"] / max(s["fy_pure"], 1.0)

        axle = "front" if s["wheel"].startswith("F") else "rear"
        road = 0.50 if s["scenario"] == "flat_mu050" else 0.75
        # Normalized combined slip: direction is determined by slip demand
        # normalized by each pure-slip peak.  Each axis is evaluated at the
        # same normalized combined magnitude.  This permits Fx and Fy to be
        # redistributed instead of multiplying both pure forces by one scale.
        zx = abs(s["kappa"]) / max(s["peak_slip_x"], 1e-9)
        zy = abs(math.degrees(s["alpha_rad"])) / max(s["peak_slip_y_deg"], 1e-9)
        rho = math.hypot(zx, zy)
        if rho > 1e-9:
            eq_x, _, _ = interp_table(tables[axle + "_fx"],
                                      rho * s["peak_slip_x"], s["fz"])
            eq_y, _, _ = interp_table(tables[axle + "_fy"],
                                      rho * s["peak_slip_y_deg"], s["fz"])
            s["normalized_fx"] = road * eq_x * zx / rho
            s["normalized_fy"] = road * eq_y * zy / rho
        else:
            s["normalized_fx"], s["normalized_fy"] = 0.0, 0.0

        # Theoretical combined slip uses the geometric slip vector
        # [kappa, tan(alpha)] without peak normalization.
        tx, ty = abs(s["kappa"]), abs(math.tan(s["alpha_rad"]))
        trho = math.hypot(tx, ty)
        if trho > 1e-9:
            eq_x, _, _ = interp_table(tables[axle + "_fx"], trho, s["fz"])
            eq_y, _, _ = interp_table(tables[axle + "_fy"],
                                      math.degrees(math.atan(trho)), s["fz"])
            s["theoretical_fx"] = road * eq_x * tx / trho
            s["theoretical_fy"] = road * eq_y * ty / trho
        else:
            s["theoretical_fx"], s["theoretical_fy"] = 0.0, 0.0

    def model_error(label, subset):
        e = []
        for s in subset:
            e.extend(((s[label + "_fx"] - s["fx_actual"]) / max(s["peak_x"], 1.0),
                      (s[label + "_fy"] - s["fy_actual"]) / max(s["peak_y"], 1.0)))
        e = np.asarray(e)
        return {"normalized_rmse": float(np.sqrt(np.mean(e * e))),
                "normalized_mae": float(np.mean(np.abs(e)))}

    # Fit only the scalar transition between the two documented CarSim modes.
    blend_scores = []
    for blend in np.linspace(0.0, 1.0, 101):
        e = []
        for s in samples:
            fx = (1.0 - blend) * s["normalized_fx"] + blend * s["theoretical_fx"]
            fy = (1.0 - blend) * s["normalized_fy"] + blend * s["theoretical_fy"]
            e.extend(((fx - s["fx_actual"]) / max(s["peak_x"], 1.0),
                      (fy - s["fy_actual"]) / max(s["peak_y"], 1.0)))
        blend_scores.append((float(np.mean(np.square(e))), float(blend)))
    blend_mse, best_blend = min(blend_scores)
    for s in samples:
        s["transition_fx"] = ((1.0 - best_blend) * s["normalized_fx"] +
                              best_blend * s["theoretical_fx"])
        s["transition_fy"] = ((1.0 - best_blend) * s["normalized_fy"] +
                              best_blend * s["theoretical_fy"])

    report = {"sample_count": len(samples), "start_s": args.start,
              "best_p": best_p, "best_p_mse": best_mse,
              "ellipse": model_error("ellipse", samples),
              "best_p_model": model_error("best_p", samples),
              "normalized_model": model_error("normalized", samples),
              "theoretical_model": model_error("theoretical", samples),
              "best_transition_blend": best_blend,
              "best_transition_mse": blend_mse,
              "transition_model": model_error("transition", samples),
              "by_scenario": {}}
    for scenario in sorted(set(s["scenario"] for s in samples)):
        subset = [s for s in samples if s["scenario"] == scenario]
        report["by_scenario"][scenario] = {
            "count": len(subset), "ellipse": model_error("ellipse", subset),
            "best_p_model": model_error("best_p", subset),
            "normalized_model": model_error("normalized", subset),
            "theoretical_model": model_error("theoretical", subset),
            "transition_model": model_error("transition", subset),
            "median_actual_fx_over_pure": float(np.median([s["actual_fx_over_pure"] for s in subset])),
            "median_actual_fy_over_pure": float(np.median([s["actual_fy_over_pure"] for s in subset])),
        }

    with open(os.path.join(args.out, "combined_slip_samples.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(samples[0]))
        w.writeheader()
        w.writerows(samples)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
