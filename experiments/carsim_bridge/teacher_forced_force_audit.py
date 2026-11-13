"""Instantaneous FastPlant tyre-force audit at measured CarSim states.

This removes open-loop state divergence from the comparison: measured pose,
body velocity, yaw rate, wheel speed, wheel load and road-wheel angle are fed
to FastPlant's compact analytic tyre model at every sample.  The resulting
force and yaw-moment residuals identify the microscopic force-law error that
drives the multiple-shooting macroscopic trajectory error.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))
from fast_plant import FastRacerPlant, load_preset  # noqa: E402

WHEELS = ("FL", "FR", "RL", "RR")


def rmse(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x * x)))


def evaluate(path: Path, combined_slip_blend: float, stiffness_exp_x,
             stiffness_exp_y):
    frame = pd.read_csv(path)
    c = frame[frame.plant == "carsim"].sort_values("t").reset_index(drop=True)
    n = len(c)
    if not n:
        return None, []
    params = load_preset("fullsize_racecar_ev")
    params["combined_slip_blend"] = float(combined_slip_blend)
    if stiffness_exp_x is not None:
        params["tire_stiffness_grip_exponent_x"] = float(stiffness_exp_x)
    if stiffness_exp_y is not None:
        params["tire_stiffness_grip_exponent_y"] = float(stiffness_exp_y)
    plant = FastRacerPlant(params, n, "cpu")
    plant.tyre_scale.fill_(float(c.mu.iloc[0]))
    s = torch.zeros((n, 10), dtype=plant.dtype)
    psi = torch.tensor(c.psi.to_numpy(float), dtype=plant.dtype)
    vxb = torch.tensor(c.vx_body.to_numpy(float), dtype=plant.dtype)
    vyb = torch.tensor(c.vy_body.to_numpy(float), dtype=plant.dtype)
    s[:, 0] = torch.tensor(c.x_world.to_numpy(float), dtype=plant.dtype)
    s[:, 1] = torch.tensor(c.y_world.to_numpy(float), dtype=plant.dtype)
    s[:, 2] = psi
    s[:, 3] = torch.cos(psi) * vxb - torch.sin(psi) * vyb
    s[:, 4] = torch.sin(psi) * vxb + torch.cos(psi) * vyb
    s[:, 5] = torch.tensor(c.yaw_rate.to_numpy(float), dtype=plant.dtype)
    for j, wheel in enumerate(WHEELS):
        s[:, 6 + j] = torch.tensor(c[f"omega_{wheel}"].to_numpy(float), dtype=plant.dtype)
    delta = torch.tensor(c[["steer_FL", "steer_FR"]].to_numpy(float), dtype=plant.dtype)
    fz = torch.tensor(c[[f"fz_{w}" for w in WHEELS]].to_numpy(float), dtype=plant.dtype)
    ex = plant._wheel_forces(s, delta, fz, plant.geometry())
    reported_kappa = torch.tensor(
        c[[f"kappa_{w}" for w in WHEELS]].to_numpy(float), dtype=plant.dtype)
    reported_alpha = torch.tensor(
        c[[f"alpha_{w}" for w in WHEELS]].to_numpy(float), dtype=plant.dtype)
    ex_reported = plant._wheel_forces(
        s, delta, fz, plant.geometry(),
        kappa_power_override=reported_kappa, alpha_override=reported_alpha)

    out = c[["scenario", "maneuver", "mu", "t", "vx_body", "vy_body",
             "yaw_rate"]].copy()
    x_pos = plant.geometry()[0][0].detach().numpy()
    y_pos = plant.geometry()[1][0].detach().numpy()
    for j, wheel in enumerate(WHEELS):
        out[f"fx_carsim_{wheel}"] = c[f"fx_{wheel}"].to_numpy(float)
        out[f"fy_carsim_{wheel}"] = c[f"fy_{wheel}"].to_numpy(float)
        out[f"fx_fast_{wheel}"] = ex["fx_wheel"][:, j].detach().numpy()
        out[f"fy_fast_{wheel}"] = ex["fy_wheel"][:, j].detach().numpy()
        out[f"fx_reported_slip_{wheel}"] = ex_reported["fx_wheel"][:, j].detach().numpy()
        out[f"fy_reported_slip_{wheel}"] = ex_reported["fy_wheel"][:, j].detach().numpy()
        out[f"kappa_carsim_{wheel}"] = c[f"kappa_{wheel}"].to_numpy(float)
        out[f"kappa_fast_{wheel}"] = ex["kappa_power"][:, j].detach().numpy()
    fx_c = c[[f"fx_{w}" for w in WHEELS]].to_numpy(float)
    fy_c = c[[f"fy_{w}" for w in WHEELS]].to_numpy(float)
    steer = c[["steer_FL", "steer_FR"]].to_numpy(float)
    cd, sd = np.ones_like(fx_c), np.zeros_like(fx_c)
    cd[:, :2], sd[:, :2] = np.cos(steer), np.sin(steer)
    fxb_c = cd * fx_c - sd * fy_c
    fyb_c = sd * fx_c + cd * fy_c
    fxb_f = ex["fx_body"].detach().numpy()
    fyb_f = ex["fy_body"].detach().numpy()
    fxb_r = ex_reported["fx_body"].detach().numpy()
    fyb_r = ex_reported["fy_body"].detach().numpy()
    out["fx_total_carsim"] = fxb_c.sum(axis=1)
    out["fx_total_fast"] = fxb_f.sum(axis=1)
    out["fy_total_carsim"] = fyb_c.sum(axis=1)
    out["fy_total_fast"] = fyb_f.sum(axis=1)
    out["mz_carsim"] = (x_pos * fyb_c - y_pos * fxb_c).sum(axis=1)
    out["mz_fast"] = (x_pos * fyb_f - y_pos * fxb_f).sum(axis=1)
    out["fx_total_reported_slip"] = fxb_r.sum(axis=1)
    out["fy_total_reported_slip"] = fyb_r.sum(axis=1)
    out["mz_reported_slip"] = (x_pos * fyb_r - y_pos * fxb_r).sum(axis=1)

    rows = []
    for name in ("fx_total", "fy_total", "mz"):
        target = out[f"{name}_carsim"].to_numpy(float)
        pred = out[f"{name}_fast"].to_numpy(float)
        scale = max(rmse(target), 1e-6)
        rows.append({"file": path.name, "scenario": c.scenario.iloc[0],
                     "maneuver": c.maneuver.iloc[0], "signal": name,
                     "rmse": rmse(pred - target), "nrmse": rmse(pred - target) / scale,
                     "bias": float(np.mean(pred - target)), "samples": n})
        pred_r = out[f"{name}_reported_slip"].to_numpy(float)
        rows.append({"file": path.name, "scenario": c.scenario.iloc[0],
                     "maneuver": c.maneuver.iloc[0], "signal": name + "_reported_slip",
                     "rmse": rmse(pred_r - target),
                     "nrmse": rmse(pred_r - target) / scale,
                     "bias": float(np.mean(pred_r - target)), "samples": n})
    for axle, wheels in (("front", ("FL", "FR")), ("rear", ("RL", "RR"))):
        for force in ("fx", "fy"):
            target = sum(out[f"{force}_carsim_{w}"].to_numpy(float) for w in wheels)
            pred = sum(out[f"{force}_fast_{w}"].to_numpy(float) for w in wheels)
            rows.append({"file": path.name, "scenario": c.scenario.iloc[0],
                         "maneuver": c.maneuver.iloc[0], "signal": f"{force}_{axle}",
                         "rmse": rmse(pred - target),
                         "nrmse": rmse(pred - target) / max(rmse(target), 1e-6),
                         "bias": float(np.mean(pred - target)), "samples": n})
            pred_r = sum(out[f"{force}_reported_slip_{w}"].to_numpy(float) for w in wheels)
            rows.append({"file": path.name, "scenario": c.scenario.iloc[0],
                         "maneuver": c.maneuver.iloc[0],
                         "signal": f"{force}_{axle}_reported_slip",
                         "rmse": rmse(pred_r - target),
                         "nrmse": rmse(pred_r - target) / max(rmse(target), 1e-6),
                         "bias": float(np.mean(pred_r - target)), "samples": n})
    return out, rows


def render(traces, out):
    fig, axes = plt.subplots(3, 2, figsize=(17, 13), sharex="col")
    selected = []
    for _, q in traces:
        if q.scenario.iloc[0] in ("flat_mu050", "flat_mu075") and \
                q.maneuver.iloc[0] == "combined_power":
            selected.append(q)
    if not selected:
        selected = [q for _, q in traces[:2]]
    for col, q in enumerate(selected[:2]):
        t = q.t
        for row, signal in enumerate(("fx_total", "fy_total", "mz")):
            ax = axes[row, col]
            ax.plot(t, q[f"{signal}_carsim"], label="CarSim measured")
            ax.plot(t, q[f"{signal}_fast"], label="FastPlant @ CarSim state", alpha=.8)
            ax.plot(t, q[f"{signal}_reported_slip"], label="FastPlant @ reported slip",
                    alpha=.75, ls="--")
            ax.set_title(f"{q.scenario.iloc[0]} {signal}")
            ax.grid(alpha=.25)
        axes[-1, col].set_xlabel("time (s)")
    if selected:
        axes[0, 0].legend()
    fig.suptitle("Teacher-forced tyre-force closure (state divergence removed)", fontsize=18)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--combined-slip-blend", type=float, default=0.49)
    ap.add_argument("--stiffness-grip-exponent", type=float, default=None,
                    help="legacy scalar override; omitted means use production preset")
    ap.add_argument("--stiffness-grip-exponent-x", type=float, default=None)
    ap.add_argument("--stiffness-grip-exponent-y", type=float, default=None)
    args = ap.parse_args()
    data, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    preset = load_preset("fullsize_racecar_ev")
    exp_x = (args.stiffness_grip_exponent_x
             if args.stiffness_grip_exponent_x is not None
             else args.stiffness_grip_exponent)
    exp_y = (args.stiffness_grip_exponent_y
             if args.stiffness_grip_exponent_y is not None
             else args.stiffness_grip_exponent)
    effective_x = (preset["tire_stiffness_grip_exponent_x"]
                   if exp_x is None else exp_x)
    effective_y = (preset["tire_stiffness_grip_exponent_y"]
                   if exp_y is None else exp_y)
    rows, traces = [], []
    for path in sorted(data.glob("*__*.csv")):
        trace, metrics = evaluate(path, args.combined_slip_blend, exp_x, exp_y)
        if trace is not None:
            trace.to_csv(out / path.name, index=False)
            traces.append((path.name, trace))
            rows.extend(metrics)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    summary = (metrics.groupby("signal", as_index=False)
               .agg(nrmse=("nrmse", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
                    rmse=("rmse", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
                    files=("file", "count")))
    summary.to_csv(out / "summary.csv", index=False)
    render(traces, out / "teacher_forced_force_closure.png")
    (out / "manifest.json").write_text(json.dumps({"dataset": str(data),
        "combined_slip_blend": args.combined_slip_blend,
        "stiffness_grip_exponent": args.stiffness_grip_exponent,
        "stiffness_grip_exponent_x": effective_x,
        "stiffness_grip_exponent_y": effective_y,
        "method": "measured pose/velocity/yaw/wheels/Fz/steer -> analytic tyre force"},
        indent=2), encoding="utf-8")
    print(metrics.sort_values(["scenario", "maneuver", "signal"]).to_string(index=False))
    print("\nSUMMARY\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
