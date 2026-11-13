"""Connect wheel-force fidelity to vehicle trajectory fidelity.

The paired system-ID CSVs historically stopped at body velocities and did not
store X/Y/Yaw.  This audit reconstructs a common inertial trajectory from the
logged body velocities and yaw rate, then reports microscopic force/slip errors
beside macroscopic position/heading/beta errors.  It also closes the force chain
by projecting wheel-frame forces to the body and integrating tire yaw moment.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WHEELS = ("FL", "FR", "RL", "RR")
LF, LR, TF, TR = 1.46257356, 1.18742644, 1.72, 1.60
MASS, IZ = 1461.4, 1693.469182
WX = np.array([LF, LF, -LR, -LR])
WY = np.array([TF / 2, -TF / 2, TR / 2, -TR / 2])


def augment(frame: pd.DataFrame) -> pd.DataFrame:
    q = frame.sort_values("t").reset_index(drop=True).copy()
    t = q.t.to_numpy(float)
    dt = np.diff(np.r_[0.0, t])
    # Compare every plant relative to its first logged sample.  The previous
    # unknown 0..dt interval must not be invented by the reconstruction.
    dt[0] = 0.0
    r = q.yaw_rate.to_numpy(float)
    # Samples are post-step values.  Trapezoidal integration avoids assigning
    # the full post-step yaw rate to the preceding interval.
    r_prev = np.r_[0.0, r[:-1]]
    psi = np.cumsum(0.5 * (r_prev + r) * dt)
    vxb, vyb = q.vx_body.to_numpy(float), q.vy_body.to_numpy(float)
    vxw = np.cos(psi) * vxb - np.sin(psi) * vyb
    vyw = np.sin(psi) * vxb + np.cos(psi) * vyb
    vxw_prev, vyw_prev = np.r_[0.0, vxw[:-1]], np.r_[0.0, vyw[:-1]]
    q["psi_recon"] = psi
    q["x_recon"] = np.cumsum(0.5 * (vxw_prev + vxw) * dt)
    q["y_recon"] = np.cumsum(0.5 * (vyw_prev + vyw) * dt)
    q["path_length"] = np.cumsum(np.hypot(vxw, vyw) * dt)
    if {"x_world", "y_world", "psi"}.issubset(q.columns):
        q["x_eval"] = q.x_world.to_numpy(float) - float(q.x_world.iloc[0])
        q["y_eval"] = q.y_world.to_numpy(float) - float(q.y_world.iloc[0])
        psi_actual = np.unwrap(q.psi.to_numpy(float))
        q["psi_eval"] = psi_actual - psi_actual[0]
        q["position_integration_closure"] = np.hypot(
            q.x_recon - q.x_eval, q.y_recon - q.y_eval)
        q["heading_integration_closure"] = np.arctan2(
            np.sin(q.psi_recon - q.psi_eval), np.cos(q.psi_recon - q.psi_eval))
    else:
        q["x_eval"], q["y_eval"], q["psi_eval"] = (
            q.x_recon, q.y_recon, q.psi_recon)
        q["position_integration_closure"] = 0.0
        q["heading_integration_closure"] = 0.0

    steer = q[[f"steer_{w}" for w in WHEELS]].to_numpy(float)
    fxw = q[[f"fx_{w}" for w in WHEELS]].to_numpy(float)
    fyw = q[[f"fy_{w}" for w in WHEELS]].to_numpy(float)
    fxb = fxw * np.cos(steer) - fyw * np.sin(steer)
    fyb = fxw * np.sin(steer) + fyw * np.cos(steer)
    q["fx_total_body"] = fxb.sum(axis=1)
    q["fy_total_body"] = fyb.sum(axis=1)
    q["fy_front_body"] = fyb[:, :2].sum(axis=1)
    q["fy_rear_body"] = fyb[:, 2:].sum(axis=1)
    q["mz_tire"] = (fyb * WX - fxb * WY).sum(axis=1)
    q["ax_force"] = q.fx_total_body / MASS
    q["ay_force"] = q.fy_total_body / MASS
    rdot = q.mz_tire.to_numpy(float) / IZ
    q["yaw_rate_from_tires"] = r[0] + np.cumsum(rdot * dt)
    q["psi_from_tires"] = np.cumsum(q.yaw_rate_from_tires.to_numpy(float) * dt)
    q["kappa_rear_mean"] = q[["kappa_RL", "kappa_RR"]].mean(axis=1)
    return q


def rmse(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


def paired_metrics(path: Path):
    raw = pd.read_csv(path)
    groups = {name: augment(g) for name, g in raw.groupby("plant")}
    if "carsim" not in groups or "fast" not in groups:
        raise ValueError(f"{path} must contain carsim and fast rows")
    c, f = groups["carsim"], groups["fast"]
    n = min(len(c), len(f))
    c, f = c.iloc[:n], f.iloc[:n]
    dpsi = np.arctan2(np.sin(f.psi_eval - c.psi_eval),
                      np.cos(f.psi_eval - c.psi_eval))
    pos = np.hypot(f.x_eval - c.x_eval, f.y_eval - c.y_eval)
    metrics = {
        "file": path.name,
        "scenario": str(c.scenario.iloc[0]),
        "maneuver": str(c.maneuver.iloc[0]),
        "duration_s": float(c.t.iloc[-1]),
        "position_rmse_m": rmse(pos),
        "position_final_m": float(pos.iloc[-1]),
        "x_rmse_m": rmse(f.x_eval - c.x_eval),
        "y_rmse_m": rmse(f.y_eval - c.y_eval),
        "heading_rmse_rad": rmse(dpsi),
        "heading_final_rad": float(dpsi.iloc[-1]),
        "beta_rmse_rad": rmse(f.beta - c.beta),
        "yaw_rate_rmse_radps": rmse(f.yaw_rate - c.yaw_rate),
        "speed_rmse_mps": rmse(f.speed - c.speed),
        "path_length_error_m": float(f.path_length.iloc[-1] - c.path_length.iloc[-1]),
        "kappa_rear_rmse": rmse(f.kappa_rear_mean - c.kappa_rear_mean),
        "front_fy_rmse_n": rmse(f.fy_front_body - c.fy_front_body),
        "rear_fy_rmse_n": rmse(f.fy_rear_body - c.fy_rear_body),
        "tire_mz_rmse_nm": rmse(f.mz_tire - c.mz_tire),
        "carsim_ax_force_closure_rmse": rmse(c.ax_force - c.ax),
        "fast_ax_force_closure_rmse": rmse(f.ax_force - f.ax),
        "carsim_ay_force_closure_rmse": rmse(c.ay_force - c.ay),
        "fast_ay_force_closure_rmse": rmse(f.ay_force - f.ay),
        "carsim_yaw_force_closure_rmse": rmse(c.yaw_rate_from_tires - c.yaw_rate),
        "fast_yaw_force_closure_rmse": rmse(f.yaw_rate_from_tires - f.yaw_rate),
        "carsim_position_integration_closure_rmse_m": rmse(c.position_integration_closure),
        "fast_position_integration_closure_rmse_m": rmse(f.position_integration_closure),
        "carsim_heading_integration_closure_rmse_rad": rmse(c.heading_integration_closure),
        "fast_heading_integration_closure_rmse_rad": rmse(f.heading_integration_closure),
    }
    return c, f, metrics


def render(c, f, metrics, out: Path):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes[0, 0].plot(c.x_eval, c.y_eval, label="CarSim", lw=2)
    axes[0, 0].plot(f.x_eval, f.y_eval, label="FastPlant", lw=2)
    axes[0, 0].set_title("inertial X-Y trajectory")
    axes[0, 0].axis("equal")
    axes[0, 0].legend()
    t = c.t.to_numpy(float)
    for ax, field, title, unit in (
        (axes[0, 1], "psi_eval", "heading", "rad"),
        (axes[0, 2], "beta", "CG sideslip beta", "rad"),
        (axes[1, 1], "yaw_rate", "yaw rate", "rad/s"),
        (axes[1, 2], "speed", "CG speed", "m/s"),
        (axes[2, 0], "kappa_rear_mean", "rear longitudinal slip", "-"),
        (axes[2, 1], "fy_rear_body", "rear-axle body Fy", "N"),
        (axes[2, 2], "mz_tire", "total tire yaw moment", "N m"),
    ):
        ax.plot(t, c[field], label="CarSim", lw=2)
        ax.plot(t, f[field], label="FastPlant", lw=2)
        ax.set_title(title)
        ax.set_ylabel(unit)
    pos = np.hypot(f.x_eval - c.x_eval, f.y_eval - c.y_eval)
    dpsi = np.arctan2(np.sin(f.psi_eval - c.psi_eval),
                      np.cos(f.psi_eval - c.psi_eval))
    axes[1, 0].plot(t, pos, label="position error (m)")
    axes[1, 0].plot(t, np.abs(dpsi), label="|heading error| (rad)")
    axes[1, 0].set_title("accumulated macroscopic divergence")
    axes[1, 0].legend()
    for ax in axes.flat:
        ax.grid(alpha=.25)
        if ax is not axes[0, 0]:
            ax.set_xlabel("time (s)")
    fig.suptitle(
        f"{metrics['scenario']} / {metrics['maneuver']}: micro-to-macro closure",
        fontsize=18)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--render", default="steer_large,combined_power")
    args = ap.parse_args()
    data, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    render_set = set(x.strip() for x in args.render.split(",") if x.strip())
    rows = []
    for path in sorted(data.glob("*.csv")):
        if "__" not in path.stem:
            continue
        c, f, result = paired_metrics(path)
        rows.append(result)
        if result["maneuver"] in render_set:
            render(c, f, result, out / f"{path.stem}__micro_macro.png")
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "metrics.csv", index=False)
    summary = {
        "dataset": str(data), "count": len(rows),
        "mean": {k: float(v) for k, v in frame.select_dtypes("number").mean().items()},
        "max": {k: float(v) for k, v in frame.select_dtypes("number").max().items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
