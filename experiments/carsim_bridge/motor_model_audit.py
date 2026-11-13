"""Probe CarSim EV internals and compare the FastPlant motor law directly."""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xcar-simulation"))
sys.path.insert(0, r"D:\CarSim2024_1_Full_Database\Extensions\Custom_Py")

from fast_plant import FastRacerPlant, load_preset  # noqa: E402
import vs_solver  # noqa: E402


BASE = [
    "Ax", "Ay", "AVz", "Steer_L1", "Steer_R1", "AVy_L1", "AVy_L2",
    "AVy_R1", "AVy_R2", "Xo", "Yo", "Yaw", "Vx", "Vy", "Beta",
    "Fx_WC_L1", "Fx_WC_L2", "Fx_WC_R1", "Fx_WC_R2", "Fy_WC_L1",
    "Fy_WC_L2", "Fy_WC_R1", "Fy_WC_R2", "Fz_WC_L1", "Fz_WC_L2",
    "Fz_WC_R1", "Fz_WC_R2", "Kappa_L1", "Kappa_L2", "Kappa_R1",
    "Kappa_R2", "Alpha_L1", "Alpha_L2", "Alpha_R1", "Alpha_R2", "AV_Motor",
]
MOTOR = [
    "M_MotCmd", "M_Motor", "PwrDrHEV", "PwrMotDm", "PwrMotor",
    "EffMotor", "PwrBttry", "PwrBtrDs", "SocBttry", "VocBttry",
    "V_Bttry", "A_Bttry", "ResBttry",
]
CHANNELS = BASE + MOTOR
IDX = {name: i for i, name in enumerate(CHANNELS)}


def run_launch(simfile: str, throttle: float, duration: float, sample_dt: float):
    vs = vs_solver.vs_solver()
    dll = ctypes.cdll.LoadLibrary(vs.get_dll_path(simfile))
    if not vs.get_api(dll):
        raise RuntimeError("VehicleSim API unavailable")
    if getattr(dll, "vs_set_opt_error_dialog", None) is not None:
        dll.vs_set_opt_error_dialog(0)
    cfg = vs.read_configuration(simfile)
    if cfg["n_export"] != len(CHANNELS):
        raise RuntimeError(f"expected {len(CHANNELS)} exports, got {cfg['n_export']}")
    exp = vs.copy_export_vars(cfg["n_export"])
    imp = [0.0, float(throttle), 0.0]
    t = cfg["t_start"]
    physics_dt = float(cfg["t_step"])
    stride = max(int(round(sample_dt / physics_dt)), 1)
    rows = []
    try:
        for k in range(int(round(duration / physics_dt))):
            t += physics_dt
            status, exp = vs.integrate_io(t, imp, exp)
            if status:
                raise RuntimeError(f"solver status {status} at {t:.6f} s")
            if (k + 1) % stride == 0:
                rows.append([t, throttle] + [float(x) for x in exp])
    finally:
        vs.terminate_run(t)
    return rows


def add_fast_static(frame: pd.DataFrame) -> pd.DataFrame:
    params = load_preset("fullsize_racecar_ev")
    plant = FastRacerPlant(params, len(frame), "cpu")
    rear_rpm = torch.tensor(frame[["AVy_L2", "AVy_R2"]].to_numpy(),
                            dtype=plant.dtype)
    rear_omega = rear_rpm * (2.0 * math.pi / 60.0)
    q = torch.tensor(frame.throttle.to_numpy(), dtype=plant.dtype)
    axle, motor, wm = plant._motor_torque(rear_omega, q, torch.zeros_like(q))
    rpm = wm * 60.0 / (2.0 * math.pi)
    eff = plant._motor_efficiency(rpm.abs(), motor.abs())
    out = frame.copy()
    out["Fast_AV_Motor"] = rpm.detach().numpy()
    out["Fast_M_Motor_static"] = motor.detach().numpy()
    out["Fast_M_axle_static"] = axle.detach().numpy()
    out["Fast_EffMotor"] = eff.detach().numpy()
    out["Fast_PwrMechanical_static"] = (motor * wm / 1000.0).detach().numpy()
    out["Fast_PwrBattery_static"] = (
        motor * wm / eff.clamp(min=.02) / 1000.0).detach().numpy()
    out["CarSim_PwrMechanical"] = (
        out["M_Motor"] * out["AV_Motor"] * (2.0 * np.pi / 60.0) / 1000.0)
    return out


def metrics(frame: pd.DataFrame):
    rows = []
    pairs = {
        "motor_speed_rpm": ("Fast_AV_Motor", "AV_Motor"),
        "motor_torque_static_vs_command_nm": ("Fast_M_Motor_static", "M_MotCmd"),
        "motor_torque_static_vs_output_nm": ("Fast_M_Motor_static", "M_Motor"),
        "motor_efficiency": ("Fast_EffMotor", "EffMotor"),
        "mechanical_motor_power_kw": ("Fast_PwrMechanical_static", "CarSim_PwrMechanical"),
        "battery_required_power_kw": ("Fast_PwrBattery_static", "PwrBttry"),
    }
    for q, group in frame.groupby("throttle"):
        for phase, selected in (("all", group), ("steady_t_ge_1s", group[group.time >= 1.0])):
            for name, (a, b) in pairs.items():
                err = selected[a].to_numpy() - selected[b].to_numpy()
                rows.append({"throttle": q, "phase": phase, "signal": name,
                             "rmse": float(np.sqrt(np.mean(err * err))),
                             "mae": float(np.mean(np.abs(err))),
                             "bias": float(np.mean(err)),
                             "max_abs": float(np.max(np.abs(err)))})
    return rows


def render(frame: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(4, 4, figsize=(20, 15), sharex=True)
    fields = (
        ("M_MotCmd", "Fast_M_Motor_static", "motor command/static torque (Nm)"),
        ("M_Motor", "Fast_M_Motor_static", "motor output/static torque (Nm)"),
        ("CarSim_PwrMechanical", "Fast_PwrMechanical_static", "mechanical shaft power (kW)"),
        ("EffMotor", "Fast_EffMotor", "motor efficiency (-)"),
    )
    for row, (q, group) in enumerate(frame.groupby("throttle")):
        for col, (cs, fast, label) in enumerate(fields):
            ax = axes[row, col]
            ax.plot(group.time, group[cs], label="CarSim", lw=2)
            ax.plot(group.time, group[fast], label="FastPlant static law", lw=2)
            ax.set_title(f"q={q:.1f}: {label}")
            ax.grid(alpha=.22)
            ax.axhline(0, color="black", lw=.7)
            if row == 3:
                ax.set_xlabel("time (s)")
    axes[0, 0].legend()
    fig.suptitle("CarSim internal EV channels vs current FastPlant motor law", fontsize=18)
    fig.tight_layout()
    fig.savefig(out / "motor_internal_timeseries.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True)
    fields2 = (
        ("PwrDrHEV", "driver demand (kW)"), ("PwrMotDm", "motor demand (kW)"),
        ("PwrBttry", "battery required power (kW)"), ("V_Bttry", "battery voltage (V)"),
        ("A_Bttry", "battery current (A)"), ("SocBttry", "battery SOC (-)"),
    )
    for q, group in frame.groupby("throttle"):
        for ax, (field, label) in zip(axes.flat, fields2):
            ax.plot(group.time, group[field], label=f"q={q:.1f}")
            ax.set_title(label)
            ax.grid(alpha=.22)
            ax.set_xlabel("time (s)")
    axes[0, 0].legend()
    fig.suptitle("CarSim power-management and battery states", fontsize=18)
    fig.tight_layout()
    fig.savefig(out / "carsim_power_battery_timeseries.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simfile", default=r"D:\carsim_wm_py\motor_probe\flat_mu100\simfile.sim")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--sample-dt", type=float, default=0.01)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in (0.1, 0.3, 0.5, 0.8):
        print(f"CarSim motor probe q={q:.1f}", flush=True)
        rows.extend(run_launch(args.simfile, q, args.duration, args.sample_dt))
    frame = pd.DataFrame(rows, columns=["time", "throttle"] + CHANNELS)
    frame = add_fast_static(frame)
    frame.to_csv(args.out / "motor_probe.csv", index=False)
    report = {
        "simfile": args.simfile,
        "duration": args.duration,
        "sample_dt": args.sample_dt,
        "metrics": metrics(frame),
        "carsim_config": {
            "driver_demand_power_w_per_throttle": 384000.0,
            "available_power_tau_s": 0.1,
            "motor_torque_tau_s": 0.05,
            "motor_inertia_kgm2": 0.0226,
            "gear_ratio": 3.44,
            "battery_packs": 120,
            "initial_soc": 0.8,
            "initial_temperature_c": 25.0,
        },
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    render(frame, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
