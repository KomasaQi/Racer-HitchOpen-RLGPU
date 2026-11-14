"""Multiple-shooting audit of FastPlant's local flow against CarSim.

Long open-loop drift trajectories are unstable, so their eventual separation
does not identify whether the local vector field is right.  This audit starts
FastPlant repeatedly from measured CarSim x/y/psi, body velocity, yaw rate and
wheel speeds, applies the same future physical commands, and scores endpoint
errors over 0.25--2.0 s horizons.  It is the bridge between wheel-force fidelity
and the macroscopic trajectory an RL policy actually experiences.
"""
from __future__ import annotations

import argparse
import hashlib
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


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def evaluate_file(path: Path, horizons, stride, solver_iters, combined_slip_blend,
                  stiffness_exp_x, stiffness_exp_y):
    all_rows = pd.read_csv(path)
    c = all_rows[all_rows.plant == "carsim"].sort_values("t").reset_index(drop=True)
    if c.empty or not {"x_world", "y_world", "psi"}.issubset(c.columns):
        raise ValueError(f"{path} does not contain actual x/y/psi")
    dt = float(np.median(np.diff(c.t.to_numpy(float))))
    mu = float(c.mu.iloc[0])
    results = []
    for horizon in horizons:
        steps = max(1, int(round(horizon / dt)))
        starts = np.arange(0, len(c) - steps, stride, dtype=int)
        if len(starts) == 0:
            continue
        params = load_preset("fullsize_racecar_ev")
        params["wheel_solver_bisection_iters"] = int(solver_iters)
        params["combined_slip_blend"] = float(combined_slip_blend)
        if stiffness_exp_x is not None:
            params["tire_stiffness_grip_exponent_x"] = float(stiffness_exp_x)
        if stiffness_exp_y is not None:
            params["tire_stiffness_grip_exponent_y"] = float(stiffness_exp_y)
        plant = FastRacerPlant(params, len(starts), "cpu")
        plant.tyre_scale.fill_(mu)
        s = torch.zeros((len(starts), 10), dtype=plant.dtype)
        first = c.iloc[starts]
        psi0 = torch.tensor(first.psi.to_numpy(float), dtype=plant.dtype)
        vxb = torch.tensor(first.vx_body.to_numpy(float), dtype=plant.dtype)
        vyb = torch.tensor(first.vy_body.to_numpy(float), dtype=plant.dtype)
        s[:, 0] = torch.tensor(first.x_world.to_numpy(float), dtype=plant.dtype)
        s[:, 1] = torch.tensor(first.y_world.to_numpy(float), dtype=plant.dtype)
        s[:, 2] = psi0
        s[:, 3] = torch.cos(psi0) * vxb - torch.sin(psi0) * vyb
        s[:, 4] = torch.sin(psi0) * vxb + torch.cos(psi0) * vyb
        s[:, 5] = torch.tensor(first.yaw_rate.to_numpy(float), dtype=plant.dtype)
        for j, wheel in enumerate(("FL", "FR", "RL", "RR")):
            s[:, 6 + j] = torch.tensor(first[f"omega_{wheel}"].to_numpy(float),
                                       dtype=plant.dtype)
        plant.reset_guess()
        # A rolling CarSim state does not have a zero-torque motor. Reconstruct
        # the hidden scalar motor state from the preceding recorded commands
        # and measured wheel speeds; document this as an estimate, not truth.
        trace_plant = FastRacerPlant(params, len(c), "cpu")
        trace_plant.tyre_scale.fill_(mu)
        trace_wheels = torch.tensor(c[["omega_RL", "omega_RR"]].to_numpy(), dtype=plant.dtype)
        trace_q = torch.tensor(c.throttle_cmd.to_numpy(), dtype=plant.dtype)
        trace_b = torch.tensor(c.brake_cmd.to_numpy(), dtype=plant.dtype)
        target_torque = trace_plant._motor_torque(trace_wheels, trace_q, trace_b)[1]
        tau_motor = float(params.get("motor_tau", 0.0))
        alpha_motor = 1.0 if tau_motor <= 0 else -np.expm1(-dt/tau_motor)
        torque_history = torch.zeros_like(target_torque)
        motor_state = torch.tensor(0., dtype=plant.dtype)
        for index, command_torque in enumerate(target_torque):
            motor_state = motor_state + alpha_motor * (command_torque - motor_state)
            torque_history[index] = motor_state
        plant._motor_torque_state.copy_(torque_history[torch.as_tensor(starts)])
        plant._ax_guess.copy_(torch.tensor(first.ax.to_numpy(), dtype=plant.dtype))
        plant._ay_guess.copy_(torch.tensor(first.ay.to_numpy(), dtype=plant.dtype))
        ex = None
        for k in range(1, steps + 1):
            future = c.iloc[starts + k]
            u = torch.tensor(future[["steer_cmd", "throttle_cmd", "brake_cmd"]].to_numpy(float),
                             dtype=plant.dtype)
            s, ex = plant.step(s, u, dt=dt, substeps=5, mode="pedals")
        target = c.iloc[starts + steps]
        target_psi = torch.tensor(target.psi.to_numpy(float), dtype=plant.dtype)
        dx = s[:, 0] - torch.tensor(target.x_world.to_numpy(float), dtype=plant.dtype)
        dy = s[:, 1] - torch.tensor(target.y_world.to_numpy(float), dtype=plant.dtype)
        dpsi = wrap(s[:, 2] - target_psi)
        cp, sp = torch.cos(s[:, 2]), torch.sin(s[:, 2])
        fast_vxb = cp * s[:, 3] + sp * s[:, 4]
        fast_vyb = -sp * s[:, 3] + cp * s[:, 4]
        fast_speed = torch.hypot(fast_vxb, fast_vyb)
        fast_beta = torch.atan2(fast_vyb, fast_vxb)
        target_speed = torch.tensor(target.speed.to_numpy(float), dtype=plant.dtype)
        target_beta = torch.tensor(target.beta.to_numpy(float), dtype=plant.dtype)
        target_yaw = torch.tensor(target.yaw_rate.to_numpy(float), dtype=plant.dtype)
        target_kappa = torch.tensor(
            target[["kappa_RL", "kappa_RR"]].to_numpy(float).mean(axis=1),
            dtype=plant.dtype)
        fast_kappa = ex["kappa_power"][:, 2:4].mean(dim=1)
        rear_radius = plant.geometry()[2][:, 2:4]
        # Derive both sides from raw omega and the authoritative preset radius;
        # legacy datasets logged rear surface speed with the old front radius.
        target_rear_surface = (
            torch.tensor(target[["omega_RL", "omega_RR"]].to_numpy(float),
                         dtype=plant.dtype) * rear_radius).mean(dim=1)
        fast_rear_surface = (s[:, 8:10] * rear_radius).mean(dim=1)

        # Raw kappa is singular when contact-patch longitudinal speed crosses
        # zero.  Keep it for diagnostics, but use bounded and log-magnitude
        # variants plus wheel surface speed to judge the physical wheel state.
        def signed_log1p(x):
            return torch.sign(x) * torch.log1p(torch.abs(x))

        def score(name, value):
            v = value.detach().numpy()
            results.append({
                "file": path.name, "scenario": str(c.scenario.iloc[0]),
                "maneuver": str(c.maneuver.iloc[0]), "horizon_s": float(horizon),
                "signal": name, "rmse": float(np.sqrt(np.mean(v * v))),
                "mae": float(np.mean(np.abs(v))), "p95": float(np.quantile(np.abs(v), .95)),
                "max_abs": float(np.max(np.abs(v))), "samples": int(len(v)),
            })

        score("position_m", torch.hypot(dx, dy))
        score("heading_rad", dpsi)
        score("beta_rad", wrap(fast_beta - target_beta))
        score("yaw_rate_radps", s[:, 5] - target_yaw)
        score("speed_mps", fast_speed - target_speed)
        score("rear_kappa", fast_kappa - target_kappa)
        score("rear_kappa_clip5", fast_kappa.clamp(-5.0, 5.0) -
              target_kappa.clamp(-5.0, 5.0))
        score("rear_kappa_log", signed_log1p(fast_kappa) - signed_log1p(target_kappa))
        score("rear_surface_speed_mps", fast_rear_surface - target_rear_surface)
    return results


def render(frame: pd.DataFrame, out: Path):
    signals = ["position_m", "heading_rad", "beta_rad", "yaw_rate_radps",
               "speed_mps", "rear_kappa_clip5", "rear_kappa_log",
               "rear_surface_speed_mps"]
    fig, axes = plt.subplots(2, 4, figsize=(21, 9))
    for ax, signal in zip(axes.flat, signals):
        q = frame[frame.signal == signal]
        for scenario, g in q.groupby("scenario"):
            agg = g.groupby("horizon_s", as_index=False).apply(
                lambda x: pd.Series({"rmse": np.sqrt(np.mean(x.rmse.to_numpy() ** 2)),
                                     "p95": float(np.max(x.p95))}),
                include_groups=False)
            ax.plot(agg.horizon_s, agg.rmse, marker="o", label=scenario)
            ax.plot(agg.horizon_s, agg.p95, ls="--", alpha=.55)
        ax.set_title(signal + " (solid RMSE, dashed worst p95)")
        ax.set_xlabel("multiple-shooting horizon (s)")
        ax.grid(alpha=.25)
    axes[0, 0].legend()
    fig.suptitle("CarSim-started FastPlant local-flow macro audit", fontsize=18)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizons", default="0.25,0.5,1.0,2.0")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--wheel-solver-iters", type=int, default=12)
    ap.add_argument("--combined-slip-blend", type=float, default=0.49)
    ap.add_argument("--stiffness-grip-exponent", type=float, default=None,
                    help="legacy scalar override; omitted means use production preset")
    ap.add_argument("--stiffness-grip-exponent-x", type=float, default=None)
    ap.add_argument("--stiffness-grip-exponent-y", type=float, default=None)
    args = ap.parse_args()
    data, out = Path(args.dataset), Path(args.out)
    source_manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("position_reference") != "instantaneous_total_vehicle_CG":
        raise ValueError("Position audit requires freshly collected Xcg_TM/Ycg_TM data")
    out.mkdir(parents=True, exist_ok=True)
    horizons = [float(x) for x in args.horizons.split(",")]
    rows = []
    exp_x = (args.stiffness_grip_exponent_x
             if args.stiffness_grip_exponent_x is not None
             else args.stiffness_grip_exponent)
    exp_y = (args.stiffness_grip_exponent_y
             if args.stiffness_grip_exponent_y is not None
             else args.stiffness_grip_exponent)
    preset = load_preset("fullsize_racecar_ev")
    effective_x = (preset["tire_stiffness_grip_exponent_x"]
                   if exp_x is None else exp_x)
    effective_y = (preset["tire_stiffness_grip_exponent_y"]
                   if exp_y is None else exp_y)
    for path in sorted(data.glob("*__*.csv")):
        rows.extend(evaluate_file(path, horizons, args.stride, args.wheel_solver_iters,
                                  args.combined_slip_blend,
                                  exp_x, exp_y))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "metrics.csv", index=False)
    summary = (frame.groupby(["horizon_s", "signal"], as_index=False)
               .agg(rmse=("rmse", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
                    worst_p95=("p95", "max"), files=("file", "count")))
    summary.to_csv(out / "summary.csv", index=False)
    render(frame, out / "local_flow_macro_horizon.png")
    payload = {"dataset": str(data), "solver_iters": args.wheel_solver_iters,
               "preset": preset,
               "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (Path(ROOT) / "xcar-simulation" / "fast_plant.py",
                                               Path(ROOT) / "xcar-simulation" / "presets.yaml",
                                               Path(__file__))},
               "position_reference": "instantaneous_total_vehicle_CG",
               "motor_initialization": "estimated by replaying recorded commands and wheel speeds",
               "combined_slip_blend": args.combined_slip_blend,
               "stiffness_grip_exponent": args.stiffness_grip_exponent,
               "stiffness_grip_exponent_x": effective_x,
               "stiffness_grip_exponent_y": effective_y,
               "horizons": horizons, "rows": len(frame)}
    (out / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
