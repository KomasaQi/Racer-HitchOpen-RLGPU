"""Render first-turn sign and yaw-balance diagnostics from paired ID logs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WHEELS = ("FL", "FR", "RL", "RR")
X_POS = np.array([1.46257356, 1.46257356, -1.18742644, -1.18742644])
Y_POS = np.array([0.860, -0.860, 0.800, -0.800])


def derived(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    steer = np.column_stack([d[f"steer_{w}"].to_numpy() for w in WHEELS])
    fxw = np.column_stack([d[f"fx_{w}"].to_numpy() for w in WHEELS])
    fyw = np.column_stack([d[f"fy_{w}"].to_numpy() for w in WHEELS])
    c, s = np.cos(steer), np.sin(steer)
    fxb = c * fxw - s * fyw
    fyb = s * fxw + c * fyw
    mz = fyb * X_POS - fxb * Y_POS
    d["fy_front"] = fyb[:, :2].sum(axis=1)
    d["fy_rear"] = fyb[:, 2:].sum(axis=1)
    d["mz_front"] = mz[:, :2].sum(axis=1)
    d["mz_rear"] = mz[:, 2:].sum(axis=1)
    d["mz_total_tire"] = mz.sum(axis=1)
    return d


def render_matrix(data: Path, out: Path) -> None:
    fig, axes = plt.subplots(3, 6, figsize=(24, 12), sharex=True)
    fields = (
        ("steer_FL", "front-left steer (rad)"),
        ("beta", "vehicle beta (rad)"),
        ("yaw_rate", "yaw rate (rad/s)"),
        ("fy_front", "front axle Fy, body (N)"),
        ("fy_rear", "rear axle Fy, body (N)"),
        ("mz_total_tire", "total tire yaw moment (Nm)"),
    )
    colors = {"fast": "tab:orange", "carsim": "tab:blue"}
    for row, scenario in enumerate(("flat_mu050", "flat_mu075", "flat_mu100")):
        raw = pd.read_csv(data / f"{scenario}__steer_large.csv")
        for plant in ("carsim", "fast"):
            d = derived(raw[raw.plant == plant])
            d = d[(d.t >= 2.9) & (d.t <= 4.5)]
            for col, (field, label) in enumerate(fields):
                axes[row, col].plot(d.t, d[field], marker="." if plant == "carsim" else None,
                                    lw=2, ms=3, color=colors[plant], label=plant)
        for col, (_, label) in enumerate(fields):
            ax = axes[row, col]
            ax.axhline(0, color="black", lw=.8)
            ax.axvline(3.0, color="0.4", lw=.8, ls="--")
            ax.grid(alpha=.22)
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(scenario)
            if row == 2:
                ax.set_xlabel("time (s)")
    axes[0, 0].legend(loc="best")
    fig.suptitle("First steering excitation: sign, trend and tire yaw-balance audit", fontsize=18)
    fig.tight_layout()
    fig.savefig(out / "three_mu_first_turn_yaw_balance.png", dpi=180)
    plt.close(fig)


def render_detail(data: Path, out: Path) -> None:
    raw = pd.read_csv(data / "flat_mu050__steer_large.csv")
    fig, axes = plt.subplots(4, 3, figsize=(17, 15), sharex=True)
    fields = (
        ("steer_FL", "steer FL (rad)"), ("vy_body", "Vy body (m/s)"),
        ("beta", "beta (rad)"), ("yaw_rate", "yaw rate (rad/s)"),
        ("ay", "Ay (m/s2)"), ("alpha_FL", "alpha FL (rad)"),
        ("fy_FL", "Fy FL, wheel (N)"), ("fy_RL", "Fy RL, wheel (N)"),
        ("fy_front", "front axle Fy, body (N)"),
        ("fy_rear", "rear axle Fy, body (N)"),
        ("mz_front", "front axle Mz (Nm)"), ("mz_rear", "rear axle Mz (Nm)"),
    )
    colors = {"fast": "tab:orange", "carsim": "tab:blue"}
    for plant in ("carsim", "fast"):
        d = derived(raw[raw.plant == plant])
        d = d[(d.t >= 2.9) & (d.t <= 4.2)]
        for ax, (field, label) in zip(axes.flat, fields):
            ax.plot(d.t, d[field], marker="." if plant == "carsim" else None,
                    lw=2, ms=4, color=colors[plant], label=plant)
            ax.set_ylabel(label)
    for ax in axes.flat:
        ax.axhline(0, color="black", lw=.8)
        ax.axvline(3.0, color="0.4", lw=.8, ls="--")
        ax.grid(alpha=.22)
        ax.set_xlabel("time (s)")
    axes[0, 0].legend(loc="best")
    fig.suptitle("flat_mu050: first-turn raw channels and axle-force decomposition", fontsize=18)
    fig.tight_layout()
    fig.savefig(out / "flat_mu050_first_turn_force_decomposition.png", dpi=180)
    plt.close(fig)


def render_drift_entry(data: Path, out: Path) -> None:
    raw = pd.read_csv(data / "flat_mu050__drift_trim_prbs.csv")
    fig, axes = plt.subplots(4, 3, figsize=(17, 15), sharex=True)
    fields = (
        ("steer_FL", "steer FL (rad)"), ("kappa_RL", "kappa RL (-)"),
        ("alpha_RL", "alpha RL (rad)"), ("beta", "beta (rad)"),
        ("yaw_rate", "yaw rate (rad/s)"), ("ay", "Ay (m/s2)"),
        ("fy_front", "front axle Fy, body (N)"),
        ("fy_rear", "rear axle Fy, body (N)"),
        ("mz_total_tire", "total tire yaw moment (Nm)"),
        ("fy_RL", "Fy RL, wheel (N)"),
        ("mz_front", "front axle Mz (Nm)"),
        ("mz_rear", "rear axle Mz (Nm)"),
    )
    colors = {"fast": "tab:orange", "carsim": "tab:blue"}
    for plant in ("carsim", "fast"):
        d = derived(raw[raw.plant == plant])
        d = d[(d.t >= 1.8) & (d.t <= 3.05)]
        for ax, (field, label) in zip(axes.flat, fields):
            ax.plot(d.t, d[field], marker="." if plant == "carsim" else None,
                    lw=2, ms=4, color=colors[plant], label=plant)
            ax.set_ylabel(label)
    for ax in axes.flat:
        ax.axhline(0, color="black", lw=.8)
        ax.axvline(2.0, color="0.4", lw=.8, ls="--")
        ax.grid(alpha=.22)
        ax.set_xlabel("time (s)")
    axes[0, 0].legend(loc="best")
    fig.suptitle("flat_mu050 drift entry: corrected initial trend and force-chain agreement",
                 fontsize=18)
    fig.tight_layout()
    fig.savefig(out / "flat_mu050_drift_entry_force_decomposition.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--drift-data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    render_matrix(args.data, args.out)
    render_detail(args.data, args.out)
    render_drift_entry(args.drift_data, args.out)
    print(args.out / "three_mu_first_turn_yaw_balance.png")
    print(args.out / "flat_mu050_first_turn_force_decomposition.png")
    print(args.out / "flat_mu050_drift_entry_force_decomposition.png")


if __name__ == "__main__":
    main()
