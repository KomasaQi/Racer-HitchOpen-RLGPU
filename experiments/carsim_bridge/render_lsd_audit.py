"""Render human-reviewable LSD/system-ID figures from stored paired data."""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RPM = np.array([-1200, -1080, -960, -840, -720, -600, -540, -480,
                -420, -360, -300, -240, -180, -120, -60, 0, 60, 120,
                180, 240, 300, 360, 420, 480, 540, 600, 720, 840,
                960, 1080, 1200], dtype=float)
DT = np.array([402.88, 382.21, 360.35, 337.07, 312.07, 284.88,
               270.26, 254.8, 238.35, 220.67, 201.44, 180.17, 156.03,
               127.4, 90.09, 0, -90.09, -127.4, -156.03, -180.17,
               -201.44, -220.67, -238.35, -254.8, -270.26, -284.88,
               -312.07, -337.07, -360.35, -382.21, -402.88])


def save_lsd_curve(out):
    x = np.linspace(-1200, 1200, 1001)
    fit = -90.09 * np.sign(x) * np.sqrt(np.abs(x) / 60.0)
    pred = -90.09 * np.sign(RPM) * np.sqrt(np.abs(RPM) / 60.0)
    fig, (ax, er) = plt.subplots(2, 1, figsize=(8.2, 6.4),
                                 gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot(x, fit, label="2-parameter generalized-viscous model", lw=2)
    ax.scatter(RPM, DT, s=22, c="black", label="CarSim 31-point carpet", zorder=3)
    ax.set_ylabel(r"$T_L-T_R$ (N m)")
    ax.grid(alpha=.25); ax.legend()
    er.axhline(0, color="black", lw=.8)
    er.scatter(RPM, pred - DT, s=20)
    er.set_xlabel(r"$n_L-n_R$ (rpm)"); er.set_ylabel("residual\n(N m)")
    er.grid(alpha=.25)
    fig.suptitle("CarSim rear LSD: data used only for offline identification")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "lsd_curve_and_residual.png"), dpi=180)
    plt.close(fig)


def save_ab(audit, out, folder="lsd_powerlaw_ab_v1", stem="lsd"):
    comp = pd.read_csv(os.path.join(audit, "data", folder, "comparison.csv"))
    val = comp[comp["split"] == "validation"].copy()
    val["delta"] = val["nrmse_candidate"] - val["nrmse_baseline"]
    val = val.sort_values("delta")
    labels = val["scenario"].str.replace("flat_", "") + ":" + val["family"]
    colors = np.where(val["delta"] <= 0, "#2ca02c", "#d62728")
    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    ax.barh(np.arange(len(val)), val["delta"], color=colors)
    ax.axvline(0, color="black", lw=.8)
    ax.set_yticks(np.arange(len(val)), labels, fontsize=7)
    ax.set_xlabel("candidate - old linear LSD validation NRMSE")
    ax.set_title("System-ID effect of structurally correct LSD (green improves)")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, stem + "_system_id_validation_delta.png"), dpi=180)
    plt.close(fig)


def save_drift(audit, out, scenario="flat_mu050"):
    old = pd.read_csv(os.path.join(audit, "data", "drift_combined_transition_v1",
                                   scenario + "__drift_trim_prbs.csv"))
    new = pd.read_csv(os.path.join(audit, "data", "drift_lsd_powerlaw_v1",
                                   scenario + "__drift_trim_prbs.csv"))
    corrected = pd.read_csv(os.path.join(audit, "data", "drift_body_lsd_corrected_v1",
                                         scenario + "__drift_trim_prbs.csv"))
    traces = [("CarSim", new[new.plant == "carsim"]),
              ("old linear LSD", old[old.plant == "fast"]),
              ("power-law LSD", new[new.plant == "fast"]),
              ("correct body + power-law LSD", corrected[corrected.plant == "fast"])]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    specs = [("kappa_RL", "rear-left kappa"), ("kappa_RR", "rear-right kappa"),
             ("beta", "sideslip beta (rad)"), ("yaw_rate", "yaw rate (rad/s)"),
             ("fx_RL", "rear-left Fx (N)"), ("fy_RL", "rear-left Fy (N)")]
    for ax, (key, label) in zip(axes.ravel(), specs):
        for name, df in traces:
            ax.plot(df.t, df[key], label=name, lw=1.2)
        ax.axvline(6.0, color="gray", ls="--", lw=.8)
        ax.set_ylabel(label); ax.grid(alpha=.22)
    axes[-1, 0].set_xlabel("time (s)"); axes[-1, 1].set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(scenario + " same-command drift replay; dashed line = PRBS/local-ID start")
    fig.tight_layout()
    fig.savefig(os.path.join(out, scenario + "_lsd_drift_replay.png"), dpi=180)
    plt.close(fig)


def save_local_drift(audit, out, scenario="flat_mu050"):
    """Plot the causal local replay, avoiding sign confusion after divergence."""
    frame = pd.read_csv(os.path.join(audit, "data", "drift_body_lsd_corrected_v1",
                                     scenario + "__drift_trim_prbs.csv"))
    traces = [("CarSim", frame[frame.plant == "carsim"]),
              ("FastPlant reset to CarSim state at t=6 s",
               frame[frame.plant == "fast_local"])]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), sharex=True)
    specs = [("kappa_RL", "rear-left kappa"), ("kappa_RR", "rear-right kappa"),
             ("alpha_RL", "rear-left alpha (rad)"), ("fy_RL", "rear-left Fy (N)"),
             ("beta", "sideslip beta (rad)"), ("yaw_rate", "yaw rate (rad/s)")]
    for ax, (key, label) in zip(axes.ravel(), specs):
        for name, df in traces:
            local = df[(df.t >= 6.0) & (df.t <= 8.0)]
            ax.plot(local.t, local[key], label=name, lw=1.5)
        ax.set_ylabel(label); ax.grid(alpha=.22)
    axes[-1, 0].set_xlabel("time (s)"); axes[-1, 1].set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(scenario + ": causal 2 s replay from the identical CarSim state")
    fig.tight_layout()
    fig.savefig(os.path.join(out, scenario + "_local_same_state_replay.png"), dpi=180)
    plt.close(fig)


def _plot_window(traces, specs, start, stop, title, path, relative=False):
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), sharex=True)
    for ax, (key, label) in zip(axes.ravel(), specs):
        for name, df in traces:
            q = df[(df.t >= start) & (df.t <= stop)].copy()
            y = q[key].to_numpy()
            if relative and len(y):
                y = y - y[0]
            ax.plot(q.t, y, marker="o", ms=3, label=name, lw=1.4)
        ax.axhline(0, color="black", lw=.65)
        ax.set_ylabel(("delta " if relative else "") + label)
        ax.grid(alpha=.22)
    axes[-1, 0].set_xlabel("time (s)"); axes[-1, 1].set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_initial_trends(audit, out, scenario="flat_mu050"):
    frame = pd.read_csv(os.path.join(audit, "data", "drift_body_lsd_corrected_v1",
                                     scenario + "__drift_trim_prbs.csv"))
    local = [("CarSim", frame[frame.plant == "carsim"]),
             ("FastPlant same-state replay", frame[frame.plant == "fast_local"])]
    specs = [("kappa_RL", "rear-left kappa"), ("kappa_RR", "rear-right kappa"),
             ("alpha_RL", "rear-left alpha (rad)"), ("fy_RL", "rear-left Fy (N)"),
             ("beta", "sideslip beta (rad)"), ("yaw_rate", "yaw rate (rad/s)")]
    _plot_window(local, specs, 6.0, 6.5,
                 scenario + ": first 0.5 s from the identical state (raw values)",
                 os.path.join(out, scenario + "_same_state_first_050s_raw.png"))
    _plot_window(local, specs, 6.0, 6.5,
                 scenario + ": first 0.5 s trend relative to each first sample",
                 os.path.join(out, scenario + "_same_state_first_050s_delta.png"),
                 relative=True)
    global_traces = [("CarSim", frame[frame.plant == "carsim"]),
                     ("FastPlant from rest", frame[frame.plant == "fast"])]
    _plot_window(global_traces, specs, 0.0, 3.0,
                 scenario + ": first drift-entry transient from rest",
                 os.path.join(out, scenario + "_from_rest_first_3s.png"))


def save_sign_matrix(audit, out):
    base = os.path.join(audit, "data", "system_id_body_lsd_corrected_v1")
    cases = [
        ("positive steer @ 3.10 s", "flat_mu100__steer_small.csv", 3.10,
         [("steer_FL", "rad"), ("vy_body", "m/s"), ("beta", "rad"),
          ("yaw_rate", "rad/s"), ("ay", "m/s2"), ("alpha_FL", "rad"),
          ("fy_FL", "N")]),
        ("positive throttle @ 1.00 s", "flat_mu100__launch_50.csv", 1.00,
         [("ax", "m/s2"), ("omega_RL", "rad/s"), ("kappa_RL", "1"),
          ("fx_RL", "N")]),
        ("brake @ 4.10 s", "flat_mu100__brake_30.csv", 4.10,
         [("ax", "m/s2"), ("kappa_FL", "1"), ("fx_FL", "N")]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for ax, (title, filename, t, fields) in zip(axes, cases):
        f = pd.read_csv(os.path.join(base, filename))
        values = []
        for plant in ("fast", "carsim"):
            q = f[f.plant == plant]
            row = q.loc[(q.t - t).abs().idxmin()]
            values.append([float(row[k]) for k, _ in fields])
        # Per-field magnitude normalization preserves sign while allowing N,
        # rad and m/s quantities to share one readable panel. Raw values are
        # printed above every bar with their units.
        raw = np.asarray(values)
        scale = np.maximum(np.max(np.abs(raw), axis=0), 1e-12)
        norm = raw / scale
        x = np.arange(len(fields)); width = .36
        b0 = ax.bar(x - width/2, norm[0], width, label="FastPlant")
        b1 = ax.bar(x + width/2, norm[1], width, label="CarSim")
        for bars, vals in ((b0, raw[0]), (b1, raw[1])):
            for bar, val, (_, unit) in zip(bars, vals, fields):
                y = bar.get_height()
                ax.text(bar.get_x()+bar.get_width()/2, y + (.04 if y >= 0 else -.08),
                        ("%.3g %s" % (val, unit)), ha="center",
                        va="bottom" if y >= 0 else "top", fontsize=7, rotation=20)
        ax.axhline(0, color="black", lw=.8)
        ax.set_xticks(x, [k for k, _ in fields])
        ax.set_ylim(-1.35, 1.35); ax.set_ylabel("signed / per-field magnitude")
        ax.set_title(title); ax.grid(axis="y", alpha=.2)
    axes[0].legend()
    fig.suptitle("First-excitation canonical sign and unit matrix (raw values annotated)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "canonical_sign_unit_first_excitation.png"), dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    save_lsd_curve(args.out)
    save_ab(args.audit, args.out)
    save_ab(args.audit, args.out, "body_lsd_vs_old_ab_v1", "correct_body_lsd")
    save_drift(args.audit, args.out)
    save_local_drift(args.audit, args.out)
    save_initial_trends(args.audit, args.out)
    save_sign_matrix(args.audit, args.out)


if __name__ == "__main__":
    main()
