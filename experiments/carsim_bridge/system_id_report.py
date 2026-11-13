"""Score a paired system-identification dataset without tuning the plant."""
import argparse
import glob
import os

import numpy as np
import pandas as pd


FIT_MANEUVERS = {"launch_10", "launch_30", "launch_50", "coast", "brake_10",
                 "steer_small", "combined_power"}
VALID_MANEUVERS = {"launch_80", "brake_30", "steer_large", "throttle_prbs"}

SCALES = {
    "speed": 10.0, "beta": 0.5, "yaw_rate": 1.0, "ax": 5.0, "ay": 5.0,
    "motor_speed": 3000.0,
}
for q, scale in (("surface_speed", 10.0), ("kappa", 0.5), ("alpha", 0.2),
                 ("fx", 5000.0), ("fy", 5000.0), ("fz", 5000.0)):
    for wheel in ("FL", "FR", "RL", "RR"):
        SCALES["%s_%s" % (q, wheel)] = scale


def angle_error(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def paired(path):
    frame = pd.read_csv(path)
    fast = frame[frame.plant == "fast"].sort_values("t").reset_index(drop=True)
    carsim = frame[frame.plant == "carsim"].sort_values("t").reset_index(drop=True)
    if len(fast) != len(carsim) or not np.allclose(fast.t, carsim.t, atol=1e-9):
        raise ValueError("unpaired timeline in %s" % path)
    return fast, carsim


def score_file(path):
    fast, carsim = paired(path)
    scenario = str(carsim.scenario.iloc[0])
    maneuver = str(carsim.maneuver.iloc[0])
    split = "fit" if maneuver in FIT_MANEUVERS else "validation"
    rows = []
    for signal, scale in SCALES.items():
        mask = np.isfinite(fast[signal]) & np.isfinite(carsim[signal])
        # Slip ratios are ill-conditioned around rest and uninformative once a
        # vehicle is reversing/spinning with |kappa| in the hundreds.
        if signal.startswith("kappa_"):
            mask &= (fast.speed >= 1.0) & (carsim.speed >= 1.0)
            mask &= (fast[signal].abs() <= 5.0) & (carsim[signal].abs() <= 5.0)
        if signal == "beta" or signal.startswith("alpha_"):
            err = angle_error(fast.loc[mask, signal].to_numpy(),
                              carsim.loc[mask, signal].to_numpy())
        else:
            err = (fast.loc[mask, signal] - carsim.loc[mask, signal]).to_numpy()
        rmse = float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan")
        mae = float(np.mean(np.abs(err))) if len(err) else float("nan")
        rows.append({"scenario": scenario, "maneuver": maneuver, "split": split,
                     "signal": signal, "n": int(len(err)),
                     "valid_fraction": float(mask.mean()), "rmse": rmse, "mae": mae,
                     "scale": scale, "nrmse": rmse / scale})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.dataset, "*.csv")))
    if not paths:
        raise ValueError("no CSV files in %s" % args.dataset)
    metrics = pd.DataFrame([row for path in paths for row in score_file(path)])
    metrics.to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    aggregate = (metrics.groupby(["split", "scenario", "signal"], as_index=False)
                 .agg(nrmse=("nrmse", "mean"), rmse=("rmse", "mean"),
                      valid_fraction=("valid_fraction", "mean"), n=("n", "sum")))
    aggregate.to_csv(os.path.join(args.out, "aggregate.csv"), index=False)
    families = metrics.assign(family=metrics.signal.str.replace(
        r"_(FL|FR|RL|RR)$", "", regex=True))
    family = (families.groupby(["split", "scenario", "family"], as_index=False)
              .agg(nrmse=("nrmse", "mean"), valid_fraction=("valid_fraction", "mean")))
    family.to_csv(os.path.join(args.out, "families.csv"), index=False)

    lines = ["# Production preset system-ID baseline", "",
             "Maneuver-level split prevents time-adjacent samples from leaking between fit and validation.", "",
             "- Fit: `%s`" % "`, `".join(sorted(FIT_MANEUVERS)),
             "- Validation: `%s`" % "`, `".join(sorted(VALID_MANEUVERS)), "",
             "NRMSE uses fixed engineering scales, not each trace's range. Kappa scoring excludes speeds below 1 m/s and |kappa| > 5.", ""]
    for split in ("fit", "validation"):
        lines += ["## %s" % split.title(), "",
                  "| scenario | family | NRMSE | valid fraction |",
                  "|---|---|---:|---:|"]
        for row in family[family.split == split].sort_values(["scenario", "family"]).itertuples():
            lines.append("| %s | %s | %.3f | %.3f |" %
                         (row.scenario, row.family, row.nrmse, row.valid_fraction))
        lines.append("")
    with open(os.path.join(args.out, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(family.pivot_table(index=["split", "scenario"], columns="family",
                             values="nrmse").round(3).to_string())


if __name__ == "__main__":
    main()
