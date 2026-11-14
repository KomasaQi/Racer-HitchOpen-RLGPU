"""Check whether exported positions and body velocities describe one point."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = []
    for path in Path(args.dataset).glob("*__*.csv"):
        df = pd.read_csv(path)
        for name, q in df.groupby("plant"):
            q = q.sort_values("t")
            vx = np.gradient(q.x_world, q.t)
            vy = np.gradient(q.y_world, q.t)
            cp, sp = np.cos(q.psi), np.sin(q.psi)
            ex = cp*vx + sp*vy - q.vx_body
            ey = -sp*vx + cp*vy - q.vy_body
            valid = (q.yaw_rate.abs() > .2) & (q.speed > 3)
            valid.iloc[:2] = False
            valid.iloc[-2:] = False
            if valid.sum() < 5:
                continue
            rate = q.yaw_rate[valid].to_numpy()
            offset_x = np.dot(rate, ey[valid]) / np.dot(rate, rate)
            offset_y = -np.dot(rate, ex[valid]) / np.dot(rate, rate)
            rows.append({"file": path.name, "plant": name,
                "fitted_position_point_x_offset_m": float(offset_x),
                "fitted_position_point_y_offset_m": float(offset_y),
                "raw_velocity_closure_rmse_mps": float(np.sqrt(np.mean(ex[valid]**2+ey[valid]**2))),
                "offset_corrected_velocity_closure_rmse_mps": float(np.sqrt(np.mean(
                    (ex[valid]+rate*offset_y)**2+(ey[valid]-rate*offset_x)**2)))})
    Path(args.output).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
