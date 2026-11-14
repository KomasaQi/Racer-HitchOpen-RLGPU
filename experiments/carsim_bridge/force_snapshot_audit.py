"""Separate load/steering/slip-state errors from the compact tire law."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xcar-simulation"))
from fast_plant import FastRacerPlant, load_preset


def audit(path):
    q = pd.read_csv(path)
    q = q[(q.plant == "carsim") & (q.t >= 6.) & (q.speed > 5.)]
    if len(q) < 4:
        return None
    p = FastRacerPlant(load_preset("fullsize_racecar_ev"), len(q), "cpu")
    p.tyre_scale.fill_(float(q.mu.iloc[0]))
    def t(columns):
        return torch.tensor(q[columns].to_numpy(), dtype=torch.float32)
    wheels = ("FL", "FR", "RL", "RR")
    s = torch.zeros(len(q), 10)
    s[:, 3:6] = t(["vx_body", "vy_body", "yaw_rate"])
    s[:, 6:10] = t(["omega_"+w for w in wheels])
    u = t(["steer_cmd", "throttle_cmd", "brake_cmd"])
    p._ax_guess.copy_(t("ax")); p._ay_guess.copy_(t("ay"))
    _, native = p.derivatives(s, u)
    steer = t(["steer_FL", "steer_FR"])
    fz = t(["fz_"+w for w in wheels])
    kappa = t(["kappa_"+w for w in wheels])
    alpha = t(["alpha_"+w for w in wheels])
    measured_loads = p._wheel_forces(s, steer, fz, p.geometry())
    measured_slips = p._wheel_forces(s, steer, fz, p.geometry(),
        kappa_power_override=kappa, alpha_override=alpha)
    def stats(a, b):
        d = (a-b).detach().numpy()
        return {"bias": np.mean(d, axis=0).tolist(),
                "rmse": np.sqrt(np.mean(d*d, axis=0)).tolist()}
    result = {"samples": len(q), "wheel_order": wheels,
              "steering_rad": stats(native["delta"], steer),
              "loads_N": stats(native["fz"], fz),
              "kinematic_kappa": stats(-measured_loads["sx"], kappa),
              "kinematic_alpha_rad": stats(measured_loads["alpha"], alpha)}
    for name, ex in (("native", native), ("measured_loads_steer", measured_loads),
                     ("measured_loads_steer_slips", measured_slips)):
        result[name] = {axis: stats(ex[axis+"_wheel"], t([axis+"_"+w for w in wheels]))
                        for axis in ("fx", "fy")}
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    result = {path.name: audit(path) for path in Path(args.dataset).glob("*__*.csv")}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
