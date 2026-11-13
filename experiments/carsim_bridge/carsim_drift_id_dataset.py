"""Collect true combined-slip CarSim data around verified drift trims.

Privileged Beta/Kappa feedback is used only as an identification fixture to
keep CarSim near the unstable operating point.  The resulting physical command
sequence is then replayed open-loop in FastPlant from the same rest condition.
No privileged signal is added to the policy observation contract.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))

from envs.carsim_flat_env import (CarSimFlatEnv, CARSIM_WHEEL_SUFFIXES,
                                  MAX_ROAD_WHEEL_RAD)  # noqa: E402
from fast_plant import FastRacerPlant, load_preset  # noqa: E402
from experiments.carsim_bridge.system_id_dataset import HEADER, _row  # noqa: E402


CONFIGS = {
    "flat_mu050": {"mu": 0.50, "beta_target": -0.25, "counter": -0.10,
                    "beta_kp": 1.0, "throttle_base": 0.55, "kappa_kp": 0.10},
    "flat_mu075": {"mu": 0.75, "beta_target": -0.40, "counter": -0.20,
                    "beta_kp": 1.0, "throttle_base": 0.55, "kappa_kp": 0.30},
}
MANEUVER = "drift_trim_prbs"


def _prbs(t):
    """Small deterministic zero-mean command perturbations after six seconds."""
    if t < 6.0:
        return 0.0, 0.0
    steer_seq = (1, -1, 1, 1, -1, -1, 1, -1)
    throttle_seq = (-1, 1, 1, -1, 1, -1, -1, 1)
    i = int((t - 6.0) / 0.40)
    return 0.018 * steer_seq[i % len(steer_seq)], \
        0.035 * throttle_seq[(i // len(steer_seq) + i) % len(throttle_seq)]


def _carsim_row(raw, scenario, mu, t, cmd):
    delta, throttle, brake = cmd
    vxb, vyb = raw["Vx"] / 3.6, raw["Vy"] / 3.6
    speed = math.hypot(vxb, vyb)
    suffix = CARSIM_WHEEL_SUFFIXES
    omega = np.array([raw["AVy_" + x] for x in suffix]) * 2.0 * math.pi / 60.0
    radius = np.array([0.332, 0.332, 0.347, 0.347])
    steer = np.array([math.radians(raw["Steer_" + suffix[0]]),
                      math.radians(raw["Steer_" + suffix[1]]), 0.0, 0.0])
    prefix = ["carsim", scenario, mu, MANEUVER, t,
              raw["Xo"], raw["Yo"], math.radians(raw["Yaw"]),
              delta, throttle, brake,
              vxb, vyb, speed, math.radians(raw["Beta"]) if speed >= 0.5 else 0.0,
              math.radians(raw["AVz"]), raw["Ax"] * 9.81, raw["Ay"] * 9.81,
              raw["AV_Motor"]]
    return _row(prefix, steer, omega, radius,
                np.array([raw["Kappa_" + x] for x in suffix]),
                np.radians([raw["Alpha_" + x] for x in suffix]),
                np.array([raw["Fx_WC_" + x] for x in suffix]),
                np.array([raw["Fy_WC_" + x] for x in suffix]),
                np.array([raw["Fz_WC_" + x] for x in suffix]))


def collect_carsim(scenario, cfg, duration, dt):
    env = CarSimFlatEnv(scenario, dt=dt)
    env.reset()
    commands, rows = [], []
    try:
        for k in range(int(round(duration / dt))):
            t = k * dt
            raw0 = env.raw()
            speed0 = math.hypot(raw0["Vx"], raw0["Vy"]) / 3.6
            beta = math.radians(raw0["Beta"]) if speed0 >= 0.5 else 0.0
            kappa = 0.5 * (raw0["Kappa_L2"] + raw0["Kappa_R2"])
            if t < 2.0:
                cmd = (0.0, 0.40, 0.0)
            else:
                ds, dq = _prbs(t)
                steer = cfg["counter"] + cfg["beta_kp"] * (beta - cfg["beta_target"]) + ds
                throttle = cfg["throttle_base"] + cfg["kappa_kp"] * (0.70 - np.clip(kappa, -2, 3)) + dq
                cmd = (float(np.clip(steer, -MAX_ROAD_WHEEL_RAD, MAX_ROAD_WHEEL_RAD)),
                       float(np.clip(throttle, 0.35, 1.0)), 0.0)
            env.step([cmd[0] / MAX_ROAD_WHEEL_RAD, cmd[1], cmd[2]])
            commands.append(cmd)
            rows.append(_carsim_row(env.raw(), scenario, cfg["mu"], (k + 1) * dt, cmd))
    finally:
        env.close()
    return commands, rows


def replay_fast(scenario, cfg, commands, dt, start_index=0, initial_row=None,
                plant_label="fast", combined_slip_model=None,
                combined_slip_blend=0.49, tire_alpha_relaxation=False,
                tire_calibration=None, power_calibration=None,
                wheel_solver_iters=None):
    params = load_preset("fullsize_racecar_ev")
    if power_calibration:
        with open(power_calibration, encoding="utf-8") as f:
            power = json.load(f)
        params["throttle_power_gain"] = float(power["throttle_power_gain"])
        params["throttle_power_rolloff"] = float(power["throttle_power_rolloff"])
    if tire_calibration:
        with open(tire_calibration, encoding="utf-8") as f:
            fits = json.load(f)["fits"]
        fx, fy = fits["front_fx"], fits["front_fy"]
        params.update({
            "bx": fx["B"], "cx": fx["C"], "ex": fx["E"],
            "by": fy["B"], "cy": fy["C"], "ey": fy["E"],
            "D": 0.5 * (fx["mu_ref"] + fy["mu_ref"]),
            "load_sensitivity": 0.5 * (fx["load_exponent"] + fy["load_exponent"]),
            "fz_ref_front": fx["fz_ref"],
            "fz_ref_rear": fits["rear_fx"]["fz_ref"],
        })
    if combined_slip_model:
        params["combined_slip_model"] = combined_slip_model
        params["combined_slip_blend"] = combined_slip_blend
    if tire_alpha_relaxation:
        params["tire_relaxation"] = True
        params["relax_kappa"] = False
        params["relax_alpha"] = True
    if wheel_solver_iters is not None:
        params["wheel_solver_bisection_iters"] = int(wheel_solver_iters)
    plant = FastRacerPlant(params, 1, "cpu")
    plant.tyre_scale.fill_(cfg["mu"])
    state = torch.zeros((1, 10))
    if initial_row is None:
        plant.init_wheels(state)
    else:
        idx = {name: i for i, name in enumerate(HEADER)}
        state[0, 0] = float(initial_row[idx["x_world"]])
        state[0, 1] = float(initial_row[idx["y_world"]])
        state[0, 2] = float(initial_row[idx["psi"]])
        c0, s0 = math.cos(float(state[0, 2])), math.sin(float(state[0, 2]))
        vxb0 = float(initial_row[idx["vx_body"]])
        vyb0 = float(initial_row[idx["vy_body"]])
        state[0, 3] = c0 * vxb0 - s0 * vyb0
        state[0, 4] = s0 * vxb0 + c0 * vyb0
        state[0, 5] = float(initial_row[idx["yaw_rate"]])
        for j, wheel in enumerate(("FL", "FR", "RL", "RR")):
            state[0, 6 + j] = float(initial_row[idx["omega_" + wheel]])
    radius = plant.geometry()[2][0].detach().numpy()
    rows = []
    for k in range(start_index, len(commands)):
        delta, throttle, brake = commands[k]
        command = torch.tensor([[delta, throttle, brake]], dtype=state.dtype)
        state, ex = plant.step(state, command, dt=dt, substeps=5, mode="pedals")
        psi = float(state[0, 2])
        c, s = math.cos(psi), math.sin(psi)
        vxw, vyw = float(state[0, 3]), float(state[0, 4])
        vxb, vyb = c * vxw + s * vyw, -s * vxw + c * vyw
        speed = math.hypot(vxb, vyb)
        prefix = [plant_label, scenario, cfg["mu"], MANEUVER, (k + 1) * dt,
                  float(state[0, 0]), float(state[0, 1]), psi,
                  delta, throttle, brake, vxb, vyb, speed,
                  math.atan2(vyb, vxb) if speed >= 0.5 else 0.0,
                  float(state[0, 5]), float(ex["ax"][0]), float(ex["ay"][0]),
                  float(ex["motor_speed"][0]) * 60.0 / (2.0 * math.pi)]
        steer = np.array([float(ex["delta"][0, 0]), float(ex["delta"][0, 1]), 0.0, 0.0])
        rows.append(_row(prefix, steer, state[0, 6:10].detach().numpy(), radius,
                         ex["kappa_power"][0].detach().numpy(),
                         ex["alpha"][0].detach().numpy(),
                         ex["fx_wheel"][0].detach().numpy(),
                         ex["fy_wheel"][0].detach().numpy(),
                         ex["fz"][0].detach().numpy()))
    return rows


def metrics(rows, start=6.0):
    idx = {name: i for i, name in enumerate(HEADER)}
    a = np.asarray([[float(row[idx[name]]) for name in
                     ("t", "steer_cmd", "throttle_cmd", "speed", "beta", "yaw_rate",
                      "kappa_RL", "kappa_RR")] for row in rows])
    a = a[a[:, 0] >= start]
    kappa = 0.5 * (a[:, 6] + a[:, 7])
    strict = ((np.abs(a[:, 4]) >= 0.20) & (np.abs(a[:, 4]) <= 0.80) &
              (kappa >= 0.30) & (kappa <= 5.0) & (a[:, 2] >= 0.40) &
              ((a[:, 1] * a[:, 5]) < 0.0))
    return {"strict_fraction_after_6s": float(strict.mean()),
            "speed_mean": float(a[:, 3].mean()), "beta_mean": float(a[:, 4].mean()),
            "yaw_mean": float(a[:, 5].mean()), "kappa_rear_mean": float(kappa.mean()),
            "throttle_mean": float(a[:, 2].mean()), "steer_mean": float(a[:, 1].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenarios", default=",".join(CONFIGS))
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--combined-slip-model",
                    choices=("isotropic", "ellipse", "compact_transition",
                             "carsim_transition"),
                    default="ellipse")
    ap.add_argument("--combined-slip-blend", type=float, default=0.49)
    ap.add_argument("--wheel-solver-iters", type=int, default=None,
                    help="candidate wheel-speed bisection count")
    ap.add_argument("--tire-alpha-relaxation", action="store_true")
    ap.add_argument("--tire-calibration", default=None,
                    help="compact analytic tyre-fit JSON used by FastPlant replay")
    ap.add_argument("--power-calibration", default=None,
                    help="pedal-to-power JSON used by FastPlant replay")
    ap.add_argument("--carsim-source", default=None,
                    help="reuse CarSim rows/commands from an existing drift dataset")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    selected = set(x.strip() for x in args.scenarios.split(",") if x.strip())
    summary = []
    for scenario, cfg in CONFIGS.items():
        if scenario not in selected:
            continue
        if args.carsim_source:
            source = os.path.join(args.carsim_source, scenario + "__" + MANEUVER + ".csv")
            with open(source, newline="") as f:
                source_rows = [row for row in csv.DictReader(f) if row["plant"] == "carsim"]
            carsim = [[row[name] for name in HEADER] for row in source_rows]
            commands = [(float(row["steer_cmd"]), float(row["throttle_cmd"]),
                         float(row["brake_cmd"])) for row in source_rows]
            print("reuse  ", scenario, flush=True)
        else:
            print("collect", scenario, flush=True)
            commands, carsim = collect_carsim(scenario, cfg, args.duration, args.dt)
        print("replay ", scenario, flush=True)
        fast = replay_fast(scenario, cfg, commands, args.dt,
                           combined_slip_model=args.combined_slip_model,
                           combined_slip_blend=args.combined_slip_blend,
                           tire_alpha_relaxation=args.tire_alpha_relaxation,
                           tire_calibration=args.tire_calibration,
                           power_calibration=args.power_calibration,
                           wheel_solver_iters=args.wheel_solver_iters)
        local_index = int(round(6.0 / args.dt))
        fast_local = replay_fast(scenario, cfg, commands, args.dt,
                                 start_index=local_index,
                                 initial_row=carsim[local_index - 1],
                                 plant_label="fast_local",
                                 combined_slip_model=args.combined_slip_model,
                                 combined_slip_blend=args.combined_slip_blend,
                                 tire_alpha_relaxation=args.tire_alpha_relaxation,
                                 tire_calibration=args.tire_calibration,
                                 power_calibration=args.power_calibration,
                                 wheel_solver_iters=args.wheel_solver_iters)
        path = os.path.join(args.out, scenario + "__" + MANEUVER + ".csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(fast + fast_local + carsim)
        summary.append({"scenario": scenario, "plant": "carsim", **metrics(carsim)})
        summary.append({"scenario": scenario, "plant": "fast", **metrics(fast)})
        summary.append({"scenario": scenario, "plant": "fast_local", **metrics(fast_local)})
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": HEADER, "duration": args.duration, "dt": args.dt,
                   "prbs_start": 6.0, "configs": CONFIGS,
                   "combined_slip_model": args.combined_slip_model,
                   "combined_slip_blend": args.combined_slip_blend,
                   "tire_alpha_relaxation": args.tire_alpha_relaxation,
                   "power_calibration": args.power_calibration,
                   "warning": "Beta/Kappa feedback is an ID fixture, not an actor input."},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
