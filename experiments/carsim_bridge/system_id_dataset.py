"""Generate paired CarSim/FastPlant system-identification trajectories.

Commands are defined in physical units (road-wheel radians, pedal fractions),
held at the 20 Hz controller rate, and applied to both plants from rest.  Every
CSV uses one canonical SI schema and contains the signals needed to identify the
powertrain, pure-slip tyres, lateral response and combined-slip behaviour.

This script collects evidence; it deliberately does not tune parameters.
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

WHEELS = ("FL", "FR", "RL", "RR")
SCENARIOS = (("flat_mu050", 0.50), ("flat_mu075", 0.75), ("flat_mu100", 1.00))


def command(name, t):
    """Return road-wheel steer [rad], throttle and brake at time ``t``."""
    if name.startswith("launch_"):
        return 0.0, float(name.split("_")[1]) / 100.0, 0.0
    if name == "coast":
        return 0.0, 0.40 if t < 3.0 else 0.0, 0.0
    if name == "brake_peak":
        # High-speed full braking provides identifiable kappa on mu100, where
        # normal launches remain almost pure rolling despite full motor torque.
        if t < 4.0:
            return 0.0, 0.80, 0.0
        if t < 5.0:
            return 0.0, 0.0, 0.0
        return 0.0, 0.0, 1.0
    if name.startswith("brake_"):
        level = float(name.split("_")[1]) / 100.0
        return 0.0, 0.40 if t < 3.0 else 0.0, level if t >= 4.0 else 0.0
    if name == "steer_small":
        steer = 0.0 if t < 3.0 else 0.04 * math.sin(2.0 * math.pi * 0.50 * (t - 3.0))
        return steer, 0.30 if t < 3.0 else 0.20, 0.0
    if name == "steer_large":
        steer = 0.0 if t < 3.0 else 0.12 * math.sin(2.0 * math.pi * 0.25 * (t - 3.0))
        return steer, 0.30 if t < 3.0 else 0.25, 0.0
    if name == "steer_peak":
        # Accelerate first, then sweep well beyond the ordinary handling range
        # so high-mu roads expose the lateral peak and post-peak branch.
        steer = 0.0 if t < 4.0 else 0.28 * math.sin(2.0 * math.pi * 0.16 * (t - 4.0))
        return steer, 0.55 if t < 4.0 else 0.12, 0.0
    if name == "combined_power":
        steer = 0.0 if t < 3.0 else 0.14 * math.sin(2.0 * math.pi * 0.22 * (t - 3.0))
        return steer, 0.30 if t < 3.0 else 0.65, 0.0
    if name == "throttle_prbs":
        if t < 2.0:
            return 0.0, 0.25, 0.0
        levels = (0.10, 0.55, 0.25, 0.80, 0.15, 0.45, 0.70, 0.30)
        return 0.0, levels[int((t - 2.0) / 0.50) % len(levels)], 0.0
    raise KeyError("unknown maneuver %s" % name)


DURATIONS = {
    "launch_10": 4.0, "launch_30": 4.0, "launch_50": 4.0, "launch_80": 4.0,
    "coast": 6.0, "brake_10": 6.0, "brake_30": 6.0,
    "steer_small": 9.0, "steer_large": 9.0, "combined_power": 9.0,
    "throttle_prbs": 8.0, "steer_peak": 11.0, "brake_peak": 9.0,
}


BASE = ["plant", "scenario", "mu", "maneuver", "t", "x_world", "y_world", "psi",
        "steer_cmd", "throttle_cmd",
        "brake_cmd", "vx_body", "vy_body", "speed", "beta", "yaw_rate", "ax", "ay",
        "motor_speed"]
PER_WHEEL = ("steer", "omega", "surface_speed", "kappa", "alpha", "fx", "fy", "fz")
HEADER = BASE + ["%s_%s" % (q, w) for q in PER_WHEEL for w in WHEELS]


def _row(prefix, steer, omega, radius, kappa, alpha, fx, fy, fz):
    values = list(prefix)
    for arr in (steer, omega, omega * radius, kappa, alpha, fx, fy, fz):
        values.extend(float(x) for x in arr)
    return values


def fast_trace(scenario, mu, maneuver, duration, dt, calibration=None,
               power_calibration=None, tire_relaxation=False,
               tire_alpha_relaxation=False, combined_slip_model=None,
               combined_slip_blend=0.49, wheel_solver_iters=None,
               stiffness_grip_exponent=None, stiffness_exp_x=None,
               stiffness_exp_y=None):
    params = load_preset("fullsize_racecar_ev")
    if calibration:
        with open(calibration, encoding="utf-8") as f:
            fits = json.load(f)["fits"]
        fx = fits["front_fx"]
        fy = fits["front_fy"]
        params.update({
            "bx": fx["B"], "cx": fx["C"], "ex": fx["E"],
            "by": fy["B"], "cy": fy["C"], "ey": fy["E"],
            "D": 0.5 * (fx["mu_ref"] + fy["mu_ref"]),
            "tyre_scale_front": 1.0, "tyre_scale_rear": 1.0,
            "mu_y_ratio": 1.0, "mu_y_ratio_rear": 1.0,
            "slip_force_plateau": 0.51,
            "load_sensitivity": 0.5 * (fx["load_exponent"] + fy["load_exponent"]),
            "load_sensitivity_model": "power",
            "fz_ref_front": fx["fz_ref"],
            "fz_ref_rear": fits["rear_fx"]["fz_ref"],
        })
    if power_calibration:
        with open(power_calibration, encoding="utf-8") as f:
            power_fit = json.load(f)
        params.update({
            "throttle_power_gain": power_fit["throttle_power_gain"],
            "throttle_power_rolloff": power_fit["throttle_power_rolloff"],
        })
    params["tire_relaxation"] = bool(tire_relaxation)
    if tire_alpha_relaxation:
        params["relax_kappa"] = False
        params["relax_alpha"] = True
    if combined_slip_model:
        params["combined_slip_model"] = combined_slip_model
        params["combined_slip_blend"] = combined_slip_blend
    if wheel_solver_iters is not None:
        params["wheel_solver_bisection_iters"] = int(wheel_solver_iters)
    exp_x = stiffness_exp_x if stiffness_exp_x is not None else stiffness_grip_exponent
    exp_y = stiffness_exp_y if stiffness_exp_y is not None else stiffness_grip_exponent
    if exp_x is not None:
        params["tire_stiffness_grip_exponent_x"] = float(exp_x)
    if exp_y is not None:
        params["tire_stiffness_grip_exponent_y"] = float(exp_y)
    plant = FastRacerPlant(params, 1, "cpu")
    plant.tyre_scale.fill_(mu)
    state = torch.zeros((1, 10))
    plant.init_wheels(state)
    radius = plant.geometry()[2][0].detach().numpy()
    rows = []
    for k in range(int(round(duration / dt))):
        t0 = k * dt
        delta, throttle, brake = command(maneuver, t0)
        cmd = torch.tensor([[delta, throttle, brake]], dtype=state.dtype)
        state, ex = plant.step(state, cmd, dt=dt, substeps=5, mode="pedals")
        psi = float(state[0, 2])
        c, s = math.cos(psi), math.sin(psi)
        vxw, vyw = float(state[0, 3]), float(state[0, 4])
        vxb, vyb = c * vxw + s * vyw, -s * vxw + c * vyw
        speed = math.hypot(vxb, vyb)
        beta = math.atan2(vyb, vxb) if speed >= 0.5 else 0.0
        motor_rpm = float(ex["motor_speed"][0]) * 60.0 / (2.0 * math.pi)
        prefix = ["fast", scenario, mu, maneuver, (k + 1) * dt,
                  float(state[0, 0]), float(state[0, 1]), psi,
                  delta, throttle, brake,
                  vxb, vyb, speed, beta, float(state[0, 5]), float(ex["ax"][0]),
                  float(ex["ay"][0]), motor_rpm]
        steer = np.array([float(ex["delta"][0, 0]), float(ex["delta"][0, 1]), 0.0, 0.0])
        rows.append(_row(prefix, steer, state[0, 6:10].detach().numpy(), radius,
                         ex["kappa_power"][0].detach().numpy(), ex["alpha"][0].detach().numpy(),
                         ex["fx_wheel"][0].detach().numpy(), ex["fy_wheel"][0].detach().numpy(),
                         ex["fz"][0].detach().numpy()))
    return rows


def carsim_trace(scenario, mu, maneuver, duration, dt):
    env = CarSimFlatEnv(scenario, dt=dt)
    env.reset()
    rows = []
    try:
        for k in range(int(round(duration / dt))):
            t0 = k * dt
            delta, throttle, brake = command(maneuver, t0)
            env.step([delta / MAX_ROAD_WHEEL_RAD, throttle, brake])
            raw = env.raw()
            vxb, vyb = raw["Vx"] / 3.6, raw["Vy"] / 3.6
            speed = math.hypot(vxb, vyb)
            suffix = CARSIM_WHEEL_SUFFIXES
            omega = np.array([raw["AVy_" + x] for x in suffix]) * 2.0 * math.pi / 60.0
            radius = np.array([0.332, 0.332, 0.347, 0.347])
            steer = np.array([math.radians(raw["Steer_" + suffix[0]]),
                              math.radians(raw["Steer_" + suffix[1]]), 0.0, 0.0])
            prefix = ["carsim", scenario, mu, maneuver, (k + 1) * dt,
                      raw["Xo"], raw["Yo"], math.radians(raw["Yaw"]),
                      delta, throttle, brake,
                      vxb, vyb, speed, math.radians(raw["Beta"]) if speed >= 0.5 else 0.0,
                      math.radians(raw["AVz"]), raw["Ax"] * 9.81, raw["Ay"] * 9.81,
                      raw["AV_Motor"]]
            rows.append(_row(
                prefix, steer, omega, radius,
                np.array([raw["Kappa_" + x] for x in suffix]),
                np.radians([raw["Alpha_" + x] for x in suffix]),
                np.array([raw["Fx_WC_" + x] for x in suffix]),
                np.array([raw["Fy_WC_" + x] for x in suffix]),
                np.array([raw["Fz_WC_" + x] for x in suffix])))
    finally:
        env.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--maneuvers", default=",".join(DURATIONS))
    ap.add_argument("--scenarios", default=",".join(x[0] for x in SCENARIOS))
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--fast-calibration", default=None,
                    help="compact CarSim tire fit JSON to test without changing the preset")
    ap.add_argument("--power-calibration", default=None,
                    help="pedal-to-power fit JSON to test without changing the preset")
    ap.add_argument("--tire-relaxation", action="store_true",
                    help="test CarSim load/slip dependent kappa/alpha relaxation")
    ap.add_argument("--tire-alpha-relaxation", action="store_true",
                    help="test only the standard alpha relaxation (not Kappa mode 2)")
    ap.add_argument("--combined-slip-model",
                    choices=("isotropic", "ellipse", "compact_transition",
                             "carsim_transition"),
                    default=None)
    ap.add_argument("--combined-slip-blend", type=float, default=0.49)
    ap.add_argument("--wheel-solver-iters", type=int, default=None,
                    help="candidate wheel-speed bisection count")
    ap.add_argument("--stiffness-grip-exponent", type=float, default=None,
                    help="legacy scalar override; omitted means use production preset")
    ap.add_argument("--stiffness-grip-exponent-x", type=float, default=None)
    ap.add_argument("--stiffness-grip-exponent-y", type=float, default=None)
    ap.add_argument("--carsim-source", default=None,
                    help="reuse CarSim rows from an existing paired dataset")
    ap.add_argument("--resume", action="store_true",
                    help="skip maneuver CSVs already present in --out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    selected_m = [x.strip() for x in args.maneuvers.split(",") if x.strip()]
    selected_s = set(x.strip() for x in args.scenarios.split(",") if x.strip())
    unknown = sorted(set(selected_m) - set(DURATIONS))
    if unknown:
        raise ValueError("unknown maneuvers: %s" % unknown)
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
    manifest = {"schema": HEADER, "dt": args.dt, "scenarios": [], "maneuvers": selected_m,
                "stiffness_grip_exponent": args.stiffness_grip_exponent,
                "stiffness_grip_exponent_x": effective_x,
                "stiffness_grip_exponent_y": effective_y,
                "fastplant_mapping": ("production preset; tyre_scale=road mu" if not args.fast_calibration
                                      else "compact tire candidate %s" % args.fast_calibration)}
    for scenario, mu in SCENARIOS:
        if scenario not in selected_s:
            continue
        manifest["scenarios"].append({"name": scenario, "mu": mu})
        for maneuver in selected_m:
            duration = DURATIONS[maneuver]
            path = os.path.join(args.out, "%s__%s.csv" % (scenario, maneuver))
            if args.resume and os.path.exists(path) and os.path.getsize(path) > 0:
                print("%-12s %-16s [existing]" % (scenario, maneuver), flush=True)
                continue
            print("%-12s %-16s" % (scenario, maneuver), flush=True)
            fast = fast_trace(scenario, mu, maneuver, duration, args.dt,
                              args.fast_calibration, args.power_calibration,
                              args.tire_relaxation, args.tire_alpha_relaxation,
                              args.combined_slip_model, args.combined_slip_blend,
                              args.wheel_solver_iters, args.stiffness_grip_exponent,
                              args.stiffness_grip_exponent_x,
                              args.stiffness_grip_exponent_y)
            if args.carsim_source:
                old_path = os.path.join(args.carsim_source, "%s__%s.csv" % (scenario, maneuver))
                with open(old_path, newline="") as old:
                    source_rows = [r for r in csv.DictReader(old) if r["plant"] == "carsim"]
                carsim = []
                for source in source_rows:
                    upgraded = {name: source.get(name, "nan") for name in HEADER}
                    # Legacy files used the 332-mm front radius for every logged
                    # surface speed.  Raw omega is authoritative, so repair only
                    # this derived diagnostic while preserving CarSim outputs.
                    for wheel, radius in zip(WHEELS, (0.332, 0.332, 0.347, 0.347)):
                        upgraded["surface_speed_" + wheel] = (
                            float(upgraded["omega_" + wheel]) * radius)
                    carsim.append([upgraded[name] for name in HEADER])
            else:
                carsim = carsim_trace(scenario, mu, maneuver, duration, args.dt)
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(HEADER)
                writer.writerows(fast + carsim)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
