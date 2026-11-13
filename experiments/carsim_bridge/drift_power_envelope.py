"""Quantify the power/friction envelope for sustained RWD power drift.

This is a feasibility bound, not a trim solver.  It combines the production
FastPlant motor map with the measured CarSim tyre peak and reports:

* friction-limit speed versus path radius;
* full-pedal steady contact force versus speed and rear slip ratio;
* the electrical power required to hold selected rear-axle forces.

Positive ``kappa`` means driven-wheel overspeed, so tyre surface speed is
``v * (1 + kappa)``.  That factor is essential: wheelspin consumes power even
when it does not increase vehicle speed.
"""
import argparse
import csv
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))

from fast_plant import FastRacerPlant, load_preset  # noqa: E402


ROAD_SCALES = (0.50, 0.75, 1.00)
RADII_M = (10.0, 15.0, 20.0, 30.0, 50.0)
SPEEDS_MPS = (10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0)
KAPPAS = (0.30, 0.70, 1.00)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    params = load_preset("fullsize_racecar_ev")
    plant = FastRacerPlant(params, 1, device="cpu", dtype=torch.float64)
    radius = float(params["rR"])
    peak_mu = float(params["D"])
    mass = float(params["m"])
    g = float(params["g"])
    rear_static_fz = mass * g * float(params["lF"]) / (
        float(params["lF"]) + float(params["lR"]))

    friction_rows = []
    for road in ROAD_SCALES:
        effective_mu = road * peak_mu
        for turn_radius in RADII_M:
            friction_rows.append({
                "road_scale": road,
                "effective_peak_mu": effective_mu,
                "radius_m": turn_radius,
                "curvature_1pm": 1.0 / turn_radius,
                "friction_limit_speed_mps": math.sqrt(effective_mu * g * turn_radius),
            })

    motor_rows = []
    for speed in SPEEDS_MPS:
        for kappa in KAPPAS:
            surface_speed = speed * (1.0 + kappa)
            omega = torch.full((1, 2), surface_speed / radius, dtype=torch.float64)
            axle, motor_torque, motor_speed = plant._motor_torque(
                omega, torch.ones(1, dtype=torch.float64),
                torch.zeros(1, dtype=torch.float64))
            rpm = float(motor_speed[0]) * 60.0 / (2.0 * math.pi)
            mechanical_power = float(motor_torque[0] * motor_speed[0])
            efficiency = float(plant._motor_efficiency(
                torch.tensor([rpm], dtype=torch.float64), motor_torque)[0])
            contact_force = float(axle[0]) / radius
            motor_rows.append({
                "speed_mps": speed,
                "kappa_rear": kappa,
                "surface_speed_mps": surface_speed,
                "motor_rpm": rpm,
                "motor_torque_nm": float(motor_torque[0]),
                "motor_efficiency": efficiency,
                "mechanical_motor_power_kw": mechanical_power / 1000.0,
                "wheel_contact_power_kw": contact_force * surface_speed / 1000.0,
                "full_pedal_contact_force_n": contact_force,
            })

    power_rows = []
    for speed in SPEEDS_MPS:
        for kappa in KAPPAS:
            surface_speed = speed * (1.0 + kappa)
            for force_fraction in (0.25, 0.50, 0.75, 1.00):
                rear_force = force_fraction * peak_mu * rear_static_fz
                # Conservative high-torque/high-speed edge of the measured
                # CarSim carpet.  The exact motor lookup is reported above.
                efficiency = 0.684
                wheel_kw = rear_force * surface_speed / 1000.0
                power_rows.append({
                    "speed_mps": speed,
                    "kappa_rear": kappa,
                    "rear_peak_force_fraction": force_fraction,
                    "rear_force_n": rear_force,
                    "wheel_power_kw": wheel_kw,
                    "electrical_power_kw_at_eff_0684": wheel_kw /
                    (float(params["drive_efficiency"]) * efficiency),
                })

    def write_csv(name, rows):
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    write_csv("friction_limit.csv", friction_rows)
    write_csv("motor_force_envelope.csv", motor_rows)
    write_csv("required_power.csv", power_rows)
    summary = {
        "mass_kg": mass,
        "rear_static_fz_n": rear_static_fz,
        "tire_peak_mu_at_road_1": peak_mu,
        "rear_peak_pure_longitudinal_force_n": peak_mu * rear_static_fz,
        "electrical_power_limit_kw": float(params["max_power_w"]) / 1000.0,
        "drive_efficiency": float(params["drive_efficiency"]),
        "interpretation": (
            "High rear slip multiplies contact-patch power by (1+kappa). "
            "At high speed the motor can sustain only a fraction of the rear "
            "pure-slip peak, so tight paths/lower friction-limit speeds are a "
            "physical curriculum requirement, not merely a reward preference."
        ),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
