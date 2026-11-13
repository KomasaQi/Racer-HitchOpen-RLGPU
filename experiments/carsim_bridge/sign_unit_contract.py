"""Verify canonical signs/units on the first excited frames of paired data."""
import argparse
import json
import os

import pandas as pd


def sample(dataset, scenario, maneuver, t):
    f = pd.read_csv(os.path.join(dataset, scenario + "__" + maneuver + ".csv"))
    out = {}
    for plant in ("fast", "carsim"):
        q = f[f.plant == plant]
        i = (q.t - t).abs().idxmin()
        out[plant] = q.loc[i]
    return out


def sgn(x, dead=1e-7):
    x = float(x)
    return 1 if x > dead else (-1 if x < -dead else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    checks = []

    def check(name, rows, field, expected):
        values = {p: float(r[field]) for p, r in rows.items()}
        signs = {p: sgn(v) for p, v in values.items()}
        ok = all(v == expected for v in signs.values())
        checks.append({"name": name, "field": field, "expected_sign": expected,
                       "values": values, "signs": signs, "pass": ok})

    steer = sample(args.dataset, "flat_mu100", "steer_small", 3.10)
    for name, field, expected in (
            ("positive command means left road-wheel steer", "steer_FL", 1),
            ("left turn body lateral velocity is positive", "vy_body", 1),
            ("left turn sideslip beta is positive", "beta", 1),
            ("counter-clockwise/left yaw rate is positive", "yaw_rate", 1),
            ("left body lateral acceleration is positive", "ay", 1),
            ("tire alpha is velocity angle in wheel frame", "alpha_FL", -1),
            ("positive Fy acts to wheel/vehicle left", "fy_FL", 1)):
        check(name, steer, field, expected)

    launch = sample(args.dataset, "flat_mu100", "launch_50", 1.00)
    for name, field, expected in (
            ("forward acceleration is positive", "ax", 1),
            ("forward wheel rotation is positive", "omega_RL", 1),
            ("driven overspeed kappa is positive", "kappa_RL", 1),
            ("tractive longitudinal force is positive", "fx_RL", 1)):
        check(name, launch, field, expected)

    brake = sample(args.dataset, "flat_mu100", "brake_30", 4.10)
    for name, field, expected in (
            ("braking longitudinal acceleration is negative", "ax", -1),
            ("braking kappa is negative", "kappa_FL", -1),
            ("braking longitudinal tire force is negative", "fx_FL", -1)):
        check(name, brake, field, expected)

    result = {"schema": "canonical SI except command pedals dimensionless",
              "checks": checks, "all_pass": all(x["pass"] for x in checks)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    for x in checks:
        print(("PASS" if x["pass"] else "FAIL"), x["name"], x["values"])
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
