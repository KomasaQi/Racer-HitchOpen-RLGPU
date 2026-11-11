"""Open-loop dynamics sanity checks for the fast plant.

Also used as the regression test after any plant or environment change: it runs
with assertions, so a failure is a non-zero exit rather than a printed number
nobody reads.

Checks
  1. straight-line acceleration with throttle
  2. brake step from a rolling start
  3. Ackermann geometry: parallel / exact / extrapolated, left/right symmetry,
     zero-steer continuity
  4. left/right symmetry of the whole plant under mirrored steering
  5. friction-circle invariant |F| <= mu * Fz per wheel
"""
import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))

from fast_plant import FastRacerPlant, preset_parameters  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print("%-52s %s %s" % (name, "PASS" if cond else "FAIL", detail))
    if not cond:
        FAILS.append(name)


def make(n=1, ack=1.0):
    p = FastRacerPlant(preset_parameters("fullsize_racecar_ev"), n, device="cpu")
    p.ackermann_scale = torch.full((n, 1), ack)
    return p


def run(p, s, cmd, seconds, dt=0.01, substeps=10):
    for _ in range(int(round(seconds / dt))):
        s, ex = p.step(s, cmd, dt=dt, substeps=substeps)
    return s, ex


def body(s):
    c, sn = torch.cos(s[:, 2]), torch.sin(s[:, 2])
    vx = c * s[:, 3] + sn * s[:, 4]
    vy = -sn * s[:, 3] + c * s[:, 4]
    return vx, vy, torch.atan2(-vy, vx)


def main():
    print("=== 1. straight line, throttle 0.5, 3 s")
    p = make()
    s = torch.zeros((1, 10))
    p.init_wheels(s)
    s, ex = run(p, s, torch.tensor([[0.0, 0.5, 0.0]]), 3.0)
    v = float(torch.hypot(s[0, 3], s[0, 4]))
    check("straight line reaches > 8 m/s", v > 8.0, "v = %.2f m/s" % v)
    check("straight line stays straight", abs(float(s[0, 1])) < 0.05,
          "lateral offset = %.3f m" % float(s[0, 1]))
    check("drive torque produces forward acceleration", float(ex["ax"]) > 1.0,
          "ax = %.2f m/s^2" % float(ex["ax"]))

    print("=== 2. brake step from 12 m/s, brake 1.0, 1 s")
    p = make()
    s = torch.zeros((1, 10))
    s[0, 3] = 12.0
    p.init_wheels(s)
    s, ex = run(p, s, torch.tensor([[0.0, 0.0, 1.0]]), 1.0)
    v = float(torch.hypot(s[0, 3], s[0, 4]))
    check("hard braking decelerates strongly", -20.0 < float(ex["ax"]) < -4.0,
          "ax = %.2f m/s^2" % float(ex["ax"]))
    check("speed decreased", v < 12.0, "v = %.2f m/s" % v)
    check("no NaN after braking", bool(torch.isfinite(s).all()))

    print("=== 3. Ackermann geometry")
    p = make(3)
    fs0 = p._front_steer(torch.zeros((3, 1)))
    check("zero steer gives exactly zero on both wheels",
          bool((fs0.abs() < 1e-6).all()), "%.6f / %.6f" % (float(fs0[0, 0]), float(fs0[0, 1])))
    for ack, label in ((0.0, "parallel"), (1.0, "exact"), (1.2, "120 %")):
        p.ackermann_scale = torch.full((3, 1), ack)
        f = p._front_steer(torch.full((3, 1), 0.3))
        inner, outer = float(f[0, 0]), float(f[0, 1])
        if ack == 0.0:
            check("ack 0 % keeps both wheels parallel", abs(inner - outer) < 1e-6,
                  "%.4f / %.4f" % (inner, outer))
        else:
            check("ack %s: left is the inner wheel" % label, inner > outer,
                  "%.4f vs %.4f" % (inner, outer))
        if ack == 1.0:
            check("ack exact matches the hand calculation",
                  abs(inner - 0.3298) < 2e-3 and abs(outer - 0.2750) < 2e-3,
                  "%.4f / %.4f" % (inner, outer))

    p.ackermann_scale = torch.full((3, 1), 1.0)
    fp = p._front_steer(torch.full((3, 1), 0.3))
    fn = p._front_steer(torch.full((3, 1), -0.3))
    check("right turn mirrors the left turn",
          abs(float(fn[0, 0]) + float(fp[0, 1])) < 1e-5 and abs(float(fn[0, 1]) + float(fp[0, 0])) < 1e-5,
          "%.4f/%.4f vs %.4f/%.4f" % (float(fn[0, 0]), float(fn[0, 1]), float(fp[0, 0]), float(fp[0, 1])))
    check("right-turn inner angle is negative (no sign flip)",
          float(fn[0, 1]) < float(fn[0, 0]) < 0.0,
          "left %.4f right %.4f" % (float(fn[0, 0]), float(fn[0, 1])))

    print("=== 4. left/right symmetry of the full plant")
    p = make(2)
    s = torch.zeros((2, 10))
    s[0, 3] = 8.0
    s[1, 3] = 8.0
    p.init_wheels(s)
    cmd = torch.tensor([[0.10, 0.30, 0.0], [-0.10, 0.30, 0.0]])
    s, _ = run(p, s, cmd, 2.0)
    check("mirrored steering mirrors the lateral response",
          abs(float(s[0, 1]) + float(s[1, 1])) < 0.2,
          "y = %.2f vs %.2f" % (float(s[0, 1]), float(s[1, 1])))
    check("mirrored steering mirrors the yaw rate",
          abs(float(s[0, 5]) + float(s[1, 5])) < 0.05,
          "r = %.3f vs %.3f" % (float(s[0, 5]), float(s[1, 5])))

    print("=== 5. friction circle")
    p = make(2)
    s = torch.zeros((2, 10))
    s[0, 3] = 8.0
    s[1, 3] = 8.0
    p.init_wheels(s)
    s, ex = run(p, s, torch.tensor([[0.2, 0.6, 0.0], [0.2, 0.6, 0.0]]), 1.0)
    fx, fy, fz = ex["fx_wheel"], ex["fy_wheel"], ex["fz"]
    mu = ex["mu"].clamp(min=1e-3)
    ratio = torch.hypot(fx, fy) / (mu * fz).clamp(min=1e-6)
    check("|F| <= mu Fz (+1 %) on every wheel",
          float(ratio.max()) < 1.01, "max ratio %.3f" % float(ratio.max()))

    print()
    if FAILS:
        print("FAILED:", ", ".join(FAILS))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
