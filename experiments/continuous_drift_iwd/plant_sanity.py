"""Open-loop dynamics sanity checks for the fast plant.

Covers: straight-line acceleration, constant-steer circle, brake step, and the
left/right symmetry of the Ackermann blend.  Run after any plant change.
"""
import math
import os
import sys

import torch

ROOT = r"E:\我的文件\大学学习\_硕士学习\博士研究\课程_世界模型-动作与探索\Final_Project\Racer-HitchOpen-RLGPU"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))

from fast_plant import FastRacerPlant, preset_parameters  # noqa: E402


def make(n):
    p = FastRacerPlant(preset_parameters("fullsize_racecar_ev"), n, device="cpu")
    return p


def run(p, s, cmd, seconds, dt=0.01, substeps=10):
    steps = int(round(seconds / dt))
    for _ in range(steps):
        s, ex = p.step(s, cmd, dt=dt, substeps=substeps)
    return s, ex


def body(s):
    psi = s[:, 2]
    c, sn = torch.cos(psi), torch.sin(psi)
    vx = c * s[:, 3] + sn * s[:, 4]
    vy = -sn * s[:, 3] + c * s[:, 4]
    return vx, vy, torch.atan2(-vy, vx)


def main():
    print("=== 1. straight line, throttle 0.5, t = 3 s")
    p = make(1)
    s = torch.zeros((1, 10))
    p.init_wheels(s)
    s, ex = run(p, s, torch.tensor([[0.0, 0.5, 0.0]]), 3.0)
    v = float(torch.hypot(s[0, 3], s[0, 4]))
    print("   v = %.2f m/s   (expect ~8-11 m/s for 384 kW at 1320 kg)" % v)
    print("   ax = %.2f m/s^2   rear slip = %.3f" % (float(ex["ax"]), float(ex["slip"][0, 2])))

    print("=== 2. constant steer 0.25 rad, throttle 0.35, t = 6 s -> should circle")
    p = make(1)
    s = torch.zeros((1, 10))
    p.init_wheels(s)
    s, ex = run(p, s, torch.tensor([[0.25, 0.35, 0.0]]), 6.0)
    vx, vy, beta = body(s)
    print("   v = %.2f m/s   yaw rate = %.2f rad/s   beta = %.3f rad"
          % (float(torch.hypot(s[0, 3], s[0, 4])), float(s[0, 5]), float(beta)))
    print("   implied radius = v/r = %.1f m" % (float(torch.hypot(s[0, 3], s[0, 4]) / max(abs(float(s[0, 5])), 1e-3))))

    print("=== 3. brake step from 12 m/s at brake 1.0, t = 1 s")
    p = make(1)
    s = torch.zeros((1, 10))
    s[0, 3] = 12.0
    p.init_wheels(s)
    s, ex = run(p, s, torch.tensor([[0.0, 0.0, 1.0]]), 1.0)
    print("   v after 1 s = %.2f m/s, ax = %.1f m/s^2  (expect a strong deceleration)"
          % (float(torch.hypot(s[0, 3], s[0, 4])), float(ex["ax"])))

    print("=== 4. Ackermann: left/right front angles at nominal 0.3 rad")
    p = make(3)
    for ack, label in ((0.0, "0 % (parallel)"), (1.0, "100 % exact"), (1.2, "120 %")):
        p.ackermann_scale = torch.full((3, 1), ack)
        d = torch.tensor([[0.3], [0.3], [0.3]])
        fs = p._front_steer(d)
        print("   %-16s left = %+.4f  right = %+.4f  (difference %+.4f)"
              % (label, float(fs[0, 0]), float(fs[0, 1]), float(fs[0, 0] - fs[0, 1])))
    p.ackermann_scale = torch.full((3, 1), 1.0)
    fs_neg = p._front_steer(torch.tensor([[-0.3], [-0.3], [-0.3]]))
    print("   symmetry: at -0.3 left = %+.4f right = %+.4f (should swap signs of the above)"
          % (float(fs_neg[0, 0]), float(fs_neg[0, 1])))
    fs0 = p._front_steer(torch.zeros((3, 1)))
    print("   at zero steer: left = %+.5f right = %+.5f (must be equal)" % (float(fs0[0, 0]), float(fs0[0, 1])))

    print("=== 5. Ackermann effect on the trajectory (0 %% vs 100 %%, same command)")
    for ack in (0.0, 1.0):
        p = make(1)
        p.ackermann_scale = torch.full((1, 1), ack)
        s = torch.zeros((1, 10))
        p.init_wheels(s)
        s, _ = run(p, s, torch.tensor([[0.30, 0.35, 0.0]]), 4.0)
        print("   ack %.0f : pos = (%.1f, %.1f)  psi = %+.2f  v = %.2f"
              % (ack, float(s[0, 0]), float(s[0, 1]), float(s[0, 2]),
                 float(torch.hypot(s[0, 3], s[0, 4]))))


if __name__ == "__main__":
    main()
