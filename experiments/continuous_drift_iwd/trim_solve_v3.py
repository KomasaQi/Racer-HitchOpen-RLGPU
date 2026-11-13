"""Float64 seven-equation steady drift trim solver for the production plant.

Unknowns are ``[beta, delta, throttle, kappa_FL, kappa_FR, kappa_RL,
kappa_RR]``.  Unlike the older solvers, throttle is not eliminated by assuming
an affine pedal-to-wheel-torque map, and finite differences are not lost in a
float32 plant.
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))
from fast_plant import FastRacerPlant, PARAM_INDEX, load_preset  # noqa: E402


class TrimV3:
    def __init__(self, effective_mu):
        self.plant = FastRacerPlant(load_preset("fullsize_racecar_ev"), 1, "cpu",
                                    dtype=torch.float64)
        self.plant.p[0, PARAM_INDEX["D"]] = float(effective_mu)
        self.g = float(self.plant.col("g"))
        self.m = float(self.plant.col("m"))
        self.iz = float(self.plant.col("Iz"))
        self.length = float(self.plant.col("lF") + self.plant.col("lR"))
        self.torque_scale = self.m * self.g * float(self.plant.col("rR")) / 4.0

    def state(self, z, curvature, speed):
        beta, delta = float(z[0]), float(z[1])
        s = torch.zeros((1, 10), dtype=torch.float64)
        s[0, 2] = -beta
        s[0, 3] = speed
        s[0, 5] = speed * curvature
        # Contact longitudinal velocities depend on body state and front steer,
        # but not wheel angular speed.  Evaluate once at the candidate delta,
        # then install the requested positive-power kappa convention.
        cmd = torch.tensor([[delta, 0.0, 0.0]], dtype=torch.float64)
        self.plant.reset_guess()
        _, ex = self.plant.derivatives(s, cmd, mode="pedals")
        vx = ex["vx_wheel"][0]
        rad = ex["wheel_radius"][0]
        floor = float(self.plant.col("slip_v_floor"))
        kx = torch.as_tensor(z[3:7], dtype=torch.float64)
        denom = vx.abs().clamp_min(floor)
        s[0, 6:10] = (vx + kx * denom) / rad
        return s

    def residual(self, z, curvature, speed):
        s = self.state(z, curvature, speed)
        cmd = torch.tensor([[float(z[1]), float(z[2]), 0.0]], dtype=torch.float64)
        self.plant.reset_guess()
        self.plant.derivatives(s, cmd, mode="pedals")
        d, ex = self.plant.derivatives(s, cmd, mode="pedals")
        psi = float(s[0, 2])
        c, sn = np.cos(psi), np.sin(psi)
        ax_b = c * float(d[0, 3]) + sn * float(d[0, 4])
        ay_b = -sn * float(d[0, 3]) + c * float(d[0, 4])
        vx_b = c * float(s[0, 3]) + sn * float(s[0, 4])
        vy_b = -sn * float(s[0, 3]) + c * float(s[0, 4])
        yaw_rate = float(s[0, 5])
        net = ex["wheel_net_torque"][0].detach().numpy()
        return np.r_[
            (ax_b + yaw_rate * vy_b) / self.g,
            (ay_b - yaw_rate * vx_b) / self.g,
            float(d[0, 5]) * self.iz / (self.m * self.g * self.length),
            net / self.torque_scale,
        ]

    def solve(self, curvature, speed):
        lo = np.array([-0.85, -0.449, 0.0, -0.15, -0.15, -0.10, -0.10])
        hi = np.array([0.85, 0.449, 1.0, 0.15, 0.15, 2.00, 2.00])
        sign = 1.0 if curvature >= 0 else -1.0
        starts = []
        for beta in (0.0, -0.20 * sign, -0.40 * sign, -0.60 * sign,
                     0.20 * sign, 0.40 * sign):
            for delta in (-0.30 * sign, -0.15 * sign, 0.10 * sign, 0.30 * sign):
                for rear_kappa in (0.08, 0.35, 0.75):
                    starts.append([beta, delta, 0.55, 0.0, 0.0,
                                   rear_kappa, rear_kappa])
        candidates = []
        scale = np.array([0.35, 0.25, 0.35, 0.08, 0.08, 0.40, 0.40])
        for z0 in starts:
            result = least_squares(
                lambda z: self.residual(z, curvature, speed),
                np.clip(z0, lo + 1e-7, hi - 1e-7), bounds=(lo, hi),
                x_scale=scale, diff_step=2e-5, max_nfev=500,
                ftol=1e-10, xtol=1e-10, gtol=1e-10)
            err = float(np.abs(result.fun).max())
            candidates.append((err, result.cost, result.x))
        feasible = [c for c in candidates if c[0] < 1e-3]
        if feasible:
            # Prefer a powered large-angle countersteer branch.
            feasible.sort(key=lambda c: (-(abs(c[2][0])), -np.mean(c[2][5:7])))
            return feasible[0], candidates
        candidates.sort(key=lambda c: (c[0], c[1]))
        return candidates[0], candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, required=True,
                    help="effective peak friction D (2.1246 * road multiplier)")
    ap.add_argument("--curvature", type=float, default=0.1)
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--limit-frac", type=float, default=0.9)
    args = ap.parse_args()
    speed = args.speed if args.speed is not None else args.limit_frac * np.sqrt(
        args.mu * 9.81 / abs(args.curvature))
    best, all_candidates = TrimV3(args.mu).solve(args.curvature, speed)
    err, cost, z = best
    print("mu %.6f curvature %.5f speed %.5f" % (args.mu, args.curvature, speed))
    print("best max_abs_res %.6e cost %.6e" % (err, cost))
    print("beta delta throttle kappa_FL kappa_FR kappa_RL kappa_RR")
    print(" ".join("%+.8f" % x for x in z))
    print("feasible", sum(c[0] < 1e-3 for c in all_candidates), "/", len(all_candidates))
    feasible = [c for c in all_candidates if c[0] < 1e-3]
    strict_branches = [c for c in feasible
                       if c[2][0] * args.curvature < 0.0
                       and c[2][1] * args.curvature < 0.0
                       and c[2][2] >= 0.40
                       and np.mean(c[2][5:7]) >= 0.30]
    print("strict-sign/power branches", len(strict_branches))
    show = strict_branches if strict_branches else feasible
    show = sorted(show, key=lambda c: (-abs(c[2][0]), -np.mean(c[2][5:7]), c[0]))
    for e, c, zz in show[:10]:
        print("branch err %.2e beta %+.4f delta %+.4f thr %.4f rear_k %.4f/%.4f" %
              (e, zz[0], zz[1], zz[2], zz[5], zz[6]))


if __name__ == "__main__":
    main()
