"""Solve the steady state drift equilibrium of the FastPlant on a circle.

For a given speed ``v`` and path curvature ``kappa`` find the sideslip, steering,
throttle and wheel speeds such that every plant derivative is zero and the car
travels on a circle of radius ``1/kappa`` with a constant sideslip.

Unknowns : [beta, delta, q, r, w_fl, w_fr, w_rl, w_rr]
Equations: dvx, dvy, dr, dw_fl, dw_fr, dw_rl, dw_rr, (r - v*kappa)

    python equilibrium.py [preset] [radius]
"""
import sys
import os

import numpy as np
import torch
from scipy.optimize import least_squares

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))

from fast_plant import FastRacerPlant, preset_parameters

preset = sys.argv[1] if len(sys.argv) > 1 else "fullsize_racecar_ev"
R = float(sys.argv[2]) if len(sys.argv) > 2 else 7.5714
plant = FastRacerPlant(preset_parameters(preset), 1, device="cpu")
kappa = 1.0 / R

LO = [-1.7, -0.45, 0.00, 0.05, 0.5, 0.5, 0.5, 0.5]
HI = [-0.05, 0.45, 1.00, 3.00, 600.0, 600.0, 600.0, 600.0]


def residual(x, v):
    beta, delta, q, r, wfl, wfr, wrl, wrr = x
    s = torch.zeros((1, 10))
    s[0, 2] = -beta                       # world velocity direction = beta + psi = 0
    s[0, 3] = v * np.cos(beta)
    s[0, 4] = v * np.sin(beta)
    s[0, 5] = r
    s[0, 6] = wfl
    s[0, 7] = wfr
    s[0, 8] = wrl
    s[0, 9] = wrr
    d, ex = plant.derivatives(s, torch.tensor([[delta, q, 0.0]]))
    # normalise the wheel spin equations (rad/s^2, O(1e3)) so the least squares
    # problem is not dominated by them
    return np.array([
        float(d[0, 3]), float(d[0, 4]), float(d[0, 5]),
        float(d[0, 6]) / 300.0, float(d[0, 7]) / 300.0,
        float(d[0, 8]) / 300.0, float(d[0, 9]) / 300.0,
        (r - v * kappa) * 10.0,
    ])


print("preset=%s  R=%.3f m" % (preset, R))
for v in (4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0):
    best = None
    for beta0 in (-0.3, -0.5, -0.7, -0.9, -1.1):
        for spin in (1.0, 1.3, 1.7, 2.2):
            x0 = [beta0, -0.36, 0.55, v * kappa,
                  v / 0.32355, v / 0.32355, spin * v / 0.3328, spin * v / 0.3328]
            try:
                sol = least_squares(residual, np.array(x0), args=(v,), bounds=(LO, HI),
                                    xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=3000)
            except Exception:
                continue
            res = float(np.abs(sol.fun).max())
            if best is None or res < best[0]:
                best = (res, sol.x)
    if best is None:
        print("v=%.1f no solution" % v)
        continue
    res, x = best
    beta, delta, q, r, wfl, wfr, wrl, wrr = x
    slip = (wrl * 0.3328 - v) / max(v, 1e-3)
    print("v=%.1f | resid=%.1e | beta=%+0.3f delta=%+0.3f q=%.3f r=%+0.3f "
          "rear_surface=%.2f m/s (slip %+.2f) steer@limit=%.2f" % (
              v, res, beta, delta, q, r, wrl * 0.3328, slip, abs(delta) / 0.45))
