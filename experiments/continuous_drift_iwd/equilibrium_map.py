"""Feasibility map of a steady drift on a circle for the FastPlant.

For a grid of (speed, sideslip) solve for the steering, throttle, yaw rate and
wheel speeds that make every plant derivative vanish while the car travels on a
circle of radius 1/kappa with a constant sideslip.  A converged solution means
the drift state is physically sustainable at that speed and sideslip.

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
G = 9.81
print("preset=%s  R=%.3f m  kappa=%.4f" % (preset, R, kappa))


def residual(x, v, beta):
    delta, q, r, wfl, wfr, wrl, wrr = x
    s = torch.zeros((1, 10))
    # Velocity along +x (path tangent) and psi = -beta, so the BODY sideslip
    # beta = veldir - psi is exactly the requested value.
    s[0, 2] = -beta
    s[0, 3] = v
    s[0, 4] = 0.0
    s[0, 5] = r
    s[0, 6] = wfl
    s[0, 7] = wfr
    s[0, 8] = wrl
    s[0, 9] = wrr
    d, ex = plant.derivatives(s, torch.tensor([[delta, q, 0.0]]))
    return np.array([
        float(d[0, 3]) / G,
        float(d[0, 4]) / G,
        float(d[0, 5]),
        float(d[0, 6]) / 500.0, float(d[0, 7]) / 500.0,
        float(d[0, 8]) / 500.0, float(d[0, 9]) / 500.0,
        (r - v * kappa) * 10.0,
    ])


LO = np.array([-0.45, 0.00, 0.02, 0.5, 0.5, 0.5, 0.5])
HI = np.array([0.45, 1.00, 3.00, 600, 600, 600, 600])

print("  v    beta | residual  delta     q    r     rear_surface  feasible")
for v in (4.0, 5.0, 6.0, 7.0):
    row = []
    for beta in (-0.30, -0.45, -0.60, -0.75, -0.90):
        best = None
        for d0 in (-0.15, -0.40):
            for q0 in (0.3, 0.7):
                for spin in (1.0, 1.6):
                    x0 = np.array([d0, q0, v * kappa,
                                   v / 0.32355, v / 0.32355,
                                   spin * v / 0.3328, spin * v / 0.3328])
                    x0 = np.clip(x0, LO + 1e-6, HI - 1e-6)
                    sol = least_squares(residual, x0, args=(v, beta), bounds=(LO, HI),
                                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=4000)
                    res = float(np.abs(sol.fun).max())
                    if best is None or res < best[0]:
                        best = (res, sol.x)
        res, x = best
        if res < 2e-3:
            row.append("%+0.2f: OK (d %+0.2f q %.2f)" % (beta, x[0], x[1]))
        else:
            row.append("%+0.2f: -- (res %.0e)" % (beta, res))
    print("v=%.1f m/s | %s" % (v, " | ".join(row)))
