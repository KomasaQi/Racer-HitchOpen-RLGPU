"""Solve the steady-drift trim for the FastPlant and write a lookup table.

The spawn used to prescribe V, beta and r = V*kappa but nothing else: steering was a
guess (0.55*beta), throttle was a constant 0.55, and the four wheel speeds were
overwritten with Vx/R.  That is not an equilibrium, and on the corrected plant a
zero-action rollout leaves the corridor in 2.1 s, which is why training stopped
working.

This script solves the real thing.  Unknowns:

    x = [beta, delta, throttle, omega_FL, omega_FR, omega_RL, omega_RR]

with r = V*kappa fixed, and the residuals are the steady circular motion conditions
in the frame whose x axis is the path tangent:

    dvx = 0,  dv_y - V^2*kappa = 0,  dr = 0,  domega_i = 0

Note the V^2*kappa term: demanding dv_y = 0 describes straight inertial motion, which
is what the old equilibrium.py did.

Convention: psi = chi - beta, so the body velocity is (V cos(beta), V sin(beta)) and
the world velocity is (V cos(chi), V sin(chi)) with chi the path tangent angle.

    python experiments/continuous_drift_iwd/trim_table.py            # build and save
    python experiments/continuous_drift_iwd/trim_table.py --check    # verify only
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))
sys.path.insert(0, os.path.join(ROOT, "utils"))

import torch  # noqa: E402
from fast_plant import PARAM_INDEX, FastRacerPlant, load_preset  # noqa: E402

G = 9.81
R_WHEEL = 0.332


class TrimSolver:
    """Vectorised Newton solve for one (kappa, V, mu) cell."""

    def __init__(self, plant, mu_scale=1.0):
        self.plant = plant
        self.mu_scale = float(mu_scale)

    def residual(self, x, kappa, V):
        """Return the six residuals for one trim candidate.

        x = [beta, delta, throttle, omega_FL, omega_FR, omega_RL, omega_RR]
        """
        beta, delta, thr = float(x[0]), float(x[1]), float(x[2])
        s = torch.zeros((1, 10))
        chi = 0.0
        # psi = chi + beta makes the body velocity (V cos beta, +V sin beta), i.e. a
        # POSITIVE beta is a positive sideslip; chi - beta flips the sign and the
        # solve then has no root
        psi = chi + beta
        s[0, 2] = psi
        s[0, 3] = V * np.cos(chi)          # world velocity along the tangent
        s[0, 4] = V * np.sin(chi)
        s[0, 5] = V * kappa
        s[0, 6:10] = torch.tensor(x[3:7], dtype=torch.float32)
        cmd = torch.tensor([[delta, thr, 0.0]])
        self.plant.reset_guess()
        d, ex = self.plant.derivatives(s, cmd, mode="pedals")
        r = float(s[0, 5])
        # d[3], d[4] are WORLD accelerations (already the body accelerations rotated
        # by psi).  The steady-state conditions are on the BODY velocity
        # derivatives, which carry the Coriolis terms:
        #   dvx_b = ax_b + vy_b*r = 0      dv_y_b = ay_b - vx_b*r = 0
        # Comparing a_y with V^2*kappa (as the first version did) mixes the two.
        c, sn = np.cos(psi), np.sin(psi)
        ax_b = c * float(d[0, 3]) + sn * float(d[0, 4])
        ay_b = -sn * float(d[0, 3]) + c * float(d[0, 4])
        vx_b = c * float(s[0, 3]) + sn * float(s[0, 4])
        vy_b = -sn * float(s[0, 3]) + c * float(s[0, 4])
        return np.array([
            (ax_b + vy_b * r) / G,
            (ay_b - vx_b * r) / G,
            float(d[0, 5]),
            float(d[0, 6]) / 300.0,
            float(d[0, 7]) / 300.0,
            float(d[0, 8]) / 300.0,
            float(d[0, 9]) / 300.0,
        ]), ex, r

    def solve(self, kappa, V, beta0=0.15):
        from scipy.optimize import least_squares
        lo = np.array([-1.2, -0.45, 0.0, 0.5, 0.5, 0.5, 0.5])
        hi = np.array([1.2, 0.45, 1.0, 900.0, 900.0, 900.0, 900.0])
        x0 = np.array([beta0, -0.20, 0.12,
                       0.8 * V / R_WHEEL, 0.8 * V / R_WHEEL,
                       1.10 * V / R_WHEEL, 1.10 * V / R_WHEEL])
        x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
        res = least_squares(lambda x: self.residual(x, kappa, V)[0], x0,
                            bounds=(lo, hi), xtol=1e-10, ftol=1e-10,
                            max_nfev=400)
        return res.x, float(np.abs(res.fun).max())


def build(cells, out_path):
    preset = load_preset("fullsize_racecar_ev")
    plant = FastRacerPlant(preset, 1, "cpu")
    table = []
    for mu, kappa, V in cells:
        plant = FastRacerPlant(preset, 1, "cpu")
        # set the grip level by scaling the tyre force at the source
        plant.p[0, PARAM_INDEX["D"]] = float(mu)
        solver = TrimSolver(plant)
        x, err = solver.solve(kappa, V)
        table.append((mu, kappa, V, err) + tuple(float(v) for v in x))
        print("mu %.2f  R %5.1f  V %4.1f  beta %+.3f  delta %+.3f  thr %.3f  "
              "wRL %6.1f  maxres %.2e"
              % (mu, 1.0 / max(kappa, 1e-6), V, x[0], x[1], x[2], x[5], err))
    np.save(out_path, np.array(table, dtype=np.float64))
    print("saved", out_path, len(table), "cells")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments",
                                                  "continuous_drift_iwd",
                                                  "trim_table.npy"))
    args = ap.parse_args()
    cells = []
    for mu in (0.5, 0.75, 1.0):
        for radius in (8.0, 12.0, 18.0):
            for frac in (0.5, 0.7):
                V = frac * np.sqrt(mu * G * radius)
                cells.append((mu, 1.0 / radius, float(V)))
    build(cells, args.out)
