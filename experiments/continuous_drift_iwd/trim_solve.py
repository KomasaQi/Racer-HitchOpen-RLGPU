"""Steady-drift trim solver (v2), following TRIM_SPEC.md.

    z = [beta, delta, kx_FL, kx_FR, kx_RL, kx_RR]
    R = [ dvxb/g, dvyb/g, Mz/(m g L), N_FL/Ts, N_FR/Ts, (N_RL-N_RR)/(2 Ts) ]

psi = -beta so a positive beta is a positive sideslip (v1 used +beta and inverted it).
Throttle is eliminated: the plant is affine in the pedal, so N_RL + N_RR = 0 is solved
by linear interpolation between the t=0 and t=1 evaluations.
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))
sys.path.insert(0, os.path.join(ROOT, "utils"))

import fast_plant  # noqa: E402
from fast_plant import FastRacerPlant, load_preset  # noqa: E402

G = 9.81
R_WHEEL = 0.332
DT = torch.float64


class Trim:
    def __init__(self, preset, mu):
        p = dict(preset)
        self.plant = FastRacerPlant(p, 1, "cpu", dtype=DT)
        self.d_idx = fast_plant.TENSOR_KEYS.index("D")
        self.plant.p[0, self.d_idx] = float(mu)

    def _eval(self, s, delta, thr):
        cmd = torch.tensor([[delta, thr, 0.0]], dtype=DT)
        self.plant.reset_guess()
        self.plant.derivatives(s, cmd, mode="pedals")   # settle the load transfer
        return self.plant.derivatives(s, cmd, mode="pedals")

    def _state(self, beta, kappa, V, kx):
        s = torch.zeros((1, 10), dtype=DT)
        s[0, 2] = -beta            # chi = 0
        s[0, 3] = V
        s[0, 4] = 0.0
        s[0, 5] = V * kappa
        _, ex = self._eval(s, 0.0, 0.0)
        u = ex["vx_wheel"][0].detach().numpy()
        rad = ex["wheel_radius"][0].detach().numpy()
        floor = float(self.plant.col("slip_v_floor").max())
        for i in range(4):
            D = max(abs(float(u[i])), floor)
            s[0, 6 + i] = (float(u[i]) + float(kx[i]) * D) / float(rad[i])
        return s

    def _thr(self, s, delta):
        _, e0 = self._eval(s, delta, 0.0)
        _, e1 = self._eval(s, delta, 1.0)
        s0 = float(e0["wheel_net_torque"][0, 2] + e0["wheel_net_torque"][0, 3])
        s1 = float(e1["wheel_net_torque"][0, 2] + e1["wheel_net_torque"][0, 3])
        return None if abs(s1 - s0) < 1e-9 else -s0 / (s1 - s0)

    def residual(self, z, kappa, V):
        s = self._state(float(z[0]), kappa, V, z[2:6])
        thr = self._thr(s, float(z[1]))
        # Out-of-range throttle must REJECT the point, not be clipped.  Clipping evaluates
        # the residual with a throttle that does not satisfy the rear torque-sum equation,
        # which silently drops that equation and admits false low residuals -- the review
        # flagged this.  A drift state needs the rear axle spinning under power, so if the
        # required throttle leaves [0, 1] the state is not attainable and must be penalised.
        if thr is None or not (0.0 <= thr <= 1.0):
            return np.full(6, 1e3)
        if thr is None:
            return np.full(6, 1e3)
        d, ex = self._eval(s, float(z[1]), float(np.clip(thr, 0.0, 1.0)))
        m = float(self.plant.col("m"))
        L = float(self.plant.col("lF") + self.plant.col("lR"))
        Iz = float(self.plant.col("Iz"))
        Ts = m * G * R_WHEEL / 4.0
        psi = float(s[0, 2])
        c, sn = np.cos(psi), np.sin(psi)
        ax_b = c * float(d[0, 3]) + sn * float(d[0, 4])
        ay_b = -sn * float(d[0, 3]) + c * float(d[0, 4])
        vx_b = c * float(s[0, 3]) + sn * float(s[0, 4])
        vy_b = -sn * float(s[0, 3]) + c * float(s[0, 4])
        r = float(s[0, 5])
        net = ex["wheel_net_torque"][0].detach().numpy()
        return np.array([(ax_b + r * vy_b) / G,
                         (ay_b - r * vx_b) / G,
                         float(d[0, 5]) * Iz / (m * G * L),
                         net[0] / Ts, net[1] / Ts,
                         (net[2] - net[3]) / (2.0 * Ts)])

    def solve(self, kappa, V):
        from scipy.optimize import least_squares
        sgn = 1.0 if kappa >= 0 else -1.0
        L = float(self.plant.col("lF") + self.plant.col("lR"))
        # Rear slip bound raised from 0.35: the drift reward asks for 0.55, and a solver
        # that cannot even search there cannot decide whether the target is feasible.
        lo = np.array([-0.65, -0.449, -0.08, -0.08, -0.02, -0.02])
        hi = np.array([0.65, 0.449, 0.05, 0.05, 1.20, 1.20])
        # The trim has more than one branch.  Normal cornering has beta and delta of
        # the SAME sign; a drift has beta large and delta OPPOSITE (countersteer).
        # Picking the lowest residual always lands on the normal branch, so converge
        # from countersteer starts and keep the solution with the LARGEST |beta| among
        # those that meet the residual tolerance.
        zs = [[0.00, sgn * np.arctan(L * abs(kappa)), -0.003, -0.003, 0.01, 0.01]]
        for b0 in (0.15, 0.25, 0.35, 0.45):
            for d0 in (-0.10, -0.20, -0.30):
                zs.append([b0 * sgn, d0 * sgn, -0.004, -0.004, 0.06, 0.06])
        cands = []
        for z0 in zs:
            try:
                r = least_squares(lambda z: self.residual(z, kappa, V),
                                  np.clip(z0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi),
                                  method="trf",
                                  x_scale=np.array([0.3, 0.2, 0.03, 0.03, 0.1, 0.1]),
                                  ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=300)
            except Exception:
                continue
            e = float(np.abs(r.fun).max())
            cands.append((e, r.x))
        if not cands:
            return None, np.inf
        ok = [c for c in cands if c[0] < 1e-3]
        if ok:
            # drift branch: the countersteer solution with the largest sideslip
            drift = [c for c in ok if c[1][0] * c[1][1] < 0.0]
            pool = drift if drift else ok
            pool.sort(key=lambda c: -abs(c[1][0]))
            return pool[0][1], pool[0][0]
        cands.sort(key=lambda c: c[0])
        return cands[0][1], cands[0][0]


def main():
    preset = load_preset("fullsize_racecar_ev")
    print("%-5s %-6s %-5s %8s %8s %8s %9s %5s" %
          ("mu", "R", "V", "beta", "delta", "thr", "|res|", "ok"))
    rows = []
    for mu in (0.5, 0.75, 1.0):
        cells = [(1.0 / r, f * np.sqrt(mu * G * r)) for r in (18.0, 12.0, 8.0)
                 for f in (0.4, 0.55, 0.7)]
        cells.sort(key=lambda c: c[1] ** 2 * abs(c[0]) / (mu * G))
        tr = Trim(preset, mu)
        for kappa, V in cells:
            z, e = tr.solve(kappa, V)
            if z is None:
                continue
            s = tr._state(float(z[0]), kappa, V, z[2:6])
            thr = tr._thr(s, float(z[1]))
            ok = bool(e < 1e-3 and thr is not None and 0.0 <= thr <= 1.0)
            print("%-5.2f %-6.1f %-5.2f %8.3f %8.3f %8.3f %9.2e %5s" %
                  (mu, 1 / kappa, V, z[0], z[1], thr if thr is not None else -1, e, ok))
            rows.append((mu, kappa, V, float(ok)) + tuple(float(v) for v in z)
                        + (float(thr) if thr is not None else -1.0,))
    out = os.path.join(ROOT, "experiments", "continuous_drift_iwd", "trim_table.npy")
    np.save(out, np.array(rows, dtype=np.float64))
    print("saved", out, len(rows), "feasible", sum(1 for r in rows if r[3]))


if __name__ == "__main__":
    main()
