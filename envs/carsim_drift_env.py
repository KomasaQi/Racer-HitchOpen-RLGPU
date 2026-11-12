"""CarSim-backed continuous-drift environment.

Only the physics is swapped: ``ContinuousDriftIWDEnv`` keeps ownership of the
reference path, the tracking, the reward, the termination logic and both
observations.  A ``CarSimPlant`` adapter implements the plant interface that the
task environment calls (``step``, ``package_extended``, ``extended_state``,
``col``, ``geometry``, ``randomize``, ``init_wheels*``, ``reset_guess``), so the
whole task stack runs unchanged on top of the real solver.

Differences from the GPU plant that the caller must know:
* ``n`` is 1 -- one CarSim solver drives one vehicle;
* the car always starts from rest at the scenario origin, because that is what a
  real car can do; the reference path is normalised to start there too;
* no OU disturbance (it belongs to training, not to evaluation).
"""
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.carsim_flat_env import (CARSIM_WHEEL_SUFFIXES, CHANNELS,  # noqa: E402
                                  DEFAULT_SCENARIOS, IDX, MAX_BRAKE_MPA,
                                  MAX_STEER_SW_DEG, CarSimFlatEnv)
from envs.continuous_drift_iwd import ContinuousDriftIWDEnv  # noqa: E402

G = 9.81
R_EFF = 0.332
# ContinuousDriftIWDEnv does not command a normalised steering value: the action's
# first component is already a ROAD WHEEL angle in radians, bounded by the plant's
# max_steer (0.45 rad).  Treating it as [-1, 1] under-commands the steering by
# 1/0.45 = 2.22x, which is exactly the "runs wide" symptom seen in the first test.
MAX_ROAD_WHEEL_RAD = 0.45
# steering actuator model (per control period): first order lag and slew limit
STEER_TAU = 0.08          # s, roughly a 12 Hz bandwidth
STEER_RATE_LIMIT = 6.0    # rad of steering wheel per second


class CarSimPlant:
    """Plant interface backed by one CarSim solver."""

    def __init__(self, simfile, device="cuda:0", grip=1.0, n=1):
        self.device = torch.device(device)
        self.n = int(n)
        self.sim = CarSimFlatEnv.__new__(CarSimFlatEnv)   # no torch state needed
        CarSimFlatEnv.__init__(self.sim, simfile=simfile, device="cpu")
        self.grip = float(grip)
        self.dtype = torch.float32
        self._last = None
        self._rdot = 0.0
        self._r_prev = 0.0
        # constants the task environment asks for
        self._cols = {
            "lR": torch.full((self.n,), 1.195, device=self.device, dtype=self.dtype),
            "lF": torch.full((self.n,), 1.455, device=self.device, dtype=self.dtype),
            "g": torch.full((self.n,), G, device=self.device, dtype=self.dtype),
            "m": torch.full((self.n,), 1985.0, device=self.device, dtype=self.dtype),
            "D": torch.full((self.n,), self.grip, device=self.device, dtype=self.dtype),
        }
        self.tyre_scale = torch.ones((self.n, 1), device=self.device, dtype=self.dtype)
        self.ackermann_scale = torch.ones((self.n, 1), device=self.device, dtype=self.dtype)
        self.g = G
        self.steer_tau = float(os.environ.get("CARSIM_STEER_TAU", STEER_TAU))
        self.steer_rate_limit = float(os.environ.get("CARSIM_STEER_RATE", STEER_RATE_LIMIT))
        self._steer_applied = 0.0
        self._steer_prev = 0.0
        self.n_physics_ok = True

    # ------------------------------------------------------------- interface
    def col(self, name):
        if name in self._cols:
            return self._cols[name]
        raise KeyError("CarSimPlant has no parameter column %r" % name)

    def geometry(self):
        x = torch.tensor([1.455, 1.455, -1.195, -1.195], device=self.device)
        y = torch.tensor([0.8, -0.8, 0.8, -0.8], device=self.device)
        return x, y, torch.full((4,), R_EFF, device=self.device), torch.zeros(4, device=self.device)

    def randomize(self, mask=None, override=None, ranges=None):
        return None

    def derivatives(self, s, command, mode="pedals"):
        """Only used by the base env to refresh ``extras`` after a reset.

        CarSim has already produced its own channel set, so this returns the same
        dictionaries as ``step`` without advancing the solver.
        """
        raw = self.sim._raw
        extras = {
            "ax": torch.tensor([raw[IDX["Ax"]] * G], device=self.device),
            "ay": torch.tensor([raw[IDX["Ay"]] * G], device=self.device),
            "yaw_acc": torch.tensor([0.0], device=self.device),
            **self._wheel_terms(raw),
        }
        return s[:, :6] * 0.0, extras

    def init_wheels(self, s):
        return None

    def init_wheels_mask(self, s, mask):
        return None

    def reset_guess(self, mask=None):
        return None

    # ------------------------------------------------------------------ step
    def step(self, s, command, dt, substeps=1, mode="pedals"):
        cmd = command.detach().cpu().numpy().reshape(-1)
        steer_cmd = (float(np.clip(cmd[0] / MAX_ROAD_WHEEL_RAD, -1.0, 1.0))
                     * MAX_STEER_SW_DEG)
        # Steering actuator: a real wheel cannot jump between two 20 Hz targets.
        # Without this the policy chatters (bang-bang steering) because the plant
        # it was trained on answered a step instantly.
        if self.steer_tau > 0.0:
            a = float(np.clip(float(dt) / (self.steer_tau + float(dt)), 0.0, 1.0))
            self._steer_applied += a * (steer_cmd - self._steer_applied)
        else:
            self._steer_applied = steer_cmd
        if self.steer_rate_limit > 0.0:
            step = self.steer_rate_limit * float(dt)
            self._steer_applied = float(np.clip(
                self._steer_applied, self._steer_prev - step, self._steer_prev + step))
        self._steer_prev = self._steer_applied
        steer = self._steer_applied
        throttle = float(np.clip(cmd[1], 0.0, 1.0))
        brake = float(np.clip(cmd[2], 0.0, 1.0)) * MAX_BRAKE_MPA
        imp = [steer, throttle, brake]
        acc = []
        n_sub = max(int(substeps), 1)
        for _ in range(n_sub):
            status = 0
            for _ in range(20):                    # 20 x 0.5 ms = 0.01 s physics
                self.sim._t += 0.0005
                status, self.sim._exp = self.sim._vs.integrate_io(
                    self.sim._t, imp, self.sim._exp)
                if status != 0:
                    raise RuntimeError("CarSim integrate_io status %d" % status)
                raw_i = np.asarray(self.sim._exp, dtype=np.float64)
                acc.append((raw_i[IDX["Ax"]] * G, raw_i[IDX["Ay"]] * G))
        raw = np.asarray(self.sim._exp, dtype=np.float64)
        self.sim._raw = raw.copy()
        self.sim._update_state(raw)
        s.copy_(self.sim.s.to(s.device))
        ax = float(np.mean([a[0] for a in acc]))
        ay = float(np.mean([a[1] for a in acc]))
        r = float(s[0, 5])
        self._rdot = (r - self._r_prev) / max(float(dt), 1e-6)
        self._r_prev = r
        self._last = raw
        extras = {
            "ax": torch.tensor([ax], device=self.device),
            "ay": torch.tensor([ay], device=self.device),
            "yaw_acc": torch.tensor([self._rdot], device=self.device),
            **self._wheel_terms(raw),
        }
        return s, extras

    def _wheel_terms(self, raw):
        dev = self.device
        kappa = torch.tensor([raw[IDX["Kappa_" + k]]
                              for k in CARSIM_WHEEL_SUFFIXES],
                             device=dev, dtype=self.dtype)
        alpha = torch.tensor([raw[IDX["Alpha_" + k]]
                              for k in CARSIM_WHEEL_SUFFIXES],
                             device=dev, dtype=self.dtype) * (math.pi / 180.0)
        fz = torch.tensor([raw[IDX["Fz_WC_" + k]]
                           for k in CARSIM_WHEEL_SUFFIXES], device=dev)
        fx = torch.tensor([raw[IDX["Fx_WC_" + k]]
                           for k in CARSIM_WHEEL_SUFFIXES], device=dev)
        fy = torch.tensor([raw[IDX["Fy_WC_" + k]]
                           for k in CARSIM_WHEEL_SUFFIXES], device=dev)
        w = torch.tensor([raw[IDX["AVy_" + k]]
                          for k in CARSIM_WHEEL_SUFFIXES], device=dev)
        w = w * (2 * math.pi / 60.0)
        vx_w = (w * R_EFF).unsqueeze(0)
        nan4 = torch.zeros((1, 4), device=dev)
        return {
            "vx_wheel": vx_w, "vy_wheel": nan4,
            "sx": kappa.unsqueeze(0), "sy": torch.tan(alpha).unsqueeze(0),
            "slip": torch.hypot(kappa, torch.tan(alpha)).unsqueeze(0),
            "mu": torch.full((1, 4), self.grip, device=dev),
            "alpha": alpha.unsqueeze(0),
            "mux": torch.full((1, 4), self.grip, device=dev),
            "muy": torch.full((1, 4), self.grip, device=dev),
            "fz": fz.unsqueeze(0), "fx_body": fx.unsqueeze(0), "fy_Body": fy.unsqueeze(0),
            "fy_body": fy.unsqueeze(0),
        }

    # --------------------------------------------------------- extended state
    def package_extended(self, s, extras):
        return self._assemble(s, extras)

    def extended_state(self, s, command=None, mode="pedals"):
        raw = self.sim._raw
        extras = {
            "ax": torch.tensor([raw[IDX["Ax"]] * G], device=self.device),
            "ay": torch.tensor([raw[IDX["Ay"]] * G], device=self.device),
            "yaw_acc": torch.tensor([0.0], device=self.device),
            **self._wheel_terms(raw),
        }
        return self._assemble(s, extras)

    def _assemble(self, s, extras):
        psi = s[:, 2]
        vxw, vyw = s[:, 3], s[:, 4]
        c, sn = torch.cos(psi), torch.sin(psi)
        ax_w = c * extras["ax"] - sn * extras["ay"]
        ay_w = sn * extras["ax"] + c * extras["ay"]
        v = torch.hypot(vxw, vyw)
        beta = torch.atan2(vyw, vxw) - psi
        beta = torch.atan2(torch.sin(beta), torch.cos(beta))
        parts = [
            vxw.unsqueeze(1), vyw.unsqueeze(1), s[:, 5].unsqueeze(1),
            ax_w.unsqueeze(1), ay_w.unsqueeze(1), extras["yaw_acc"].unsqueeze(1),
            v.unsqueeze(1), beta.unsqueeze(1),
            extras["vx_wheel"], extras["vy_wheel"],
            extras["sx"], extras["sy"], extras["slip"], extras["mu"],
            extras["alpha"], extras["mux"], extras["muy"],
            extras["fz"], extras["fx_body"], extras["fy_body"],
        ]
        return torch.cat(parts, dim=1)


class CarSimDriftEnv(ContinuousDriftIWDEnv):
    """``ContinuousDriftIWDEnv`` with CarSim as the plant, n = 1."""

    def __init__(self, scenario="flat_mu050", simfile=None, device="cuda:0",
                 stall_window=20, stall_move=0.6, **kwargs):
        # CarSim releases the car FROM REST, while training spawned at 4.5 m/s.
        # The default stall rule (10 steps / 1.0 m) therefore fires during the
        # launch: in 0.5 s from standstill the car covers only about 0.5 m.
        kwargs.setdefault("stall_window", stall_window)
        kwargs.setdefault("stall_move", stall_move)
        if simfile is None:
            key = {"flat_mu050": 0.50, "flat_mu075": 0.75, "flat_mu100": 1.00}
            grip = key.get(str(scenario), 1.0)
            simfile = DEFAULT_SCENARIOS[grip]
        else:
            grip = 1.0
        super().__init__("fullsize_racecar_ev", 1, device, ref_mode="hybrid",
                         control_mode="pedals", train=True, **kwargs)
        self.car = CarSimPlant(simfile, device=device, grip=grip, n=1)
        self.plant_name = "fast"
        self.plant = self.car
        # the base class cached the GPU plant's radii before the swap
        self.wheel_radius = self.car.geometry()[2]
        self.grip = grip
        self.scenario = scenario
        self._normalise_paths()
        self._startup_speed = 2.0
        self._launched = False

    def _normalise_paths(self):
        """Move every reference road so it starts at the origin heading +x.

        CarSim always releases the car at the scenario origin pointing along +x, so
        the reference has to start there too -- otherwise the first tracking error
        would be the distance between the two frames.
        """
        bank = self.path_bank[self.current_generation]      # [paths, W, 4]
        x0 = bank[:, 0, 0:1]
        y0 = bank[:, 0, 1:2]
        d0 = bank[:, 0, 2:3]
        c, s = torch.cos(-d0), torch.sin(-d0)
        dx = bank[:, :, 0] - x0
        dy = bank[:, :, 1] - y0
        bank[:, :, 0] = c * dx - s * dy
        bank[:, :, 1] = s * dx + c * dy
        bank[:, :, 2] = bank[:, :, 2] - d0

    def reset(self):
        super().reset()
        # the car is at rest at the origin; put the task bookkeeping there too
        self.progress.zero_()
        self.step_progress.zero_()
        self.travel.zero_()
        self.no_progress_count.zero_()
        self.is_done.zero_()
        self.u_cmd.zero_()
        self._stall_anchor.copy_(self.s[:, :2])
        self.car.sim.reset()
        self.s.copy_(self.car.sim.s.to(self.s.device))
        self.car._steer_applied = 0.0
        self.car._steer_prev = 0.0
        self.car._r_prev = float(self.s[0, 5])
        self.car._rdot = 0.0
        _, self.extras = self.car.derivatives(
            self.s, self.u_cmd, mode=self.control_mode)
        self.es = self.car.package_extended(self.s, self.extras)
        self._track_pending = True
        self._now = None
        self._prev = None
        return self.obs()

    def close(self):
        self.car.sim.close()
