# -*- coding: utf-8 -*-
"""FastPlant: GPU-vectorized planar full-size RWD EV plant.

The plant keeps XCAR's isotropic combined-slip Pacejka tire (a single
``mu(slip)`` curve for the whole contact patch, see ``IWDCarDynamics``) and adds
the intermediate vehicle modules that the reference lab page
(``理解Pacejka模型/xcar-tire-lab_aero.html``, sections 07-08) uses:

* simplified electric motor: speed dependent torque envelope + power limit;
* viscous limited-slip differential on the driven axle;
* driven wheel rotational dynamics (wheel inertia, tire reaction torque, brake
  torque, rolling resistance);
* longitudinal *and* lateral quasi-static load transfer with an aero coupling;
* simplified longitudinal/lateral aerodynamics acting at one pressure center
  (drag, side force, lift/downforce and the resulting yaw moment).

The plant is deliberately *not* a full Magic-Formula 6.2 implementation.  It is
a "fast plant" for large scale GPU RL training, whose interface (steer /
throttle / brake in, 10 states out) matches the CarSim vehicle.

State (10)::

    [x, y, psi, vx_w, vy_w, r, omega_fl, omega_fr, omega_rl, omega_rr]

``vx_w``/``vy_w`` are the *world frame* velocity components (same convention as
the original XCAR environment, so ``state[:, :6]`` keeps its meaning).  Because
the body force projection rotates with the heading, the world-frame velocity
derivative is exactly ``R(psi) @ [Fx/m, Fy/m]`` -- the centripetal terms of the
body equations are already contained in that rotation.

Wheel order everywhere: ``[FL, FR, RL, RR]`` (front-left, front-right, rear-left,
rear-right), matching ``IWDCarDynamics``.  ``y`` points to the left, ``x``
forward, positive yaw is a left turn, positive ``ay`` is a left turn, positive
``ax`` is forward acceleration.

Numerical scheme
----------------
The 6 body DOF use explicit Euler at the physics step (100 Hz by default).  The
four wheel speeds use a linearized (semi-implicit) update
``omega <- omega + h * tau / (J + h * R * dFx/domega)`` because the tire
longitudinal stiffness makes the wheel-spin mode much stiffer than the body
modes at low speed; plain explicit Euler is unstable there.
"""

from __future__ import annotations

import math
import os
import copy

import torch
import yaml

# ---------------------------------------------------------------------------
# Preset defaults
# ---------------------------------------------------------------------------

G = 9.81

#: 384 kW / 2400 Nm direct-drive motor map of the workspace EV conversion
#: (``改造电车车辆/CarSim_motor_384kW_direct_drive.csv``).
EV_MOTOR_384KW = [
    [0.0, 2400.0], [250.0, 2400.0], [500.0, 2400.0], [750.0, 2400.0],
    [1000.0, 2400.0], [1250.0, 2400.0], [1500.0, 2400.0], [1600.0, 2293.62],
    [1800.0, 2038.77], [2000.0, 1834.89], [2500.0, 1467.91], [3000.0, 1223.26],
    [3500.0, 1048.51], [4000.0, 917.45], [4500.0, 815.51], [5000.0, 733.96],
    [5500.0, 667.23], [6000.0, 611.63], [6500.0, 564.58], [7000.0, 524.26],
    [7500.0, 489.30], [8000.0, 458.72], [8500.0, 431.74], [9000.0, 407.75],
    [9200.0, 398.89],
]

#: Keys every preset is expected to define.  Anything missing falls back to
#: ``DEFAULT_PARAMS`` (then partly to mass-scaled values, see ``resolve_params``).
DEFAULT_PARAMS = {
    # --- body ---
    "lF": 1.20, "lR": 1.20, "T": 1.60, "Tf": None, "Tr": None,
    "m": 1300.0, "h": 0.40, "g": G, "Iz": 1300.0,
    # --- wheels ---
    "rF": 0.32, "rR": 0.32, "IwF": 1.8, "IwR": 2.0,
    # --- tire: XCAR isotropic combined slip (B, C, D, E) ---
    "B": 10.0, "C": 1.90, "D": 1.40, "E": 0.97,
    "tyre_scale_front": 1.0, "tyre_scale_rear": 1.0,
    "slip_v_floor": 1.0,          # m/s, denominator floor of the slip ratio
    "slip_alpha_v_floor": 1.0,    # m/s, denominator floor of the slip angle
    "slip_max": 5.0,              # evaluated slip clamp (post-peak force plateau)
    "load_sensitivity": 0.0,      # 0 = pure XCAR (no tire load sensitivity)
    # --- lateral load transfer ---
    "roll_share_front": 0.55,     # lambda_f: share of lateral transfer on front
    # --- aerodynamics (one pressure center, x forward / y left / z up) ---
    "rho_air": 1.206,
    "aero_Ax": 2.20, "aero_Ay": 6.50,
    "aero_Cdx": 0.36, "aero_Cdy": 1.00,
    "aero_Clx": -0.77, "aero_Cly": 0.0,
    "aero_xp": 0.0, "aero_zp": 0.0,
    "wind_x": 0.0, "wind_y": 0.0,
    # --- drivetrain ---
    "drivetrain": "rwd",          # "rwd" | "fwd" | "awd" | "iwd"
    "gear_ratio": 3.44,
    "drive_efficiency": 0.92,
    "max_power_w": 384000.0,
    "motor": EV_MOTOR_384KW,
    "coast_torque_nm": 0.0,       # axle total, scaled by (1 - throttle)
    "wheel_speed_kp": 300.0,      # legacy wheel-speed command adapter gain
    # --- limited slip differential ---
    "lsd_kd": 20.0,               # N m s/rad, viscous coupling
    "lsd_tau_max": 300.0,         # N m, per side coupling cap
    # --- brakes ---
    "brake_pressure_max_mpa": 6.0,
    "brake_torque_front_nm_per_mpa": 400.0,
    "brake_torque_rear_nm_per_mpa": 175.0,
    # --- misc ---
    "rolling_resistance": 0.012,
    "max_steer_rad": 0.45,
    "load_iters": 2,              # load transfer passes per derivative (>=2)
    # Trapezoidal weight of the semi-implicit wheel update.  1.0 = backward
    # Euler (most damping, first order), 0.5 = trapezoidal (less bias while
    # accelerating/braking, still A-stable).
    "wheel_implicit_weight": 0.5,
}

#: tensor parameter order of ``FastRacerPlant.p`` (per environment row)
TENSOR_KEYS = (
    "lF", "lR", "Tf", "Tr", "m", "h", "g", "Iz",
    "rF", "rR", "IwF", "IwR",
    "B", "C", "D", "E", "tyre_scale_front", "tyre_scale_rear",
    "slip_v_floor", "slip_alpha_v_floor", "load_sensitivity",
    "roll_share_front",
    "rho_air", "aero_Ax", "aero_Ay", "aero_Cdx", "aero_Cdy",
    "aero_Clx", "aero_Cly", "aero_xp", "aero_zp", "wind_x", "wind_y",
    "gear_ratio", "drive_efficiency", "max_power_w", "coast_torque_nm",
    "wheel_speed_kp", "lsd_kd", "lsd_tau_max",
    "brake_pressure_max_mpa", "brake_torque_front_nm_per_mpa",
    "brake_torque_rear_nm_per_mpa",
    "rolling_resistance", "max_steer_rad",
)
PARAM_INDEX = {k: i for i, k in enumerate(TENSOR_KEYS)}

NUM_STATES = 10
NUM_ACTIONS = 3          # pedals mode: steer, throttle, brake
NUM_ACTIONS_LEGACY = 5   # wheel-speed mode: steer + 4 wheel surface speeds


def resolve_params(raw: dict) -> dict:
    """Fill a preset parameter dict with defaults and derived geometry."""
    p = dict(DEFAULT_PARAMS)
    p.update({k: v for k, v in raw.items() if v is not None})
    p["Tf"] = float(raw.get("Tf", raw.get("T", p["T"])))
    p["Tr"] = float(raw.get("Tr", raw.get("T", p["T"])))
    p["T"] = 0.5 * (p["Tf"] + p["Tr"])
    for key in ("motor",):
        if p.get(key) is not None:
            p[key] = [[float(a), float(b)] for a, b in p[key]]
    p["motor"] = p["motor"] if p.get("motor") else copy.deepcopy(EV_MOTOR_384KW)
    p["drivetrain"] = str(p.get("drivetrain", "rwd")).lower()
    return p


def load_preset(preset_name: str, presets_path: str | None = None) -> dict:
    """Load one preset from ``presets.yaml`` next to this module."""
    if presets_path is None:
        presets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets.yaml")
    with open(presets_path, encoding="utf-8") as f:
        presets = yaml.safe_load(f)
    if preset_name not in presets:
        raise KeyError(f"unknown car preset {preset_name!r}; available: {sorted(presets)}")
    entry = presets[preset_name]
    params = resolve_params(entry.get("parameters", {}))
    params["_name"] = preset_name
    params["_visualization"] = dict(entry.get("visualization", {}))
    return params


def preset_parameters(preset_name: str, presets_path: str | None = None) -> dict:
    """Return only the resolved numeric/other parameters (no metadata keys)."""
    p = load_preset(preset_name, presets_path)
    return {k: v for k, v in p.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Plant
# ---------------------------------------------------------------------------

class FastRacerPlant:
    """Planar 7-DOF (body 3 + 4 wheel spin) RWD-EV plant, vectorized over n envs."""

    def __init__(self, params: dict, n: int, device="cuda:0", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.n = int(n)
        self.g = float(params.get("g", G))
        self.drivetrain = str(params.get("drivetrain", "rwd")).lower()
        self.load_iters = max(int(params.get("load_iters", 2)), 1)
        self.slip_max = float(params.get("slip_max", 5.0))
        self.wheel_implicit_weight = float(params.get("wheel_implicit_weight", 0.5))

        # per-env float parameters, shape (n, len(TENSOR_KEYS))
        values = []
        for key in TENSOR_KEYS:
            v = params.get(key, DEFAULT_PARAMS.get(key))
            if key in ("drive_efficiency",) and v is None:
                v = 1.0
            values.append(float(v) if not isinstance(v, (list, tuple)) else 0.0)
        p0 = torch.tensor(values, device=self.device, dtype=self.dtype)
        self.p = p0.unsqueeze(0).repeat(self.n, 1).clone()
        self.p_default = p0.clone()

        # driven wheel mask (fixed per preset): [FL, FR, RL, RR]
        drive = torch.zeros(4, device=self.device, dtype=self.dtype)
        if self.drivetrain in ("rwd", "awd", "iwd"):
            drive[2:] = 1.0
        if self.drivetrain in ("fwd", "awd", "iwd"):
            drive[:2] = 1.0
        self.drive_mask = drive
        self.is_rwd = self.drivetrain == "rwd"

        # motor torque envelope (shared across envs; per-env scaling via max_power_w)
        motor = torch.as_tensor(params.get("motor", EV_MOTOR_384KW), device=self.device, dtype=self.dtype)
        order = torch.argsort(motor[:, 0])
        motor = motor[order]
        self.motor_rpm = motor[:, 0].contiguous()
        self.motor_torque = motor[:, 1].contiguous()
        self.motor_rpm_max = float(self.motor_rpm[-1].item())

        # cached constant geometry tensors
        self.corner_x = torch.tensor([1.0, 1.0, -1.0, -1.0], device=self.device, dtype=self.dtype)
        self.corner_y_sign = torch.tensor([1.0, -1.0, 1.0, -1.0], device=self.device, dtype=self.dtype)
        self.wheel_is_front = torch.tensor([1.0, 1.0, 0.0, 0.0], device=self.device, dtype=self.dtype)

        self.command_mode = "pedals"
        self._ax_guess = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self._ay_guess = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        #: optional per-env multiplicative friction scale (time varying disturbance)
        self.tyre_scale = torch.ones((self.n, 1), device=self.device, dtype=self.dtype)

    # -- helpers ----------------------------------------------------------
    def col(self, key: str) -> torch.Tensor:
        """Per-env column of a tensor parameter, shape (n,)."""
        return self.p[:, PARAM_INDEX[key]]

    def geometry(self):
        n = self.n
        lf, lr = self.col("lF"), self.col("lR")
        tf, tr = self.col("Tf"), self.col("Tr")
        x_pos = torch.stack([lf, lf, -lr, -lr], dim=1)                     # (n,4)
        y_pos = torch.stack([tf * 0.5, -tf * 0.5, tr * 0.5, -tr * 0.5], dim=1)
        radius = torch.stack([self.col("rF"), self.col("rF"), self.col("rR"), self.col("rR")], dim=1)
        inertia = torch.stack([self.col("IwF"), self.col("IwF"), self.col("IwR"), self.col("IwR")], dim=1)
        return x_pos, y_pos, radius, inertia

    # -- randomization ----------------------------------------------------
    def randomize(self, mask=None, override=None, ranges=None):
        """Randomize scalar parameters for the selected environments.

        ``override`` sets fixed values, ``ranges`` gives ``{key: [lo, hi]}``
        uniform ranges (both per key).  ``key`` must be in ``TENSOR_KEYS``.
        """
        if mask is None:
            mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        num = int(mask.sum().item())
        if num == 0:
            return
        override = override or {}
        ranges = ranges or {}
        for key, value in override.items():
            if key in PARAM_INDEX:
                self.p[mask, PARAM_INDEX[key]] = float(value)
        for key, (lo, hi) in ranges.items():
            if key in PARAM_INDEX:
                self.p[mask, PARAM_INDEX[key]] = lo + (hi - lo) * torch.rand(num, device=self.device, dtype=self.dtype)

    def reset_randomization(self, mask=None):
        if mask is None:
            self.p[:] = self.p_default
        else:
            self.p[mask] = self.p_default

    def reset_guess(self, mask=None):
        """Clear the load-transfer warm start (call on episode reset)."""
        if mask is None:
            self._ax_guess.zero_()
            self._ay_guess.zero_()
        else:
            self._ax_guess[mask] = 0.0
            self._ay_guess[mask] = 0.0

    # -- state ------------------------------------------------------------
    def zero_state(self, n: int | None = None) -> torch.Tensor:
        n = self.n if n is None else n
        return torch.zeros((n, NUM_STATES), device=self.device, dtype=self.dtype)

    def init_wheels(self, s: torch.Tensor, slip: float = 0.0) -> torch.Tensor:
        """Set wheel speeds to the pure-rolling value implied by the body state."""
        _, _, radius, _ = self.geometry()
        vx = s[:, 3]
        omega = vx.unsqueeze(1) * (1.0 + slip) / radius
        s[:, 6:10] = omega
        return s

    def settle_wheels(self, s: torch.Tensor) -> torch.Tensor:
        """Idempotent helper for reset paths: if wheel speeds are still zero,
        initialise them from the body speed."""
        zero = (s[:, 6:10].abs().sum(dim=1) < 1e-9)
        if zero.any():
            tmp = s[zero]
            self.init_wheels(tmp)
            s[zero] = tmp
        return s

    # -- tire -------------------------------------------------------------
    @staticmethod
    def _mf(slip, B, C, D, E):
        """XCAR isotropic Magic Formula and its derivative w.r.t. the slip."""
        bs = B * slip
        at = torch.atan(bs)
        phi = bs - E * (bs - at)
        mu = D * torch.sin(C * torch.atan(phi))
        dphi = B * (1.0 - E * (bs * bs) / (1.0 + bs * bs))
        dmu = D * torch.cos(C * torch.atan(phi)) * (C / (1.0 + phi * phi)) * dphi
        return mu, dmu

    def _aero(self, s: torch.Tensor):
        """Aerodynamic forces/moments in the body frame (section 08)."""
        psi = s[:, 2]
        vxw, vyw, r = s[:, 3], s[:, 4], s[:, 5]
        c, sn = torch.cos(psi), torch.sin(psi)
        vxb = c * vxw + sn * vyw
        vyb = -sn * vxw + c * vyw

        rho = self.col("rho_air")
        ax_a, ay_a = self.col("aero_Ax"), self.col("aero_Ay")
        cdx, cdy = self.col("aero_Cdx"), self.col("aero_Cdy")
        clx, cly = self.col("aero_Clx"), self.col("aero_Cly")
        xp, zp = self.col("aero_xp"), self.col("aero_zp")

        wx, wy = self.col("wind_x"), self.col("wind_y")
        wx_b = c * wx + sn * wy
        wy_b = -sn * wx + c * wy

        U = vxb - wx_b
        V = vyb + r * xp - wy_b
        fx = -0.5 * rho * ax_a * cdx * U * torch.abs(U)
        fy = -0.5 * rho * ay_a * cdy * V * torch.abs(V)
        fz = 0.5 * rho * (ax_a * clx * U * U + ay_a * cly * V * V)
        mx = -zp * fy
        my = zp * fx - xp * fz
        mz = xp * fy
        return fx, fy, fz, mx, my, mz

    def _wheel_forces(self, s, delta, fz, geometry):
        """Combined-slip isotropic tire forces in wheel and body frames.

        Returns a dict with the per-wheel forces plus the slip diagnostics.
        """
        x_pos, y_pos, radius, _ = geometry
        vxw, vyw, r = s[:, 3], s[:, 4], s[:, 5]
        c, sn = torch.cos(s[:, 2]), torch.sin(s[:, 2])
        vxb = c * vxw + sn * vyw
        vyb = -sn * vxw + c * vyw

        vx_b = vxb.unsqueeze(1) - r.unsqueeze(1) * y_pos
        vy_b = vyb.unsqueeze(1) + r.unsqueeze(1) * x_pos
        cd, sd = torch.cos(delta).unsqueeze(1), torch.sin(delta).unsqueeze(1)
        vx_l = vx_b.clone()
        vy_l = vy_b.clone()
        vx_l[:, :2] = cd * vx_b[:, :2] + sd * vy_b[:, :2]
        vy_l[:, :2] = -sd * vx_b[:, :2] + cd * vy_b[:, :2]

        B, C, D, E = self.col("B").unsqueeze(1), self.col("C").unsqueeze(1), \
            self.col("D").unsqueeze(1), self.col("E").unsqueeze(1)
        axle_scale = torch.where(self.wheel_is_front.unsqueeze(0) > 0.5,
                                 self.col("tyre_scale_front").unsqueeze(1),
                                 self.col("tyre_scale_rear").unsqueeze(1))
        d_eff = D * axle_scale * self.tyre_scale
        if float(self.col("load_sensitivity").abs().max().item()) > 0.0:
            fz_ref = (self.col("m") * self.col("g") * 0.25).unsqueeze(1).clamp(min=1.0)
            sens = self.col("load_sensitivity").unsqueeze(1)
            d_eff = d_eff * torch.clamp(1.0 - sens * (fz / fz_ref - 1.0), 0.4, 1.4)

        vf = self.col("slip_v_floor").unsqueeze(1)
        vfa = self.col("slip_alpha_v_floor").unsqueeze(1)
        denom = torch.maximum(vx_l.abs(), vf)
        denom_a = torch.maximum(vx_l.abs(), vfa)

        omega = s[:, 6:10]
        sx = (vx_l - omega * radius) / denom
        sy = vy_l / denom_a
        slip_raw = torch.hypot(sx, sy)
        # The Magic Formula keeps falling past a very large slip (for C < 2 it
        # tends to D*sin(C*pi/2)).  Real tires show a plateau; clamp the evaluated
        # slip so a spinning wheel keeps a physically sensible force instead of
        # losing almost all grip at 100x the peak slip.
        slip = torch.clamp(slip_raw, max=self.slip_max)
        mu, dmu = self._mf(slip, B, C, d_eff, E)
        alpha = torch.atan2(sy, sx)
        mux = -torch.cos(alpha) * mu
        muy = -torch.sin(alpha) * mu
        fx_wheel = mux * fz
        fy_wheel = muy * fz

        fx_body = fx_wheel.clone()
        fy_body = fy_wheel.clone()
        fx_body[:, :2] = cd * fx_wheel[:, :2] - sd * fy_wheel[:, :2]
        fy_body[:, :2] = sd * fx_wheel[:, :2] + cd * fy_wheel[:, :2]

        # |dFx/domega| used by the semi-implicit wheel update.  The tangent alone
        # under-estimates the stiffness when the step crosses toward zero slip
        # (the slip ratio is very sensitive at low speed); take the larger of the
        # local tangent and the secant to the zero-force point so the implicit
        # step cannot overshoot the rolling equilibrium and chatter.
        with torch.no_grad():
            ds_dsx = torch.where(slip > 1e-9, sx / slip.clamp(min=1e-9),
                                 torch.zeros_like(slip))
            kappa = (fz * torch.cos(alpha).abs() * dmu.abs()
                     * ds_dsx.abs() * radius / denom)
            kappa = torch.where(slip_raw < self.slip_max, kappa, torch.zeros_like(kappa))
            secant = fx_wheel.abs() / torch.clamp(sx.abs(), min=1e-3) * radius / denom
            kappa = torch.maximum(kappa, secant).clamp(min=0.0)

        return {
            "fx_wheel": fx_wheel, "fy_wheel": fy_wheel,
            "fx_body": fx_body, "fy_body": fy_body,
            "fz": fz, "sx": sx, "sy": sy, "slip": slip, "mu": mu,
            "alpha": alpha, "mux": mux, "muy": muy,
            "dFx_domega": kappa, "vx_wheel": vx_l, "vy_wheel": vy_l,
        }

    def _loads(self, ax, ay, aero, geometry):
        """Quasi-static wheel loads with longitudinal + lateral transfer (section F/IV)."""
        x_pos, y_pos, _, _ = geometry
        m, h, g = self.col("m"), self.col("h"), self.col("g")
        lf, lr = self.col("lF"), self.col("lR")
        tf, tr = self.col("Tf"), self.col("Tr")
        lam = self.col("roll_share_front")

        fx_a, fy_a, fz_a, mx_a, my_a, _ = aero
        total_raw = m * g - fz_a
        total = torch.clamp(total_raw, min=0.0)
        front_raw = (lr * total_raw + my_a - h * (m * ax - fx_a)) / (lf + lr)
        front = torch.maximum(torch.minimum(front_raw, total), torch.zeros_like(total))
        rear = total - front

        q_lat = h * (m * ay - fy_a) + mx_a
        df_front_raw = lam * q_lat / tf
        df_rear_raw = (1.0 - lam) * q_lat / tr
        df_front = torch.maximum(torch.minimum(df_front_raw, front * 0.5), -front * 0.5)
        df_rear = torch.maximum(torch.minimum(df_rear_raw, rear * 0.5), -rear * 0.5)

        fz_fl = front * 0.5 - df_front
        fz_fr = front * 0.5 + df_front
        fz_rl = rear * 0.5 - df_rear
        fz_rr = rear * 0.5 + df_rear
        fz = torch.stack([fz_fl, fz_fr, fz_rl, fz_rr], dim=1)
        limited = (total_raw < 0) | (torch.abs(df_front - df_front_raw) > 1e-6) | \
                  (torch.abs(df_rear - df_rear_raw) > 1e-6)
        return fz, limited

    # -- motor / LSD / brakes --------------------------------------------
    def _motor_torque(self, omega_pair, throttle, brake):
        """Axle torque from the pedals (total for the axle seen by the differential)."""
        gear = self.col("gear_ratio")
        eff = self.col("drive_efficiency")
        wm = 0.5 * gear * (omega_pair[:, 0] + omega_pair[:, 1])
        rpm = wm.abs() * (60.0 / (2.0 * math.pi))

        idx = torch.bucketize(rpm, self.motor_rpm)
        idx = torch.clamp(idx, 1, self.motor_rpm.numel() - 1)
        r0 = self.motor_rpm[idx - 1]
        r1 = self.motor_rpm[idx]
        t0 = self.motor_torque[idx - 1]
        t1 = self.motor_torque[idx]
        t_map = t0 + (t1 - t0) * (rpm - r0) / torch.clamp(r1 - r0, min=1e-6)
        t_map = torch.where(rpm > self.motor_rpm_max, torch.zeros_like(t_map), t_map)

        p_max = self.col("max_power_w")
        t_power = p_max / torch.clamp(wm.abs(), min=1.0)
        t_max = torch.minimum(t_map, t_power)

        # pedal mapping: throttle requests torque, brake overrides it
        q = torch.clamp(throttle - brake, min=0.0)
        t_motor = q * t_max
        # zero-pedal driveline drag (EV regen / in-gear coast), axle total
        coast = self.col("coast_torque_nm") * (1.0 - torch.clamp(throttle + brake, max=1.0)) \
            * torch.tanh(gear * wm / 30.0)
        axle = (t_motor - coast) * gear * eff
        return axle, t_motor, wm

    def _lsd(self, axle_torque, omega_pair):
        """Viscous limited-slip coupling on one axle (section E of the lab page)."""
        kd = self.col("lsd_kd")
        cap = self.col("lsd_tau_max")
        dw = omega_pair[:, 0] - omega_pair[:, 1]
        coupling = torch.clamp(kd * dw, min=-cap, max=cap)
        coupling = torch.where(cap > 0, coupling, torch.zeros_like(coupling))
        return axle_torque * 0.5 - coupling, axle_torque * 0.5 + coupling, coupling

    # -- derivatives ------------------------------------------------------
    def derivatives(self, s: torch.Tensor, command: torch.Tensor, mode: str = "pedals"):
        """State derivative and diagnostics for one physics step."""
        geometry = self.geometry()
        x_pos, y_pos, radius, inertia = geometry

        if mode == "pedals":
            steer, throttle, brake = command[:, 0], command[:, 1], command[:, 2]
        else:
            steer = command[:, 0]
            throttle = torch.zeros_like(steer)
            brake = torch.zeros_like(steer)
        delta = torch.clamp(steer, -self.col("max_steer_rad"), self.col("max_steer_rad"))

        m = self.col("m")
        aero = self._aero(s)

        # Quasi-static load transfer.  The body accelerations of the previous
        # evaluation are an excellent initial guess (they change slowly), so one
        # predictor pass plus one final force evaluation are enough at 100 Hz.
        ax = self._ax_guess
        ay = self._ay_guess
        fz, forces = None, None
        for _ in range(self.load_iters):
            fz, limited = self._loads(ax, ay, aero, geometry)
            forces = self._wheel_forces(s, delta, fz, geometry)
            ax = (forces["fx_body"].sum(dim=1) + aero[0]) / m
            ay = (forces["fy_body"].sum(dim=1) + aero[1]) / m
        self._ax_guess = ax.detach()
        self._ay_guess = ay.detach()

        fx_b, fy_b = forces["fx_body"], forces["fy_body"]
        fx_w, fy_w = forces["fx_wheel"], forces["fy_wheel"]
        omega = s[:, 6:10]

        # --- yaw and body accelerations ---
        mz = (x_pos * fy_b - y_pos * fx_b).sum(dim=1) + aero[5]
        rd = mz / self.col("Iz")

        psi = s[:, 2]
        c, sn = torch.cos(psi), torch.sin(psi)
        dvx = c * ax - sn * ay
        dvy = sn * ax + c * ay
        vxw, vyw = s[:, 3], s[:, 4]
        dx = c * vxw - sn * vyw
        dy = sn * vxw + c * vyw

        # --- wheel torques ---
        drive_torque = torch.zeros_like(omega)
        if mode == "pedals":
            axle, t_motor, wm = self._motor_torque(
                torch.stack([omega[:, 2], omega[:, 3]], dim=1), throttle, brake)
            t_l, t_r, coupling = self._lsd(axle, torch.stack([omega[:, 2], omega[:, 3]], dim=1))
            if self.drivetrain in ("rwd", "awd", "iwd"):
                share = 1.0 if self.drivetrain == "rwd" else 0.5
                drive_torque = torch.stack([
                    torch.zeros_like(t_l), torch.zeros_like(t_l), t_l * share, t_r * share], dim=1)
            if self.drivetrain == "fwd":
                drive_torque = torch.stack([t_l, t_r, torch.zeros_like(t_l), torch.zeros_like(t_l)], dim=1)
            if self.drivetrain == "awd":
                drive_torque[:, 0] = axle * 0.25
                drive_torque[:, 1] = axle * 0.25
            extras_motor = {"axle_torque": axle, "motor_torque": t_motor,
                            "motor_speed": wm, "coupling": coupling}
        else:
            # legacy wheel-speed command adapter: feed-forward + proportional servo
            cmd_v = command[:, 1:5]
            omega_cmd = cmd_v / radius
            kp = self.col("wheel_speed_kp").unsqueeze(1)
            drive_torque = radius * fx_w + kp * (omega_cmd - omega)
            extras_motor = {"axle_torque": torch.zeros_like(m),
                            "motor_torque": torch.zeros_like(m),
                            "motor_speed": torch.zeros_like(m),
                            "coupling": torch.zeros_like(m)}

        # brakes (per wheel, saturating at standstill)
        p_brake = brake * self.col("brake_pressure_max_mpa")
        t_front = p_brake * self.col("brake_torque_front_nm_per_mpa")
        t_rear = p_brake * self.col("brake_torque_rear_nm_per_mpa")
        brake_torque = torch.stack([t_front, t_front, t_rear, t_rear], dim=1)
        brake_torque = brake_torque * torch.tanh(omega / 0.5)

        roll_torque = (self.col("rolling_resistance").unsqueeze(1) * fz * radius
                       * torch.tanh(omega / 0.5))

        net = drive_torque - brake_torque - roll_torque - radius * fx_w
        domega = net / inertia

        deriv = torch.cat([
            dx.unsqueeze(1), dy.unsqueeze(1), s[:, 5].unsqueeze(1),
            dvx.unsqueeze(1), dvy.unsqueeze(1), rd.unsqueeze(1),
            domega,
        ], dim=1)

        extras = dict(forces)
        extras.update(extras_motor)
        extras.update({
            "ax": ax, "ay": ay, "yaw_acc": rd, "aero_fx": aero[0], "aero_fy": aero[1],
            "aero_fz": aero[2], "aero_mz": aero[5], "load_limited": limited,
            "delta": delta, "throttle": throttle, "brake": brake,
            "drive_torque": drive_torque, "brake_torque": brake_torque,
            "wheel_net_torque": net, "wheel_inertia": inertia, "wheel_radius": radius,
        })
        return deriv, extras

    # -- integration ------------------------------------------------------
    def step(self, s: torch.Tensor, command: torch.Tensor, dt: float,
             substeps: int = 1, mode: str = "pedals", keep_wheel_sign: bool = True):
        """Advance the plant.

        ``dt`` is the *control* period; it is split into ``substeps`` physics
        steps over which ``command`` is held constant (zero order hold).
        """
        h = float(dt) / max(int(substeps), 1)
        for _ in range(max(int(substeps), 1)):
            deriv, extras = self.derivatives(s, command, mode=mode)
            s[:, :6] = s[:, :6] + h * deriv[:, :6]
            # semi-implicit wheel spin: backward Euler linearised on the tire
            # longitudinal stiffness, stable at low speed where the slip ratio
            # denominator makes the wheel mode very stiff
            j_eff = (extras["wheel_inertia"]
                     + self.wheel_implicit_weight * h * extras["wheel_radius"] * extras["dFx_domega"])
            s[:, 6:10] = s[:, 6:10] + h * extras["wheel_net_torque"] / j_eff
            if keep_wheel_sign:
                # a wheel may not reverse through zero within one physics step
                s[:, 6:10] = torch.clamp(s[:, 6:10], min=0.0, max=2500.0)
            s[:, 2] = torch.atan2(torch.sin(s[:, 2]), torch.cos(s[:, 2]))
        return s, extras

    # -- observation ------------------------------------------------------
    def package_extended(self, s: torch.Tensor, extras: dict) -> torch.Tensor:
        """Assemble the 56-column diagnostic state from one derivative evaluation.

        Cheap: it only reuses the state and the ``extras`` of the last physics
        substep, so no additional tire force evaluation is needed.
        """
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
            extras["sx"], extras["sy"], extras["slip"], extras["mu"], extras["alpha"],
            extras["mux"], extras["muy"], extras["fz"],
            extras["fx_body"], extras["fy_body"],
        ]
        return torch.cat(parts, dim=1)

    def extended_state(self, s: torch.Tensor, command: torch.Tensor | None = None,
                       mode: str = "pedals") -> torch.Tensor:
        """56-column diagnostic state, layout compatible with ``IWDCarDynamics``.

        ``[vx, vy, r, ax_w, ay_w, rdot, v, beta, vx_wheel(4), vy_wheel(4),
        sx(4), sy(4), slip(4), mu(4), alpha(4), mux(4), muy(4), fz(4), fx(4),
        fy(4)]``
        """
        if command is None:
            command = torch.zeros((s.shape[0], NUM_ACTIONS), device=self.device, dtype=s.dtype)
        deriv, ex = self.derivatives(s, command, mode=mode)
        psi = s[:, 2]
        vxw, vyw = s[:, 3], s[:, 4]
        v = torch.hypot(vxw, vyw)
        beta = torch.atan2(vyw, vxw) - psi
        beta = torch.atan2(torch.sin(beta), torch.cos(beta))
        psi_dot = s[:, 5]
        ax_w = deriv[:, 3]
        ay_w = deriv[:, 4]
        vec = lambda t: t if t.dim() == 2 else t.unsqueeze(1)
        parts = [
            vxw.unsqueeze(1), vyw.unsqueeze(1), psi_dot.unsqueeze(1),
            ax_w.unsqueeze(1), ay_w.unsqueeze(1), deriv[:, 5].unsqueeze(1),
            v.unsqueeze(1), beta.unsqueeze(1),
            ex["vx_wheel"], ex["vy_wheel"],
            ex["sx"], ex["sy"], ex["slip"], ex["mu"], ex["alpha"],
            ex["mux"], ex["muy"], ex["fz"], ex["fx_body"], ex["fy_body"],
        ]
        return torch.cat([vec(t) for t in parts], dim=1)


# ---------------------------------------------------------------------------
# Small self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    params = preset_parameters("fullsize_racecar_ev" if os.environ.get("PLANT_PRESET") is None
                               else os.environ["PLANT_PRESET"])
    plant = FastRacerPlant(params, 1, device=device)
    s = plant.zero_state(1)
    s[:, 3] = 0.0
    plant.init_wheels(s)
    cmd = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    for i in range(300):
        s, ex = plant.step(s, cmd, dt=0.01, substeps=1)
        if i % 50 == 0:
            print(f"t={i*0.01:5.2f}s v={float(s[0,3]):7.3f} m/s "
                  f"ax={float(ex['ax']):6.2f} omega_rl={float(s[0,8]):7.2f} "
                  f"slip={float(ex['slip'][0,2]):6.3f} Fz_r={float(ex['fz'][0,2]):7.1f}")
