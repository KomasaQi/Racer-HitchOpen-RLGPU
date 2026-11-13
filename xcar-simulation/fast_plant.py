# -*- coding: utf-8 -*-
"""FastPlant: GPU-vectorized planar full-size RWD EV plant.

The plant keeps XCAR's compact analytic Pacejka philosophy.  It supports the
original isotropic radial curve and a calibrated longitudinal/lateral variant;
neither performs a runtime tyre-table lookup.  Around that compact tyre it adds
the intermediate vehicle modules that the reference lab page
(``鐞嗚ВPacejka妯″瀷/xcar-tire-lab_aero.html``, sections 07-08) uses:

* simplified electric motor: speed dependent torque envelope + power limit;
* compact generalized-viscous limited-slip differential on the driven axle;
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
#: (``鏀归€犵數杞﹁溅杈?CarSim_motor_384kW_direct_drive.csv``).
EV_MOTOR_384KW = [
    [0.0, 2400.0], [250.0, 2400.0], [500.0, 2400.0], [750.0, 2400.0],
    [1000.0, 2400.0], [1250.0, 2400.0], [1500.0, 2400.0], [1600.0, 2293.62],
    [1800.0, 2038.77], [2000.0, 1834.89], [2500.0, 1467.91], [3000.0, 1223.26],
    [3500.0, 1048.51], [4000.0, 917.45], [4500.0, 815.51], [5000.0, 733.96],
    [5500.0, 667.23], [6000.0, 611.63], [6500.0, 564.58], [7000.0, 524.26],
    [7500.0, 489.30], [8000.0, 458.72], [8500.0, 431.74], [9000.0, 407.75],
    [9200.0, 398.89],
]

# Positive-torque half of CarSim's r_eff_motor_CARPET.  The source function is
# 2D_LINEAR, so torque above 305 Nm follows linear extrapolation (with CarSim's
# final 0.02 efficiency floor); the motor itself can produce up to 2400 Nm.
EV_EFF_RPM = [0., 500., 1000., 1500., 2000., 2500., 3000., 3500., 4000., 4500., 6000.]
EV_EFF_TORQUE = [0., 5., 35., 65., 95., 125., 155., 185., 215., 245., 275., 305.]
EV_EFFICIENCY = [
    [0.85975] * 12,
    [0.85975, 0.8265, 0.855, 0.8075, 0.760, 0.722, 0.684, 0.646, 0.6175, 0.589, 0.5605, 0.532],
    [0.85975, 0.8075, 0.893, 0.8645, 0.8455, 0.817, 0.7885, 0.7695, 0.741, 0.722, 0.703, 0.684],
    [0.85975, 0.7885, 0.893, 0.8835, 0.874, 0.855, 0.836, 0.817, 0.741, 0.722, 0.703, 0.684],
    [0.85975, 0.7790, 0.9025, 0.9025, 0.8835, 0.874, 0.836, 0.817, 0.741, 0.722, 0.703, 0.684],
    [0.85975, 0.7695, 0.9025, 0.9025, 0.893, 0.874, 0.836, 0.817, 0.741, 0.722, 0.703, 0.684],
    [0.85975, 0.7695, 0.9025, 0.9120, 0.9025, 0.874, 0.836, 0.817, 0.741, 0.722, 0.703, 0.684],
    [0.85975, 0.7600, 0.9025, 0.9120, 0.9025, 0.874, 0.836, 0.817, 0.741, 0.722, 0.703, 0.684],
    [0.85975, 0.7600, 0.9025, 0.9025, 0.9025, 0.874, 0.836, 0.817, 0.741, 0.722, 0.703, 0.684],
    [0.85975, 0.7505, 0.9025, 0.9025, 0.9025, 0.874, 0.836, 0.817, 0.741, 0.722, 0.703, 0.684],
    [0.85975, 0.7505, 0.9025, 0.9025, 0.9025, 0.874, 0.836, 0.817, 0.741, 0.722, 0.703, 0.684],
]


def _scalar_motor_efficiency(rpm: float, torque: float) -> float:
    """Nominal CarSim efficiency carpet for one offline/precompute point."""
    r = min(max(abs(rpm), EV_EFF_RPM[0]), EV_EFF_RPM[-1])
    q = max(abs(torque), EV_EFF_TORQUE[0])

    def bracket(grid, value):
        for i in range(1, len(grid)):
            if value <= grid[i]:
                return i - 1, i
        return len(grid) - 2, len(grid) - 1

    ir0, ir1 = bracket(EV_EFF_RPM, r)
    iq0, iq1 = bracket(EV_EFF_TORQUE, q)
    wr = (r - EV_EFF_RPM[ir0]) / (EV_EFF_RPM[ir1] - EV_EFF_RPM[ir0])
    wq = (q - EV_EFF_TORQUE[iq0]) / (
        EV_EFF_TORQUE[iq1] - EV_EFF_TORQUE[iq0])
    e0 = EV_EFFICIENCY[ir0][iq0] * (1.0 - wq) + EV_EFFICIENCY[ir0][iq1] * wq
    e1 = EV_EFFICIENCY[ir1][iq0] * (1.0 - wq) + EV_EFFICIENCY[ir1][iq1] * wq
    return min(max(e0 * (1.0 - wr) + e1 * wr, 0.02), 0.98)


def _build_battery_torque_envelope():
    """Precompute the nominal analytic battery ceiling once, never per step."""
    rpm_grid = [float(x) for x in range(0, 9201, 25)]
    pmax = 720.0 * (936.936 - 720.0) / 0.12
    torque_grid = []
    for rpm in rpm_grid:
        omega = rpm * (2.0 * math.pi / 60.0)
        if omega < 1e-9:
            torque_grid.append(2400.0)
            continue
        lo, hi = 0.0, 2400.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            required = mid * omega / _scalar_motor_efficiency(rpm, mid)
            if required <= pmax:
                lo = mid
            else:
                hi = mid
        torque_grid.append(lo)
    return rpm_grid, torque_grid


EV_BATTERY_LIMIT_RPM, EV_BATTERY_LIMIT_TORQUE = _build_battery_torque_envelope()


def _build_effective_motor_envelope():
    """Merge the source motor and battery ceilings into one runtime map."""
    out = []
    for rpm, battery_limit in zip(EV_BATTERY_LIMIT_RPM, EV_BATTERY_LIMIT_TORQUE):
        j = 1
        while j < len(EV_MOTOR_384KW) and rpm > EV_MOTOR_384KW[j][0]:
            j += 1
        if j >= len(EV_MOTOR_384KW):
            motor_limit = 0.0
        else:
            r0, t0 = EV_MOTOR_384KW[j - 1]
            r1, t1 = EV_MOTOR_384KW[j]
            motor_limit = t0 + (t1 - t0) * (rpm - r0) / max(r1 - r0, 1e-9)
        out.append(min(motor_limit, battery_limit))
    return out


EV_EFFECTIVE_MOTOR_TORQUE = _build_effective_motor_envelope()

# Dimensionless multipliers from the CarSim L_RELAX_X/Y tables.  The physical
# gains are RRE/5 longitudinally and 2*RRE laterally; values outside the table
# use the flat edge, matching SPLINE_FLAT.
RELAX_X_ABS = [0., .005, .01, .015, .02, .025, .03, .035, .04, .045,
               .05, .055, .06, .065, .07, .075, .08, .085, .09, .095, .10]
RELAX_X_FACTOR = [1., .90707, .7393, .59275, .48251, .40397, .34575,
                  .30301, .26989, .24389, .22324, .20631, .1926, .18085,
                  .17087, .16251, .1552, .14862, .1431, .13793, .13526]
RELAX_Y_DEG = [0., .5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5,
               6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10.]
RELAX_Y_FACTOR = [1., .89674, .71033, .5475, .42501, .33775, .27305,
                  .22557, .18877, .15988, .13693, .11812, .10289, .089829,
                  .078749, .069451, .061335, .054018, .047886, .042142, .039175]

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
    "bx": 10.0, "cx": 1.90, "ex": 0.97,
    "by": 8.0, "cy": 1.30, "ey": -0.5,
    "tyre_scale_front": 1.0, "tyre_scale_rear": 1.0,
    "slip_v_floor": 1.0,          # m/s, denominator floor of the slip ratio
    "slip_alpha_v_floor": 1.0,    # m/s, denominator floor of the slip angle
    "slip_max": 5.0,              # evaluated slip clamp (post-peak force plateau)
    "slip_force_plateau": 0.30,   # nominal B=6,C=1.9,E=.97 first Fx peak
    "slip_alpha_force_plateau_rad": 1.40,
    # All modes are compact analytic Magic-Formula approximations; none performs
    # a runtime lookup into a CarSim tyre table. ``compact_transition`` allows
    # different longitudinal/lateral curves and a one-scalar combined-slip map.
    "combined_slip_model": "compact_transition",
    "combined_slip_blend": 0.50,
    # CarSim road friction scales peak force but preserves much of the tire's
    # small-slip stiffness, shifting peak slip inward as grip falls.  Exponent
    # zero is the legacy D-only scaling; one gives exact stiffness preservation.
    "tire_mu_reference": 2.12460,
    "tire_stiffness_grip_exponent_x": 0.0,
    "tire_stiffness_grip_exponent_y": 0.0,
    "kappa_peak": 0.067524,
    "alpha_peak_rad": 0.115537,
    "load_sensitivity": 0.0,      # 0 = pure XCAR (no tire load sensitivity)
    "fz_ref_front": 0.0,          # 0 -> static m*g/4 compatibility reference
    "fz_ref_rear": 0.0,
    "tire_relaxation": False,     # CarSim load/slip-dependent kappa/alpha lag
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
    "drive_efficiency": 0.99,     # differential/final-drive mechanical efficiency
    "motor_efficiency": 1.0,      # multiplier on the CarSim efficiency carpet
    "throttle_power_gain": 1.0,
    "throttle_power_rolloff": 0.0,
    "max_power_w": 384000.0,
    "motor": EV_MOTOR_384KW,
    # Rotating inertias from the expanded CarSim powertrain pars.  The motor is
    # upstream of the 3.44 final drive and is shared by both driven wheels; its
    # reflected inertia therefore contributes I_motor * ratio^2 / 2 per wheel.
    "motor_inertia": 0.0226,
    "diff_input_inertia": 0.013,
    "halfshaft_inertia": 0.009,
    # Equivalent total-pack battery values at the configured initial SOC=.8.
    # Source: 120 series packs, 7.8078 V OCV/cell, 6 V minimum/cell and
    # 1 mOhm/cell.  The analytic discharge ceiling is
    # V_min * (V_oc - V_min) / R = 1.3016 MW at reset.
    "battery_voc_v": 120.0 * 7.8078,
    "battery_min_voltage_v": 120.0 * 6.0,
    "battery_resistance_ohm": 120.0 * 0.001,
    "battery_power_limit_scale": 1.0,
    "use_battery_torque_envelope": False,
    "coast_torque_nm": 0.0,       # axle total, scaled by (1 - throttle)
    "wheel_speed_kp": 300.0,      # legacy wheel-speed command adapter gain
    # --- limited slip differential ---
    # Generalized viscous coupling identified from CarSim's rear-differential
    # carpet.  The carpet is used for offline identification only: at runtime
    # FastPlant evaluates the two-parameter power law
    #   |T_L-T_R| = lsd_torque_ref * (|dn_rpm|/lsd_ref_rpm)^lsd_exponent.
    # The old linear coefficient is retained only for legacy presets.
    "lsd_model": "power_law",
    "lsd_ref_rpm": 60.0,
    "lsd_torque_ref": 90.09,      # N m LEFT-minus-RIGHT at 60 rpm magnitude
    "lsd_exponent": 0.50,         # 1 = Newtonian viscous; 0.5 = CarSim nominal
    "lsd_kd": 20.0,               # legacy N m s/rad per-side coupling
    "lsd_tau_max": 300.0,         # N m, per-side numerical safety cap
    # --- brakes ---
    "brake_pressure_max_mpa": 6.0,
    # MEASURED against CarSim (round 68).  At 0.05 / 0.10 / 0.20 pedal the GPU plant
    # changed speed by +0.15 / +0.41 / +0.61 m/s over one second while CarSim changed
    # -0.39 / -1.02 / -2.27: the GPU brakes were 2.5-3.7x too weak, and at 0.05 they did
    # not even overcome the coast.  A policy that trims with a small brake on the GPU
    # therefore brakes HARD in CarSim -- which is exactly why the transferred car was
    # too slow.  Scaled by 3.0 to match.
    "brake_torque_front_nm_per_mpa": 1200.0,
    "brake_torque_rear_nm_per_mpa": 525.0,
    # --- misc ---
    "rolling_resistance": 0.012,
    "max_steer_rad": 0.45,
    "load_iters": 2,              # load transfer passes per derivative (>=2)
    # Trapezoidal weight of the semi-implicit wheel update.  1.0 = backward
    # Euler (most damping, first order), 0.5 = trapezoidal (less bias while
    # accelerating/braking, still A-stable).
    # 0.5 made the implicit denominator more than 100x J at low speed (the secant
    # to the zero-force point is huge there), so the driven wheel could only creep
    # and the >=0 clamp then pinned it locked.  0.15 is enough damping to stop the
    # step overshooting without suppressing the spin-up.
    "wheel_implicit_weight": 0.15,
    # Scalar monotone wheel-speed solve.  Kept outside the tensor/DR contract:
    # this is a numerical-accuracy setting, not a physical parameter.
    "wheel_solver_bracket_iters": 8,
    "wheel_solver_bisection_iters": 12,
}

#: tensor parameter order of ``FastRacerPlant.p`` (per environment row)
TENSOR_KEYS = (
    "lF", "lR", "Tf", "Tr", "m", "h", "g", "Iz",
    "rF", "rR", "IwF", "IwR",
    "B", "C", "D", "E", "bx", "cx", "ex", "by", "cy", "ey",
    "tyre_scale_front", "tyre_scale_rear",
    "slip_v_floor", "slip_alpha_v_floor", "load_sensitivity",
    "combined_slip_blend", "tire_mu_reference",
    "tire_stiffness_grip_exponent_x", "tire_stiffness_grip_exponent_y",
    "fz_ref_front", "fz_ref_rear",
    "roll_share_front",
    "rho_air", "aero_Ax", "aero_Ay", "aero_Cdx", "aero_Cdy",
    "aero_Clx", "aero_Cly", "aero_xp", "aero_zp", "wind_x", "wind_y",
    "gear_ratio", "drive_efficiency", "motor_efficiency", "throttle_power_gain",
    "throttle_power_rolloff", "max_power_w", "coast_torque_nm",
    "motor_inertia", "diff_input_inertia", "halfshaft_inertia",
    "battery_voc_v", "battery_min_voltage_v", "battery_resistance_ohm",
    "battery_power_limit_scale",
    "wheel_speed_kp", "lsd_torque_ref", "lsd_exponent",
    "lsd_kd", "lsd_tau_max",
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
    for dst, src in (("bx", "B"), ("cx", "C"), ("ex", "E")):
        if dst not in raw:
            p[dst] = p[src]
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
        self.grip_bias = float(params.get("grip_bias", 1.0))
        self.drivetrain = str(params.get("drivetrain", "rwd")).lower()
        self.lsd_model = str(params.get("lsd_model", "power_law")).lower()
        self.lsd_ref_rpm = float(params.get("lsd_ref_rpm", 60.0))
        self.load_iters = max(int(params.get("load_iters", 2)), 1)
        self.slip_max = float(params.get("slip_max", 5.0))
        self.slip_force_plateau = float(params.get("slip_force_plateau", 0.30))
        self.slip_alpha_force_plateau = float(
            params.get("slip_alpha_force_plateau_rad", 1.40))
        self.combined_slip_model = str(
            params.get("combined_slip_model", "compact_transition"))
        # Backward compatibility for datasets/checkpoints produced before the
        # model was given a simulator-neutral name.  The implementation has
        # always been analytic; it never queried a CarSim table at runtime.
        if self.combined_slip_model == "carsim_transition":
            self.combined_slip_model = "compact_transition"
        self.combined_slip_blend = float(params.get("combined_slip_blend", 0.50))
        self.kappa_peak = float(params.get("kappa_peak", 0.067524))
        self.alpha_peak = float(params.get("alpha_peak_rad", 0.115537))
        self.load_sensitivity_model = str(params.get("load_sensitivity_model", "linear"))
        self.mu_y_ratio = float(params.get("mu_y_ratio", 1.05))
        self.mu_y_ratio_rear = float(params.get("mu_y_ratio_rear", 0.85))
        self.wheel_implicit_weight = float(params.get("wheel_implicit_weight", 0.5))
        self.wheel_solver_bracket_iters = max(
            int(params.get("wheel_solver_bracket_iters", 8)), 1)
        self.wheel_solver_bisection_iters = max(
            int(params.get("wheel_solver_bisection_iters", 12)), 1)
        _relax_all = bool(params.get("tire_relaxation", False))
        self.relax_kappa = bool(params.get("relax_kappa", _relax_all))
        self.relax_alpha = bool(params.get("relax_alpha", _relax_all))
        self.tire_relaxation = self.relax_kappa or self.relax_alpha

        # per-env float parameters, shape (n, len(TENSOR_KEYS))
        values = []
        for key in TENSOR_KEYS:
            v = params.get(key, DEFAULT_PARAMS.get(key))
            if key in ("drive_efficiency", "motor_efficiency") and v is None:
                v = 1.0
            values.append(float(v) if not isinstance(v, (list, tuple)) else 0.0)
        p0 = torch.tensor(values, device=self.device, dtype=self.dtype)
        self.p = p0.unsqueeze(0).repeat(self.n, 1).clone()
        self.p_default = p0.clone()

        # driven wheel mask (fixed per preset): [FL, FR, RL, RR]
        drive = torch.zeros(4, device=self.device, dtype=self.dtype)
        if self.drivetrain in ("rwd", "awd", "iwd"):
            drive[2:] = 1.0
        if self.drivetrain in ("fwd", "awd"):
            drive[:2] = 1.0
        self.drive_mask = drive
        self.is_rwd = self.drivetrain == "rwd"

        # Motor torque envelope (shared across envs).  For the CarSim EV preset,
        # its analytic battery ceiling is pre-merged into this same one-dimensional
        # map, so the training hot path pays no second lookup.
        if bool(params.get("use_battery_torque_envelope", False)):
            motor_source = list(zip(EV_BATTERY_LIMIT_RPM, EV_EFFECTIVE_MOTOR_TORQUE))
        else:
            motor_source = params.get("motor", EV_MOTOR_384KW)
        motor = torch.as_tensor(motor_source, device=self.device, dtype=self.dtype)
        order = torch.argsort(motor[:, 0])
        motor = motor[order]
        self.motor_rpm = motor[:, 0].contiguous()
        self.motor_torque = motor[:, 1].contiguous()
        self.motor_rpm_max = float(self.motor_rpm[-1].item())
        self.eff_rpm = torch.tensor(EV_EFF_RPM, device=self.device, dtype=self.dtype)
        self.eff_torque = torch.tensor(EV_EFF_TORQUE, device=self.device, dtype=self.dtype)
        self.eff_table = torch.tensor(EV_EFFICIENCY, device=self.device, dtype=self.dtype)
        # The analytic battery/efficiency solve is collapsed once into a dense
        # one-dimensional envelope.  Runtime cost is one linear interpolation,
        # preserving the XCAR-style hot path for massive GPU batches.
        self.battery_limit_rpm = torch.tensor(
            EV_BATTERY_LIMIT_RPM, device=self.device, dtype=self.dtype)
        self.battery_limit_torque = torch.tensor(
            EV_BATTERY_LIMIT_TORQUE, device=self.device, dtype=self.dtype)
        self.relax_x_grid = torch.tensor(RELAX_X_ABS, device=self.device, dtype=self.dtype)
        self.relax_x_factor = torch.tensor(RELAX_X_FACTOR, device=self.device, dtype=self.dtype)
        self.relax_y_grid = torch.tensor(RELAX_Y_DEG, device=self.device, dtype=self.dtype)
        self.relax_y_factor = torch.tensor(RELAX_Y_FACTOR, device=self.device, dtype=self.dtype)

        # cached constant geometry tensors
        self.corner_x = torch.tensor([1.0, 1.0, -1.0, -1.0], device=self.device, dtype=self.dtype)
        self.corner_y_sign = torch.tensor([1.0, -1.0, 1.0, -1.0], device=self.device, dtype=self.dtype)
        self.wheel_is_front = torch.tensor([1.0, 1.0, 0.0, 0.0], device=self.device, dtype=self.dtype)

        self.command_mode = "pedals"
        self._ax_guess = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self._ay_guess = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        #: optional per-env multiplicative friction scale (time varying disturbance)
        self.tyre_scale = torch.ones((self.n, 1), device=self.device, dtype=self.dtype)
        #: per-env Ackermann blend for the front axle: 0 = parallel steering,
        #: 1 = exact Ackermann geometry.  Randomised per episode by the env.
        self.ackermann_scale = torch.ones((self.n, 1), device=self.device, dtype=self.dtype)
        # Tire relaxation states are plant memory, not actor observations.  They
        # are intentionally kept outside the public 10-state vehicle contract.
        self._sx_relaxed = torch.zeros((self.n, 4), device=self.device, dtype=self.dtype)
        self._alpha_relaxed = torch.zeros((self.n, 4), device=self.device, dtype=self.dtype)
        self._relax_initialized = torch.zeros(self.n, device=self.device, dtype=torch.bool)

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
        # REFLECTED DRIVELINE INERTIA.  The previous patch used a guessed motor
        # inertia of 0.30 kg m^2 and added the *whole* reflected inertia to both
        # rear wheels.  CarSim's expanded pars state IMOTOR=0.0226, IDS=0.013 and
        # IHS_LR=IHS_RR=0.009.  One motor/differential is shared by two wheels, so
        # the upstream terms are split between them on the equal-speed mode.
        _i_dl = (
            0.5 * self.col("motor_inertia") * self.col("gear_ratio") ** 2
            + 0.5 * self.col("diff_input_inertia")
            + self.col("halfshaft_inertia")
        ).reshape(-1, 1)
        if self.drivetrain in ("rwd", "iwd"):
            inertia = inertia.clone()
            inertia[:, 2:] = inertia[:, 2:] + _i_dl
        elif self.drivetrain == "fwd":
            inertia = inertia.clone()
            inertia[:, :2] = inertia[:, :2] + _i_dl
        elif self.drivetrain == "awd":
            inertia = inertia.clone()
            inertia = inertia + 0.5 * _i_dl
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
                # D is the friction level; grip_bias > 1 biases the sample towards
                # the LOW end while keeping the full range, so a policy can be
                # trained with more low-grip experience without narrowing D
                u = torch.rand(num, device=self.device, dtype=self.dtype)
                if key == "D" and self.grip_bias != 1.0:
                    u = u.pow(self.grip_bias)
                self.p[mask, PARAM_INDEX[key]] = lo + (hi - lo) * u

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
            self._sx_relaxed.zero_()
            self._alpha_relaxed.zero_()
            self._relax_initialized.zero_()
        else:
            self._ax_guess[mask] = 0.0
            self._ay_guess[mask] = 0.0
            self._sx_relaxed[mask] = 0.0
            self._alpha_relaxed[mask] = 0.0
            self._relax_initialized[mask] = False

    # -- state ------------------------------------------------------------
    def zero_state(self, n: int | None = None) -> torch.Tensor:
        n = self.n if n is None else n
        return torch.zeros((n, NUM_STATES), device=self.device, dtype=self.dtype)

    def init_wheels(self, s: torch.Tensor, slip: float = 0.0) -> torch.Tensor:
        """Set wheel speeds to the pure-rolling value implied by the body state.

        The rolling speed follows the *body* longitudinal velocity
        ``vx*cos(psi) + vy*sin(psi)``, not the world frame ``vx``: at a non zero
        heading (or a large sideslip) those differ a lot and a reset that used
        the world component would start with completely wrong wheel speeds.
        """
        _, y_pos, radius, _ = self.geometry()
        psi = s[:, 2]
        vx_body = torch.cos(psi) * s[:, 3] + torch.sin(psi) * s[:, 4]
        # a wheel at lateral offset y has contact velocity vx_body - r*y: the yaw term
        # is about +/-0.36 m/s at r = 0.42 rad/s and cannot be neglected
        vx_contact = vx_body.unsqueeze(1) - s[:, 5:6] * y_pos
        s[:, 6:10] = (1.0 + slip) * vx_contact / radius
        return s

    def init_wheels_mask(self, s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, y_pos, radius, _ = self.geometry()
        psi = s[mask, 2]
        vx_body = torch.cos(psi) * s[mask, 3] + torch.sin(psi) * s[mask, 4]
        vx_contact = vx_body.unsqueeze(1) - s[mask, 5:6] * y_pos[mask]
        s[mask, 6:10] = vx_contact / radius[mask]
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

    def _front_steer(self, delta):
        """Left/right front wheel angles from the nominal steer (radians).

        ``ackermann_scale`` is per env: 0 keeps the front wheels parallel, 1
        applies the exact Ackermann geometry about the rear axle centre, and
        values in between blend the two (1.2 extrapolates past exact).  The
        geometry is built from magnitudes and the sign is applied afterwards,
        because a signed radius would make the inner angle flip sign on the
        right-hand side.
        """
        if delta.dim() == 1:
            delta = delta.unsqueeze(1)
        L = self.col("lF").unsqueeze(1) + self.col("lR").unsqueeze(1)
        half = 0.5 * self.col("Tf").unsqueeze(1)
        ack = self.ackermann_scale
        mag = delta.abs()
        small = mag < 1e-4
        safe = torch.where(small, torch.full_like(mag, 1e-4), mag)
        radius = L / torch.tan(safe)
        # keep the inner radius positive and away from zero at large steer
        r_in = torch.clamp(radius - half, min=0.2 * L)
        r_out = torch.clamp(radius + half, min=0.2 * L)
        d_in = torch.atan(L / r_in) * torch.sign(delta)
        d_out = torch.atan(L / r_out) * torch.sign(delta)
        inner = delta + ack * (d_in - delta)
        outer = delta + ack * (d_out - delta)
        # left is the inner wheel for a left turn (delta > 0)
        left = torch.where(delta > 0, inner, outer)
        right = torch.where(delta > 0, outer, inner)
        left = torch.where(small, delta, left)
        right = torch.where(small, delta, right)
        return torch.cat([left, right], dim=1)

    def _wheel_kinematics(self, s, delta, geometry):
        """Wheel-frame contact velocities and front-steer rotation matrices."""
        x_pos, y_pos, radius, _ = geometry
        vxw, vyw, r = s[:, 3], s[:, 4], s[:, 5]
        c, sn = torch.cos(s[:, 2]), torch.sin(s[:, 2])
        vxb = c * vxw + sn * vyw
        vyb = -sn * vxw + c * vyw

        vx_b = vxb.unsqueeze(1) - r.unsqueeze(1) * y_pos
        vy_b = vyb.unsqueeze(1) + r.unsqueeze(1) * x_pos
        # ``delta`` is (n, 2): the left and right front wheel angles.  The rear
        # wheels are not steered.
        if delta.dim() == 1:
            delta = delta.unsqueeze(1).expand(-1, 2)
        cd = torch.ones_like(vx_b)
        sd = torch.zeros_like(vx_b)
        cd[:, :2] = torch.cos(delta)
        sd[:, :2] = torch.sin(delta)
        vx_l = vx_b.clone()
        vy_l = vy_b.clone()
        vx_l[:, :2] = cd[:, :2] * vx_b[:, :2] + sd[:, :2] * vy_b[:, :2]
        vy_l[:, :2] = -sd[:, :2] * vx_b[:, :2] + cd[:, :2] * vy_b[:, :2]
        return vx_l, vy_l, cd, sd

    def _wheel_forces(self, s, delta, fz, geometry,
                      kappa_power_override=None, alpha_override=None):
        """Combined-slip isotropic tire forces in wheel and body frames.

        Returns a dict with the per-wheel forces plus the slip diagnostics.
        """
        _, _, radius, _ = geometry
        vx_l, vy_l, cd, sd = self._wheel_kinematics(s, delta, geometry)

        B, C, D, E = self.col("B").unsqueeze(1), self.col("C").unsqueeze(1), \
            self.col("D").unsqueeze(1), self.col("E").unsqueeze(1)
        axle_scale = torch.where(self.wheel_is_front.unsqueeze(0) > 0.5,
                                 self.col("tyre_scale_front").unsqueeze(1),
                                 self.col("tyre_scale_rear").unsqueeze(1))
        d_eff = D * axle_scale * self.tyre_scale
        d_surface = d_eff
        if float(self.col("load_sensitivity").abs().max().item()) > 0.0:
            static_ref = (self.col("m") * self.col("g") * 0.25).unsqueeze(1)
            fzf = self.col("fz_ref_front").unsqueeze(1)
            fzr = self.col("fz_ref_rear").unsqueeze(1)
            fz_ref = torch.cat([fzf, fzf, fzr, fzr], dim=1)
            fz_ref = torch.where(fz_ref > 0.0, fz_ref, static_ref.expand(-1, 4)).clamp(min=1.0)
            sens = self.col("load_sensitivity").unsqueeze(1)
            if self.load_sensitivity_model == "power":
                d_eff = d_eff * torch.clamp(fz / fz_ref, min=0.1, max=4.0).pow(-sens)
            else:
                d_eff = d_eff * torch.clamp(1.0 - sens * (fz / fz_ref - 1.0), 0.4, 1.4)

        vf = self.col("slip_v_floor").unsqueeze(1)
        vfa = self.col("slip_alpha_v_floor").unsqueeze(1)
        denom = torch.maximum(vx_l.abs(), vf)
        denom_a = torch.maximum(vx_l.abs(), vfa)

        omega = s[:, 6:10]
        sx_raw = (vx_l - omega * radius) / denom
        sy_raw = vy_l / denom_a
        alpha_raw = torch.atan(sy_raw)
        # Offline identification may inject CarSim's reported lagged tire-slip
        # states to separate the static force law from slip-state dynamics.
        # Production calls leave both overrides at None and pay no tensor cost.
        if kappa_power_override is not None:
            sx_raw = -kappa_power_override
        if alpha_override is not None:
            alpha_raw = alpha_override
            sy_raw = torch.tan(alpha_raw).clamp(-5.0, 5.0)
        if self.tire_relaxation:
            ready = self._relax_initialized.unsqueeze(1)
            sx = (torch.where(ready, self._sx_relaxed, sx_raw)
                  if self.relax_kappa else sx_raw)
            alpha_force = (torch.where(ready, self._alpha_relaxed, alpha_raw)
                           if self.relax_alpha else alpha_raw)
            sy = torch.tan(alpha_force).clamp(-5.0, 5.0)
        else:
            sx, sy, alpha_force = sx_raw, sy_raw, alpha_raw
        slip_raw = torch.hypot(sx_raw, sy_raw)
        # The Magic Formula keeps falling past a very large slip (for C < 2 it
        # tends to D*sin(C*pi/2)).  Real tires show a plateau; clamp the evaluated
        # slip so a spinning wheel keeps a physically sensible force instead of
        # losing almost all grip at 100x the peak slip.
        # --- separate longitudinal and lateral pure-slip curves, then a friction
        # ellipse.  The isotropic form mu(hypot(sx, sy)) with the force direction from
        # atan2(sy, sx) gives a tyre only about 0.48 of whose force is lateral at
        # kx = 0.55, and yaw balance needs roughly 55 % of the lateral force from the
        # REAR (lF 1.455 against lR 1.195).  With the total already reduced by the
        # falling Magic Formula at high longitudinal slip the rear cannot deliver that
        # share while spinning, which is why the trim solver finds no power-drift state
        # in either sign convention.  A real tyre's lateral and longitudinal peaks occur
        # at very different slips, so they are modelled separately here.
        _sx_abs = sx.abs()
        _sy_abs = sy.abs()
        # Hold longitudinal force at the final measured CarSim carpet point.
        # The fitted table ends at |kappa|=0.51 (where force is about 93 % of its
        # first peak) and is flat-extrapolated by CarSim.  Evaluating the compact
        # Magic Formula beyond that point would invent an unmeasured force decay
        # and made harder throttle spuriously slow the low-mu car.
        _sx_eval = torch.clamp(_sx_abs, max=self.slip_force_plateau)
        bx, cx, ex = (self.col(k).unsqueeze(1) for k in ("bx", "cx", "ex"))
        by, cy, ey = (self.col(k).unsqueeze(1) for k in ("by", "cy", "ey"))
        _mu_ref = self.col("tire_mu_reference").unsqueeze(1).clamp(min=1e-3)
        _grip_ratio = torch.clamp(d_surface / _mu_ref, min=0.15, max=2.0)
        _stiff_scale_x = _grip_ratio.pow(
            -self.col("tire_stiffness_grip_exponent_x").unsqueeze(1))
        _stiff_scale_y = _grip_ratio.pow(
            -self.col("tire_stiffness_grip_exponent_y").unsqueeze(1))
        bx = bx * _stiff_scale_x
        by = by * _stiff_scale_y
        _shape_x, _dshape_x = self._mf(_sx_eval, bx, cx, 1.0, ex)
        _dshape_x = torch.where(_sx_abs > self.slip_force_plateau,
                                torch.zeros_like(_dshape_x), _dshape_x)
        _sy_plateau = math.tan(self.slip_alpha_force_plateau)
        _sy_eval = torch.clamp(_sy_abs, max=_sy_plateau)
        _shape_y, _dshape_y = self._mf(_sy_eval, by, cy, 1.0, ey)
        _dshape_y = torch.where(_sy_abs > _sy_plateau,
                                torch.zeros_like(_dshape_y), _dshape_y)
        mux_peak = d_eff                      # D_eff is the longitudinal friction level
        # Lateral peak is per AXLE: a rear-drive car that can hold a slide needs the rear
        # to saturate laterally before the front, and the measured behaviour was a 0.4 rad
        # sideslip decaying to zero within 3 s precisely because the rear had MORE lateral
        # grip than the front (the 1.40 rear longitudinal scale was bleeding into it).
        _ratio = torch.tensor([self.mu_y_ratio, self.mu_y_ratio,
                               self.mu_y_ratio_rear, self.mu_y_ratio_rear],
                              dtype=d_eff.dtype, device=d_eff.device)
        muy_peak = d_eff * _ratio             # lateral peak relative to longitudinal
        _fx0 = -torch.sign(sx) * mux_peak * fz * _shape_x
        _fy0 = -torch.sign(sy) * muy_peak * fz * _shape_y
        if self.combined_slip_model == "isotropic":
            # Exact structural replica of XCAR's simplified isotropic tire.
            # One shared Magic-Formula curve acts on the magnitude of the
            # combined slip vector; no separate Fx/Fy peaks or friction ellipse
            # are introduced.  ``slip_max`` is only a numerical far-tail guard
            # (the original XCAR operating range never approaches its default 5).
            _slip_force = torch.hypot(sx, sy)
            _slip_eval = torch.clamp(_slip_force, max=self.slip_max)
            _mu_iso, _dmu_iso = self._mf(_slip_eval, B, C, d_eff, E)
            _dmu_iso = torch.where(_slip_force > self.slip_max,
                                   torch.zeros_like(_dmu_iso), _dmu_iso)
            _inv_slip = torch.where(
                _slip_force > 1e-9, 1.0 / _slip_force.clamp(min=1e-9),
                torch.zeros_like(_slip_force))
            fx_wheel = -fz * _mu_iso * sx * _inv_slip
            fy_wheel = -fz * _mu_iso * sy * _inv_slip
            mux = fx_wheel / fz.clamp(min=1.0)
            muy = fy_wheel / fz.clamp(min=1.0)
            mu = _mu_iso
            dmu = _dmu_iso
        elif self.combined_slip_model == "compact_transition":
            # CarSim's tabular tyre does not apply one common ellipse scale to
            # the two independently evaluated pure forces.  It transitions
            # between normalized and theoretical combined-slip vectors.  A
            # fit on real drift data gives blend ~=0.49 for this tyre.
            _zx = _sx_abs / max(self.kappa_peak, 1e-6)
            _zy = alpha_force.abs() / max(self.alpha_peak, 1e-6)
            _rho_n = torch.hypot(_zx, _zy).clamp(min=1e-9)
            _eq_x_n = torch.clamp(_rho_n * self.kappa_peak,
                                  max=self.slip_force_plateau)
            _eq_y_n = torch.tan(torch.clamp(_rho_n * self.alpha_peak,
                                            max=self.slip_alpha_force_plateau))
            _shape_x_n, _ = self._mf(_eq_x_n, bx, cx, 1.0, ex)
            _shape_y_n, _ = self._mf(_eq_y_n, by, cy, 1.0, ey)
            _fx_n = (-torch.sign(sx) * mux_peak * fz * _shape_x_n *
                     _zx / _rho_n)
            _fy_n = (-torch.sign(sy) * muy_peak * fz * _shape_y_n *
                     _zy / _rho_n)

            _tx, _ty = _sx_abs, _sy_abs
            _rho_t = torch.hypot(_tx, _ty).clamp(min=1e-9)
            _eq_x_t = torch.clamp(_rho_t, max=self.slip_force_plateau)
            _eq_y_t = torch.clamp(_rho_t, max=_sy_plateau)
            _shape_x_t, _ = self._mf(_eq_x_t, bx, cx, 1.0, ex)
            _shape_y_t, _ = self._mf(_eq_y_t, by, cy, 1.0, ey)
            _fx_t = (-torch.sign(sx) * mux_peak * fz * _shape_x_t *
                     _tx / _rho_t)
            _fy_t = (-torch.sign(sy) * muy_peak * fz * _shape_y_t *
                     _ty / _rho_t)
            _blend = self.col("combined_slip_blend").unsqueeze(1).clamp(0.0, 1.0)
            fx_wheel = (1.0 - _blend) * _fx_n + _blend * _fx_t
            fy_wheel = (1.0 - _blend) * _fy_n + _blend * _fy_t
        elif self.combined_slip_model == "ellipse":
            # ellipse projection: scale both together when combined demand exceeds 1
            _q = torch.sqrt((_fx0 / mux_peak.clamp(min=1e-3) / fz.clamp(min=1.0)) ** 2
                            + (_fy0 / muy_peak.clamp(min=1e-3) / fz.clamp(min=1.0)) ** 2)
            _scale = torch.where(_q > 1.0, 1.0 / _q.clamp(min=1e-6), torch.ones_like(_q))
            fx_wheel = _scale * _fx0
            fy_wheel = _scale * _fy0
        else:
            raise ValueError("unknown combined_slip_model %r" % self.combined_slip_model)
        if self.combined_slip_model != "isotropic":
            mux = fx_wheel / fz.clamp(min=1.0)
            muy = fy_wheel / fz.clamp(min=1.0)
            mu = torch.hypot(mux, muy)
            dmu = torch.zeros_like(mu)
        # Tire slip angle is the wheel-frame velocity angle, not the direction
        # of the combined (longitudinal,lateral) slip vector.  atan2(sy,sx)
        # jumps to +/-90 deg whenever longitudinal slip is zero and cannot be
        # compared with CarSim Alpha_*.
        alpha = alpha_raw

        fx_body = fx_wheel.clone()
        fy_body = fy_wheel.clone()
        fx_body[:, :2] = cd[:, :2] * fx_wheel[:, :2] - sd[:, :2] * fy_wheel[:, :2]
        fy_body[:, :2] = sd[:, :2] * fx_wheel[:, :2] + cd[:, :2] * fy_wheel[:, :2]

        # |dFx/domega| used by the semi-implicit wheel update.  The tangent alone
        # under-estimates the stiffness when the step crosses toward zero slip
        # (the slip ratio is very sensitive at low speed); take the larger of the
        # local tangent and the secant to the zero-force point so the implicit
        # step cannot overshoot the rolling equilibrium and chatter.
        with torch.no_grad():
            if self.combined_slip_model == "isotropic":
                # d[mu(s)*sx/s]/d(sx), with s=hypot(sx,sy).  This preserves
                # wheel-spin stiffness when lateral slip is present.
                _s = _slip_force.clamp(min=1e-6)
                _local = (_dmu_iso.abs() * sx.square() / _s.square()
                          + _mu_iso.abs() * sy.square() / _s.pow(3))
                kappa = fz * _local * radius / denom
            else:
                # longitudinal stiffness of the separate-curve A/B models
                kappa = (fz * mux_peak * _dshape_x.abs() * radius / denom)
            secant = fx_wheel.abs() / torch.clamp(sx.abs(), min=1e-3) * radius / denom
            kappa = torch.maximum(kappa, secant).clamp(min=0.0)

        return {
            "fx_wheel": fx_wheel, "fy_wheel": fy_wheel,
            "fx_body": fx_body, "fy_body": fy_body,
            "fz": fz, "sx": sx_raw, "sy": sy_raw, "slip": slip_raw, "mu": mu,
            # Canonical power-slip convention shared with CarSim: positive means
            # wheel circumferential speed exceeds contact-patch longitudinal speed.
            "kappa_power": -sx_raw,
            "alpha": alpha, "sx_force": sx, "alpha_force": alpha_force,
            "mux": mux, "muy": muy,
            "dFx_domega": kappa, "vx_wheel": vx_l, "vy_wheel": vy_l,
        }

    @staticmethod
    def _interp_flat(x, grid, values):
        x = x.clamp(grid[0], grid[-1])
        i = torch.bucketize(x, grid).clamp(1, grid.numel() - 1)
        x0, x1 = grid[i - 1], grid[i]
        y0, y1 = values[i - 1], values[i]
        return y0 + (y1 - y0) * (x - x0) / torch.clamp(x1 - x0, min=1e-8)

    def _update_tire_relaxation(self, s, delta, fz, h):
        """Advance CarSim OPT_TIRE_LAG_KAPPA=2 / ALPHA=1 states.

        Both modes transition to instantaneous slip when tau < 12*TSTEP;
        Kappa mode 2 additionally becomes instantaneous whenever |kappa| is
        decreasing.  This distinction is why an unconditional first-order lag
        produced catastrophic wheel spin in the rejected SI experiment.
        """
        if not self.tire_relaxation:
            return
        geometry = self.geometry()
        vx_l, vy_l, _, _ = self._wheel_kinematics(s, delta, geometry)
        radius = geometry[2]
        denom = torch.maximum(vx_l.abs(), self.col("slip_v_floor").unsqueeze(1))
        denom_a = torch.maximum(vx_l.abs(), self.col("slip_alpha_v_floor").unsqueeze(1))
        sx_raw = (vx_l - s[:, 6:10] * radius) / denom
        alpha_raw = torch.atan(vy_l / denom_a)
        new = ~self._relax_initialized
        if bool(new.any()):
            self._sx_relaxed[new] = sx_raw[new]
            self._alpha_relaxed[new] = alpha_raw[new]
            self._relax_initialized[new] = True
        old = ~new
        if not bool(old.any()) or h <= 0.0:
            return
        static_ref = (self.col("m") * self.col("g") * 0.25).unsqueeze(1)
        fzf = self.col("fz_ref_front").unsqueeze(1)
        fzr = self.col("fz_ref_rear").unsqueeze(1)
        fz_ref = torch.cat([fzf, fzf, fzr, fzr], dim=1)
        fz_ref = torch.where(fz_ref > 0.0, fz_ref,
                             static_ref.expand(-1, 4)).clamp(min=1.0)
        load = (fz / fz_ref).clamp(0.0, 2.0)
        load_x = 0.2 + 0.8 * load
        load_y = torch.where(load <= 1.0, 0.1 + 0.9 * load,
                             1.0 + 0.5805 * (load - 1.0))
        lx = (radius / 5.0) * self._interp_flat(
            sx_raw.abs(), self.relax_x_grid, self.relax_x_factor) * load_x
        ly = (2.0 * radius) * self._interp_flat(
            torch.rad2deg(alpha_raw.abs()), self.relax_y_grid,
            self.relax_y_factor) * load_y
        # CarSim VLOW_KAPPA/VLOW_ALPHA are 2 km/h for these tires.
        speed = vx_l.abs().clamp(min=2.0 / 3.6)
        tau_x = lx.clamp(min=1e-4) / speed
        tau_y = ly.clamp(min=1e-4) / speed
        ax = 1.0 - torch.exp(-float(h) / tau_x)
        ay = 1.0 - torch.exp(-float(h) / tau_y)
        # The source says "transition" rather than exposing its blend curve.
        # A hard switch at the documented threshold is deterministic and avoids
        # inventing another calibration parameter.
        ax = torch.where(tau_x < 12.0 * float(h), torch.ones_like(ax), ax)
        ay = torch.where(tau_y < 12.0 * float(h), torch.ones_like(ay), ay)
        decreasing = sx_raw.abs() < self._sx_relaxed.abs()
        ax = torch.where(decreasing, torch.ones_like(ax), ax)
        select = old.unsqueeze(1)
        self._sx_relaxed = torch.where(
            select, self._sx_relaxed + ax * (sx_raw - self._sx_relaxed),
            self._sx_relaxed)
        self._alpha_relaxed = torch.where(
            select, self._alpha_relaxed + ay * (alpha_raw - self._alpha_relaxed),
            self._alpha_relaxed)

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
    def _motor_efficiency(self, rpm, torque):
        """Bilinear lookup/extrapolation of the CarSim efficiency carpet.

        The source function is ``2D_LINEAR`` (interpolation *and extrapolation*).
        CarSim then floors efficiency at 0.02.  Clamping torque to the last
        305-Nm column hid the large electrical draw at the 400--2400 Nm drift
        operating points even though it did not change mechanical shaft power.
        """
        r = rpm.clamp(self.eff_rpm[0], self.eff_rpm[-1])
        q = torque.abs().clamp(min=self.eff_torque[0])
        ir = torch.bucketize(r, self.eff_rpm).clamp(1, len(EV_EFF_RPM) - 1)
        iq = torch.bucketize(q, self.eff_torque).clamp(1, len(EV_EFF_TORQUE) - 1)
        r0, r1 = self.eff_rpm[ir - 1], self.eff_rpm[ir]
        q0, q1 = self.eff_torque[iq - 1], self.eff_torque[iq]
        wr = (r - r0) / (r1 - r0).clamp(min=1e-6)
        wq = (q - q0) / (q1 - q0).clamp(min=1e-6)
        e00, e01 = self.eff_table[ir - 1, iq - 1], self.eff_table[ir - 1, iq]
        e10, e11 = self.eff_table[ir, iq - 1], self.eff_table[ir, iq]
        value = (e00 * (1.0 - wq) + e01 * wq) * (1.0 - wr) + \
            (e10 * (1.0 - wq) + e11 * wq) * wr
        value = value * self.col("motor_efficiency")
        return value.clamp(0.02, 0.98)

    def _battery_limited_motor_torque(self, rpm, omega, requested):
        """Fast lookup of the precomputed analytic battery torque envelope."""
        voc = self.col("battery_voc_v")
        vmin = self.col("battery_min_voltage_v")
        resistance = self.col("battery_resistance_ohm").clamp(min=1e-6)
        p_battery_max = vmin * torch.clamp(voc - vmin, min=0.0) / resistance
        p_battery_max = p_battery_max * self.col("battery_power_limit_scale")
        p_nominal = 720.0 * (936.936 - 720.0) / 0.12

        r = rpm.clamp(self.battery_limit_rpm[0], self.battery_limit_rpm[-1])
        idx = torch.bucketize(r, self.battery_limit_rpm).clamp(
            1, self.battery_limit_rpm.numel() - 1)
        r0, r1 = self.battery_limit_rpm[idx - 1], self.battery_limit_rpm[idx]
        t0 = self.battery_limit_torque[idx - 1]
        t1 = self.battery_limit_torque[idx]
        ceiling = t0 + (t1 - t0) * (r - r0) / (r1 - r0)
        # First-order scaling is sufficient for deliberately broad DR; the
        # nominal curve itself retains the exact nonlinear efficiency shape.
        scale = (p_battery_max / p_nominal) * self.col("motor_efficiency")
        return torch.minimum(requested, ceiling * scale.clamp(0.25, 2.0))

    def _motor_torque(self, omega_pair, throttle, brake):
        """Axle torque from the pedals (total for the axle seen by the differential)."""
        gear = self.col("gear_ratio")
        diff_eff = self.col("drive_efficiency")
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

        # CarSim's pedal table is driver DEMAND POWER
        # (PWR_DRV_THROTTLE_COEFFICIENT=384 kW), not a percentage of maximum
        # motor torque.  At low speed even a small positive pedal can therefore
        # reach the 2400 N m torque ceiling; the old q*t_max mapping made 10 %
        # pedal roughly three times too weak in the 0-3 s launch transient.
        q = torch.clamp(throttle - brake, min=0.0)
        q = q * torch.clamp(
            self.col("throttle_power_gain") - self.col("throttle_power_rolloff") * q,
            min=0.0, max=2.0)
        # CarSim internal-channel evidence fixes an important interpretation:
        # PwrDrHEV/PwrMotDm is requested MECHANICAL motor-shaft power.  Motor
        # efficiency is applied downstream when computing battery power; it
        # must not reduce shaft torque here.  The old efficiency multiplication
        # under-delivered 15--60 % torque in the powered-slide range and made
        # rear Kappa collapse.  Example: q=.8 at 3723 rpm commands about 788 Nm
        # (=307.2 kW/omega) even though CarSim reports efficiency about .375.
        t_power_demand = q * p_max / torch.clamp(wm.abs(), min=1.0)
        t_motor = torch.minimum(t_map, t_power_demand)
        # zero-pedal driveline drag (EV regen / in-gear coast), axle total
        coast = self.col("coast_torque_nm") * (1.0 - torch.clamp(throttle + brake, max=1.0)) \
            * torch.tanh(gear * wm / 30.0)
        # MMOTOR_MAX is mechanical output torque and must not be reduced by the
        # electrical efficiency carpet.  Final-drive loss applies downstream.
        axle = (t_motor - coast) * gear * diff_eff
        return axle, t_motor, wm

    def _lsd(self, axle_torque, omega_pair):
        """Split axle torque with a compact generalized-viscous LSD.

        CarSim specifies ``(left rpm - right rpm) -> (left torque - right
        torque)``.  Its nominal 31-point curve is, to the printed precision,
        exactly an odd square-root law.  We retain that physical structure
        rather than carrying the lookup table into the GPU plant.  ``coupling``
        is the positive torque shifted from the faster left wheel to the right
        wheel, so ``T_L-T_R = -2*coupling``.
        """
        cap = self.col("lsd_tau_max")
        dw = omega_pair[:, 0] - omega_pair[:, 1]
        if self.lsd_model in ("power_law", "generalized_viscous"):
            dn_rpm = dw * (60.0 / (2.0 * math.pi))
            ratio = torch.abs(dn_rpm) / max(self.lsd_ref_rpm, 1e-6)
            torque_diff_mag = self.col("lsd_torque_ref") * torch.pow(
                ratio, self.col("lsd_exponent").clamp(0.05, 1.50))
            # Faster side receives less torque: sign(delta T)=-sign(delta rpm).
            torque_diff = -torch.sign(dn_rpm) * torque_diff_mag
            coupling = -0.5 * torque_diff
        else:
            coupling = self.col("lsd_kd") * dw
        coupling = torch.clamp(coupling, min=-cap, max=cap)
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
        delta = self._front_steer(delta)

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
        # ax, ay are BODY accelerations -> rotate once to the world frame.
        dvx = c * ax - sn * ay
        dvy = sn * ax + c * ay
        # s[:, 3:5] is already the WORLD velocity (the spawn sets
        # (v cos(beta+psi), v sin(beta+psi))), so the position derivative must NOT
        # rotate it again.  Rotating twice made the car travel along a direction
        # rotated by its own heading, i.e. its path was roughly twice as curved as
        # the dynamics implied.
        vxw, vyw = s[:, 3], s[:, 4]
        dx = vxw
        dy = vyw

        # --- wheel torques ---
        drive_torque = torch.zeros_like(omega)
        if mode == "pedals":
            axle, t_motor, wm = self._motor_torque(
                torch.stack([omega[:, 2], omega[:, 3]], dim=1), throttle, brake)
            t_l, t_r, coupling = self._lsd(axle, torch.stack([omega[:, 2], omega[:, 3]], dim=1))
            if self.drivetrain in ("rwd", "awd", "iwd"):
                # "iwd" is the compatibility name used by the continuous-drift
                # task.  This plant has a single rear motor/LSD, so it is RWD
                # here: it used to send only half the axle torque rearward and
                # silently discard the other half (awd is what splits it).
                share = 0.5 if self.drivetrain == "awd" else 1.0
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
        n_sub = max(int(substeps), 1)
        ax_sum = None
        ay_sum = None
        for _ in range(n_sub):
            deriv, extras = self.derivatives(s, command, mode=mode)
            # The accelerometer reading is the average over the control period,
            # not the instantaneous value of the last physics substep.
            if ax_sum is None:
                ax_sum = extras["ax"].clone()
                ay_sum = extras["ay"].clone()
            else:
                ax_sum += extras["ax"]
                ay_sum += extras["ay"]
            s[:, :6] = s[:, :6] + h * deriv[:, :6]
            # Constrained backward-Euler solve for the four wheel speeds.
            #
            # The previous update was a linearised one-step formula with a clamp, which
            # cannot represent a locked wheel: a 240 Nm brake against 0.17 kg m^2 removes
            # 14 rad/s in one 10 ms substep, so the wheel always overshoots negative, and
            # whatever projection ran afterwards either restored it to rolling speed (and
            # thereby erased all braking slip and force) or let it run away.  Measured
            # consequence: every pedal above 0.10 gave an identical -0.01 m/s per second
            # while CarSim removes 2.27 m/s at 0.20.
            #
            # Solve, per wheel and per substep, with body, loads, steering and drivetrain
            # torque frozen:
            #
            #   G(w) = J (w - w_old) - h [ T_d - T_b - T_rr - R Fx(w) ]  = 0
            #
            # G is monotone once the tyre force is clamped at its FIRST peak, so a bracket
            # and a bisection are enough.  G(0) >= 0 means even the maximum brake cannot
            # hold a positive wheel speed: the wheel locks at zero, which is exactly what
            # a brake produces, and Fx is then evaluated on the sliding plateau.
            w_old = s[:, 6:10].clone()
            delta_w = extras["delta"]
            fz_w = extras["fz"]
            geom = self.geometry()
            radius_w = geom[2]
            j_w = extras["wheel_inertia"]
            # roll_torque is not exported, so recover it from the net torque the
            # derivative already assembled: net = drive - brake - roll - R*fx
            # net = drive - brake - roll - R*fx  =>  drive - brake - roll = net + R*fx
            tau = (extras["wheel_net_torque"] + radius_w * extras["fx_wheel"])
            # tau = drive - brake - roll.  The tire force is implicit below, and
            # the electric drive must be implicit as well because its power limit
            # changes sharply with motor speed.  Freezing drive torque produced a
            # two-cycle: rear wheel surface speed alternated between rolling and
            # 10-100 m/s on adjacent substeps.
            resist = extras["drive_torque"] - tau  # brake + rolling resistance

            def drive_of(omega):
                if mode != "pedals":
                    return extras["drive_torque"]
                axle_i, _, _ = self._motor_torque(
                    torch.stack([omega[:, 2], omega[:, 3]], dim=1),
                    command[:, 1], command[:, 2])
                tl_i, tr_i, _ = self._lsd(
                    axle_i, torch.stack([omega[:, 2], omega[:, 3]], dim=1))
                out = torch.zeros_like(omega)
                if self.drivetrain in ("rwd", "iwd"):
                    out[:, 2], out[:, 3] = tl_i, tr_i
                elif self.drivetrain == "fwd":
                    out[:, 0], out[:, 1] = tl_i, tr_i
                elif self.drivetrain == "awd":
                    out[:, 0] = axle_i * 0.25
                    out[:, 1] = axle_i * 0.25
                    out[:, 2] = tl_i * 0.5
                    out[:, 3] = tr_i * 0.5
                return out

            def g_of(omega):
                trial = s.clone()
                trial[:, 6:10] = omega
                f = self._wheel_forces(trial, delta_w, fz_w, geom)
                # G = J(w-w_old) - h[drive(w)-brake-roll-R*Fx(w)]
                return j_w * (omega - w_old) - h * (
                    drive_of(omega) - resist - radius_w * f["fx_wheel"])

            g_lo = g_of(torch.zeros_like(w_old))
            locked = g_lo >= 0.0
            lo = torch.zeros_like(w_old)
            hi = (2.0 * w_old + 10.0).clamp(min=10.0)
            for _ in range(self.wheel_solver_bracket_iters):
                g_hi = g_of(hi)
                need = g_hi < 0.0
                if not bool(need.any()):
                    break
                hi = torch.where(need, 2.0 * hi + 10.0, hi)
            for _ in range(self.wheel_solver_bisection_iters):
                mid = 0.5 * (lo + hi)
                go_right = g_of(mid) < 0.0
                lo = torch.where(go_right, mid, lo)
                hi = torch.where(go_right, hi, mid)
            omega_new = torch.where(locked, torch.zeros_like(w_old), 0.5 * (lo + hi))
            s[:, 6:10] = omega_new.clamp(0.0, 2500.0)
            s[:, 2] = torch.atan2(torch.sin(s[:, 2]), torch.cos(s[:, 2]))
            self._update_tire_relaxation(s, delta_w, fz_w, h)
        # ``extras`` came from the beginning of the final physics substep while
        # ``s`` now contains its updated body and wheel speeds.  Refresh the cheap
        # slip diagnostics so rewards and logged Kappa are not one substep stale.
        vx_l, vy_l, _, _ = self._wheel_kinematics(s, extras["delta"], geom)
        denom = torch.maximum(vx_l.abs(), self.col("slip_v_floor").unsqueeze(1))
        denom_a = torch.maximum(
            vx_l.abs(), self.col("slip_alpha_v_floor").unsqueeze(1))
        sx_now = (vx_l - s[:, 6:10] * radius_w) / denom
        sy_now = vy_l / denom_a
        extras.update({
            "vx_wheel": vx_l, "vy_wheel": vy_l,
            "sx": sx_now, "sy": sy_now,
            "slip": torch.hypot(sx_now, sy_now),
            "kappa_power": -sx_now,
            "alpha": torch.atan(sy_now),
        })
        if self.tire_relaxation:
            extras["sx_force"] = self._sx_relaxed.clone()
            extras["alpha_force"] = self._alpha_relaxed.clone()
        extras["ax"] = ax_sum / n_sub
        extras["ay"] = ay_sum / n_sub
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
