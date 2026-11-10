# FastPlant — full-size RWD EV vehicle model

`xcar-simulation/fast_plant.py` is the plant used for RL training. It keeps the
simplicity of XCAR (one isotropic combined-slip Pacejka curve, planar body,
vectorized over tens of thousands of environments on one GPU) and adds the
intermediate modules needed to line the interface up with CarSim.

Everything below is a *fast plant*, not a Magic-Formula 6.2 replacement. The
full MF formula is deliberately not implemented.

## Interface

| | |
|---|---|
| state (10) | `x, y, psi, vx, vy, r, omega_FL, omega_FR, omega_RL, omega_RR` |
| action (pedals) | `steer [rad], throttle [0,1], brake [0,1]` — mutually exclusive |
| action (legacy) | `steer, v_FL, v_FR, v_RL, v_RR` (wheel surface speed, m/s) |
| timing | control `dt` (default **20 Hz**), physics `physics_dt` (default **100 Hz**), command zero-order held over `dt/physics_dt` substeps |
| wheel order | `[FL, FR, RL, RR]`, `x` forward, `y` left, positive yaw = left turn |
| frames | `vx, vy` are **world frame** (same convention as XCAR); tyre forces are computed in the body/wheel frames |

`vx, vy` in the world frame is not an approximation: with body-frame forces and
a heading rotation, `d(v_world)/dt = R(psi) @ [Fx/m, Fy/m]`, i.e. the
centripetal terms are already inside the rotation.

## Equations

**Tyre (XCAR isotropic, unchanged)** — one combined slip drives one friction
value, so the friction circle is exact by construction:

```
sx = (vx_wheel - R*omega) / max(|vx_wheel|, slip_v_floor)
sy = vy_wheel            / max(|vx_wheel|, slip_alpha_v_floor)
s  = min(hypot(sx, sy), slip_max)
mu = D * sin(C * atan(B*s - E*(B*s - atan(B*s))))
alpha = atan2(sy, sx);  fx = -cos(alpha)*mu*Fz;  fy = -sin(alpha)*mu*Fz
```

`D` is the peak friction coefficient (dimensionless) and can be scaled per axle
(`tyre_scale_front`, `tyre_scale_rear`) and per environment
(`plant.tyre_scale`, used by the disturbance mode). `slip_max` keeps the force
on a high-slip plateau instead of letting the Magic Formula decay to
`D*sin(C*pi/2)` at absurd slip ratios.

**Load transfer (longitudinal + lateral, with aero coupling)** — solved
quasi-statically with two relaxed passes per derivative (the previous step's
accelerations are the initial guess):

```
N   = m*g - Fz_aero
Fzf = clamp((lr*N + My_aero - h*(m*ax - Fx_aero)) / (lf+lr), 0, N);  Fzr = N - Fzf
Q   = h*(m*ay - Fy_aero) + Mx_aero
dFzf = clamp(roll_share_front * Q / tf, -Fzf/2, Fzf/2)
dFzr = clamp((1-roll_share_front) * Q / tr, -Fzr/2, Fzr/2)
Fz  = [Fzf/2 - dFzf, Fzf/2 + dFzf, Fzr/2 - dFzr, Fzr/2 + dFzr]
```

**Aerodynamics** — one pressure center `P = (xp, 0, zp)` relative to the CG,
separate longitudinal and lateral drag and lift:

```
U = vx_body - wind_x_body;   V = vy_body + r*xp - wind_y_body
Fx = -0.5*rho*Ax*Cdx*U*|U|;  Fy = -0.5*rho*Ay*Cdy*V*|V|
Fz =  0.5*rho*(Ax*Clx*U^2 + Ay*Cly*V^2)          # Clx, Cly < 0 = downforce
Mx = -zp*Fy;  My = zp*Fx - xp*Fz;  Mz = xp*Fy
```

**Powertrain** — one motor behind a single reduction, torque envelope from the
`motor` table, hard power limit, then a viscous limited-slip differential on
the driven axle:

```
wm   = gear_ratio * (omega_RL + omega_RR)/2;  rpm = |wm|*60/(2*pi)
Tmax = min(table(rpm), max_power_w / max(|wm|,1))
Tq   = clamp(throttle - brake, 0, 1) * Tmax - coast(rpm)
T_axle = Tq * gear_ratio * drive_efficiency
coupling = clamp(lsd_kd * (omega_RL - omega_RR), -lsd_tau_max, +lsd_tau_max)
T_RL = T_axle/2 - coupling;   T_RR = T_axle/2 + coupling
```

**Wheel spin, brakes, rolling resistance**

```
T_brake_i = brake * p_max * Nm_per_MPa_i * tanh(omega_i / 0.5)
domega_i  = (T_drive_i - T_brake_i - c_rr*Fz_i*R_i*tanh(omega_i/0.5) - R_i*Fx_i) / Iw_i
```

**Body**

```
m(ax) = sum Fx_body + Fx_aero,  m(ay) = sum Fy_body + Fy_aero
Iz*rdot = sum(x_i*Fy_i - y_i*Fx_i) + Mz_aero
```

## Numerics

* Body DOF: explicit Euler at 100 Hz.
* Wheel DOF: semi-implicit (trapezoidal) update linearised on the tyre
  longitudinal stiffness, `J_eff = J + 0.5*h*R*dFx/domega`. The stiffness is the
  larger of the local tangent and the secant to the zero-force point, which
  prevents chattering around the rolling equilibrium at low speed.
* `torchdiffeq` is not used; the substep loop is explicit.

Known limitations (all are deliberate simplifications, listed so they are not
mistaken for validated physics):

* no tyre relaxation length, no load sensitivity (`load_sensitivity` is
  available but defaults to 0), no camber, no aligning moment;
* no suspension dynamics, roll/pitch states or unsprung mass;
* no gearbox, clutch, engine inertia or brake hydraulics (brake torque is
  instantaneous);
* the wheel-spin equilibrium during acceleration/braking carries a first order
  numerical bias of order `h*R*stiffness*a`, visible as extra drag on free
  rolling wheels at high longitudinal acceleration;
* the plant is **not calibrated to CarSim yet** — it is the nominal start point
  for that calibration.

## Presets

`fullsize_racecar_ev` is the CarSim-matched target: 2019 CarSim "Exotic
Racecar" chassis (sprung mass 1200 kg, CG height 375 mm, wheelbase 2650 mm,
lF 1455 / lR 1195 mm, Izz 1065.2 kg m², total mass 1320 kg with unsprung
corners, total yaw inertia 1350 kg m²), the workspace 384 kW direct-drive motor
map with the 3.44:1 final drive, 6 MPa brakes with 400 / 175 N m per MPa, and
CdA 0.80 / ClA 1.70. `racecar`, `sensorcar` and `xcar` keep their original
XCAR parameters and only gained the fields the new plant needs.

Check the plant before training:

```bash
cd xcar-simulation
python plant_check.py fullsize_racecar_ev
```

It prints acceleration, braking, steady-state cornering, lateral limit, LSD,
aero, wheel-spin and throughput probes.

## Measured throughput (RTX 5070 Laptop, 8 GB)

| what | envs | wall time per control step | env-steps / s |
|---|---|---|---|
| FastPlant, raw step (5 substeps) | 20 000 | 28.6 ms | 0.70 M |
| FastPlant, raw step (5 substeps) | 2 000 | 29.3 ms | 0.07 M |
| full `continuous_drift_iwd` inside PPO | 8 192 | 47 ms | 0.17 M |
| legacy XCAR plant (100 Hz, 1 step) | 20 000 | 4.9 ms | 4.1 M |
| legacy full-size plant (v3.5.7, 2 substeps) | 4 096 | ~84 ms | 0.049 M |

Per *simulated* second (what actually matters for RL) the new plant costs
20 x 28.6 ms = 572 ms per wall second at 20 000 envs, against 100 x 4.9 ms =
489 ms for the original XCAR plant: ~17 % more for 10 states, five physics
substeps per action and a much richer model.  Against the legacy full-size
plant it is roughly an order of magnitude faster per simulated second.

The cost is dominated by kernel launch overhead (every tensor is only
`(n, 4)`), not by arithmetic. Reducing it further means fewer, larger kernels
(fusing the substep loop with `torch.compile`/CUDA graphs) — a framework
change to JAX would not help and is not practical on native Windows anyway.

## Running

```bash
# 20 Hz control / 100 Hz physics, pedals interface, CarSim-sized EV
python run.py train continuous_drift_iwd --car-preset fullsize_racecar_ev \
    --num-parallel 16384 --horizon 128 --gamma 0.999 --tau 0.97 \
    --plant fast --control-mode pedals --exp-name ev_pedals_v1

# same plant with the legacy XCAR wheel-speed action interface
python run.py train continuous_drift_iwd --car-preset fullsize_racecar_ev \
    --plant fast --control-mode wheel_speed --exp-name ev_ws_v1

# legacy XCAR plant (6 states, 100 Hz, torchdiffeq euler)
python run.py train continuous_drift_iwd --car-preset xcar --plant iwd --exp-name xcar_legacy
```
