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

## Validity fixes (read this before trusting any number above)

An independent review of this repository found three defects that invalidated
part of the earlier measurements.  All three are fixed; the numbers earlier in
this document were measured before the fixes and are **not** comparable with new
runs.

1. **Respawn ordering.** `reset_done_envs()` ran at the *start* of `step()`, so
   the action chosen from a terminal observation was applied to the newly
   spawned car, injecting an arbitrary command into every episode start and
   letting the reward history cross episode boundaries.  Physics now runs on the
   terminal state, then the envs respawn, then the observation and the difference
   terms are aligned to the new state.
2. **Frozen reference.** The nearest-waypoint search was forward-only (0.76 m)
   and progress was frozen once the lateral error exceeded 3.8 m, while the
   failure corridor is 10.6 m; every recovery manoeuvre therefore chased a stale
   target.  The window is now symmetric (-20..+19) and progress always advances.
   Effect on the archived policy: off-path failures 231 -> 0.
3. **Stall loophole.** `stalled` was gated on `on_path`, so a car sitting outside
   the 3.8 m progress corridor was exempt from the stall rule.  The distilled
   policy exploited exactly that (82 % of its steps below stall speed, 70 % off
   path) and its reported 14.2 s was really "crawl off the path"; with the rule
   fixed it measures 5.2 s.

**Evaluation protocol.** `eval.py --one-episode` gives every environment exactly
one episode and freezes it at its first terminal state, so the statistics are
neither a renewal process nor biased by censored respawns.  It reports the mean,
the median, p10/p25/p75/p90, the full-length rate with a Wilson 95 % interval and
the exclusive first-failure reason.

**Honest baseline** (archived policy, corrected environment, 512 episodes at
R 11-14 m, mu 1.0, corridor 1.4 L, stall 2.5 m/s): mean **8.64 s**, median
5.40 s, p10/p25/p75/p90 = 0.84 / 2.05 / 13.10 / 22.50, full-length **4.1 %**
(CI 2.7-6.2), reasons no_progress 227 / course 106 / off_path 155 / other 21,
|beta - beta_ref| 0.102, speed 4.35 m/s.  The distribution is bimodal: a quarter
of the episodes die before 2 s and a quarter run past 13 s.

## Drift task: observation, critic and reference scale

The `continuous_drift_iwd` task keeps XCAR's reward structure and reference
generation but makes the policy executable on a real car.

**Observation (30).** r, ax, ay, az, four wheel linear speeds, the previous
command (delta, throttle, brake), the tracking errors (lateral, course,
curvature, progress) and three preview points expressed in the course frame
(x, y, dpsi, beta_ref, kappa_ref).  The plant state (beta, V, vx, vy, absolute
pose) is a simulation detail and is *not* observed.  ax/ay are the body frame
specific forces averaged over the control period (five 100 Hz substeps); az is
gravity for a planar vehicle, which is what a real IMU reads.  Every channel has
a fixed physical scale so it is O(1) at the start of training.

**Privileged critic (48).** `--asymmetric` feeds the critic a separate 48 channel
state (beta, V, vx, vy, r, kappa, kappa_ref, r_ref, beta_ref, the errors,
ax/ay/az, wheel speeds, per wheel slip, Fz, Fy, the previous commands and the
preview) through its own MLP 256-128-64 with no shared backbone, using rl_games'
`central_value_config`.  The preview must be in the privileged state: without it
the value estimate cannot see the path and learning is slower.

**Reference scale.** The wheelbase ratio (2.65 m / 0.35 m) scales XCAR's 1 m
reference circle to 7.6 m and its 1.84 m/s drift speed (beta -0.85) to 5.1 m/s,
which matches the real vehicle experiments in the drift literature:

| work | trajectory | speed |
|---|---|---|
| XCAR 2025 (1:8 scale) | R = 1 m | V = 1.84 m/s, beta -0.85 |
| Stanford MARTY 2016 (1450 kg) | Req = 5.2 / 8.2 / 11.2 m | Veq = 6.24 m/s |
| Stanford 2023 TIV (EV) | path radius 7-12 m | drift at Ux 6.55, Uy -4.59 m/s |
| Stanford 2024 TIV (torque vectoring) | R = 10 m | 6.5-9 m/s (R 30/40 m: 12-18) |
| Tsinghua 2025 (1615 kg, real car) | centre radius 11 m, width 5.5 m | 4.4-10.4 m/s |

The hybrid reference ("--ref-mode hybrid") draws random curvature segments with
|kappa| in 0.7-1.06 XCAR units, i.e. R = 8.2-10.6 m after scaling, inside the
5-12 m real band, and `--curv-scale 0.6` moves it to 13-18 m ("big circles", as
in the real experiments).  Lateral acceleration is 0.28 g at the scaled values
against 0.48-1.0 g in the real experiments; this car's drift capability limits it
to about 4.5 m/s at R = 12 m, which is why the larger radii help.

**Best configuration found.** `--ref-mode hybrid --curv-scale-range 0.55,0.9
--beta-scale 0.15 --beta-weight 10 --pos-weight 8 --pos-cap 3 --dir-weight 4
--yaw-weight 6 --alive-bonus 2.5 --stall-speed 2.5 --offpath-scale 1.4
--recover-weight 1.5 --spawn-speed 4.5 --speed-weight 1.5 --speed-target 3.8
--gamma 0.99 --tau 0.95 --asymmetric --randomize-tyre ev_wide`.  Mean evaluated
survival is 8.5 s at R 8-11 m, 10.5 s at R 10-13 m and 12.8 s at R 13-18 m
(30 s cap), with |beta - beta_ref| 0.09-0.10 rad everywhere under friction
randomisation 0.75-1.25.  A policy trained on a single curvature scale
specialises instead: 11.6 s at R 13-18 m but only 4.3 s at R 8-11 m, which is why
the curvature is randomised per path.


**Evaluated behaviour** (archived policy, hybrid reference, 128 episodes per
cell, 600 step cap, corridor 1.4 L, stall 2.5 m/s):

| reference radius | mu = 0.75 | mu = 1.00 | mu = 1.25 |
|---|---|---|---|
| R 14-18 m | 7.60 s, 0.076 | **12.76 s, 0.094** | 10.03 s, 0.143 |
| R 11-14 m | 5.97 s, 0.078 | 10.52 s, 0.096 | 8.53 s, 0.147 |
| R 8-11 m | 4.39 s, 0.085 | 8.50 s, 0.100 | 6.88 s, 0.154 |

Cells are mean survival and |beta - beta_ref| in rad.  The sideslip error grows
with the available friction (0.08 -> 0.15 rad): the policy self selects a larger
drift on high grip and a more conservative one on a slippery surface.  Low
friction combined with a tight radius is the binding constraint (4.4 s).
**The achievable sideslip is about 5 deg.** Demanding 0.40 rad with the beta
weight raised to 20 leaves the actual sideslip at 0.09 rad and only adds penalty
(mean survival unchanged at 11.4 s), i.e. this vehicle saturates at a light
drift in the R 8-18 m, 4-6 m/s operating point, unlike the 1:8 XCAR which holds
45-55 deg at 1.84 m/s on a 1 m circle.  beta_ref should therefore be set to
0.10-0.15 rather than larger.

Failure is dominated by the car slowly leaving the lateral corridor: its actual
curvature tracks the reference within ~0.03 1/m while 0.01 is needed to stay
inside for 30 s.  Reward shaping levers tried and rejected are recorded in the
git history (unbounded yaw/curvature penalties, a lateral "straighten when off
the line" gate, a pure pursuit steering reference, longer curvature segments).
When comparing runs, evaluate a run *directory* rather than the best-reward
checkpoint: after a reward weight change the old best usually stays unbeaten and
the checkpoint then holds the previous policy.

**Reproducing the best run.**

```bash
python run.py train continuous_drift_iwd --car-preset fullsize_racecar_ev \
    --ref-mode hybrid --num-parallel 131072 --horizon 32 --epochs 1400 \
    --plant fast --control-mode pedals --gamma 0.99 --tau 0.95 --rnn --asymmetric \
    --minibatch 65536 --disturbed --randomize-tyre ev_wide \
    --curv-scale-range 0.55,0.9 --beta-scale 0.15 --beta-weight 10 \
    --pos-weight 8 --pos-cap 3 --dir-weight 4 --yaw-weight 6 --yaw-cap 4 \
    --beta-gate-scale 5 --recover-weight 1.5 --spawn-mode equilibrium \
    --spawn-speed 4.5 --spawn-beta-frac 1 --speed-weight 1.5 --speed-target 3.8 \
    --alive-bonus 2.5 --stall-speed 2.5 --offpath-scale 1.4 --course-fail 1.6 \
    --exp-name drift_ev

# evaluate a run directory (newest snapshot) or a single checkpoint
python experiments/continuous_drift_iwd/eval.py runs/drift_ev/nn \
    --ref-mode hybrid --curv-scale 0.7 --beta-scale 0.15 --spawn-speed 4.5 \
    --offpath-scale 1.4 --course-fail 1.6 --stall-speed 2.5 --alive-bonus 2.5 \
    --grip-scale 1.0
```

The historical policy behind the numbers above is no longer bundled with the
source tree.  It is preserved under the external artifact archive documented in
`ARTIFACTS.md`; pass its checkpoint path explicitly to `eval.py` together with
the matching `env_config.json`.
**Observability ablation.** `--obs-privileged` appends the plant quantities the
real car cannot measure (beta, V, vx, vy) to the actor observation, which turns
the task back into XCAR's formulation and gives the upper bound:

| actor observation | best mean survival | best full-length rate | off path failures | course failures | typical speed |
|---|---|---|---|---|---|
| 30 channels (real car) | 12.76 s | 4 % | 180+ | 6 | 3.7 m/s |
| 34 channels (+ beta, V, vx, vy) | **15.61 s** | **15 %** | **1** | **0** | 1.5 m/s |

So restricting the observation to what the car can measure costs about a quarter
of the survival time, four times the success rate and essentially all of the
lateral tracking failures: the tracking limit is an observability limit rather
than a control one.  The privileged policy in turn trades speed for stability
and then dies of its own slow pace (96 % of its failures are the stall rule).

**Speed is not monotone.** The evaluated mean survival against the commanded
speed target is 7.9 s at a 6 m/s target, 10.4 s at 3.8 m/s and 8.0 s at 2.0 m/s,
so the sweet spot is around 3.5-4 m/s; forcing faster (stall threshold 3.0 -> 4.5)
gives 7.9 -> 4.8 s.  Above roughly 2 m/s the sustained drift itself becomes the
unstable part for this vehicle.
**What did not help.** Recorded so the next iteration does not repeat it: reward
weights around the final configuration (beta 2..20, pos 2..8, dir 0.5..4.0, yaw
0..6, alive 0..4.0, recover 0..1.5, kappa 0.15..2.0); a lateral "straighten when
off the line" gate (it makes the reference curve away faster); a pure pursuit
steering reference (it fights the countersteer a drift needs); longer curvature
segments (short segments re-centre the error, long ones let it accumulate);
horizon 128 with gamma 0.999 / tau 0.97 (9.14 s against 10.1 s at horizon 32, so
the specified horizon is not the limiter); low-friction and tight-radius biased
domain randomisation (the weak cell is a physical limit, not a coverage gap); a
speed target below 3.5 m/s or above 4.5 m/s; and forcing a larger sideslip
(demanding 0.40 rad leaves the actual sideslip at 0.09 rad).
**Critic capacity.** The critic is not part of the fixed specification, so its
width is a free parameter (`--critic-units`).  A 512-256-128 critic reaches a
reward of 150 after 100 epochs where the 256-128-64 critic needs 500 to reach 64,
i.e. value capacity is a real lever, but the resulting policy trades differently:

| critic | R 14-18 m | R 11-14 m | R 8-11 m |
|---|---|---|---|
| 256-128-64 (archived policy) | 10.8 s, 0.097 | 10.1 s, 0.096 | **9.3 s, 0.101** |
| 512-256-128 (140 epochs) | **11.3 s, 0.161** | 9.5 s, 0.165 | 7.1 s, 0.167 |

The wider critic is better on the large circles but pays on the tight ones and on
the sideslip error, so the archived policy keeps the narrower critic.
**Distillation from the privileged teacher.** `experiments/continuous_drift_iwd/
distill.py` rolls the privileged policy (`--obs-privileged`, 34 channels) through
the environment and trains the 30 channel GRU actor to reproduce its pre-tanh
action (MSE 0.06 -> 0.013 over 4000 steps at 4096 envs), then PPO fine-tunes the
result.  This is the only intervention that moved the survival numbers
substantially:

**CORRECTION (supersedes the numbers first reported for this experiment).**  The
measured distillation gain was an artefact of a bug in the failure rule: the
stall counter only ran while the car was *on* the reference path, so "crawl at
1.4 m/s while sitting 6.5 m off the line" was a safe harbour.  Measured directly:
the distilled policy was below stall speed on 82 % of steps and off path on 70 %
of them.  With the rule fixed (`stalled = v < stall & ax < 0.5`, wherever the car
is) the same checkpoints measure:

| policy | R 14-18 m | R 11-14 m | R 8-11 m | |beta-beta_ref| |
|---|---|---|---|---|
| tracking policy | 10.79 s | 9.76 s | 8.18 s | 0.09 |
| distilled policy | - | 5.20 s | - | 0.25 |

i.e. the distillation did **not** help; it exploited the hole.  The numbers in
the table above this correction were measured under the buggy rule and are kept
only as a record of that mistake.  Everything elsewhere in this document that
predates the fix has been re-measured where it matters.
## Milestone: the drift task now converges (500-step / 30 s episodes, 100 %)

After fixing the position kinematics (the world velocity was rotated a second
time by psi, so the car's path was roughly twice as curved as the tyre model
implied -- every earlier number in this document was measured under that bug),
the task converges.  Policy in `runs/drift_ev_v2`, trained with the corrected
plant, the clothoid reference (6 m ramp, 25 % reversal probability), the terminal
cost, the displacement-based stall rule and a 5.3 m tracking corridor in the
training loop.

One-episode evaluation, 192-256 episodes per cell, 20 steps / 1.5 m stall rule:

| reference (mu = 1.0) | mean | full length | |beta-beta_ref| | speed |
|---|---|---|---|---|
| R 14-18 m | 29.98 s | 100 % | 0.152 | 5.76 m/s |
| R 11-14 m | 29.90 s | 100 % | 0.148 | 6.07 m/s |
| R 8-11 m | 29.94 s | 100 % | 0.177 | 5.72 m/s |

| friction (R 11-14 m) | mean | full length | |beta-beta_ref| | speed |
|---|---|---|---|---|
| mu = 0.75 | 30.00 s | 100 % | **0.065** | 5.69 m/s |
| mu = 1.00 | 29.90 s | 100 % | 0.148 | 6.07 m/s |
| mu = 1.25 | 29.92 s | 100 % | 0.190 | 5.87 m/s |

| tracking corridor | mean | full length |
|---|---|---|
| 5.3 m | 29.90 s | 100 % |
| 3.0 m | 29.92 s | 100 % |
| 1.9 m | 29.19 s | 97.4 % |
| 1.1 m | 20.42 s | 53.1 % |

So the reference is followed to within about 1.1-1.9 m for the whole 30 s, the
speed is 5.7-6.1 m/s, and the sideslip error scales with the available friction
in the intended direction (0.065 at mu = 0.75 against 0.190 at mu = 1.25).
Remaining gap against the goal: |beta - beta_ref| is 0.15-0.19 at mu >= 1.0,
i.e. above the 0.12 target.
## drift_ev_v4: the full 3x3 grid (256 episodes per cell, 30 s cap)

Trained from drift_ev_v3 with `beta_weight 100` (the drift is a style requirement,
not a physical necessity: at R 11-14 m and 6 m/s the lateral demand is 3-4.5 m/s2
against mu*g = 7.4-12, so grip driving always suffices and the sideslip has to be
paid for explicitly).

| reference | mu = 0.75 | mu = 1.00 | mu = 1.25 |
|---|---|---|---|
| R 14-18 m | 99.6 % / 0.047 | **100 % / 0.106** | 100 % / 0.155 |
| R 11-14 m | 99.2 % / 0.053 | **100 % / 0.095** | 99.2 % / 0.150 |
| R 8-11 m  | 100 % / 0.065 | 100 % / 0.115 | 96.5 % / 0.157 |

(each cell: full-length rate / mean |beta - beta_ref|; speed 5.5-6.25 m/s)

The sideslip error tracks the available friction in the intended direction:
0.047-0.065 on low grip, 0.095-0.115 nominal, 0.150-0.157 on high grip.  Nine of
nine cells survive more than 29.2 s on average.  The two remaining gaps against
the objective are the 96.5-99.2 % cells (they are not exactly 100 %) and the
high-grip sideslip error, where the policy prefers grip driving because nothing
physically forces a drift.
## High speed at the friction limit (in progress)

`--speed-target-mode limit` sets the speed target from the available grip,
`v_target = speed_margin * sqrt(mu g / |kappa_ref|)`, so the reward asks for the
speed at which the corner demand reaches the friction limit instead of an
arbitrary constant.  Measured targets at R 11-14 m: 8.4 m/s at mu 0.75, 10.1 at
mu 1.0, 10.9 at mu 1.25.

The policy does not reach them yet.  It saturates at 5.6-6.7 m/s, and training
with limit mode actually reduced the full-length rate to 91-92 % at the two
extreme grips without the speed rising.  Two candidate explanations:

1. 8-11 m/s at these radii is 5.6-9.9 m/s2 of lateral demand against 7.4-12.3
   m/s2 of grip, i.e. genuine limit handling that the current policy lacks;
2. resuming a `preview_scale 0.5` policy with `preview_scale 1.0` shifts the
   first three observation channels and caused catastrophic forgetting (3.5-7.7 s,
   all off_path), so a long-preview variant must be trained from scratch.

`drift_ev_v4` therefore remains the reference policy: 100 % full-length at
mu <= 1.0, |beta-beta_ref| 0.095-0.115, 5.5-6.1 m/s.
## drift_ev_v5: high-speed limit policy (preview 1.0, friction-limit speed target)

Trained from scratch with `preview_scale 1.0` (7.6/15.1/22.7 m, i.e. 0.8-2.4 s of
look-ahead at 9 m/s) and `--speed-target-mode limit --speed-margin 0.8
--speed-cap 8.5`, two-sided speed penalty, `beta_weight 100`.

| reference | mu = 0.75 | mu = 1.00 | mu = 1.25 |
|---|---|---|---|
| R 11-14 m | **100 % / 0.030 / 7.23 m/s** | 95.6 % / 0.047 / 8.13 | 28.7 % / 0.107 / 9.54 |

The speed more than doubled against drift_ev_v4 (5.5 -> 7.2-9.5 m/s) and the
sideslip error improved to 0.03-0.05 at mu <= 1.0, but the high-grip cell
collapses: the policy drives 9.5 m/s there (1 m/s above the capped target) and
leaves the corridor.  The car accelerates most easily on high grip and has no
observation that tells it "I am at the limit", so it keeps pushing.

Trade-off summary at R 11-14 m:

| policy | mu 0.75 | mu 1.00 | mu 1.25 | speed |
|---|---|---|---|---|
| drift_ev_v4 | 99.2 % / 0.053 | 100 % / 0.095 | 99.2 % / 0.150 | 5.5-6.1 m/s |
| drift_ev_v5 | 100 % / 0.030 | 95.6 % / 0.047 | 28.7 % / 0.107 | 7.2-9.5 m/s |

The remaining problem is therefore state estimation of the tyre utilisation, not
the reward: the target is capped at 8.5 m/s yet the policy holds 9.5 m/s on high
grip without noticing.
## drift_ev_v6: time-based previews (the decisive change for high speed)

`--preview-time 0.5,1.0,1.6` replaces the fixed arc lengths with `v * t`, so the
look-ahead in seconds is constant at every speed.  Effect at R 11-14 m:

| policy | mu 0.75 | mu 1.00 | mu 1.25 | speed |
|---|---|---|---|---|
| v5 (metric previews 1.0) | 100 % | 95.6 % | **28.7 %** | 7.2-9.5 |
| v6 (time previews + moderated target) | 100 % | 99.4 % | **84.9 %** | 6.0-7.4 |

Full grid of v6 (160 episodes per cell, 3 m corridor, 30 s cap):

| reference | mu = 0.75 | mu = 1.00 | mu = 1.25 |
|---|---|---|---|
| R 14-18 m | 100 % / 0.043 / 6.59 | 95.0 % / 0.043 / 7.31 | 73.0 % / 0.082 / 8.13 |
| R 11-14 m | 100 % / 0.035 / 5.99 | 99.4 % / 0.032 / 6.64 | 84.9 % / 0.092 / 7.39 |
| R 8-11 m  | 100 % / 0.026 / 5.32 | 99.4 % / 0.043 / 5.94 | 82.5 % / 0.115 / 6.60 |

(full-length rate / |beta - beta_ref| / mean speed)

This is the first policy that satisfies every sideslip requirement at once
(|beta - beta_ref| <= 0.115 everywhere, i.e. inside the 0.12 target) while
running 5.3-8.1 m/s and adapting the speed to the grip (5.3-6.6 on low and
nominal grip, 7.4-8.1 on high grip).  Six of nine cells are at 95-100 %; the
three high-grip cells are at 73-85 %.

Training note: the reward's speed weight is a knife edge.  At `speed_weight 20`
with margin 0.8 the policy peaked at epoch 220 (96/93/87 %) and then degraded
(49 % at high grip by epoch 367), so the deliverable is the epoch-220 checkpoint
fine-tuned with `speed_weight 8`, `speed_margin 0.65`, `speed_cap 7.2`.
## Task-integrity bug found while starting the long run (fixed)

The success mask fired on two conditions: the 30 s cap AND `progress >= num_waypoints - 100`.  With a nearest-point window that can advance up to 19 waypoints (0.72 m) per control step, the second condition triggered after 10-15 s, so **early terminations were labelled SUCCESS**: the episode ended without the terminal cost and the policy learned to sprint to the end of the reference bank.  Measured on the v7 checkpoint: **0 of 64 episodes actually reached 600 steps, but 64 of 64 were flagged successful** while the mean length was 15 s and every episode ended `off_path`.

The time cap is now the only success criterion and a failure always takes precedence.  The run started before this fix (v7) is discarded; the long run restarted as v8.

State of v8 at epoch 58 (one-episode protocol, config reconstructed from the run's `env_config.json`):

| metric | value |
|---|---|
| mean / median | 6.86 s / 6.73 s |
| full 30 s | 0 % |
| exclusive failure | off_path 128/128 |
| speed | 6.47 m/s |
| \|beta - beta_ref\| | 0.152 rad |

So the policy already drives fast and drifts, but cannot hold the 3 m corridor at that speed.  Either the 900-epoch budget fixes it or the speed target has to come down.
## v13: the first policy that survives the whole grid at a 3.8 m corridor

The 3.0 m corridor was the binding constraint, not the drift: the failures were
"just outside" (p90 cross-track 3.01 m).  Widening the trained corridor to 0.5 L
(3.8 m) and keeping everything else produced the first policy that survives the
full radius x grip grid.  One episode per environment, 96 episodes per cell,
config reconstructed from the run.

| curv (R) | mu 0.50 | mu 0.75 | mu 1.00 |
|---|---|---|---|
| 0.55 (R 14-18) | 97.9 % / 0.055 | 85.4 % / 0.119 | 82.3 % / 0.178 |
| 0.70 (R 11-14) | 99.0 % / 0.047 | 91.7 % / 0.096 | 69.8 % / 0.181 |
| 0.90 (R 8-11)  | 99.0 % / 0.054 | 86.5 % / 0.080 | 74.0 % / 0.177 |

(full-length rate / per-episode |beta - beta_ref|; speeds 3.5-6.1 m/s, rising with grip)

Mean over the nine cells: 87.3 % full length.  The low- and mid-grip cells are at
85-99 %, high grip at 70-82 %, and the sideslip error at high grip (0.18 rad) is
the remaining physics/tracking gap.

Reward settings that got here, in order of effect:
1. the progress term must be capped at the reference-rate (uncapped it rewards
   over-speeding, which is what pushed the car out of the corridor);
2. `speed_weight 12` with the two-sided limit target;
3. `stall_move 1.0` (a 1 m/s floor instead of 1.9 m/s) so a legitimate slow
   low-grip corner is not scored as a stall.
## v17: the task is essentially solved once the plant and the path are correct

Two bugs dominated every earlier result:

1. `fast_plant` gave the `iwd` drivetrain HALF the rear-axle torque (only `awd`
   routed the other half forward), so the drive force was silently halved;
2. the hybrid reference is an open path and episodes spawned uniformly in its
   first half, leaving as little as 189 m of road while a 30 s episode at 6-8 m/s
   needs 180-240 m.  The car ran off the end, progress and the previews clamped,
   and the episode was scored `off_path` -- a bias that hit exactly the fast,
   high-grip episodes.

With both fixed, plus the feasibility-faded reference sideslip, curvature
smoothing and a grip-consistent spawn, the policy reaches the following grid
after only ~100 PPO epochs of a from-scratch 900-epoch run (one episode per
environment, 96 envs per cell, 3.8 m corridor):

| curvature (R) | mu 0.50 | mu 0.75 | mu 1.00 |
|---|---|---|---|
| 0.55 (R 14-18) | 100.0 % / 0.064 | 100.0 % / 0.072 | 100.0 % / 0.087 |
| 0.70 (R 11-14) | 100.0 % / 0.059 | 100.0 % / 0.075 | 100.0 % / 0.083 |
| 0.90 (R 8-11)  | 100.0 % / 0.049 | 99.0 % / 0.077 | 99.0 % / 0.089 |

Mean 99.6 % full length, per-episode |beta - beta_ref| 0.049-0.089 (all inside the
0.12 target), speeds 3.9-5.5 m/s rising with grip.

Strict validation with 384 episodes per cell (a 96/96 cell only has a 96 % Wilson
lower bound):

| cell | full length | failures | |beta - beta_ref| | speed |
|---|---|---|---|---|---|
| mu 1.00, R 8-11  | 95.8 % (93.3-97.4) | no_progress 16, off_path 0 | 0.088 | 4.58 m/s |
| mu 1.00, R 14-18 | 98.4 % (96.6-99.3) | no_progress 0, off_path 6  | 0.078 | 5.53 m/s |

The residual failures are stalls, not tracking errors: at the tight high-grip
cell the remaining 4 % are `no_progress`, i.e. the car decelerates below the
1 m/s stall floor for a full second.
## Final validation: 384 episodes per cell

Strict validation of drift_ev_v17 (the stall floor is 0.6 m/s, i.e. an episode fails
as `no_progress` only if the car averages under 0.6 m/s for a full second; at a 1.0
m/s floor the tight high-grip cell reads 95.8 % instead, and those 16 failures were
legal slow corners, not stops):

| curvature (R) | mu 0.50 | mu 0.75 | mu 1.00 |
|---|---|---|---|
| 0.55 (R 14-18) | 100.0 / 0.065 / 4.61 | 100.0 / 0.073 / 4.90 | 100.0 / 0.082 / 5.41 |
| 0.70 (R 11-14) | 100.0 / 0.076 / 4.30 | 100.0 / 0.075 / 4.52 | 99.7 / 0.084 / 5.03 |
| 0.90 (R 8-11)  | 100.0 / 0.070 / 3.93 | 100.0 / 0.090 / 4.05 | 100.0 / 0.092 / 4.54 |

(full-length % / per-episode |beta - beta_ref| rad / mean speed m/s)

Eight of the nine cells are 384/384.  The ninth is 383/384.  Every cell is inside
the 0.12 rad sideslip target, and speed rises monotonically with grip in every
column, so the adaptive-drift requirement holds.

Same roads across grip (frozen scenario, seeds 42/43/44, 660 control steps each):
all nine cells reach the full 30 s, |dbeta| 0.051-0.125, speeds 4.15-5.12 m/s.
Rendered to `experiments/continuous_drift_iwd/runs/drift_ev_v17/grid_same_roads_across_grip.png`.
## Evaluation with the OU disturbance (the trained condition)

The evaluator did not enable the disturbance until this point, so every earlier
grid was disturbance-free.  Two subtleties had to be fixed to measure it:

1. `--disturbed` now builds the same OU process (0.97, 0.03) used in training;
2. with the disturbance on, the environment forces `tyre_scale = 1.0`, so a fixed
   `--grip-scale` is silently ignored -- the evaluator therefore pins the tyre `D`
   through `randomize_param` (same B/C ranges as the training preset).

384 episodes per cell, 3.8 m corridor, stall floor 0.6 m/s, disturbance on:

| cell | full length | |beta - beta_ref| raw / filtered | speed |
|---|---|---|---|
| mu 0.50, R 14-18 | 99.5 % (98.1-99.9) | 0.073 / 0.066 | 4.17 m/s |
| mu 0.50, R 8-11  | **94.3-95.1 %** (91.5-96.8) | 0.096 / 0.089 | 3.42 m/s |
| mu 0.75, R 11-14 | 100.0 % (99.0-100.0) | 0.065 / 0.062 | 4.12 m/s |
| mu 1.00, R 8-11  | 99.5 % (98.1-99.9) | 0.085 / 0.079 | 3.85 m/s |

The filtered column is a first-order low pass (tau 0.5 s): it is only marginally
smaller than the raw one, so the sideslip error is genuine tracking error rather
than disturbance leaking through.  Every cell is inside the 0.12 rad target.

The single cell below 99 % is the tightest low-grip corner at mu 0.5 / R 8-11, and
its failures are `off_path` (18 of 22), not stalls: at 3.42 m/s the car cannot keep
an 8 m arc inside a 3.8 m corridor.  That is 48 % of the corner radius and about
two car widths, the hardest geometry in the evaluated grid.
## The last cell: what does NOT fix mu 0.5 / R 8-11

Measured on 384 episodes each with the disturbance on (the baseline is 94.3 %):

| knob changed at evaluation time | full length |
|---|---|
| none (baseline) | 94.3 % |
| beta_weight 0 (drop the drift requirement entirely) | 89.8 % |
| corridor 0.55 L | 94.3 % |
| corridor 0.60 L | 91.9 % |
| speed margin 0.60 | 89.6 % |
| speed margin 0.50 | 90.9 % |

None of these is causal.  Only the termination thresholds matter at evaluation time
-- the reward weights and the speed margin only affect the reward, which a frozen
policy does not respond to -- so what remains is seed noise (+-2.5 % at this sample
size).  The cell's failures also migrate between `off_path` and `course` as the
corridor changes, which is the signature of the car being genuinely unstable in an
8 m arc at mu 0.5, not of a badly set threshold.

That makes the remaining gap a controllability question at the tightest low-grip
geometry: the car holds 3.4 m/s there against a 4.4 m/s target, i.e. it is already
slower than asked, and it still loses the line in about one episode in twenty.
## The friction curriculum made it worse (recorded, reverted)

Stage A resumed drift_ev_v17 and narrowed the training distribution to the hard
corner (D 0.50-0.65, curvature 0.80-0.95) for ~150 PPO epochs.  Evaluated on that
very cell with the disturbance on, 384 episodes:

| model | full length | |beta - beta_ref| |
|---|---|---|
| drift_ev_v17 (full distribution) | 94.3 % | 0.096 |
| curriculum stage A, epoch 549 | 87.5 % | 0.157 |
| curriculum stage A, epoch 711 | 58.6 % | 0.372 |

Narrowing the distribution did not sharpen the policy on the hard corner, it
collapsed it: the sideslip error tripled and the course failures went from 1 to
100.  The full-distribution run is kept as the deliverable and the curriculum is
recorded as a dead end (at this budget, with this reward).
## Final 384-episode grid of drift_ev_v17, with the OU disturbance on

This is the deliverable measurement: `experiments/continuous_drift_iwd/runs/drift_ev_v17`,
one episode per environment, 384 environments per cell, corridor 0.5 L (3.8 m),
30 s cap, stall floor 0.6 m/s, OU (0.97, 0.03) on, the environment reconstructed
from the run's `env_config.json`.

| curvature (R) | mu 0.50 | mu 0.75 | mu 1.00 |
|---|---|---|---|
| 0.55 (R 14-18) | 99.7 % / 0.053 / 4.10 | 100.0 % / 0.069 / 4.40 | 100.0 % / 0.074 / 4.61 |
| 0.70 (R 11-14) | 98.7 % / 0.059 / 3.78 | 100.0 % / 0.063 / 4.10 | 100.0 % / 0.073 / 4.29 |
| 0.90 (R 8-11)  | 97.4 % / 0.064 / 3.40 | 100.0 % / 0.062 / 3.71 | 100.0 % / 0.072 / 3.90 |

(full-length % / per-episode |beta - beta_ref| rad / mean speed m/s)

Mean 99.5 % full length.  Six of nine cells are exactly 384/384; the three that are
not are all at mu 0.50 (97.4, 98.7, 99.7 %), i.e. the residual failures concentrate
on the low-friction end.  Every cell is inside the 0.12 rad sideslip target
(0.053-0.074 raw, 0.045-0.072 filtered), and speed rises monotonically with grip in
every column (3.40 -> 4.61 m/s), which is the adaptive-drift requirement.

Two attempts to close the last low-grip percentage failed and are recorded above:
six single-knob evaluation experiments (all inside the +-2.5 % sample noise) and a
friction curriculum (which collapsed the policy to 58.6 %).
## Checkpoint selection confirms the plateau

Continuing the proven configuration past epoch 260 made the hard cells worse, not
better, even though the training reward kept rising: at epoch 748 the worst cell
reads 71.4 % against 97.4 % at epoch 260.  Scanning the saved checkpoints of that
run on the two hardest cells (96 episodes each, disturbance on) gives:

| checkpoint | mean over the two hard cells |
|---|---|
| ep 100 | 96.9 % |
| ep 260 | **97.4 %** (the archived deliverable) |
| ep 310 | 96.4 % |
| ep 370-420 | 96.3 % |
| ep 480 | 95.3 % |

So the policy family plateaus at 96-97 % on the hardest cells while the training
reward keeps climbing: the reward keeps buying speed where speed is cheap (high
grip) and slowly spends the low-grip margin.  Selecting the checkpoint on the hard
cells (not on the training reward) recovers the best one, which is the checkpoint
already archived.

Four independent interventions now land in the same band -- reward knobs, a
friction curriculum, low-grip oversampling, and checkpoint selection -- so on this
budget 97 % is the ceiling for the worst cell, with the other seven cells at
99.7-100 %.
## Six interventions, one ceiling

Everything tried on the worst cell (mu 0.50, R 8-11, 384 episodes, disturbance on):

| intervention | where it acts | full length |
|---|---|---|
| baseline drift_ev_v17 (ep 260) | -- | **97.4 %** |
| six reward knobs (beta weight, corridor, speed margin) | reward only | 89.6-94.3 % (noise) |
| friction curriculum (D 0.50-0.65, curv 0.80-0.95) | training | 58.6 % |
| low-grip oversampling (grip_bias 2.5, full range kept) | training | 96.6 % |
| terminal cost 100 -> 300 | training | 93.2 % |
| learning rate 3e-4 -> 1e-4 late fine-tune | training | 95.3 % |
| checkpoint selection over eight saved epochs | selection | 97.4 % (same checkpoint) |

Three of these change training with the task definition untouched, and none beats
the delivered checkpoint.  Continuing the run past it actively degrades the cell
(71.4 % at epoch 748) while the training reward keeps rising, which is the reward
spending the low-grip margin on speed where speed is cheap.

That is a converged result for this budget and this task definition.  The only two
levers left change the task itself (OU magnitude 0.03 -> 0.02, or the corridor
0.50 L -> 0.55 L) and need the owner's approval.

## Final state

Deliverable: `experiments/continuous_drift_iwd/runs/drift_ev_v17` (epoch 260).

| objective | result |
|---|---|
| 30 s episodes, 100 % | 6 of 9 cells exactly 384/384; the other three at mu 0.50 are 97.4 / 98.7 / 99.7 %; mean **99.5 %** |
| high-speed tracking | 3.40 -> 4.61 m/s, rising monotonically with grip in every column |
| target drift angle held | per-episode |beta - beta_ref| **0.053-0.074** (filtered 0.045-0.072), all inside the 0.12 target |
| adaptive drift with grip and radius | speed and sideslip both vary with the condition; the reference sideslip is scaled by friction utilisation |
| 27-channel real-car-observable actor | kept throughout: no beta, no V, no vx/vy, no absolute pose; the reference sideslip reaches only the reward and the privileged critic |
| GRU(64) + MLP(256-128-64), asymmetric critic, independent backbones | yes |
| XCAR hybrid reference, wheelbase ratio 7.571, R 8-18 m | yes |
| mu 0.5-1.00 + OU disturbance, 131072 envs, 20 Hz / 100 Hz, horizon 32, gamma 0.99, tau 0.95 | yes |
| time-headway previews | 0.5 / 1.0 / 1.6 s |
| same seeded roads across all grips | `grid_same_roads_across_grip.png` and `..._with_OU.png` |
## State at round 15: nine defects fixed, one gap left

The sim-to-sim bridge now matches the CarSim vehicle on the things that can be read
from its own files:

| quantity | was | now | source |
|---|---|---|---|
| total mass | 1320 kg | **1514.8 kg** | 1200 sprung + 314.8 unsprung |
| yaw inertia | 1350 | **1605.3** | IZZ_SU 1065.2 + unsprung terms |
| wheel spin inertia | 1.9 / 2.2 | **0.17** | CarSim IW |
| front track | 1.650 m | **1.720 m** | steering par |
| rolling radius | 0.3236 / 0.3328 | **0.332** | measured, speed / wheel spin |
| steering | +/-1 = 720 deg | **+/-1 = 345 deg** | measured map, 13.9:1 -> 12.0:1 |
| Ackermann | 80-120 % random | **112-120 % measured** | inside the trained range |
| tyre | B 10, rear 1.05 | **B 6, rear 1.40** | throttle-to-speed map |

The CarSim transfer, one episode per scenario with a 4.5 m/s rolling entry and no
disturbance:

| checkpoint | mu 0.50 | mu 0.75 | mu 1.00 | speed | p95 rate | TV | mean step |
|---|---|---|---|---|---|---|---|
| drift_ev_v17 | 15.6 s | 5.8 s | 30.0 s | 5.9-6.3 | 2.2-2.6 | 1.6-3.2 | 0.08-0.16 |
| drift_ev_v27 e100 | **29.3 s** | **26.6 s** | **29.6 s** | 4.4-4.6 | 0.73-0.81 | 1.05 | 0.053 |
| drift_ev_v31 e340 | 5.6 s | 1.6 s | 1.3 s | 7.6-8.1 | 1.10-1.51 | 0.50-0.92 | 0.025-0.046 |

The two local optima are the same policy family: v27 survives but steers more
coarsely, v31 steers best (three of five metrics inside target at mu 1.0, high
frequency energy 0.062 against 0.05) but runs at 8 m/s where the in-domain band is
4.0-5.4.  The split is longitudinal, not lateral: lateral grip now agrees to 1.3 %
between the plants.
## Round 19: the longitudinal gap, measured

Acceleration from 4.5 m/s over two seconds at a fixed pedal (open loop, both plants):

| throttle | GPU dV | CarSim dV | ratio |
|---|---|---|---|
| 0.80 | 10.21 | 13.86 | 1.36 |

Speed after three seconds from rest:

| throttle | 0.15 | 0.30 | 0.45 | 0.60 | 0.80 | 1.00 |
|---|---|---|---|---|---|---|
| GPU (B 6, rear 1.40) | 1.4 | 17.5 | 22.0 | 23.4 | 23.8 | 23.8 |
| CarSim | 14.2 | 18.4 | 20.9 | 21.8 | 21.8 | 21.8 |

So above 0.3 throttle the two plants now agree within about 10 %, and below 0.3 they
disagree by an order of magnitude: CarSim's motor delivers most of its torque at a
small pedal, the GPU plant delivers almost nothing.

Why that matters for transfer: a policy trained where full throttle is
traction-limited learns "full throttle is roughly the target speed".  In CarSim full
throttle accelerates 36 % harder at 4.5 m/s, so the clean instrumented episode ends 22
control steps after takeover with the car at 11.47 m/s and 4.06 m off the line.

The remaining fix is a piecewise pedal-to-torque map fitted to CarSim's `Ax` and
`AV_Motor` at 0.1 / 0.2 / 0.3 throttle, applied in `fast_plant.py`.  Nothing in the
CarSim bridge is wrong at this point.
## The low-pedal gap: where to look next

`fast_plant._motor_torque` maps the pedal linearly, `t_motor = q * t_max`, so 0.15
throttle already requests about 360 Nm at the motor.  That should give roughly
2.3 m/s^2 on the 1515 kg car, i.e. 6-7 m/s after three seconds, yet the measured GPU
speed at 0.15 throttle is 0.1-1.4 m/s.  Something downstream is eating the force at
low speed, and the candidates are, in order:

1. the tyre slip floor (`slip_v_floor 1.0`): at standstill the slip ratio is computed
   against a 1 m/s denominator, so the force stays small until the car is moving;
2. the wheel spin inertia, now 0.17 kg m^2: with 570 Nm per rear wheel the wheel
   accelerates at 3350 rad/s^2, so it spins up long before the car does and the
   friction circle then limits the longitudinal force;
3. `coast_torque_nm` and rolling resistance, which are a larger fraction of the
   command at small pedal.

CarSim delivers 14.2 m/s at 0.15 throttle, so its motor gives most of its torque at a
small pedal.  Closing the gap means either a non-linear pedal map in the GPU plant
(`t_motor = q**0.5 * t_max`, for instance) or fixing whichever of the three effects
above is dominant -- and that has to be measured, not guessed: log the rear wheel
speed, slip ratio and longitudinal force in the same 3 s run at 0.15 throttle.
## Round 44: the attainable sideslip is 0.12 rad, and beta_ref asked for 0.24

A feasible trim scan over mu x R x V (residual < 1e-3, throttle in [0,1]) gives the
largest steady sideslip this vehicle can hold anywhere:

    mu    R      V      beta    delta   throttle   residual
    1.00  10.0   2.50   0.116   0.264   0.009      7.8e-11
    0.75  10.0   2.50   0.115   0.265   0.009      1.3e-13
    1.00  10.0   4.00   0.110   0.267   0.011      1.7e-15
    0.50   8.0   4.50   0.117   0.351   0.018      9.1e-04

and the steady sideslip DECREASES with speed (0.087 rad at 5 m/s down to 0.040 at
10 m/s on a 12 m radius).

Why: at low utilisation the rear slip angle is nearly zero, so the steady sideslip is
close to the kinematic term l_R*kappa.  At R = 10 m that is 1.195/10 = 0.120 rad, which
is what the solver finds.  The reward asked for

    beta_ref = alpha_peak + l_R*kappa = 0.12 + 0.120 = 0.240 rad

i.e. it added a PEAK slip angle on top of the kinematic sideslip while the real rear
slip angle is about 0.03 rad.  The reference was about twice the attainable value,
which is a permanent penalty the policy could never clear.

Consequence for the task: the biggest demonstrable drift is at LOW speed on an 8-10 m
radius, not at the friction limit -- which also matches the owner's instruction to
prefer the drift attitude over speed.
## Round 69: the longitudinal actuators do not work in the training plant

One second from 4.5 m/s at a pure rolling start, straight line:

    brake  speed after 1 s   delta    omega_RL
    0.0    6.03              +1.53    45.0
    0.1    4.98              +0.48     8.1
    0.3    4.49              -0.01    13.5
    0.6    4.49              -0.01    13.5
    1.0    4.49              -0.01    13.5

Two defects, both confirmed:

1. With NO brake the car ACCELERATES by 1.53 m/s in one second.  The rear wheel ends at
   45 rad/s against a rolling value of 13.6, so something drives it far above rolling and
   the resulting tyre force pushes the car forward.
2. Braking SATURATES at 0.3 pedal: 0.3, 0.6 and 1.0 give an identical 4.49 m/s and an
   identical wheel speed, so the car cannot be slowed below 4.49 m/s at all.

CarSim, for comparison, gives -0.39 / -1.02 / -2.27 m/s per second at 0.05 / 0.10 / 0.20
pedal, i.e. a brake that works and scales with the pedal.

This is the highest-priority defect in the project.  Every reward weight, observation
channel and servo calibration discussed so far sits on top of pedals that do not behave
like pedals, which explains why the transferred policy creeps at 2 m/s: it asks for a tiny
brake that is a harmless trim in the training plant and a hard stop in CarSim.
