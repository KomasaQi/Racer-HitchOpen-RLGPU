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

The policy behind the numbers above is archived at
`experiments/continuous_drift_iwd/runs/drift_ev_best/`; the evaluation command
above works on that directory unchanged.
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