# Trim solver specification (from the review, round 41)

The current `trim_table.py` does not converge (residuals 0.25-0.82 against a 1e-2
target).  The review identified six separate causes, all of them in the formulation
rather than in the physics.  This file records the design to implement next; the
solver must not be wired into training until every acceptance check below passes.

## What is wrong with the current solver

1. **Sign of beta.** The state is built with `psi = chi + beta`, which gives the body
   velocity `(V cos b, -V sin b)`.  So the conventional sideslip is `-beta`, and every
   "beta" this solver reported (up to +0.88 rad) means -0.88 conventional.  Use
   `psi = chi - beta` so that a positive `beta` is a positive sideslip, and mirror
   `beta`/`delta` for negative curvature.

2. **float32 plant, float64 optimiser.** SciPy builds a finite-difference Jacobian
   with steps appropriate for float64 against a float32 plant, so the derivative can
   be zero or noise.  Build the plant with `FastRacerPlant(preset, 1, "cpu",
   dtype=torch.float64)` and the state/command with `dtype=torch.float64`, or set
   `diff_step=1e-4` explicitly and give up on `xtol=1e-12`.

3. **Load transfer is solved from a cold guess.** `reset_guess()` on every evaluation
   means the residual is computed against two load iterations starting from zero, not
   the converged state the rollout uses.  Re-evaluate the same candidate until
   `extras["ax"]` and `extras["ay"]` stop changing (3-5 calls) before reading the
   residual.

4. **Throttle is a free unknown.** It should be eliminated.  Rear wheel equilibrium is
   `Ta/2 - c - Q_RL = 0` and `Ta/2 + c - Q_RR = 0` with `Q_i = T_roll,i + R_i Fx_i`
   and `c = clip(klsd (w_RL - w_RR))`, hence

       Ta*  = Q_RL + Q_RR
       t*   = (Ta* - Ta(0)) / (Ta(1) - Ta(0))

   (`coast_torque_nm` is 0 for this preset, so it is required over available axle
   torque).  Reject the point if `t*` is outside [0, 1].

5. **Wheel rows are dynamic and arbitrarily scaled.** `domega/3000` hides the
   physically meaningful balance.  Solve in per-wheel longitudinal slip instead:

       u_i  = extras["vx_wheel"][i]            # exact, includes yaw/track/Ackermann
       D_i  = max(|u_i|, v_floor)
       w_i  = (u_i + kx_i D_i) / R_i
       kx_i = (R_i w_i - u_i) / D_i            # = -sx_i for this plant

   Six unknowns: `z = [beta, delta, kx_FL, kx_FR, kx_RL, kx_RR]`.  Do NOT share one
   slip per axle: Ackermann, yaw and the viscous LSD all need left/right difference.

6. **Six residuals**, with `Ts = m g R / 4 ~ 1230 Nm` and `L = lF + lR`:

       [ dvxb/g, dvyb/g, (d[5] * Iz)/(m g L),
         N_FL/Ts, N_FR/Ts, (N_RL - N_RR)/(2 Ts) ]

   where `dvxb = a_xb + r v_yb`, `dvyb = a_yb - r v_xb` and
   `N_i = T_drive,i - T_roll,i - R_i Fx_i`.

## Bounds, starts, continuation

    lo = [-0.65, -0.449, -0.08, -0.08, -0.02, -0.02]
    hi = [ 0.65,  0.449,  0.05,  0.05,  0.35,  0.35]

Front slips slightly negative (rolling resistance), rear positive.  Solve cells in
increasing `V^2 kappa / (mu g)` with continuation from the nearest feasible
lower-demand cell.  Without a neighbour, three starts for positive curvature:

    normal      beta  0.00  delta  atan(L kappa)  kF -0.003  kR 0.01
    mild drift  beta  0.20  delta -0.10           kF -0.003  kR 0.08
    deep drift  beta  0.40  delta -0.25           kF -0.003  kR 0.15

    least_squares(method="trf",
                  x_scale=[0.3, 0.2, 0.03, 0.03, 0.10, 0.10],
                  ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=300)

## Acceptance criteria (all of them)

    max |scaled residual| < 1e-3
    |dvxb|, |dvyb|        < 0.02 m/s^2
    |rdot|                < 0.03 rad/s^2
    |wheel_net_torque_i|  < 1.5 Nm
    t* in [0, 1]
    no tyre on the slip_max clamp
    2-3 s rollout under (delta*, t*) stays close to the circle

The rollout check matters because the trim must use the same Ackermann scale, tyre
scale and randomised parameters as the environment that spawns it.

## Infeasible cells

Do not fall back to a smaller beta: beta is an outcome of the solve, so changing it at
fixed (V, kappa, mu) just produces another non-equilibrium state.  Instead mark the
cell infeasible and resample: same mu and kappa at the highest feasible speed below
the requested one; failing that, same mu at lower curvature.  Store and apply the
whole feasible tuple (state, four wheel speeds, steering, throttle).  Interpolate only
inside a connected feasible region.

## The last point, and the most important one

"Zero action" must mean zero CORRECTION around the trim `(delta*, throttle*)`.  If the
action interface sends literal zero steering and throttle, even a perfect trim state
leaves equilibrium immediately -- which is exactly what the 2.1 s zero-action test in
round 37 measured.  Either the actor output is an increment on the trim command, or
the environment must hold the trim command when the action is zero.  This has to be
settled before any spawn retest, or the same false negative will reappear.
