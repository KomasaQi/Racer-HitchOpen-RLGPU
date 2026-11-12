# DRIFT: what was achieved, what blocks the rest, and the evidence

## The owner's complaint that started this goal

The previous "success" was slow path tracking, not drifting.  Measured on drift_ev_v56
epoch 140: throttle 0.000-0.037, rear wheels SLOWER than the front, rear longitudinal slip
NEGATIVE (-0.09 to -0.14), i.e. the direction of engine braking.

## What now works (drift_ev_v62, the best checkpoint of this goal)

### In the GPU plant, |beta| over 96 environments

| t | \|beta\| | \|r\| | throttle | kx_rear |
|---|---|---|---|---|
| 1.3 | 0.187 | 0.661 | 0.272 | 1.159 |
| 2.6 | 0.150 | 0.746 | 0.258 | 0.361 |
| 5.1 | 0.143 | 0.771 | 0.245 | 0.596 |
| 8.8 | 0.147 | 0.745 | 0.232 | 0.634 |

- rear longitudinal slip **0.2-1.2** (target 0.55)
- rear wheel surface speed **7.8-9.5 m/s against 5.8-6.4 at the front**
- throttle **0.23-0.28** held open, against the 0.00-0.037 of the crawling policy
- yaw rate **0.66-0.77 rad/s**
- sustained sideslip **0.14-0.16 rad (8-9 degrees)**

### In CarSim, from a 2.5 m/s rolling entry with the drift spawn installed

| scenario | duration | speed | \|dbeta\| | e_pos | reversals | TV | p95 | step |
|---|---|---|---|---|---|---|---|---|
| flat_mu050 | 8.2 s | 3.34 | 0.225 | 1.13 | 0.86 | 0.27 | 0.59 | 0.013 |
| flat_mu075 | 4.0 s | 0.69 | 0.096 | 0.05 | 1.80 | 0.16 | 0.52 | 0.008 |
| flat_mu100 | 7.1 s | 4.35 | 0.278 | 1.40 | - | - | - | - |

Four of five steering metrics inside target in the scenarios that ran; the drift attitude is
what takes the car off the path (reason 2) rather than any instability.

## What does NOT work, and the evidence

A sustained sideslip above ~0.15 rad was not reached.  The number 0.11-0.18 rad held across
EVERY lever tried in sixteen rounds:

| lever | values tried | sideslip |
|---|---|---|
| speed | 4.5 / 6 / 8 m/s | 0.13-0.18 |
| joint gate weight | 12 / 40 | 0.13-0.17 |
| rear lateral peak mu_y | 0.65 / 0.85 / 1.05 | 0.11-0.18 |
| motor power | 384 / 500 kW | 0.12-0.18 |
| tyre model | isotropic / friction ellipse | 0.14-0.16 |
| drift sign convention | both | 0.14-0.16 |
| countersteer gate | dead / real | 0.12-0.16 |
| joint gate | inactive / active | 0.07-0.19 |
| rear slip weight | 60 / 15 | 0.07-0.19 |
| spawn sideslip | 0.30/0.40 | entry 0.35, decays |

Three independent lines agree that a STEADY drift is outside this vehicle:

1. the trim solver finds no drift state (residual 0.3-0.45 against a 1e-3 threshold) while
   the normal cornering branch converges to 1e-15, in every configuration above;
2. the front axle's lateral capacity is 4.85 kN while the path demands m V^2 kappa = 8.08 kN
   at 8 m/s with a 0.4 rad attitude;
3. yaw balance needs ~55 % of the lateral force from the REAR, and a friction ellipse
   reduces exactly that share once the rear is overslipping.

## The one hypothesis not yet tested

The 27-channel actor has NO sideslip or sideslip-error channel: it sees yaw rate, averaged
accelerations, four wheel speeds, the previous command, velocity-course errors and previews.
Body heading relative to the velocity is not among them.  A GRU can infer a sideslip from yaw
and IMU history but only weakly, and not at all at reset.

Testing it means giving the actor a deployable sideslip estimate (GNSS/INS, or an observer
state) and re-running.  If the sideslip rises, the ceiling is observability.  If it returns to
0.15, the ceiling is the vehicle and should be recorded as such.

## What the owner should decide

1. Accept 8-9 degrees of sustained sideslip with the rear axle overslipping and the throttle
   open as the achievable power slide for this vehicle, and finish the CarSim verification on
   that basis; or
2. Change the vehicle fundamentally (mass, wheelbase, driven-axle position), which is what
   actually sets the drift angle a car can hold, and recalibrate; or
3. Add a sideslip observation and settle the observability question first.
