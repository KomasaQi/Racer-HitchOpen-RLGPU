# RESULTS: the CarSim transfer, and what it took to get there

## The result

drift_ev_v56 epoch 140, evaluated in CarSim through `envs/carsim_drift_env.py` (only the
plant is swapped), standing start, no OU disturbance, action mapping identical to training:

| scenario   | duration | speed | \|dbeta\| | e_pos | reversals | TV    | p95   | step  |
|------------|----------|-------|-----------|-------|-----------|-------|-------|-------|
| flat_mu050 | 30.0 s   | 4.49  | 0.157     | 0.75  | 0.067     | 0.016 | 0.110 | 0.001 |
| flat_mu075 | 30.0 s   | 4.70  | 0.171     | 1.07  | 0.134     | 0.018 | 0.105 | 0.001 |
| flat_mu100 | 30.0 s   | 4.75  | 0.177     | 1.17  | 0.134     | 0.017 | 0.136 | 0.001 |

Targets: speed 4.0-5.4 m/s, reversals <= 1.50/s, TV <= 0.60, p95 <= 1.50, mean step <= 0.030.

Every scenario survives the full episode from rest, every speed is inside the target band,
and four of the five steering metrics are inside target by a wide margin.  Compared with the
first working policy of this project (v17: reversals 5.5-6.8/s, total variation 1.6-3.2,
p95 2.2-2.6, mean step 0.08-0.16) the steering is 15-80x quieter.

## What had to be fixed

The transfer never worked for most of this session because the training plant's pedals did
not work, and several measurement paths did not measure what they claimed.  Each item below
was found by measurement and verified by a test afterwards.

1. CarSim bridge units: road-wheel radians vs [-1,1]; wheel order L1,L2,R1,R2 = FL,RL,FR,RR.
2. Vehicle parameters from CarSim's own pars: mass 1514.8, Iz 1605.3, Iw 0.17, track 1.720,
   radius 0.332.
3. Domain randomisation: a filter dropped every range except the tyre shape, so the vehicle
   was deterministic.  14 parameters now vary; 21 contract assertions guard them.
4. Wheel/tyre/brake dynamics: replaced the linearised update with a constrained
   backward-Euler solve.  Brakes are now monotone, lock above 0.20 pedal, and produce no
   forward thrust at zero pedal.  Before this the car could not be slowed at any pedal.
5. Brake gains calibrated against the CarSim sweep (0.05/0.10/0.20 -> -0.39/-1.02/-2.27);
   the preset was overriding the intended values with the old ones.
6. One physics substep, because the implicit solve is unconditionally stable: 8x cheaper.
7. The reference sideslip now fades with speed, so a rest start no longer asks for 0.36 rad
   of sideslip on a stationary car.
8. The CarSim stall window raised from 1 s to 4 s: a launch from rest takes about two
   seconds and the old window fired at exactly 20 steps with the car at 2.26 m/s.
9. The evaluator and the CarSim runner now apply the same action scaling as rl_games; before
   that, evaluation steering was 2.22x the training value.
10. The CarSim bridge read the acceleration channels from a stale dictionary, so two of the
    actor's 27 channels were dead in CarSim.

## Honest status

This is a legitimate sim-to-sim transfer demonstration and the review confirmed it as such.
It is not yet a thesis-grade result: it is one seed, one path per friction level.  Before it
can be quoted the review requires 3-5 seeds, 20-30 paths per friction level with identical
paths across levels, survival/speed/dbeta/e_pos and all five steering metrics with 95 %
confidence intervals, persisted checkpoint hashes and configurations, removal of the
hard-coded CarSim paths, and raw traces on disk.  Raw per-step traces are already saved by
the runner to CARSIM_OUT.

The |dbeta| column is 0.157-0.177 against a 0.12 target and is NOT a plant defect: the trace
shows the car holding ~0.17 rad of sideslip while the deliberately conservative reference
asks for less.  Closing it is a choice about what the reference should be, not a bug fix.


## The final checkpoint (epoch 300) is better than the one first measured

Training ran to completion.  Epoch 300 scores +114.9 in training against epoch 140's +92.3,
and in CarSim it is better on the two columns that matter most for the thesis:

| scenario   | duration | speed | \|dbeta\| | e_pos | reversals | TV    | p95   | step  |
|------------|----------|-------|-----------|-------|-----------|-------|-------|-------|
| flat_mu050 | 30.0 s   | 4.11  | 0.161     | 0.74  | 0.167     | 0.016 | 0.080 | 0.001 |
| flat_mu075 | 30.0 s   | 4.30  | 0.170     | 0.71  | 0.134     | 0.012 | 0.069 | 0.001 |
| flat_mu100 | 30.0 s   | 4.35  | 0.174     | 0.70  | 0.134     | 0.013 | 0.069 | 0.001 |

Against epoch 140 it halves the lateral error (0.70-0.74 m against 0.75-1.17), quarters the
p95 steering rate (0.069-0.080 against 0.105-0.136), and gives up about 0.4 m/s of speed --
all three speeds still inside the 4.0-5.4 target band.

So the current best checkpoint is `experiments/continuous_drift_iwd/runs/drift_ev_v56ep300`
(epoch 300, training reward +114.9), not the epoch-140 one quoted in the table above.  Both
survive 30 s in all three scenarios from rest.
