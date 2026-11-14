# Current project status

Updated 2026-11-14. The sustained FastPlant-to-CarSim drift objective is **not achieved**.

## Active run

v11: `audit_repaired_gru128_v11_resume_v9ep60_sensor50ms`,32768 environments,
full DR, streaming random paths, horizon40, sequence10, minibatch81920.
Resumes the CarSim-validated v9 checkpoint60, including actor, independent
critic, normalizers, optimizer states and counters. Source checkpoint unchanged.
User clarified that only degraded checkpoints are prohibited from continuation.
Fresh v10 was stopped and retained. Observation latency0..50ms; dropout hold off.
Exact CPU restore comparison and a post-restore PPO update passed.
Actor: observable bypass + shared GRU128 +256/124/64 mirrored-history calls;
beta head antisymmetric. Independent privileged critic uses body-frame velocity.

Adaptive actor learning rate is now actually bounded to 3e-5..3e-4.
Installed rl_games ignored previous YAML minimum and used 1e-6..1e-2;
v8 logged LR0.006667, KL94.85 and saturated actions. v8 was stopped after
checkpoints65/70 failed all six CarSim fixtures with zero throttle/full brake.
The LR defect is fixed; this does not prove future training cannot collapse.

## Verified scope

Mirror core, PPO sequence likelihood, paired recurrent resets, auxiliary and
critic gradient updates, checkpoint deployment parity, legacy regressions,
32768-environment GPU rollout/update, and short three-road CarSim interface
smoke passed. These are infrastructure checks, not successful drift evidence.
Initial v9 total training throughput is about47k-50k frames/s on this machine.

## Evaluation and unresolved work

Use fixed R15 circles in both directions and R12/R15 unequal-circle eight,
with rest starts, known seeds, original CarSim scenarios, and archived native
results copied to LastRun. Require actual drift persistence, tracking, speed,
powered rear slip, pedal usage and steering smoothness; survival is insufficient.
V11 checkpoint75 completes all nine60s circle/eight fixtures. The R12/R15
unequal-circle eight also completes120s on all three roads for each of two
seeds20261115/20261116. This is not general success: lowgrip ellipse30x20
fails off-path at12.55s; mid/high ellipse survive120s but mid tracking and
braking remain weak. Steering/beta oscillations are unresolved.
FastPlant eight CPU8 tests complete30s for all24 samples, with mid/high
braking weaknesses also present. Ellipse cross-plant diagnosis is running.
See `v11_ep075_validation_update.json` and `v11_ep075_ellipse_diagnostic.png`.
The original CarSim motor efficiency extrapolation defect and low-mu macro
model mismatch remain unresolved; isolated calibration candidates are not adopted.

Evaluation update: user-approved throttle threshold is now0.15 (previously0.35),
for both FastPlant and CarSim powered-drift metrics. Reward is unchanged. New
outputs record the threshold; historical CarSim recalculation is in
`v9_metric_throttle015_recomputed.json`, preserving original result files.

## Reproduction and offline records

All run outputs, source snapshots and detailed decisions are outside this repo:
`../Codex_Drift_Audit_2026-11-13/review_20261114/` and its sibling `training_runs/`.
Start with `v11_launch.json`, `v11_resume_effective_check.json`, `mirror_integration_progress.json`,
`v8_collapse_lr_finding.json`, and `v11_training.log`.
Previous STATUS.md is archived as `repository_status_before_v9.md` there.

Repository checks:
- `experiments/continuous_drift_iwd/audit_mirror_contract.py --output PATH`
- `experiments/continuous_drift_iwd/ppo_integration_check.py --mirror-policy ...`
- `experiments/continuous_drift_iwd/audit_regressions.py --output PATH`

The active experiment's source snapshot is authoritative for its implementation;
subsequent test/documentation additions do not alter its running process.


Optional streaming-reference controls are now wired through training and evaluation:
`--reference-segments` (default2) and `--reference-transition-min-m` (default unset).
Short spans bound their curvature change by the configured derivative limit.
Initialization, reset and continuation were checked; the default generator is
bitwise equal to the previous implementation. V11 remains on its original
source snapshot/config. The candidate is not proven to improve training.
Evidence: `optional_reference_config_checks.json`, `optional_reference_eval_cli.json`.

## Active continuation comparisons (2026-11-14)

Main v11 training was stopped at the user's request; epoch175 is preserved.
Do not assume the latest checkpoint is best: high-grip eight speed fell and
braking/steering oscillation increased. Healthy v11 epoch105 is the common
starting point for matched continuation experiments; degraded checkpoints
must not be resumed.

`--adaptive-speed-coupling` now supports `precision` (unchanged default),
`independent_position` (speed term independent of position error, retaining
course alignment), and `corridor` (broad smooth position/course gates).
Only the performance reward changes; drift-quality precision gates remain.
The independent-position and precision comparison uses frozen external source,
16384 environments per branch, full DR, identical seeds and20 updates.
Neither candidate is accepted based on reward or survival alone.

Detailed source hashes, configuration comparison and validation are in
`../Codex_Drift_Audit_2026-11-13/review_20261114/reward_speed_pair/`,
`speed_coupling_candidate_source/validation.json`, and
`speed_coupling_integration.json`. Path-sampling pilot results are separate in
`path_sampling_training_pair/`. Steering reversal shaping is still an offline
prototype, not an active training reward.


## 2026-11-14 experiment drain before user pause

User requested completion of already-running experiments then a stop. All launched training/evaluation jobs have now exited; final training exit codes are zero and checkpoint epoch/frame budgets and SHA256 were verified. No new experiment is queued. Details: `../Codex_Drift_Audit_2026-11-13/review_20261114/drain_before_user_pause.json` and `drain_training_completion_verified.json` in that directory.

Reward smooth03/tight10 reached ep125 (13,107,200 added frames each); path baseline/short reached ep145 (6,553,600 each). Final followup and short-path checkpoints remain unvalidated in CarSim; their final evaluations were not launched during drain. No candidate was adopted. Speed-position decoupling improved ep125 high-grip eight speed in two evaluation seeds, but tracking/braking gains were inconsistent and mid-grip circle tracking worsened. Early smoothness/tracking-only changes did not resolve high-grip straight oscillation. Low-grip ellipse failures remain. Overall transfer goal is NOT achieved. Wait for user to resume before further experiments.
