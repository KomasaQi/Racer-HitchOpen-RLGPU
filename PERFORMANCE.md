# Performance audit — 2026-11-17

PPO150 takes about78–80 minutes in the current concurrent experiment. Existing
rl_games phase logs attribute81–83% to environment stepping,5–6% to policy
inference,12–13% to update and other work. This is phase timing, not a CUDA
kernel profile. CPU diagnostics independently found110 full tire evaluations
per50ms control step in one test state, mostly from implicit wheel-speed solves.

Prioritize caching per-substep invariants and fusing the wheel residual solver
while preserving10ms physics, iterations, motor/LSD coupling and all dynamics.
Do not reduce physical fidelity or recurrent observations to manufacture speed.
No production acceleration has been applied or benchmarked yet.

Full timing CSV, figures, latest evaluation snapshot, reproducible CPU diagnostic
and validation plan are outside the repository in
`../DSH_Day_2026-11-17/analysis_speed/REPORT.txt`.

PPO7 first seed improves between105 and150; nine previews trade improved
straight smoothness for lower speed and less medium-grip figure-eight drift.
DSAC-T evaluation has missing straight results due to geometry assertions;
exclude missing results from success claims and audit before ranking algorithms.

This documentation snapshot does not change active experiment code/configs.
The original DSH seal belongs to commit c82d690 and must remain intact. The
currently running scheduler continues; do not restart the old sealed plan at a
new HEAD or rewrite its seal to hide the difference.

## Equivalent implicit solver acceleration

Developed in the isolated `codex/fastplant-speed` worktree. The running DSH
experiment remains on the original checkout/code. After completing that frozen
matrix, use this version with `--fastplant-solver cuda_graph`. The option is
saved as `implicit_solver_backend` in env_config.json. `original` preserves the
reference implementation; `cached` (the CLI default) removes repeated invariant
work; `cuda_graph` additionally replays the wheel refinement with the original
arithmetic operations. CPU evaluation uses cached equations automatically.

The solver keeps5x10ms substeps per50ms control period,12 bisections,2 coupled
Newton corrections, dynamic bracket expansion, motor dynamics and LSD coupling.
Only compact-transition non-relaxed RWD/IWD tires use the graph specialization;
other cases retain the original general path. Explicit unsupported graph
configurations raise an error. Run GPU rollouts under torch.no_grad(), as PPO
does. Initial graph capture is a startup cost and uses extra GPU memory.

32768-car synchronized short tests while other training was running:
physics median1.286s ->0.242s per control step (5.32x); full environment median
1.320s ->0.277s (4.76x). These are preliminary concurrent timings, not an
isolated150-epoch speed claim. The original78-minute training could project to
roughly28minutes if phase proportions and speedup hold; verify end to end first.

Correctness: the full48-car training environment matched states, returned reward,
actor and critic observations bitwise over40steps including16automatic resets.
Explicit parameter DR updates after graph capture also matched bitwise. A3-epoch
2048-env full-DR PPO smoke and a separate CLI/config-save smoke passed, as did
all existing contract tests. The48-car three-road600-step (30s) trajectory comparison, with selected resets,
matched all10states and Fx bitwise. Long trajectory results and raw benchmark/equivalence
JSON are in `../FastPlant_Optimization_20261117/`.

Reproduce solver checks with:
`python experiments/continuous_drift_iwd/solver_equivalence.py --device cuda --backend cuda_graph --steps 600 --output ABSOLUTE_RESULT_JSON`
This test deliberately samples parameter ranges and changes them after capture.
It is a numerical-equivalence test, not a learned-policy drift success test.
