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
