# Artifact policy

This repository contains source code, compact configuration, contract tests and concise technical documentation only. Checkpoints, rollout CSV files, rendered media, profiler output and large scenario snapshots are intentionally stored outside Git.

Local project artifact root:

`../Codex_Drift_Audit_2026-11-13/`

Locations:

- `repo_archive/legacy_repo_artifacts/`: files removed from the historical repository tree without deleting them;
- `data/`: CarSim/FastPlant calibration, paired A/B results, policy evaluations and figures;
- `training_runs/`: completed Codex training run directories after the producing process exits;
- the Markdown files in the artifact root are the detailed offline engineering memory.

Every promoted policy must be accompanied outside the repository by its checkpoint, `env_config.json`, `train_args.json`, evaluation outputs and a checksum manifest. No script may silently select a checkpoint by modification time; evaluators require an explicit checkpoint and its matching configuration.

Formal training commands pass `--run-root ../Codex_Drift_Audit_2026-11-13/training_runs`
(normally as an absolute path). The default remains `runs` only for upstream-compatible
ad-hoc use; repository-local run output is ignored by Git and must not be committed.
Evaluation environments do not write reference trajectories implicitly. A caller that
needs one must pass an explicit external `trajectory_output_dir`.
