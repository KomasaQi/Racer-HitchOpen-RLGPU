# Artifact policy

This repository contains source code, compact configuration, contract tests and concise technical documentation only. Checkpoints, rollout CSV files, rendered media, profiler output and large scenario snapshots are intentionally stored outside Git.

Historical calibration and audit root:

`../Codex_Drift_Audit_2026-11-13/`

Locations:

- `repo_archive/legacy_repo_artifacts/`: files removed from the historical repository tree without deleting them;
- `data/`: CarSim/FastPlant calibration, paired A/B results, policy evaluations and figures;
- `training_runs/`: completed Codex training run directories after the producing process exits;
- the Markdown files in the artifact root are the detailed offline engineering memory.

Every promoted policy must be accompanied outside the repository by its checkpoint, `env_config.json`, `train_args.json`, evaluation outputs and a checksum manifest. No script may silently select a checkpoint by modification time; evaluators require an explicit checkpoint and its matching configuration.

Formal training commands pass `--run-root ../Codex_Drift_Audit_2026-11-13/training_runs`
(normally as an absolute path). The PPO9 profile defaults to the sibling `training_runs` directory. Legacy CLI runs use `runs` unless overridden.
Evaluation environments do not write reference trajectories implicitly. A caller that
needs one must pass an explicit external `trajectory_output_dir`.

## 2026-11-18整理

PPO9_Review_20261118/repository_before_cleanup/tracked_source.zip 为整理前完整跟踪文件备份；retired_orchestration/ 保留过期night调度器。完成的加速实验和封印worktree均留在仓库外，不删除历史策略。今后策略评测仅CarSim，所有输出使用外部路径。默认PPO9训练输出为 ../training_runs。
