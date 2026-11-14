"""Measure the GPU reset hot path at production environment counts.

This is a compact regression gate, not an experiment artifact.  It reconstructs
the exact training MDP from ``env_config.json`` and writes measurements only to
an explicit output path outside the source repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from envs.continuous_drift_iwd import ContinuousDriftIWDEnv  # noqa: E402


def timed_ms(fn, repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append(1e3 * (time.perf_counter() - start))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=262144)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--fractions", default="0.0001,0.001,0.01,0.1,1.0")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--ablate", default="",
                        help="comma list: randomize,sensor,wheels,guess,extras,path")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        cfg = json.load(stream)
    cfg = dict(cfg)
    ref_mode = cfg.pop("ref_mode_cli", "closed_random")
    cfg.pop("obs_privileged", None)
    cfg["train"] = True
    cfg["quiet"] = True
    cfg["device"] = args.device
    cfg["plant"] = cfg.get("plant", "fast")
    cfg["control_mode"] = cfg.get("control_mode", "pedals")
    plant = cfg.pop("plant")
    control_mode = cfg.pop("control_mode")
    device = cfg.pop("device")

    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", args.n, device, ref_mode=ref_mode,
        plant=plant, control_mode=control_mode, **cfg
    )
    env.reset()
    torch.cuda.synchronize()
    ablate = {name for name in args.ablate.split(",") if name}
    if "randomize" in ablate:
        env.randomize = lambda *_args, **_kwargs: None
    if "sensor" in ablate:
        env._reset_sensor_model = lambda *_args, **_kwargs: None
    if "wheels" in ablate:
        env.plant.init_wheels_mask = lambda state, _mask: state
    if "guess" in ablate:
        env.plant.reset_guess = lambda *_args, **_kwargs: None
    if "path" in ablate:
        env._try_path_refresh = lambda *_args, **_kwargs: False
    if "extras" in ablate:
        old_extras = env.extras
        old_es = env.es
        env._eval_extras = lambda *_args, **_kwargs: old_extras
        env.plant.package_extended = lambda *_args, **_kwargs: old_es

    fractions = [float(value) for value in args.fractions.split(",")]
    results = []
    # Contiguous masks make the requested reset count exact and reproducible.
    for fraction in fractions:
        count = max(1, min(args.n, int(round(args.n * fraction))))
        mask = torch.zeros(args.n, dtype=torch.bool, device=args.device)
        mask[:count] = True

        def reset_selected() -> None:
            env.is_done.zero_()
            env.is_done[mask] = 1
            env.reset_done_envs()

        reset_selected()  # allocator/kernel warm-up for this shape
        samples = timed_ms(reset_selected, args.repeats)
        results.append({
            "fraction": fraction,
            "count": count,
            "mean_ms": sum(samples) / len(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "samples_ms": samples,
        })

    payload = {
        "config": os.path.abspath(args.config),
        "n": args.n,
        "device": args.device,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "ablate": sorted(ablate),
        "results": results,
    }
    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        count = max(1, int(round(args.n * 0.01)))
        mask = torch.zeros(args.n, dtype=torch.bool, device=args.device)
        mask[:count] = True
        env.is_done.zero_()
        env.is_done[mask] = 1
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True) as prof:
            env.reset_done_envs()
            torch.cuda.synchronize()
        payload["profile_cuda"] = prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=30)
        payload["profile_cpu"] = prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=30)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
