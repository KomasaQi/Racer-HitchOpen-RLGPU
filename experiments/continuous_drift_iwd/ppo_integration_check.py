"""Single disposable PPO update to verify rollout/auxiliary plumbing, no checkpoint."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--num-parallel", type=int, default=4096)
p.add_argument("--horizon", type=int, default=10)
p.add_argument("--seq-length", type=int, default=5)
p.add_argument("--minibatch", type=int, default=20480)
p.add_argument("--steer-actuator-profile", choices=["legacy", "responsive"], default="legacy")
p.add_argument("--device", default="cuda:0")
p.add_argument("--mirror-policy", action="store_true")
p.add_argument("--ref-mode", default="closed_random")
p.add_argument("--beta-ref-mode", default="adaptive_drift")
p.add_argument("--preview-time", default=None)
p.add_argument("--adaptive-shape", choices=["legacy", "smooth_v1", "smooth_task_v2"], default="legacy")
p.add_argument("--adaptive-speed-penalty", choices=["target", "envelope", "none"], default="target")
p.add_argument("--adaptive-path-scale-m", type=float, default=None)
args = p.parse_args()
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
sys.argv = [str(ROOT / "run.py"), "train", "continuous_drift_iwd",
            "--car-preset", "fullsize_racecar_ev", "--rnn", "--observable-gru",
            "--asymmetric", "--num-parallel", str(args.num_parallel), "--horizon", str(args.horizon),
            "--seq-length", str(args.seq_length), "--minibatch", str(args.minibatch), "--mini-epochs", "1",
            "--device", args.device, "--steer-actuator-profile", args.steer_actuator_profile,
            "--ref-mode", args.ref_mode, "--closed-random-radius", "2",
            "--continuing-task", "--randomize-tyre", "ev_drift",
            "--sensor-dr", "--actuator-dr",
            "--spawn-mode", "legacy", "--drift-spawn-trim-frac", "0",
            "--drift-spawn-spin-frac", "0", "--run-root", str(out),
            "--reward-mode", "adaptive_envelope", "--drift-mode",
            "--beta-ref-mode", args.beta_ref_mode]
if args.mirror_policy:
    sys.argv += ["--mirror-policy"]
if args.preview_time is not None:
    sys.argv += ["--preview-time", args.preview_time]
sys.argv += ["--adaptive-speed-penalty", args.adaptive_speed_penalty, "--adaptive-shape", args.adaptive_shape]
if args.adaptive_shape in ("smooth_v1", "smooth_task_v2"):
    sys.argv += ["--adaptive-performance-weight", "1.5", "--adaptive-drift-weight", "2.3"]
if args.adaptive_path_scale_m is not None:
    sys.argv += ["--adaptive-path-scale-m", str(args.adaptive_path_scale_m)]
if args.device == "cpu":
    torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("audit_run_config", ROOT / "run.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
runner = module.runner
agent = runner.algo_factory.create(runner.algo_name, base_name="plumbing", params=runner.params)
agent.init_tensors()
agent.obs = agent.env_reset()
assert agent.vec_env.env.steer_actuator_profile == args.steer_actuator_profile
assert agent.vec_env.env.adaptive_shape == args.adaptive_shape
assert agent.vec_env.env.adaptive_speed_penalty == args.adaptive_speed_penalty
assert agent.vec_env.env.adaptive_path_scale_m == args.adaptive_path_scale_m
agent.set_eval()
if args.device != "cpu":
    torch.cuda.synchronize()
start = time.perf_counter()
batch = agent.play_steps_rnn()
if args.device != "cpu":
    torch.cuda.synchronize()
rollout_seconds = time.perf_counter() - start
agent.prepare_dataset(batch)
scale = agent.vec_env.env.priv_scale[:2]
torch.testing.assert_close(agent.dataset.values_dict["beta_labels"], batch["states"][:, :2]/scale)
net = agent.model.a2c_network
before = net.beta_head[0].weight.detach().clone()
minibatch = agent.dataset[0]
agent.train_actor_critic(minibatch)
critic_params = dict(agent.central_value_net.model.named_parameters())
critic_before = {name: value.detach().clone() for name, value in critic_params.items()}
agent.central_value_net.train_critic(agent.central_value_net.dataset[0])
critic_delta = max(float((value-critic_before[name]).abs().max())
                   for name, value in critic_params.items())
assert critic_delta > 0
delta = (net.beta_head[0].weight.detach() - before).abs().max().item()
assert delta > 0
assert not agent.has_value_loss, "actor must not train a second value function"
assert net.beta_labels is None and net.beta_prediction is None
assert all(torch.isfinite(t).all() for t in net.parameters())
result = {"rollout_frames": int(batch["obses"].shape[0]),
          "device": args.device, "ref_mode": args.ref_mode,
          "mirror_policy": args.mirror_policy,
          "effective_lr_bounds": [agent.scheduler.min_lr, agent.scheduler.max_lr],
          "beta_ref_mode": args.beta_ref_mode, "preview_time": args.preview_time,
          "labels_aligned": True, "beta_head_parameter_change": delta,
          "actor_has_value_loss": agent.has_value_loss,
          "separate_critic": agent.has_central_value,
          "critic_parameter_change": critic_delta,
          "full_physics_sensor_actuator_dr": True,
          "rollout_seconds": rollout_seconds,
          "rollout_frames_per_second": batch["obses"].shape[0]/rollout_seconds,
          "peak_gpu_memory_MiB": torch.cuda.max_memory_allocated()/2**20 if args.device != "cpu" else None,
          "reserved_gpu_memory_MiB": torch.cuda.memory_reserved()/2**20 if args.device != "cpu" else None,
          "free_gpu_memory_MiB": torch.cuda.mem_get_info()[0]/2**20 if args.device != "cpu" else None,
          "checkpoint_saved": False}
(out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
agent.writer.close()
