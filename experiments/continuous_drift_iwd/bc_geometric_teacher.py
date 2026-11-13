"""Warm-start the 27->3 GRU from an actor-observable geometric teacher.

The teacher follows the same closed-random A0 task with corrected pure pursuit
and a speed PI loop.  The student receives only the production 27 channels.
The output is a normal rl_games checkpoint and is subsequently fine-tuned with
PPO; no teacher signal is present at deployment.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.continuous_drift_iwd import ContinuousDriftIWDEnv  # noqa: E402
from distill import Actor, load_actor_weights  # noqa: E402
from observable_teacher import ObservableEnvelopeTeacher  # noqa: E402


def physical_to_raw(action, env):
    low = torch.as_tensor(env.action_space.low, device=action.device, dtype=action.dtype)
    high = torch.as_tensor(env.action_space.high, device=action.device, dtype=action.dtype)
    return (2.0 * (action - low) / (high - low) - 1.0).clamp(-1.0, 1.0)


def raw_to_physical(raw, env):
    low = torch.as_tensor(env.action_space.low, device=raw.device, dtype=raw.dtype)
    high = torch.as_tensor(env.action_space.high, device=raw.device, dtype=raw.dtype)
    return low + 0.5 * (raw.clamp(-1.0, 1.0) + 1.0) * (high - low)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=370000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dagger", action="store_true",
                    help="execute a teacher/student blend to collect recovery states")
    ap.add_argument("--blend-start", type=float, default=0.8)
    ap.add_argument("--blend-end", type=float, default=0.15)
    ap.add_argument("--pedal-weight", type=float, default=0.35)
    ap.add_argument("--long-weight", type=float, default=0.0,
                    help="direct loss on executable throttle-minus-brake demand")
    ap.add_argument("--speed-margin", type=float, default=0.45,
                    help="fraction of the local friction-limit speed used by the teacher")
    ap.add_argument("--speed-cap", type=float, default=22.0)
    ap.add_argument("--closed-random-radius", type=float, default=2.0,
                    help="geometric scale of the closed-random A0 path bank")
    ap.add_argument("--teacher-mode", choices=("causal", "privileged"), default="causal",
                    help="causal uses only deployable sensor/history signals; privileged is archival")
    ap.add_argument("--mu-floor", type=float, default=1.0623)
    ap.add_argument("--mu-ceil", type=float, default=2.1246)
    ap.add_argument("--mu-explore-rate", type=float, default=0.18,
                    help="causal grip-hypothesis increase per second while tracking safely")
    ap.add_argument("--mu-retreat-rate", type=float, default=0.90,
                    help="causal grip-hypothesis decrease per second after observable distress")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", args.n, args.device, ref_mode="closed_random",
        closed_random_radius=args.closed_random_radius, plant="fast", control_mode="pedals",
        train=True, asymmetric=True, continuing_task=True,
        reward_mode="adaptive_envelope", drift_mode=True,
        spawn_beta_frac=0.0, drift_spawn_spin_frac=0.0,
        drift_spawn_trim_frac=0.0, speed_target_mode="limit",
        speed_margin=args.speed_margin, speed_cap=args.speed_cap, preview_time="0.35,0.7,1.2",
        steer_actuator_tau=0.08, steer_rate_limit=1.5,
        randomize_param={"D": [1.0623, 2.1246]}, random_seed=args.seed)
    actor = Actor(27).to(dev)
    used, total = load_actor_weights(actor, args.source)
    print("initialised %d/%d actor tensors" % (used, total), flush=True)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    source_model = {k.replace("_orig_mod.", ""): v.float()
                    for k, v in checkpoint["model"].items()}
    mean = source_model["running_mean_std.running_mean"].to(dev)
    var = source_model["running_mean_std.running_var"].to(dev)
    if mean.numel() != 27:
        raise ValueError("source actor normalizer is not 27-dimensional")
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    obs = env.reset()
    hidden = torch.zeros(1, args.n, 64, device=dev)
    causal_teacher = ObservableEnvelopeTeacher(
        args.n, dev, speed_margin=args.speed_margin, speed_cap=args.speed_cap,
        mu_floor=args.mu_floor, mu_ceil=args.mu_ceil,
        explore_rate=args.mu_explore_rate, retreat_rate=args.mu_retreat_rate)
    losses = []
    for step in range(args.steps):
        actor_obs = obs["obs"] if isinstance(obs, dict) else obs
        if args.teacher_mode == "privileged":
            mu = (env.plant.col("D") * env.plant.tyre_scale.squeeze(1)).clamp(0.3, 3.0)
            kap = torch.maximum(env.kappa_ref.abs(),
                                env.kappa_preview.abs().amax(1)).clamp_min(1e-2)
            v_target = (args.speed_margin * torch.sqrt(mu * 9.81 / kap)).clamp(
                2.0, args.speed_cap)
            wheel_v = env.s[:, 6:10] * env.wheel_radius
            v_observed = wheel_v[:, :2].mean(1).clamp_min(0.0)
            longitudinal = (0.12 + 0.18 * (v_target - v_observed)).clamp(-1.0, 1.0)
            teacher_physical = torch.stack((
                env.delta_pp.clamp(-env.max_steer, env.max_steer),
                longitudinal.clamp(min=0.0), (-longitudinal).clamp(min=0.0)), dim=1)
            teacher_mu_mean = float(mu.mean())
        else:
            teacher_physical, diag = causal_teacher.command(env)
            teacher_mu_mean = float(diag["mu_hat"].mean())
        target = physical_to_raw(teacher_physical, env)

        x = torch.clamp((actor_obs - mean) / torch.sqrt(var + 1e-5), -5.0, 5.0)
        pred, next_hidden = actor(x, hidden.detach())
        # Steering is the A0 bottleneck; pedal supervision remains present but
        # lower-weighted because the target speed is inferred over GRU history.
        loss_steer = nn.functional.smooth_l1_loss(pred[:, 0], target[:, 0])
        loss_pedal = nn.functional.smooth_l1_loss(pred[:, 1:], target[:, 1:])
        pred_long = 0.5 * (pred[:, 1] - pred[:, 2])
        target_long = teacher_physical[:, 1] - teacher_physical[:, 2]
        loss_long = nn.functional.smooth_l1_loss(pred_long, target_long)
        loss = (loss_steer + args.pedal_weight * loss_pedal +
                args.long_weight * loss_long)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))

        if args.dagger:
            frac = step / max(args.steps - 1, 1)
            teacher_blend = args.blend_start + frac * (args.blend_end - args.blend_start)
            student_physical = raw_to_physical(pred.detach(), env)
            executed = teacher_blend * teacher_physical + (1.0 - teacher_blend) * student_physical
        else:
            executed = teacher_physical
        obs, _, done, _ = env.step(executed)
        hidden = next_hidden.detach()
        hidden[:, done.bool(), :] = 0.0
        if args.teacher_mode == "causal" and done.any():
            causal_teacher.reset(done.bool())
        if (step + 1) % 50 == 0:
            print("step %4d loss %.6f mean50 %.6f mu_hat %.3f" %
                  (step + 1, losses[-1], sum(losses[-50:]) / 50.0,
                   teacher_mu_mean), flush=True)

    dst = {k.replace("_orig_mod.", ""): k for k in checkpoint["model"]}
    replaced = 0
    for key, value in actor.state_dict().items():
        checkpoint_key = dst.get(key)
        if checkpoint_key is not None:
            checkpoint["model"][checkpoint_key] = value.detach().cpu()
            replaced += 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(checkpoint, args.out)
    print("wrote %s; replaced %d tensors; final mean50 %.6f" %
          (args.out, replaced, sum(losses[-50:]) / min(len(losses), 50)), flush=True)


if __name__ == "__main__":
    main()
