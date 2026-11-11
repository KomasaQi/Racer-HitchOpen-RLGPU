"""Distil the privileged-observation policy into the real-car observation actor.

The observability ablation showed that giving the actor beta / V / vx / vy turns
the task from ~12.8 s into ~15.6 s mean survival and removes essentially every
lateral tracking failure, i.e. the tracking limit is an observability limit.  A
teacher-student pass is the cheapest way to transfer that skill: roll the
privileged policy through the environment, train the 30 channel GRU actor to
reproduce its (pre-tanh) action on the same states, then hand the result to PPO
as a warm start.

    python distill.py teacher.pth student_src.pth out.pth [--steps 6000]

``teacher.pth`` is an ``--obs-privileged`` checkpoint (34 channels) and
``student_src.pth`` any 30 channel checkpoint whose normalisation and value head
should be kept (v57 is fine).  The output keeps everything from ``student_src``
except the actor, which is replaced by the distilled one.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))
sys.path.insert(0, os.path.join(ROOT, "utils"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "continuous_drift_iwd"))

from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
from eval import Policy


class Actor(nn.Module):
    """rl_games compatible actor: ``a2c_network.rnn.rnn`` + ``actor_mlp`` + ``mu``."""

    def __init__(self, obs_dim, units=(256, 128, 64), hidden=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.a2c_network = nn.Module()
        self.a2c_network.rnn = nn.Module()
        self.a2c_network.rnn.rnn = nn.GRU(obs_dim, hidden, num_layers=1, batch_first=True)
        layers = []
        prev = hidden
        for i, u in enumerate(units):
            layers += [nn.Linear(prev, u), nn.ELU()]
            prev = u
        self.a2c_network.actor_mlp = nn.Sequential(*layers)
        self.a2c_network.mu = nn.Linear(prev, 3)

    def forward(self, x, h):
        out, h = self.a2c_network.rnn.rnn(x.unsqueeze(1), h)
        y = self.a2c_network.actor_mlp(out.squeeze(1))
        return self.a2c_network.mu(y), h


def load_actor_weights(actor, path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    w = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    own = actor.state_dict()
    used = 0
    for k, v in own.items():
        if k in w and w[k].shape == v.shape:
            own[k] = w[k].float()
            used += 1
    actor.load_state_dict(own)
    return used, len(own)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teacher")
    ap.add_argument("student_src")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--curv-scale", type=float, default=0.7)
    args = ap.parse_args()

    dev = torch.device(args.device)
    teacher = Policy(args.teacher, dev)

    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", args.n, args.device, ref_mode="hybrid", control_mode="pedals",
        train=True, asymmetric=True, obs_privileged=True, curv_scale=args.curv_scale,
        beta_scale=0.15, spawn_mode="equilibrium", spawn_speed=4.5, spawn_beta_frac=1.0,
        offpath_scale=1.4, course_fail=1.6, stall_speed=2.5, speed_target=3.8)
    student = Actor(env.num_states - 4).to(dev)
    used, total = load_actor_weights(student, args.student_src)
    print("initialised %d/%d actor tensors from %s" % (used, total, os.path.basename(args.student_src)))

    # the student normalises with the 30 channel statistics of the source run
    sd_src = torch.load(args.student_src, map_location="cpu", weights_only=False)["model"]
    w_src = {k.replace("_orig_mod.", ""): v.float() for k, v in sd_src.items()}
    mean = w_src["running_mean_std.running_mean"].to(dev)
    var = w_src["running_mean_std.running_var"].to(dev)
    assert mean.numel() == student.obs_dim, (mean.numel(), student.obs_dim)

    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    env.reset()
    env.is_done[:] = 1
    env.reset_done_envs()
    def actor_obs(o):
        return o["obs"] if isinstance(o, dict) else o

    obs = actor_obs(env.obs())                    # 34 channels: teacher sees all
    h_t = torch.zeros(1, args.n, 64, device=dev)
    h_s = torch.zeros(1, args.n, 64, device=dev)
    teacher.reset_hidden(torch.ones(args.n, dtype=torch.bool, device=dev))

    student.eval()
    losses = []
    for step in range(args.steps):
        with torch.no_grad():
            target = teacher.act(obs)
        student.train()
        x = torch.clamp((obs[:, : student.obs_dim] - mean) / torch.sqrt(var + 1e-5), -5.0, 5.0)
        pred, h_s = student(x, h_s.detach())
        loss = nn.functional.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))

        done_prev = env.is_done.clone()
        obs, _, _, _ = env.step(target)
        obs = actor_obs(obs)
        newly = (done_prev == 0) & (env.is_done != 0)
        h_s[:, newly, :] = 0.0
        h_t[:, newly, :] = 0.0
        env.is_done[:] = 0
        if (step + 1) % 500 == 0:
            print("step %5d  mse %.5f  (last 500 mean %.5f)" % (
                step + 1, losses[-1], sum(losses[-500:]) / 500), flush=True)

    sd = torch.load(args.student_src, map_location="cpu", weights_only=False)
    model = sd["model"]
    dst = {k.replace("_orig_mod.", ""): k for k in model}
    replaced = 0
    with torch.no_grad():
        for k, v in student.state_dict().items():
            key = dst.get(k)
            if key is None:
                continue
            model[key] = v.detach().cpu()
            replaced += 1
    torch.save(sd, args.out)
    print("wrote %s, replaced %d actor tensors" % (args.out, replaced))


if __name__ == "__main__":
    main()
