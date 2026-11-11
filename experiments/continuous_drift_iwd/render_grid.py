"""Seeded-road grid render: the SAME reference roads across every grip level.

The path bank is generated from the torch RNG at environment construction, so
re-seeding identically for each grip produces the same roads and the same spawn
points.  That makes it possible to compare the policy across mu on exactly the
same task instances, which is the protocol the training evaluation should use.

Usage
    python experiments/continuous_drift_iwd/render_grid.py runs/drift_ev_v6 \
        --grips 0.5,0.75,1.0 --n 3 --seeds 7,11,23 --out grid.png
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))
sys.path.insert(0, os.path.join(ROOT, "utils"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.continuous_drift_iwd import ContinuousDriftIWDEnv  # noqa: E402
from eval import Policy  # noqa: E402


def resolve(ckpt):
    if os.path.isdir(ckpt) or ckpt.endswith("nn"):
        import glob
        cands = glob.glob(os.path.join(ckpt, "last_*.pth"))
        if not cands:
            cands = [os.path.join(ckpt, "continuous_drift_iwd.pth")]
        ckpt = max(cands, key=os.path.getmtime)
    return ckpt


def rollout(a, grip, seed, policy):
    """One episode per env, all envs sharing one seed => identical roads."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = dict(ref_mode=a.ref_mode, control_mode="pedals", train=True,
              curv_scale=a.curv_scale, curv_smooth=a.curv_smooth,
              preview_time=a.preview_time, preview_scale=a.preview_scale,
              flip_prob=a.flip_prob, beta_scale=a.beta_scale,
              spawn_mode="equilibrium", spawn_speed=a.spawn_speed,
              spawn_beta_frac=1.0, spawn_speed_min=a.spawn_speed_min,
              spawn_deficit_frac=a.spawn_deficit_frac,
              offpath_scale=a.offpath_scale, course_fail=a.course_fail,
              stall_speed=a.stall_speed, stall_window=a.stall_window,
              stall_move=a.stall_move, grip_scale=grip,
              alive_bonus=a.alive_bonus, yaw_weight=a.yaw_weight,
              beta_weight=a.beta_weight, dir_weight=a.dir_weight,
              pos_weight=a.pos_weight)
    env = ContinuousDriftIWDEnv(a.car_preset, a.n, a.device, **kw)
    env.reset()
    env.is_done[:] = 1
    env.reset_done_envs()
    obs = env.obs()
    if getattr(a, "scenario", None):
        import numpy as _np
        sc = _np.load(a.scenario)
        all_ids = [int(x) for x in str(a.scenario_ids).split(",")] if getattr(a, "scenario_ids", None) else list(range(len(sc["s"])))
        sid = all_ids[getattr(a, "current_scenario_id", 0) % len(all_ids)]
        idx = [sid]
        sub = {"path": sc["path"][idx], "progress": sc["progress"][idx],
               "s": sc["s"][idx], "u_cmd": sc["u_cmd"][idx]}
        obs = env.apply_scenario(sub)
    policy.reset_hidden(torch.ones(a.n, dtype=torch.bool, device=a.device))

    rec = [dict(xy=[], psi=[], v=[], beta=[], bref=[], prog=[], d=[], q=[]) for _ in range(a.n)]
    pid0 = [None] * a.n
    gen0 = [0] * a.n
    end = [-1] * a.n
    for step in range(a.steps):
        act = policy.act(obs)
        obs, rew, done, info = env.step(act)
        policy.reset_hidden(done.bool())
        alive = done.detach().cpu().numpy() == 0
        xy = env.s[:, :2].detach().cpu().numpy()
        v = env.es[:, 6].detach().cpu().numpy()
        b = env.es[:, 7].detach().cpu().numpy()
        br = env.beta_ref.detach().cpu().numpy()
        psi = env.s[:, 2].detach().cpu().numpy()
        pr = env.progress.detach().cpu().numpy()
        dd = env.u_cmd[:, 0].detach().cpu().numpy()
        qq = env.u_cmd[:, 1].detach().cpu().numpy()
        for k in range(a.n):
            if end[k] >= 0 or not alive[k]:
                if end[k] < 0:
                    end[k] = step + 1
                continue
            if pid0[k] is None:
                pid0[k] = int(env.path_id.detach().cpu().numpy()[k])
                gen0[k] = int(env.path_generation.detach().cpu().numpy()[k])
            rec[k]["xy"].append(xy[k].copy())
            rec[k]["psi"].append(float(psi[k]))
            rec[k]["v"].append(float(v[k]))
            rec[k]["beta"].append(float(b[k]))
            rec[k]["bref"].append(float(br[k]))
            rec[k]["prog"].append(int(pr[k]))
            rec[k]["d"].append(float(dd[k]))
            rec[k]["q"].append(float(qq[k]))
        if all(e >= 0 for e in end):
            break
    for k in range(a.n):
        rec[k]["pid"] = pid0[k]
        rec[k]["gen"] = gen0[k]
    return env, rec


def main(args):
    ckpt = resolve(args.checkpoint)
    policy = Policy(ckpt, torch.device(args.device))
    grips = [float(x) for x in args.grips.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    # one episode per column, so every column is exactly one seeded road
    n = len(seeds)
    args.n = 1

    # one column per seed; inside a column every grip runs on the same roads
    grids = []
    for si, seed in enumerate(seeds[:n]):
        args.current_scenario_id = si
        per_grip = []
        for grip in grips:
            env, rec = rollout(args, grip, seed, policy)
            per_grip.append((grip, env, rec))
        grids.append((seed, per_grip))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon

    nrow, ncol = len(grips), n
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.8 * nrow), squeeze=False)
    for ci, (seed, per_grip) in enumerate(grids):
        for ri, (grip, env, rec) in enumerate(per_grip):
            ax = axes[ri][ci]
            r = rec[0]
            ts = len(r["xy"]) * env.dt
            if not r["xy"]:
                ax.set_axis_off()
                continue
            pts = np.array(r["xy"])
            sp = np.array(r["v"])
            segs = np.concatenate([pts.reshape(-1, 1, 2)[:-1], pts.reshape(-1, 1, 2)[1:]], axis=1)
            lc = LineCollection(segs, cmap="turbo", linewidth=2.6)
            lc.set_array(sp[:-1])
            lc.set_clim(0, 10.0)
            ax.add_collection(lc)
            bank = env.path_bank[r["gen"], r["pid"], :, :2].detach().cpu().numpy()
            pr = np.array(r["prog"])
            lo, hi = max(pr.min() - 60, 0), min(pr.max() + 60, len(bank))
            ax.plot(bank[lo:hi, 0], bank[lo:hi, 1], color="0.2", lw=1.0, alpha=0.8)
            for i in range(0, len(pts), max(args.frames, 1)):
                c_, s_ = math.cos(r["psi"][i]), math.sin(r["psi"][i])
                L, W = 4.5, 1.9
                corners = np.array([[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2]])
                ax.add_patch(Polygon(corners @ np.array([[c_, -s_], [s_, c_]]).T + pts[i],
                                     closed=True, fill=False, ec="k", lw=0.55, alpha=0.55))
            ax.plot(pts[0, 0], pts[0, 1], "o", color="lime", ms=7)
            ax.plot(pts[-1, 0], pts[-1, 1], "X", color="red", ms=8)
            ax.set_aspect("equal")
            ax.grid(alpha=0.25)
            db = float(np.mean(np.abs(np.array(r["beta"]) - np.array(r["bref"]))))
            ax.set_title("seed %d  mu %.2f  %.1f s  |db| %.3f  v %.2f"
                         % (seed, grip, ts, db, float(sp.mean())), fontsize=9)
            if ci == 0:
                ax.set_ylabel("mu %.2f" % grip, fontsize=11)
            if ri == 0:
                ax.set_xlabel("seed %d" % seed, fontsize=10)
    fig.suptitle("same seeded roads across grip levels (colour = speed)", fontsize=12)
    fig.tight_layout()
    out = args.out
    if not out.endswith(".png"):
        out = os.path.join(out, "grid.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=115)
    print("wrote", out)

    # numeric table
    print("\nseed  grip  dur[s]  full  mean_v  |dbeta|")
    for seed, per_grip in grids:
        for ci, (grip, env, rec) in enumerate(per_grip):
            r = rec[0]
            if not r["xy"]:
                continue
            dur = len(r["xy"]) * env.dt
            db = float(np.mean(np.abs(np.array(r["beta"]) - np.array(r["bref"]))))
            print("%4d  %.2f  %6.2f  %4s  %6.2f  %.3f"
                  % (seed, grip, dur, "yes" if dur > 29.5 else "no",
                     float(np.mean(r["v"])), db))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--car-preset", default="fullsize_racecar_ev")
    ap.add_argument("--ref-mode", default="hybrid")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seeds", type=str, default="7,11,23")
    ap.add_argument("--grips", type=str, default="0.5,0.75,1.0")
    ap.add_argument("--steps", type=int, default=640)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--out", default="grid.png")
    ap.add_argument("--scenario", type=str, default=None,
                    help="frozen scenario npz: same roads and starts for every grip")
    ap.add_argument("--scenario-ids", type=str, default=None)
    ap.add_argument("--beta-scale", type=float, default=0.15)
    ap.add_argument("--offpath-scale", type=float, default=0.4)
    ap.add_argument("--course-fail", type=float, default=1.6)
    ap.add_argument("--curv-scale", type=float, default=0.7)
    ap.add_argument("--curv-smooth", type=int, default=160)
    ap.add_argument("--preview-time", type=str, default="0.5,1.0,1.6")
    ap.add_argument("--preview-scale", type=float, default=1.0)
    ap.add_argument("--flip-prob", type=float, default=0.25)
    ap.add_argument("--stall-speed", type=float, default=2.5)
    ap.add_argument("--stall-window", type=int, default=20)
    ap.add_argument("--stall-move", type=float, default=1.5)
    ap.add_argument("--spawn-speed", type=float, default=4.5)
    ap.add_argument("--spawn-speed-min", type=float, default=3.15)
    ap.add_argument("--spawn-deficit-frac", type=float, default=0.3)
    ap.add_argument("--alive-bonus", type=float, default=2.5)
    ap.add_argument("--yaw-weight", type=float, default=6.0)
    ap.add_argument("--beta-weight", type=float, default=100.0)
    ap.add_argument("--dir-weight", type=float, default=4.0)
    ap.add_argument("--pos-weight", type=float, default=8.0)
    args = ap.parse_args()
    main(args)
