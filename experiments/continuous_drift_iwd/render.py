"""Render one episode of the drift policy: reference, corridor and the car.

Produces
  * a GIF/MP4 animation of the run (car body, speed-coloured trajectory), and
  * a static summary PNG: reference centreline, corridor, the driven path
    coloured by speed, and a body marker every N control steps.

Usage
    python experiments/continuous_drift_iwd/render.py runs/drift_ev_v6 \
        --curv-scale 0.7 --grip-scale 1.0 --preview-time 0.5,1.0,1.6 \
        --offpath-scale 0.4 --out render_v6 --frames 4
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


def build_env(a, n):
    kw = dict(ref_mode=a.ref_mode, control_mode="pedals", train=True,
              curv_scale=a.curv_scale, curv_smooth=a.curv_smooth,
              preview_scale=a.preview_scale, preview_time=a.preview_time,
              flip_prob=a.flip_prob, beta_scale=a.beta_scale,
              spawn_mode="equilibrium", spawn_speed=a.spawn_speed,
              spawn_beta_frac=1.0, spawn_speed_min=args.spawn_speed_min,
              spawn_deficit_frac=args.spawn_deficit_frac,
              offpath_scale=a.offpath_scale, course_fail=a.course_fail,
              stall_speed=a.stall_speed, stall_window=a.stall_window,
              stall_move=a.stall_move, grip_scale=a.grip_scale,
              alive_bonus=a.alive_bonus, yaw_weight=a.yaw_weight,
              beta_weight=a.beta_weight, dir_weight=a.dir_weight,
              pos_weight=a.pos_weight)
    return ContinuousDriftIWDEnv(a.car_preset, n, a.device, **kw)


def resolve(ckpt):
    if os.path.isdir(ckpt) or ckpt.endswith("nn"):
        import glob
        cands = glob.glob(os.path.join(ckpt, "last_*.pth"))
        if not cands:
            cands = [os.path.join(ckpt, "continuous_drift_iwd.pth")]
        ckpt = max(cands, key=os.path.getmtime)
    return ckpt


def main(args):
    torch.manual_seed(args.seed)
    ckpt = resolve(args.checkpoint)
    n = args.n
    env = build_env(args, n)
    env.reset()
    env.is_done[:] = 1
    env.reset_done_envs()
    obs = env.obs()

    policy = Policy(ckpt, torch.device(args.device))
    policy.reset_hidden(torch.ones(n, dtype=torch.bool, device=args.device))

    traj = [[] for _ in range(n)]
    speeds = [[] for _ in range(n)]
    betas = [[] for _ in range(n)]
    brefs = [[] for _ in range(n)]
    headings = [[] for _ in range(n)]
    progress = [[] for _ in range(n)]
    epos = [[] for _ in range(n)]
    pid0 = [None] * n   # path id AT EPISODE START (re-randomised on respawn)
    gen0 = [0] * n      # ring generation at episode start
    deltas = [[] for _ in range(n)]
    throttles = [[] for _ in range(n)]
    alive = [0] * n
    end_step = [-1] * n

    for step in range(args.steps):
        a = policy.act(obs)
        obs, rew, done, info = env.step(a)
        policy.reset_hidden(done.bool())
        pos = env.s[:, :2].detach().cpu().numpy()
        v = env.es[:, 6].detach().cpu().numpy()
        b = env.es[:, 7].detach().cpu().numpy()
        b_ref = env.beta_ref.detach().cpu().numpy()
        psi = env.s[:, 2].detach().cpu().numpy()
        d = env.u_cmd[:, 0].detach().cpu().numpy()
        q = env.u_cmd[:, 1].detach().cpu().numpy()
        dn = done.detach().cpu().numpy()
        # on the terminal step the environment has ALREADY respawned, so the state
        # below belongs to the next episode: record the previous sample instead
        alive_now = dn == 0
        for k in range(n):
            if end_step[k] >= 0 or not alive_now[k]:
                if end_step[k] < 0:
                    end_step[k] = step + 1
                continue
                continue
            traj[k].append(pos[k].copy())
            speeds[k].append(float(v[k]))
            betas[k].append(float(b[k]))
            brefs[k].append(float(b_ref[k]))
            headings[k].append(float(psi[k]))
            progress[k].append(int(env.progress.detach().cpu().numpy()[k]))
            epos[k].append(float(env.signed_pos_diff.detach().cpu().numpy()[k]))
            if pid0[k] is None:
                pid0[k] = int(env.path_id.detach().cpu().numpy()[k])
                gen0[k] = int(env.path_generation.detach().cpu().numpy()[k])
            deltas[k].append(float(d[k]))
            throttles[k].append(float(q[k]))
            alive[k] += 1
            if dn[k] != 0:
                end_step[k] = step + 1
        if all(e >= 0 for e in end_step):
            break

    os.makedirs(args.out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Rectangle
    from matplotlib.collections import LineCollection

    n_show = min(args.show, n)
    refs = {}
    for k in range(n_show):
        pid = pid0[k]   # the id at episode start; env.path_id is re-randomised on respawn
        if (gen0[k], pid) not in refs:
            refs[(gen0[k], pid)] = env.path_bank[gen0[k], pid, :, :2].detach().cpu().numpy()
    corridor = env.offpath_limit

    # ---- static summary -------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon

    # the tracked reference point at every step: because the reference path
    # self-intersects, drawing the whole bank would mix branches, so plot exactly
    # what the car was asked to follow at each instant
    refxy = []
    for k in range(n_show):
        pr = np.array(progress[k])
        refxy.append(env.path_bank[gen0[k], pid0[k], pr, :2].detach().cpu().numpy())

    fig, axes = plt.subplots(3, n_show, figsize=(6.0 * n_show, 12.6), squeeze=False,
                             gridspec_kw={"height_ratios": [1.5, 1.0, 0.85]})
    for k in range(n_show):
        ax = axes[0][k]
        pts = np.array(traj[k])
        sp = np.array(speeds[k])
        segs = np.concatenate([pts.reshape(-1, 1, 2)[:-1], pts.reshape(-1, 1, 2)[1:]], axis=1)
        lc = LineCollection(segs, cmap="turbo", linewidth=3.0)
        lc.set_array(sp[:-1])
        lc.set_clim(0, 9.0)
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, fraction=0.042, pad=0.02, label="speed [m/s]")
        ax.plot(refxy[k][:, 0], refxy[k][:, 1], color="0.15", lw=1.1, alpha=0.75,
                label="tracked reference")
        for i in range(0, len(pts), max(args.frames, 1)):
            psi = headings[k][i]
            L, W = 4.5, 1.9
            c_, s_ = math.cos(psi), math.sin(psi)
            corners = np.array([[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2]])
            corners = corners @ np.array([[c_, -s_], [s_, c_]]).T + pts[i]
            ax.add_patch(Polygon(corners, closed=True, fill=False, ec="k", lw=0.6, alpha=0.6))
        ax.plot(pts[0, 0], pts[0, 1], "o", color="lime", ms=8, label="start")
        ax.plot(pts[-1, 0], pts[-1, 1], "X", color="red", ms=9, label="end")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=7)
        ax.set_title("env %d: %.1f s, mean %.2f m/s, |dbeta| %.3f"
                     % (k, len(pts) * env.dt, float(sp.mean()) if len(sp) else 0.0,
                        float(np.mean(np.abs(np.array(betas[k]) - np.array(brefs[k])))) if len(sp) else 0.0),
                     fontsize=10)

        ax2 = axes[1][k]
        t = np.arange(len(sp)) * env.dt
        ax2.plot(t, sp, color="tab:blue", lw=1.6, label="speed [m/s]")
        ax2.set_ylabel("speed [m/s]", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax2.grid(alpha=0.25)
        ax3 = ax2.twinx()
        ax3.plot(t, np.array(betas[k]), color="tab:red", lw=1.2, label="beta")
        ax3.plot(t, np.array(brefs[k]), color="tab:red", lw=1.0, ls="--", alpha=0.8, label="beta_ref")
        ax3.set_ylabel("beta [rad]", color="tab:red")
        ax3.tick_params(axis="y", labelcolor="tab:red")
        ax2.set_xlabel("time [s]")

        ax4 = axes[2][k]
        ax4.plot(t, np.array(epos[k]), color="tab:green", lw=1.3)
        ax4.axhline(corridor, color="tab:green", ls="--", lw=0.9, alpha=0.7)
        ax4.axhline(-corridor, color="tab:green", ls="--", lw=0.9, alpha=0.7)
        ax4.set_ylabel("lateral error [m]")
        ax4.set_xlabel("time [s]")
        ax4.grid(alpha=0.25)
        ax4.set_ylim(-corridor * 1.6, corridor * 1.6)
    fig.suptitle("drift policy: top-down (colour = speed, boxes every %d steps) and time series"
                 % max(args.frames, 1), fontsize=11)
    fig.tight_layout()
    png = os.path.join(args.out_dir, "summary.png")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print("wrote", png)

    # ---- animation ------------------------------------------------------
    if args.gif:
        n_anim = min(n_show, 2)
        fig, axes = plt.subplots(1, n_anim, figsize=(6.2 * n_anim, 5.8), squeeze=False)
        artists = []
        max_len = max(len(traj[k]) for k in range(n_anim))
        for k in range(n_anim):
            ax = axes[0][k]
            pts = np.array(traj[k])
            pad = 8.0
            ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
            ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
            ax.set_aspect("equal")
            ax.grid(alpha=0.25)
            rp = refxy[k]
            jumps = np.linalg.norm(np.diff(rp, axis=0), axis=1) > 1.0
            for seg_i in np.split(np.arange(len(rp)), np.nonzero(jumps)[0] + 1):
                if len(seg_i) > 1:
                    ax.plot(rp[seg_i, 0], rp[seg_i, 1], color="0.2", lw=1.2, alpha=0.7)
            line, = ax.plot([], [], color="tab:blue", lw=2.6)
            body = Polygon(np.zeros((4, 2)), closed=True, fc="tab:orange", ec="k", alpha=0.95)
            ax.add_patch(body)
            txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9,
                          bbox=dict(fc="white", alpha=0.85, ec="none"))
            artists.append((ax, line, body, txt, k))

        def init():
            return [a for ax, line, body, txt, k in artists for a in (line, body, txt)]

        def update(frame):
            for ax, line, body, txt, k in artists:
                m = min(frame + 1, len(traj[k]))
                pts = np.array(traj[k][:m])
                line.set_data(pts[:, 0], pts[:, 1])
                psi = headings[k][m - 1]
                L, W = 4.5, 1.9
                c_, s_ = math.cos(psi), math.sin(psi)
                corners = np.array([[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2]])
                body.set_xy(corners @ np.array([[c_, -s_], [s_, c_]]).T + pts[-1])
                txt.set_text("t=%.2f s   v=%.2f m/s\nbeta=%+.3f  beta_ref=%+.3f\ndelta=%+.2f  throttle=%.2f"
                             % (m * env.dt, speeds[k][m - 1], betas[k][m - 1], brefs[k][m - 1],
                                deltas[k][m - 1], throttles[k][m - 1]))
            return [a for ax, line, body, txt, k in artists for a in (line, body, txt)]

        anim = animation.FuncAnimation(fig, anim_update_factory(update) if False else update,
                                       frames=max_len, init_func=init,
                                       interval=1000 / args.fps, blit=False)
        gif = os.path.join(args.out_dir, "episode.gif")
        anim.save(gif, writer=animation.PillowWriter(fps=args.fps))
        plt.close(fig)
        print("wrote", gif)

    for k in range(n_show):
        sp = np.array(speeds[k])
        print("env %d: %.2f s, mean v %.2f m/s, |dbeta| %.3f, end_step %d"
              % (k, len(sp) * env.dt, float(sp.mean()) if len(sp) else 0.0,
                 float(np.mean(np.abs(np.array(betas[k]) - np.array(brefs[k])))), end_step[k]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--car-preset", default="fullsize_racecar_ev")
    ap.add_argument("--ref-mode", default="hybrid")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=640)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--show", type=int, default=3, help="how many envs to plot")
    ap.add_argument("--frames", type=int, default=20, help="body marker every N steps")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--out-dir", default="render_out")
    ap.add_argument("--beta-scale", type=float, default=0.15)
    ap.add_argument("--offpath-scale", type=float, default=0.4)
    ap.add_argument("--course-fail", type=float, default=1.6)
    ap.add_argument("--curv-scale", type=float, default=0.7)
    ap.add_argument("--curv-smooth", type=int, default=160)
    ap.add_argument("--preview-scale", type=float, default=1.0)
    ap.add_argument("--preview-time", type=str, default="0.5,1.0,1.6")
    ap.add_argument("--flip-prob", type=float, default=0.25)
    ap.add_argument("--stall-speed", type=float, default=2.5)
    ap.add_argument("--stall-window", type=int, default=20)
    ap.add_argument("--stall-move", type=float, default=1.5)
    ap.add_argument("--spawn-speed", type=float, default=4.5)
    ap.add_argument("--spawn-speed-min", type=float, default=None)
    ap.add_argument("--spawn-deficit-frac", type=float, default=0.0)
    ap.add_argument("--grip-scale", type=float, default=1.0)
    ap.add_argument("--alive-bonus", type=float, default=2.5)
    ap.add_argument("--yaw-weight", type=float, default=6.0)
    ap.add_argument("--beta-weight", type=float, default=100.0)
    ap.add_argument("--dir-weight", type=float, default=4.0)
    ap.add_argument("--pos-weight", type=float, default=8.0)
    args = ap.parse_args()
    if args.spawn_speed_min is None:
        args.spawn_speed_min = args.spawn_speed
    main(args)
