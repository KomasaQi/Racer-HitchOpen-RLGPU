"""Run the trained drift policy inside CarSim (no OU disturbance).

One episode per scenario: the car starts from rest at the origin, follows one of our
own reference roads, and the shared task logic scores it.
"""
import json
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))
sys.path.insert(0, os.path.join(ROOT, "utils"))

from envs.carsim_drift_env import CarSimDriftEnv  # noqa: E402
from experiments.continuous_drift_iwd.eval import Policy, physical_action  # noqa: E402
from experiments.carsim_bridge.steer_metrics import steering_metrics, format_metrics  # noqa: E402

CKPT = os.path.join(ROOT, "experiments", "continuous_drift_iwd", "runs",
                os.environ.get("CARSIM_RUN", "drift_ev_v17"), "nn")
RUN_DIR = os.path.dirname(CKPT)
# select by the EPOCH NUMBER, not the filename: "ep_90" sorts after "ep_290"
# alphabetically, so the old loop silently evaluated the wrong checkpoint
_cands = [os.path.join(CKPT, f) for f in os.listdir(CKPT)
          if f.startswith("last_") and f.endswith(".pth")]
if not _cands:
    raise FileNotFoundError("no last_*.pth in " + CKPT)
CKPT_FILE = max(_cands, key=lambda p: int(re.search(r"ep_(\d+)", p).group(1))
                if re.search(r"ep_(\d+)", p) else -1)


def run(scenario, policy_path, steps=600, seed=0, entry_speed=0.0):
    torch.manual_seed(seed)
    # reconstruct the trained task configuration, otherwise the actor sees a
    # different preview geometry from the one it was trained with
    with open(os.path.join(RUN_DIR, "env_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("plant", "control_mode", "train", "ref_mode_cli",
                "disturbance_param", "randomize_param", "quiet", "num_paths"):
        cfg.pop(key, None)
    cfg.pop("curv_scale", None)      # a list here, the env wants the tuple marker
    cfg["curv_scale"] = float(os.environ.get("CARSIM_CURV", "0.7"))
    # FEASIBILITY: a drifting car can hold at most about eta*mu*g/V^2 of curvature.  At
    # 6 m/s and kappa 0.40 the demand is 12 m/s^2, above even mu = 1.0, so the car MUST run
    # wide no matter how it is controlled.  Scale the reference down to what the friction
    # can supply, using eta 0.65 for a sliding rear axle.
    if os.environ.get("CARSIM_FEASIBLE_CURV") is not None:
        _v_tgt = float(os.environ.get("CARSIM_VTGT", "6.0"))
        _mu = float(scenario.split("mu")[-1]) / 100.0
        _eta = float(os.environ["CARSIM_FEASIBLE_CURV"])
        _kmax = _eta * _mu * 9.81 / (_v_tgt * _v_tgt)
        cfg["curv_scale"] = min(cfg["curv_scale"], _kmax / 0.12)
    # A CarSim episode starts from REST and needs about two seconds to build 4.5 m/s,
    # so the training stall window of 20 steps (1 s) fires on every legitimate launch:
    # measured, no_progress_count reached exactly 20 at t = 1.00 s with the car already
    # at 2.26 m/s and accelerating.
    cfg["stall_window"] = 80
    # a drifting car legitimately needs more room than a gripping one: the trace showed it
    # running progressively wide while holding 10-11 degrees of sideslip, and ending when
    # it touched a 3.8 m corridor.  Allow an override so the corridor can be studied.
    if os.environ.get("CARSIM_OFFPATH") is not None:
        cfg["offpath_scale"] = float(os.environ["CARSIM_OFFPATH"])
    if os.environ.get("CARSIM_BRAKE_MARGIN") is not None:
        cfg["brake_margin"] = float(os.environ["CARSIM_BRAKE_MARGIN"])
    if os.environ.get("CARSIM_DEADBAND") is not None:
        cfg["steer_deadband"] = float(os.environ["CARSIM_DEADBAND"])
    env = CarSimDriftEnv(scenario, **cfg)
    # one physical episode: never tensor-respawn while the solver keeps its state
    env.train = False
    obs = env.reset()
    env.asymmetric = True
    # rolling entry: training spawned at 4.5 m/s on a drift equilibrium, so let
    # the car reach that speed in a straight line before the policy takes over.
    # This is evaluation-only and tests whether the rest start is what breaks it.
    if entry_speed > 0.0:
        # Launch along the REFERENCE, not straight: a straight launch on a
        # curving road leaves the car off the path by the takeover point.
        launch = torch.zeros((1, 3), device="cuda:0")
        for _ in range(200):
            # pure pursuit toward a point 3 m down the reference
            pr = min(int(env.progress[0].item()) + 80, env.num_waypoints - 1)
            gp, pid = int(env.path_generation[0].item()), int(env.path_id[0].item())
            ref = env.path_bank[gp, pid, pr, :2].cpu().numpy()
            cur = env.s[0, :2].cpu().numpy()
            psi = float(env.s[0, 2])
            des = float(np.arctan2(ref[1] - cur[1], ref[0] - cur[0]))
            err = float(np.arctan2(np.sin(des - psi), np.cos(des - psi)))
            launch[0, 0] = float(np.clip(2.5 * err, -0.35, 0.35))
            # BUG: the launch command only set the steering -- the throttle stayed zero, so the
            # car could never accelerate to the requested entry speed and every "rolling entry"
            # run was silently a from-rest run.  The throttle has to be opened for the launch.
            launch[0, 1] = 0.65
            obs, _, done, _ = env.step(launch)
            v_now = float(env.es.reshape(1, -1)[0, 6])
            if v_now >= entry_speed:
                break
        # the launch is not part of the scored episode: re-anchor the reference
        # and clear the task counters so the 30 s timer starts here
        obs = env.begin_policy_episode()
    pol = Policy(policy_path, torch.device("cuda:0"))
    pol.reset_hidden(torch.ones(1, dtype=torch.bool, device="cuda:0"))
    speed, beta_err, epos, n = 0.0, 0.0, 0.0, 0
    trace = []
    dsteer, prev_steer, sign_flips, prev_dsign = 0.0, None, 0, None
    applied = []
    done_step = steps
    for k in range(steps):
        actor_obs = obs["obs"] if isinstance(obs, dict) else obs
        # rl_games clips to [-1, 1] then rescales onto the action space; without
        # this the raw steering 0.2 became 0.2 rad here and 0.09 rad in training
        raw = pol.act(actor_obs.float())
        a = physical_action(raw, env)
        s_now = float(a[0, 0])
        if prev_steer is not None:
            d = s_now - prev_steer
            dsteer += abs(d)
            sg = 1 if d > 0 else -1
            if prev_dsign is not None and sg != prev_dsign and abs(d) > 0.01:
                sign_flips += 1
            prev_dsign = sg
        prev_steer = s_now
        applied.append(float(env.u_cmd[0, 0]))
        obs, rew, done, info = env.step(a)
        # every control step, from the state itself; the old sampler ran at 4 Hz and read
        # beta out of a reshaped extended state, and a direct measurement on the same
        # checkpoint gave 0.013 rad against the 0.177 reported here
        if True:
            es = env.es.reshape(1, -1)
            br = env.beta_ref.reshape(1, -1)
            sp = env.signed_pos_diff.reshape(1, -1)
            _db = float(es[0, 7] - br[0, 0])
            trace.append(((k + 1) * float(env.dt),
                          float(torch.hypot(env.s[0, 3], env.s[0, 4])),
                          float(es[0, 7]), float(br[0, 0]), _db,
                          float(sp[0, 0]), float(env.u_cmd[0, 0]),
                          float(env.u_cmd[0, 1]), float(env.u_cmd[0, 2])))
            # the extended state is not refreshed by begin_policy_episode() until the
            # first step, which is why this used to print 0.00; the world velocity is
            # always current
            speed += float(torch.hypot(env.s[0, 3], env.s[0, 4]))
            beta_err += float((es[0, 7] - br[0, 0]).abs())
            epos += float(sp[0, 0].abs())
            n += 1
            if torch.is_tensor(done) and int(done.sum()) != 0:
                done_step = k + 1
                break
    env.close()
    _reason = ""
    try:
        if getattr(env, "_term_reason", None) is not None:
            _reason = str(int(env._term_reason.reshape(-1)[0]))
    except Exception:
        _reason = "?"
    _out = os.environ.get("CARSIM_OUT", ".")
    with open(os.path.join(_out, "trace_%s.csv" % scenario), "w") as fh:
        fh.write("t,v,beta,beta_ref,dbeta,epos,steer,thr,brk\n")
        for row in trace:
            fh.write(",".join("%.6f" % float(x) for x in row) + "\n")
    return {"scenario": scenario, "steps": done_step, "reason": _reason,
            "dsteer": dsteer / max(n, 1), "flips": sign_flips,
            "metrics": steering_metrics(applied, dt=env.dt),
            "seconds": done_step * env.dt,
            "speed": speed / max(n, 1),
            "beta_err": beta_err / max(n, 1),
            "epos": epos / max(n, 1)}


ENTRY = float(os.environ.get("CARSIM_ENTRY_SPEED", "0"))

if __name__ == "__main__":
    print("checkpoint:", os.path.basename(CKPT_FILE))
    rows = []
    for sc in ("flat_mu050", "flat_mu075", "flat_mu100"):
        r = run(sc, CKPT_FILE, entry_speed=ENTRY)
        rows.append(r)
        print("%-12s %6.1f s  v %5.2f m/s  |dbeta| %.3f  |e_pos| %.2f m  reason %s"
              % (r["scenario"], r["seconds"], r["speed"], r["beta_err"], r["epos"],
                 r.get("reason", "?")))
        print("             steer: " + format_metrics(r["metrics"]))
