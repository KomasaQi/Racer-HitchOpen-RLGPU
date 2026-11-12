"""Run the trained drift policy inside CarSim (no OU disturbance).

One episode per scenario: the car starts from rest at the origin, follows one of our
own reference roads, and the shared task logic scores it.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))
sys.path.insert(0, os.path.join(ROOT, "utils"))

from envs.carsim_drift_env import CarSimDriftEnv  # noqa: E402
from experiments.continuous_drift_iwd.eval import Policy  # noqa: E402
from experiments.carsim_bridge.steer_metrics import steering_metrics, format_metrics  # noqa: E402

CKPT = os.path.join(ROOT, "experiments", "continuous_drift_iwd", "runs",
                os.environ.get("CARSIM_RUN", "drift_ev_v17"), "nn")
RUN_DIR = os.path.dirname(CKPT)
CKPT_FILE = None
for f in os.listdir(CKPT):
    if f.startswith("last_") and f.endswith(".pth"):
        CKPT_FILE = os.path.join(CKPT, f)


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
    cfg["curv_scale"] = 0.7
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
        launch = torch.zeros((1, 3), device="cuda:0")
        launch[0, 1] = 0.35
        for _ in range(200):
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
    dsteer, prev_steer, sign_flips, prev_dsign = 0.0, None, 0, None
    applied = []
    done_step = steps
    for k in range(steps):
        actor_obs = obs["obs"] if isinstance(obs, dict) else obs
        a = pol.act(actor_obs.float())
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
        if k % 5 == 0:
            es = env.es.reshape(1, -1)
            br = env.beta_ref.reshape(1, -1)
            sp = env.signed_pos_diff.reshape(1, -1)
            speed += float(es[0, 6])
            beta_err += float((es[0, 7] - br[0, 0]).abs())
            epos += float(sp[0, 0].abs())
            n += 1
            if torch.is_tensor(done) and int(done.sum()) != 0:
                done_step = k + 1
                break
    env.close()
    return {"scenario": scenario, "steps": done_step,
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
        print("%-12s %6.1f s  v %5.2f m/s  |dbeta| %.3f  |e_pos| %.2f m"
              % (r["scenario"], r["seconds"], r["speed"], r["beta_err"], r["epos"]))
        print("             steer: " + format_metrics(r["metrics"]))
