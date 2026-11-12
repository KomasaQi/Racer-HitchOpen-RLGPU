"""Run the trained drift policy inside CarSim (no OU disturbance).

One episode per scenario: the car starts from rest at the origin, follows one of our
own reference roads, and the shared task logic scores it.
"""
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

CKPT = os.path.join(ROOT, "experiments", "continuous_drift_iwd", "runs",
                    "drift_ev_v17", "nn")
CKPT_FILE = None
for f in os.listdir(CKPT):
    if f.startswith("last_") and f.endswith(".pth"):
        CKPT_FILE = os.path.join(CKPT, f)


def run(scenario, policy_path, steps=600, seed=0):
    torch.manual_seed(seed)
    env = CarSimDriftEnv(scenario)
    obs = env.reset()
    env.asymmetric = True
    pol = Policy(policy_path, torch.device("cuda:0"))
    pol.reset_hidden(torch.ones(1, dtype=torch.bool, device="cuda:0"))
    speed, beta_err, epos, n = 0.0, 0.0, 0.0, 0
    done_step = steps
    for k in range(steps):
        a = pol.act(obs["obs"] if isinstance(obs, dict) else obs)
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
            "seconds": done_step * env.dt,
            "speed": speed / max(n, 1),
            "beta_err": beta_err / max(n, 1),
            "epos": epos / max(n, 1)}


if __name__ == "__main__":
    print("checkpoint:", os.path.basename(CKPT_FILE))
    rows = []
    for sc in ("flat_mu050", "flat_mu075", "flat_mu100"):
        r = run(sc, CKPT_FILE)
        rows.append(r)
        print("%-12s %6.1f s  v %5.2f m/s  |dbeta| %.3f  |e_pos| %.2f m"
              % (r["scenario"], r["seconds"], r["speed"], r["beta_err"], r["epos"]))
