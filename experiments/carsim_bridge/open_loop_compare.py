"""Open-loop comparison of the GPU plant and CarSim from the same state.

Same initial condition (4.5 m/s, straight, wheels rolling), same pedal inputs, and
the response of the channels the actor actually sees is compared.
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))
sys.path.insert(0, os.path.join(ROOT, "utils"))

from envs.carsim_flat_env import CarSimFlatEnv  # noqa: E402
from fast_plant import FastRacerPlant, load_preset  # noqa: E402


def gpu_rollout(steer, throttle, brake, n=40):
    params = load_preset("fullsize_racecar_ev")
    plant = FastRacerPlant(params, 1, "cpu")
    s = torch.zeros((1, 10))
    v0 = 4.5
    s[0, 3] = v0
    for i in range(4):
        s[0, 6 + i] = v0 / 0.332
    cmd = torch.tensor([[steer, throttle, brake]])
    rows = []
    for k in range(n):
        s, ex = plant.step(s, cmd, dt=0.05, substeps=5, mode="pedals")
        es = plant.package_extended(s, ex)
        rows.append([float(es[0, 6]), float(es[0, 3]), float(es[0, 4]),
                     float(s[0, 5]), float(ex["ax"]), float(ex["ay"])])
    return np.array(rows)


def carsim_rollout(steer, throttle, brake, n=40, scenario="flat_mu100"):
    env = CarSimFlatEnv(scenario)
    env.reset()
    # get rolling at 4.5 m/s in a straight line first
    imp = [0.0, 0.35, 0.0]
    for _ in range(400):
        env.step([0.0, 0.35 / 1.0, 0.0])
        if float(np.hypot(env.s[0, 3], env.s[0, 4])) >= 4.5:
            break
    rows = []
    for k in range(n):
        env.step([steer, throttle, brake])
        raw = env.raw()
        rows.append([float(np.hypot(env.s[0, 3], env.s[0, 4])), float(env.s[0, 3]),
                     float(env.s[0, 4]), float(env.s[0, 5]),
                     raw["Ax"] * 9.81, raw["Ay"] * 9.81])
    env.close()
    return np.array(rows)


if __name__ == "__main__":
    cases = [("throttle 0.8", 0.0, 0.8, 0.0),
             ("steer +0.2 rad", 0.2, 0.3, 0.0),
             ("steer -0.2 rad", -0.2, 0.3, 0.0)]
    print("%-16s %-22s %-22s" % ("case", "GPU dv/ay/yawr", "CarSim dv/ay/yawr"))
    for name, st, th, br in cases:
        g = gpu_rollout(st, th, br)
        c = carsim_rollout(st, th, br)
        print("%-16s dV %5.2f ay %5.2f r %6.3f   dV %5.2f ay %5.2f r %6.3f"
              % (name,
                 g[-1, 0] - 4.5, g[-1, 5], g[-1, 3],
                 c[-1, 0] - 4.5, c[-1, 5], c[-1, 3]))
