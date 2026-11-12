"""Contract regression tests: the defects that cost this project the most time.

Each test pins one invariant that was silently violated at some point.  They are
cheap (no training, no CarSim) and must pass before any training run.

    python experiments/continuous_drift_iwd/contract_tests.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))
sys.path.insert(0, os.path.join(ROOT, "utils"))

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s   %s" % (name, detail))
        FAILURES.append(name)


def test_preset():
    """The preset must carry the values measured from the CarSim vehicle."""
    from fast_plant import load_preset
    p = load_preset("fullsize_racecar_ev")
    check("mass 1514.8 kg", abs(p["m"] - 1514.8) < 0.5, "got %s" % p["m"])
    check("Iz 1605.3", abs(p["Iz"] - 1605.3) < 1.0, "got %s" % p["Iz"])
    check("wheel inertia 0.17", abs(p["IwF"] - 0.17) < 1e-6, "got %s" % p["IwF"])
    check("front track 1.720", abs(p["Tf"] - 1.720) < 1e-3, "got %s" % p["Tf"])
    check("radius 0.332", abs(p["rR"] - 0.332) < 1e-4, "got %s" % p["rR"])
    check("nominal B 6.0", abs(p["B"] - 6.0) < 0.05, "got %s" % p["B"])


def test_randomisation_covers_nominal():
    """The DR range must be centred on the calibrated tyre, not an old one."""
    import re
    src = open(os.path.join(ROOT, "run.py"), encoding="utf-8").read()
    block = src.split('"ev_drift": {', 1)[1].split("}", 1)[0]
    b = [float(x) for x in re.search(r'"B":\s*\[([-\d.,\s]+)\]', block).group(1).split(",")]
    check("DR B brackets 6.0", b[0] <= 6.0 <= b[1], "got %s" % (b,))


def test_steering_conversion():
    """Command +/-1 must be the 0.45 rad road-wheel angle the GPU plant uses."""
    from envs.carsim_flat_env import (MAX_ROAD_WHEEL_RAD, MAX_STEER_SW_DEG,
                                      road_wheel_rad_to_steering_wheel_deg)
    check("full command is 345 deg of steering wheel",
          abs(road_wheel_rad_to_steering_wheel_deg(MAX_ROAD_WHEEL_RAD)
              - MAX_STEER_SW_DEG) < 1e-6)
    check("0.20 rad is about 153 deg",
          abs(road_wheel_rad_to_steering_wheel_deg(0.20) - 153.3) < 0.5,
          "got %.2f" % road_wheel_rad_to_steering_wheel_deg(0.20))


def test_wheel_order():
    """CarSim L1,L2,R1,R2 is FL,RL,FR,RR; the contract is FL,FR,RL,RR."""
    from envs.carsim_flat_env import CARSIM_WHEEL_SUFFIXES
    check("wheel order FL,FR,RL,RR",
          CARSIM_WHEEL_SUFFIXES == ("L1", "R1", "L2", "R2"),
          "got %s" % (CARSIM_WHEEL_SUFFIXES,))


def test_actuator():
    """One step at the slew limit moves exactly rate*dt, and the pedals are exclusive."""
    from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
    env = ContinuousDriftIWDEnv("fullsize_racecar_ev", 2, "cuda:0", ref_mode="hybrid",
                                control_mode="pedals", train=True,
                                steer_actuator_tau=0.06, steer_rate_limit=1.8,
                                steer_deadband=0.0075, brake_margin=0.0)
    env.reset()
    a = torch.zeros((2, 3), device="cuda:0")
    a[:, 0] = 0.45
    a[:, 1] = 0.6
    a[:, 2] = 0.6
    env.step(a)
    check("first step is rate*dt", abs(float(env.u_cmd[0, 0]) - 1.8 * env.dt) < 0.003,
          "got %.4f" % float(env.u_cmd[0, 0]))
    check("pedals exclusive (both 0.6 -> 0/0)",
          abs(float(env.u_cmd[0, 1])) < 1e-6 and abs(float(env.u_cmd[0, 2])) < 1e-6)
    for _ in range(20):
        env.step(a)
    check("command never passes the target",
          float(env.u_cmd[0, 0]) <= 0.45 + 1e-5, "got %.4f" % float(env.u_cmd[0, 0]))


def test_observation_contract():
    """27 actor channels, 52 critic channels, and no privileged leak into the actor."""
    from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
    env = ContinuousDriftIWDEnv("fullsize_racecar_ev", 2, "cuda:0", ref_mode="hybrid",
                                control_mode="pedals", train=True, asymmetric=True)
    obs = env.reset()
    check("actor is 27", tuple(obs["obs"].shape) == (2, 27), str(tuple(obs["obs"].shape)))
    check("critic is 52", tuple(obs["states"].shape) == (2, 52),
          str(tuple(obs["states"].shape)))


if __name__ == "__main__":
    print("contract tests")
    for fn in (test_preset, test_randomisation_covers_nominal, test_steering_conversion,
               test_wheel_order, test_actuator, test_observation_contract):
        print(fn.__name__)
        fn()
    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        sys.exit(1)
    print("ALL CONTRACT TESTS PASSED")
