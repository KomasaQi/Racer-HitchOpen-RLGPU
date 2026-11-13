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
    check("CarSim total laden mass 1461.4 kg",
          abs(p["m"] - 1461.4) < 1e-6, "got %s" % p["m"])
    check("CarSim total laden Iz 1693.469",
          abs(p["Iz"] - 1693.469182) < 1e-6, "got %s" % p["Iz"])
    check("CarSim wheel+tire spin inertia 1.77/2.27",
          abs(p["IwF"] - 1.77) < 1e-6 and abs(p["IwR"] - 2.27) < 1e-6,
          "got %s / %s" % (p["IwF"], p["IwR"]))
    check("CarSim total CG geometry",
          abs(p["lF"] - 1.46257356) < 1e-8 and
          abs(p["lR"] - 1.18742644) < 1e-8 and
          abs(p["h"] - 0.373483988) < 1e-9)
    check("front track 1.720", abs(p["Tf"] - 1.720) < 1e-3, "got %s" % p["Tf"])
    check("CarSim effective radii front 0.332 / rear 0.347",
          abs(p["rF"] - 0.332) < 1e-4 and abs(p["rR"] - 0.347) < 1e-4,
          "got %s / %s" % (p["rF"], p["rR"]))
    check("CarSim longitudinal B 29.773", abs(p["bx"] - 29.7734) < 0.01,
          "got %s" % p["bx"])
    check("CarSim lateral B 17.298", abs(p["by"] - 17.2982) < 0.01,
          "got %s" % p["by"])
    check("CarSim tire peak mu 2.1246", abs(p["D"] - 2.1246) < 0.001,
          "got %s" % p["D"])
    check("compact analytic transition is production",
          p["combined_slip_model"] == "compact_transition" and
          abs(p["combined_slip_blend"] - 0.49) < 1e-6,
          "got %s / %s" % (p["combined_slip_model"], p["combined_slip_blend"]))
    check("CarSim road-grip stiffness coupling 0.20/0.40",
          abs(p["tire_stiffness_grip_exponent_x"] - 0.20) < 1e-8 and
          abs(p["tire_stiffness_grip_exponent_y"] - 0.40) < 1e-8)
    check("motor inertia 0.0226", abs(p["motor_inertia"] - 0.0226) < 1e-8,
          "got %s" % p["motor_inertia"])
    check("load references are part of tensor contract",
          "fz_ref_front" in p and "fz_ref_rear" in p)
    check("CarSim LSD uses compact square-root law",
          p["lsd_model"] == "power_law" and
          abs(p["lsd_torque_ref"] - 90.09) < 1e-6 and
          abs(p["lsd_exponent"] - 0.5) < 1e-6,
          "got %s / %s / %s" % (p["lsd_model"], p["lsd_torque_ref"],
                                  p["lsd_exponent"]))


def test_driveline_inertia_and_slip_sign():
    """CarSim inertias are shared correctly and positive kappa means overspeed."""
    from fast_plant import FastRacerPlant, load_preset
    plant = FastRacerPlant(load_preset("fullsize_racecar_ev"), 1, "cpu")
    inertia = plant.geometry()[3][0]
    expected = 2.27 + 0.5 * 0.0226 * 3.44 ** 2 + 0.5 * 0.013 + 0.009
    check("rear effective inertia from CarSim pars",
          abs(float(inertia[2]) - expected) < 1e-5,
          "got %.6f expected %.6f" % (float(inertia[2]), expected))
    s = torch.zeros((1, 10))
    s[0, 3] = 5.0
    s[0, 6:8] = 5.0 / 0.332
    s[0, 8:10] = 8.0 / 0.332
    _, extras = plant.derivatives(s, torch.zeros((1, 3)))
    check("positive kappa_power is wheel overspeed",
          float(extras["kappa_power"][0, 2:].mean()) > 0.5,
          "got %.4f" % float(extras["kappa_power"][0, 2:].mean()))
    s2 = torch.zeros((1, 10))
    s2[0, 3] = 10.0
    s2[0, 4] = 1.0
    s2[0, 6:10] = 10.0 / 0.332
    _, ex2 = plant.derivatives(s2, torch.zeros((1, 3)))
    check("alpha is tire slip angle, not combined-slip direction",
          abs(float(ex2["alpha"][0, 2]) - torch.atan(torch.tensor(0.1)).item()) < 0.01,
          "got %.4f rad" % float(ex2["alpha"][0, 2]))
    eff = plant._motor_efficiency(torch.tensor([2000.0, 2000.0]),
                                  torch.tensor([65.0, 305.0]))
    check("CarSim motor efficiency carpet lookup",
          abs(float(eff[0]) - 0.9025) < 1e-5 and abs(float(eff[1]) - 0.684) < 1e-5,
          "got %s" % [float(x) for x in eff])


def test_carsim_motor_power_semantics():
    """CarSim demand power is mechanical shaft power, not battery power."""
    from fast_plant import FastRacerPlant, load_preset
    plant = FastRacerPlant(load_preset("fullsize_racecar_ev"), 1, "cpu")
    # Measured internal channels at q=.8 and ~3723 rpm give M_MotCmd ~=788 Nm:
    # exactly 307.2 kW / omega, independent of the ~.375 efficiency reported
    # at that operating point.
    motor_rpm = torch.tensor([3723.0])
    wm = motor_rpm * (2.0 * torch.pi / 60.0)
    rear_omega = torch.stack([wm / 3.44, wm / 3.44], dim=1)
    _, torque, _ = plant._motor_torque(
        rear_omega, torch.tensor([0.8]), torch.tensor([0.0]))
    expected = 0.8 * 384000.0 / wm
    check("CarSim demand maps to mechanical shaft power",
          bool(torch.allclose(torque, expected, rtol=2e-3, atol=1.0)),
          "got %.3f expected %.3f" % (float(torque), float(expected)))


def test_lsd_generalized_viscous_contract():
    """The analytic LSD must reproduce CarSim's curve and conserve axle torque."""
    from fast_plant import FastRacerPlant, load_preset
    plant = FastRacerPlant(load_preset("fullsize_racecar_ev"), 5, "cpu")
    rpm = torch.tensor([0.0, 60.0, 120.0, 600.0, -1200.0])
    omega = rpm * (2.0 * torch.pi / 60.0)
    pair = torch.stack([omega, torch.zeros_like(omega)], dim=1)
    axle = torch.full((5,), 1000.0)
    left, right, coupling = plant._lsd(axle, pair)
    delta = left - right
    expected = torch.tensor([0.0, -90.09, -127.40, -284.88, 402.88])
    check("analytic LSD matches CarSim printed points",
          bool(torch.allclose(delta, expected, atol=0.08, rtol=0.0)),
          "got %s" % [round(float(x), 3) for x in delta])
    check("LSD conserves total axle torque",
          bool(torch.allclose(left + right, axle, atol=1e-5, rtol=0.0)))
    check("LSD transfers torque away from faster wheel",
          bool((coupling[1:4] > 0).all()),
          "coupling %s" % [float(x) for x in coupling])


def test_calibrated_parameters_affect_dynamics():
    """Calibrated fields must change equations, not merely exist in YAML/p."""
    from fast_plant import FastRacerPlant, PARAM_INDEX, load_preset
    p = load_preset("fullsize_racecar_ev")
    pair = torch.tensor([[100.0, 0.0], [100.0, 0.0]])
    lsd = FastRacerPlant(p, 2, "cpu")
    lsd.p[:, PARAM_INDEX["lsd_torque_ref"]] = torch.tensor([45.0, 90.0])
    left, right, _ = lsd._lsd(torch.full((2,), 1000.0), pair)
    check("LSD strength changes actual torque split",
          abs(float((left-right)[1])) > 1.9 * abs(float((left-right)[0])))
    wheel = FastRacerPlant(p, 2, "cpu")
    wheel.p[:, PARAM_INDEX["IwR"]] = torch.tensor([1.0, 4.0])
    s = wheel.zero_state(2); wheel.init_wheels(s)
    s, _ = wheel.step(s, torch.tensor([[0.0, 0.2, 0.0]]).repeat(2, 1),
                      dt=0.05, substeps=5)
    check("wheel+tire inertia changes wheel-speed transient",
          float(s[0, 8:10].mean()) > float(s[1, 8:10].mean()))
    yaw = FastRacerPlant(p, 2, "cpu")
    yaw.p[:, PARAM_INDEX["Iz"]] = torch.tensor([900.0, 1800.0])
    sy = yaw.zero_state(2); sy[:, 3] = 12.0; yaw.init_wheels(sy)
    ds, _ = yaw.derivatives(sy, torch.tensor([[0.2, 0.0, 0.0]]).repeat(2, 1))
    check("total Iz changes yaw acceleration",
          abs(float(ds[0, 5])) > 1.8 * abs(float(ds[1, 5])))


def test_physical_sign_contract():
    """Canonical left/CCW/overspeed signs must hold in the plant equations."""
    from fast_plant import FastRacerPlant, load_preset
    plant = FastRacerPlant(load_preset("fullsize_racecar_ev"), 1, "cpu")
    s = plant.zero_state(1); s[:, 3] = 12.0; plant.init_wheels(s)
    ds, ex = plant.derivatives(s, torch.tensor([[0.04, 0.0, 0.0]]))
    check("positive steer creates left/CCW response",
          float(ds[0, 4]) > 0 and float(ds[0, 5]) > 0 and
          float(ex["ay"][0]) > 0)
    check("left-turn tire alpha/Fy convention",
          float(ex["alpha"][0, 0]) < 0 and float(ex["fy_wheel"][0, 0]) > 0)
    launch = plant.zero_state(1); plant.init_wheels(launch)
    for _ in range(10):
        launch, lex = plant.step(launch, torch.tensor([[0.0, 0.5, 0.0]]),
                                  dt=0.05, substeps=5)
    check("positive throttle gives positive omega/kappa/Fx",
          float(launch[0, 8]) > 0 and float(lex["kappa_power"][0, 2]) > 0 and
          float(lex["fx_wheel"][0, 2]) > 0)
    rolling = plant.zero_state(1); rolling[:, 3] = 12.0; plant.init_wheels(rolling)
    rolling, bex = plant.step(rolling, torch.tensor([[0.0, 0.0, 0.3]]),
                               dt=0.05, substeps=5)
    check("brake gives negative Ax/kappa/Fx",
          float(bex["ax"][0]) < 0 and float(bex["kappa_power"][0, 0]) < 0 and
          float(bex["fx_wheel"][0, 0]) < 0)


def test_xcar_isotropic_tire_contract():
    """The XCAR candidate must use one radial MF curve, not an Fx/Fy ellipse."""
    from fast_plant import FastRacerPlant, load_preset
    p = load_preset("fullsize_racecar_ev")
    p["combined_slip_model"] = "isotropic"
    p["load_sensitivity"] = 0.0
    plant = FastRacerPlant(p, 2, "cpu")
    s = torch.zeros((2, 10))
    s[:, 3] = 10.0
    radius = p["rR"]
    s[0, 6:10] = 12.0 / radius       # sx = -0.2, sy = 0
    s[1, 4] = 2.0
    s[1, 6:10] = 10.0 / radius       # sx = 0, sy = 0.2
    fz = torch.full((2, 4), 3500.0)
    force = plant._wheel_forces(s, torch.zeros((2, 2)), fz, plant.geometry())
    fx = force["fx_wheel"][0, 2].abs()
    fy = force["fy_wheel"][1, 2].abs()
    check("XCAR isotropic tire is radially symmetric",
          abs(float(fx - fy)) < 1e-2,
          "|Fx| %.4f |Fy| %.4f" % (float(fx), float(fy)))
    check("XCAR isotropic force follows combined-slip direction",
          abs(float(force["fy_wheel"][0, 2])) < 1e-2 and
          abs(float(force["fx_wheel"][1, 2])) < 1e-2)


def test_power_limited_wheel_solver_is_stable():
    """The implicit motor/wheel solve must not alternate overspeed and rolling."""
    from fast_plant import FastRacerPlant, load_preset
    plant = FastRacerPlant(load_preset("fullsize_racecar_ev"), 1, "cpu")
    s = plant.zero_state(1)
    plant.init_wheels(s)
    cmd = torch.tensor([[0.0, 0.10, 0.0]])
    rear_surface = []
    for _ in range(30):
        s, extras = plant.step(s, cmd, dt=0.05, substeps=5)
        rear_surface.append(float(s[0, 8:10].mean() * extras["wheel_radius"][0, 2:].mean()))
    jumps = [abs(b - a) for a, b in zip(rear_surface, rear_surface[1:])]
    check("power-limited rear wheel solve has no two-cycle",
          max(jumps) < 5.0,
          "max adjacent jump %.3f m/s; first samples %s"
          % (max(jumps), [round(x, 3) for x in rear_surface[:6]]))
    check("wheel solve remains finite",
          all(torch.isfinite(torch.tensor(rear_surface))), str(rear_surface[:6]))


def test_tire_relaxation_candidate():
    """The optional CarSim lag must delay force slip and reset its memory."""
    from fast_plant import FastRacerPlant, load_preset
    params = load_preset("fullsize_racecar_ev")
    params["tire_relaxation"] = True
    plant = FastRacerPlant(params, 1, "cpu")
    s = plant.zero_state(1)
    s[0, 3] = 0.6
    plant.init_wheels(s)
    s, _ = plant.step(s, torch.zeros((1, 3)), dt=0.01, substeps=1)
    s, ex = plant.step(s, torch.tensor([[0.01, 0.0, 0.0]]), dt=0.01, substeps=1)
    raw = float(ex["alpha"][0, :2].abs().mean())
    lagged = float(ex["alpha_force"][0, :2].abs().mean())
    check("low-speed lateral force slip lags a steer step", 0.0 < lagged < raw,
          "raw %.6f lagged %.6f" % (raw, lagged))
    plant.reset_guess()
    check("tire relaxation memory clears on reset",
          not bool(plant._relax_initialized.any()))


def test_randomisation_covers_nominal():
    """The DR range must be centred on the calibrated tyre, not an old one."""
    import re
    src = open(os.path.join(ROOT, "run.py"), encoding="utf-8").read()
    block = src.split('"ev_drift": {', 1)[1].split("}", 1)[0]
    bx = [float(x) for x in re.search(r'"bx":\s*\[([-\d.,\s]+)\]', block).group(1).split(",")]
    by = [float(x) for x in re.search(r'"by":\s*\[([-\d.,\s]+)\]', block).group(1).split(",")]
    check("DR bx brackets 29.773", bx[0] <= 29.7734 <= bx[1], "got %s" % (bx,))
    check("DR by brackets 17.298", by[0] <= 17.2982 <= by[1], "got %s" % (by,))


def test_pretrain_gate_matches_training_dr():
    """The cheap rollout gate must exercise the same plant uncertainty as PPO."""
    import re
    from experiments.continuous_drift_iwd.pretrain_rollout_gate import EV_DR
    src = open(os.path.join(ROOT, "run.py"), encoding="utf-8").read()
    block = src.split('"ev_drift": {', 1)[1].split("},", 1)[0]
    critical = ("m", "Iz", "IwF", "IwR", "rF", "rR",
                "tire_stiffness_grip_exponent_x",
                "tire_stiffness_grip_exponent_y", "lsd_torque_ref",
                "lsd_exponent", "lsd_tau_max", "D")
    for key in critical:
        match = re.search(r'"%s":\s*\[([-\d.,\s]+)\]' % re.escape(key), block)
        limits = [float(x) for x in match.group(1).split(",")]
        check("pretrain gate uses training DR %s" % key,
              EV_DR.get(key) == limits,
              "got %s" % EV_DR.get(key))


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
    # The hybrid training reset may sample a large steering trim.  That can put
    # a 0.45-rad target inside the slew-limit distance and makes this nominal
    # actuator test depend on the random reset draw.  Test the actuator itself
    # from the canonical zero command/trim state.
    env.u_trim.zero_()
    env.u_cmd.zero_()
    env._steer_state.zero_()
    env._steer_prev.zero_()
    env._action_delay_buffer.zero_()
    a = torch.zeros((2, 3), device="cuda:0")
    a[:, 0] = 0.45
    a[:, 1] = 0.6
    a[:, 2] = 0.6
    trim = env.u_trim[0].clone()
    env.step(a)
    # the actor emits a CORRECTION around the trim, so the actuator moves rate*dt away
    # from the trim, not from zero
    moved = float(env.u_cmd[0, 0]) - float(trim[0])
    check("first step moves rate*dt from the trim", abs(moved - 1.8 * env.dt) < 0.003,
          "moved %.4f from trim %.3f" % (moved, float(trim[0])))
    # both pedals maxed: the correction cancels and the trim is throttle 0.02, so the
    # net longitudinal command is the trim throttle alone
    check("pedals exclusive (both 0.6 -> trim throttle only)",
          abs(float(env.u_cmd[0, 1]) - float(trim[1])) < 1e-5
          and abs(float(env.u_cmd[0, 2])) < 1e-6,
          "thr %.4f brake %.4f trim thr %.4f"
          % (float(env.u_cmd[0, 1]), float(env.u_cmd[0, 2]), float(trim[1])))
    for _ in range(20):
        env.step(a)
    check("command never passes the target",
          float(env.u_cmd[0, 0]) <= 0.45 + 1e-5, "got %.4f" % float(env.u_cmd[0, 0]))

    # rl_games has already converted normalized network output to this physical
    # action.  With final-curriculum trim disabled, the environment must preserve
    # road-wheel angle exactly instead of applying a hidden second gain.
    env._steer_tau_env.zero_()
    env.u_trim.zero_()
    physical = torch.tensor([[0.20, 0.60, 0.10], [-0.17, 0.10, 0.60]], device="cuda:0")
    applied = env.preprocess_action(physical)
    check("physical steering action is not rescaled twice",
          bool(torch.allclose(applied[:, 0], physical[:, 0], atol=1e-7)),
          str(applied[:, 0].tolist()))


def test_observation_contract():
    """27 actor channels, 52 critic channels, and no privileged leak into the actor."""
    from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
    env = ContinuousDriftIWDEnv("fullsize_racecar_ev", 2, "cuda:0", ref_mode="hybrid",
                                control_mode="pedals", train=True, asymmetric=True)
    obs = env.reset()
    check("actor is 27", tuple(obs["obs"].shape) == (2, 27), str(tuple(obs["obs"].shape)))
    check("critic is 52", tuple(obs["states"].shape) == (2, 52),
          str(tuple(obs["states"].shape)))


def test_compiled_resume_key_contract():
    """Compiled asymmetric critic checkpoints must be restorable by rl_games."""
    from collections import OrderedDict
    from utils.rlgame_utils import normalise_compiled_central_state
    value = torch.tensor([1.0])
    state = OrderedDict([
        ("model._orig_mod.a2c_network.mu.weight", value),
        ("unrelated", value),
    ])
    fixed = normalise_compiled_central_state(state)
    check("compiled central-value prefix is removed",
          "model.a2c_network.mu.weight" in fixed)
    check("unrelated checkpoint keys are preserved", "unrelated" in fixed)


def test_sensor_actuator_randomisation():
    """DR must vary hardware timing without changing or leaking the actor contract."""
    from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", 256, "cuda:0", ref_mode="closed_random",
        control_mode="pedals", train=True, asymmetric=True,
        sensor_dr=True, sensor_delay_max=2, sensor_dropout_prob=0.01,
        actuator_dr=True, action_delay_max=2)
    obs = env.reset()
    cached = env.obs()
    check("sensor DR preserves 27 actor channels", tuple(obs["obs"].shape) == (256, 27),
          str(tuple(obs["obs"].shape)))
    check("repeated obs call returns same sensor packet",
          bool(torch.equal(obs["obs"], cached["obs"])),
          "obs() advanced noise/delay without physics")
    check("steering time constant is randomised",
          float(env._steer_tau_env.max() - env._steer_tau_env.min()) > 0.05)
    check("pedal time constant is randomised",
          float(env._pedal_tau_env.max() - env._pedal_tau_env.min()) > 0.05)
    check("command delay covers multiple steps",
          int(env._action_delay_steps.max()) > int(env._action_delay_steps.min()),
          str(torch.unique(env._action_delay_steps).tolist()))
    after, _, _, _ = env.step(torch.zeros((256, 3), device="cuda:0"))
    check("sensor DR stays finite", bool(torch.isfinite(after["obs"]).all()))


def test_drift_stack_is_opt_in():
    """A throttle-only ablation must not silently enable the drift stack."""
    from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
    env = ContinuousDriftIWDEnv("fullsize_racecar_ev", 2, "cuda:0", ref_mode="hybrid",
                                control_mode="pedals", train=True, drift_mode=False,
                                throttle_weight=3.0)
    env.reset()
    env.step(torch.zeros((2, 3), device="cuda:0"))
    for key in ("drift_kappa", "drift_throttle", "drift_counter", "drift_joint"):
        check("%s disabled outside drift_mode" % key,
              abs(env.rew_stats[key]) < 1e-8, "got %.6f" % env.rew_stats[key])


def test_rest_spawn_has_no_countersteer():
    """A zero-speed curriculum spawn cannot begin with an established-drift lock."""
    from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", 64, "cuda:0", ref_mode="hybrid",
        control_mode="pedals", train=True, drift_mode=True,
        spawn_mode="equilibrium", spawn_speed=0.0, spawn_speed_min=0.0,
        spawn_deficit_frac=1.0)
    env.reset()
    check("rest drift spawn has zero steering trim",
          float(env.u_trim[:, 0].abs().max()) < 1e-6,
          "max |trim steer| %.6f" % float(env.u_trim[:, 0].abs().max()))
    check("rest drift spawn has no artificial wheel spin",
          float(env.s[:, 8:10].abs().max()) < 1e-6,
          "max rear omega %.6f" % float(env.s[:, 8:10].abs().max()))
    physical = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", 64, "cuda:0", ref_mode="closed_random",
        control_mode="pedals", train=True, drift_mode=True,
        spawn_mode="equilibrium", spawn_speed=6.0,
        spawn_beta_frac=0.0, drift_spawn_spin_frac=0.0,
        drift_spawn_trim_frac=0.0)
    physical.reset()
    rear_kappa = physical.extras["kappa_power"][:, 2:4]
    check("physical curriculum can disable rear pre-spin",
          float(rear_kappa.abs().max()) < 0.03,
          "max |rear kappa| %.4f" % float(rear_kappa.abs().max()))
    check("physical curriculum installs no privileged drift trim",
          float(physical.u_trim.abs().max()) < 1e-6,
          "max |trim| %.4f" % float(physical.u_trim.abs().max()))
    fresh = physical._eval_extras(physical.u_cmd)
    check("respawn refreshes extras instead of reusing terminal frame",
          bool(torch.allclose(physical.extras["kappa_power"], fresh["kappa_power"], atol=1e-6)),
          "stored and recomputed kappa differ")



def test_vehicle_randomisation():
    """Every calibrated vehicle parameter must actually vary at reset."""
    from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
    rng = {"bx": [26.8, 32.8], "by": [15.6, 19.0], "D": [1.0623, 2.1246],
           "m": [1417.0, 1505.0], "Iz": [1558.0, 1829.0],
           "IwF": [1.55, 1.99], "IwR": [2.00, 2.54],
           "Tf": [1.70, 1.74], "Tr": [1.58, 1.62],
           "rF": [0.329, 0.335], "rR": [0.3435, 0.3505],
           "load_sensitivity": [0.12, 0.19], "drive_efficiency": [0.975, 0.995],
           "motor_efficiency": [0.90, 1.10],
           "combined_slip_blend": [0.35, 0.65],
           "tire_stiffness_grip_exponent_x": [0.10, 0.30],
           "tire_stiffness_grip_exponent_y": [0.30, 0.50],
           "lsd_torque_ref": [72.0, 108.0], "lsd_exponent": [0.40, 0.60],
           "lsd_tau_max": [240.0, 360.0],
           "max_power_w": [345600.0, 422400.0]}
    env = ContinuousDriftIWDEnv("fullsize_racecar_ev", 2048, "cuda:0",
                                ref_mode="hybrid", control_mode="pedals",
                                train=True, randomize_param=rng)
    env.randomize()
    for key in ("m", "Iz", "IwF", "rR", "bx", "by", "motor_efficiency",
                "max_power_w", "load_sensitivity", "combined_slip_blend",
                "tire_stiffness_grip_exponent_x", "tire_stiffness_grip_exponent_y",
                "lsd_torque_ref", "lsd_exponent", "lsd_tau_max"):
        v = env.plant.col(key)
        spread = float(v.max() - v.min())
        check("DR varies %s" % key, spread > 1e-6, "spread %.3e" % spread)


def test_continuing_task_contract():
    """A rollout/time boundary must not reset a continuing control task."""
    from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", 2, "cuda:0", ref_mode="closed_random",
        control_mode="pedals", train=True, continuing_task=True, num_paths=4)
    env.reset()
    env.max_steps = 1
    _, _, done, info = env.step(torch.zeros((2, 3), device="cuda:0"))
    check("continuing horizon does not emit done", not bool(done.any()), str(done.tolist()))
    check("continuing horizon is not a timeout", not bool(info["time_outs"].any()),
          str(info["time_outs"].tolist()))
    env.progress.fill_(env.num_waypoints - 1)
    env.step_progress.fill_(3)
    env.progress.add_(env.step_progress.long()).remainder_(env.num_waypoints)
    check("closed reference progress wraps", bool((env.progress == 2).all()),
          str(env.progress.tolist()))
    route = env.path_bank[0, 0]
    seam_gap = float(torch.norm(route[0, :2] - route[-1, :2]))
    seam_heading = float(torch.atan2(torch.sin(route[0, 2] - route[-1, 2]),
                                     torch.cos(route[0, 2] - route[-1, 2])).abs())
    check("random closed path has a continuous seam",
          seam_gap < 0.03 and seam_heading < 0.02,
          "position %.5f m heading %.5f rad" % (seam_gap, seam_heading))
    env.travel.fill_(1e6)
    obs = env.obs()
    actor_obs = obs["obs"] if isinstance(obs, dict) else obs
    check("continuing actor has no unbounded odometer",
          float(actor_obs[:, 14].abs().max()) < 5.0,
          "channel 14 max %.3f" % float(actor_obs[:, 14].abs().max()))
    try:
        ContinuousDriftIWDEnv(
            "fullsize_racecar_ev", 1, "cuda:0", ref_mode="hybrid",
            control_mode="pedals", train=True, continuing_task=True)
    except ValueError as exc:
        rejected = "seamless path streaming" in str(exc)
    else:
        rejected = False
    check("open path cannot masquerade as continuing", rejected)
    check("continuing evaluation pass means full window, not success terminal",
          env.continuing_task and not bool(env.is_done.any()))


def test_preview_steering_sign():
    """A left-curving reference must request positive (left) road-wheel steer."""
    from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", 32, "cuda:0", ref_mode="circle",
        control_mode="pedals", train=True, spawn_beta_frac=0.0,
        drift_spawn_spin_frac=0.0, drift_spawn_trim_frac=0.0,
        preview_time="0.35,0.7,1.2")
    env.reset()
    env.obs()
    same_sign = (env.delta_pp * env.kappa_ref) > 0.0
    # Spawn position/heading noise intentionally puts a minority of cars across
    # the line, where recovery steering can oppose nominal curvature.
    check("pure-pursuit steer has reference curvature sign",
          float(same_sign.float().mean()) > 0.80 and
          float((env.delta_pp * env.kappa_ref).mean()) > 0.0,
          "same-sign %.1f%% delta %.4f kappa %.4f" %
          (100.0 * float(same_sign.float().mean()), float(env.delta_pp.mean()),
           float(env.kappa_ref.mean())))


def test_closed_random_refresh_contract():
    """Fresh closed roads install atomically without changing an active road."""
    from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", 4, "cuda:0", ref_mode="closed_random",
        control_mode="pedals", train=True, continuing_task=True, num_paths=4)
    env.reset()
    active_generation = int(env.current_generation)
    active_roads = env.path_bank[active_generation].clone()
    active_env_road = env.waypoints_at(
        torch.zeros((env.n, 1), dtype=torch.long, device=env.device)).clone()
    installed = env.request_path_refresh()
    check("closed_random accepts path refresh", installed)
    check("closed_random advances generation", int(env.current_generation) != active_generation)
    check("refresh preserves old generation for active environments",
          bool(torch.equal(env.path_bank[active_generation], active_roads)))
    check("refresh does not change road under active environments",
          bool(torch.equal(env.waypoints_at(
              torch.zeros((env.n, 1), dtype=torch.long, device=env.device)),
              active_env_road)))
    new_generation = int(env.current_generation)
    check("refreshed closed bank is genuinely new",
          not bool(torch.equal(env.path_bank[new_generation], active_roads)))
    route = env.path_bank[new_generation, 0]
    seam_gap = float(torch.norm(route[0, :2] - route[-1, :2]))
    seam_heading = float(torch.atan2(torch.sin(route[0, 2] - route[-1, 2]),
                                     torch.cos(route[0, 2] - route[-1, 2])).abs())
    check("refreshed path keeps a continuous closed seam",
          seam_gap < 0.03 and seam_heading < 0.02,
          "position %.5f m heading %.5f rad" % (seam_gap, seam_heading))


def test_bounded_reward_contract():
    """No live step may be negative and a failure must be strictly negative."""
    from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", 64, "cuda:0", ref_mode="circle",
        control_mode="pedals", train=True, drift_mode=True,
        reward_mode="bounded_joint", terminal_cost=100.0)
    env.reset()
    _, live_rew, done, _ = env.step(torch.zeros((64, 3), device="cuda:0"))
    check("bounded live reward is nonnegative",
          float(live_rew[done == 0].min()) >= 0.0,
          "minimum %.6f" % float(live_rew.min()))
    env._now["failed"] = torch.ones(64, device="cuda:0")
    fail_rew = env.reward()
    check("bounded failure reward is strictly negative",
          float(fail_rew.max()) < 0.0, "maximum %.6f" % float(fail_rew.max()))
    max_live = (env.bounded_alive + env.bounded_tracking + env.bounded_drift +
                env.bounded_joint + env.bounded_smooth)
    check("terminal cost dominates maximum live bonus",
          env.terminal_cost > max_live,
          "terminal %.3f max live %.3f" % (env.terminal_cost, max_live))


def test_adaptive_reward_contract():
    """Normal tracking is valid; drift is optional and failure remains dominated."""
    from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", 64, "cuda:0", ref_mode="closed_random",
        control_mode="pedals", train=True, drift_mode=True,
        reward_mode="adaptive_envelope", terminal_cost=100.0,
        spawn_beta_frac=0.0, drift_spawn_spin_frac=0.0,
        drift_spawn_trim_frac=0.0, speed_target_mode="limit",
        continuing_task=True)
    env.reset()
    _, reward, done, _ = env.step(torch.zeros((64, 3), device="cuda:0"))
    live = done == 0
    check("adaptive live reward is nonnegative",
          bool((reward[live] >= 0.0).all()))
    no_drift = live & (env._now["beta"].abs() < 0.12) & (
        env._now["kappa_power"][:, 2:4].mean(1) < 0.25)
    check("normal tracking receives reward without mandatory drift",
          bool(no_drift.any()) and float(reward[no_drift].mean()) > 0.0,
          "normal samples %d" % int(no_drift.sum()))
    env._now["failed"] = torch.ones(64, device="cuda:0")
    fail = env.reward()
    check("adaptive failure is strictly negative", float(fail.max()) < 0.0)
    max_live = (env.adaptive_alive + env.adaptive_tracking +
                env.adaptive_performance + env.adaptive_drift +
                env.adaptive_guidance + env.adaptive_smooth)
    check("adaptive terminal cost dominates maximum live bonus",
          env.terminal_cost > max_live,
          "terminal %.3f max live %.3f" % (env.terminal_cost, max_live))


def test_carsim_policy_config_contract():
    """CarSim policy evaluation must reuse the trained continuing reference."""
    import inspect
    from envs.carsim_drift_env import (CarSimDriftEnv,
                                       effective_mu_from_road_scale,
                                       policy_takeover_trim)
    from experiments.carsim_bridge.run_policy_carsim import prepare_task_config
    cfg = prepare_task_config({
        "plant": "fast", "control_mode": "pedals", "train": True,
        "ref_mode_cli": "closed_random", "continuing_task": True,
        "closed_random_radius": 2.0, "randomize_param": {"D": [1.0, 2.0]},
    })
    check("CarSim env exposes reference-mode selection",
          "ref_mode" in inspect.signature(CarSimDriftEnv.__init__).parameters)
    check("CarSim evaluator preserves trained closed reference",
          cfg.get("ref_mode") == "closed_random" and cfg.get("continuing_task") is True)
    check("CarSim evaluator strips FastPlant-only randomisation",
          "plant" not in cfg and "randomize_param" not in cfg)
    k = torch.tensor([0.1], device="cuda:0")
    beta = torch.tensor([-0.2], device="cuda:0")
    speed = torch.tensor([8.0], device="cuda:0")
    steer, throttle = policy_takeover_trim(k, beta, speed, True, 0.0, 0.45)
    check("CarSim A0 takeover has exactly zero privileged drift trim",
          float(steer.abs().max()) == 0.0 and float(throttle.abs().max()) == 0.0)
    steer1, throttle1 = policy_takeover_trim(k, beta, speed, True, 1.0, 0.45)
    check("CarSim later drift trim obeys curriculum fraction",
          abs(float(steer1[0]) + 0.25) < 1e-7 and
          abs(float(throttle1[0]) - 0.50) < 1e-7)
    check("CarSim road multipliers map to FastPlant effective mu",
          abs(effective_mu_from_road_scale(0.50) - 1.0623) < 1e-8 and
          abs(effective_mu_from_road_scale(0.75) - 1.59345) < 1e-8 and
          abs(effective_mu_from_road_scale(1.00) - 2.1246) < 1e-8)


def test_observable_teacher_contract():
    """The deployment teacher cannot label identical observations from true mu."""
    from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
    from experiments.continuous_drift_iwd.observable_teacher import ObservableEnvelopeTeacher
    from fast_plant import PARAM_INDEX
    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", 2, "cuda:0", ref_mode="closed_random",
        control_mode="pedals", train=True, continuing_task=True,
        closed_random_radius=2.0, preview_time="0.35,0.7,1.2",
        spawn_beta_frac=0.0, drift_spawn_spin_frac=0.0,
        drift_spawn_trim_frac=0.0)
    env.reset()
    # Make all deployable signals identical while deliberately changing the
    # hidden plant friction.  A privileged speed teacher would now disagree.
    env.s[1].copy_(env.s[0])
    for name in ("signed_pos_diff", "angle_diff", "kappa_ref", "delta_pp"):
        value = getattr(env, name)
        value[1].copy_(value[0])
    env.kappa_preview[1].copy_(env.kappa_preview[0])
    env.plant.p[0, PARAM_INDEX["D"]] = 1.0623
    env.plant.p[1, PARAM_INDEX["D"]] = 2.1246
    teacher = ObservableEnvelopeTeacher(2, env.device)
    action, _ = teacher.command(env)
    check("observable teacher ignores hidden true friction for identical histories",
          bool(torch.equal(action[0], action[1])), str(action.detach().cpu().tolist()))


def test_xcar_reset_replica_contract():
    """The lightweight PPO path can reproduce XCAR's broad/zero reset mixture."""
    from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", 1024, "cuda:0", ref_mode="closed_random",
        control_mode="pedals", train=True, spawn_mode="legacy",
        spawn_beta_mag=0.40, legacy_zero_prob=0.20,
        closed_random_radius=2.0)
    env.reset()
    speed = torch.hypot(env.s[:, 3], env.s[:, 4])
    zero = speed < 1e-7
    zero_rate = float(zero.float().mean())
    beta = env.es[:, 7].abs()
    check("XCAR replica reset retains about 20 percent true rest states",
          0.15 <= zero_rate <= 0.25, "rate %.3f" % zero_rate)
    check("XCAR replica reset covers broad nonzero sideslip",
          float(beta[~zero].quantile(0.90)) > 0.25,
          "beta p90 %.3f" % float(beta[~zero].quantile(0.90)))


def test_adaptive_drift_reference_contract():
    """Large beta is requested only for feasible grip/curvature/speed cells."""
    from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
    from fast_plant import PARAM_INDEX
    env = ContinuousDriftIWDEnv(
        "fullsize_racecar_ev", 3, "cuda:0", ref_mode="circle",
        control_mode="pedals", train=False, drift_mode=True,
        beta_ref_mode="adaptive_drift", drift_beta_target=0.40)
    env.plant.p[:, PARAM_INDEX["D"]] = torch.tensor(
        [1.0623, 1.59345, 2.1246], device=env.device)
    kappa = torch.full((3,), 0.08, device=env.device)
    fast = torch.full((3,), 14.0, device=env.device)
    ref = env._reference_beta(kappa, fast).abs()
    check("adaptive beta decreases as nominal grip rises",
          float(ref[0]) > float(ref[1]) > float(ref[2]) - 1e-8,
          str(ref.detach().cpu().tolist()))
    straight = env._reference_beta(torch.zeros_like(kappa), fast)
    stopped = env._reference_beta(kappa, torch.zeros_like(fast))
    check("adaptive beta is zero on a straight", float(straight.abs().max()) < 1e-8)
    check("adaptive beta is zero at rest", float(stopped.abs().max()) < 1e-8)
    env.plant.p[:, PARAM_INDEX["D"]] = 1.0623
    gentle = env._reference_beta(torch.full_like(kappa, 0.02), fast).abs()
    tight = env._reference_beta(torch.full_like(kappa, 0.08), fast).abs()
    check("adaptive beta increases with path curvature",
          bool(torch.all(tight > gentle)),
          "%s vs %s" % (tight.detach().cpu().tolist(), gentle.detach().cpu().tolist()))

if __name__ == "__main__":
    print("contract tests")
    for fn in (test_preset, test_driveline_inertia_and_slip_sign,
               test_carsim_motor_power_semantics,
               test_lsd_generalized_viscous_contract,
               test_calibrated_parameters_affect_dynamics,
               test_physical_sign_contract,
               test_xcar_isotropic_tire_contract,
               test_power_limited_wheel_solver_is_stable,
               test_tire_relaxation_candidate,
               test_randomisation_covers_nominal,
               test_pretrain_gate_matches_training_dr, test_steering_conversion,
               test_wheel_order, test_actuator, test_observation_contract,
               test_compiled_resume_key_contract,
               test_sensor_actuator_randomisation,
               test_drift_stack_is_opt_in, test_rest_spawn_has_no_countersteer,
               test_vehicle_randomisation, test_continuing_task_contract,
               test_preview_steering_sign,
               test_closed_random_refresh_contract,
               test_bounded_reward_contract, test_adaptive_reward_contract,
               test_carsim_policy_config_contract,
               test_observable_teacher_contract,
               test_xcar_reset_replica_contract,
               test_adaptive_drift_reference_contract):
        print(fn.__name__)
        fn()
    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        sys.exit(1)
    print("ALL CONTRACT TESTS PASSED")
