"""Behavioural regressions from the 2026-11-14 audit; no policy training."""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "xcar-simulation"), str(ROOT / "utils")]
from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
from fast_plant import PARAM_INDEX, load_preset
from utils.drift_network import DriftBuilder, beta_supervision
from experiments.continuous_drift_iwd.eval import Policy
from rl_games.algos_torch.models import ModelA2CContinuousLogStd


def make(train=True, beta_ref_mode="adaptive_drift"):
    torch.manual_seed(19)
    e = ContinuousDriftIWDEnv("fullsize_racecar_ev", 2, "cpu",
        ref_mode="closed_random", num_paths=2, closed_random_radius=2,
        control_mode="pedals", train=train, continuing_task=True,
        spawn_mode="legacy", equilibrium_reset_prob=0, legacy_zero_prob=0,
        drift_mode=True, beta_ref_mode=beta_ref_mode,
        reward_mode="adaptive_envelope", drift_spawn_trim_frac=0,
        drift_spawn_spin_frac=0, sensor_dr=False, actuator_dr=False, stall_window=11)
    e.reset()
    return e


def test_environment():
    rest = make()
    rest.legacy_zero_prob = 1.
    rest.is_done[:] = True
    rest.reset_done_envs()
    start = rest.path_bank[rest.path_generation, rest.path_id, 0]
    torch.testing.assert_close(rest.s[:, :3], start[:, :3])
    torch.testing.assert_close(rest.s[:, 3:6], torch.zeros_like(rest.s[:, 3:6]))
    assert torch.all(rest.progress == 0)
    a, b = make(), make()
    for e in (a, b):
        e.step_count[:] = 20
        e._stall_anchor.copy_(e.s[:, :2])
        e.no_progress_count.zero_()
        e.stall_move_min = 1000
    b.no_progress_count[1] = 11
    action = torch.tensor([[.2, .4, 0.], [.2, .4, 0.]])
    oa, ra, da, _ = a.step(action)
    ob, rb, db, _ = b.step(action)
    assert not da[0] and not db[0] and db[1]
    for x, y in ((oa[0], ob[0]), (ra[0], rb[0]), (a.s[0], b.s[0]),
                 (a.no_progress_count[0], b.no_progress_count[0])):
        torch.testing.assert_close(x, y)
    for name in ("_prev", "_now"):
        for key in getattr(a, name):
            torch.testing.assert_close(getattr(a, name)[key][0], getattr(b, name)[key][0])
    history = {k: v.clone() for k, v in a._prev.items()}
    a.obs(); a.obs()
    for key in history:
        torch.testing.assert_close(history[key], a._prev[key])
    e = make(False)
    e._track_pending = False
    e.s[:, 3:5] = torch.tensor([10., 0.])
    first = e.obs().clone()
    e.s[:, 3:5] = torch.tensor([10 * math.cos(.3), 10 * math.sin(.3)])
    second = e.obs().clone()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    throttle = torch.linspace(0, 1, 101)
    score = e._positive_target_gate(throttle, e.drift_throttle_gate)
    assert torch.all(torch.diff(score) >= 0) and score[-1] == 1
    # Test the constraint where electrical efficiency CAN affect shaft torque.
    e.plant.p[:, PARAM_INDEX["motor_efficiency"]] = torch.tensor([.5, 1.5])
    torques = e.plant._motor_torque(torch.full((2, 2), 240.),
                                   torch.ones(2), torch.zeros(2))[0]
    assert torques[1] > torques[0] * 1.05, torques
    return {"rest_spawn_matches_path_pose": True,
            "batch_independence": True, "repeat_obs_idempotent": True,
            "hidden_course_actor_maxdiff": float((first-second).abs().max()),
            "throttle_monotonic": True, "battery_limited_axle_torque": torques.tolist()}


def test_reward_ranking(beta_ref_mode="adaptive_drift"):
    """Counterfactual score ordering; this does not prove dynamical feasibility."""
    e = make(False, beta_ref_mode=beta_ref_mode)
    e.speed_target_mode = "limit"
    e.plant.tyre_scale.fill_(.75)
    e.kappa_ref.fill_(.075)
    e.kappa_preview.fill_(.075)
    speed = e._reference_speed()
    target = e._reference_beta(e.kappa_ref[:, None], speed).squeeze(1)
    base = {key: value.clone() for key, value in e._now.items()}
    def score(beta_scale=1., speed_scale=1., slip=.7, throttle=.6, delta=-.25,
              brake=0., raw_jump=0., offpath=0.):
        e._now = {key: value.clone() for key, value in base.items()}
        for key in ("failed", "angle_diff", "e_pos", "signed_pos_diff"):
            e._now[key].zero_()
        e._now["signed_pos_diff"].fill_(offpath)
        e._now["beta"] = target * beta_scale
        e._now["v"] = speed * speed_scale
        e._now["kappa_ref"].fill_(.075)
        e._now["kappa_power"][:, 2:4] = slip
        e._now["delta"].fill_(delta)
        e._now["raw_delta"].fill_(delta)
        e._prev = {key: value.clone() for key, value in e._now.items()}
        e._prev["raw_delta"] -= raw_jump
        e.u_cmd[:] = torch.tensor([delta, throttle, brake])
        return float(e.reward().mean())
    values = {"qualified_drift": score(),
              "same_speed_grip": score(beta_scale=0, slip=.02, delta=.15),
              "slow_grip": score(beta_scale=0, speed_scale=.2, slip=.02, delta=.15),
              "wrong_side_slide": score(beta_scale=-1),
              "raw_steer_chatter": score(raw_jump=.2),
              "below_target_no_brake": score(speed_scale=.8),
              "below_target_braking": score(speed_scale=.8, brake=.5),
              "offpath_drift": score(offpath=e.offpath_limit)}
    assert values["qualified_drift"] > values["same_speed_grip"] > values["slow_grip"], values
    for key in ("wrong_side_slide", "raw_steer_chatter", "offpath_drift"):
        assert values[key] < values["qualified_drift"], values
    assert values["below_target_braking"] < values["below_target_no_brake"], values
    e.plant.tyre_scale.fill_(1.)
    high_grip = e._reference_beta(e.kappa_ref[:, None], e._reference_speed())
    assert high_grip.abs().max() < 1e-6
    e.kappa_ref.zero_()
    straight = e._reference_beta(e.kappa_ref[:, None], e._reference_speed())
    assert straight.abs().max() < 1e-6
    return {"scores": values, "high_grip_and_straight_beta_target_zero": True,
            "scope": "counterfactual reward semantics, not proof of reachable states"}


def test_terminal_zero_contract():
    from types import SimpleNamespace
    from rl_games.common.a2c_common import A2CBase
    e = make()
    e.legacy_zero_prob = 1.
    e.terminal_cost = 0.
    e.is_done[:] = True
    e.reset_done_envs(); e.obs(); e.train = False
    for _ in range(12):
        _, _, done, info = e.step(torch.zeros((2, 3)))
        if done.all():
            break
    assert torch.all(done == 1)
    assert not info['time_outs'].any()
    assert torch.all(e.reward() == 0)
    dummy = SimpleNamespace(horizon_length=2, gamma=.999, tau=.97)
    values = torch.zeros(2, 1, 1)
    rewards = torch.tensor([[[1.]], [[0.]]])
    before_dones = torch.zeros(2, 1)
    def advantages(last, done):
        return A2CBase.discount_values(dummy, torch.tensor([done]),
            torch.tensor([[last]]), before_dones, values, rewards)
    torch.testing.assert_close(advantages(0., 1.), advantages(1000., 1.))
    assert advantages(1000., 0.)[-1].item() > 900
    return {'zero_cost_failure_remains_terminal': True,
            'failure_has_no_timeout_bootstrap': True,
            'gae_failure_cuts_future_value': True,
            'continuing_rollout_keeps_bootstrap': True}


def test_path_curvature_cap():
    e = make(False)
    e.trajectory_generator.switch_fraction = 1.
    torch.manual_seed(123)
    original = e._generate_fresh_path_bank()
    assert original[..., 3].abs().max() > .3
    e.closed_curvature_cap = .14
    torch.manual_seed(123)
    bounded = e._generate_fresh_path_bank()
    assert bounded[..., 3].abs().max() <= .140001
    assert torch.all(bounded[..., 3].min(1).values < -.13)
    assert torch.all(bounded[..., 3].max(1).values > .13)
    factor = original[..., 3].abs().amax(1) / .14
    torch.testing.assert_close(bounded[..., :2], original[..., :2]*factor[:,None,None], atol=.003, rtol=.001)
    return {'geometry_scaled_not_labels_clipped': True,
            'fresh_path_generations_respect_cap': True,
            'max_curvature_inv_m': float(bounded[...,3].abs().max())}


def test_network():
    cfg = yaml.safe_load((ROOT / "runner_config.yaml").read_text())["params"]["network"]
    cfg["mlp"]["units"] = [256, 124, 64]
    cfg["rnn"].update(units=128, layers=1, concat_output=True, before_mlp=True)
    builder = DriftBuilder(); builder.load(cfg)
    model = ModelA2CContinuousLogStd(builder).build({
        "actions_num": 3, "input_shape": (27,), "num_seqs": 2,
        "value_size": 1, "normalize_input": True, "normalize_value": True})
    model.eval()
    obs = torch.randn(2, 27)
    def infer(states):
        return model({"is_train": False, "obs": obs, "rnn_states": states})
    h = model.get_default_rnn_state()
    with torch.no_grad():
        base = infer(h)
        model.a2c_network.beta_labels = torch.tensor([[.4, 10.], [-.3, 8.]])
        supervised = infer(h)
        torch.testing.assert_close(base["mus"], supervised["mus"], atol=0, rtol=0)
    model.a2c_network.beta_labels = None
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "actor.pth"
        torch.save({"model": model.state_dict()}, path)
        policy = Policy(path, "cpu")
        torch.testing.assert_close(base["mus"], policy.act(obs), atol=1e-6, rtol=1e-6)
        with torch.no_grad():
            nxt = infer(base["rnn_states"])
        torch.testing.assert_close(nxt["mus"], policy.act(obs), atol=1e-6, rtol=1e-6)
    model.a2c_network.beta_labels = torch.tensor([[.4, 10.], [-.3, 8.]])
    infer(h)
    aux = model.get_aux_loss()["beta"]
    aux.backward()
    assert model.a2c_network.rnn.rnn.weight_ih_l0.grad.abs().sum() > 0
    assert model.a2c_network.beta_head[0].weight.grad.abs().sum() > 0
    # Auxiliary objective must not directly update the action MLP.
    assert model.a2c_network.actor_mlp[0].weight.grad is None
    pred = torch.tensor([[.4]], requires_grad=True)
    low = beta_supervision(pred, torch.tensor([[0., 0.]]))
    mid = beta_supervision(pred, torch.tensor([[0., 2.5]]))
    high = beta_supervision(pred, torch.tensor([[0., 4.]]))
    assert low == 0 and 0 < mid < high
    low.backward()
    assert pred.grad == 0
    return {"architecture": "obs27 + GRU128 -> 256/124/64 -> action3",
            "inference_parity": True, "labels_do_not_change_action": True,
            "beta_loss_low_mid_high": [float(x) for x in (low, mid, high)],
            "auxiliary_updates_gru": True}


def test_projection_sensors_motor():
    e = make(False)
    for start in (0, e.num_waypoints - 60):
        e.progress.fill_(start)
        e.s[:, :2] = e.waypoints_at(e.progress[:, None])[:, 0, :2]
        e._projection_xy = e.s[:, :2].clone()
        expected = (e.progress + 100).remainder(e.num_waypoints)
        e.s[:, :2] = e.waypoints_at(expected[:, None])[:, 0, :2]
        e._tracking_mask = None
        e.find_nearest_waypoint(e.s[:, 0], e.s[:, 1], e.s[:, 2],
                               e.es[:, 7], torch.zeros(e.n))
        assert torch.equal(e.progress, expected), (e.progress, expected)
        assert not e.projection_search_edge.any()
    bank = e.path_bank[0]
    spacing = torch.linalg.vector_norm(torch.roll(bank[..., :2], -1, 1)-bank[..., :2], dim=-1)
    assert float((spacing.max(1).values / spacing.min(1).values).max()) < 1.005
    e._track_pending = False
    before = e.obs().clone()
    e.wheel_radius *= 1.1
    torch.testing.assert_close(before, e.obs(), atol=0, rtol=0)
    e.plant._motor_torque_state[:] = torch.tensor([100., 200.])
    e.plant.reset_guess(torch.tensor([1]))
    torch.testing.assert_close(e.plant._motor_torque_state, torch.tensor([100., 0.]))
    from generate_segment_racetrack import TrajGen
    generator = TrajGen(2512, device="cpu")
    generator.switch_fraction = 1.
    eight = TrajGen.resample_closed(generator.generate_closed_random_waypoints(2, radius=2.))
    assert torch.all(eight[:, :, 3].min(1).values < -.1)
    assert torch.all(eight[:, :, 3].max(1).values > .1)
    return {"projection_100_waypoints_and_seam": True, "uniform_arclength": True,
            "unknown_radius_does_not_leak": True, "motor_reset_is_local": True,
            "smooth_eight_changes_curvature_sign": True}


def test_carsim():
    from envs.carsim_drift_env import CarSimDriftEnv
    e = None
    try:
        e = CarSimDriftEnv("flat_mu075", device="cpu", ref_mode="closed_random",
            num_paths=2, closed_random_radius=2, drift_mode=True,
            beta_ref_mode="adaptive_drift", equilibrium_reset_prob=1,
            drift_spawn_trim_frac=0, drift_spawn_spin_frac=0, sensor_dr=True,
            sensor_delay_max=2, sensor_dropout_prob=0, actuator_dr=False)
        for key in ("_sensor_white_std", "_sensor_bias_std", "_sensor_rw_std"):
            getattr(e, key).zero_()
        initial = e.reset().clone()
        e._sensor_pending.fill_(True)
        torch.testing.assert_close(initial, e.obs(), atol=0, rtol=0)
        torch.testing.assert_close(e._stall_anchor, e.s[:, :2])
        assert e.no_progress_count.item() == 0
        p = load_preset("fullsize_racecar_ev")
        assert abs(e.car.col("m").item() - p["m"]) < .001
        torch.testing.assert_close(e.wheel_radius, torch.tensor([[p["rF"]]*2+[p["rR"]]*2]))
        return {"first_packet_valid": True, "mass": e.car.col("m").item(),
                "radii": e.wheel_radius.tolist(), "initial_stall_count": 0}
    finally:
        if e is not None:
            e.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--carsim", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = {"environment": test_environment(), "reward_ranking": test_reward_ranking(),
               "geometry_reward_ranking": test_reward_ranking("geometry_drift"),
               "terminal_zero": test_terminal_zero_contract(),
               "path_curvature_cap": test_path_curvature_cap(),
               "network": test_network(),
               "projection_sensor_motor": test_projection_sensors_motor()}
    if args.carsim:
        results["carsim"] = test_carsim()
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
