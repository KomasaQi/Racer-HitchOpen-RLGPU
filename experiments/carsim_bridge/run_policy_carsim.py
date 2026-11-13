"""Run the trained drift policy inside CarSim (no OU disturbance).

One episode per scenario: the car starts from rest at the origin, follows one of our
own reference roads, and the shared task logic scores it.
"""
import argparse
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

def resolve_checkpoint(path):
    """Resolve a checkpoint file or an nn directory without import-time I/O."""
    if os.path.isfile(path):
        return os.path.abspath(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    cands = [os.path.join(path, f) for f in os.listdir(path)
             if f.startswith("last_") and f.endswith(".pth")]
    if not cands:
        fallback = os.path.join(path, "continuous_drift_iwd.pth")
        if os.path.isfile(fallback):
            return fallback
        raise FileNotFoundError("no checkpoint in " + path)
    # Select by epoch number: ep_90 sorts lexically after ep_290.
    return max(cands, key=lambda p: int(re.search(r"ep_(\d+)", p).group(1))
               if re.search(r"ep_(\d+)", p) else -1)


def prepare_task_config(raw):
    """Strip training-only fields while preserving the trained reference task."""
    cfg = dict(raw)
    ref_mode = cfg.get("ref_mode_cli", cfg.get("ref_mode", "hybrid"))
    for key in ("plant", "control_mode", "train", "ref_mode_cli",
                "disturbance_param", "randomize_param", "quiet", "num_paths"):
        cfg.pop(key, None)
    cfg["ref_mode"] = ref_mode
    return cfg


def run(scenario, policy_path, config_path, steps=600, seed=0, entry_speed=0.0,
        output_dir="."):
    torch.manual_seed(seed)
    # reconstruct the trained task configuration, otherwise the actor sees a
    # different preview geometry from the one it was trained with
    with open(config_path, encoding="utf-8") as f:
        cfg = prepare_task_config(json.load(f))
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
    # a drifting car legitimately sits at low speed for long stretches (it is sliding
    # sideways rather than driving forward), so the stall rule must not fire on it
    if os.environ.get("CARSIM_STALL_SPEED") is not None:
        cfg["stall_speed"] = float(os.environ["CARSIM_STALL_SPEED"])
    if os.environ.get("CARSIM_STALL_WINDOW") is not None:
        cfg["stall_window"] = int(os.environ["CARSIM_STALL_WINDOW"])
    if os.environ.get("CARSIM_COURSE_FAIL") is not None:
        cfg["course_fail"] = float(os.environ["CARSIM_COURSE_FAIL"])
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
    _solver_t0 = float(env.car.sim._t)
    speed, beta_err, epos, eyaw, n = 0.0, 0.0, 0.0, 0.0, 0
    tracking_hits = 0
    envelope_hits = 0
    drift_active_hits = 0
    drift_quality_hits = 0
    trace = []
    strict_hits = 0
    strict_samples = 0
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
            _wheel = env.s[0, 6:10] * 0.332
            _kp = float(env.extras["kappa_power"][0, 2:4].mean())
            _util = float(es[0, 6] ** 2 * env.kappa_ref[0].abs()
                          / max(env.grip * 9.81, 1e-6))
            _ratio = float(_wheel[2:4].abs().mean()
                           / _wheel[:2].abs().mean().clamp_min(0.5))
            _counter = bool(float(env.u_cmd[0, 0] * env.kappa_ref[0]) < 0.0
                            and abs(float(env.u_cmd[0, 0])) >= 0.03)
            _track = (abs(float(sp[0, 0])) <= 1.50
                      and abs(float(env.angle_diff.reshape(-1)[0])) <= 0.35)
            # The CarSim flat_muNNN road multiplier scales the nominal tyre
            # peak.  2.1246 is the production fullsize tyre's measured nominal
            # peak used by the FastPlant trim/speed-envelope contract.
            _surface_scale = float(scenario.split("mu")[-1]) / 100.0
            _mu_eff = 2.1246 * _surface_scale
            _kap_limit = max(abs(float(env.kappa_ref[0])),
                             float(env.kappa_preview[0].abs().max()), 1e-2)
            _speed_target = min(float(cfg.get("speed_cap", 1e9)),
                                float(cfg.get("speed_margin", 0.9))
                                * (_mu_eff * 9.81 / _kap_limit) ** 0.5)
            _speed_ratio = float(es[0, 6]) / max(_speed_target, 0.5)
            _envelope = _track and 0.70 <= _speed_ratio <= 1.20
            _drift_active = (float(env.u_cmd[0, 1]) >= 0.35 and _kp >= 0.25
                             and abs(float(es[0, 7])) >= 0.20 and _ratio >= 1.15)
            _drift_quality = _drift_active and _counter and _envelope
            _strict = (float(env.u_cmd[0, 1]) >= 0.35 and _kp >= 0.25
                       and abs(float(es[0, 7])) >= 0.20 and _ratio >= 1.15
                       and _counter and 0.70 <= _util <= 1.10
                       and abs(float(sp[0, 0])) <= 1.50)
            strict_hits += int(_strict)
            strict_samples += 1
            tracking_hits += int(_track)
            envelope_hits += int(_envelope)
            drift_active_hits += int(_drift_active)
            drift_quality_hits += int(_drift_quality)
            trace.append(((k + 1) * float(env.dt),
                          float(torch.hypot(env.s[0, 3], env.s[0, 4])),
                          float(es[0, 7]), float(br[0, 0]), _db,
                          float(sp[0, 0]), float(env.u_cmd[0, 0]),
                          float(env.u_cmd[0, 1]), float(env.u_cmd[0, 2]),
                          # rear wheel surface speeds and the rear longitudinal slip, so a
                          # drift can be judged directly instead of inferred: a spin gives a
                          # large |beta| but no rear overslip, and a coasting car gives neither
                          float(_wheel[0]), float(_wheel[1]),
                          float(_wheel[2]), float(_wheel[3]), _kp, _ratio,
                          _util, float(_counter), float(_strict)))
            # the extended state is not refreshed by begin_policy_episode() until the
            # first step, which is why this used to print 0.00; the world velocity is
            # always current
            speed += float(torch.hypot(env.s[0, 3], env.s[0, 4]))
            beta_err += float((es[0, 7] - br[0, 0]).abs())
            epos += float(sp[0, 0].abs())
            eyaw += abs(float(env.angle_diff.reshape(-1)[0]))
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
    _out = output_dir
    os.makedirs(_out, exist_ok=True)
    with open(os.path.join(_out, "trace_%s.csv" % scenario), "w") as fh:
        fh.write("t,v,beta,beta_ref,dbeta,epos,steer,thr,brk,"
                 "wFL,wFR,wRL,wRR,kappa_rear,wheel_ratio,friction_util,counter,strict\n")
        for row in trace:
            fh.write(",".join("%.6f" % float(x) for x in row) + "\n")
    # the task clock and the solver clock are not the same thing: env.dt is the
    # control period and env.car.sim._t is what CarSim actually integrated.  Report both
    # so the two can be compared instead of assuming they agree.
    return {"scenario": scenario, "steps": done_step, "reason": _reason,
            "solver_seconds": float(env.car.sim._t) - _solver_t0,
            "dsteer": dsteer / max(n, 1), "flips": sign_flips,
            "metrics": steering_metrics(applied, dt=env.dt),
            "seconds": done_step * env.dt,
            "speed": speed / max(n, 1),
            "beta_err": beta_err / max(n, 1),
            "epos": epos / max(n, 1),
            "eyaw": eyaw / max(n, 1),
            "tracking_duty": tracking_hits / max(strict_samples, 1),
            "envelope_duty": envelope_hits / max(strict_samples, 1),
            "drift_active_duty": drift_active_hits / max(strict_samples, 1),
            "drift_quality_duty": drift_quality_hits / max(strict_samples, 1),
            "strict_duty": strict_hits / max(strict_samples, 1)}


def main():
    ap = argparse.ArgumentParser(description="Evaluate one explicit policy/config in CarSim")
    ap.add_argument("checkpoint", help="checkpoint file or nn directory")
    ap.add_argument("--config", required=True, help="matching env_config.json")
    ap.add_argument("--scenarios", default="flat_mu050,flat_mu075,flat_mu100")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--entry-speed", type=float, default=0.0)
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()
    ckpt = resolve_checkpoint(args.checkpoint)
    config = os.path.abspath(args.config)
    if not os.path.isfile(config):
        raise FileNotFoundError(config)
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    print("checkpoint:", ckpt)
    print("config:", config)
    rows = []
    for sc in scenarios:
        # Fair surface A/B: every CarSim scenario receives the same random path
        # and actor initial state.  Incrementing the seed by scenario silently
        # confounded grip with road geometry in the former evaluator.
        r = run(sc, ckpt, config, steps=args.steps, seed=args.seed,
                entry_speed=args.entry_speed, output_dir=args.output_dir)
        rows.append(r)
        print("%-12s %6.1f s  v %5.2f m/s  |dbeta| %.3f  |e_pos| %.2f m  reason %s"
              % (r["scenario"], r["seconds"], r["speed"], r["beta_err"], r["epos"],
                 r.get("reason", "?")))
        print("             clocks: task %.2f s   solver %.2f s   (%d control steps)"
              % (r["seconds"], r["solver_seconds"], r["steps"]))
        print("             steer: " + format_metrics(r["metrics"]))
        print("             adaptive: tracking %.2f %%  near-envelope %.2f %%  "
              "powered-slide %.2f %%  tracked-quality-drift %.2f %%"
              % (100.0 * r["tracking_duty"], 100.0 * r["envelope_duty"],
                 100.0 * r["drift_active_duty"], 100.0 * r["drift_quality_duty"]))
        print("             strict powered-drift duty: %.2f %%" % (100.0 * r["strict_duty"]))
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
