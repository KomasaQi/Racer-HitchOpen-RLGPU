import os
import sys
file_path = os.path.dirname(__file__)
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from utils.rlgame_utils import RLGPUEnv, RLGPUAlgoObserver
from envs.fixed_circle_iwd import FixedCircleIWDEnv
from envs.eight_drift_iwd import EightDriftIWDEnv
from envs.state_tracker_iwd import StateTrackerIWDEnv
from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
import torch
import yaml
import argparse
import glob
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from icecream import ic
import pprint

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
ic.configureOutput(argToStringFunction=lambda x: pprint.pformat(x, sort_dicts=False))

parser = argparse.ArgumentParser()
parser.add_argument("train_or_test", type=str, help="Train or test")
parser.add_argument("env", type=str)
parser.add_argument("--car-preset", type=str, default="racecar")
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--plant", type=str, default="fast", choices=["fast", "iwd"],
                    help="fast = FastPlant (10 states, pedals); iwd = legacy XCAR plant")
parser.add_argument("--control-mode", type=str, default="pedals", choices=["pedals", "wheel_speed"])
parser.add_argument("--dt", type=float, default=None,
                    help="control period; default 0.05 s (20 Hz) for the fast plant, 0.01 s for iwd")
parser.add_argument("--physics-dt", type=float, default=0.01, help="physics step of the fast plant (100 Hz)")
parser.add_argument("--rnn", action='store_true')
parser.add_argument("--disturbed", action='store_true')
parser.add_argument("--randomize-tyre", nargs='?', const='big', default=None, type=str)
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--exp-name", type=str, default="default")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--num-parallel", type=int, default=16384)
parser.add_argument("--minibatch", type=int, default=65536,
                    help="PPO minibatch size; capped at num_parallel*horizon")
parser.add_argument("--lr-schedule", type=str, default="adaptive")
parser.add_argument("--mini-epochs", type=int, default=5)
parser.add_argument("--mlp-size-last", type=int, default=64)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--tau", type=float, default=0.97)
parser.add_argument("--horizon", type=int, default=128)
parser.add_argument("--score-to-win", type=int, default=20000)
parser.add_argument("--save-freq", type=int, default=10)
parser.add_argument("--epoch-index", type=int, default=-1, help="For test only, -1 for using latest")
parser.add_argument("--latent-size", type=int, default=8)
parser.add_argument("--quiet", action='store_true')
parser.add_argument("--aux-reward-decay-steps", type=int, default=0)
parser.add_argument("--aux-reward-coef", type=float, default=1.)
parser.add_argument("--env-variant", type=str, default="")
parser.add_argument("--ref-mode", type=str, default="hybrid")
parser.add_argument("--beta-scale", type=float, default=1.0,
                    help="scale of the reference sideslip target (1.0 = XCAR, +-1 rad)")
parser.add_argument("--beta-weight", type=float, default=1.6,
                    help="reward weight of the sideslip tracking term")
parser.add_argument("--pos-weight", type=float, default=2.4,
                    help="reward weight of the lateral tracking error")
parser.add_argument("--spawn-mode", type=str, default="equilibrium",
                    choices=["equilibrium", "legacy"])
parser.add_argument("--spawn-speed", type=float, default=6.5,
                    help="drift speed used by the equilibrium spawn [m/s]")
parser.add_argument("--spawn-beta-frac", type=float, default=1.0,
                    help="fraction of the reference sideslip at spawn (curriculum)")
parser.add_argument("--speed-weight", type=float, default=0.0,
                    help="weight of the speed regulation term around --speed-target")
parser.add_argument("--speed-target", type=float, default=None,
                    help="speed target of the regulation term (default: spawn speed)")
parser.add_argument("--sigma-init", type=float, default=0.0,
                    help="initial log std of the action distribution (0 -> std 1.0)")
parser.add_argument("--entropy-coef", type=float, default=0.0)
parser.add_argument("--dir-weight", type=float, default=0.5,
                    help="reward weight of the course (heading) error")
parser.add_argument("--yaw-weight", type=float, default=0.0,
                    help="reward weight of the yaw rate tracking term")
parser.add_argument("--yaw-scale", type=float, default=0.5, help="normalisation of the yaw error [rad/s]")
parser.add_argument("--alive-bonus", type=float, default=0.0,
                    help="constant reward per surviving control step")
parser.add_argument("--asymmetric", action="store_true",
                    help="privileged critic on a separate network (no shared backbone)")
parser.add_argument("--resume", type=str, default=None,
                    help="checkpoint to warm start training from (curriculum continuation)")
parser.add_argument("--stall-speed", type=float, default=2.0,
                    help="speed below which a car counts as stalled (episode failed)")
parser.add_argument("--offpath-scale", type=float, default=1.0,
                    help="off path failure threshold in vehicle length scales")
parser.add_argument("--course-fail", type=float, default=1.0,
                    help="course error failure threshold [rad]")
parser.add_argument("--pos-cap", type=float, default=1.0,
                    help="saturation of the squared lateral error penalty")
parser.add_argument("--yaw-cap", type=float, default=4.0,
                    help="saturation of the squared yaw rate error")
parser.add_argument("--kappa-cap", type=float, default=4.0,
                    help="saturation of the squared curvature error")
parser.add_argument("--curv-scale", type=float, default=1.0,
                    help="multiplier on the hybrid reference curvature (1/radius)")
parser.add_argument("--curv-scale-range", type=str, default=None,
                    help="per path curvature scale range, e.g. 0.55,1.1 (randomised radius)")
parser.add_argument("--beta-gate-scale", type=float, default=0.8,
                    help="lateral error (vehicle lengths) at which the demanded drift fades out")
parser.add_argument("--recover-weight", type=float, default=0.0,
                    help="weight of the lateral error reduction (potential) term")
parser.add_argument("--kappa-weight", type=float, default=0.15,
                    help="weight of the curvature (yaw rate / speed) tracking term")
parser.add_argument("--pp-weight", type=float, default=0.0,
                    help="weight of the pure pursuit steering reference")
parser.add_argument("--seg-scale", type=float, default=1.0,
                    help="multiplier on the random reference segment lengths")
parser.add_argument("--obs-privileged", action="store_true",
                    help="append beta/V/vx/vy to the actor observation (upper bound)")
parser.add_argument("--speed-hinge", action="store_true",
                    help="one sided linear speed penalty instead of the quadratic")
parser.add_argument("--critic-units", type=str, default="256,128,64",
                    help="hidden units of the privileged critic MLP")
parser.add_argument("--seq-length", type=int, default=8,
                    help="recurrent chunk length for BPTT; must divide --horizon")
parser.add_argument("--num-paths", type=int, default=256,
                    help="how many reference geometries the bank holds")
parser.add_argument("--curv-smooth", type=int, default=0,
                    help="moving average width (waypoints) applied to the reference curvature")
parser.add_argument("--preview-scale", type=float, default=1.0,
                    help="multiplier on the preview distances (1 = 200/400/600 waypoints)")
parser.add_argument("--flip-prob", type=float, default=None,
                    help="curvature sign flip probability per segment (None = 0.5, independent)")
parser.add_argument("--terminal-cost", type=float, default=100.0,
                    help="reward penalty applied on the terminal step")
parser.add_argument("--stall-window", type=int, default=None,
                    help="steps over which the displacement is measured for the stall rule")
parser.add_argument("--stall-move", type=float, default=1.0,
                    help="metres the car must travel per stall window to count as moving")
parser.add_argument("--spawn-speed-min", type=float, default=None,
                    help="lower bound of the speed-deficit spawn range")
parser.add_argument("--spawn-deficit-frac", type=float, default=0.0,
                    help="fraction of spawns drawn from the speed-deficit range")
parser.add_argument("--speed-target-mode", type=str, default="signal",
                    choices=["signal", "limit"],
                    help="limit derives the target from sqrt(mu g / |kappa|)")
parser.add_argument("--speed-margin", type=float, default=0.9,
                    help="fraction of the friction-limit speed used as the target")
parser.add_argument("--speed-cap", type=float, default=None,
                    help="upper clamp on the friction-limit speed target")
parser.add_argument("--preview-time", type=str, default=None,
                    help="comma separated preview headways in seconds, e.g. 0.5,1.0,1.6")
parser.add_argument("--sigma-per-dim", type=str, default=None,
                    help="initial action std per dimension, e.g. 0.6,0.25,0.25")
parser.add_argument("--ackermann-lo", type=float, default=0.8)
parser.add_argument("--ackermann-hi", type=float, default=1.2)
parser.add_argument("--beta-max", type=float, default=0.32)
parser.add_argument("--alpha-peak", type=float, default=0.12,
                    help="rear slip angle at the tyre peak, sets the reference sideslip")
parser.add_argument("--path-refresh-every", type=int, default=8,
                    help="request a fresh path generation every N PPO epochs (0 disables)")
parser.add_argument("--beta-ref-mode", type=str, default="physics")
parser.add_argument("--prog-weight", type=float, default=0.2,
                    help="weight on raw path progress; 0 removes the incentive to over-speed")
parser.add_argument("--grip-range", type=str, default=None,
                    help="override the D range of the tyre preset, e.g. 0.6,0.9")
args = parser.parse_args()

if args.train_or_test == "test":
    # Turn off disturbance and randomization when testing
    args.disturbed = False
    args.randomize_tyre = None

def get_num_parallel():
    if args.train_or_test == "train":
        return args.num_parallel
    elif args.train_or_test == "test":
        return 1

envs = {
    "fixed_circle_iwd": lambda **kwargs: FixedCircleIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
    "eight_drift_iwd": lambda **kwargs: EightDriftIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
    "state_tracker_iwd": lambda **kwargs: StateTrackerIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
    "continuous_drift_iwd": lambda **kwargs: ContinuousDriftIWDEnv(args.car_preset, get_num_parallel(), args.device, args.ref_mode, **kwargs),
}

# Common env config
default_env_config = {
    "dt": args.dt if args.dt is not None else (0.01 if args.plant == "iwd" else 0.05),
    "plant": args.plant,
    "control_mode": args.control_mode,
    "physics_dt": args.physics_dt,
    "beta_scale": args.beta_scale,
    "beta_weight": args.beta_weight,
    "pos_weight": args.pos_weight,
    "dir_weight": args.dir_weight,
    "yaw_weight": args.yaw_weight,
    "alive_bonus": args.alive_bonus,
    "stall_speed": args.stall_speed,
    "offpath_scale": args.offpath_scale,
    "course_fail": args.course_fail,
    "pos_cap": args.pos_cap,
    "yaw_cap": args.yaw_cap,
    "kappa_cap": args.kappa_cap,
    "curv_scale": (tuple(float(x) for x in args.curv_scale_range.split(","))
                   if args.curv_scale_range else args.curv_scale),
    "beta_gate_scale": args.beta_gate_scale,
    "recover_weight": args.recover_weight,
    "kappa_weight": args.kappa_weight,
    "pp_weight": args.pp_weight,
    "seg_scale": args.seg_scale,
    "obs_privileged": args.obs_privileged,
    "seq_length": args.seq_length,
    "num_paths": args.num_paths,
    "curv_smooth": args.curv_smooth,
    "preview_scale": args.preview_scale,
    "preview_time": args.preview_time,
    "flip_prob": args.flip_prob,
    "terminal_cost": args.terminal_cost,
    "stall_window": (args.stall_window if args.stall_window is not None else max(int(round(0.2 * (2.65 / 0.35) ** 0.5 / 0.05)), 2)),
    "stall_move": args.stall_move,
    "speed_hinge": args.speed_hinge,
    "asymmetric": args.asymmetric,
    "yaw_scale": args.yaw_scale,
    "spawn_mode": args.spawn_mode,
    "spawn_speed": args.spawn_speed,
    "spawn_beta_frac": args.spawn_beta_frac,
    "spawn_speed_min": (args.spawn_speed_min if args.spawn_speed_min is not None else args.spawn_speed),
    "spawn_deficit_frac": args.spawn_deficit_frac,
    "speed_target_mode": args.speed_target_mode,
    "speed_margin": args.speed_margin,
    "ackermann_lo": args.ackermann_lo,
    "ackermann_hi": args.ackermann_hi,
    "beta_ref_mode": args.beta_ref_mode,
    "beta_max": args.beta_max,
    "alpha_peak": args.alpha_peak,
    "prog_weight": args.prog_weight,

    "speed_cap": (args.speed_cap if args.speed_cap is not None else float("inf")),
    "speed_weight": args.speed_weight,
    "speed_target": (args.speed_target if args.speed_target is not None
                     else args.spawn_speed),
    "disturbance_param": (0.97, 0.03) if args.disturbed else None,
    "randomize_param": {
        None: {},
        "big": {
            "B": [0.2, 3.],
            "C": [0.5, 3.],
            "D": [0.1, 0.5],
        },
        "small": {
            "B": [0.8, 1.],
            "C": [2, 2.5],
            "D": [0.3, 0.4],
        },
        # the 1:8 XCAR preset of this plant (nominal B 4.5, C 1.8, D 0.35):
        # +/-20 % shape and D in [0.30, 0.40], as randomised in the XCAR paper
        "xcar_scale": {
            "B": [3.6, 5.4],
            "C": [1.6, 2.0],
            "D": [0.30, 0.40],
        },
        # full-size EV preset: +/-20 % on the Magic Formula shape and +/-10 % peak
        # friction, loosely following the XCAR paper's tire randomization
        "ev": {
            "B": [8., 12.],
            "C": [1.7, 2.1],
            "D": [0.9, 1.1],
        },
        # "different road surfaces": peak friction 0.75 - 1.25
        # high friction biased: the evaluated weak spot is the grippy surface,
        # where the policy prefers grip driving over the reference sideslip
        "ev_high": {
            "B": [7., 13.],
            "C": [1.6, 2.2],
            "D": [1.05, 1.40],
        },
        # road friction for the drift task: a real drift lives between 0.5 and
        # 1.0, so 1.25 is dropped.  B and C keep the shape spread but are not
        # widened further (an uncalibrated tyre does not need more shape spread).
        "ev_drift": {
            "B": [8.0, 12.0],
            "C": [1.7, 2.1],
            "D": [0.50, 1.00],
        },
        "ev_wide": {
            "B": [7., 13.],
            "C": [1.6, 2.2],
            "D": [0.75, 1.25],
        },
        # low friction biased: the evaluated weak spot is the slippery surface
        "ev_low": {
            "B": [9., 11.],
            "C": [1.8, 2.05],
            "D": [0.60, 1.05],
        },
    }[args.randomize_tyre],
    "random_seed": args.seed,
    "quiet": args.quiet,
    "aux_reward_decay_steps": args.aux_reward_decay_steps,
    "aux_reward_coef": args.aux_reward_coef,
    "gamma": args.gamma,
    "train": (args.train_or_test == "train"),
}

# --grip-range narrows or widens the friction spread of the selected tyre preset
# so a curriculum can ramp D without editing the table above
if args.grip_range is not None:
    _g = [float(x) for x in args.grip_range.split(",")]
    if len(_g) != 2 or not (0.0 < _g[0] <= _g[1]):
        parser.error("--grip-range must be LO,HI with 0 < LO <= HI")
    # randomize_param is already the SELECTED preset dict, not the whole table
    default_env_config["randomize_param"] = dict(default_env_config["randomize_param"])
    default_env_config["randomize_param"]["D"] = _g

# Environment-specific config
if args.env == "state_tracker_iwd":
    # Reference generation mode
    if args.train_or_test == "train":
        default_env_config["ref_mode"] = "hybrid"
    else:
        env_variant_to_ref_mode = {
            "hybrid": "hybrid",
            "fixed": "fixed_circle_ccw",
            "fixed_cw": "fixed_circle_cw",
            "eight": "eight_drift",
        }
        if args.env_variant in env_variant_to_ref_mode:
            default_env_config["ref_mode"] = env_variant_to_ref_mode[args.env_variant]
        else:
            raise ValueError(f"Unknown env variant {args.env_variant}")

blacklist_keys = lambda d, blacklist: {k: d[k] for k in d if not (k in blacklist)}
vecenv.register('RLGPU',
                lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'vecenv_type': 'RLGPU',
    'env_creator': lambda **env_config: envs[args.env](
        **blacklist_keys(default_env_config, env_config.keys()),
        **env_config,
    ),
})


if args.sigma_per_dim:
    _std = [float(x) for x in args.sigma_per_dim.split(",")]
    if len(_std) != 3:
        parser.error("--sigma-per-dim requires steer,throttle,brake")


    class PerDimSigmaObserver(RLGPUAlgoObserver):
        """Set the initial action std per dimension.

        rl_games keeps one log-std vector on the actor (a2c_network.sigma), so the
        per-dimension values must be written there after the actor is built.  The
        runner restores the checkpoint afterwards, which overwrites them on resume.
        """

        def __init__(self, std, path_refresh_every=8):
            super().__init__(path_refresh_every=path_refresh_every)
            self.std = std

        def after_init(self, algo):
            super().after_init(algo)
            net = algo.model.a2c_network
            if not hasattr(net, "sigma"):
                raise RuntimeError("actor has no a2c_network.sigma")
            std = torch.as_tensor(self.std, dtype=net.sigma.dtype, device=net.sigma.device)
            if std.shape != net.sigma.shape:
                raise ValueError("sigma shape %s != %s" % (tuple(std.shape), tuple(net.sigma.shape)))
            if torch.any(std <= 0):
                raise ValueError("all action std must be positive")
            with torch.no_grad():
                net.sigma.copy_(std.log())
            print("per-dim action std set to", self.std)
else:
    PerDimSigmaObserver = None

observer = (PerDimSigmaObserver(_std, path_refresh_every=args.path_refresh_every)
            if args.sigma_per_dim
            else RLGPUAlgoObserver(path_refresh_every=args.path_refresh_every))
runner = Runner(observer)
file_path = os.path.dirname(__file__)
with open(os.path.join(file_path, "runner_config.yaml")) as f:
    runner_config = yaml.safe_load(f)
full_experiment_name = args.env + "_" + args.exp_name
# persist the resolved environment configuration next to the run so that
# evaluation can reconstruct the exact MDP instead of guessing defaults
if args.train_or_test == "train":
    # ref_mode is already passed positionally to the env constructor, so it is
    # recorded under a separate key that only the evaluator reads
    default_env_config["ref_mode_cli"] = args.ref_mode
    import json
    _cfg_dir = os.path.join("runs", full_experiment_name)
    os.makedirs(_cfg_dir, exist_ok=True)
    with open(os.path.join(_cfg_dir, "env_config.json"), "w") as _f:
        json.dump(default_env_config, _f, indent=2, default=str)
    with open(os.path.join(_cfg_dir, "train_args.json"), "w") as _f:
        json.dump(vars(args), _f, indent=2, default=str)

runner_config["params"]["seed"] = args.seed
runner_config["params"]["config"]["num_actors"] = args.num_parallel
runner_config["params"]["config"]["max_epochs"] = args.epochs
# the CLI knob must actually reach PPO, otherwise --seq-length is inert
runner_config["params"]["config"]["seq_length"] = args.seq_length

    # Per-dimension initial action std.  rl_games uses one scalar sigma for all
    # action dims, but the ranges are +-0.45 rad for steering and [0, 1] for each
    # pedal, so a std of 1.0 is 220 % of the steering range and 100 % of a pedal.
    # Patching the model right after construction gives each dim its own value.
batch = args.num_parallel * args.horizon
runner_config["params"]["config"]["minibatch_size"] = max(
    min(args.minibatch, batch), min(batch, 1024))
runner_config["params"]["config"]["games_to_track"] = args.num_parallel
runner_config["params"]["config"]["mini_epochs"] = args.mini_epochs
runner_config["params"]["config"]["lr_schedule"] = args.lr_schedule
runner_config["params"]["config"]["gamma"] = args.gamma
runner_config["params"]["config"]["horizon_length"] = args.horizon
runner_config["params"]["config"]["score_to_win"] = args.score_to_win
runner_config["params"]["config"]["name"] = args.env
runner_config["params"]["config"]["full_experiment_name"] = full_experiment_name
runner_config["params"]["network"]["mlp"]["units"] = [args.mlp_size_last * i for i in (4, 2, 1)]
runner_config["params"]["config"]["save_frequency"] = args.save_freq
runner_config["params"]["config"]["entropy_coef"] = args.entropy_coef
if args.asymmetric:
    # Privileged critic: its own MLP over the drift related + absolute state
    # channels, no shared backbone with the GRU actor.
    runner_config["params"]["config"]["central_value_config"] = {
        "minibatch_size": runner_config["params"]["config"]["minibatch_size"],
        "mini_epochs": args.mini_epochs,
        "learning_rate": 3.e-4,
        "lr_schedule": None,
        "clip_value": True,
        "normalize_input": True,
        "normalize_value": True,
        "model": {"name": "central_value"},
        "network": {
            "name": "actor_critic",
            "central_value": True,
            "mlp": {
                "units": [int(u) for u in args.critic_units.split(",")],
                "activation": "elu",
                "initializer": {"name": "default"},
                "regularizer": {"name": "None"},
            },
            "space": runner_config["params"]["network"]["space"],
        },
    }
runner_config["params"]["network"]["space"]["continuous"]["sigma_init"]["val"] = args.sigma_init
# rl_games uses `device` for the training algorithm. `device_name` is used by
# the player, so keep both values aligned with the environment device.
runner_config["params"]["config"]["device"] = args.device
runner_config["params"]["config"]["device_name"] = args.device
if not args.rnn:
    runner_config["params"]["network"].pop("rnn")

if args.quiet:
    with suppress_stdout_stderr():
        runner.load(runner_config)
else:
    runner.load(runner_config)

if __name__ == "__main__":
    if args.train_or_test == "train":
        run_args = {'train': True}
        if args.resume is not None:
            run_args['checkpoint'] = args.resume
        runner.run(run_args)
    elif args.train_or_test == "test":
        checkpoint_dir = f"runs/{full_experiment_name}/nn"
        if args.epoch_index == -1:
            checkpoint_name = f"{checkpoint_dir}/{args.env}.pth"
        else:
            list_of_files = glob.glob(f"{checkpoint_dir}/last_{args.env}_ep_{args.epoch_index}_rew_*.pth")
            checkpoint_name = max(list_of_files, key=os.path.getctime)
        runner.run({
            'train': False,
            'play': True,
            'checkpoint' : checkpoint_name,
        })
