import os
import sys
import gym
import torch
import time
import math
import csv
from pathlib import Path
import numpy as np
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../xcar-simulation"))
sys.path.append(os.path.join(file_path, "../utils"))
from gpu_vectorized_car_env import GPUVectorizedCarEnv
from generate_segment_racetrack import TrajGen


class ContinuousDriftIWDEnv(GPUVectorizedCarEnv):
    """Continuous drifting on random / fixed reference trajectories.

    ``control_mode="pedals"`` (default with the FastPlant): the policy commands
    ``[steer, throttle, brake]`` at ``dt`` (20 Hz by default) while the plant is
    integrated at ``physics_dt`` (100 Hz).

    ``control_mode="wheel_speed"``: the legacy XCAR action
    ``[steer, v_fl, v_fr, v_rl, v_rr]`` is kept so the same task can be trained
    with the old command interface (the FastPlant then runs a wheel speed
    servo instead of the motor + LSD).
    """

    def __init__(self, preset_name, n, device, ref_mode="hybrid", initial_state=None,
                 plant="fast", control_mode="pedals", **kwargs):
        super().__init__(preset_name, n, device=device, plant=plant,
                         control_mode=control_mode, drivetrain="iwd",
                         initial_state=initial_state, **kwargs)

        # A continuing task has no success/time-limit terminal.  PPO's horizon
        # remains a rollout boundary and therefore bootstraps the value without
        # resetting plant state or recurrent state.  Only physical/task failures
        # emit done.  Open random paths require a streaming reference and are
        # rejected below until one is available; clamping at the last waypoint
        # would silently turn the task into straight-line holding.
        self.continuing_task = bool(kwargs.get("continuing_task", False))
        self.critic_velocity_frame = kwargs.get("critic_velocity_frame", "world")
        if self.critic_velocity_frame not in ("world", "body"):
            raise ValueError("critic_velocity_frame must be world or body")
        self._cyclic_reference = ref_mode in ("circle", "eight", "closed_random")
        self._streaming_reference = ref_mode == "streaming_random"
        if self._streaming_reference and kwargs.get('preview_time') is None:
            raise ValueError('streaming_random requires explicit time preview headways')
        self.closed_curvature_cap = kwargs.get("closed_curvature_cap", None)
        if self.closed_curvature_cap is not None:
            self.closed_curvature_cap = float(self.closed_curvature_cap)
            if self.closed_curvature_cap <= 0:
                raise ValueError("closed_curvature_cap must be positive (1/m)")
        if self.continuing_task and ref_mode == "hybrid":
            raise ValueError(
                "continuing_task with hybrid references requires seamless path streaming; "
                "use circle/eight for the continuing curriculum stage")

        # env parameters
        self.max_steps = int(round(30.0 / self.dt))
        self.quiet = kwargs.get("quiet", False)
        self.smooth_weight = float(kwargs.get("smooth_weight", 0.015))
        self.smooth_scale = 0.01 / self.dt   # keep reward per unit time constant
        self.reward_mode = str(kwargs.get("reward_mode", "legacy"))
        if self.reward_mode not in ("legacy", "bounded_joint", "adaptive_envelope"):
            raise ValueError("reward_mode must be legacy, bounded_joint or adaptive_envelope")
        self.bounded_alive = float(kwargs.get("bounded_alive", 1.0))
        self.bounded_tracking = float(kwargs.get("bounded_tracking", 0.75))
        self.bounded_drift = float(kwargs.get("bounded_drift", 1.0))
        self.bounded_joint = float(kwargs.get("bounded_joint", 2.0))
        self.bounded_smooth = float(kwargs.get("bounded_smooth", 0.10))
        # Tracking/performance is the task. Powered drift is an optional bonus
        # only when it coexists with both, so a high-grip normal limit corner is
        # valid instead of being forced into a fake large-angle slide.
        self.adaptive_alive = float(kwargs.get("adaptive_alive", 0.10))
        self.adaptive_tracking = float(kwargs.get("adaptive_tracking", 1.00))
        self.adaptive_performance = float(kwargs.get("adaptive_performance", 3.00))
        self.adaptive_drift = float(kwargs.get("adaptive_drift", 1.00))
        self.adaptive_smooth = float(kwargs.get("adaptive_smooth", 0.10))
        self.adaptive_guidance = float(kwargs.get("adaptive_guidance", 0.0))
        # Opt-in: retain old checkpoint reward semantics when absent.
        self.adaptive_speed_penalty = str(kwargs.get("adaptive_speed_penalty", "target"))
        if self.adaptive_speed_penalty not in ("target", "envelope", "none"):
            raise ValueError('adaptive_speed_penalty must be target, envelope or none')
        self.adaptive_speed_coupling = str(kwargs.get("adaptive_speed_coupling", "precision"))
        if self.adaptive_speed_coupling not in ("precision", "corridor", "independent_position"):
            raise ValueError("adaptive_speed_coupling must be precision, corridor or independent_position")
        self.adaptive_path_scale_m = kwargs.get("adaptive_path_scale_m", None)
        if self.adaptive_path_scale_m is not None:
            self.adaptive_path_scale_m = float(self.adaptive_path_scale_m)
            if not math.isfinite(self.adaptive_path_scale_m) or self.adaptive_path_scale_m <= 0:
                raise ValueError('adaptive_path_scale_m must be finite and positive')

        self.adaptive_shape = str(kwargs.get("adaptive_shape", "legacy"))
        if self.adaptive_shape not in ("legacy", "smooth_v1", "smooth_task_v2"):
            raise ValueError("adaptive_shape must be legacy, smooth_v1, or smooth_task_v2")

        self.pedals = self.control_mode == "pedals"
        # observed: yaw rate + 3 accelerometer axes + previous commands (3/4)
        # + wheel speeds (4) + lateral/course/curvature errors and progress (4)
        # + 3 preview points x 5.  The plant state (10) is a simulation detail and
        # is not observed.

        # The reference geometry of this task was authored for the 0.35 m XCAR.
        # A full-size car cannot follow a 1 m radius curve (minimum steering
        # radius is L/tan(max_steer) ~ 5.4 m), so the whole reference is scaled
        # with the wheelbase ratio.  Everything metric is expressed in XCAR
        # units and multiplied by ``length_scale`` on the way in, everything
        # with the unit of time by ``speed_scale = sqrt(length_scale)``.
        wheelbase = float(self.preset["lF"] + self.preset["lR"])
        self.wheelbase = wheelbase
        self.length_scale = wheelbase / 0.35
        self.speed_scale = math.sqrt(self.length_scale)
        self.speed_ref = 10.0 * self.speed_scale
        # "no progress" (0.2 s for the XCAR) stretched with the vehicle time scale
        self.no_progress_limit = int(kwargs.get(
            "stall_window", max(int(round(0.2 * self.speed_scale / self.dt)), 2)))
        self.stall_window = self.no_progress_limit
        # Reference sideslip magnitude.  XCAR asks for +-1 rad (57 deg); with a
        # 26 deg steering limit a 57 deg sustained drift needs ~46 deg of
        # countersteer, which this car does not have, so the target is scaled to
        # what the vehicle can hold (--beta-scale 1.0 restores the XCAR value).
        self.beta_scale = float(kwargs.get("beta_scale", 1.0))
        # reference sideslip: "constant" keeps the old |beta_ref| = beta_scale,
        # "physics" derives it from the friction utilisation at each point
        self.beta_ref_mode = str(kwargs.get("beta_ref_mode", "physics"))
        self.beta_max = float(kwargs.get("beta_max", 0.32))
        self.alpha_peak = float(kwargs.get("alpha_peak", 0.12))
        # the geometric sideslip target is only demanded when the corner actually
        # uses this fraction of the available grip
        self.beta_util_ref = float(kwargs.get("beta_util_ref", 0.45))
        self.beta_util_lo = float(kwargs.get("beta_util_lo", 0.30))
        # Adaptive drift reference.  These are reward/critic-only quantities;
        # true mu and true speed never enter the 27-channel actor observation.
        self.drift_util_lo = float(kwargs.get("drift_util_lo", 0.55))
        self.drift_util_hi = float(kwargs.get("drift_util_hi", 0.90))
        self.drift_curve_lo = float(kwargs.get("drift_curve_lo", 0.02))
        self.drift_curve_hi = float(kwargs.get("drift_curve_hi", 0.06))
        self.drift_curve_fade_lo = float(kwargs.get("drift_curve_fade_lo", 0.10))
        self.drift_curve_fade_hi = float(kwargs.get("drift_curve_fade_hi", 0.15))
        self.drift_mu_lo = float(kwargs.get("drift_mu_lo", 1.0623))
        self.drift_mu_hi = float(kwargs.get("drift_mu_hi", 2.1246))
        # reference-sideslip map: beta_ref = beta_max * clamp(v_limit / v_char, lo, 1)
        # Ackermann blend range (0 = parallel steering, 1 = exact geometry)
        self.ackermann_lo = float(kwargs.get("ackermann_lo", 0.8))
        self.ackermann_hi = float(kwargs.get("ackermann_hi", 1.2))
        self.beta_weight = float(kwargs.get("beta_weight", 1.6))
        self.pos_weight = float(kwargs.get("pos_weight", 2.4))
        self.dir_weight = float(kwargs.get("dir_weight", 0.5))
        # yaw rate tracking: the under-turning failure mode.  r is an IMU
        # measurement and ``r_ref = v_est * kappa_ref`` is computable on the car
        # (wheel speed estimate x previewed curvature), so this is a legal,
        # dense signal for "keep rotating at the rate the path needs".
        # Every task term except progress is a penalty, so with an all negative
        # per step reward "die quickly" is a local optimum (observed: the policy
        # learned to stop, which ends the episode).  A constant alive bonus makes
        # surviving strictly better than terminating.
        self.alive_bonus = float(kwargs.get("alive_bonus", 0.0))
        self.terminal_cost = float(kwargs.get("terminal_cost", 100.0))
        self.yaw_weight = float(kwargs.get("yaw_weight", 0.0))
        self.yaw_scale = float(kwargs.get("yaw_scale", 0.5))
        self.progress_gate = 0.5 * self.length_scale
        self.stall_speed = float(kwargs.get("stall_speed", 2.0))
        # Failure thresholds as multiples of the vehicle length scale.  Survival
        # is the first objective, so the corridor and the course limits are
        # configurable and can be tightened later (curriculum).
        self.offpath_limit = float(kwargs.get("offpath_scale", 1.0)) * self.length_scale
        self.pos_cap = float(kwargs.get("pos_cap", 1.0))
        self.yaw_cap = float(kwargs.get("yaw_cap", 4.0))
        self.kappa_cap = float(kwargs.get("kappa_cap", 4.0))
        self.course_limit = float(kwargs.get("course_fail", 1.0))
        # 0 disables the gate: the reward then targets beta_ref exactly
        self.beta_gate_scale = float(kwargs.get("beta_gate_scale", 0.0))
        self.recover_weight = float(kwargs.get("recover_weight", 0.0))
        self.kappa_weight = float(kwargs.get("kappa_weight", 0.15))
        self.pp_weight = float(kwargs.get("pp_weight", 0.0))
        self.prog_weight = float(kwargs.get("prog_weight", 0.2))
        # --- DRIFT terms (added after the owner reported that the car was only crawling
        # along the path).  Measured on the previous best policy: throttle 0.000-0.037,
        # rear wheels SLOWER than the front, rear longitudinal slip NEGATIVE (-0.09 to
        # -0.14), i.e. the direction of engine braking.  No existing term asked the rear
        # axle to spin, so coasting was optimal.
        self.drift_kappa_target = float(kwargs.get("drift_kappa_target", 0.55))
        # Curriculum only: 1 seeds the rear wheel at the target slip to teach
        # unstable drift maintenance; 0 is the physical final task and never
        # pre-spins a wheel.  Evaluation/CarSim must always use 0.
        self.drift_spawn_spin_frac = float(kwargs.get("drift_spawn_spin_frac", 1.0))
        self.drift_spawn_trim_frac = float(kwargs.get("drift_spawn_trim_frac", 1.0))
        self.drift_kappa_weight = float(kwargs.get("drift_kappa_weight", 60.0))
        self.drift_throttle_gate = float(kwargs.get("drift_throttle_gate", 0.30))
        self.drift_throttle_weight = float(kwargs.get("drift_throttle_weight", 4.0))
        self.drift_counter_weight = float(kwargs.get("drift_counter_weight", 8.0))
        # MEASURED: 40 made it WORSE -- the rear slip fell from 0.2-1.2 to 0.06-0.29 and the
        # throttle from 0.23-0.28 to 0.15-0.24, because the joint gate then demanded
        # |beta| 0.18-0.36 while the policy could hold 0.15, so it gave up on the gate
        # entirely and minimised the slip penalty instead.  12 is the measured optimum
        # so far and is restored.
        self.drift_joint_weight = float(kwargs.get("drift_joint_weight", 12.0))
        # Standalone "hold the pedal open" reward, DEFAULT 0 so nothing changes unless it
        # is explicitly requested.  The v56 configuration survives a real 30 s in CarSim
        # with a half-metre line and smooth steering but its throttle mean is 0.006, i.e.
        # it coasts; this is the single increment to add on top of it, and the weight is
        # deliberately small against the pos_weight 8 and beta_weight 100 already there.
        self.throttle_weight = float(kwargs.get("throttle_weight", 0.0))
        self.throttle_gate = float(kwargs.get("throttle_gate", 0.25))
        # The faded kinematic reference was correct for path tracking but it is the wrong
        # target for a drift: measured with the drift slip reward active, the car reached
        # throttle 0.18-0.29 and a positive rear slip of 0.03-0.36 (both good) while its
        # sideslip stayed at 0.003-0.017 rad, i.e. a fast forward slide.  A power slide needs
        # an explicit large sideslip, so when drift_mode is on the reference becomes a
        # constant magnitude of this many radians with the sign of the local curvature.
        self.drift_mode = bool(kwargs.get("drift_mode", False))
        # In drift mode the lateral error is only penalised BEYOND a deadband, so the car has
        # room to hold a drift attitude without being pushed toward zero sideslip by a
        # uniform quadratic.  A uniform penalty is what made every attempt trade attitude for
        # line, one for one.
        self.drift_corridor_deadband = float(kwargs.get("drift_corridor_deadband", 0.6))
        self.drift_beta_target = float(kwargs.get("drift_beta_target", 0.40))

        n_ctrl = 3 if self.pedals else 4
        if not self.pedals:
            # the 27-channel actor contract assumes the pedal triple; the legacy
            # wheel-speed mode would make it 28 channels
            raise ValueError("continuous_drift_iwd uses the 27-channel pedals actor")
        # Production keeps the fixed 27-channel real-observable actor contract.
        # The optional four exact plant channels are an explicitly non-deployable
        # observability upper bound; sensor DR still acts only on physical sensors.
        self.sensor_num_states = (1 + 3 + n_ctrl + 4 + 4) + 12
        self.obs_privileged = bool(kwargs.get("obs_privileged", False))
        self.num_states = self.sensor_num_states + (4 if self.obs_privileged else 0)
        # training; rl_games adds its own running normalisation on top.
        wheel_scale = 3.0 / self.speed_ref
        ctrl_scale = ([1.0 / self.max_steer, 1.0, 1.0] if self.pedals
                      else [1.0 / self.max_steer] + [wheel_scale] * 4)
        scale = [0.5, 0.1, 0.1, 0.1]          # r, ax, ay, az
        scale += ctrl_scale
        scale += [wheel_scale] * 4
        scale += [1.0, 1.0, 1.0,
                  (1.0 / self.speed_ref if self.continuing_task else 0.05)]
        # e_pos, e_dir, e_kappa, encoder speed (continuing) or legacy odometry
        for _ in range(3):
            scale += [1.0 / self.length_scale, 1.0 / self.length_scale, 1.0, 1.0]
        self.obs_scale = torch.tensor(scale, device=self.device, dtype=torch.float32)
        # Actor-only sensor/fusion uncertainty.  Noise is specified after the
        # fixed physical normalization above; reward and asymmetric critic keep
        # true simulation quantities.  Repeated obs() calls return the cached
        # packet, so rendering/evaluation cannot advance delay or random walk.
        self.sensor_dr = bool(kwargs.get("sensor_dr", False))
        self.sensor_delay_max = max(int(kwargs.get("sensor_delay_max", 1)), 0)
        self.sensor_dropout_prob = float(kwargs.get("sensor_dropout_prob", 0.0))
        yaw_n = 0.005 * scale[0]
        acc_n = 0.10 * scale[1]
        steer_n = 0.002 * scale[4]
        wheel_n = 0.03 * wheel_scale
        pos_n = 0.05 / self.length_scale
        curv_n = 0.001 * self.length_scale
        white = ([yaw_n, acc_n, acc_n, acc_n, steer_n, 0.005, 0.005]
                 + [wheel_n] * 4
                 + [pos_n, 0.005, curv_n, 0.03 / self.speed_ref]
                 + [pos_n, pos_n, 0.005, curv_n] * 3)
        bias = ([0.010 * scale[0], 0.15 * scale[1], 0.15 * scale[1], 0.15 * scale[1],
                 0.003 * scale[4], 0.005, 0.005]
                + [0.02 * wheel_scale] * 4
                + [pos_n, 0.005, curv_n, 0.02 / self.speed_ref]
                + [pos_n, pos_n, 0.005, curv_n] * 3)
        self._sensor_white_std = torch.tensor(white, device=self.device)
        self._sensor_bias_std = torch.tensor(bias, device=self.device)
        self._sensor_rw_std = 0.001 * self._sensor_bias_std
        self._sensor_bias = torch.zeros((self.n, self.sensor_num_states), device=self.device)
        self._sensor_rw = torch.zeros_like(self._sensor_bias)
        self._sensor_noise_scale = torch.ones((self.n, 1), device=self.device)
        self._sensor_delay_steps = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self._sensor_history = torch.zeros(
            (self.sensor_delay_max + 1, self.n, self.sensor_num_states), device=self.device)
        self._sensor_cached = torch.zeros((self.n, self.sensor_num_states), device=self.device)
        self._sensor_pending = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self._sensor_needs_fill = torch.ones_like(self._sensor_pending)
        self.priv_concat_scale = torch.tensor([1.0, 0.25, 0.25, 0.25], device=self.device)
        # privileged critic channels: beta, v, r, vx, vy, kappa, kappa_ref, r_ref,
        # beta_ref, e_pos, e_dir, ax, ay, az, wheels, slip, fz, fy, commands
        pscale = ([1.0, 0.1, 0.5, 0.1, 0.1, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,
                   0.1, 0.1, 0.1]
                  + [wheel_scale] * 4 + [0.5] * 4 + [1.0] * 4 + [1.0] * 4 + [1.0] * 4
                  + list(ctrl_scale))
        for _ in range(3):
            pscale += [1.0 / self.length_scale, 1.0 / self.length_scale, 1.0, 1.0, 1.0]
        self.priv_scale = torch.tensor(pscale, device=self.device, dtype=torch.float32)
        # "equilibrium" spawns start near the drift equilibrium of the local
        # reference (see gen_random_state); "legacy" keeps the XCAR wide
        # randomization.  spawn_speed is the target drift speed on the reference.
        # Asymmetric actor critic: when enabled, reset/step return
        # {"obs": actor_obs, "states": privileged_state} and the critic is a
        # separate network that sees the drift / absolute state quantities the
        # actor cannot measure.
        self.asymmetric = bool(kwargs.get("asymmetric", False))
        self.spawn_mode = str(kwargs.get("spawn_mode", "equilibrium"))
        default_eq_prob = 1.0 if self.spawn_mode == "equilibrium" else 0.0
        self.equilibrium_reset_prob = float(
            kwargs.get("equilibrium_reset_prob", default_eq_prob))
        if not 0.0 <= self.equilibrium_reset_prob <= 1.0:
            raise ValueError("equilibrium_reset_prob must be in [0, 1]")
        # Kept explicitly for reset-distribution diagnostics and contract tests.
        # It is not part of either actor or critic observation.
        self._last_equilibrium_reset = torch.zeros(
            self.n, dtype=torch.bool, device=self.device)
        self.spawn_speed = float(kwargs.get("spawn_speed", 6.5))
        self.spawn_beta_frac = float(kwargs.get("spawn_beta_frac", 1.0))
        self.spawn_beta_noise = float(kwargs.get("spawn_beta_noise", 0.10))
        self.spawn_speed_noise = float(kwargs.get("spawn_speed_noise", 0.10))
        # Speed-deficit curriculum: a fraction of the spawns start below the
        # reference speed so the policy has to LEARN TO REGAIN it.  The dominant
        # failure at the moment is "4 m/s -> 1 m/s in a second and never back".
        self.spawn_beta_mag = float(kwargs.get("spawn_beta_mag", 0.15))
        self.legacy_zero_prob = float(kwargs.get("legacy_zero_prob", 0.0))
        if not 0.0 <= self.legacy_zero_prob <= 1.0:
            raise ValueError("legacy_zero_prob must be in [0, 1]")
        self.spawn_speed_min = float(kwargs.get("spawn_speed_min", self.spawn_speed))
        self.spawn_deficit_frac = float(kwargs.get("spawn_deficit_frac", 0.0))
        # Speed regulation around the drift operating speed.  XCAR's rspeed only
        # punishes V < 0.5 m/s, which at our scale (drift speed ~6.5 m/s) leaves
        # "slow down until the drift collapses" as a reward improving local
        # optimum; this term closes that hole.
        self.speed_target = float(kwargs.get("speed_target", self.spawn_speed))
        # "signal" = fixed target; "limit" = target derived from the local friction
        # limit, v_lim = sqrt(mu * g / |kappa_ref|), which is what makes the car
        # drive AT the limit of adhesion instead of at an arbitrary speed.
        self.speed_target_mode = str(kwargs.get("speed_target_mode", "signal"))
        self.speed_margin = float(kwargs.get("speed_margin", 0.9))
        if self.adaptive_speed_penalty == "envelope" and (
                self.speed_target_mode != "limit" or not 0.0 < self.speed_margin <= 1.0):
            raise ValueError('envelope speed penalty requires limit speed mode and 0 < speed_margin <= 1')
        # The limit target assumes the full mu can be used laterally, but the car
        # sustains only a light sideslip, so at high grip it becomes too
        # optimistic (measured: 9.3 m/s at mu 1.25 with 30 % full length).  The cap
        # keeps the target inside what the drift envelope allows.
        self.speed_cap = float(kwargs.get("speed_cap", float("inf")))
        self.speed_lo = float(kwargs.get("speed_lo", 3.5))
        self.speed_weight = float(kwargs.get("speed_weight", 0.0))
        self.speed_hinge = bool(kwargs.get("speed_hinge", False))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
        self.state_space = self.observation_space
        if self.asymmetric:
            self.num_priv = 52
            self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_priv,))

        dummy_obs = torch.zeros((self.n, self.num_states), device=self.device)
        self.recent_obs = [dummy_obs, dummy_obs]  # most recent first
        self.train = kwargs.get("train", False)
        self.wheel_radius = self.plant.geometry()[2]
        # Encoder conversion uses a known calibration, never the hidden
        # episode-specific rolling radius drawn by physics DR.
        self.encoder_radius = self.wheel_radius.clone()
        # >1 biases the friction draw towards the low end while keeping the range
        self.plant.grip_bias = float(kwargs.get("grip_bias", 1.0))

        # Only generated reference families may refresh their bank.  Keep this
        # separate from ``_is_hybrid``: hybrid is an OPEN road and has special
        # end-of-road spawn guards, while closed_random is cyclic but must still
        # receive fresh roads during training to prevent finite-bank memorisation.
        self._is_hybrid = False
        self._refreshable_reference = False
        self._path_generator_kind = None
        self.path_regen_every = int(kwargs.get("path_regen_every", 0))
        self._resets_since_regen = 0

        if ref_mode == "hybrid":
            self.num_waypoints = 10000
            path_length = 50
            self._path_length_xcar = path_length
            self.trajectory_generator = TrajGen(self.num_waypoints, path_length, device=device)
            cs = kwargs.get("curv_scale", 1.0)
            self.trajectory_generator.curv_scale = cs if isinstance(cs, (tuple, list)) else float(cs)
            self.trajectory_generator.seg_scale = float(kwargs.get("seg_scale", 1.0))
            # an unsmoothed generator contains instantaneous curvature sign
            # jumps, which is not a trackable road at 6-8 m/s
            self.trajectory_generator.curv_smooth = int(kwargs.get("curv_smooth", 200))
            self.trajectory_generator.flip_prob = kwargs.get("flip_prob", None)
            self.num_paths = int(kwargs.get("num_paths", 256))
            self.path_bank = self._scale_waypoints(
                self.trajectory_generator.generate_random_waypoints(self.num_paths))
            # the bank is regenerated periodically: 256 fixed roads shared by
            # 131072 environments would be memorised, so fresh roads are drawn
            # whenever enough episodes have been spawned (see reset_done_envs)
            self._is_hybrid = True
            self._refreshable_reference = True
            self._path_generator_kind = "open_random"
        elif self._streaming_reference:
            from utils.random_reference import RollingReference
            initial = torch.zeros((self.n, 4), device=self.device)
            initial[:, 3] = (torch.rand(self.n, device=self.device)*2-1)*.09
            self._rolling_reference = RollingReference(
                initial, segments=int(kwargs.get("reference_segments", 2)),
                transition_min_m=kwargs.get("reference_transition_min_m", None))
            self.path_bank = self._rolling_reference.roads
            self.num_waypoints = self.path_bank.shape[1]
            self._path_length_xcar = 240. / self.length_scale
        elif ref_mode == "closed_random":
            self.num_waypoints = 2512
            self.closed_random_radius = float(kwargs.get("closed_random_radius", 1.0))
            if self.closed_random_radius <= 0.0:
                raise ValueError("closed_random_radius must be positive")
            self._path_length_xcar = 2.0 * math.pi * self.closed_random_radius
            self.num_paths = int(kwargs.get("num_paths", 256))
            self.trajectory_generator = TrajGen(
                self.num_waypoints, self._path_length_xcar, device=device)
            self.trajectory_generator.switch_fraction = float(kwargs.get("closed_switch_fraction", 0.0))
            self.path_bank = self._scale_waypoints(
                self.trajectory_generator.generate_closed_random_waypoints(
                    self.num_paths, radius=self.closed_random_radius))
            self._refreshable_reference = True
            self._path_generator_kind = "closed_random"
        # circle trajectory
        elif ref_mode == "circle":
            self.num_waypoints = 2512      # one waypoint per 0.005m
            self._path_length_xcar = 4.0 * math.pi
            radius = 1.
            waypoint_x = radius * torch.sin(torch.arange(self.num_waypoints) / self.num_waypoints * 4 * np.pi)
            waypoint_y = radius - radius * torch.cos(torch.arange(self.num_waypoints) / self.num_waypoints * 4 * np.pi)
            waypoint_veldir = torch.arange(self.num_waypoints) / self.num_waypoints * 4 * np.pi
            waypoint_veldir = torch.atan2(torch.sin(waypoint_veldir), torch.cos(waypoint_veldir))
            waypoint_kappa = torch.ones((self.num_waypoints, )) / radius
            self.path_bank = self._scale_waypoints(torch.stack((waypoint_x, waypoint_y, waypoint_veldir, waypoint_kappa), dim=1).to(self.device)).unsqueeze(0)
        # eight-shaped trajectory
        elif ref_mode == "eight":
            self.num_waypoints = 2512      # one waypoint per 0.005m
            self._path_length_xcar = 4.0 * math.pi
            radius = 1.
            waypoint_x = radius * torch.sin(torch.arange(self.num_waypoints) / self.num_waypoints * 4 * np.pi)
            waypoint_y = torch.cat((
                radius - radius * torch.cos(torch.arange(self.num_waypoints // 2) / self.num_waypoints * 4 * np.pi),
                -radius + radius * torch.cos(torch.arange(self.num_waypoints // 2) / self.num_waypoints * 4 * np.pi),
            ))
            waypoint_veldir = torch.cat((
                torch.arange(self.num_waypoints // 2) / self.num_waypoints * 2,
                self.num_waypoints // 2 - torch.arange(self.num_waypoints // 2) / self.num_waypoints * 2,
            )) * 2 * np.pi
            waypoint_veldir = torch.atan2(torch.sin(waypoint_veldir), torch.cos(waypoint_veldir))

            waypoint_kappa = torch.cat((
                torch.ones((self.num_waypoints // 2, )) / radius,
                -torch.ones((self.num_waypoints // 2, )) / radius,
            ))
            self.path_bank = self._scale_waypoints(torch.stack((waypoint_x, waypoint_y, waypoint_veldir, waypoint_kappa), dim=1).to(self.device)).unsqueeze(0)
        elif ref_mode == "three":
            path_length = 12 * torch.pi
            self.num_waypoints = int(path_length / 0.005)

            traj_gen = TrajGen(num_waypoints=self.num_waypoints, path_length=path_length, device=self.device)

            segment_length = torch.tensor([(1 + 1 / 12) * 2 * torch.pi, (1 + 5 / 6) * 2 * torch.pi, (1 + 1 / 6) * 2 * torch.pi, (1 + 11 / 12) * 2 * torch.pi], device=self.device)

            curvature = torch.tensor([1, -1, 1, -1], device=self.device)

            self.path_bank = self._scale_waypoints(traj_gen.generate_custom_waypoints(segment_length, curvature))
        elif ref_mode == "olympic":
            path_length = 18 * torch.pi
            self.num_waypoints = int(path_length / 0.005)

            traj_gen = TrajGen(num_waypoints=self.num_waypoints, path_length=path_length, device=self.device)

            segment_length = torch.tensor([(1 + 1 / 4) * 2 * torch.pi, (1 + 5 / 6) * 2 * torch.pi, (1 + 1 / 3) * 2 * torch.pi, (1 + 2 / 3) * 2 * torch.pi, (1 + 1 / 6) * 2 * torch.pi, (1 + 3 / 4) * 2 * torch.pi], device=self.device)

            curvature = torch.tensor([1, -1, 1, -1, 1, -1], device=self.device)

            self.path_bank = self._scale_waypoints(traj_gen.generate_custom_waypoints(segment_length, curvature))
        elif ref_mode == "variable_curvature":
            circle_length_small_start = 4.5 * torch.pi
            circle_length_small_mid = 5 * torch.pi
            circle_length_small_end = 4.5 * torch.pi
            sin_length = math.asin(1 / 2) * 8
            circle_length_large = (torch.pi - (math.asin(1 / 2) + math.sqrt(3) / 2 - 1) * 8) * 2 * 2

            path_length = circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large + sin_length + circle_length_small_end
            self.num_waypoints = int(path_length / 0.005)

            traj_gen = TrajGen(num_waypoints=self.num_waypoints, path_length=path_length, device=self.device)
            self.path_bank = self._scale_waypoints(traj_gen.generate_variable_curvature_waypoints())
        else:
            raise ValueError("Invalid ref_mode")

        # One shared reference bank + a per environment path index.  Storing a
        # (n, W, 5) reference per environment costs gigabytes at large n, and
        # regenerating a path at every reset launched thousands of tiny kernels.
        # A bank of ``num_paths`` paths is a few tens of MB and a reset becomes
        # a single ``randint``.
        if self.path_bank.ndim != 3 or self.path_bank.shape[-1] != 4:
            raise ValueError("path generator must return [num_paths, W, 4], got %s"
                             % (tuple(self.path_bank.shape),))
        self.num_paths = int(self.path_bank.shape[0])
        # three-generation ring: an environment keeps its generation for the whole
        # episode, and a refresh only overwrites a slot that no episode is using
        self.path_bank = (self.path_bank.unsqueeze(0) if self._streaming_reference
                          else self.path_bank.unsqueeze(0).repeat(3, 1, 1, 1))
        self._path_ds = torch.linalg.vector_norm(
            torch.roll(self.path_bank[..., :2], -1, dims=2) - self.path_bank[..., :2],
            dim=-1).mean(dim=-1)
        self.current_generation = 0
        self.path_generation = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self._path_refresh_pending = False
        self.path_id = torch.zeros(self.n, dtype=torch.long, device=self.device)
        if self._streaming_reference:
            self.path_id = torch.arange(self.n, device=self.device)
            self._path_ds.fill_(self._rolling_reference.ds)
        elif self.num_paths > 1:
            self.path_id.random_(0, self.num_paths)

        # Evaluation used to create ``data/traj-*.csv`` in the process working
        # directory unconditionally.  That silently polluted the source tree
        # during read-only gates.  Export is now explicit and caller-owned.
        trajectory_output_dir = kwargs.get("trajectory_output_dir")
        if not self.train and trajectory_output_dir:
            basename = time.strftime("%Y%m%d-%H%M%S")
            Path(trajectory_output_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(trajectory_output_dir,
                                   "traj-" + basename + ".csv"), 'w') as f:
                writer = csv.writer(f, delimiter=',')
                for i in range(self.num_waypoints):
                    x, y, psi = (self.path_bank[0, 0, i, 0].item(),
                                 self.path_bank[0, 0, i, 1].item(),
                                 self.path_bank[0, 0, i, 2].item())
                    writer.writerow([x, y, psi])

        self.progress = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.step_progress = torch.zeros(self.n, dtype=torch.int16, device=self.device)
        self.travel = torch.zeros(self.n, device=self.device)
        self.win_back = 20
        self.search_offsets = torch.arange(-self.win_back, 20, device=self.device)
        self.stall_move_min = float(kwargs.get("stall_move", 1.0))
        self._stall_anchor = torch.zeros((self.n, 2), device=self.device)
        self.preview_scale = float(kwargs.get("preview_scale", 1.0))
        # time-based preview headways (seconds); None keeps the metric previews
        pt = kwargs.get("preview_time", None)
        self.preview_time = None if pt is None else torch.as_tensor(
            [float(x) for x in (pt.split(',') if isinstance(pt, str) else pt)],
            device=self.device, dtype=torch.float32)
        if self.preview_time is not None:
            values = self.preview_time
            if (values.shape != (3,) or not bool(torch.isfinite(values).all())
                    or not bool((values > 0).all()) or not bool((values.diff() > 0).all())):
                raise ValueError('preview_time requires three increasing positive finite headways')
        pd = kwargs.get("preview_distances_m", None)
        self.preview_distances_m = None
        if pd is not None:
            if pt is not None:
                raise ValueError('Choose metric or time preview, not both')
            self.preview_distances_m = torch.as_tensor(
                [float(x) for x in (pd.split(',') if isinstance(pd, str) else pd)],
                device=self.device, dtype=torch.float32)
            values = self.preview_distances_m
            if (values.shape != (3,) or not bool(torch.isfinite(values).all())
                    or not bool((values > 0).all()) or not bool((values.diff() > 0).all())):
                raise ValueError('preview_distances_m requires three increasing positive finite distances')
        # arc length of one waypoint, needed to turn a time headway into an index
        self.waypoint_ds = float(getattr(self, "_path_length_xcar", 50.0) * self.length_scale / max(self.num_waypoints - 1, 1))

        # The hybrid reference is an OPEN path.  Starting anywhere in the first
        # half left as little as 189 m of reference, while a 30 s episode at
        # 6-8 m/s covers 180-240 m: the car would run off the end, the progress
        # and previews clamp and it was then scored off_path.  That penalised
        # exactly the fast, high-grip episodes.
        self._max_spawn_progress = max(self.num_waypoints // 2 - 1, 0)
        if getattr(self, "_is_hybrid", False):
            try:
                _cap = float(self.speed_cap)
            except (TypeError, ValueError):
                _cap = float("inf")
            guard_speed = _cap if _cap == _cap and _cap != float("inf") else max(
                float(self.spawn_speed), float(self.speed_target))
            if getattr(self, "preview_time", None) is not None:
                preview_guard = guard_speed * float(self.preview_time.max().item())
            else:
                preview_guard = (float(self.preview_distances_m[-1])
                                 if self.preview_distances_m is not None
                                 else 600.0 * self.preview_scale * self.waypoint_ds)
            reserve = (1.15 * guard_speed * self.max_steps * self.dt
                       + preview_guard + 2.0 * self.offpath_limit)
            usable = max((self.num_waypoints - 1) * self.waypoint_ds - reserve, 0.0)
            self._max_spawn_progress = min(
                int(usable / self.waypoint_ds), self.num_waypoints - 1)
        self.is_done = torch.zeros(self.n, dtype=torch.uint8, device=self.device)   # 0 = not done, 1 = failed, 2 = succeeded
        self.no_progress_count = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        self._now = None
        self._prev = None

    def _scale_waypoints(self, wp):
        """Scale an XCAR-sized reference geometry to this vehicle's size."""
        out = wp.clone()
        if abs(self.length_scale - 1.0) > 1e-9:
            out[..., 0] *= self.length_scale
            out[..., 1] *= self.length_scale
            out[..., 3] /= self.length_scale
        if False:  # the path no longer carries a reference sideslip
            pass
        if self._cyclic_reference:
            if self.closed_curvature_cap is not None:
                # Enlarge the whole geometry, never clip only its curvature
                # labels. Heading and curvature must describe the same path.
                factor = (out[..., 3].abs().amax(dim=-1, keepdim=True)
                          / self.closed_curvature_cap).clamp_min(1.)
                out[..., :2] *= factor.unsqueeze(-1)
                out[..., 3] /= factor
            out = (TrajGen.resample_closed(out.unsqueeze(0))[0] if out.ndim == 2
                   else TrajGen.resample_closed(out))
        return out

    def update_recent_obs(self, obs):
        self.recent_obs[1] = self.recent_obs[0]
        self.recent_obs[0] = obs

    def _reset_sensor_model(self, mask):
        if not self.sensor_dr:
            return
        n = mask.numel() if mask.dtype != torch.bool else int(mask.sum().item())
        if n == 0:
            return
        self._sensor_noise_scale[mask] = 0.5 + torch.rand((n, 1), device=self.device)
        self._sensor_bias[mask] = (torch.randn((n, self.sensor_num_states), device=self.device)
                                   * self._sensor_bias_std)
        self._sensor_rw[mask] = 0.0
        self._sensor_delay_steps[mask] = torch.randint(
            0, self.sensor_delay_max + 1, (n,), device=self.device)
        self._sensor_pending[mask] = True
        self._sensor_needs_fill[mask] = True

    def _prepare_sensor_error(self):
        """Sample once so derived previews and encoder channels share a reading."""
        if not hasattr(self, '_sensor_error_prepared'):
            self._sensor_error_prepared = torch.zeros_like(self._sensor_pending)
            self._sensor_error = torch.zeros_like(self._sensor_cached)
        mask = self._sensor_pending & ~self._sensor_error_prepared
        if not bool(mask.any()):
            return self._sensor_error
        n = int(mask.sum().item())
        scale = self._sensor_noise_scale[mask]
        self._sensor_rw[mask] += (torch.randn((n, self.sensor_num_states), device=self.device)
                                  * self._sensor_rw_std * scale)
        limit = 3.0 * self._sensor_bias_std
        self._sensor_rw[mask] = torch.maximum(torch.minimum(self._sensor_rw[mask], limit), -limit)
        self._sensor_error[mask] = (self._sensor_bias[mask] + self._sensor_rw[mask]
            + torch.randn_like(self._sensor_cached[mask]) * self._sensor_white_std * scale)
        self._sensor_error_prepared[mask] = True
        return self._sensor_error

    def _actor_sensor_packet(self, clean):
        """Return one delayed/noisy packet while preserving the 27-D contract."""
        if not self.sensor_dr:
            return clean
        # CarSim exposes its native state as float64 while the deployable actor,
        # sensor biases and delay queue are float32.  Letting the first addition
        # promote ``sample`` to float64 made reset fail when it was assigned to
        # the float32 history buffer.  The real sensor packet contract is
        # float32, independent of the plant solver precision.
        clean = clean.to(dtype=self._sensor_history.dtype)
        mask = self._sensor_pending
        if not bool(mask.any()):
            return self._sensor_cached
        error = self._prepare_sensor_error()
        sample = clean[mask] + error[mask]
        if self.sensor_delay_max > 0:
            self._sensor_history[1:, mask] = self._sensor_history[:-1, mask].clone()
        self._sensor_history[0, mask] = sample
        fill = mask & self._sensor_needs_fill
        if bool(fill.any()):
            self._sensor_history[:, fill] = self._sensor_history[0, fill].unsqueeze(0)
            self._sensor_needs_fill[fill] = False
        env_i = torch.arange(self.n, device=self.device)
        delayed = self._sensor_history[self._sensor_delay_steps, env_i]
        dropout = mask & ~fill & (torch.rand(self.n, device=self.device) < self.sensor_dropout_prob)
        delivered = torch.where(dropout.unsqueeze(1), self._sensor_cached, delayed)
        self._sensor_cached[mask] = delivered[mask]
        self._sensor_error_prepared[mask] = False
        self._sensor_pending[mask] = False
        return self._sensor_cached

    def waypoints_at(self, ind):
        """Reference waypoints: (n, k, 4) from each environment's own generation."""
        return self.path_bank[
            self.path_generation.unsqueeze(1), self.path_id.unsqueeze(1), ind
        ]

    def _generate_fresh_path_bank(self):
        """Draw one bank matching the current generated reference family."""
        if self._path_generator_kind == "open_random":
            fresh = self.trajectory_generator.generate_random_waypoints(self.num_paths)
        elif self._path_generator_kind == "closed_random":
            fresh = self.trajectory_generator.generate_closed_random_waypoints(
                self.num_paths, radius=self.closed_random_radius)
        else:
            raise RuntimeError("reference family is not refreshable")
        return self._scale_waypoints(fresh)

    def _try_path_refresh(self, releasing=None):
        """Install a pending generation when its target slot is unused.

        ``releasing`` marks environments whose episode ends in this reset call;
        they no longer count as users of their old generation.
        """
        if not self._refreshable_reference or not self._path_refresh_pending:
            return False
        target = (self.current_generation + 1) % 3
        in_use = self.path_generation == target
        if releasing is not None:
            in_use = in_use & ~releasing
        if bool(in_use.any()):
            return False
        fresh = self._generate_fresh_path_bank()
        if fresh.shape != self.path_bank[target].shape:
            raise RuntimeError("refreshed bank shape %s != %s"
                               % (tuple(fresh.shape), tuple(self.path_bank[target].shape)))
        self.path_bank[target].copy_(fresh)
        self._path_ds[target] = torch.linalg.vector_norm(
            torch.roll(fresh[..., :2], -1, dims=1) - fresh[..., :2], dim=-1).mean(dim=-1)
        self.current_generation = target
        self._path_refresh_pending = False
        return True

    def request_path_refresh(self):
        """Ask for one fresh road generation (the observer calls this per N epochs)."""
        if not self._refreshable_reference:
            return False
        self._path_refresh_pending = True
        return self._try_path_refresh()

    def find_nearest_waypoint(self, x, y, veldir, beta, kappa):
        if getattr(self, '_streaming_reference', False):
            self.progress, _ = self._rolling_reference.advance(
                self.progress, active=getattr(self, '_tracking_mask', None))
        # A partial respawn must not advance another car's tracking history.
        mask = getattr(self, "_tracking_mask", None)
        saved = {}
        if mask is not None:
            for name in ("progress", "step_progress", "_stall_anchor", "no_progress_count",
                         "angle_diff", "signed_pos_diff", "beta_diff", "beta_ref",
                         "kappa_diff", "kappa_ref"):
                value = getattr(self, name, None)
                if torch.is_tensor(value):
                    saved[name] = value.clone()
        # Predict index motion from localised displacement, not hidden velocity.
        # The correction window now covers projection error rather than imposing
        # a 19-waypoint/control-step speed limit.
        last = self.waypoints_at(self.progress[:, None])[:, 0]
        nxt_idx = self.progress + 1
        nxt_idx = (nxt_idx.remainder(self.num_waypoints) if self._cyclic_reference
                   else nxt_idx.clamp(max=self.num_waypoints - 1))
        nxt = self.waypoints_at(nxt_idx[:, None])[:, 0]
        ds = torch.linalg.vector_norm(nxt[:, :2] - last[:, :2], dim=1).clamp_min(1e-4)
        previous_xy = getattr(self, "_projection_xy", self.s[:, :2])
        displacement = self.s[:, :2] - previous_xy
        along = displacement[:, 0] * torch.cos(last[:, 2]) + displacement[:, 1] * torch.sin(last[:, 2])
        advance = (along / ds).round().long()
        if mask is not None:
            advance = torch.where(mask, advance, torch.zeros_like(advance))
        # Local projection around the predicted progress.  The window is
        # symmetric because a car that has been pushed off the line must be able
        # to re-acquire a reference point slightly BEHIND the last one; the old
        # forward-only window plus the "freeze progress while off path" rule made
        # the target stale exactly inside the recovery band (progress froze at
        # 3.8 m of lateral error while the failure corridor is 10.6 m).
        offs = self.search_offsets
        ind = (self.progress + advance).unsqueeze(1) + offs
        if self._cyclic_reference:
            ind = ind.remainder(self.num_waypoints)
        else:
            ind = ind.clamp_(0, self.num_waypoints - 1)
        waypoints_filtered = self.waypoints_at(ind)
        # four reference channels: [x, y, veldir, kappa]; the nearest point is
        # chosen on position only, so ``beta`` is not part of the comparison
        current = torch.stack((x, y, veldir, kappa), dim=1)
        diff = current.unsqueeze(1) - waypoints_filtered
        # wrap angle diff to (-pi, pi)
        angle_diff = torch.atan2(torch.sin(diff[:, :, 2]), torch.cos(diff[:, :, 2]))
        pos_diff = torch.hypot(diff[:, :, 0], diff[:, :, 1])
        # the path no longer carries a beta column: column 3 is the curvature
        beta_diff = torch.zeros_like(diff[:, :, 0])
        kappa_diff = diff[:, :, 3]
        best = torch.argmin(pos_diff, 1)
        bidx = best.unsqueeze(1)
        self.step_progress = (advance + best - self.win_back).to(torch.int16)
        self.projection_search_edge = (best == 0) | (best == offs.numel() - 1)
        self.progress = self.progress + self.step_progress.long()
        if self._cyclic_reference:
            self.progress.remainder_(self.num_waypoints)
        else:
            self.progress.clamp_(0, self.num_waypoints - 1)
        best_dist = pos_diff.gather(1, bidx).squeeze(1)
        # "stalled" means the car is actually stopped (or crawling).  Tying it to
        # the reference progress instead let a 0.7 m/s crawl survive forever,
        # while killing drifting cars that legitimately lose ground for a moment.
        # A car that is slow but accelerating (recovering from a slide) is not
        # stalled, so the longitudinal acceleration gates the counter.
        #
        # The rule must NOT be gated on ``on_path``: doing so made "be slow while
        # sitting outside the progress corridor" a safe harbour (measured: the
        # distilled policy was below stall speed 82 % of the time and off path
        # 70 % of the time, which is what its long episodes actually were).
        # The stall rule is purely about displacement: the counter measures how
        # many control steps have passed since the car last covered
        # ``stall_move_min`` metres.  A speed gate and a longitudinal-acceleration
        # gate were both exploitable (a car can creep while still accelerating),
        # and a drift legitimately dips below any instantaneous speed threshold.
        pos_now = self.s[:, :2]
        moved = torch.norm(pos_now - self._stall_anchor, dim=1)
        reached = moved >= self.stall_move_min
        if getattr(self, "_physics_advanced", False):
            self._stall_anchor = torch.where(reached.unsqueeze(1), pos_now, self._stall_anchor)
            self.no_progress_count[reached] = 0
            self.no_progress_count[~reached] += 1

        # u = vector from nearest waypoint to current pos; v = velocity direction at nearest waypoint
        # theta = angle from u to v
        # theta > 0 means current pos is to the right of waypoint
        best = self.step_progress.unsqueeze(1)
        ux = diff[:, :, 0].gather(1, bidx).squeeze(1)
        uy = diff[:, :, 1].gather(1, bidx).squeeze(1)
        u_angle = torch.atan2(uy, ux)
        v_angle = waypoints_filtered[:, :, 2].gather(1, bidx).squeeze(1)
        theta = v_angle - u_angle
        theta = torch.atan2(torch.sin(theta), torch.cos(theta))
        pos_diff_sign = torch.sign(theta)
        self.angle_diff = angle_diff.gather(1, bidx).squeeze(1)
        self.signed_pos_diff = pos_diff_sign * pos_diff.gather(1, bidx).squeeze(1)
        self.beta_diff = beta_diff.gather(1, bidx).squeeze(1)
        self.beta_ref = torch.zeros_like(self.beta_diff)   # set from the curvature map later
        self.kappa_diff = kappa_diff.gather(1, bidx).squeeze(1)
        self.kappa_ref = waypoints_filtered[:, :, 3].gather(1, bidx).squeeze(1)
        for name, old in saved.items():
            value = getattr(self, name)
            select = mask.view(-1, *([1] * (value.ndim - 1)))
            setattr(self, name, torch.where(select, value, old))
        if mask is None or not hasattr(self, "_projection_xy"):
            self._projection_xy = self.s[:, :2].clone()
        else:
            self._projection_xy[mask] = self.s[mask, :2]

    @staticmethod
    def _dense_target_gate(value, target, scale):
        """Smooth [0, 1] gate with nonzero gradient away from its target."""
        return torch.exp(-0.5 * ((value - target) / scale).square())

    @staticmethod
    def _positive_target_gate(value, target):
        """Zero-baseline progress gate for a signed positive requirement.

        Unlike a Gaussian target gate, zero behaviour earns exactly zero.  The
        clamp boundary retains an autograd direction at zero.
        """
        target = torch.as_tensor(target, device=value.device, dtype=value.dtype)
        return torch.clamp(value / target.clamp_min(1e-6), 0.0, 1.0)

    @staticmethod
    def _xcar_target_improvement(value, target):
        """Signed quadratic improvement over doing nothing, in [-1, 1].

        XCAR learns drift from ``-(value-target)^2``: samples on the wrong side
        are worse, rather than indistinguishable from zero.  Subtracting the
        zero-behaviour score preserves that direction while making zero exactly
        neutral and the target exactly one.
        """
        target = torch.as_tensor(target, device=value.device, dtype=value.dtype)
        denom = target.abs().clamp_min(1e-6).square()
        improvement = (target.square() - (value - target).square()) / denom
        return torch.clamp(improvement, -1.0, 1.0)

    @staticmethod
    def _weighted_drift_progress(progress):
        """Dense zero-baseline drift progress without an easy-component loophole.

        Columns are rear kappa, throttle, signed beta and countersteer.  Attitude
        receives 75% of the mass, so pure forward wheelspin can earn at most 25%
        of this shaping term.  The separate strict product still requires every
        component at once.
        """
        weights = progress.new_tensor((0.15, 0.10, 0.40, 0.35))
        return (progress * weights).sum(dim=1)

    def _adaptive_beta_shaping(self, beta, beta_target):
        """Keep adaptive drift learnable without forcing drift at zero demand.

        A raw quadratic loses its exploration gradient quadratically when the
        physics map asks for a smaller (but still meaningful) sideslip.  At a
        0.15 rad target it is about seven times weaker than at 0.40 rad, and the
        kappa term used to be attenuated a second time.  Separate target
        *magnitude* from target *eligibility*: the requested angle stays exactly
        the physical map output, while a smooth demand gate and normalized error
        keep a comparable learning signal.  Below 0.15 rad the gate fades
        smoothly to zero; at zero demand the neutral beta penalty remains.
        """
        target_abs = beta_target.abs()
        x = torch.clamp(target_abs / 0.15, 0.0, 1.0)
        demand = x * x * (3.0 - 2.0 * x)
        # The square root is essential because the error is squared below.  It
        # makes d(loss)/d(beta) at beta=0 approximately independent of target
        # magnitude instead of over-amplifying small targets.
        scale = torch.sqrt(self.drift_beta_target / target_abs.clamp_min(0.10))
        cap = max(self.beta_max, self.drift_beta_target + 0.15) ** 2
        target_loss = torch.clamp(((beta - beta_target) * scale) ** 2, max=cap)
        neutral_loss = torch.clamp(beta ** 2, max=cap)
        return -(demand * target_loss + (1.0 - demand) * neutral_loss), demand

    def _reference_beta(self, kappa, speed, idx=None):
        """Reference sideslip: the tight-corner target, faded by feasibility.

        The geometric target is the rear-slip-at-peak relation

            beta_geo = alpha_peak + lR * |kappa|

        which a tighter corner makes larger.  It is only reachable when the car
        is actually using a large fraction of the available grip: measured at
        mu 1.0, R 16 m, 6.1 m/s the car achieves 0.056 rad against a 0.193 rad
        request with random sign, i.e. it simply drives with grip.  The target is
        therefore scaled by the friction utilisation

            util = v^2 |kappa| / (mu g),  factor = clamp(util / util_ref, lo, 1)

        so the reference never asks for a sideslip the car cannot produce.  Both
        the speed and the grip are privileged, so this stays reward/critic only.
        """
        kap = kappa.abs()
        extra = (1,) * max(kappa.dim() - 1, 0)
        # ``idx`` selects the environments being spawned: the plant columns are
        # full batch tensors, while a respawn only draws a subset of kappa
        lr = self.plant.col("lR").reshape(-1, *extra)
        if idx is not None:
            lr = lr[idx]
        mu = (self.plant.col("D") * self.plant.tyre_scale.squeeze(1)).clamp_min(0.05).reshape(-1, *extra)
        if idx is not None:
            mu = mu[idx]
        gg = self.plant.col("g").reshape(-1, *extra)
        if idx is not None:
            gg = gg[idx]
        if not torch.is_tensor(speed):
            speed = torch.full_like(kappa, float(speed))
        if speed.ndim == 0:
            speed = speed.expand_as(kappa)
        while speed.ndim < kappa.ndim:
            speed = speed.unsqueeze(-1)
        # MEASURED against the trim solver (rounds 43-44): the steady sideslip the
        # car can actually hold is l_R*kappa plus a rear slip angle that is SMALL and
        # slightly negative at these utilisations -- 0.087 rad at (mu 1, R 12, 5 m/s)
        # against l_R*kappa = 0.0996, and 0.040 at 10 m/s.  The old expression added a
        # PEAK slip angle on top and then faded it UP with speed, both backwards: it
        # asked 0.220 rad where 0.074 was attainable, so the penalty was permanent.
        beta_geo = torch.clamp(self.beta_scale * lr * kap, max=self.beta_max)
        util = (speed.square() * kap / (mu * gg)).clamp(0.0, 1.0)
        # the attainable sideslip FALLS as the car uses more grip (trim: 0.087 at
        # util 0.21, 0.074 at 0.42, 0.040 at 0.85 as a fraction of l_R*kappa), so the
        # fade has to go DOWN with utilisation -- the old one went up, which is why
        # the reference was worst exactly at the operating point
        factor = (1.05 - 0.75 * util).clamp(0.35, 1.0)
        # A car at standstill has NO sideslip, so the reference must fade to zero with
        # speed as well.  Without this the very first observation of a rest start said
        # "you are 0.356 rad off target" (measured in CarSim: e_beta = -0.3559 at v = 0),
        # the actor answered with throttle 0.62 AND brake 0.47, pedal exclusivity kept the
        # brake, and the car never launched -- which is why every CarSim episode ended at
        # 1.1 s under the corrected action mapping.
        speed_fade = (speed / 3.0).clamp(0.0, 1.0)
        if self.drift_mode:
            if self.beta_ref_mode in ("adaptive_drift", "geometry_drift"):
                # Demand large sideslip only where it is useful and physically
                # plausible: a sufficiently tight corner, near the friction
                # limit, on a surface where power oversteer is accessible.
                # Smoothstep gates avoid reward discontinuities at each boundary.
                def _smooth_gate(x):
                    x = x.clamp(0.0, 1.0)
                    return x * x * (3.0 - 2.0 * x)

                util_gate = _smooth_gate(
                    (util - self.drift_util_lo) /
                    max(self.drift_util_hi - self.drift_util_lo, 1e-6))
                if self.beta_ref_mode == "geometry_drift":
                    # A path reference should not oscillate with the ratio of
                    # local curvature to the most demanding preview point.
                    # Geometry/grip gates below still suppress unsuitable bends.
                    util_gate = torch.ones_like(util_gate)
                curve = lr * kap
                curve_gate = _smooth_gate(
                    (curve - self.drift_curve_lo) /
                    max(self.drift_curve_hi - self.drift_curve_lo, 1e-6))
                # Power countersteer is not monotonically easier as a corner
                # gets tighter.  CarSim and FastPlant both hold the mu075 drift
                # branch near lR*kappa ~= 0.08, while lR*kappa ~= 0.16 has only
                # a steer-into-turn / low-throttle equilibrium.  Fade the drift
                # request back out above that verified band; path tracking then
                # remains the objective in hairpins.
                curve_gate = curve_gate * _smooth_gate(
                    (self.drift_curve_fade_hi - curve) /
                    max(self.drift_curve_fade_hi - self.drift_curve_fade_lo, 1e-6))
                grip_gate = _smooth_gate(
                    (self.drift_mu_hi - mu) /
                    max(self.drift_mu_hi - self.drift_mu_lo, 1e-6))
                # The verified mu075 equilibrium is a genuine large-angle
                # branch (|beta| ~= 0.54), not a half-sized interpolation
                # between drift and normal cornering.  Keep the full drift
                # target through the low/mid-grip range and use the gate only
                # to fade the request near the high-grip endpoint.
                grip_gate = _smooth_gate(grip_gate / 0.5)
                target = self.drift_beta_target * util_gate * curve_gate * grip_gate
                return -torch.sign(kappa) * target * speed_fade
            # explicit drift target: a large sideslip, faded in only at very low speed and
            # faded out above the point where the car cannot hold it any more
            # The trim solver's drift branch has beta and delta of OPPOSITE sign with beta
            # taking the sign of the curvature (trim_solve.py), so the target is
            # +sign(kappa)*target, NOT -sign(kappa).  The old sign asked for beta = -0.4 on
            # a positive-curvature corner while the physically correct drift is +0.4, which
            # made a real drift score exactly the same as no sideslip at all.
            # SIGN: in this repo beta = velocity heading - body heading, so a left-hand
            # power slide at positive curvature has the nose pointing LEFT of the velocity
            # vector, i.e. beta < 0 and delta < 0 (beta*delta > 0).  The review derived this
            # from the rear slip angle at beta = +0.4 coming out positive (0.29-0.34) and the
            # tyre then pushing OUTWARD, i.e. the wrong way for a left turn.  The target is
            # therefore -sign(kappa)*target.
            _tgt = torch.full_like(kappa, self.drift_beta_target)
            return -torch.sign(kappa) * _tgt * (speed / 2.0).clamp(0.0, 1.0)
        return -torch.sign(kappa) * beta_geo * factor * speed_fade

    def obs(self):
        """Real-vehicle observable state, after the XCAR observation design.

        Removed relative to the original XCAR observation (all of these are not
        measurable on the real car): sideslip angle ``beta``, speed ``V``, the
        world frame ``vx, vy``, the contact-patch velocities ``vflx, vfrx`` and
        the sideslip error ``beta - beta_ref``.  What remains is what the car
        actually has: yaw rate (IMU), wheel speeds (encoders), previous
        commands, tracking errors and the previewed reference -- so the policy
        has to infer the hidden state (speed, sideslip) from history, which is
        why the actor is a GRU.

        Preview positions and tangent headings use the localised BODY heading.
        True course error is reserved for reward/critic; the actor receives
        body-heading error. Localisation and IMU/encoder packets are subjected
        to the same sensor model before either actor branch sees them.
        """
        x = self.s[:, 0]
        y = self.s[:, 1]
        psi = self.s[:, 2]
        psi = torch.atan2(torch.sin(psi), torch.cos(psi))
        r = self.es[:, 2]
        # privileged quantities: used by the reward only, never observed
        beta = self.es[:, 7]
        v = self.es[:, 6]
        vflx = self.es[:, 8]
        vfrx = self.es[:, 9]
        wheel_speed = self.s[:, 6:10] * self.encoder_radius
        # curvature estimate from measurable signals (yaw rate / wheel speed)
        # Use the FRONT wheels for the speed estimate: during a drift the rear
        # wheels spin far above the ground speed, so averaging all four
        # over-estimates the speed by ~50 % and makes r_ref = v_est*kappa_ref
        # drive the policy into over-rotation (the observed course failures).
        v_est = wheel_speed[:, :2].mean(dim=1).clamp(min=0.5)
        kappa_est = (r / v_est).clamp(-2.0, 2.0)   # bounded curvature estimate

        xd = self.s[:, 3]
        yd = self.s[:, 4]
        veldir = torch.atan2(yd, xd)
        veldir[torch.hypot(xd, yd) < 0.01] = psi[torch.hypot(xd, yd) < 0.01]

        self._v_hint = v
        # obs() runs once per control step in training but is called again by the
        # evaluator and the renderer.  Advancing the nearest-point search twice
        # would corrupt progress and the stall counter, so guard it.
        update_history = getattr(self, "_track_pending", True)
        if update_history:
            self.find_nearest_waypoint(x, y, veldir, beta, kappa_est)
            self._track_pending = False

        # --- reference sideslip: reward / critic only ---------------------
        # The target is the steady-drift sideslip that the local curvature and the
        # available friction imply.  util = v^2|kappa| / (mu g) is the fraction of
        # the available lateral grip the corner actually asks for, so a gentle
        # corner at low speed asks for almost no sideslip and a corner at the
        # friction limit asks for beta_max.  It uses the TRUE speed and the TRUE
        # grip, so it is privileged information and never reaches the actor.
        # Every reference mode is state dependent: even ``adaptive_drift`` must
        # be recomputed after the plant advances because its utilisation gate
        # depends on the current speed.  Previously this condition omitted the
        # adaptive mode, leaving ``beta_ref`` at its all-zero constructor value
        # during training while reset_done_envs() spawned from a nonzero
        # adaptive target.  That made the reset distribution and reward demand
        # contradictory and disabled the drift-opportunity gate entirely.
        if self.beta_ref_mode in ("physics", "constant", "adaptive_drift", "geometry_drift"):
            self.beta_ref = self._reference_beta(self.kappa_ref.unsqueeze(1), v).squeeze(1)
            self.beta_diff = torch.atan2(torch.sin(beta - self.beta_ref),
                                         torch.cos(beta - self.beta_ref))
        angle_diff = self.angle_diff
        signed_pos_diff = self.signed_pos_diff
        beta_diff = self.beta_diff
        beta_ref = self.beta_ref
        kappa_diff = self.kappa_diff
        kappa_ref = self.kappa_ref
        path_heading = self.waypoints_at(self.progress.unsqueeze(1))[:, 0, 2]
        heading_error = torch.atan2(torch.sin(psi - path_heading),
                                    torch.cos(psi - path_heading))

        # look ahead 3m (scaled with the reference geometry)
        # Preview points.  With --preview-time they are specified as TIME
        # headways and converted to an arc length with the current speed, so the
        # look-ahead in seconds stays constant from 4 to 10 m/s (a fixed metric
        # preview is 2.3 s at 4 m/s but only 0.85 s at 10 m/s).
        if self.preview_distances_m is not None:
            ds_wp = (self._path_ds[self.path_generation, self.path_id].unsqueeze(1)
                     if self._cyclic_reference else self.s.new_full((self.n, 1), self.waypoint_ds))
            offs = (self.preview_distances_m[None] / ds_wp.clamp_min(1e-6)).round().long().clamp_min(1)
            inds = self.progress[:, None] + offs
            inds = (inds.remainder(self.num_waypoints) if self._cyclic_reference
                    else inds.clamp(0, self.num_waypoints - 1))
            la_inds = [inds[:, k:k+1] for k in range(3)]
        elif self.preview_time is not None:
            ds_wp = (self._path_ds[self.path_generation, self.path_id].unsqueeze(1)
                     if self._cyclic_reference else self.waypoint_ds)
            # use the ENCODER-derived speed, not the true speed: the preview
            # positions are actor-visible, so the true speed must not leak into
            # them through the time headway.
            preview_speed = wheel_speed[:, :2].mean(dim=1).clamp_min(0.)
            # Preview uses the clean encoder-derived speed as the filtered
            # deployment signal proxy. Do not inject wheel measurement noise
            # into waypoint selection; true translational speed is never used.
            offs = (preview_speed.unsqueeze(1) * self.preview_time.unsqueeze(0) / ds_wp).round().long()
            offs = offs.clamp(min=1, max=self.num_waypoints - 1)
            if self._cyclic_reference:
                la_inds = [((self.progress.unsqueeze(1) + offs[:, k:k + 1]) % self.num_waypoints)
                           for k in range(3)]
            else:
                la_inds = [(self.progress.unsqueeze(1) + offs[:, k:k + 1]).clamp(0, self.num_waypoints - 1)
                           for k in range(3)]
        elif not self._cyclic_reference:
            la_inds = [(self.progress + i).unsqueeze(-1)
                       for i in (int(200 * self.preview_scale), int(400 * self.preview_scale),
                                 int(600 * self.preview_scale))]
            for la_ind in la_inds:
                la_ind[la_ind >= self.num_waypoints] = self.num_waypoints - 1
        else:
            la_inds = [(self.progress + i).unsqueeze(-1) % self.num_waypoints
                       for i in (int(200 * self.preview_scale), int(400 * self.preview_scale),
                                 int(600 * self.preview_scale))]

        la = [self.waypoints_at(la_ind).squeeze(1) for la_ind in la_inds]
        # curvature at the preview points, used by the friction-limit speed target
        self.kappa_preview = torch.stack([w[:, 3] for w in la], dim=1)
        if self.reward_mode == "adaptive_envelope" and self.beta_ref_mode in ("adaptive_drift", "geometry_drift"):
            self.beta_ref = self._reference_beta(
                self.kappa_ref[:, None], self._reference_speed()).squeeze(1)
            self.beta_diff = torch.atan2(torch.sin(beta - self.beta_ref),
                                        torch.cos(beta - self.beta_ref))
            beta_ref, beta_diff = self.beta_ref, self.beta_diff


        def convert(waypoints):
            """Cast a waypoint into the measured body-heading frame."""
            x1 = waypoints[:, 0]
            y1 = waypoints[:, 1]
            veldir1 = waypoints[:, 2]
            kappa1 = waypoints[:, 3]
            theta = torch.atan2(y1 - y, x1 - x) - psi
            l = torch.hypot(y1 - y, x1 - x)
            x2 = l * torch.cos(theta)
            y2 = l * torch.sin(theta)
            veldir2 = veldir1 - psi
            veldir2 = torch.atan2(torch.sin(veldir2), torch.cos(veldir2))
            kappa2 = kappa1 * self.length_scale          # O(1) normalisation
            x2, y2, veldir2, kappa2 = map(lambda t: torch.unsqueeze(t, 1),
                                                 [x2, y2, veldir2, kappa2])
            # reference sideslip is NOT observable to the actor: it belongs to the reward and to the critic only.  kappa_ref is kept because it is the
            # geometric cue from which the required sideslip can be inferred.
            return torch.cat([x2, y2, veldir2, kappa2], 1)

        la_converted = [convert(waypoints) for waypoints in la]

        # Pure pursuit steering for the middle preview point: delta_pp is the
        # steering angle that would put the car on a circular arc through that
        # point.  It is computable on the car from the preview alone, and serves
        # as the "steer towards the line" reference for the reward.
        # Sign: FastPlant/CarSim contract uses positive road-wheel angle for a
        # positive-y/positive-yaw (left) turn.  A preview point to the left must
        # therefore produce positive steering.  The old extra minus sign made
        # every pure-pursuit teacher/reward steer away from the reference.
        pp = la_converted[1]
        ld2 = (pp[:, 0] ** 2 + pp[:, 1] ** 2).clamp(min=1e-3)
        self.delta_pp = torch.atan2(2.0 * self.wheelbase * pp[:, 1], ld2)

        # grace period instead of the original "progress >= 5" guard: the progress
        # based guard never fires for a car that stops on the first waypoints
        fail_npc = self.no_progress_count >= self.no_progress_limit
        fail_course = torch.abs(angle_diff) > self.course_limit
        fail_off = torch.abs(signed_pos_diff) > self.offpath_limit
        failed = (self.step_count > 10) & (fail_npc | fail_course | fail_off)
        # snapshot of the terminal conditions, so a caller can attribute the
        # failure after the environment has already respawned
        self._term_masks = torch.stack((fail_npc, fail_course, fail_off), dim=1)
        if self.continuing_task:
            succeeded = torch.zeros_like(failed)
        elif self.train:
            # the episode cap is the ONLY success criterion: reaching the end of the
            # reference bank is not a task objective and, with a progress index that
            # can advance 19 waypoints per step, it fired after a few seconds and
            # mislabelled early terminations as successes
            succeeded = self.step_count >= self.max_steps
        else:
            succeeded = (self.step_count >= self.max_steps)
        # a failure on the final step must not be overwritten by the time-out
        succeeded = succeeded & ~failed
        self.is_done[failed] = 1
        self.is_done[succeeded] = 2
        # exclusive first-failure attribution, read by eval.py after the respawn
        self._term_reason = torch.where(
            succeeded, torch.full_like(self.step_count, 3),
            torch.where(fail_npc, torch.full_like(self.step_count, 0),
                        torch.where(fail_course, torch.full_like(self.step_count, 1),
                                    torch.where(fail_off, torch.full_like(self.step_count, 2),
                                                torch.full_like(self.step_count, 4)))))

        if self.pedals:
            ctrl = [self.u_cmd[:, 0], self.u_cmd[:, 1], self.u_cmd[:, 2]]
        else:
            ctrl = [self.u[:, 1], self.u[:, 2], self.u[:, 3], self.u[:, 4]]

        now = {
            "beta": beta, "v": v, "kappa": r / (v + 1e-6),   # true curvature: reward only
            "r": r, "r_ref": v_est * kappa_ref,
            "vflx": vflx, "vfrx": vfrx,
            "delta": self.u_cmd[:, 0],
            "raw_delta": getattr(self, "_raw_action", self.u_cmd)[:, 0],
            "wheel": wheel_speed,
            "kappa_power": (
                self.extras["kappa_power"]
                if self.extras is not None and "kappa_power" in self.extras
                else (-self.extras["sx"] if self.extras is not None
                      else torch.zeros_like(wheel_speed))
            ),
            "angle_diff": angle_diff, "signed_pos_diff": signed_pos_diff,
            "e_pos": signed_pos_diff / self.length_scale,
            "beta_diff": beta_diff, "kappa_ref": kappa_ref,
            "prog": self.step_progress, "failed": failed.float(),
        }
        if update_history:
            mask = getattr(self, "_tracking_mask", None)
            if mask is None or self._now is None:
                self._prev = self._now
                self._now = {k: t.clone() for k, t in now.items()}
            else:
                # Only initialise the newborn rows; survivors retain the
                # previous real control sample used for smoothness rewards.
                for k, t in now.items():
                    m = mask.view(-1, *([1] * (t.ndim - 1)))
                    self._now[k] = torch.where(m, t, self._now[k])

        # observed: ego state + previous commands + tracking errors + preview
        e_pos = signed_pos_diff / self.length_scale
        e_kappa = (kappa_diff * self.length_scale).clamp(-5.0, 5.0)
        # accelerometer: body frame specific force averaged over the control
        # period; az is gravity for a planar vehicle (a real IMU still reads it)
        ax_acc = self.extras["ax"] if self.extras is not None else torch.zeros_like(r)
        ay_acc = self.extras["ay"] if self.extras is not None else torch.zeros_like(r)
        az_acc = torch.full_like(r, float(self.plant.g))
        scalars = [r, ax_acc, ay_acc, az_acc] + ctrl
        scalars += [wheel_speed[:, i] for i in range(4)]
        # travel: integral of the SLOWEST wheel speed.  In a drift the rear
        # wheels spin far above ground speed and the fronts are the honest ones,
        # so the slowest wheel is the conservative progress measurement a real
        # car could integrate from the four encoders.
        v_min = wheel_speed.abs().amin(dim=1)
        if getattr(self, "_physics_advanced", False):
            self.travel = self.travel + v_min * self.dt
            self._physics_advanced = False
        # actor channels: tracking errors + the encoder-derived travel (no reference
        # sideslip here - that is reward/critic only)
        # A cumulative odometer is non-stationary in a continuing task and will
        # grow without bound.  Keep the same real encoder source and channel
        # count, but expose the stationary conservative speed estimate there.
        # Episodic mode preserves legacy semantics for checkpoint replay.
        encoder_feature = v_min if self.continuing_task else self.travel / self.length_scale
        scalars += [e_pos, heading_error, e_kappa, encoder_feature]
        obs_list = [t.unsqueeze(1) for t in scalars]
        obs_list.extend(la_converted)
        obs = torch.cat(obs_list, 1) * self.obs_scale
        actor_obs = self._actor_sensor_packet(obs)
        if self.obs_privileged:
            priv = torch.stack([self.es[:, 7], self.es[:, 6], self.es[:, 3], self.es[:, 4]], dim=1)
            actor_obs = torch.cat([actor_obs, priv * self.priv_concat_scale], dim=1)
        if self.asymmetric:
            # privileged critic input: everything the simulation knows and the
            # car cannot measure (sideslip, absolute speed, tyre forces/slips)
            slip = self.extras["slip"] if self.extras is not None else torch.zeros_like(wheel_speed)
            fz = self.extras["fz"] if self.extras is not None else torch.zeros_like(wheel_speed)
            fy = self.extras["fy_body"] if self.extras is not None else torch.zeros_like(wheel_speed)
            fx = self.extras["fx_body"] if self.extras is not None else torch.zeros_like(wheel_speed)
            # forces are normalised by the static wheel load m*g/4, so Fz is O(1)
            # (it can reach ~2 at the front under braking) instead of O(10^4)
            f_ref = (self.plant.col("m") * self.plant.col("g") / 4.0).unsqueeze(1).clamp(min=1.0)
            # Versioned for fresh mirror runs; old checkpoints retain world velocities.
            critic_vx, critic_vy = self.s[:, 3], self.s[:, 4]
            if self.critic_velocity_frame == "body":
                c_psi, s_psi = self.s[:, 2].cos(), self.s[:, 2].sin()
                critic_vx, critic_vy = (c_psi * critic_vx + s_psi * critic_vy,
                                        -s_psi * critic_vx + c_psi * critic_vy)
            priv = [
                beta, v, r, critic_vx, critic_vy,
                (r / (v + 1e-6)), kappa_ref, self._now["r_ref"], beta_ref,
                signed_pos_diff / self.length_scale, angle_diff,
                ax_acc, ay_acc, az_acc,
            ]
            priv_list = [t.unsqueeze(1) for t in priv]
            priv_list.append(wheel_speed)
            priv_list.append(slip)
            priv_list.append(fz / f_ref)
            priv_list.append(fx / f_ref)
            priv_list.append(fy / f_ref)
            priv_list.append(torch.stack(ctrl, dim=1))
            # the preview is what makes the value estimate informative about the
            # future; without it the privileged critic is blind to the path
            # critic previews keep the reference sideslip (5 channels); the actor
            # version above does not have it
            beta_preview = self._reference_beta(self.kappa_preview, v[:, None])
            for j in range(3):
                priv_list.append(torch.cat([la_converted[j], beta_preview[:, j:j + 1]], dim=1))
            states = torch.cat(priv_list, 1) * self.priv_scale
            return {"obs": actor_obs, "states": states}
        return actor_obs

    def _reference_speed(self):
        """Heuristic attainable-speed scale; not a claimed friction bound."""
        if self.speed_target_mode != "limit":
            return torch.full_like(self.kappa_ref, self.speed_target)
        amplitude = (self.plant.col("D") * self.plant.tyre_scale.squeeze(1)).clamp(.3, 3.)
        curvature = torch.maximum(self.kappa_ref.abs(),
                                  self.kappa_preview.abs().amax(dim=1)).clamp_min(.01)
        return (self.speed_margin * torch.sqrt(amplitude * self.plant.col("g") /
                                                curvature)).clamp(2., self.speed_cap)

    def reward(self):
        now, prev = self._now, self._prev
        beta = now["beta"]
        v = now["v"]
        vfrx, vflx = now["vfrx"], now["vflx"]
        delta = now["delta"]
        wheel = now["wheel"]
        angle_diff = now["angle_diff"]
        signed_pos_diff = now["signed_pos_diff"]
        beta_diff = now["beta_diff"]
        kappa_ref = now["kappa_ref"]
        prog = now["prog"]
        is_failed = now["failed"]

        if prev is None:
            return torch.zeros(self.n, device=self.device)

        # Metric quantities are normalised back to XCAR units (the reference
        # geometry scale) so that the reward weights keep their meaning across
        # vehicle sizes; dimensionless quantities are used directly.
        s = self.length_scale
        # saturate the lateral penalty: far off the path the episode is about to
        # fail anyway, and an unbounded quadratic made long episodes massively
        # negative (reward -2000 per episode at a relaxed corridor)
        rew_pos = - torch.clamp((signed_pos_diff / s) ** 2, max=self.pos_cap)
        if self.drift_mode:
            # only the excess beyond the deadband is penalised
            # now["e_pos"] is normalised by length_scale; the old code compared
            # it with offpath_limit in metres, making this term identically zero
            # throughout the legal corridor.
            _cap = max(self.offpath_limit / self.length_scale, 1e-6)
            _band = self.drift_corridor_deadband * _cap
            _excess = torch.clamp(now["e_pos"].abs() - _band, min=0.0)
            rew_pos = -torch.clamp((_excess / max(_cap - _band, 1e-3)) ** 2, max=1.0)
        rew_dir = - angle_diff ** 2
        # progress is capped at the rate the reference speed implies, exactly like
        # the original XCAR formulation caps it: a car that goes faster than the
        # reference must not earn extra reward for it (that was the over-speed
        # incentive).  ``v_tgt`` is filled below.
        _prog_raw = prog
        rew_smooth = self.smooth_scale * (
            - ((delta - prev["delta"]) ** 2
               + 1e-2 * (((wheel - prev["wheel"]) / self.speed_ref) ** 2).sum(dim=1)))
        # Adaptive drift target: holding a fixed sideslip locks the drift radius,
        # so a car that has drifted away from the line cannot recover while it is
        # still demanded to drift (observed: the policy holds the reference beta
        # and runs off the path, which is by far the dominant failure).  The
        # demanded sideslip fades out as the lateral error grows, i.e. drift on
        # the line, straighten to come back.
        if self.beta_gate_scale > 0:
            beta_gate = torch.clamp(
                1.0 - (signed_pos_diff / (self.beta_gate_scale * s)) ** 2, 0.0, 1.0)
        else:
            beta_gate = torch.ones_like(signed_pos_diff)
        # pure pursuit tracking: the steering that would follow the preview point
        rew_pp = - torch.clamp(((delta - self.delta_pp) / 0.3) ** 2, max=4.0)
        # the gate is optional: with the geometry-derived reference the reward
        # should target exactly beta_ref, so the gate is disabled when its scale is 0
        beta_target = self.beta_ref * beta_gate if self.beta_gate_scale > 0 else self.beta_ref
        # Rear-wheel slip is required only where the reference map actually asks
        # for a power slide.  With ``adaptive_drift`` the high-grip/straight
        # target is zero; an unconditional kappa target there would contradict
        # the user's objective and reward tyre smoke during a valid normal turn.
        # The fixed low-grip drift task still receives a factor of one.
        _drift_opportunity = torch.clamp(
            beta_target.abs() / max(self.drift_beta_target, 1e-6), 0.0, 1.0)
        # cap the shaping term: an uncapped quadratic lets a single spin dominate the
        # return with hundreds of negative reward in one step, which swamps the critic
        # The clamp must never be active at beta = 0, or the quadratic has zero gradient
        # there and PPO gets no direction at all.  With the drift target at 0.40 rad the old
        # cap of beta_max**2 = 0.32**2 was already saturated at beta = 0, which is why the
        # sideslip collapsed to zero and stayed there.
        _b_cap = max(self.beta_max, self.drift_beta_target + 0.15) ** 2
        rew_beta = -torch.clamp((beta - beta_target) ** 2, max=_b_cap)
        if self.drift_mode and self.beta_ref_mode in ("adaptive_drift", "geometry_drift"):
            rew_beta, _drift_opportunity = self._adaptive_beta_shaping(
                beta, beta_target)
        # Potential based recovery term: reward REDUCING the lateral error, not
        # just having a small one.  The static quadratic gives no direction at the
        # operating point and the car never learns to come back to the line.
        if prev is not None and prev.get("e_pos") is not None:
            rew_recover = (torch.abs(prev["e_pos"]) - torch.abs(now["e_pos"])) / self.dt
        else:
            rew_recover = torch.zeros_like(rew_beta)
        rew_lowspeed = torch.clamp(v, 0., 0.5 * self.speed_scale) - 0.5 * self.speed_scale
        rew_sf = - ((vfrx - wheel[:, 1]) / self.speed_scale) ** 2 \
                 - ((vflx - wheel[:, 0]) / self.speed_scale) ** 2
        # clamp both curvature and yaw terms: during a spin the instantaneous
        # curvature / yaw error diverges and an unbounded square reached -100 per
        # step, swamping the alive bonus and every other term
        rew_kappa = - torch.clamp(((now["kappa"] - kappa_ref) * s) ** 2, max=self.kappa_cap)
        rew_yaw = - torch.clamp(((now["r"] - now["r_ref"]) / self.yaw_scale) ** 2, max=self.yaw_cap)
        # One sided linear speed term: the quadratic form has almost no gradient
        # at low speed, so the policy settles at 1.5 m/s where the alive bonus
        # dominates and then dies of its own slow pace.  Only being BELOW the
        # target is penalised, linearly, so the incentive is constant.
        if self.speed_target_mode == "limit":
            # Speed target at the limit of adhesion: v_lim = sqrt(mu g / |kappa|).
            # The grip multiplier is available to the reward but not to the actor,
            # so the policy must infer how fast it may go from the vehicle response,
            # which is exactly the adaptive behaviour we want.
            v_tgt = self._reference_speed()
        else:
            v_tgt = self.speed_target
        if self.speed_target_mode == "limit":
            # Two-sided: the limit target is a GOAL, so over-driving it is as
            # wrong as under-driving.  With a floor-only term the policy kept
            # 9.7 m/s on high grip against an 8.5 m/s target and failed 87 % of
            # the episodes by running out of the corridor.
            # TWO-SIDED: exceeding the limit target is as wrong as falling short,
            # otherwise the policy over-drives at high grip and leaves the corridor
            rew_speed = -torch.clamp(((v - v_tgt) / v_tgt) ** 2, max=1.0)
        else:
            rew_speed = -torch.clamp(((v - v_tgt) / v_tgt) ** 2, max=1.0)

        # progress is capped at the waypoints the reference speed covers in one
        # step, exactly as the original XCAR reward caps it: going faster than the
        # reference must not earn extra progress reward (that was the over-speed
        # incentive that pushed the car out of the corridor at high grip).
        _rate = v_tgt * self.dt / self.waypoint_ds
        if not torch.is_tensor(_rate):
            _rate = torch.full_like(v, float(_rate))
        rew_prog = torch.minimum(_prog_raw.clamp(min=0.0), _rate)
        # Over-speeding must not pay.  Measured at epoch 200 of drift_ev_v48: the car ran
        # at 7.37 m/s where the limit target was about 5.4, because progress (+0.7) and
        # the alive bonus (+2.5) outweighed the speed penalty (-1.6) and the car simply
        # left the corridor (94/96 failures).  Progress is now gated on being at or below
        # the target speed, so the only way to keep earning it is to hold the limit.
        if getattr(self, "speed_weight", 0.0) > 0.0:
            _v_gate = torch.where(self.es[:, 6] <= v_tgt,
                                  torch.ones_like(rew_prog),
                                  torch.zeros_like(rew_prog))
            rew_prog = rew_prog * _v_gate
        rew_final = -is_failed

        # --- drift quantities (computed before the dict; see the config block) -----
        _w = now["wheel"]
        # Rear longitudinal slip in one plant-independent convention.  This uses
        # the local wheel contact velocity, not total vehicle speed (which is
        # wrong by construction at large sideslip).
        _kx_rear = now["kappa_power"][:, 2:4].mean(dim=1)
        rew_drift_kappa = -torch.clamp((_kx_rear - self.drift_kappa_target) ** 2, max=4.0)
        # a power slide needs the driver holding the throttle open
        # Hinge penalty: demand the minimum pedal but do not pay extra for blind
        # full throttle after the requirement is met.
        rew_drift_throttle = -torch.clamp(
            self.drift_throttle_gate - self.u_cmd[:, 1], min=0.0) ** 2
        # countersteer: the wheel must point out of the turn while the car slides
        # r/V is near zero during a forward slide, so keying the countersteer off it asks
        # for no countersteer exactly when one is needed.  Use the reference curvature.
        _sig = torch.sign(now.get("kappa_ref", now["kappa"]))
        # countersteer target is -0.25*sign(kappa): delta and beta then share a sign,
        # which is the drift convention this repo uses
        rew_drift_counter = -torch.clamp((self.u_cmd[:, 0] + 0.25 * _sig) ** 2, max=1.0)
        # soft gates, each in [0, 1], multiplied together
        _g_slip = torch.clamp((_kx_rear - 0.20) / 0.20, 0.0, 1.0)
        _b = now["beta"]
        _kap = now["kappa_ref"]
        # gate ramps from 45 % to 90 % of the drift target, so as the target rises the
        # gate tracks it instead of saturating at a hard-coded 0.30 rad
        # The gate must be ACTIVE where the policy currently lives, or it is worth zero and --
        # worse -- its gradient is zero there too, which is why raising its weight to 40 could
        # not help.  The measured solution sits at |beta| 0.15 and throttle 0.23-0.28, so the
        # ramps start BELOW that and pay for improvement rather than for an unreachable target.
        _b_lo = 0.10
        _b_hi = 0.30
        _g_beta = torch.clamp((_b.abs() - _b_lo) / max(_b_hi - _b_lo, 1e-6), 0.0, 1.0)
        _sign_ok = ((_b * _kap) < 0.0).to(_b.dtype)
        # the countersteer gate was accidentally a CONSTANT 1, so the joint term never
        # checked countersteer at all.  Clamp((steer - target) / band) in the correct
        # direction: the wheel must sit near -0.25*sign(kappa).
        _g_delta = torch.clamp(1.0 - (self.u_cmd[:, 0] + 0.25 * _sig).abs() / 0.20,
                               0.0, 1.0)
        _g_thr = torch.clamp((self.u_cmd[:, 1] - 0.20) / 0.15, 0.0, 1.0)
        _g_turn = torch.clamp((_kap.abs() - 0.02) / 0.02, 0.0, 1.0)
        rew_drift_joint = (_g_slip * _g_beta * _sign_ok * _g_delta * _g_thr * _g_turn)

        _rew_thr = torch.clamp(self.u_cmd[:, 1] - self.throttle_gate, min=0.0)

        if self.reward_mode in ("bounded_joint", "adaptive_envelope"):
            # All non-terminal terms are in [0, 1].  This makes every live step
            # non-negative and a failure strictly negative, so terminating early
            # can never avoid an accumulating shaping debt.  The additive soft
            # gates provide gradients before the hard joint behavior is reached;
            # the product bonus is paid only when tracking and every drift
            # criterion hold at the same instant.
            _path_gate = torch.clamp(1.0 - signed_pos_diff.abs() / self.offpath_limit,
                                     0.0, 1.0)
            if self.reward_mode == "adaptive_envelope" and self.adaptive_path_scale_m is not None:
                # Precision is independent of the failure boundary. A rational
                # score retains recovery signal even far from the centreline.
                _path_gate = 1.0 / (1.0 + (signed_pos_diff / self.adaptive_path_scale_m).square())
            _course_gate = torch.clamp(1.0 - angle_diff.abs() / self.course_limit,
                                       0.0, 1.0)
            _v_target_tensor = torch.as_tensor(
                v_tgt, device=v.device, dtype=v.dtype)
            _speed_gate = torch.clamp(1.0 - (v - _v_target_tensor).abs() /
                                      torch.clamp(0.30 * _v_target_tensor, min=0.5),
                                      0.0, 1.0)
            _slip_gate = torch.clamp((_kx_rear - 0.25) /
                                     max(self.drift_kappa_target - 0.25, 1e-3), 0.0, 1.0)
            _thr_gate = torch.clamp((self.u_cmd[:, 1] - 0.25) /
                                    max(self.drift_throttle_gate - 0.25, 0.05), 0.0, 1.0)
            # The former clipped |beta| and hard sign gates were exactly zero at
            # the forward-wheelspin solution.  Their product therefore gave PPO
            # no direction toward signed sideslip or countersteer.  Gaussian
            # target gates remain bounded/nonnegative while providing a dense
            # gradient from beta=0 and delta=0 toward the physical drift attitude.
            _beta_gate = self._dense_target_gate(_b, beta_target, 0.30)
            _counter_gate = (self._dense_target_gate(
                delta, -0.25 * _sig, 0.20) * _g_turn)
            _steer_step = (delta - prev["delta"]).abs()
            _smooth_gate = torch.clamp(1.0 - _steer_step / 0.05, 0.0, 1.0)
            _track_mean = (_path_gate + _course_gate + _speed_gate) / 3.0
            _drift_mean = (_slip_gate + _thr_gate + _beta_gate + _counter_gate) / 4.0
            _joint = (_path_gate * _course_gate * _speed_gate * _slip_gate *
                      _thr_gate * _beta_gate * _counter_gate * _smooth_gate)
            live = 1.0 - is_failed
            if self.reward_mode == "adaptive_envelope":
                # Eligibility follows the reference operating point. Slowing
                # down cannot make the requested drift disappear; actual speed
                # still gates whether a sample earns the optional bonus.
                if self.drift_mode and self.beta_ref_mode in ("adaptive_drift", "geometry_drift"):
                    beta_target = self._reference_beta(
                        self.kappa_ref[:, None], _v_target_tensor).squeeze(1) * beta_gate
                    demand = (beta_target.abs() / 0.15).clamp(0.0, 1.0)
                    _drift_opportunity = demand.square() * (3.0 - 2.0 * demand)
                    _beta_gate = self._dense_target_gate(_b, beta_target, 0.30)
                    _counter_gate = self._positive_target_gate(
                        -delta * _sig, 0.25) * _g_turn
                _track_joint = _path_gate * _course_gate
                _speed_ratio = v / _v_target_tensor.clamp_min(0.5)
                _speed_use = torch.clamp(_speed_ratio, 0.0, 1.0)
                _speed_ceiling_ratio = torch.ones_like(_speed_ratio)
                if self.adaptive_speed_penalty == "envelope":
                    # The margin-reduced speed is a performance reference, not
                    # a hard upper bound. Allow self-selected equilibria up to
                    # the un-margined reference, retaining preview slowdown and
                    # the configured speed cap. This is a nominal envelope,
                    # not a guarantee of combined-slip feasibility.
                    _ceiling = (_v_target_tensor / self.speed_margin).clamp(max=self.speed_cap)
                    _speed_ceiling_ratio = torch.maximum(
                        _speed_ceiling_ratio, _ceiling / _v_target_tensor.clamp_min(0.5))
                _overspeed_safe = torch.clamp(
                    1.0 - torch.clamp(_speed_ratio - _speed_ceiling_ratio, min=0.0) / 0.25,
                    0.0, 1.0)
                if self.adaptive_speed_penalty == "none":
                    _overspeed_safe = torch.ones_like(_speed_ratio)
                _performance = _track_joint * _speed_use * _overspeed_safe
                # Only the speed component is decoupled. Drift quality keeps
                # its original precision gate; default checkpoint semantics stay exact.
                _speed_performance = _performance
                if self.adaptive_speed_coupling == "independent_position":
                    _speed_performance = _course_gate * _speed_use * _overspeed_safe
                elif self.adaptive_speed_coupling == "corridor":
                    def fade_abs(value, start, end):
                        t = ((value.abs() - start) / (end - start)).clamp(0., 1.)
                        return 1. - t.square() * (3. - 2. * t)
                    _corridor = fade_abs(signed_pos_diff, 1.5, 3.5)
                    _direction = fade_abs(angle_diff, 0.35 * self.course_limit, self.course_limit)
                    _speed_performance = _corridor * _direction * _speed_use * _overspeed_safe
                # A strict product erases all exploration signal when one factor
                # is absent.  Preserve XCAR's essential signed quadratic target
                # information: correct-side samples improve over zero, while
                # wrong-side beta/countersteer are explicitly worse than zero.
                # The weighted sum remains zero at zero behaviour and the strict
                # product remains the simultaneous-behaviour bonus.
                _beta_aligned = _b * torch.sign(beta_target)
                _counter_aligned = -delta * _sig
                _drift_progress = torch.stack((
                    self._positive_target_gate(_kx_rear, self.drift_kappa_target),
                    self._positive_target_gate(
                        self.u_cmd[:, 1], self.drift_throttle_gate),
                    self._xcar_target_improvement(
                        _beta_aligned, beta_target.abs().clamp_min(0.15)),
                    torch.clamp(_counter_aligned / 0.25, -1.0, 1.0),
                ), dim=1)
                _drift_dense = self._weighted_drift_progress(_drift_progress)
                if self.adaptive_shape in ("smooth_v1", "smooth_task_v2"):
                    # Compact S curves: no bonus for zero/negative powered slip,
                    # but progress starts before the former hard thresholds.
                    def s_curve(x):
                        x = x.clamp(0.0, 1.0)
                        return x.square() * (3.0 - 2.0 * x)
                    _slip_gate = s_curve(_kx_rear / self.drift_kappa_target)
                    _thr_gate = s_curve(self.u_cmd[:, 1] / self.drift_throttle_gate)
                    _counter_gate = s_curve(_counter_aligned / 0.25) * _g_turn
                    # Same rational half-bell family as path precision; 0.05
                    # radians per control step gives half smoothness credit.
                    _smooth_gate = 1.0 / (1.0 + (_steer_step / 0.05).square())
                # Necessary countersteer must not erase the drift bonus.
                # v2 retains the independent smoothness term and bounded raw
                # command-change cost, while actuator limits remain physical.
                _strict_smooth = (1.0 if self.adaptive_shape == "smooth_task_v2"
                                  else _smooth_gate)
                _drift_strict = (_slip_gate * _thr_gate * _beta_gate *
                                 _counter_gate * _strict_smooth)
                # During drift entry the course error necessarily grows before
                # countersteer arrests it.  Multiplying *dense* progress by the
                # full course gate erased precisely those transition samples;
                # weights 10 and 30 changed return but not behaviour.  Keep the
                # car on-road and near its feasible speed while learning the
                # transition, but reserve the strict bonus for simultaneous
                # path+course+speed tracking.
                _transition_support = (_path_gate * _speed_use * _overspeed_safe)
                _dense_share = 0.5 if self.adaptive_shape in ("smooth_v1", "smooth_task_v2") else 0.8
                _drift_quality = _drift_opportunity * (
                    _dense_share * _transition_support * _drift_dense
                    + (1.0 - _dense_share) * _performance * _drift_strict)
                # A0-only geometric teacher.  delta_pp is computed exclusively
                # from actor-visible preview/encoder signals; its weight is
                # explicitly annealed to zero before adaptive mode selection.
                # Gaussian support remains informative even when the initial
                # policy is more than 0.2 rad from the teacher.  The former
                # clipped ramp was identically zero there and PPO could not
                # correlate exploration with better guidance reward.
                _guidance_gate = torch.exp(
                    -0.5 * ((delta - self.delta_pp) / 0.20).square())
                _raw_change_sq = ((now["raw_delta"] - prev["raw_delta"]) / 0.10).square()
                _raw_change_cost = (_raw_change_sq / (1.0 + _raw_change_sq)
                                    if self.adaptive_shape in ("smooth_v1", "smooth_task_v2")
                                    else _raw_change_sq.clamp(max=1.0))
                # In drift-eligible corners, grip-only tracking must not collect
                # the entire task reward while avoiding the entry manoeuvre.
                # Preserve a dense floor, then restore the base reward smoothly
                # as signed sideslip AND positive rear power slip develop.
                # Eligibility is reference-based; high-grip/straight rewards
                # are exactly unchanged. Older reward profiles are untouched.
                _task_factor = torch.ones_like(_performance)
                if self.adaptive_shape == "smooth_task_v2":
                    _beta_progress = s_curve(
                        _beta_aligned / beta_target.abs().clamp_min(0.15))
                    _powered_slide = _beta_progress * _slip_gate
                    _task_factor = 1.0 - 0.75 * _drift_opportunity * (1.0 - _powered_slide)
                adaptive_terms = {
                    "alive": self.adaptive_alive * live,
                    "tracking": self.adaptive_tracking * _track_joint * _task_factor * live,
                    "performance": self.adaptive_performance * _speed_performance * _task_factor * live,
                    "optional_drift": self.adaptive_drift * _drift_quality * live,
                    "guidance": self.adaptive_guidance * _guidance_gate * live,
                    "smooth": self.adaptive_smooth * _smooth_gate * live,
                    "raw_steer_change": -0.05 * _raw_change_cost * live,
                    # Retain brake for approaching a tighter/slower segment or
                    # overspeed recovery; discourage braking below target speed.
                    "unneeded_brake": -0.10 * self.u_cmd[:, 2] * (
                        1. - _speed_ratio).clamp(0., 1.) * live,
                    "terminal": -self.terminal_cost * is_failed,
                }
                live_rew = torch.zeros_like(rew_pos)
                for name, term in adaptive_terms.items():
                    if name != "terminal":
                        live_rew = live_rew + term
                # Wrong-side samples must rank below zero behaviour, but a long
                # shaping debt must never make deliberate failure profitable.
                rew = live_rew.clamp_min(0.0) + adaptive_terms["terminal"]
                with torch.no_grad():
                    self.rew_stats = {k: float(t.detach().mean())
                                      for k, t in adaptive_terms.items()}
                return rew
            bounded_terms = {
                "alive": self.bounded_alive * live,
                "tracking_soft": self.bounded_tracking * _track_mean * live,
                "drift_soft": self.bounded_drift * _drift_mean * live,
                "joint": self.bounded_joint * _joint * live,
                "smooth_soft": self.bounded_smooth * _smooth_gate * live,
                "terminal": -self.terminal_cost * is_failed,
            }
            rew = torch.zeros_like(rew_pos)
            for term in bounded_terms.values():
                rew = rew + term
            with torch.no_grad():
                self.rew_stats = {k: float(t.detach().mean())
                                  for k, t in bounded_terms.items()}
            return rew

        _drift_on = 1.0 if self.drift_mode else 0.0
        terms = {
            "pos": self.pos_weight * rew_pos,
            "recover": self.recover_weight * rew_recover,
            "pure_pursuit": self.pp_weight * rew_pp,
            "dir": self.dir_weight * rew_dir,
            # the speed target already defines how fast to go; a raw distance
            # reward double-counts it and always favours going faster, which is
            # what pushed the car past the corridor at high grip
            "prog": self.prog_weight * rew_prog,
            # steering chatter is smooth-band energy; the weight is a knob because
            # the default was far too small to compete with the alive bonus
            "smooth": self.smooth_weight * rew_smooth,
            "throttle": self.throttle_weight * _rew_thr,
            # All drift-stack terms are conditional on drift_mode.  Previously
            # their nonzero defaults silently changed every nominal/v56-style
            # run, so a claimed throttle-only ablation was not an ablation.
            "drift_kappa": (_drift_on * self.drift_kappa_weight *
                            _drift_opportunity * rew_drift_kappa),
            "drift_throttle": _drift_on * self.drift_throttle_weight * rew_drift_throttle,
            # throttle and countersteer are ELIGIBILITY gates in the joint term now, not
            # standalone rewards: a positive standalone throttle reward pays full throttle
            # even with no drift at all, which is one of the ways the policy escaped into
            # the forward-wheelspin basin.
            "drift_counter": _drift_on * self.drift_counter_weight * rew_drift_counter,
            # JOINT drift bonus: a PRODUCT of soft gates, so it pays only when every
            # element of the drift is present at once.  A sum can always be collected one
            # term at a time -- rear slip without sideslip, throttle without slip -- and
            # that is exactly the basin the policy has been escaping into.
            "drift_joint": _drift_on * self.drift_joint_weight * rew_drift_joint,
            "beta": self.beta_weight * rew_beta,
            "lowspeed": 0.1 * rew_lowspeed,
            "sf": 0.005 * rew_sf,
            "kappa": self.kappa_weight * rew_kappa,
            "speed": self.speed_weight * rew_speed,
            "yaw": self.yaw_weight * rew_yaw,
            "alive": torch.full_like(rew_pos, self.alive_bonus),
            # without an explicit terminal cost most shaping terms are negative, so
            # braking into termination was genuinely profitable (the review caught
            # that rew_final was computed and then dropped)
            "terminal": self.terminal_cost * rew_final,
        }
        rew = torch.zeros_like(rew_pos)
        for term in terms.values():
            rew = rew + term
        # diagnostics: running mean of every reward component (read by eval.py)
        with torch.no_grad():
            self.rew_stats = {k: float(t.detach().mean()) for k, t in terms.items()}
        return rew

    def done(self):
        return self.is_done

    def reset(self):
        if self._refreshable_reference:
            self.path_generation.fill_(self.current_generation)
        self._track_pending = True
        self._sensor_pending.fill_(True)
        self._physics_advanced = False
        super().reset()
        self.progress = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.step_progress = torch.zeros(self.n, dtype=torch.int16, device=self.device)
        self.travel = torch.zeros(self.n, device=self.device)
        self.win_back = 20
        self.search_offsets = torch.arange(-self.win_back, 20, device=self.device)
        self._stall_anchor.copy_(self.s[:, :2])
        self.no_progress_count = torch.zeros(self.n, dtype=torch.int32, device=self.device)
        self.is_done = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        self._now = None
        self._prev = None
        # super().reset() puts every car at the plant's zero state, so without this the
        # FIRST episode of every environment started from rest instead of from the
        # equilibrium spawn the task advertises.  reset_done_envs() does the spawning,
        # so mark everything done and let it do its job.
        self.is_done.fill_(1)
        self._track_pending = True
        self.reset_done_envs()
        return self.obs()

    def info(self):
        info_dict = {
            "time_outs": (self.is_done == 2)
        }
        return info_dict

    def _refresh_reset_diagnostics(self, indices):
        """Install the exact kinematics and neutral one-sample IMU at a respawn.

        The terminal physics step has already produced valid diagnostics for all
        surviving environments.  Re-evaluating tire forces for the entire batch
        merely to populate the first observation of a few respawned cars cost
        16--18 ms at 262k environments.  A respawn is a teleport, so acceleration
        at that mathematical instant is undefined; use a neutral IMU sample and
        exact pose/speed/beta/wheel kinematics.  The first real physics substep
        replaces every field with the full plant solution before any new reward.
        """
        if indices.numel() == 0:
            return
        state = self.s[indices]
        psi = state[:, 2]
        speed = torch.hypot(state[:, 3], state[:, 4])
        beta = torch.atan2(state[:, 4], state[:, 3]) - psi
        beta = torch.atan2(torch.sin(beta), torch.cos(beta))
        wheel_surface = state[:, 6:10] * self.wheel_radius[indices]

        # Clear stale terminal diagnostics only on the respawned rows.
        if self.es is None:
            # Direct reset_done_envs() is a supported lightweight test path even
            # before reset(); the production reset already owns this buffer.
            self.es = torch.zeros((self.n, 56), device=self.device, dtype=self.s.dtype)
        self.es[indices] = 0.0
        self.es[indices, 0:2] = state[:, 3:5]
        self.es[indices, 2] = state[:, 5]
        self.es[indices, 6] = speed
        self.es[indices, 7] = beta
        self.es[indices, 8:12] = wheel_surface

        if self.extras is None:
            return
        for key in ("ax", "ay", "yaw_acc"):
            if key in self.extras:
                self.extras[key][indices] = 0.0
        for key in ("sx", "sy", "slip", "alpha", "mux", "muy",
                    "fx_body", "fy_body", "kappa_power"):
            if key in self.extras:
                self.extras[key][indices] = 0.0
        if "vx_wheel" in self.extras:
            self.extras["vx_wheel"][indices] = wheel_surface
        if "vy_wheel" in self.extras:
            self.extras["vy_wheel"][indices] = 0.0
        if "fz" in self.extras:
            mass = self.plant.col("m")[indices]
            gravity = self.plant.col("g")[indices]
            lf = self.plant.col("lF")[indices]
            lr = self.plant.col("lR")[indices]
            front = mass * gravity * lr / (2.0 * (lf + lr))
            rear = mass * gravity * lf / (2.0 * (lf + lr))
            self.extras["fz"][indices] = torch.stack(
                (front, front, rear, rear), dim=1)

    def reset_done_envs(self):
        """Only reset envs that are already done.

        No host/device synchronisation here: every op below is a no-op for an
        empty selection, so the ``any()`` check that used to guard it (and cost
        one CUDA sync per control step) is not needed.
        """
        is_done = self.is_done.bool()
        done_idx = is_done.nonzero(as_tuple=False).squeeze(1)
        size = done_idx.numel()
        if size == 0:
            return
        if self._streaming_reference:
            from utils.random_reference import generate_reference_chunk
            initial = torch.zeros((size, 4), device=self.device,
                                  dtype=self._rolling_reference.roads.dtype)
            initial[:, 3] = (torch.rand(size, device=self.device)*2-1)*.09
            options = self._rolling_reference.options
            first = generate_reference_chunk(initial, **options)
            second = generate_reference_chunk(first[:, -1], **options)
            self._rolling_reference.roads[done_idx] = torch.cat((first, second[:, 1:]), dim=1)
            self._rolling_reference.offset[done_idx] = 0

        # a deferred refresh may become installable now that these episodes end
        if self._refreshable_reference:
            self._try_path_refresh(releasing=is_done)
            self.path_generation[done_idx] = self.current_generation

        self.step_count[done_idx] = 0
        if self._streaming_reference:
            self.path_id[done_idx] = done_idx
        elif self.num_paths > 1:
            self.path_id[done_idx] = torch.randint(0, self.num_paths, (size,), device=self.device)
        self.progress[done_idx] = torch.randint(
            self._max_spawn_progress + 1, (size,), device=self.device)
        self.step_progress[done_idx] = 0
        self.no_progress_count[done_idx] = 0
        self.travel[done_idx] = 0
        # periodically draw a fresh set of roads so the policy cannot memorise
        # a fixed bank (256 roads x 131072 envs would be heavily overfitted)
        if self._refreshable_reference and self.path_regen_every > 0:
            self._resets_since_regen += size
            if self._resets_since_regen >= self.path_regen_every:
                # never replace the [generation, path, w, ch] ring by a 3-D bank
                self.request_path_refresh()
                self._resets_since_regen = 0
        # Ackermann blend randomised per episode: 0.8 - 1.2 of the exact geometry
        if self.plant_name == "fast":
            self.plant.ackermann_scale[done_idx] = (
                self.ackermann_lo + (self.ackermann_hi - self.ackermann_lo)
                * torch.rand((size, 1), device=self.device))
        # sample the episode tyre FIRST: the equilibrium spawn below is grip
        # dependent and must see this episode's D, not the previous episode's
        self.randomize(done_idx)
        if self.plant_name == "fast" and hasattr(self.plant, "p"):
            # Encoder conversion must follow the episode's randomized rolling
            # radii.  Keeping the constructor-time radius here silently made
            # rF/rR randomization affect physics but not the actor observation.
            self.wheel_radius[done_idx, 0:2] = self.plant.col("rF")[done_idx, None]
            self.wheel_radius[done_idx, 2:4] = self.plant.col("rR")[done_idx, None]
        self._reset_sensor_model(done_idx)

        # A gradual state-coverage curriculum is essential for pedal control:
        # equilibrium rows expose a maintainable powered-slide state, while
        # legacy rows require the actor to enter it from ordinary rolling/rest.
        # Draw the mixture per respawn so both populations share one PPO batch.
        if self.equilibrium_reset_prob <= 0.0:
            equilibrium_mask = torch.zeros(size, dtype=torch.bool, device=self.device)
        elif self.equilibrium_reset_prob >= 1.0:
            equilibrium_mask = torch.ones(size, dtype=torch.bool, device=self.device)
        else:
            equilibrium_mask = (torch.rand(size, device=self.device) <
                                self.equilibrium_reset_prob)
        equilibrium_idx = done_idx[equilibrium_mask]
        legacy_idx = done_idx[~equilibrium_mask]
        self._last_equilibrium_reset[done_idx] = equilibrium_mask

        def gen_random_state(indices, mode):
            # The XCAR code spawned 20 % of the episodes at the path start at
            # rest (bernoulli mask multiplied into the state).  For a 1320 kg car
            # that is "enter a drift from standstill", which it cannot do yet and
            # which consumed a fifth of every batch; the equilibrium spawn below
            # is used for all episodes instead.
            legacy_active = torch.ones(len(indices), device=self.device)
            if mode == "legacy" and self.legacy_zero_prob > 0.0:
                legacy_active = (torch.rand(len(indices), device=self.device)
                                 >= self.legacy_zero_prob).to(torch.float32)
                # Exact XCAR reset semantics: zero-state samples begin at the
                # start of the reference rather than at a random progress index.
                self.progress[indices] *= legacy_active.bool()
            prog = self.progress[indices]
            waypoints = self.path_bank[
                self.path_generation[indices], self.path_id[indices], prog, :]

            x = waypoints[:, 0].clone()
            y = waypoints[:, 1].clone()
            veldir = waypoints[:, 2]

            kappa = waypoints[:, 3]

            x += 0.1 * self.length_scale * torch.randn_like(x)
            y += 0.1 * self.length_scale * torch.randn_like(y)

            if mode == "equilibrium":
                # Start on the drift equilibrium of the local reference: the
                # sideslip magnitude follows the reference sign, the speed and
                # the yaw rate follow the local curvature.  The XCAR paper
                # randomizes beta over [-1,1] rad and V over [0,3] m/s, which for
                # a 7.6x larger car means dropping it into arbitrary spins; the
                # policy then never sees the equilibrium it has to hold, so the
                # task becomes recovering from a spin rather than drifting.
                # the spawn sideslip is its own distribution, NOT the path beta
                # column (which is now only a legacy slot)

                deficit = (torch.rand_like(kappa) < self.spawn_deficit_frac)
                if self.speed_target_mode == "limit":
                    mu0 = (self.plant.col("D")[indices] *
                           self.plant.tyre_scale[indices, 0]).clamp(0.3, 3.0)
                    g0 = self.plant.col("g")[indices]
                    base_speed = (self.speed_margin * torch.sqrt(
                        mu0 * g0 / kappa.abs().clamp(min=1e-2))).clamp(
                            min=2.0, max=self.speed_cap)
                else:
                    base_speed = torch.full_like(kappa, self.spawn_speed)
                v_floor = torch.minimum(torch.full_like(base_speed, self.spawn_speed_min),
                                        base_speed)
                v_low = v_floor + (base_speed - v_floor) * torch.rand_like(kappa)
                v0 = torch.where(deficit, v_low, base_speed * torch.clamp(
                    1.0 + self.spawn_speed_noise * torch.randn_like(kappa), 0.6, 1.3))
                beta0 = self.spawn_beta_frac * self._reference_beta(
                    kappa.unsqueeze(1), v0, idx=indices).squeeze(1)
                # A car that starts from standstill has no sideslip and no countersteer:
                # carrying the drift trim into a rest start asks the policy to hold a state
                # the physics forbids at v = 0.  Fade both in over the first 3 m/s so the
                # spawn distribution covers rest through full drift continuously.
                _fade = (v0 / 3.0).clamp(0.0, 1.0)
                beta0 = beta0 * _fade
                beta0 = beta0 * torch.clamp(
                    1.0 + self.spawn_beta_noise * torch.randn_like(beta0), 0.4, 1.4)
                psi = veldir - beta0 + 0.1 * torch.randn_like(veldir)
                xd = v0 * torch.cos(beta0 + psi)
                yd = v0 * torch.sin(beta0 + psi)
                psid = v0 * kappa
            else:
                if self.speed_target_mode == "limit":
                    # Preserve XCAR's broad 0..V reset, but scale its upper end
                    # with the local friction-limit speed.  The fixed model-car
                    # range scales to only ~8.2 m/s and never reaches the known
                    # mu075 drift branch (CarSim: ~14.6 m/s), so a policy cannot
                    # learn to maintain a state absent from its reset support.
                    mu0 = (self.plant.col("D")[indices] *
                           self.plant.tyre_scale[indices, 0]).clamp(0.3, 3.0)
                    g0 = self.plant.col("g")[indices]
                    v_limit = (self.speed_margin * torch.sqrt(
                        mu0 * g0 / kappa.abs().clamp(min=1e-2))).clamp(
                            min=2.0, max=self.speed_cap)
                    random_V = torch.rand_like(kappa) * 1.10 * v_limit
                else:
                    random_V = (torch.rand(prog.shape, device=self.device) *
                                3 * self.speed_scale)
                if self.drift_mode and self.beta_ref_mode in ("adaptive_drift", "geometry_drift"):
                    # XCAR-style broad resets must agree with the task at each
                    # friction/curvature cell.  A fixed +/-0.4 rad reset on a
                    # high-grip cell whose valid target is zero created an
                    # impossible recovery distribution and made nearly every
                    # full-DR episode terminate.  Low-grip/tight/fast cells
                    # still span the complete requested drift attitude.
                    beta_cap = self._reference_beta(
                        kappa.unsqueeze(1), random_V, idx=indices).squeeze(1)
                    random_beta = torch.rand_like(random_V) * beta_cap
                else:
                    random_beta = (-torch.sign(kappa) * torch.rand(
                        prog.shape, device=self.device) * self.spawn_beta_mag)
                psi = veldir - random_beta
                psi += 0.1 * torch.randn_like(psi)
                xd = random_V * torch.cos(random_beta + psi)
                yd = random_V * torch.sin(random_beta + psi)
                if self.speed_target_mode == "limit":
                    # Once the reset speed is scaled to the full-size friction
                    # limit, yaw rate must be scaled jointly as r = V*kappa.
                    # Retaining XCAR's independent model-car [1,3] rad/s draw
                    # made the high-speed state kinematically inconsistent with
                    # its path before the actor issued a command.
                    psid = random_V * kappa
                else:
                    psid = (-torch.sign(random_beta) * (1 + torch.rand(
                        prog.shape, device=self.device) * 2) / self.speed_scale)

            x, y, psi, xd, yd, psid = map(lambda t: torch.unsqueeze(t, 1), [x, y, psi, xd, yd, psid])
            state = torch.cat([x, y, psi, xd, yd, psid], 1)
            if mode == "legacy":
                # Rest is zero velocity at the reference start, not zero world
                # pose: closed paths need not begin at (0, 0) or heading zero.
                resting = legacy_active == 0
                state[resting, 0:2] = waypoints[resting, 0:2]
                state[resting, 2] = veldir[resting]
                state[:, 3:6] *= legacy_active.unsqueeze(1)
            return state

        self.s[done_idx, :] = 0
        if equilibrium_idx.numel() > 0:
            self.s[equilibrium_idx, :6] = gen_random_state(
                equilibrium_idx, "equilibrium")
        if legacy_idx.numel() > 0:
            self.s[legacy_idx, :6] = gen_random_state(legacy_idx, "legacy")
        self._stall_anchor[done_idx] = self.s[done_idx, :2]
        self.u[done_idx, :] = 0
        self.u_cmd[done_idx, :] = 0
        if self.pedals and equilibrium_idx.numel() > 0:
            # A car teleported into a drift was, physically, already
            # countersteering and on the throttle.  The observation contains the
            # previous command, and a zero previous command at a 0.75 rad
            # sideslip spawn is inconsistent -- without this the policy has no
            # information at all at the first step (its recurrent state is
            # empty) and learns to lift off, which collapses the drift.
            # per-env curvature from the ring, then the map-B target (the helper
            # works on whole batches, so evaluate it for all envs and index)
            _di = equilibrium_idx
            kappa_all = self.path_bank[self.path_generation, self.path_id, self.progress, 3]
            beta_all = self._reference_beta(
                kappa_all.unsqueeze(1),
                torch.hypot(self.s[:, 3], self.s[:, 4])).squeeze(1)
            # the trim must use the ACTUAL spawn speed, not spawn_speed: with rest
            # starts in the distribution the two differ by up to 4.5 m/s.squeeze(1)
            # Trim commands measured with the solver (rounds 43-45): the steer needed to
            # hold a steady sideslip is about 2.4-2.6x that sideslip on an 8-12 m radius,
            # and the throttle needed is 1-3 %, NOT the 0.55 that used to sit here.
            # A 0.55 throttle on the corrected plant is a hard acceleration command, which
            # is why the car left the corridor in 2.1 s with no action at all.
            # the trim steer must fade with speed as well, or a rest start gets a full
            # countersteer while its sideslip is zero
            _v_now = torch.hypot(self.s[_di, 3], self.s[_di, 4])
            _fade_cmd = (_v_now / 3.0).clamp(0.0, 1.0)
            if self.drift_mode and equilibrium_idx.numel() > 0:
                # In drift mode the countersteer target is -0.25*sign(kappa) and the spawn must
                # be consistent with it.  The old 2.5*beta gave -1.0*sign(kappa) (clamped to the
                # steering limit), i.e. a full opposite lock, which the joint gate's countersteer
                # factor correctly rejected -- so the joint bonus was worth zero at the spawn.
                # At rest there is no drift attitude to countersteer.  Failing to apply the
                # same speed fade as the spawned beta taught the actor an impossible CarSim
                # initial condition: zero speed with a fixed opposite lock.
                self.u_cmd[_di, 0] = (-0.25 * torch.sign(kappa_all[_di]) * _fade_cmd *
                                      self.drift_spawn_trim_frac).clamp(
                    -self.max_steer, self.max_steer)
            else:
                self.u_cmd[_di, 0] = (2.5 * beta_all[_di] * _fade_cmd).clamp(
                    -self.max_steer, self.max_steer)
            # The throttle needed to HOLD a given speed is the one thing about the plant
            # that CarSim does not share with the GPU model: the GPU policy settles on
            # 0.004 because the trim supplies 0.02, and in CarSim that command lets the car
            # decelerate to a standstill.  Randomising the trim throttle over a wide range
            # forces the policy to close the speed loop from the wheel-speed channels
            # instead of memorising one value.
            _lo = float(getattr(self, "trim_throttle_lo", 0.0))
            _hi = float(getattr(self, "trim_throttle_hi", 0.25))
            if self.drift_mode:
                # a drift spawn must satisfy the joint gate's throttle condition from the
                # first step, otherwise the state is inside the drift region geometrically
                # but still scores zero and the policy gets no gradient
                _lo, _hi = 0.35, 0.70
            self.u_cmd[_di, 1] = (_lo + (_hi - _lo) * torch.rand(
                len(_di), device=self.device, dtype=self.u_cmd.dtype))
            if self.drift_mode:
                self.u_cmd[_di, 1] *= self.drift_spawn_trim_frac
        # the actor emits a CORRECTION on top of this trim, so store it
        self.u_trim = getattr(self, "u_trim", None)
        if self.u_trim is None:
            self.u_trim = torch.zeros_like(self.u_cmd)
        self.u_trim[done_idx] = self.u_cmd[done_idx]
        # the equilibrium spawn installs a physical steering command; keep the
        # actuator state consistent with it or the first steps fight the wheel
        if hasattr(self, "_steer_tau_env"):
            self._steer_state[done_idx] = self.u_cmd[done_idx, 0]
            self._steer_prev[done_idx] = self.u_cmd[done_idx, 0]
        if hasattr(self, "_pedal_state"):
            self._pedal_state[done_idx] = self.u_cmd[done_idx, 1:3]
        if hasattr(self, "_action_delay_buffer"):
            self._action_delay_buffer[:, done_idx] = self.u_cmd[done_idx].unsqueeze(0)
        if self.plant_name == "fast":
            self.plant.init_wheels_mask(self.s, done_idx)
            if self.drift_mode:
                # DRIFT SPAWN.  The joint drift gate pays 12 per step only when the rear is
                # already overslipping AND the sideslip is large AND correctly signed.  The
                # gates need |beta| > 0.18 while the policy was measured at 0.003-0.022, so
                # the gate was unreachable by gradient: to collect it the policy would first
                # have to do a destabilising transition it never takes.  Seeding the rear
                # axle with the target slip means the episode STARTS inside the drift
                # region, so the gradient exists from the first step and the policy's job is
                # to hold the state rather than to discover it.
                _spin = 1.0 + self.drift_spawn_spin_frac * self.drift_kappa_target
                # Scale the already initialized contact-velocity rolling speed.
                # Using total vehicle speed here silently injected Kappa even at
                # spin_frac=0 whenever the car had sideslip/yaw rate.
                self.s[equilibrium_idx, 8:10] = (
                    self.s[equilibrium_idx, 8:10] * _spin).clamp(min=0.0)
            self.plant.reset_guess(done_idx)
            if self.disturbance_param is not None:
                self.tyre_scale[done_idx] = 1.0
        self.is_done.zero_()
        if self.plant_name == "fast":
            # Refresh the extended (diagnostic / reward) state for the respawned
            # envs.  Without this the first reward after a reset is computed from
            # the previous episode's state, and any controller that reads ``es``
            # sees stale values on the spawn step.
            if size == self.n:
                # Preserve the historical, fully physical initial observation
                # for deterministic evaluation and whole-batch reset.  This is
                # paid once, not on the high-frequency partial-respawn path.
                self.extras = self._eval_extras(self.u_cmd)
                self.es = self.plant.package_extended(self.s, self.extras)
            else:
                self._refresh_reset_diagnostics(done_idx)
        # the next observation belongs to the new pose/path position, not to the
        # terminal tracking solution of the episode that just ended
        self._track_pending = True
        self._physics_advanced = False
        self._sensor_pending[done_idx] = True
        if hasattr(self, "_raw_action"):
            self._raw_action[done_idx] = self.u_cmd[done_idx]
        self._tracking_mask = is_done.clone()
        if not hasattr(self, "_projection_xy"):
            self._projection_xy = self.s[:, :2].clone()
        self._projection_xy[done_idx] = self.s[done_idx, :2]

    # ---------------------------------------------------------------- scenarios
    def capture_scenario(self):
        """Freeze everything needed to replay the episode: the ROADS themselves,

        not just their indices.  The ring may legally overwrite a slot later, so
        an index-only scenario would replay the state on a different road.
        """
        sel = self.path_bank[self.path_generation, self.path_id]
        scenario = {"path": sel.detach().cpu().numpy().copy(),
                "progress": self.progress.detach().cpu().numpy().copy(),
                "s": self.s.detach().cpu().numpy().copy(),
                "u_cmd": self.u_cmd.detach().cpu().numpy().copy()}
        if hasattr(self.plant, "_motor_torque_state"):
            scenario["motor_torque"] = self.plant._motor_torque_state.detach().cpu().numpy().copy()
        return scenario

    def apply_scenario(self, sc):
        """Replay a frozen scenario: install its roads, then its start states."""
        import numpy as _np
        if "path" in sc:
            roads = torch.as_tensor(_np.asarray(sc["path"]), device=self.device, dtype=self.path_bank.dtype)
            if roads.shape[0] != self.n:
                raise ValueError(
                    "scenario has %d environments but eval requested %d"
                    % (roads.shape[0], self.n))
            if self.n > self.num_paths:
                raise ValueError(
                    "exact replay needs n <= num_paths; got %d > %d"
                    % (self.n, self.num_paths))
            n = self.n
            self.path_bank[self.current_generation, :n].copy_(roads[:n])
            self._path_ds[self.current_generation, :n] = torch.linalg.vector_norm(
                torch.roll(roads[..., :2], -1, dims=1) - roads[..., :2], dim=-1).mean(dim=-1)
            self.path_id[:n] = torch.arange(n, device=self.device)
            self.path_generation[:n] = self.current_generation
        prog = torch.as_tensor(_np.asarray(sc["progress"]), device=self.device, dtype=self.progress.dtype)
        st = torch.as_tensor(_np.asarray(sc["s"]), device=self.device, dtype=self.s.dtype)
        n = min(prog.numel(), self.n)
        self.progress[:n] = prog[:n]
        self.s[:n] = st[:n]
        self._projection_xy = self.s[:, :2].clone()
        self._tracking_mask = None
        self._sensor_pending.fill_(True)
        self._sensor_needs_fill.fill_(True)
        self.u[:n] = 0
        if "u_cmd" in sc:
            uc = torch.as_tensor(_np.asarray(sc["u_cmd"]), device=self.device, dtype=self.u_cmd.dtype)
            self.u_cmd[:n] = uc[:n]
        else:
            self.u_cmd[:n] = 0
        self.step_count[:n] = 0
        self.step_progress[:n] = 0
        self.is_done[:n] = 0
        self.no_progress_count[:n] = 0
        self.travel[:n] = 0
        self._stall_anchor[:n] = self.s[:n, :2]
        if self.plant_name == "fast":
            self.plant.reset_guess(torch.ones(self.n, dtype=torch.bool, device=self.device))
            self.extras = self._eval_extras(self.u_cmd)
            self.es = self.plant.package_extended(self.s, self.extras)
        self._track_pending = True
        self._physics_advanced = False
        if "motor_torque" in sc and hasattr(self.plant, "_motor_torque_state"):
            self.plant._motor_torque_state.copy_(torch.as_tensor(
                sc["motor_torque"], device=self.device, dtype=self.s.dtype))
        return self.obs()

    def step(self, action, **kwargs):
        self._raw_action = action.detach().clone()
        # Physics first, THEN respawn: the action was chosen from the terminal
        # observation, so it has to act on the terminal state.  Resetting before
        # the step (the previous order) applied that action to the freshly
        # spawned car, which injected an arbitrary command into the start of
        # every episode and made the reward history cross episode boundaries.
        self._track_pending = True   # the tracking update belongs to this step
        self._tracking_mask = None
        self._physics_advanced = True   # only a real physics step may advance travel
        self._sensor_pending.fill_(True)
        obs, reward, done, info = super().step(action, **kwargs)
        # ``done`` aliases ``self.is_done``; the respawn below clears it, so keep
        # a copy of the terminal flags before resetting
        done = done.clone()
        if self.train:
            newly = (done != 0)
            if bool(newly.any()):
                # the observation refresh below recomputes the failure masks for
                # the new spawn, so keep the terminal attribution
                keep_reason = getattr(self, "_term_reason", None)
                keep_masks = getattr(self, "_term_masks", None)
                self.reset_done_envs()
                fresh = self.obs()
                if keep_reason is not None:
                    self._term_reason = keep_reason
                if keep_masks is not None:
                    self._term_masks = keep_masks
                if isinstance(obs, dict):
                    obs = {k: torch.where(newly.unsqueeze(-1), fresh[k], obs[k])
                           for k in obs}
                else:
                    obs = torch.where(newly.unsqueeze(-1), fresh, obs)
                # the first reward of the new episode must not see the previous
                # episode's state through the difference terms
                if self._prev is not None and self._now is not None:
                    for k in self._now:
                        if k in self._prev and torch.is_tensor(self._prev[k]):
                            m = newly.view(-1, *([1] * (self._prev[k].dim() - 1)))
                            self._prev[k] = torch.where(m, self._now[k], self._prev[k])
        self.update_recent_obs(obs)
        return obs, reward, done, info

