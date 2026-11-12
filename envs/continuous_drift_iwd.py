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

        # env parameters
        self.max_steps = int(round(30.0 / self.dt))
        self.quiet = kwargs.get("quiet", False)
        self.smooth_scale = 0.01 / self.dt   # keep reward per unit time constant

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

        n_ctrl = 3 if self.pedals else 4
        if not self.pedals:
            # the 27-channel actor contract assumes the pedal triple; the legacy
            # wheel-speed mode would make it 28 channels
            raise ValueError("continuous_drift_iwd uses the 27-channel pedals actor")
        # the 27-channel actor contract is fixed: privileged plant quantities
        # (beta, V, vx, vy) are only ever exposed through the asymmetric critic
        self.obs_privileged = False
        self.num_states = (1 + 3 + n_ctrl + 4 + 4) + 12
        # training; rl_games adds its own running normalisation on top.
        wheel_scale = 3.0 / self.speed_ref
        ctrl_scale = ([1.0 / self.max_steer, 1.0, 1.0] if self.pedals
                      else [1.0 / self.max_steer] + [wheel_scale] * 4)
        scale = [0.5, 0.1, 0.1, 0.1]          # r, ax, ay, az
        scale += ctrl_scale
        scale += [wheel_scale] * 4
        scale += [1.0, 1.0, 1.0, 0.05]        # e_pos, e_dir, e_kappa, travel/length_scale
        for _ in range(3):
            scale += [1.0 / self.length_scale, 1.0 / self.length_scale, 1.0, 1.0]
        self.obs_scale = torch.tensor(scale, device=self.device, dtype=torch.float32)
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
        self.spawn_speed = float(kwargs.get("spawn_speed", 6.5))
        self.spawn_beta_frac = float(kwargs.get("spawn_beta_frac", 1.0))
        self.spawn_beta_noise = float(kwargs.get("spawn_beta_noise", 0.10))
        self.spawn_speed_noise = float(kwargs.get("spawn_speed_noise", 0.10))
        # Speed-deficit curriculum: a fraction of the spawns start below the
        # reference speed so the policy has to LEARN TO REGAIN it.  The dominant
        # failure at the moment is "4 m/s -> 1 m/s in a second and never back".
        self.spawn_beta_mag = float(kwargs.get("spawn_beta_mag", 0.15))
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
        # >1 biases the friction draw towards the low end while keeping the range
        self.plant.grip_bias = float(kwargs.get("grip_bias", 1.0))

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
            # NOTE: an in-place regeneration would change the road underneath
            # environments that are still driving on it, so it is OFF by default
            # until the multi-generation ring exists.
            self.path_regen_every = int(kwargs.get("path_regen_every", 0))
            self._resets_since_regen = 0
        # circle trajectory
        elif ref_mode == "circle":
            self.num_waypoints = 2512      # one waypoint per 0.005m
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
        self.path_bank = self.path_bank.unsqueeze(0).repeat(3, 1, 1, 1)
        self.current_generation = 0
        self.path_generation = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self._path_refresh_pending = False
        self.path_id = torch.zeros(self.n, dtype=torch.long, device=self.device)
        if self.num_paths > 1:
            self.path_id.random_(0, self.num_paths)

        if not self.train:
            basename = time.strftime("%Y%m%d-%H%M%S")
            Path("data").mkdir(parents=True, exist_ok=True)
            with open(os.path.join("data", "traj-" + basename + ".csv"), 'w') as f:
                writer = csv.writer(f, delimiter=',')
                for i in range(self.num_waypoints):
                    x, y, psi = self.path_bank[0, i, 0].item(), self.path_bank[0, i, 1].item(), self.path_bank[0, i, 2].item()
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
                preview_guard = 600.0 * self.preview_scale * self.waypoint_ds
            reserve = (1.15 * guard_speed * self.max_steps * self.dt
                       + preview_guard + 2.0 * self.offpath_limit)
            usable = max((self.num_waypoints - 1) * self.waypoint_ds - reserve, 0.0)
            self._max_spawn_progress = min(
                int(usable / self.waypoint_ds), self.num_waypoints - 1)
        self.preview_time = None if pt is None else torch.as_tensor(
            [float(x) for x in (pt if not isinstance(pt, str) else pt.split(","))],
            device=self.device, dtype=torch.float32)
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
        return out

    def update_recent_obs(self, obs):
        self.recent_obs[1] = self.recent_obs[0]
        self.recent_obs[0] = obs

    def waypoints_at(self, ind):
        """Reference waypoints: (n, k, 4) from each environment's own generation."""
        return self.path_bank[
            self.path_generation.unsqueeze(1), self.path_id.unsqueeze(1), ind
        ]

    def _try_path_refresh(self, releasing=None):
        """Install a pending generation when its target slot is unused.

        ``releasing`` marks environments whose episode ends in this reset call;
        they no longer count as users of their old generation.
        """
        if not self._is_hybrid or not self._path_refresh_pending:
            return False
        target = (self.current_generation + 1) % 3
        in_use = self.path_generation == target
        if releasing is not None:
            in_use = in_use & ~releasing
        if bool(in_use.any()):
            return False
        fresh = self._scale_waypoints(
            self.trajectory_generator.generate_random_waypoints(self.num_paths))
        if fresh.shape != self.path_bank[target].shape:
            raise RuntimeError("refreshed bank shape %s != %s"
                               % (tuple(fresh.shape), tuple(self.path_bank[target].shape)))
        self.path_bank[target].copy_(fresh)
        self.current_generation = target
        self._path_refresh_pending = False
        return True

    def request_path_refresh(self):
        """Ask for one fresh road generation (the observer calls this per N epochs)."""
        if not self._is_hybrid:
            return False
        self._path_refresh_pending = True
        return self._try_path_refresh()

    def find_nearest_waypoint(self, x, y, veldir, beta, kappa):
        # Local projection around the last known progress.  The window is
        # symmetric because a car that has been pushed off the line must be able
        # to re-acquire a reference point slightly BEHIND the last one; the old
        # forward-only window plus the "freeze progress while off path" rule made
        # the target stale exactly inside the recovery band (progress froze at
        # 3.8 m of lateral error while the failure corridor is 10.6 m).
        offs = self.search_offsets
        ind = (self.progress.unsqueeze(1) + offs).clamp_(0, self.num_waypoints - 1)
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
        self.step_progress = (best - self.win_back).to(torch.int16)
        self.progress = (self.progress + self.step_progress.long()).clamp_(0, self.num_waypoints - 1)
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
        if speed.dim() < kappa.dim():
            speed = speed.reshape(-1, *((1,) * (kappa.dim() - speed.dim())))
        beta_geo = torch.clamp(
            self.beta_scale * (self.alpha_peak + lr * kap), max=self.beta_max)
        util = (speed.square() * kap / (mu * gg)).clamp(0.0, 1.0)
        factor = (util / self.beta_util_ref).clamp(self.beta_util_lo, 1.0)
        return -torch.sign(kappa) * beta_geo * factor

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

        The reference/error part is unchanged in *meaning* from XCAR: preview
        points are expressed in the course aligned frame (position, course
        error, curvature and the reference sideslip sign), and the tracking
        errors are the lateral error, the course error and the curvature error.
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
        wheel_speed = self.s[:, 6:10] * self.wheel_radius       # encoder measurement
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
        if getattr(self, "_track_pending", True):
            self.find_nearest_waypoint(x, y, veldir, beta, kappa_est)
            self._track_pending = False

        # --- reference sideslip: reward / critic only ---------------------
        # The target is the steady-drift sideslip that the local curvature and the
        # available friction imply.  util = v^2|kappa| / (mu g) is the fraction of
        # the available lateral grip the corner actually asks for, so a gentle
        # corner at low speed asks for almost no sideslip and a corner at the
        # friction limit asks for beta_max.  It uses the TRUE speed and the TRUE
        # grip, so it is privileged information and never reaches the actor.
        if self.beta_ref_mode in ("physics", "constant"):
            self.beta_ref = self._reference_beta(self.kappa_ref.unsqueeze(1), v).squeeze(1)
            self.beta_diff = torch.atan2(torch.sin(beta - self.beta_ref),
                                         torch.cos(beta - self.beta_ref))
        angle_diff = self.angle_diff
        signed_pos_diff = self.signed_pos_diff
        beta_diff = self.beta_diff
        beta_ref = self.beta_ref
        kappa_diff = self.kappa_diff
        kappa_ref = self.kappa_ref

        # look ahead 3m (scaled with the reference geometry)
        # Preview points.  With --preview-time they are specified as TIME
        # headways and converted to an arc length with the current speed, so the
        # look-ahead in seconds stays constant from 4 to 10 m/s (a fixed metric
        # preview is 2.3 s at 4 m/s but only 0.85 s at 10 m/s).
        if self.preview_time is not None:
            ds_wp = self.waypoint_ds
            # use the ENCODER-derived speed, not the true speed: the preview
            # positions are actor-visible, so the true speed must not leak into
            # them through the time headway.
            offs = (v_est.unsqueeze(1) * self.preview_time.unsqueeze(0) / ds_wp).round().long()
            offs = offs.clamp(min=1, max=self.num_waypoints // 4)
            if self.train:
                la_inds = [(self.progress.unsqueeze(1) + offs[:, k:k + 1]).clamp(0, self.num_waypoints - 1)
                           for k in range(3)]
            else:
                la_inds = [((self.progress.unsqueeze(1) + offs[:, k:k + 1]) % self.num_waypoints)
                           for k in range(3)]
        elif self.train:
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


        def convert(waypoints):
            """Cast a waypoint into the current course aligned frame."""
            x1 = waypoints[:, 0]
            y1 = waypoints[:, 1]
            veldir1 = waypoints[:, 2]
            kappa1 = waypoints[:, 3]
            theta = torch.atan2(y1 - y, x1 - x) - veldir
            l = torch.hypot(y1 - y, x1 - x)
            x2 = l * torch.cos(theta)
            y2 = l * torch.sin(theta)
            veldir2 = veldir1 - veldir
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
        # Sign: positive steering turns right, so the lateral offset is negated.
        pp = la_converted[1]
        ld2 = (pp[:, 0] ** 2 + pp[:, 1] ** 2).clamp(min=1e-3)
        self.delta_pp = -torch.atan2(2.0 * self.wheelbase * pp[:, 1], ld2)

        # grace period instead of the original "progress >= 5" guard: the progress
        # based guard never fires for a car that stops on the first waypoints
        fail_npc = self.no_progress_count >= self.no_progress_limit
        fail_course = torch.abs(angle_diff) > self.course_limit
        fail_off = torch.abs(signed_pos_diff) > self.offpath_limit
        failed = (self.step_count > 10) & (fail_npc | fail_course | fail_off)
        # snapshot of the terminal conditions, so a caller can attribute the
        # failure after the environment has already respawned
        self._term_masks = torch.stack((fail_npc, fail_course, fail_off), dim=1)
        if self.train:
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

        self._prev = self._now
        self._now = {
            "beta": beta, "v": v, "kappa": r / (v + 1e-6),   # true curvature: reward only
            "r": r, "r_ref": v_est * kappa_ref,
            "vflx": vflx, "vfrx": vfrx,
            "delta": self.u_cmd[:, 0],
            "wheel": wheel_speed,
            "angle_diff": angle_diff, "signed_pos_diff": signed_pos_diff,
            "e_pos": signed_pos_diff / self.length_scale,
            "beta_diff": beta_diff, "kappa_ref": kappa_ref,
            "prog": self.step_progress, "failed": failed.float(),
        }

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
        scalars += [e_pos, angle_diff, e_kappa, self.travel / self.length_scale]
        obs_list = [t.unsqueeze(1) for t in scalars]
        obs_list.extend(la_converted)
        obs = torch.cat(obs_list, 1) * self.obs_scale
        if self.obs_privileged:
            priv = torch.stack([self.es[:, 7], self.es[:, 6], self.es[:, 3], self.es[:, 4]], dim=1)
            obs = torch.cat([obs, priv * self.priv_concat_scale], dim=1)
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
            priv = [
                beta, v, r, self.s[:, 3], self.s[:, 4],
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
            return {"obs": obs, "states": states}
        return obs

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
        rew_beta = - (beta - beta_target) ** 2
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
            mu = (self.plant.col("D") * self.plant.tyre_scale.squeeze(1)).clamp(0.3, 1.8)
            kap = torch.maximum(self.kappa_ref.abs(),
                                  self.kappa_preview.abs().amax(dim=1)).clamp(min=1e-2)
            v_tgt = (self.speed_margin * torch.sqrt(mu * 9.81 / kap)).clamp(2.0, self.speed_cap)
        else:
            v_tgt = self.speed_target
        if self.speed_target_mode == "limit":
            # Two-sided: the limit target is a GOAL, so over-driving it is as
            # wrong as under-driving.  With a floor-only term the policy kept
            # 9.7 m/s on high grip against an 8.5 m/s target and failed 87 % of
            # the episodes by running out of the corridor.
            # TWO-SIDED: exceeding the limit target is as wrong as falling short,
            # otherwise the policy over-drives at high grip and leaves the corridor
            rew_speed = - ((v - v_tgt) / v_tgt) ** 2
        else:
            rew_speed = - ((v - v_tgt) / v_tgt) ** 2

        # progress is capped at the waypoints the reference speed covers in one
        # step, exactly as the original XCAR reward caps it: going faster than the
        # reference must not earn extra progress reward (that was the over-speed
        # incentive that pushed the car out of the corridor at high grip).
        _rate = v_tgt * self.dt / self.waypoint_ds
        if not torch.is_tensor(_rate):
            _rate = torch.full_like(v, float(_rate))
        rew_prog = torch.minimum(_prog_raw.clamp(min=0.0), _rate)
        rew_final = -is_failed

        terms = {
            "pos": self.pos_weight * rew_pos,
            "recover": self.recover_weight * rew_recover,
            "pure_pursuit": self.pp_weight * rew_pp,
            "dir": self.dir_weight * rew_dir,
            # the speed target already defines how fast to go; a raw distance
            # reward double-counts it and always favours going faster, which is
            # what pushed the car past the corridor at high grip
            "prog": self.prog_weight * rew_prog,
            "smooth": 0.015 * rew_smooth,
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
        if getattr(self, "_is_hybrid", False):
            self.path_generation.fill_(self.current_generation)
        self._track_pending = True
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
        # super().reset() already called obs() with stale task counters
        self._track_pending = True
        return self.obs()

    def info(self):
        info_dict = {
            "time_outs": (self.is_done == 2)
        }
        return info_dict

    def reset_done_envs(self):
        """Only reset envs that are already done.

        No host/device synchronisation here: every op below is a no-op for an
        empty selection, so the ``any()`` check that used to guard it (and cost
        one CUDA sync per control step) is not needed.
        """
        is_done = self.is_done.bool()
        size = torch.sum(is_done)

        # a deferred refresh may become installable now that these episodes end
        if getattr(self, "_is_hybrid", False):
            self._try_path_refresh(releasing=is_done)
            self.path_generation[is_done] = self.current_generation

        self.step_count[is_done] = 0
        if self.num_paths > 1:
            self.path_id[is_done] = torch.randint(0, self.num_paths, (size,), device=self.device)
        self.progress[is_done] = torch.randint(
            self._max_spawn_progress + 1, (size,), device=self.device)
        self.step_progress[is_done] = 0
        self.no_progress_count[is_done] = 0
        self.travel[is_done] = 0
        # periodically draw a fresh set of roads so the policy cannot memorise
        # a fixed bank (256 roads x 131072 envs would be heavily overfitted)
        if getattr(self, "_is_hybrid", False) and getattr(self, "path_regen_every", 0) > 0:
            self._resets_since_regen += int(size.item())
            if self._resets_since_regen >= self.path_regen_every:
                # never replace the [generation, path, w, ch] ring by a 3-D bank
                self.request_path_refresh()
                self._resets_since_regen = 0
        # Ackermann blend randomised per episode: 0.8 - 1.2 of the exact geometry
        if self.plant_name == "fast":
            n_ack = int(torch.sum(is_done).item())
            if n_ack:
                self.plant.ackermann_scale[is_done] = (self.ackermann_lo + (self.ackermann_hi - self.ackermann_lo) * torch.rand((n_ack, 1), device=self.device))
        # sample the episode tyre FIRST: the equilibrium spawn below is grip
        # dependent and must see this episode's D, not the previous episode's
        self.randomize(is_done)

        def gen_random_state(indices):
            # The XCAR code spawned 20 % of the episodes at the path start at
            # rest (bernoulli mask multiplied into the state).  For a 1320 kg car
            # that is "enter a drift from standstill", which it cannot do yet and
            # which consumed a fifth of every batch; the equilibrium spawn below
            # is used for all episodes instead.
            prog = self.progress[indices]
            waypoints = self.path_bank[
                self.path_generation[indices], self.path_id[indices], prog, :]

            x = waypoints[:, 0]
            y = waypoints[:, 1]
            veldir = waypoints[:, 2]

            kappa = waypoints[:, 3]

            x += 0.1 * self.length_scale * torch.randn_like(x)
            y += 0.1 * self.length_scale * torch.randn_like(y)

            if self.spawn_mode == "equilibrium":
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
                v_low = self.spawn_speed_min + (self.spawn_speed - self.spawn_speed_min) * torch.rand_like(kappa)
                v0 = torch.where(deficit, v_low, self.spawn_speed * torch.clamp(
                    1.0 + self.spawn_speed_noise * torch.randn_like(kappa), 0.6, 1.3))
                beta0 = self.spawn_beta_frac * self._reference_beta(
                    kappa.unsqueeze(1), v0, idx=indices).squeeze(1)
                beta0 = beta0 * torch.clamp(
                    1.0 + self.spawn_beta_noise * torch.randn_like(beta0), 0.4, 1.4)
                psi = veldir - beta0 + 0.1 * torch.randn_like(veldir)
                xd = v0 * torch.cos(beta0 + psi)
                yd = v0 * torch.sin(beta0 + psi)
                psid = v0 * kappa
            else:
                random_V = torch.rand(prog.shape, device=self.device) * 3 * self.speed_scale
                random_beta = -torch.sign(kappa) * torch.rand(prog.shape, device=self.device) * self.spawn_beta_mag
                psi = veldir - random_beta
                psi += 0.1 * torch.randn_like(psi)
                xd = random_V * torch.cos(random_beta + psi)
                yd = random_V * torch.sin(random_beta + psi)
                psid = -torch.sign(random_beta) * (1 + torch.rand(prog.shape, device=self.device) * 2) / self.speed_scale

            x, y, psi, xd, yd, psid = map(lambda t: torch.unsqueeze(t, 1), [x, y, psi, xd, yd, psid])
            return torch.cat([x, y, psi, xd, yd, psid], 1)

        self.s[is_done, :] = 0
        self.s[is_done, :6] = gen_random_state(is_done.nonzero().squeeze(-1))
        self._stall_anchor[is_done] = self.s[is_done, :2]
        self.u[is_done, :] = 0
        self.u_cmd[is_done, :] = 0
        if self.pedals and self.spawn_mode == "equilibrium":
            # A car teleported into a drift was, physically, already
            # countersteering and on the throttle.  The observation contains the
            # previous command, and a zero previous command at a 0.75 rad
            # sideslip spawn is inconsistent -- without this the policy has no
            # information at all at the first step (its recurrent state is
            # empty) and learns to lift off, which collapses the drift.
            # per-env curvature from the ring, then the map-B target (the helper
            # works on whole batches, so evaluate it for all envs and index)
            _di = is_done.nonzero(as_tuple=False).squeeze(1)
            kappa_all = self.path_bank[self.path_generation, self.path_id, self.progress, 3]
            beta_all = self._reference_beta(kappa_all.unsqueeze(1), self.spawn_speed).squeeze(1)
            self.u_cmd[_di, 0] = (0.55 * beta_all[_di]).clamp(-self.max_steer, self.max_steer)
            self.u_cmd[_di, 1] = 0.55
        if self.plant_name == "fast":
            self.plant.init_wheels_mask(self.s, is_done)
            self.plant.reset_guess(is_done)
            if self.disturbance_param is not None:
                self.tyre_scale[is_done] = 1.0
        self.is_done[:] = 0
        if self.plant_name == "fast":
            # Refresh the extended (diagnostic / reward) state for the respawned
            # envs.  Without this the first reward after a reset is computed from
            # the previous episode's state, and any controller that reads ``es``
            # sees stale values on the spawn step.
            self.es = self.plant.extended_state(self.s, self.u_cmd)
        # the next observation belongs to the new pose/path position, not to the
        # terminal tracking solution of the episode that just ended
        self._track_pending = True
        self._physics_advanced = False

    # ---------------------------------------------------------------- scenarios
    def capture_scenario(self):
        """Freeze everything needed to replay the episode: the ROADS themselves,

        not just their indices.  The ring may legally overwrite a slot later, so
        an index-only scenario would replay the state on a different road.
        """
        sel = self.path_bank[self.path_generation, self.path_id]
        return {"path": sel.detach().cpu().numpy().copy(),
                "progress": self.progress.detach().cpu().numpy().copy(),
                "s": self.s.detach().cpu().numpy().copy(),
                "u_cmd": self.u_cmd.detach().cpu().numpy().copy()}

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
            self.path_id[:n] = torch.arange(n, device=self.device)
            self.path_generation[:n] = self.current_generation
        prog = torch.as_tensor(_np.asarray(sc["progress"]), device=self.device, dtype=self.progress.dtype)
        st = torch.as_tensor(_np.asarray(sc["s"]), device=self.device, dtype=self.s.dtype)
        n = min(prog.numel(), self.n)
        self.progress[:n] = prog[:n]
        self.s[:n] = st[:n]
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
            self.es = self.plant.extended_state(self.s, self.u_cmd)
        self._track_pending = True
        self._physics_advanced = False
        return self.obs()

    def step(self, action, **kwargs):
        # Physics first, THEN respawn: the action was chosen from the terminal
        # observation, so it has to act on the terminal state.  Resetting before
        # the step (the previous order) applied that action to the freshly
        # spawned car, which injected an arbitrary command into the start of
        # every episode and made the reward history cross episode boundaries.
        self._track_pending = True   # the tracking update belongs to this step
        self._physics_advanced = True   # only a real physics step may advance travel
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
