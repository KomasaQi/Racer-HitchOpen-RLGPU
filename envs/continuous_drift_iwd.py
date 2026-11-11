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
        self.no_progress_limit = max(int(round(0.2 * self.speed_scale / self.dt)), 2)
        # Reference sideslip magnitude.  XCAR asks for +-1 rad (57 deg); with a
        # 26 deg steering limit a 57 deg sustained drift needs ~46 deg of
        # countersteer, which this car does not have, so the target is scaled to
        # what the vehicle can hold (--beta-scale 1.0 restores the XCAR value).
        self.beta_scale = float(kwargs.get("beta_scale", 1.0))
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
        self.beta_gate_scale = float(kwargs.get("beta_gate_scale", 0.8))
        self.recover_weight = float(kwargs.get("recover_weight", 0.0))
        self.kappa_weight = float(kwargs.get("kappa_weight", 0.15))
        self.pp_weight = float(kwargs.get("pp_weight", 0.0))

        n_ctrl = 3 if self.pedals else 4
        # --obs-privileged appends the plant quantities the real car cannot
        # measure (beta, V, vx, vy) to the actor observation, which turns the
        # task back into the original XCAR formulation and gives the upper bound
        # the restricted observation is compared against.
        self.obs_privileged = bool(kwargs.get("obs_privileged", False))
        self.num_states = (1 + 3 + n_ctrl + 4 + 4) + 15 + (4 if self.obs_privileged else 0)
        # Fixed per channel scaling so every channel is O(1) at the start of
        # training; rl_games adds its own running normalisation on top.
        wheel_scale = 3.0 / self.speed_ref
        ctrl_scale = ([1.0 / self.max_steer, 1.0, 1.0] if self.pedals
                      else [1.0 / self.max_steer] + [wheel_scale] * 4)
        scale = [0.5, 0.1, 0.1, 0.1]          # r, ax, ay, az
        scale += ctrl_scale
        scale += [wheel_scale] * 4
        scale += [1.0, 1.0, 1.0, 0.1]         # e_pos, e_dir, e_kappa, progress
        for _ in range(3):
            scale += [1.0 / self.length_scale, 1.0 / self.length_scale, 1.0, 1.0, 1.0]
        self.obs_scale = torch.tensor(scale, device=self.device, dtype=torch.float32)
        self.priv_concat_scale = torch.tensor([1.0, 0.25, 0.25, 0.25], device=self.device)
        # privileged critic channels: beta, v, r, vx, vy, kappa, kappa_ref, r_ref,
        # beta_ref, e_pos, e_dir, ax, ay, az, wheels, slip, fz, fy, commands
        pscale = ([1.0, 0.1, 0.5, 0.1, 0.1, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,
                   0.1, 0.1, 0.1]
                  + [wheel_scale] * 4 + [0.5] * 4 + [0.3] * 4 + [0.3] * 4
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
        # Speed regulation around the drift operating speed.  XCAR's rspeed only
        # punishes V < 0.5 m/s, which at our scale (drift speed ~6.5 m/s) leaves
        # "slow down until the drift collapses" as a reward improving local
        # optimum; this term closes that hole.
        self.speed_target = float(kwargs.get("speed_target", self.spawn_speed))
        self.speed_weight = float(kwargs.get("speed_weight", 0.0))
        self.speed_hinge = bool(kwargs.get("speed_hinge", False))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
        self.state_space = self.observation_space
        if self.asymmetric:
            self.num_priv = 48
            self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_priv,))

        dummy_obs = torch.zeros((self.n, self.num_states), device=self.device)
        self.recent_obs = [dummy_obs, dummy_obs]  # most recent first
        self.train = kwargs.get("train", False)
        self.wheel_radius = self.plant.geometry()[2]

        if ref_mode == "hybrid":
            self.num_waypoints = 10000
            path_length = 50
            self.trajectory_generator = TrajGen(self.num_waypoints, path_length, device=device)
            cs = kwargs.get("curv_scale", 1.0)
            self.trajectory_generator.curv_scale = cs if isinstance(cs, (tuple, list)) else float(cs)
            self.trajectory_generator.seg_scale = float(kwargs.get("seg_scale", 1.0))
            self.trajectory_generator.curv_smooth = int(kwargs.get("curv_smooth", 0))
            self.trajectory_generator.flip_prob = kwargs.get("flip_prob", None)
            self.num_paths = int(kwargs.get("num_paths", 256))
            self.path_bank = self._scale_waypoints(
                self.trajectory_generator.generate_random_waypoints(self.num_paths))
        # circle trajectory
        elif ref_mode == "circle":
            self.num_waypoints = 2512      # one waypoint per 0.005m
            radius = 1.
            waypoint_x = radius * torch.sin(torch.arange(self.num_waypoints) / self.num_waypoints * 4 * np.pi)
            waypoint_y = radius - radius * torch.cos(torch.arange(self.num_waypoints) / self.num_waypoints * 4 * np.pi)
            waypoint_veldir = torch.arange(self.num_waypoints) / self.num_waypoints * 4 * np.pi
            waypoint_veldir = torch.atan2(torch.sin(waypoint_veldir), torch.cos(waypoint_veldir))
            waypoint_betaref = -torch.ones((self.num_waypoints, ))
            waypoint_kappa = torch.ones((self.num_waypoints, )) / radius
            self.path_bank = self._scale_waypoints(torch.stack((waypoint_x, waypoint_y, waypoint_veldir, waypoint_betaref, waypoint_kappa), dim=1).to(self.device)).unsqueeze(0)
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
            waypoint_betaref = torch.cat((
                -torch.ones((self.num_waypoints // 2, )),
                torch.ones((self.num_waypoints // 2, )),
            ))
            waypoint_kappa = torch.cat((
                torch.ones((self.num_waypoints // 2, )) / radius,
                -torch.ones((self.num_waypoints // 2, )) / radius,
            ))
            self.path_bank = self._scale_waypoints(torch.stack((waypoint_x, waypoint_y, waypoint_veldir, waypoint_betaref, waypoint_kappa), dim=1).to(self.device)).unsqueeze(0)
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
        self.num_paths = int(self.path_bank.shape[0])
        self.path_id = torch.zeros(self.n, dtype=torch.long, device=self.device)
        if self.num_paths > 1:
            self.path_id = torch.randint(0, self.num_paths, (self.n,), device=self.device)

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
        self.win_back = 20
        self.search_offsets = torch.arange(-self.win_back, 20, device=self.device)
        self.stall_move_min = 1.0
        self._stall_anchor = torch.zeros((self.n, 2), device=self.device)
        self.preview_scale = float(kwargs.get("preview_scale", 1.0))
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
            out[..., 4] /= self.length_scale
        if abs(self.beta_scale - 1.0) > 1e-9:
            out[..., 3] *= self.beta_scale
        return out

    def update_recent_obs(self, obs):
        self.recent_obs[1] = self.recent_obs[0]
        self.recent_obs[0] = obs

    def waypoints_at(self, ind):
        """Reference waypoints for each env, ``ind`` is (n, k) long indices."""
        return self.path_bank[self.path_id.unsqueeze(1), ind]

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
        current = torch.stack((x, y, veldir, beta, kappa), dim=1)
        diff = current.unsqueeze(1) - waypoints_filtered
        # wrap angle diff to (-pi, pi)
        angle_diff = torch.atan2(torch.sin(diff[:, :, 2]), torch.cos(diff[:, :, 2]))
        pos_diff = torch.hypot(diff[:, :, 0], diff[:, :, 1])
        beta_diff = torch.atan2(torch.sin(diff[:, :, 3]), torch.cos(diff[:, :, 3]))
        kappa_diff = diff[:, :, 4]
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
        ax_now = self.extras["ax"] if self.extras is not None else torch.zeros_like(self._v_hint)
        slow = (self._v_hint < self.stall_speed) & (ax_now < 0.5)
        # "stalled" must mean NOT MOVING, not "slow": a drift legitimately dips
        # below the stall speed mid corner.  The counter only keeps running while
        # the car has also travelled less than ``stall_move_min`` metres since the
        # window opened.
        pos_now = self.s[:, :2]
        self._stall_anchor = torch.where(slow.unsqueeze(1), self._stall_anchor, pos_now)
        moved = torch.norm(pos_now - self._stall_anchor, dim=1)
        stalled = slow & (moved < self.stall_move_min)
        self.no_progress_count[~stalled] = 0
        self.no_progress_count[stalled] += 1

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
        self.beta_ref = waypoints_filtered[:, :, 3].gather(1, bidx).squeeze(1)
        self.kappa_diff = kappa_diff.gather(1, bidx).squeeze(1)
        self.kappa_ref = waypoints_filtered[:, :, 4].gather(1, bidx).squeeze(1)

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
        self.find_nearest_waypoint(x, y, veldir, beta, kappa_est)
        angle_diff = self.angle_diff
        signed_pos_diff = self.signed_pos_diff
        beta_diff = self.beta_diff
        beta_ref = self.beta_ref
        kappa_diff = self.kappa_diff
        kappa_ref = self.kappa_ref

        # look ahead 3m (scaled with the reference geometry)
        if self.train:
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

        def convert(waypoints):
            """Cast a waypoint into the current course aligned frame."""
            x1 = waypoints[:, 0]
            y1 = waypoints[:, 1]
            veldir1 = waypoints[:, 2]
            beta1 = waypoints[:, 3]
            kappa1 = waypoints[:, 4]
            theta = torch.atan2(y1 - y, x1 - x) - veldir
            l = torch.hypot(y1 - y, x1 - x)
            x2 = l * torch.cos(theta)
            y2 = l * torch.sin(theta)
            veldir2 = veldir1 - veldir
            veldir2 = torch.atan2(torch.sin(veldir2), torch.cos(veldir2))
            # reference sideslip sign only; the measured sideslip is not observed
            beta2 = beta1
            kappa2 = kappa1 * self.length_scale          # O(1) normalisation
            x2, y2, veldir2, beta2, kappa2 = map(lambda t: torch.unsqueeze(t, 1),
                                                 [x2, y2, veldir2, beta2, kappa2])
            return torch.cat([x2, y2, veldir2, beta2, kappa2], 1)

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
            succeeded = (self.step_count >= self.max_steps) | (self.progress >= self.num_waypoints - 100)
        else:
            succeeded = (self.step_count >= self.max_steps)
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
        scalars += [e_pos, angle_diff, e_kappa, self.step_progress.float()]
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
            priv = [
                beta, v, r, self.s[:, 3], self.s[:, 4],
                (r / (v + 1e-6)), kappa_ref, self._now["r_ref"], beta_ref,
                signed_pos_diff / self.length_scale, angle_diff,
                ax_acc, ay_acc, az_acc,
            ]
            priv_list = [t.unsqueeze(1) for t in priv]
            priv_list.append(wheel_speed)
            priv_list.append(slip)
            priv_list.append(fz / 1000.0)
            priv_list.append(fy / 1000.0)
            priv_list.append(torch.stack(ctrl, dim=1))
            # the preview is what makes the value estimate informative about the
            # future; without it the privileged critic is blind to the path
            priv_list.extend(la_converted)
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
        rew_prog = torch.clamp(prog, 0.0, 5.0 * s)
        rew_smooth = self.smooth_scale * (
            - ((delta - prev["delta"]) ** 2
               + 1e-2 * (((wheel - prev["wheel"]) / self.speed_ref) ** 2).sum(dim=1)))
        # Adaptive drift target: holding a fixed sideslip locks the drift radius,
        # so a car that has drifted away from the line cannot recover while it is
        # still demanded to drift (observed: the policy holds the reference beta
        # and runs off the path, which is by far the dominant failure).  The
        # demanded sideslip fades out as the lateral error grows, i.e. drift on
        # the line, straighten to come back.
        beta_gate = torch.clamp(1.0 - (signed_pos_diff / (self.beta_gate_scale * s)) ** 2, 0.0, 1.0)
        # pure pursuit tracking: the steering that would follow the preview point
        rew_pp = - torch.clamp(((delta - self.delta_pp) / 0.3) ** 2, max=4.0)
        rew_beta = - (beta - self.beta_ref * beta_gate) ** 2
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
        if self.speed_hinge:
            rew_speed = - torch.clamp((self.speed_target - v) / self.speed_target, min=0.0)
        else:
            rew_speed = - ((v - self.speed_target) / self.speed_target) ** 2
        rew_final = -is_failed

        terms = {
            "pos": self.pos_weight * rew_pos,
            "recover": self.recover_weight * rew_recover,
            "pure_pursuit": self.pp_weight * rew_pp,
            "dir": self.dir_weight * rew_dir,
            "prog": 0.2 * rew_prog,
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
        super().reset()
        self.progress = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.step_progress = torch.zeros(self.n, dtype=torch.int16, device=self.device)
        self.win_back = 20
        self.search_offsets = torch.arange(-self.win_back, 20, device=self.device)
        self.stall_move_min = 1.0
        self._stall_anchor = torch.zeros((self.n, 2), device=self.device)
        self.no_progress_count = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        self.is_done = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        self._now = None
        self._prev = None
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
        self.step_count[is_done] = 0
        if self.num_paths > 1:
            # hybrid mode: every episode draws a reference from the shared bank
            self.path_id[is_done] = torch.randint(0, self.num_paths, (size,), device=self.device)
        self.progress[is_done] = torch.randint(self.num_waypoints // 2, (size,), device=self.device)
        self.step_progress[is_done] = 0
        self.no_progress_count[is_done] = 0

        def gen_random_state(indices):
            # The XCAR code spawned 20 % of the episodes at the path start at
            # rest (bernoulli mask multiplied into the state).  For a 1320 kg car
            # that is "enter a drift from standstill", which it cannot do yet and
            # which consumed a fifth of every batch; the equilibrium spawn below
            # is used for all episodes instead.
            prog = self.progress[indices]
            waypoints = self.path_bank[self.path_id[indices], prog, :]

            x = waypoints[:, 0]
            y = waypoints[:, 1]
            veldir = waypoints[:, 2]
            beta = waypoints[:, 3]
            kappa = waypoints[:, 4]

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
                beta0 = beta * self.spawn_beta_frac * torch.clamp(
                    1.0 + self.spawn_beta_noise * torch.randn_like(beta), 0.4, 1.4)
                v0 = self.spawn_speed * torch.clamp(
                    1.0 + self.spawn_speed_noise * torch.randn_like(beta), 0.6, 1.3)
                psi = veldir - beta0 + 0.1 * torch.randn_like(veldir)
                xd = v0 * torch.cos(beta0 + psi)
                yd = v0 * torch.sin(beta0 + psi)
                psid = v0 * kappa
            else:
                random_V = torch.rand(prog.shape, device=self.device) * 3 * self.speed_scale
                random_beta = torch.sign(beta) * torch.rand(prog.shape, device=self.device) * self.beta_scale
                psi = veldir - random_beta
                psi += 0.1 * torch.randn_like(psi)
                xd = random_V * torch.cos(random_beta + psi)
                yd = random_V * torch.sin(random_beta + psi)
                psid = -torch.sign(beta) * (1 + torch.rand(prog.shape, device=self.device) * 2) / self.speed_scale

            x, y, psi, xd, yd, psid = map(lambda t: torch.unsqueeze(t, 1), [x, y, psi, xd, yd, psid])
            return torch.cat([x, y, psi, xd, yd, psid], 1)

        self.s[is_done, :] = 0
        self.s[is_done, :6] = gen_random_state(is_done.nonzero().squeeze(-1))
        self.u[is_done, :] = 0
        self.u_cmd[is_done, :] = 0
        if self.pedals and self.spawn_mode == "equilibrium":
            # A car teleported into a drift was, physically, already
            # countersteering and on the throttle.  The observation contains the
            # previous command, and a zero previous command at a 0.75 rad
            # sideslip spawn is inconsistent -- without this the policy has no
            # information at all at the first step (its recurrent state is
            # empty) and learns to lift off, which collapses the drift.
            beta_wp = self.path_bank[self.path_id[is_done], self.progress[is_done], 3]
            self.u_cmd[is_done, 0] = (0.55 * beta_wp).clamp(-self.max_steer, self.max_steer)
            self.u_cmd[is_done, 1] = 0.55
        self.randomize(is_done)
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

    def step(self, action, **kwargs):
        # Physics first, THEN respawn: the action was chosen from the terminal
        # observation, so it has to act on the terminal state.  Resetting before
        # the step (the previous order) applied that action to the freshly
        # spawned car, which injected an arbitrary command into the start of
        # every episode and made the reward history cross episode boundaries.
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
                self.reset_done_envs()
                fresh = self.obs()
                if keep_reason is not None:
                    self._term_reason = keep_reason
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
