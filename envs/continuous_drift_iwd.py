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
        # observed: yaw rate (1) + previous commands (3/4) + wheel speeds (4)
        # + lateral/course/curvature errors and progress (4) + 3 preview points
        # x 5.  The plant state (10) is a simulation detail and not observed.
        self.num_states = (1 + (3 if self.pedals else 4) + 4 + 4) + 15

        # The reference geometry of this task was authored for the 0.35 m XCAR.
        # A full-size car cannot follow a 1 m radius curve (minimum steering
        # radius is L/tan(max_steer) ~ 5.4 m), so the whole reference is scaled
        # with the wheelbase ratio.  Everything metric is expressed in XCAR
        # units and multiplied by ``length_scale`` on the way in, everything
        # with the unit of time by ``speed_scale = sqrt(length_scale)``.
        wheelbase = float(self.preset["lF"] + self.preset["lR"])
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
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
        self.state_space = self.observation_space

        dummy_obs = torch.zeros((self.n, self.num_states), device=self.device)
        self.recent_obs = [dummy_obs, dummy_obs]  # most recent first
        self.train = kwargs.get("train", False)
        self.wheel_radius = self.plant.geometry()[2]

        if ref_mode == "hybrid":
            self.num_waypoints = 10000
            path_length = 50
            self.trajectory_generator = TrajGen(self.num_waypoints, path_length, device=device)
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
        self.step_progress = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
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
        # Look ahead 20 steps
        if self.train:
            ind = torch.stack([self.progress + i for i in range(20)], dim=1)
            ind[ind >= self.num_waypoints] = self.num_waypoints - 1
        else:
            ind = torch.stack([self.progress + i for i in range(20)], dim=1) % self.num_waypoints
        waypoints_filtered = self.waypoints_at(ind)
        current = torch.stack((x, y, veldir, beta, kappa), dim=1)
        diff = current.unsqueeze(1) - waypoints_filtered
        # wrap angle diff to (-pi, pi)
        angle_diff = torch.atan2(torch.sin(diff[:, :, 2]), torch.cos(diff[:, :, 2]))
        pos_diff = torch.hypot(diff[:, :, 0], diff[:, :, 1])
        beta_diff = torch.atan2(torch.sin(diff[:, :, 3]), torch.cos(diff[:, :, 3]))
        kappa_diff = diff[:, :, 4]
        self.step_progress = torch.argmin(pos_diff, 1)
        self.progress = (self.progress + self.step_progress)
        self.no_progress_count[self.step_progress != 0] = 0
        self.no_progress_count[self.step_progress == 0] += 1

        # u = vector from nearest waypoint to current pos; v = velocity direction at nearest waypoint
        # theta = angle from u to v
        # theta > 0 means current pos is to the right of waypoint
        best = self.step_progress.unsqueeze(1)
        ux = diff[:, :, 0].gather(1, best).squeeze(1)
        uy = diff[:, :, 1].gather(1, best).squeeze(1)
        u_angle = torch.atan2(uy, ux)
        v_angle = waypoints_filtered[:, :, 2].gather(1, best).squeeze(1)
        theta = v_angle - u_angle
        theta = torch.atan2(torch.sin(theta), torch.cos(theta))
        pos_diff_sign = torch.sign(theta)
        self.angle_diff = angle_diff.gather(1, best).squeeze(1)
        self.signed_pos_diff = pos_diff_sign * pos_diff.gather(1, best).squeeze(1)
        self.beta_diff = beta_diff.gather(1, best).squeeze(1)
        self.beta_ref = waypoints_filtered[:, :, 3].gather(1, best).squeeze(1)
        self.kappa_diff = kappa_diff.gather(1, best).squeeze(1)
        self.kappa_ref = waypoints_filtered[:, :, 4].gather(1, best).squeeze(1)

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
        v_est = wheel_speed.mean(dim=1).clamp(min=0.5)
        kappa_est = r / v_est

        xd = self.s[:, 3]
        yd = self.s[:, 4]
        veldir = torch.atan2(yd, xd)
        veldir[torch.hypot(xd, yd) < 0.01] = psi[torch.hypot(xd, yd) < 0.01]

        self.find_nearest_waypoint(x, y, veldir, beta, kappa_est)
        angle_diff = self.angle_diff
        signed_pos_diff = self.signed_pos_diff
        beta_diff = self.beta_diff
        beta_ref = self.beta_ref
        kappa_diff = self.kappa_diff
        kappa_ref = self.kappa_ref

        # look ahead 3m (scaled with the reference geometry)
        if self.train:
            la_inds = [(self.progress + i).unsqueeze(-1) for i in (200, 400, 600)]
            for la_ind in la_inds:
                la_ind[la_ind >= self.num_waypoints] = self.num_waypoints - 1
        else:
            la_inds = [(self.progress + i).unsqueeze(-1) % self.num_waypoints for i in (200, 400, 600)]
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

        failed = (self.progress >= 5) & ((self.no_progress_count >= self.no_progress_limit)
                                        | (torch.abs(angle_diff) > 1.)
                                        | (torch.abs(signed_pos_diff) > self.length_scale))
        if self.train:
            succeeded = (self.step_count >= self.max_steps) | (self.progress >= self.num_waypoints - 100)
        else:
            succeeded = (self.step_count >= self.max_steps)
        self.is_done[failed] = 1
        self.is_done[succeeded] = 2

        if self.pedals:
            ctrl = [self.u_cmd[:, 0], self.u_cmd[:, 1], self.u_cmd[:, 2]]
        else:
            ctrl = [self.u[:, 1], self.u[:, 2], self.u[:, 3], self.u[:, 4]]

        self._prev = self._now
        self._now = {
            "beta": beta, "v": v, "kappa": r / (v + 1e-6),   # true curvature: reward only
            "vflx": vflx, "vfrx": vfrx,
            "delta": self.u_cmd[:, 0],
            "wheel": wheel_speed,
            "angle_diff": angle_diff, "signed_pos_diff": signed_pos_diff,
            "beta_diff": beta_diff, "kappa_ref": kappa_ref,
            "prog": self.step_progress, "failed": failed.float(),
        }

        # observed: ego state + previous commands + tracking errors + preview
        e_pos = signed_pos_diff / self.length_scale
        e_kappa = kappa_diff * self.length_scale
        scalars = [r] + ctrl
        scalars += [wheel_speed[:, i] for i in range(4)]
        scalars += [e_pos, angle_diff, e_kappa, self.step_progress.float()]
        obs_list = [t.unsqueeze(1) for t in scalars]
        obs_list.extend(la_converted)
        obs = torch.cat(obs_list, 1)
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
        rew_pos = - (signed_pos_diff / s) ** 2
        rew_dir = - angle_diff ** 2
        rew_prog = torch.clamp(prog, 0.0, 5.0 * s)
        rew_smooth = self.smooth_scale * (
            - ((delta - prev["delta"]) ** 2
               + 1e-2 * (((wheel - prev["wheel"]) / self.speed_ref) ** 2).sum(dim=1)))
        rew_beta = - beta_diff ** 2
        rew_lowspeed = torch.clamp(v, 0., 0.5 * self.speed_scale) - 0.5 * self.speed_scale
        rew_sf = - ((vfrx - wheel[:, 1]) / self.speed_scale) ** 2 \
                 - ((vflx - wheel[:, 0]) / self.speed_scale) ** 2
        rew_kappa = - ((now["kappa"] - kappa_ref) * s) ** 2
        rew_final = -is_failed

        rew = 2.4 * rew_pos + 0.5 * rew_dir + 0.2 * rew_prog + 0.015 * rew_smooth + self.beta_weight * rew_beta + 0.1 * rew_lowspeed + 0.005 * rew_sf + 0.15 * rew_kappa
        return rew

    def done(self):
        return self.is_done

    def reset(self):
        super().reset()
        self.progress = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.step_progress = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
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
            random_indices = torch.bernoulli(0.8 * torch.ones((len(indices),), device=self.device))
            self.progress[indices] = self.progress[indices] * random_indices.bool()
            prog = self.progress[indices]
            waypoints = self.path_bank[self.path_id[indices], prog, :]

            x = waypoints[:, 0]
            y = waypoints[:, 1]
            veldir = waypoints[:, 2]
            beta = waypoints[:, 3]
            kappa = waypoints[:, 4]

            x += 0.1 * self.length_scale * torch.randn_like(x)
            y += 0.1 * self.length_scale * torch.randn_like(y)

            random_V = torch.rand(prog.shape, device=self.device) * 3 * self.speed_scale
            random_beta = torch.sign(beta) * torch.rand(prog.shape, device=self.device) * self.beta_scale

            psi = veldir - random_beta
            psi += 0.1 * torch.randn_like(psi)

            xd = random_V * torch.cos(random_beta + psi)
            yd = random_V * torch.sin(random_beta + psi)
            psid = -torch.sign(beta) * (1 + torch.rand(prog.shape, device=self.device) * 2) / self.speed_scale

            x, y, psi, xd, yd, psid = map(lambda t: torch.unsqueeze(t, 1), [x, y, psi, xd, yd, psid])
            return torch.cat([x, y, psi, xd, yd, psid], 1) * random_indices.unsqueeze(1)

        self.s[is_done, :] = 0
        self.s[is_done, :6] = gen_random_state(is_done.nonzero().squeeze(-1))
        self.u[is_done, :] = 0
        self.u_cmd[is_done, :] = 0
        self.randomize(is_done)
        if self.plant_name == "fast":
            self.plant.init_wheels_mask(self.s, is_done)
            self.plant.reset_guess(is_done)
            if self.disturbance_param is not None:
                self.tyre_scale[is_done] = 1.0
        self.is_done[:] = 0

    def step(self, action, **kwargs):
        if self.train:
            self.reset_done_envs()
        obs, reward, done, info = super().step(action, **kwargs)
        self.update_recent_obs(obs)
        return obs, reward, done, info
