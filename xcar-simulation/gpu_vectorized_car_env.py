# -*- coding: utf-8 -*-
"""GPU vectorized car environment.

Two plants are available:

``plant="fast"`` (default)
    :class:`fast_plant.FastRacerPlant`, the full-size RWD EV plant: 10 states
    ``(x, y, psi, vx, vy, r, omega_FL..RR)``, isotropic XCAR combined-slip tire,
    motor torque envelope + power limit, viscous LSD, wheel spin dynamics,
    longitudinal + lateral load transfer and a one-pressure-center aerodynamic
    model.  It is meant to line up with the CarSim vehicle interface.

``plant="iwd"``
    The original XCAR :class:`IWDCarDynamics` (6 states, wheel speed commands,
    torchdiffeq euler).  Kept bit-for-bit for the legacy environments.

Control modes
-------------
``control_mode="pedals"``
    action = ``[steer, throttle, brake]``.  Matches the CarSim input channels.
    Throttle and brake are made mutually exclusive (``q = max(q-b, 0)``,
    ``b = max(b-q, 0)``) as in the project proposal.

``control_mode="wheel_speed"``
    action = ``[steer, v_FL, v_FR, v_RL, v_RR]``, the legacy XCAR wheel surface
    speed command (m/s).  It is mapped to per-wheel torques by a
    feed-forward + proportional servo inside the plant, so old observations,
    rewards and action layouts keep working.

Timing
------
``dt`` is the *control* period and ``physics_dt`` the physics step; the command
is held constant over ``dt / physics_dt`` physics substeps (zero order hold).
The default 20 Hz / 100 Hz split therefore integrates 5 substeps per action.
"""

import csv
import os
import random
import time
from pathlib import Path

import gym
import numpy as np
import torch
import yaml

try:  # the package import path differs between runners
    from .IWDCarDynamics import IWDCarDynamics
    from .fast_plant import (FastRacerPlant, NUM_ACTIONS, NUM_ACTIONS_LEGACY,
                             load_preset, resolve_params)
except ImportError:  # legacy sys.path based imports
    from IWDCarDynamics import IWDCarDynamics
    from fast_plant import (FastRacerPlant, NUM_ACTIONS, NUM_ACTIONS_LEGACY,
                            load_preset, resolve_params)


class GPUVectorizedCarEnv:
    def __init__(self,
                 preset_name,
                 n,
                 dt=None,
                 solver="euler",
                 device="cuda:0",
                 drivetrain="iwd",
                 plant="fast",
                 control_mode=None,
                 physics_dt=0.01,
                 disturbance_param=None,
                 randomize_param=None,
                 random_seed=None,
                 initial_state=None,
                 **kwargs):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)

        self.n = int(n)
        self.device = torch.device(device)
        self.preset_name = preset_name
        self.plant_name = plant
        self.solver = solver
        self.drivetrain = drivetrain
        self.train = kwargs.get("train", False)
        self.randomize_param = randomize_param or {}
        self.disturbance_param = disturbance_param
        self.steer_deadband = float(kwargs.get("steer_deadband", 0.0))
        self.brake_margin = float(kwargs.get("brake_margin", 0.0))
        self.steer_actuator_tau = float(kwargs.get("steer_actuator_tau", 0.0))
        self.steer_rate_limit = float(kwargs.get("steer_rate_limit", 0.0))
        self.pedal_actuator_tau = float(kwargs.get("pedal_actuator_tau", 0.0))
        self.actuator_dr = bool(kwargs.get("actuator_dr", False))
        self.action_delay_max = int(kwargs.get("action_delay_max", 2 if self.actuator_dr else 0))
        self.action_delay_max = max(self.action_delay_max, 0)
        self._steer_state = torch.zeros((self.n,), device=self.device)
        self._steer_prev = torch.zeros((self.n,), device=self.device)
        self.initial_state = initial_state

        # ------------------------------------------------------------- timing
        if control_mode is None:
            control_mode = "wheel_speed" if plant == "iwd" else "pedals"
        self.control_mode = control_mode
        if dt is None:
            dt = 0.01 if plant == "iwd" else 0.05
        self.dt = float(dt)
        self.physics_dt = float(physics_dt)
        if plant == "iwd":
            self.substeps = 1
        else:
            # The wheel root is implicit, but motor torque/power and body/load
            # transfer are frozen within each root solve.  At one 50 ms step the
            # rear wheel alternated between ~100 m/s and rolling speed on adjacent
            # control ticks.  Keep the advertised 100 Hz plant integration.
            ratio = self.dt / self.physics_dt
            self.substeps = max(int(round(ratio)), 1)
            if abs(self.substeps * self.physics_dt - self.dt) > 1e-9:
                raise ValueError("dt must be an integer multiple of physics_dt")

        self.num_states = 6
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.state_space = self.observation_space

        # ------------------------------------------------------------ actions
        if plant == "iwd":
            self.num_actions, self.action_space, self.cast_action = self._legacy_actions(drivetrain)
            self.preset = load_preset(preset_name)
            self._build_legacy_plant()
        else:
            self.preset = load_preset(preset_name)
            self.max_steer = float(self.preset.get("max_steer_rad", 0.45))
            if control_mode == "pedals":
                self.num_actions = NUM_ACTIONS
                self.action_space = gym.spaces.Box(
                    low=np.array([-self.max_steer, 0.0, 0.0], dtype=np.float32),
                    high=np.array([self.max_steer, 1.0, 1.0], dtype=np.float32),
                    dtype=np.float32)
                self.cast_action = lambda u: u
            else:
                self.num_actions = NUM_ACTIONS_LEGACY
                self.action_space = gym.spaces.Box(
                    low=np.array([-self.max_steer, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    high=np.array([self.max_steer, 7.0, 7.0, 7.0, 7.0], dtype=np.float32),
                    dtype=np.float32)
                self.cast_action = lambda u: u
            self.plant = FastRacerPlant(self.preset, self.n, device=self.device)
            grip = kwargs.get("grip_scale", None)
            if grip is not None:
                # uniform tyre friction multiplier (road/tyre grip level)
                self.plant.tyre_scale = torch.full((self.n, 1), float(grip), device=self.device)
            self.num_states = 10

        # -------------------------------------------------------------- state
        self.s = torch.zeros((self.n, self.num_states), device=self.device)
        if initial_state is not None:
            self.s[:, :6] = torch.as_tensor(initial_state, device=self.device, dtype=torch.float32)
        self.u = torch.zeros((self.n, self.num_actions), device=self.device)
        self.u_cmd = self.u.clone()
        # Per-environment actuator parameters.  With actuator_dr disabled these
        # are exact broadcasts of the legacy scalar configuration.
        self._steer_tau_env = torch.full((self.n,), self.steer_actuator_tau, device=self.device)
        self._steer_rate_env = torch.full((self.n,), self.steer_rate_limit, device=self.device)
        self._steer_deadband_env = torch.full((self.n,), self.steer_deadband, device=self.device)
        self._pedal_tau_env = torch.full((self.n,), self.pedal_actuator_tau, device=self.device)
        self._steer_gain_env = torch.ones(self.n, device=self.device)
        self._throttle_gain_env = torch.ones(self.n, device=self.device)
        self._brake_gain_env = torch.ones(self.n, device=self.device)
        self._action_delay_steps = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self._action_delay_buffer = torch.zeros(
            (self.action_delay_max + 1, self.n, self.num_actions), device=self.device)
        self._pedal_state = torch.zeros((self.n, 2), device=self.device)
        self.es = None
        self.extras = None
        self.current_obs = None
        self.dynamics = None
        self.step_count = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.total_step_count = 0
        self.saved_data = []
        if disturbance_param is not None:
            self.disturbance = torch.zeros((self.n, 8), device=self.device)
            self.tyre_scale = torch.ones((self.n, 1), device=self.device)

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _legacy_actions(drivetrain):
        if drivetrain == "4wd":
            space = gym.spaces.Box(low=np.array([-0.45, 0.]), high=np.array([0.45, 7.]), shape=(2,))
            cast = lambda u: torch.cat([u[:, :1], u[:, 1:2], u[:, 1:2], u[:, 1:2], u[:, 1:2]], 1)
            return 2, space, cast
        if drivetrain == "2iwd":
            space = gym.spaces.Box(low=np.array([-0.45, 0., 0.]), high=np.array([0.45, 7., 7.]), shape=(3,))
            cast = lambda u: torch.cat([u[:, :1], u[:, 1:2], u[:, 1:2], u[:, 2:3], u[:, 2:3]], 1)
            return 3, space, cast
        if drivetrain == "rwd":
            space = gym.spaces.Box(low=np.array([-0.45, 0.]), high=np.array([0.45, 7.]), shape=(2,))
            cast = lambda u: torch.cat([u[:, :1], u[:, 1:2], u[:, 1:2], u[:, 1:2], u[:, 1:2]], 1)
            return 2, space, cast
        space = gym.spaces.Box(low=np.array([-0.45, 0., 0., 0., 0.]),
                               high=np.array([0.45, 7., 7., 7., 7.]), shape=(5,))
        return 5, space, (lambda u: u)

    def _build_legacy_plant(self):
        params = self.preset
        self.p_body = torch.zeros((self.n, 8), device=self.device)
        for i, key in enumerate(("lF", "lR", "m", "h", "g", "Iz", "T")):
            self.p_body[:, i] = params[key]
        self.p_body[:, 7] = params.get("rF", 0.0)
        self.p_tyre = torch.zeros((self.n, 4), device=self.device)
        for i, key in enumerate(("B", "C", "D", "E")):
            self.p_tyre[:, i] = params[key]

    # ---------------------------------------------------------- randomization
    def randomize_item_(self, env_mask, override, key, target):
        num = int(torch.sum(env_mask).item())
        if key in override:
            target[env_mask] = override[key]
        elif key in self.randomize_param:
            lo, hi = self.randomize_param[key]
            target[env_mask] = lo + (hi - lo) * torch.rand(num, device=self.device)

    def randomize(self, env_mask=None, override=None):
        if override is None:
            override = {}
        if env_mask is None:
            env_mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        if self.plant_name == "fast":
            # pass EVERY range: the plant knows which parameters it can randomize
            # (mass, inertia, geometry, tyre, powertrain).  Filtering to the four
            # tyre shape parameters left the whole vehicle deterministic, which is
            # exactly what makes a policy impossible to transfer.
            ranges = dict(self.randomize_param)
            self.plant.randomize(env_mask, override=override, ranges=ranges)
            self._randomize_actuators(env_mask)
        else:
            rand_item_ = lambda key, target: self.randomize_item_(env_mask, override, key, target)
            rand_item_("B", self.p_tyre[:, 0])
            rand_item_("C", self.p_tyre[:, 1])
            rand_item_("D", self.p_tyre[:, 2])

    def _randomize_actuators(self, env_mask):
        """Draw bounded physical actuator uncertainty for selected episodes."""
        n = int(env_mask.sum().item())
        if n == 0:
            return
        if not self.actuator_dr:
            self._steer_tau_env[env_mask] = self.steer_actuator_tau
            self._steer_rate_env[env_mask] = self.steer_rate_limit
            self._steer_deadband_env[env_mask] = self.steer_deadband
            self._pedal_tau_env[env_mask] = self.pedal_actuator_tau
            self._steer_gain_env[env_mask] = 1.0
            self._throttle_gain_env[env_mask] = 1.0
            self._brake_gain_env[env_mask] = 1.0
            self._action_delay_steps[env_mask] = 0
            return
        rand = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=self.device)
        self._steer_tau_env[env_mask] = rand(0.05, 0.15)
        self._steer_rate_env[env_mask] = rand(0.8, 2.5)
        self._steer_deadband_env[env_mask] = rand(0.0, 0.010)
        self._pedal_tau_env[env_mask] = rand(0.04, 0.15)
        self._steer_gain_env[env_mask] = rand(0.95, 1.05)
        self._throttle_gain_env[env_mask] = rand(0.90, 1.10)
        self._brake_gain_env[env_mask] = rand(0.90, 1.10)
        self._action_delay_steps[env_mask] = torch.randint(
            0, self.action_delay_max + 1, (n,), device=self.device)

    # ------------------------------------------------------------------ hooks
    def obs(self):
        return self.s

    def reward(self):
        return torch.zeros(self.n, device=self.device)

    def done(self):
        return torch.zeros(self.n, device=self.device)

    def info(self):
        return {}

    def get_number_of_agents(self):
        return self.n

    def _legacy_dynamics(self):
        if self.disturbance_param is None:
            return IWDCarDynamics(self.cast_action(self.u), self.p_body, self.p_tyre,
                                 drivetrain=self.drivetrain)
        a, w = self.disturbance_param
        self.disturbance = a * self.disturbance + w * torch.randn((self.n, 8), device=self.device)
        return IWDCarDynamics(self.cast_action(self.u), self.p_body, self.p_tyre,
                             disturbance=self.disturbance, drivetrain=self.drivetrain)

    # -------------------------------------------------------------- lifecycle
    def reset(self):
        self._steer_state.zero_()
        self._steer_prev.zero_()
        self.randomize()
        if self.initial_state is not None:
            self.s[:, :6] = torch.as_tensor(self.initial_state, device=self.device, dtype=torch.float32)
        else:
            self.s.zero_()
        self.u.zero_()
        self.u_cmd.zero_()
        self._pedal_state.zero_()
        self._action_delay_buffer.zero_()
        if self.plant_name == "fast":
            self.plant.init_wheels(self.s)
            self.plant.reset_guess()
            self.extras = self._eval_extras(self.u_cmd)
            self.es = self.plant.package_extended(self.s, self.extras)
        else:
            self.dynamics = self._legacy_dynamics()
            self.es = self.dynamics.compute_extended_state(self.s)
        self.step_count.zero_()
        self.total_step_count = 0
        self.saved_data = []
        obs = self.obs()
        self.current_obs = obs
        return obs

    def _eval_extras(self, command):
        _, extras = self.plant.derivatives(self.s, command, mode=self.control_mode)
        return extras

    def preprocess_action(self, action):
        """Clamp to the physical command range and make the pedals exclusive."""
        if self.plant_name != "fast":
            return action
        u = action.clone()
        # RESIDUAL action around the trim: a drift is an unstable equilibrium, so an
        # actor near zero output removes the trim on its first step and the car
        # departs.  During the seeded curriculum action is a correction on top of
        # the per-environment trim.  The action has ALREADY been affinely mapped by
        # rl_games into the declared physical action space, so steering must not be
        # scaled again here.  With trim=0 (deployment/final curriculum), the actor's
        # steering channel is exactly road-wheel angle in radians.
        if getattr(self, "u_trim", None) is not None:
            u[:, 0] = u[:, 0] + self.u_trim[:, 0]
            u[:, 1] = u[:, 1] + self.u_trim[:, 1]
            u[:, 2] = u[:, 2] + self.u_trim[:, 2]
        target = torch.clamp(u[:, 0] * self._steer_gain_env,
                             -self.max_steer, self.max_steer)
        if self.control_mode == "pedals":
            q = torch.clamp(u[:, 1] * self._throttle_gain_env, 0.0, 1.0)
            b = torch.clamp(u[:, 2] * self._brake_gain_env, 0.0, 1.0)
            longitudinal = q - b
            u[:, 1] = torch.clamp(longitudinal, min=0.0)
            u[:, 2] = torch.clamp(-longitudinal - self.brake_margin, min=0.0)
        else:
            u[:, 1:] = torch.clamp(u[:, 1:], 0.0, 7.0)
        u[:, 0] = target

        # Command transport delay is sampled per episode.  Fill the buffer with
        # the installed trim on reset so delay never injects an artificial zero.
        if self.action_delay_max > 0:
            self._action_delay_buffer[1:] = self._action_delay_buffer[:-1].clone()
        self._action_delay_buffer[0] = u
        env_i = torch.arange(self.n, device=self.device)
        u = self._action_delay_buffer[self._action_delay_steps, env_i].clone()
        target = u[:, 0]

        active_steer = self._steer_tau_env > 0.0
        if bool(active_steer.any()):
            # Saturated first order servovalve: the rate is limited BEFORE it is
            # integrated, so a step cannot cross the target and the actuator can
            # never be faster than the slew limit.
            # a deadband holds the wheel for tiny target errors, which is what a
            # driver does and what removes sub-limit dither without touching real
            # countersteer
            error = target - self._steer_state
            error = torch.where(error.abs() <= self._steer_deadband_env,
                                torch.zeros_like(error), error)
            effective_target = self._steer_state + error
            rate = error / self._steer_tau_env.clamp(min=1e-6)
            limit = self._steer_rate_env
            rate = torch.where(limit > 0.0, torch.maximum(torch.minimum(rate, limit), -limit), rate)
            nxt = self._steer_state + self.dt * rate
            nxt = torch.where(effective_target >= self._steer_state,
                              torch.minimum(nxt, effective_target),
                              torch.maximum(nxt, effective_target))
            self._steer_state = torch.where(active_steer, nxt, target)
            self._steer_prev = self._steer_state.clone()
            u[:, 0] = self._steer_state
        else:
            u[:, 0] = target
        if self.control_mode == "pedals":
            target_pedal = u[:, 1:3]
            active_pedal = self._pedal_tau_env > 0.0
            alpha = (self.dt / self._pedal_tau_env.clamp(min=self.dt)).clamp(max=1.0)
            nxt = self._pedal_state + alpha.unsqueeze(1) * (target_pedal - self._pedal_state)
            self._pedal_state = torch.where(active_pedal.unsqueeze(1), nxt, target_pedal)
            # Apply the same exclusive net-demand contract after actuator lag.
            longitudinal = self._pedal_state[:, 0] - self._pedal_state[:, 1]
            u[:, 1] = torch.clamp(longitudinal, min=0.0, max=1.0)
            u[:, 2] = torch.clamp(-longitudinal - self.brake_margin, min=0.0, max=1.0)
        return u

    def step(self, u, override_s=None):
        self.u = u
        if self.plant_name == "fast":
            self.u_cmd = self.preprocess_action(u)
            self._update_disturbance()
            self.s, self.extras = self.plant.step(
                self.s, self.u_cmd, dt=self.dt, substeps=self.substeps, mode=self.control_mode)
            self.es = self.plant.package_extended(self.s, self.extras)
        else:
            self.u_cmd = u
            self.dynamics = self._legacy_dynamics()
            if override_s is None:
                from torchdiffeq import odeint
                t = torch.tensor([0.0, self.dt], device=self.device)
                self.s = odeint(self.dynamics, self.s, t, method=self.solver)[1, :, :]
            else:
                self.s[:] = torch.as_tensor(override_s, device=self.device).unsqueeze(0)
            self.es = self.dynamics.compute_extended_state(self.s)

        self.step_count += 1
        self.total_step_count += 1
        obs = self.obs()
        reward = self.reward() if self.train else torch.zeros(self.n, device=self.device)
        done = self.done()
        info = self.info()
        self.current_obs = obs
        if self.saved_data and bool(torch.all(done)):
            self.save_data()
            raise SystemExit(0)
        return obs, reward, done, info

    def _update_disturbance(self):
        """Mean-one AR(1) disturbance on the effective tyre friction.

        The process must fluctuate AROUND 1, not decay: with
        ``x <- a*x + w*N(0,1)`` and no mean reversion the expectation is a^t, so
        after 100 steps of a=0.97 it is 0.048 and the clamp pins every episode to
        the 0.4 floor -- the friction a car experiences then has nothing to do
        with the nominal value of the cell being evaluated.
        """
        if self.disturbance_param is None or self.plant_name != "fast":
            return
        a, w = self.disturbance_param
        self.tyre_scale = (
            1.0 + a * (self.tyre_scale - 1.0) + w * torch.randn_like(self.tyre_scale)
        )
        self.tyre_scale.clamp_(0.4, 1.6)
        self.plant.tyre_scale = self.tyre_scale

    def reset_done(self):
        if hasattr(self, "reset_done_envs"):
            self.reset_done_envs()
            return self.obs()
        return self.reset()

    # ------------------------------------------------------------- recording
    def save_data(self):
        basename = time.strftime("%Y%m%d-%H%M%S")
        Path("data").mkdir(parents=True, exist_ok=True)
        with open(os.path.join("data", basename + ".csv"), "w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            for s, u, es, _ in self.saved_data:
                x, y, psi = s[0].item(), s[1].item(), s[2].item()
                x_dot, y_dot, psi_dot = s[3].item(), s[4].item(), s[5].item()
                u5 = self.cast_action(u.unsqueeze(0)).squeeze(0)
                delta = u5[0].item()
                wheels = [u5[i].item() for i in range(1, u5.shape[0])]
                r = es[2].item()
                beta = es[7].item()
                v = es[6].item()
                writer.writerow([x, y, psi, x_dot, y_dot, psi_dot, delta, *wheels, r, beta, v])
        torch.save(self.saved_data, os.path.join("data", basename + ".pth"))
        print("Total steps:", self.total_step_count)

    def render(self, **kwargs):
        self.saved_data.append((self.s[0, :].cpu(), self.u[0, :].cpu(),
                                self.es[0, :].cpu(), self.current_obs[0, :].cpu()))

    def detach(self):
        self.s = self.s.detach()
        self.u = self.u.detach()
        self.es = self.es.detach()
