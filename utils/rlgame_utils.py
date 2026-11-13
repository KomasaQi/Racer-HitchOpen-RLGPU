# This file contains code copied from Isaac Gym Benchmark Environments (IsaacGymEnvs)
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This file contains code copied from Isaac Gym Benchmark Environments (IsaacGymEnvs).
Original source: https://github.com/isaac-sim/IsaacGymEnvs
"""

import torch
from collections import OrderedDict
from pathlib import Path
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch import torch_ext


def normalise_compiled_central_state(state):
    """Remove the torch.compile wrapper segment from central-value keys only.

    rl_games restores the actor after compiling it (so actor ``_orig_mod`` keys
    are correct) but restores the asymmetric central-value model before its
    compile wrapper exists.  Its saved ``model._orig_mod.*`` keys must therefore
    become ``model.*``.  Parameter values and optimizer state are untouched.
    """
    return OrderedDict((key.replace("model._orig_mod.", "model."), value)
                       for key, value in state.items())


def resume_compatible_checkpoint(path):
    """Return a checkpoint path loadable by asymmetric rl_games + torch.compile."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("assymetric_vf_nets")
    if not state or not any("model._orig_mod." in key for key in state):
        return path
    payload["assymetric_vf_nets"] = normalise_compiled_central_state(state)
    source = Path(path)
    output = source.with_name(source.stem + ".resume_compat" + source.suffix)
    torch.save(payload, output)
    return str(output)

class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()
    
    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        if hasattr(self.env, "amp_observation_space"):
            info['amp_observation_space'] = self.env.amp_observation_space

        if self.env.num_states > 0:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info

class RLGPUAlgoObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats. """

    def __init__(self, path_refresh_every=8, warm_start=None):
        self.path_refresh_every = int(path_refresh_every)
        self.warm_start = warm_start

    def after_init(self, algo):
        self.algo = algo
        self.mean_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.ep_infos = []
        self.direct_info = {}
        self.writer = self.algo.writer
        if self.warm_start:
            payload = torch.load(self.warm_start, map_location=self.algo.ppo_device,
                                 weights_only=False)
            model_state = OrderedDict(
                (key.replace("_orig_mod.", ""), value)
                for key, value in payload["model"].items())
            self.algo.model.load_state_dict(model_state, strict=True)
            central = payload.get("assymetric_vf_nets")
            if central is not None and getattr(self.algo, "central_value_net", None) is not None:
                self.algo.central_value_net.load_state_dict(
                    normalise_compiled_central_state(central), strict=True)
            print("warm-started model weights (fresh optimizer) from", self.warm_start)

    def process_infos(self, infos, done_indices):
        assert isinstance(infos, dict), "RLGPUAlgoObserver expects dict info"
        if isinstance(infos, dict):
            if 'episode' in infos:
                self.ep_infos.append(infos['episode'])

            if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
                self.direct_info = {}
                for k, v in infos.items():
                    # only log scalars
                    if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                        self.direct_info[k] = v

    def after_clear_stats(self):
        self.mean_scores.clear()

    def _request_path_refresh(self):
        """Walk the wrapper chain and ask the env for a fresh road generation."""
        target = getattr(self.algo, "vec_env", None)
        seen = set()
        while target is not None and id(target) not in seen:
            seen.add(id(target))
            fn = getattr(target, "request_path_refresh", None)
            if callable(fn):
                return bool(fn())
            target = getattr(target, "env", None) or getattr(target, "venv", None)
        return False

    def after_print_stats(self, frame, epoch_num, total_time):
        # every N PPO epochs ask for a new path generation; the env installs it
        # only when the target ring slot has no active episode
        if (self.path_refresh_every > 0 and epoch_num > 0
                and epoch_num % self.path_refresh_every == 0):
            installed = self._request_path_refresh()
            if getattr(self, "writer", None) is not None:
                self.writer.add_scalar("paths/refresh_installed", float(installed), epoch_num)
        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.algo.device)
                for ep_info in self.ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.algo.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, epoch_num)
            self.ep_infos.clear()
        
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f'{k}/frame', v, frame)
            self.writer.add_scalar(f'{k}/iter', v, epoch_num)
            self.writer.add_scalar(f'{k}/time', v, total_time)

        if self.mean_scores.current_size > 0:
            mean_scores = self.mean_scores.get_mean()
            self.writer.add_scalar('scores/mean', mean_scores, frame)
            self.writer.add_scalar('scores/iter', mean_scores, epoch_num)
            self.writer.add_scalar('scores/time', mean_scores, total_time)
