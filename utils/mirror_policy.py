"""Exact left/right symmetry by shared recurrent evaluation, not a loss penalty.

Inputs use the 27/35/43/51-channel observable actor contract. Hidden state stores both
histories as [2, layers, batch, hidden]; reflecting a history swaps its branches.
Enabled explicitly by the --mirror-policy training option.
"""
import os
import torch
from torch import nn


class ActorMirror(nn.Module):
    def __init__(self, preview_count=3):
        super().__init__()
        width = 15 + 4 * preview_count
        perm = torch.arange(width)
        perm[7:11] = torch.tensor([8, 7, 10, 9])
        sign = torch.ones(width)
        sign[[0, 2, 4, 11, 12, 13] + [15+4*j+k for j in range(preview_count) for k in (1,2,3)]] = -1
        self.register_buffer("permutation", perm)
        self.register_buffer("sign", sign)
        self.register_buffer("action_sign", torch.tensor([-1., 1., 1.]))

    def forward(self, obs):
        if obs.shape[-1] != self.sign.numel():
            raise ValueError("input does not match the configured mirror contract")
        return obs[..., self.permutation] * self.sign

    def normalize(self, obs, mean, variance):
        # Project moments onto the mirror-invariant distribution. Include the
        # between-mean variance term when pooling exchanged wheel channels.
        reflected_mean = self(mean)
        pooled_mean = (mean + reflected_mean) * .5
        pooled_variance = ((variance + mean.square() +
                            variance[self.permutation] + reflected_mean.square()) * .5
                           - pooled_mean.square()).clamp_min(0.)
        return ((obs - pooled_mean) / (pooled_variance + 1e-5).sqrt()).clamp(-5., 5.)


class SharedMirrorActor(nn.Module):
    """Shared GRU128 + observable bypass + 256/124/64 action branch.

    forward accepts time-major [T,B,15+4N] raw (environment-scaled) observations.
    done_before[t] resets both histories before consuming observation t.
    Fixed diagonal Gaussian log-std is mirror invariant; coupled samples use
    mirrored steering noise, while arbitrary samples are symmetric in law.
    """
    def __init__(self, preview_count=3, beta_feedback=False, hidden_size=128, actor_units=(256,124,64), road_encoder="none"):
        super().__init__()
        self.beta_feedback = bool(beta_feedback)
        width = 15 + 4 * preview_count
        self.mirror = ActorMirror(preview_count)
        self.hidden_size=int(hidden_size)
        self.preview_count=preview_count
        self.gru = nn.GRU(width, self.hidden_size)
        units=tuple(actor_units);layers=[];previous=self.hidden_size+width+int(beta_feedback)+(64 if road_encoder!='none' else 0)
        for unit in units:
            layers.extend([nn.Linear(previous,unit),nn.ELU()]);previous=unit
        self.mlp=nn.Sequential(*layers)
        self.mu=nn.Linear(units[-1],3)
        self.beta=nn.Sequential(nn.Linear(self.hidden_size,64),nn.ELU(),nn.Linear(64,1))
        self.logstd=nn.Parameter(torch.tensor([-.99,-.80,-1.39]))
        self.road=None
        if road_encoder!='none':
            from utils.road_encoder import RoadEncoder
            self.road=RoadEncoder(preview_count,road_encoder)

    def forward(self, obs, mean, variance, hidden=None, done_before=None):
        precision=getattr(self,"inference_precision","float32")
        if precision not in ("float32","float16"):raise ValueError("unknown actor inference precision: "+precision)
        if precision=="float16" and obs.is_cuda and not torch.is_grad_enabled() and not torch.is_autocast_enabled("cuda"):
            with torch.autocast("cuda",dtype=torch.float16):
                return self._execute(obs,mean,variance,hidden,done_before)
        return self._execute(obs,mean,variance,hidden,done_before)

    def _execute(self, obs, mean, variance, hidden=None, done_before=None):
        backend = getattr(self, "execution_backend", os.environ.get("MIRROR_NETWORK_BACKEND", "reference"))
        if backend not in ("reference", "cuda_graph"):
            raise ValueError("unknown mirror execution backend: " + backend)
        if backend != "reference" and obs.is_cuda:
            from utils.exact_network_graph import actor_graph_call
            return actor_graph_call(self, obs, mean, variance, hidden, done_before)
        if torch.is_grad_enabled():self._reference_training_used=True
        return self._forward_reference(obs, mean, variance, hidden, done_before)

    def _forward_reference(self, obs, mean, variance, hidden=None, done_before=None):
        if obs.ndim != 3:
            raise ValueError("expected [time,batch,observation_width]")
        batch = obs.shape[1]
        if hidden is None:
            hidden = obs.new_zeros(2, 1, batch, self.hidden_size)
        outputs, betas = [], []
        for t in range(obs.shape[0]):
            if done_before is not None:
                hidden = hidden * (~done_before[t].bool())[None, None, :, None]
            x = (obs[t] if mean is None else
                 self.mirror.normalize(obs[t], mean, variance))
            pair = torch.cat((x, self.mirror(x)), dim=0)
            h = torch.cat((hidden[0], hidden[1]), dim=1)
            recurrent, h = self.gru(pair.unsqueeze(0), h)
            hidden = torch.stack(h.split(batch, dim=1), dim=0)
            recurrent = recurrent[0]
            beta_raw = self.beta(recurrent)
            features = [recurrent, pair]
            if self.road is not None:features.append(self.road(pair[:,15:].reshape(-1,self.preview_count,4)))
            if self.beta_feedback:
                b0, b1 = beta_raw.split(batch, dim=0)
                estimate = ((b0 - b1) * .5).detach()
                features.append(torch.cat((estimate, -estimate), dim=0))
            raw = self.mu(self.mlp(torch.cat(features, dim=-1)))
            left, right = raw.split(batch, dim=0)
            outputs.append((left + right * self.mirror.action_sign) * .5)
            b0, b1 = beta_raw.split(batch, dim=0)
            betas.append((b0 - b1) * .5)
        return torch.stack(outputs), self.logstd, torch.stack(betas), hidden


class CriticMirror(ActorMirror):
    """52/62/72/82-channel privileged contract with BODY velocity at indices 3 and 4.

    Never apply this mapping to legacy world-velocity critic observations.
    """
    def __init__(self, preview_count=3):
        super().__init__(preview_count)
        perm = torch.arange(37 + 5 * preview_count)
        for start in (14, 18, 22, 26, 30):
            perm[start:start + 4] = torch.tensor([start+1, start, start+3, start+2])
        sign = torch.ones(37 + 5 * preview_count)
        sign[[0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 30, 31, 32, 33, 34] +
             [37+5*j+k for j in range(preview_count) for k in (1,2,3,4)]] = -1
        self.permutation = perm
        self.sign = sign


class SharedMirrorCritic(nn.Module):
    """Independent shared-weight MLP with exactly invariant scalar output."""
    def __init__(self, units=(256, 128, 64), preview_count=3):
        super().__init__()
        self.mirror = CriticMirror(preview_count)
        layers, width = [], 37 + 5 * preview_count
        for next_width in units:
            layers.extend((nn.Linear(width, next_width), nn.ELU()))
            width = next_width
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, states, mean, variance):
        backend = getattr(self, "execution_backend", os.environ.get("MIRROR_NETWORK_BACKEND", "reference"))
        if backend in ("cuda_graph",) and states.is_cuda:
            from utils.exact_network_graph import critic_graph_call
            return critic_graph_call(self, states, mean, variance)
        if torch.is_grad_enabled():self._reference_training_used=True
        return self._forward_reference(states, mean, variance)

    def _forward_reference(self, states, mean, variance):
        x = (states if mean is None else
             self.mirror.normalize(states, mean, variance))
        n = x.shape[0]
        values = self.mlp(torch.cat((x, self.mirror(x)), dim=0))
        return (values[:n] + values[n:]) * .5
