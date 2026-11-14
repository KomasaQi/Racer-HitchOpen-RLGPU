"""Exact left/right symmetry by shared recurrent evaluation, not a loss penalty.

Inputs use the 27-channel observable actor contract. Hidden state stores both
histories as [2, layers, batch, hidden]; reflecting a history swaps its branches.
Enabled explicitly by the --mirror-policy training option.
"""
import torch
from torch import nn


class ActorMirror(nn.Module):
    def __init__(self):
        super().__init__()
        perm = torch.arange(27)
        perm[7:11] = torch.tensor([8, 7, 10, 9])
        sign = torch.ones(27)
        sign[[0, 2, 4, 11, 12, 13, 16, 17, 18, 20, 21, 22, 24, 25, 26]] = -1
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

    forward accepts time-major [T,B,27] raw (environment-scaled) observations.
    done_before[t] resets both histories before consuming observation t.
    Fixed diagonal Gaussian log-std is mirror invariant; coupled samples use
    mirrored steering noise, while arbitrary samples are symmetric in law.
    """
    def __init__(self):
        super().__init__()
        self.mirror = ActorMirror()
        self.gru = nn.GRU(27, 128)
        self.mlp = nn.Sequential(nn.Linear(155, 256), nn.ELU(),
                                 nn.Linear(256, 124), nn.ELU(),
                                 nn.Linear(124, 64), nn.ELU())
        self.mu = nn.Linear(64, 3)
        self.beta = nn.Sequential(nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))
        self.logstd = nn.Parameter(torch.tensor([-.99, -.80, -1.39]))

    def forward(self, obs, mean, variance, hidden=None, done_before=None):
        if obs.ndim != 3:
            raise ValueError("expected [time,batch,27]")
        batch = obs.shape[1]
        if hidden is None:
            hidden = obs.new_zeros(2, 1, batch, 128)
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
            raw = self.mu(self.mlp(torch.cat((recurrent, pair), dim=-1)))
            left, right = raw.split(batch, dim=0)
            outputs.append((left + right * self.mirror.action_sign) * .5)
            b0, b1 = self.beta(recurrent).split(batch, dim=0)
            betas.append((b0 - b1) * .5)
        return torch.stack(outputs), self.logstd, torch.stack(betas), hidden


class CriticMirror(ActorMirror):
    """52-channel privileged contract with BODY velocity at indices 3 and 4.

    Never apply this mapping to legacy world-velocity critic observations.
    """
    def __init__(self):
        super().__init__()
        perm = torch.arange(52)
        for start in (14, 18, 22, 26, 30):
            perm[start:start + 4] = torch.tensor([start+1, start, start+3, start+2])
        sign = torch.ones(52)
        sign[[0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 30, 31, 32, 33, 34,
              38, 39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 51]] = -1
        self.permutation = perm
        self.sign = sign


class SharedMirrorCritic(nn.Module):
    """Independent shared-weight MLP with exactly invariant scalar output."""
    def __init__(self, units=(256, 128, 64)):
        super().__init__()
        self.mirror = CriticMirror()
        layers, width = [], 52
        for next_width in units:
            layers.extend((nn.Linear(width, next_width), nn.ELU()))
            width = next_width
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, states, mean, variance):
        x = (states if mean is None else
             self.mirror.normalize(states, mean, variance))
        n = x.shape[0]
        values = self.mlp(torch.cat((x, self.mirror(x)), dim=0))
        return (values[:n] + values[n:]) * .5
