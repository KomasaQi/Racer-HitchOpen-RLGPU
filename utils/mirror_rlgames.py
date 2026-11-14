"""rl_games adapters for shared-call mirror actors and independent critics."""
import torch
from torch import nn
from rl_games.algos_torch import network_builder, model_builder, models
from utils.mirror_policy import SharedMirrorActor, SharedMirrorCritic


class SymmetricNormalization:
    def norm_obs(self, observation):
        if not self.normalize_input:
            return observation
        with torch.no_grad():
            # Maintain normal checkpoint statistics; pool their first/second
            # moments symmetrically for actual normalization on both branches.
            if self.training:
                self.running_mean_std(observation)
            return self.a2c_network.mirror.normalize(
                observation, self.running_mean_std.running_mean,
                self.running_mean_std.running_var).to(observation.dtype)


class MirrorActorModel(models.ModelA2CContinuousLogStd):
    class Network(SymmetricNormalization, models.ModelA2CContinuousLogStd.Network):
        pass


class MirrorValueModel(models.ModelCentralValue):
    class Network(SymmetricNormalization, models.ModelCentralValue.Network):
        pass


class MirrorActorBuilder(network_builder.A2CBuilder):
    class Network(nn.Module):
        def __init__(self, params, **kwargs):
            super().__init__()
            if tuple(kwargs["input_shape"]) != (27,) or kwargs["actions_num"] != 3:
                raise ValueError("mirror actor requires 27 observable inputs and 3 pedal actions")
            self.core = SharedMirrorActor()
            self.num_seqs = kwargs.get("num_seqs", 1)
            self.beta_aux_weight = float(params.get("beta_aux_weight", .1))
            self.beta_labels = self.beta_mask = self.beta_prediction = None
            self.value = nn.Linear(64, 1)  # API-only: independent critic supplies PPO values.
            with torch.no_grad():
                initial = params.get("initial_action_mean", [0., 0., -1.])
                self.core.mu.bias.copy_(torch.tensor(initial))
                sigma = params["space"]["continuous"]["sigma_init"]["val"]
                self.core.logstd.fill_(sigma)

        @property
        def mirror(self): return self.core.mirror
        @property
        def sigma(self): return self.core.logstd
        @property
        def beta_head(self): return self.core.beta
        def is_rnn(self): return True
        def is_separate_critic(self): return False
        def get_value_layer(self): return self.value
        def get_default_rnn_state(self):
            return tuple(torch.zeros(1, self.num_seqs, 128) for _ in range(2))

        def forward(self, data):
            obs = data["obs"]
            length = data.get("seq_length", 1)
            count = obs.shape[0] // length
            seq = obs.reshape(count, length, 27).transpose(0, 1)
            states = data.get("rnn_states")
            hidden = torch.stack(states) if states is not None and len(states) else None
            done = data.get("dones")
            if done is not None:
                done = done.reshape(count, length).transpose(0, 1)
            mu, logstd, beta, hidden = self.core(seq, None, None, hidden, done)
            mu = mu.transpose(0, 1).reshape(-1, 3)
            self.beta_prediction = (beta.transpose(0, 1).reshape(-1, 1)
                                    if self.beta_labels is not None else None)
            # The actor never learns a second value function.
            value = mu[:, :1] * 0.
            return mu, logstd.expand_as(mu), value, tuple(hidden.unbind(0))

        def get_aux_loss(self):
            if self.beta_labels is None: return None
            from utils.drift_network import beta_supervision
            return {"beta": self.beta_aux_weight * beta_supervision(
                self.beta_prediction, self.beta_labels, self.beta_mask)}

    def build(self, name, **kwargs): return self.Network(self.params, **kwargs)


class MirrorValueBuilder(network_builder.A2CBuilder):
    class Network(nn.Module):
        def __init__(self, params, **kwargs):
            super().__init__()
            if tuple(kwargs["input_shape"]) != (52,):
                raise ValueError("mirror critic requires 52 body-frame privileged channels")
            self.core = SharedMirrorCritic(params["mlp"]["units"])
        @property
        def mirror(self): return self.core.mirror
        def is_rnn(self): return False
        def get_default_rnn_state(self): return None
        def get_value_layer(self): return self.core.mlp[-1]
        def get_aux_loss(self): return None
        def forward(self, data):
            return self.core(data["obs"], None, None), data.get("rnn_states")
    def build(self, name, **kwargs): return self.Network(self.params, **kwargs)


def register():
    model_builder.register_network("drift_mirror_actor", MirrorActorBuilder)
    model_builder.register_network("drift_mirror_value", MirrorValueBuilder)
    model_builder.register_model("drift_mirror_actor", MirrorActorModel)
    model_builder.register_model("drift_mirror_value", MirrorValueModel)
