"""Observable recurrent actor with training-only sideslip supervision.

The labels travel through the PPO dataset, never through the model's inputs.
Uses rl_games' recurrent sequence ordering and done masks without a PPO fork.
"""
import torch
from torch import nn
from torch.nn import functional as F
from rl_games.algos_torch import network_builder, model_builder
from rl_games.algos_torch.a2c_continuous import A2CAgent


def beta_supervision(prediction, labels, mask=None, low=1.0, high=4.0):
    """Wrapped Huber loss, continuously faded to zero below 1 m/s.

    Divide by valid sample count, NOT sum of speed weights: otherwise fading
    would cancel when all samples have similar low speed.
    """
    beta, speed = labels.unbind(-1)
    fraction = ((speed - low) / (high - low)).clamp(0.0, 1.0)
    weight = fraction.square() * (3.0 - 2.0 * fraction)
    error = prediction.float().squeeze(-1) - beta
    error = torch.atan2(torch.sin(error), torch.cos(error))
    loss = F.huber_loss(error, torch.zeros_like(error), reduction="none", delta=0.2)
    valid = torch.ones_like(weight) if mask is None else mask.reshape(-1).float()
    return (loss * weight * valid).sum() / valid.sum().clamp_min(1.0)


class DriftBuilder(network_builder.A2CBuilder):
    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            if (self.separate or not self.has_rnn or not self.is_rnn_before_mlp
                    or not self.rnn_concat_output or self.rnn_layers != 1):
                raise ValueError("drift actor requires one GRU before MLP and observation bypass")
            initial_mean = params.get("initial_action_mean")
            if initial_mean is not None:
                bias = torch.as_tensor(initial_mean, dtype=self.mu.bias.dtype)
                if (bias.shape != self.mu.bias.shape or not torch.isfinite(bias).all()
                        or (bias.abs() > 1).any()):
                    raise ValueError("initial_action_mean must match actions and lie in [-1, 1]")
                # Initialization only: no runtime offset, mask or frozen brake.
                # Loading a checkpoint replaces this like any other parameter.
                with torch.no_grad():
                    self.mu.bias.copy_(bias)
            self.beta_head = nn.Sequential(nn.Linear(self.rnn_units, 64),
                                           nn.ELU(), nn.Linear(64, 1))
            self.beta_aux_weight = float(params.get("beta_aux_weight", 0.1))
            self.beta_labels = None
            self.beta_mask = None
            self.beta_prediction = None
            self.actor_mlp.register_forward_pre_hook(self._capture_beta)

        def _capture_beta(self, module, inputs):
            # rl_games concat_output order is [GRU output, normalised obs].
            # Nothing from the privileged critic is part of this tensor.
            if self.beta_labels is not None:
                self.beta_prediction = self.beta_head(inputs[0][:, :self.rnn_units])

        def get_aux_loss(self):
            if self.beta_labels is None:
                return None
            return {"beta": self.beta_aux_weight * beta_supervision(
                self.beta_prediction, self.beta_labels, self.beta_mask)}

    def build(self, name, **kwargs):
        return self.Network(self.params, **kwargs)


class DriftAgent(A2CAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.has_central_value:
            raise ValueError("drift PPO requires the independent privileged critic")
        if self.is_adaptive_lr:
            # rl_games AdaptiveScheduler ignores min/max_learning_rate config
            # and otherwise silently expands to [1e-6, 1e-2]. Wire them here.
            import math
            initial = float(self.config["learning_rate"])
            low = float(self.config.get("min_learning_rate", initial * .1))
            high = float(self.config.get("max_learning_rate", initial))
            if not (all(math.isfinite(v) for v in (low, initial, high))
                    and 0 < low <= initial <= high):
                raise ValueError("require 0 < min_learning_rate <= learning_rate <= max_learning_rate")
            self.scheduler.min_lr, self.scheduler.max_lr = low, high
            print(f"Effective adaptive actor LR bounds: [{low:g}, {high:g}]")


    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        # These are pre-normalisation critic states stored at the SAME time
        # as actor observations, before the action and before any respawn.
        scale = self.vec_env.env.priv_scale[:2]
        self.dataset.values_dict["beta_labels"] = (
            batch_dict["states"][:, :2] / scale).detach()

    def calc_gradients(self, input_dict):
        net = self.model.a2c_network
        net.beta_labels = input_dict["beta_labels"]
        net.beta_mask = input_dict.get("rnn_masks")
        try:
            return super().calc_gradients(input_dict)
        finally:
            net.beta_labels = net.beta_mask = net.beta_prediction = None


def register(runner):
    from utils.mirror_rlgames import register as register_mirror
    register_mirror()
    model_builder.register_network("drift_observable", DriftBuilder)
    runner.algo_factory.register_builder(
        "drift_ppo", lambda **kwargs: DriftAgent(**kwargs))
