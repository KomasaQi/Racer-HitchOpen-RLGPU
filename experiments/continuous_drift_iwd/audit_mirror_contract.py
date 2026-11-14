"""Mirror architecture contract: sequence symmetry, masks and likelihoods.

Usage: python experiments/continuous_drift_iwd/audit_mirror_contract.py --output PATH
This tests semantics, not successful driving or sim-to-sim transfer.
"""
import argparse
import json
import sys
from pathlib import Path
import torch
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.mirror_policy import SharedMirrorActor, SharedMirrorCritic
from utils.mirror_rlgames import MirrorActorBuilder, MirrorActorModel


def check():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    actor = SharedMirrorActor()
    obs = torch.randn(9, 7, 27)
    mean, var = torch.randn(27), torch.rand(27) + .1
    hidden = torch.randn(2, 1, 7, 128)
    done = torch.rand(9, 7) < .2
    mu, _, beta, h = actor(obs, mean, var, hidden, done)
    mirrored, _, mirrored_beta, mirrored_h = actor(
        actor.mirror(obs), mean, var, hidden.flip(0), done)
    errors = dict(action=float((mirrored-mu*actor.mirror.action_sign).abs().max()),
                  beta=float((mirrored_beta+beta).abs().max()),
                  hidden=float((mirrored_h-h.flip(0)).abs().max()))
    assert max(errors.values()) < 2e-6, errors
    assert torch.equal(actor.mirror(actor.mirror(obs)), obs)
    stream, h_stream = [], hidden
    for t in range(len(obs)):
        value, _, _, h_stream = actor(obs[t:t+1], mean, var, h_stream, done[t:t+1])
        stream.append(value)
    torch.testing.assert_close(torch.cat(stream), mu)
    (mu.square().mean()+beta.square().mean()).backward()
    assert actor.gru.weight_ih_l0.grad.abs().sum() > 0
    critic = SharedMirrorCritic()
    states, cm, cv = torch.randn(19, 52), torch.randn(52), torch.rand(52)+.1
    value = critic(states, cm, cv)
    critic_error = float((value-critic(critic.mirror(states), cm, cv)).abs().max())
    assert critic_error < 2e-6
    value.square().mean().backward()
    assert critic.mlp[0].weight.grad.abs().sum() > 0
    assert not ({id(p) for p in actor.parameters()} & {id(p) for p in critic.parameters()})

    cfg = yaml.safe_load((ROOT/"runner_config.yaml").read_text())["params"]["network"]
    builder = MirrorActorBuilder(); builder.load(cfg)
    model = MirrorActorModel(builder).build(dict(actions_num=3, input_shape=(27,),
        num_seqs=7, value_size=1, normalize_input=True, normalize_value=True))
    model.eval()
    flat = obs.transpose(0, 1).reshape(-1, 27)
    actions = torch.randn(len(flat), 3)
    masks = done.transpose(0, 1).reshape(-1)
    h0 = tuple(hidden.unbind(0))
    mirror = model.a2c_network.mirror
    with torch.no_grad():
        a = model(dict(obs=flat.clone(), prev_actions=actions, rnn_states=h0,
                       dones=masks, seq_length=9, is_train=True))
        b = model(dict(obs=mirror(flat), prev_actions=actions*mirror.action_sign,
                       rnn_states=h0[::-1], dones=masks, seq_length=9, is_train=True))
    likelihood_error = float((a["prev_neglogp"]-b["prev_neglogp"]).abs().max())
    assert likelihood_error < 2e-5
    torch.testing.assert_close(b["mus"], a["mus"]*mirror.action_sign, atol=2e-6, rtol=0)
    return dict(actor_errors=errors, critic_error=critic_error,
                likelihood_error=likelihood_error, sequence_streaming_equal=True,
                gradients_nonzero=True, independent_critic=True,
                scope="Architecture only; body-frame critic contract required")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = check()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
