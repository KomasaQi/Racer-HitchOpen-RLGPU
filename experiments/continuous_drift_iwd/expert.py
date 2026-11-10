"""Hand crafted drift expert for continuous_drift_iwd.

Uses the privileged simulation state (yaw rate, sideslip, speed and the tracking
errors) to produce the pedal commands: a sideslip / yaw rate feedback loop that
stabilises the drift, plus a path tracking term so the car stays on the scaled
XCAR reference.  Its only purpose is to generate demonstrations for behaviour
cloning and as a sanity reference for what the task should look like.

The policy never sees any of these quantities, so the cloned network still has
to infer them from the observation history.
"""
import torch


def expert_action(env, v_target=6.5, k_ff=0.6, kb=1.2, kr=-0.5, kp=0.10, kd=-0.6,
                  q0=0.55, kv=0.08, steer_max=None):
    beta = env.es[:, 7]
    r = env.es[:, 2]
    v = env.es[:, 6]
    e_pos = env.signed_pos_diff
    e_dir = env.angle_diff
    beta_ref = env.beta_ref
    r_ref = v * env.kappa_ref

    # slow down a little when the course error is large
    v_ref = v_target * (1.0 - 0.25 * torch.tanh(2.0 * e_dir.abs()))
    # k_ff is the feed-forward countersteer that holds the drift equilibrium
    # (delta ~ beta_ref + lf/R - alpha_f); without it the feedback has zero
    # countersteer exactly at the reference sideslip and the car spins.
    steer = k_ff * beta_ref + kb * (beta - beta_ref) + kr * (r - r_ref) \
        + kp * e_pos + kd * e_dir
    m = steer_max if steer_max is not None else env.max_steer
    steer = torch.clamp(steer, -m, m)
    q = torch.clamp(kv * (v_ref - v) + q0, 0.0, 1.0)
    return torch.stack([steer, q, torch.zeros_like(q)], dim=1)
