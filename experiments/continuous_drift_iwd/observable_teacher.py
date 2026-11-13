"""Deployable-signal geometric teacher for the adaptive A0 curriculum.

The teacher deliberately does not read true speed, sideslip, tyre force or the
plant friction parameter.  Its grip hypothesis is a recurrent state updated
only from encoder speeds, previous tracking success and visible wheel overspeed.
It therefore defines a causal mapping that the production 27-channel GRU can
represent, unlike a teacher that labels identical observations with different
commands after reading privileged ``plant.col('D')``.
"""
import torch


class ObservableEnvelopeTeacher:
    def __init__(self, n, device, speed_margin=0.45, speed_cap=22.0,
                 mu_floor=1.0623, mu_ceil=2.1246,
                 explore_rate=0.18, retreat_rate=0.90):
        self.n = int(n)
        self.device = torch.device(device)
        self.speed_margin = float(speed_margin)
        self.speed_cap = float(speed_cap)
        self.mu_floor = float(mu_floor)
        self.mu_ceil = float(mu_ceil)
        self.explore_rate = float(explore_rate)
        self.retreat_rate = float(retreat_rate)
        self.mu_hat = torch.full((self.n,), self.mu_floor, device=self.device)
        self.prev_target = torch.full((self.n,), 2.0, device=self.device)

    def reset(self, mask=None):
        if mask is None:
            self.mu_hat.fill_(self.mu_floor)
            self.prev_target.fill_(2.0)
            return
        mask = mask.bool()
        self.mu_hat[mask] = self.mu_floor
        self.prev_target[mask] = 2.0

    @torch.no_grad()
    def command(self, env):
        # Every signal below is in the 27-channel production observation or is a
        # deterministic transform of its RTK/preview/encoder channels.
        wheel_v = env.s[:, 6:10] * env.wheel_radius
        v_front = wheel_v[:, :2].mean(1).clamp_min(0.0)
        rear_ratio = (wheel_v[:, 2:4].abs().mean(1) /
                      wheel_v[:, :2].abs().mean(1).clamp_min(0.5))
        track_good = ((env.signed_pos_diff.abs() < 0.75)
                      & (env.angle_diff.abs() < 0.15)
                      & ((rear_ratio - 1.0).abs() < 0.10))
        at_probe = v_front >= 0.85 * self.prev_target
        distress = ((env.signed_pos_diff.abs() > 1.25)
                    | (env.angle_diff.abs() > 0.28)
                    | ((rear_ratio - 1.0).abs() > 0.18))
        self.mu_hat.add_(env.dt * self.explore_rate * (track_good & at_probe).float())
        self.mu_hat.sub_(env.dt * self.retreat_rate * distress.float())
        self.mu_hat.clamp_(self.mu_floor, self.mu_ceil)

        kap = torch.maximum(env.kappa_ref.abs(),
                            env.kappa_preview.abs().amax(1)).clamp_min(1e-2)
        v_target = (self.speed_margin * torch.sqrt(
            self.mu_hat * 9.81 / kap)).clamp(2.0, self.speed_cap)
        self.prev_target.copy_(v_target)
        distress_level = torch.maximum(
            env.signed_pos_diff.abs() / 1.25,
            torch.maximum(env.angle_diff.abs() / 0.28,
                          (rear_ratio - 1.0).abs() / 0.18))
        longitudinal = (0.12 + 0.18 * (v_target - v_front)
                        - 0.55 * (distress_level - 1.0).clamp_min(0.0)).clamp(-1.0, 1.0)
        action = torch.stack((
            env.delta_pp.clamp(-env.max_steer, env.max_steer),
            longitudinal.clamp(min=0.0),
            (-longitudinal).clamp(min=0.0)), dim=1)
        diagnostics = {
            "mu_hat": self.mu_hat,
            "v_front": v_front,
            "v_target": v_target,
            "rear_ratio": rear_ratio,
            "distress": distress,
        }
        return action, diagnostics
