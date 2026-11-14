import torch
import math

class TrajGen:
    def __init__(self, num_waypoints: int = 4800, path_length: int = 24, device: str = "cuda"):
        self.num_waypoints = num_waypoints
        self.path_length = path_length
        self.device = device

    def generate_random_waypoints(self, n: int, theta0: float = 0.) -> torch.Tensor:
        # Waypoints along the path
        s = torch.linspace(0, self.path_length, self.num_waypoints, device=self.device)
        ds = s[1] - s[0]  # Step size

        num_scenes = n
        # Random segment lengths and curvature
        # Segment length in XCAR units; seg_scale lengthens the constant curvature
        # stretches, i.e. makes the reference a longer circular arc per segment
        # (curriculum: fewer curvature changes per episode).
        seg_scale = getattr(self, "seg_scale", 1.0)
        segment_length = (torch.rand((num_scenes, self.path_length // 3), device=self.device) * 5.0 + 3.0) * seg_scale

        threshold = torch.cumsum(segment_length, dim=1)  # (n, path_length)
        
        # XCAR units: |kappa| in [kmin, kmax] per 1 m of path.  ``curv_scale``
        # multiplies the curvature, i.e. divides the radii, so the reference can
        # be moved towards the larger drift circles used in real vehicle
        # experiments without touching the vehicle scaling.
        # curv_scale can be a scalar or a (lo, hi) pair; in the paired case every
        # scene draws its own radius so one bank spans the whole radius range
        # (domain randomisation over the trajectory size).
        curv_scale = getattr(self, "curv_scale", 1.0)
        if isinstance(curv_scale, (tuple, list)):
            lo, hi = curv_scale
            curv_scale = (torch.rand((num_scenes, 1), device=self.device) * (hi - lo) + lo)
        magnitude = (torch.rand_like(threshold, device=self.device) * 0.36 + 0.7) * curv_scale
        flip_prob = getattr(self, "flip_prob", None)
        if flip_prob is None:
            curvature = magnitude * torch.sign(torch.randn_like(threshold))
        else:
            # Correlated signs: an independent sign per segment makes a curvature
            # REVERSAL the default at every boundary, which is far harder than a
            # series of same-direction arcs.  A figure of eight stays in the
            # benchmark through the remaining probability.
            flip = torch.rand_like(threshold) < float(flip_prob)
            flip[:, 0] = False
            parity = torch.cumsum(flip.long(), dim=1) & 1
            init = torch.where(torch.rand((num_scenes, 1), device=self.device) < 0.5,
                               -torch.ones((num_scenes, 1), device=self.device),
                               torch.ones((num_scenes, 1), device=self.device))
            curvature = magnitude * init * (1.0 - 2.0 * parity.float())
        
        # Vectorised segment lookup. ``torch.searchsorted`` accepts a batched
        # sorted sequence, which replaces the per-scene python loop.  That loop
        # launched one kernel per scene and dominated the cost of every episode
        # reset once thousands of environments finished episodes in parallel.
        query = s.unsqueeze(0).expand(num_scenes, -1)
        seg = torch.searchsorted(threshold.contiguous(), query.contiguous()).clamp(max=threshold.shape[1] - 1)
        kappa = torch.gather(curvature, 1, seg)                        # (n, W)
        # (the reference sideslip is no longer stored on the path: it is a
        # function of the local curvature and the grip, computed by the env)

        # Curvature smoothing: the generator draws a piecewise CONSTANT curvature
        # with an independent random sign per segment, so kappa jumps
        # discontinuously at every boundary (a sign flip asks for a yaw rate jump
        # of about 2*V*kappa = 0.6-1.0 rad/s).  A moving average of width W turns
        # each jump into a linear ramp, i.e. a clothoid-like transition, which is
        # what a real reference would have.
        smooth_w = int(getattr(self, "curv_smooth", 0))
        if smooth_w > 1:
            import torch.nn.functional as Fn
            kern = torch.ones(1, 1, smooth_w, device=self.device) / smooth_w
            kappa = Fn.conv1d(kappa.unsqueeze(1), kern, padding=smooth_w // 2).squeeze(1)
            # an even kernel produces num_waypoints + 1 outputs, which silently
            # made the bank one waypoint longer than the path
            kappa = kappa[:, :self.num_waypoints]
        
        # Integrate curvature to get heading angle theta
        theta = torch.cumsum(kappa * ds, dim=1)  # Numeric integration of kappa to get angle
        
        # Starting conditions
        x0, y0, theta0 = 0, 0, theta0
        
        # Compute x, y positions by integrating the heading angle
        x = torch.cumsum(torch.cos(theta + theta0) * ds, dim=1) + x0
        y = torch.cumsum(torch.sin(theta + theta0) * ds, dim=1) + y0
        
        # Velocity direction only: the reference sideslip is NOT a path property
        # any more (it is a function of the local curvature and the grip, built by
        # the environment for the reward and the critic).
        vel_dir = torch.atan2(torch.sin(theta + theta0), torch.cos(theta + theta0))

        # Stack the output: [x, y, velocity direction, curvature]
        waypoints = torch.stack((x, y, vel_dir, kappa), dim=2)

        return waypoints

    def generate_closed_random_waypoints(self, n: int, radius: float = 1.0) -> torch.Tensor:
        """Smooth periodic route family for genuinely continuing rollouts.

        Low-amplitude Fourier modes perturb a circle while retaining exact
        positional/tangent/curvature continuity at the seam.  Coefficients are
        bounded to avoid self intersections.  Clockwise and counter-clockwise
        routes are both represented.
        """
        theta = torch.arange(self.num_waypoints, device=self.device,
                             dtype=torch.float32) * (2.0 * math.pi / self.num_waypoints)
        t = theta.unsqueeze(0)
        coeff = 2.0 * torch.rand((n, 4), device=self.device) - 1.0
        # Third derivatives amplify high harmonics quickly; these bounds keep
        # the resulting full-size curvature inside the steerable drift envelope.
        a2 = 0.04 * radius * coeff[:, 0:1]
        a3 = 0.01 * radius * coeff[:, 1:2]
        b2 = 0.04 * radius * coeff[:, 2:3]
        b3 = 0.01 * radius * coeff[:, 3:4]
        direction = torch.where(torch.rand((n, 1), device=self.device) < 0.5,
                                -torch.ones((n, 1), device=self.device),
                                torch.ones((n, 1), device=self.device))

        x = radius * torch.cos(t) + a2 * torch.cos(2.0 * t) + a3 * torch.sin(3.0 * t)
        y0 = radius * torch.sin(t) + b2 * torch.sin(2.0 * t) + b3 * torch.cos(3.0 * t)
        dx = -radius * torch.sin(t) - 2.0 * a2 * torch.sin(2.0 * t) + 3.0 * a3 * torch.cos(3.0 * t)
        dy0 = radius * torch.cos(t) + 2.0 * b2 * torch.cos(2.0 * t) - 3.0 * b3 * torch.sin(3.0 * t)
        ddx = -radius * torch.cos(t) - 4.0 * a2 * torch.cos(2.0 * t) - 9.0 * a3 * torch.sin(3.0 * t)
        ddy0 = -radius * torch.sin(t) - 4.0 * b2 * torch.sin(2.0 * t) - 9.0 * b3 * torch.cos(3.0 * t)
        y, dy, ddy = direction * y0, direction * dy0, direction * ddy0
        heading = torch.atan2(dy, dx)
        curvature = (dx * ddy - dy * ddx) / torch.clamp(dx * dx + dy * dy, min=1e-6).pow(1.5)
        fraction = float(getattr(self, "switch_fraction", 0.0))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("switch_fraction must be in [0,1]")
        if fraction > 0:
            # Smooth Gerono eight: finite tangent speed at the crossing and
            # continuous curvature reversal, unlike two joined circles.
            select = torch.rand((n, 1), device=self.device) < fraction
            ex = radius * torch.cos(t).expand(n, -1)
            ey = direction * 0.5 * radius * torch.sin(2*t)
            edx = -radius * torch.sin(t).expand(n, -1)
            edy = direction * radius * torch.cos(2*t)
            eddx = -radius * torch.cos(t).expand(n, -1)
            eddy = -2 * direction * radius * torch.sin(2*t)
            eh = torch.atan2(edy, edx)
            ek = (edx*eddy - edy*eddx) / (edx.square()+edy.square()).clamp_min(1e-6).pow(1.5)
            x, y = torch.where(select, ex, x), torch.where(select, ey, y)
            heading, curvature = torch.where(select, eh, heading), torch.where(select, ek, curvature)
        return torch.stack((x, y, heading, curvature), dim=2)

    @staticmethod
    def resample_closed(waypoints):
        """Uniform arclength samples, interpolating headings across +/-pi."""
        count = waypoints.shape[1]
        closed = torch.cat((waypoints, waypoints[:, :1]), dim=1)
        ds = torch.linalg.vector_norm(torch.diff(closed[..., :2], dim=1), dim=-1)
        arc = torch.cat((torch.zeros_like(ds[:, :1]), ds.cumsum(1)), dim=1)
        query = arc[:, -1:] * (torch.arange(count, device=waypoints.device) / count)
        right = torch.searchsorted(arc.contiguous(), query.contiguous(), right=True).clamp(1, count)
        left = right - 1
        a = torch.gather(closed, 1, left[..., None].expand(-1, -1, 4))
        b = torch.gather(closed, 1, right[..., None].expand(-1, -1, 4))
        fraction = ((query - arc.gather(1, left)) /
                    (arc.gather(1, right) - arc.gather(1, left)).clamp_min(1e-8))
        out = a + fraction[..., None] * (b-a)
        angle = torch.atan2(torch.sin(b[..., 2]-a[..., 2]), torch.cos(b[..., 2]-a[..., 2]))
        out[..., 2] = a[..., 2] + fraction * angle
        out[..., 2] = torch.atan2(torch.sin(out[..., 2]), torch.cos(out[..., 2]))
        return out
    def generate_custom_waypoints(self, segment_length: torch.Tensor, curvature: torch.Tensor, theta0: float = 0.) -> torch.Tensor:
        assert segment_length.shape[0] == curvature.shape[0], "segment_length and curvature must have the same number of scenes"

        # Waypoints along the path
        s = torch.linspace(0, self.path_length, self.num_waypoints, device=self.device)
        ds = s[1] - s[0] # Step size

        num_scenes = 1
        threshold = torch.cumsum(segment_length.unsqueeze(0), dim=1)
        curvature = curvature.unsqueeze(0)

        kappa = torch.zeros((num_scenes, self.num_waypoints), device=self.device)
        kappa[0] = curvature[0][torch.bucketize(s, threshold[0])]

        # Integrate curvature to get heading angle theta
        theta = torch.cumsum(kappa * ds, dim=1)  # Numeric integration of kappa to get angle

        # Starting conditions
        x0, y0, theta0 = 0, 0, theta0

        # Compute x, y positions by integrating the heading angle
        x = torch.cumsum(torch.cos(theta + theta0) * ds, dim=1) + x0
        y = torch.cumsum(torch.sin(theta + theta0) * ds, dim=1) + y0

        # Velocity direction only: the reference sideslip is NOT a path property
        # any more (it is a function of the local curvature and the grip, built by
        # the environment for the reward and the critic).
        vel_dir = torch.atan2(torch.sin(theta + theta0), torch.cos(theta + theta0))

        return waypoints
    
    def generate_variable_curvature_waypoints(self, theta0: float = 0.) -> torch.Tensor:
        # Define the boundaries for each segment
        s = torch.linspace(0, self.path_length, self.num_waypoints, device=self.device)
        ds = s[1] - s[0]  # Step size

        # length of each segment
        circle_length_small_start = 4.5 * torch.pi
        circle_length_small_mid = 5 * torch.pi
        circle_length_small_end = 4.5 * torch.pi
        sin_length = math.asin(1 / 2) * 8
        circle_length_large = (torch.pi - (math.asin(1 / 2) + math.sqrt(3) / 2 - 1) * 8)  * 2 * 2
        k = 8
        constant_curvature = 1 / 2

        # Define curvature as a function of s
        curvature = torch.zeros_like(s, device=self.device)

        # Segment 1: curvature = 1 for s in [0, 4.5pi]
        curvature[s <= circle_length_small_start] = 1

        # Segment 2: curvature = 1 - sin((s - 4.5pi) / k) for s in (4.5pi, 4.5pi + sin_length]
        curvature[(s > circle_length_small_start) & (s <= circle_length_small_start + sin_length)] = 1 - torch.sin((s[(s > circle_length_small_start) & (s <= circle_length_small_start + sin_length)] - circle_length_small_start) / k)

        # Segment 3: curvature = constant_curvature for s in (4.5pi + sin_length, 4.5pi + sin_length + circle_length_large]
        curvature[(s > circle_length_small_start + sin_length) & (s <= circle_length_small_start + sin_length + circle_length_large)] = constant_curvature

        # Segment 4: curvature = 1 + sin((s - (4.5pi + sin_length + circle_length_large + sin_length)) / k) for s in (4.5pi + sin_length + circle_length_large, 4.5pi + sin_length + circle_length_large + sin_length]
        curvature[(s > circle_length_small_start + sin_length + circle_length_large) & (s <= circle_length_small_start + sin_length + circle_length_large + sin_length)] = 1 + torch.sin((s[(s > circle_length_small_start + sin_length + circle_length_large) & (s <= circle_length_small_start + sin_length + circle_length_large + sin_length)] - (circle_length_small_start + sin_length + circle_length_large + sin_length)) / k)

        # Segment 5: curvature = 1 for s in (4.5pi + sin_length + circle_length_large + sin_length, 9.5pi + sin_length + circle_length_large + sin_length]
        curvature[(s > circle_length_small_start + sin_length + circle_length_large + sin_length) & (s <= circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid)] = 1

        # Segment 6: curvature = 1 - sin((s - 9pi - sin_length - circle_length_large - sin_length) / k) for s in (9pi + sin_length + circle_length_large + sin_length, 9pi + sin_length + circle_length_large + sin_length + sin_length]
        curvature[(s > circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid) & (s <= circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length)] = 1 - torch.sin((s[(s > circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid) & (s <= circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length)] - (circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid)) / k)

        # Segment 7: curvature = constant_curvature for s in (9pi + sin_length + circle_length_large + sin_length + sin_length, 9pi + sin_length + circle_length_large + sin_length + sin_length + circle_length_large]
        curvature[(s > circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length) & (s <= circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large)] = constant_curvature

        # Segment 8: curvature = 1 + sin((s - 9pi - sin_length - circle_length_large - sin_length - circle_length_large) / k) for s in (9pi + sin_length + circle_length_large + sin_length + sin_length + circle_length_large, 9pi + sin_length + circle_length_large + sin_length + sin_length + circle_length_large + sin_length]
        curvature[(s > circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large) & (s <= circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large + sin_length)] = 1 + torch.sin((s[(s > circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large) & (s <= circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large + sin_length)] - (circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large + sin_length)) / k)

        # Segment 9: curvature = 1 for s in (9pi + sin_length + circle_length_large + sin_length + sin_length + circle_length_large + sin_length, 13.5pi + sin_length + circle_length_large + sin_length + sin_length + circle_length_large + sin_length]
        curvature[s > circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large + sin_length] = 1

        # Initialize kappa (curvature values at waypoints)
        kappa = curvature.unsqueeze(0)

        # Numeric integration of curvature to get heading angle
        theta = torch.cumsum(kappa * ds, dim=1)  # Integrate curvature to compute heading

        # Initial position and angle
        x0, y0, theta0 = 0, 0, theta0

        # Compute x, y positions by integrating the heading angle
        x = torch.cumsum(torch.cos(theta + theta0) * ds, dim=1) + x0
        y = torch.cumsum(torch.sin(theta + theta0) * ds, dim=1) + y0

        # Velocity direction only: the reference sideslip is NOT a path property
        # any more (it is a function of the local curvature and the grip, built by
        # the environment for the reward and the critic).
        vel_dir = torch.atan2(torch.sin(theta + theta0), torch.cos(theta + theta0))

        return waypoints

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    
    ## three circles
    path_length = 12 * torch.pi
    num_waypoints = int(path_length / 0.005)

    traj_gen = TrajGen(num_waypoints=num_waypoints, path_length=path_length)

    segment_length = torch.tensor([(1 + 1 / 12) * 2 * torch.pi, (1 + 5 / 6) * 2 * torch.pi, (1 + 1 / 6) * 2 * torch.pi, (1 + 11 / 12) * 2 * torch.pi], device='cuda')

    curvature = torch.tensor([1, -1, 1, -1], device='cuda')

    waypoints = traj_gen.generate_custom_waypoints(segment_length, curvature)

    ## olympic rings
    path_length = 18 * torch.pi
    num_waypoints = int(path_length / 0.005)

    traj_gen = TrajGen(num_waypoints=num_waypoints, path_length=path_length)

    segment_length = torch.tensor([(1 + 1 / 4) * 2 * torch.pi, (1 + 5 / 6) * 2 * torch.pi, (1 + 1 / 3) * 2 * torch.pi, (1 + 2 / 3) * 2 * torch.pi, (1 + 1 / 6) * 2 * torch.pi, (1 + 3 / 4) * 2 * torch.pi], device='cuda')

    curvature = torch.tensor([1, -1, 1, -1, 1, -1], device='cuda')

    waypoints = traj_gen.generate_custom_waypoints(segment_length, curvature, theta0=-0.5 * torch.pi)

    # random racetrack
    traj_gen = TrajGen(num_waypoints=2400, path_length=24)
    waypoints = traj_gen.generate_random_waypoints(1)

    # variable curvature racetrack
    circle_length_small_start = 4.5 * torch.pi
    circle_length_small_mid = 5 * torch.pi
    circle_length_small_end = 4.5 * torch.pi
    sin_length = math.asin(1 / 2) * 8
    circle_length_large = (torch.pi - (math.asin(1 / 2) + math.sqrt(3) / 2 - 1) * 8)  * 2 * 2
    traj_gen = TrajGen(num_waypoints=int((circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large + sin_length + circle_length_small_end) / 0.005), path_length=(circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large + sin_length + circle_length_small_end))
    waypoints = traj_gen.generate_variable_curvature_waypoints()
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(waypoints[0, :, 0].cpu().numpy(), waypoints[0, :, 1].cpu().numpy())
    ax.set_aspect('equal')
    plt.savefig("racetrack.png")
