"""Install a physical-unit reference at rest, without editing vehicle state."""
import numpy as np
import torch


def install_reference(env, filename, *, closed=True):
    if getattr(env, '_streaming_reference', False):
        raise ValueError('Fixed fixtures require a non-streaming environment')
    if bool((env.step_count != 0).any()):
        raise ValueError('Reference fixtures must be installed immediately after reset')
    # Dynamic resets derive speed, beta, yaw rate and wheel slip from the old
    # random path. Rotating a replacement fixture to body heading does not make
    # those states consistent with its tangent/curvature. Require a genuine
    # rest start until a fixture-aware dynamic initialization is implemented.
    if bool((env.s[:, 3:].abs() > 1e-4).any()):
        raise ValueError('Reference fixtures require a rest state; dynamic resets '
                         'must be initialized from the fixture itself')
    if closed and not env._cyclic_reference:
        raise ValueError('Reference fixtures require a cyclic task')
    if not closed and env.preview_distances_m is None and env.preview_time is None:
        raise ValueError('Open reference fixtures require physical distance or time previews')
    data = np.load(filename, allow_pickle=False)
    if data.shape != (env.num_waypoints, 4) or not np.isfinite(data).all():
        raise ValueError('Expected finite [waypoints, x/y/heading/curvature] array')
    path = torch.as_tensor(data, dtype=env.s.dtype, device=env.device)
    xy = path[:, :2] - path[0, :2]
    angle = env.s[:, 2] - path[0, 2]
    c, s = angle.cos()[:, None], angle.sin()[:, None]
    roads = path[None].repeat(env.n, 1, 1)
    roads[:, :, 0] = c*xy[:, 0] - s*xy[:, 1] + env.s[:, 0:1]
    roads[:, :, 1] = s*xy[:, 0] + c*xy[:, 1] + env.s[:, 1:2]
    roads[:, :, 2] = torch.atan2(torch.sin(path[:, 2]+angle[:, None]),
                                torch.cos(path[:, 2]+angle[:, None]))
    if env.n > env.num_paths:
        raise ValueError('Need at least one path slot per evaluation environment')
    env.path_bank[env.current_generation, :env.n].copy_(roads)
    differences = (torch.roll(roads[:, :, :2], -1, dims=1)-roads[:, :, :2]
                   if closed else roads[:, 1:, :2]-roads[:, :-1, :2])
    spacing = torch.linalg.vector_norm(differences, dim=2).mean(1)
    env._path_ds[env.current_generation, :env.n] = spacing
    env._cyclic_reference = bool(closed)
    if not closed:
        env.waypoint_ds = float(spacing[0])
    env.path_id[:] = torch.arange(env.n, device=env.device)
    env.path_generation.fill_(env.current_generation)
    env.progress.zero_()
    env.step_progress.zero_()
    env._now = env._prev = None
    env._projection_xy = env.s[:, :2].clone()
    env._tracking_mask = None
    env._track_pending = True
    env._physics_advanced = False
    env._sensor_pending.fill_(True)
    env._sensor_needs_fill.fill_(True)
    return env.obs()
