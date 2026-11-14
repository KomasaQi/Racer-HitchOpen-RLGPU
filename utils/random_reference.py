"""Batched smooth random open roads in metres, radians and inverse metres.

Chunks include both endpoints. Pass the final [x,y,heading,curvature] row as
``start`` for continuation, dropping the duplicate joint when concatenating.
Bounds are geometric; they do not certify vehicle dynamic feasibility.
"""
import math
import torch


def generate_reference_chunk(start, *, points=2513, length=400., segments=6,
                             curvature_limit=.09, curvature_rate_limit=.006,
                             generator=None, transition_min_m=None):
    if start.ndim != 2 or start.shape[1] != 4 or not start.is_floating_point():
        raise ValueError('start must be a floating [batch,4] tensor')
    if points < 3 or segments < 1 or curvature_limit <= 0 or curvature_rate_limit <= 0:
        raise ValueError('Invalid geometry bounds')
    if not bool(torch.isfinite(start).all()) or bool((start[:, 3].abs() > curvature_limit).any()):
        raise ValueError('Initial state must be finite and curvature within bounds')
    # Optional short spans clip each sampled curvature change to its own
    # quintic derivative bound. The default retains the original distribution.
    # Maximum derivative of quintic smoothstep is 1.875. Reserve enough
    # distance for even a full -limit to +limit reversal in every segment.
    minimum = (1.875 * 2 * curvature_limit / curvature_rate_limit if transition_min_m is None else float(transition_min_m))
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError('transition_min_m must be positive')
    if length < minimum * segments:
        raise ValueError('Chunk too short for requested segments and curvature-rate bound')
    n = start.shape[0]
    def rand(*shape):
        return torch.rand(shape, device=start.device, dtype=start.dtype, generator=generator)
    weights = rand(n, segments) + .1
    spans = minimum + (length-minimum*segments)*weights/weights.sum(1, keepdim=True)
    ends = spans.cumsum(1)
    knots = torch.cat((start.new_zeros(n, 1), ends), 1)
    targets = (rand(n, segments)*2-1)*curvature_limit
    targets = torch.where(rand(n, segments)<.2, torch.zeros_like(targets), targets)
    if transition_min_m is not None:
        bounded=[]
        previous=start[:,3]
        for j in range(segments):
            max_delta=curvature_rate_limit*spans[:,j]/1.875
            previous=previous+(targets[:,j]-previous).clamp(-max_delta,max_delta)
            bounded.append(previous)
        targets=torch.stack(bounded,1)
    values = torch.cat((start[:, 3:4], targets), 1)
    arc = torch.linspace(0, length, points, device=start.device, dtype=start.dtype)
    query = arc[None].expand(n, -1).contiguous()
    index = torch.searchsorted(ends.contiguous(), query).clamp(max=segments-1)
    left = knots.gather(1, index)
    span = spans.gather(1, index)
    u = ((query-left)/span).clamp(0, 1)
    blend = u**3*(10+u*(-15+6*u))
    ka, kb = values.gather(1,index), values.gather(1,index+1)
    curvature = ka+(kb-ka)*blend
    # Force exact endpoints despite floating-point cumsum/division roundoff.
    curvature[:, 0] = start[:, 3]
    curvature[:, -1] = targets[:, -1]
    ds = length/(points-1)
    heading = torch.cat((start[:, 2:3], start[:, 2:3]+torch.cumsum(
        .5*(curvature[:, 1:]+curvature[:, :-1])*ds, 1)), 1)
    x = torch.cat((start[:, 0:1], start[:, 0:1]+torch.cumsum(
        .5*(heading[:, 1:].cos()+heading[:, :-1].cos())*ds, 1)), 1)
    y = torch.cat((start[:, 1:2], start[:, 1:2]+torch.cumsum(
        .5*(heading[:, 1:].sin()+heading[:, :-1].sin())*ds, 1)), 1)
    return torch.stack((x,y,heading,curvature), dim=2)


class RollingReference:
    """Independent two-chunk windows; caller remaps progress without reset.

    Only consumed chunks are discarded. Active overlap retains exact samples,
    so local projection and previews do not jump during a window shift.
    This storage object does not alter plant, policy, reward or sensor state.
    """

    def __init__(self, start, *, points=513, length=120., segments=2,
                 curvature_limit=.09, curvature_rate_limit=.006, generator=None, transition_min_m=None):
        self.options = dict(points=points, length=length, segments=segments,
                            curvature_limit=curvature_limit,
                            curvature_rate_limit=curvature_rate_limit, generator=generator,
                            transition_min_m=transition_min_m)
        self.stride = points-1
        self.ds = length/self.stride
        first = generate_reference_chunk(start, **self.options)
        second = generate_reference_chunk(first[:, -1], **self.options)
        self.roads = torch.cat((first, second[:, 1:]), dim=1)
        self.offset = torch.zeros(start.shape[0], device=start.device, dtype=torch.long)

    def advance(self, progress, *, retained_behind=32, active=None):
        """Return remapped indices and shifted mask; don't reset episode state."""
        if retained_behind < 1 or retained_behind >= self.stride:
            raise ValueError('retained_behind must lie inside one chunk')
        if progress.shape != self.offset.shape:
            raise ValueError('One progress index required per reference')
        if bool(((progress < 0) | (progress >= self.roads.shape[1])).any()):
            raise ValueError('Progress outside available reference')
        shifted = progress >= self.stride + retained_behind
        if active is not None:
            shifted &= active
        ids = shifted.nonzero(as_tuple=False).squeeze(1)
        if ids.numel():
            next_chunk = generate_reference_chunk(self.roads[ids, -1], **self.options)
            retained = self.roads[ids, self.stride:].clone()
            self.roads[ids] = torch.cat((retained, next_chunk[:, 1:]), dim=1)
            self.offset[ids] += self.stride
        return progress - shifted.long()*self.stride, shifted



