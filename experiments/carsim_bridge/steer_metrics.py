"""Steering smoothness metrics, as specified in the review.

Raw sign changes are misleading: during a one-direction drift the angle can stay on
one side of zero while still chattering.  These metrics are computed on the APPLIED
road-wheel angle and use excursion + dwell thresholds so that a real countersteer is
not counted as chatter.
"""
import numpy as np


def _lowpass(x, dt, fc):
    """First order low pass, used to strip the chatter band before analysis."""
    a = dt / (1.0 / (2.0 * np.pi * fc) + dt)
    y = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc += a * (v - acc)
        y[i] = acc
    return y


def steering_metrics(delta, dt=0.05, fc=3.0, min_excursion=0.02,
                     min_interval=0.10):
    """``delta`` is the applied road-wheel angle in radians over one episode."""
    d = np.asarray(delta, dtype=np.float64)
    if d.size < 3:
        return {"tv": 0.0, "p95_rate": 0.0, "reversals": 0.0, "hf_ratio": 0.0,
                "mean_step": 0.0}
    smooth = _lowpass(d, dt, fc)
    rate = np.diff(smooth) / dt
    duration = max((d.size - 1) * dt, 1e-6)

    tv = float(np.abs(np.diff(d)).sum() / duration)
    p95_rate = float(np.percentile(np.abs(rate), 95))
    mean_step = float(np.abs(np.diff(d)).mean())

    # meaningful reversals: sign flip of the rate, with a real excursion since the
    # previous reversal and a minimum dwell time
    n_rev = 0
    last_rev_i = 0
    prev_sign = 0
    for i, r in enumerate(rate):
        s = 0 if r == 0 else (1 if r > 0 else -1)
        if s == 0:
            continue
        if prev_sign != 0 and s != prev_sign:
            if (abs(smooth[i] - smooth[last_rev_i]) >= min_excursion
                    and (i - last_rev_i) * dt >= min_interval):
                n_rev += 1
                last_rev_i = i
        prev_sign = s
    reversals = n_rev / duration

    # high-frequency energy: power above fc over total power of the rate signal
    if rate.size > 8:
        spec = np.abs(np.fft.rfft(rate - rate.mean())) ** 2
        freqs = np.fft.rfftfreq(rate.size, dt)
        hf = float(spec[freqs > fc].sum() / max(spec.sum(), 1e-12))
    else:
        hf = 0.0
    return {"tv": tv, "p95_rate": p95_rate, "reversals": reversals,
            "hf_ratio": hf, "mean_step": mean_step}


TARGETS = {"reversals": 1.5, "tv": 0.6, "p95_rate": 1.5, "hf_ratio": 0.05,
           "mean_step": 0.03}


def format_metrics(m):
    parts = []
    for k in ("reversals", "tv", "p95_rate", "hf_ratio", "mean_step"):
        ok = "ok" if m[k] <= TARGETS[k] else "HIGH"
        parts.append("%s %.3f/%.2f %s" % (k, m[k], TARGETS[k], ok))
    return "  ".join(parts)
