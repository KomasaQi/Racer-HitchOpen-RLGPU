# -*- coding: utf-8 -*-
"""FastPlant sanity checks.

Run from ``xcar-simulation``::

    python plant_check.py [preset]

The script prints a compact table for each probe:

* straight line acceleration (burn-out vs. slip limited launch, top speed)
* braking from 100 km/h
* steady state cornering (radius, lateral g, left/right and front/rear loads)
* peak lateral acceleration sweep
* limited slip differential action (left/right wheel speed and torque split)
* aerodynamics (drag and downforce at constant speed)
* wheel spin transient under a throttle step
* GPU throughput

Nothing here is a CarSim validation; it only proves that every module of the
plant produces monotone, physically plausible numbers before an RL run.
"""

from __future__ import annotations

import sys
import time

import torch

from fast_plant import FastRacerPlant, preset_parameters, resolve_params


def new_plant(preset: str, device="cuda:0", n: int = 1, overrides: dict | None = None):
    params = preset_parameters(preset)
    if overrides:
        params.update(overrides)
        params = resolve_params(params)
    return FastRacerPlant(params, n, device=device)


def cmd(steer: float, throttle: float, brake: float, device) -> torch.Tensor:
    return torch.tensor([[steer, throttle, brake]], device=device, dtype=torch.float32)


def speed_of(state) -> float:
    return float(torch.hypot(state[0, 3], state[0, 4]))


def probe_acceleration(preset, device):
    dt = 0.01

    # (a) full throttle from rest: pure burn-out, no driver model
    plant = new_plant(preset, device)
    s = plant.zero_state(1)
    plant.init_wheels(s)
    t20 = None
    vmax = 0.0
    for i in range(3000):
        s, ex = plant.step(s, cmd(0.0, 1.0, 0.0, device), dt=dt, substeps=1)
        v = speed_of(s)
        vmax = max(vmax, v)
        if t20 is None and v >= 20.0:
            t20 = (i + 1) * dt
    print(f"  full throttle from rest: t(20 m/s)={t20 if t20 else float('nan'):5.2f} s  "
          f"v_max={vmax * 3.6:6.1f} km/h  (burn-out expected)")

    # (b) traction limited launch: PI on the rear slip ratio target
    plant = new_plant(preset, device)
    s = plant.zero_state(1)
    plant.init_wheels(s)
    q = 0.0
    t100 = None
    vmax = 0.0
    for i in range(6000):
        s, ex = plant.step(s, cmd(0.0, q, 0.0, device), dt=dt, substeps=1)
        slip_r = float(ex["sx"][0, 2:].mean())
        q = max(0.0, min(1.0, q + 30.0 * (slip_r + 0.10) * dt))
        v = speed_of(s)
        vmax = max(vmax, v)
        if t100 is None and v * 3.6 >= 100.0:
            t100 = (i + 1) * dt
    print(f"  slip limited launch  : t(100 km/h)={t100 if t100 else float('nan'):5.2f} s  "
          f"v_max={vmax * 3.6:6.1f} km/h (target rear slip -0.10)")


def probe_braking(preset, device, v0=27.78):
    for level in (0.5, 1.0):
        plant = new_plant(preset, device)
        s = plant.zero_state(1)
        s[0, 3] = v0
        plant.init_wheels(s)
        t = 0.0
        dt = 0.01
        peak = 0.0
        while t < 8.0 and float(s[0, 3]) > 0.5:
            s, ex = plant.step(s, cmd(0.0, 0.0, level, device), dt=dt, substeps=1)
            peak = min(peak, float(ex["ax"]))
            t += dt
        print(f"  brake {level:.1f}: 100-0 km/h in {t:5.2f} s "
              f"({(v0 / max(t, 1e-3) / 9.81):4.2f} g avg, peak {abs(peak) / 9.81:4.2f} g)")


def probe_cornering(preset, device):
    """Steady state cornering at a target lateral acceleration."""
    params = preset_parameters(preset)
    wheelbase = params["lF"] + params["lR"]
    for v0, ay_target in ((8.0, 2.0), (12.0, 4.0), (18.0, 6.0), (25.0, 8.0)):
        radius = v0 * v0 / ay_target
        steer = wheelbase / radius
        plant = new_plant(preset, device)
        s = plant.zero_state(1)
        s[0, 3] = v0
        plant.init_wheels(s)
        q = 0.0
        for _ in range(800):
            v = speed_of(s)
            q = max(0.0, min(1.0, q + 0.4 * (v0 - v)))
            s, ex = plant.step(s, cmd(steer, q, 0.0, device), dt=0.01, substeps=1)
        v = speed_of(s)
        r = float(s[0, 5])
        ay = v * r
        fz = [round(float(x)) for x in ex["fz"][0]]
        print(f"  steer {steer:5.3f} rad @ {v0:4.1f} m/s target: v={v:5.2f} r={r:6.3f} "
              f"ay={ay:5.2f} m/s2 radius={abs(v / r) if abs(r) > 1e-4 else float('inf'):6.1f} m "
              f"Fz={fz}")


def probe_lateral_limit(preset, device):
    """Ramp the steering at constant speed and find the peak lateral g."""
    for v0 in (15.0, 25.0):
        plant = new_plant(preset, device)
        s = plant.zero_state(1)
        s[0, 3] = v0
        plant.init_wheels(s)
        best_ay = 0.0
        best_steer = 0.0
        q = 0.0
        for i in range(1200):
            steer = min(0.45, i * 0.0004)
            v = speed_of(s)
            q = max(0.0, min(1.0, q + 0.4 * (v0 - v)))
            s, ex = plant.step(s, cmd(steer, q, 0.0, device), dt=0.01, substeps=1)
            ay = abs(float(ex["ay"]))
            if ay > best_ay:
                best_ay, best_steer = ay, steer
        print(f"  speed {v0:4.1f} m/s: peak |ay|={best_ay:5.2f} m/s2 ({best_ay / 9.81:4.2f} g) "
              f"at steer {best_steer:5.3f} rad")


def probe_lsd(preset, device):
    for kd, cap in ((0.0, 0.0), (20.0, 300.0), (200.0, 1200.0)):
        plant = new_plant(preset, device, overrides={"lsd_kd": kd, "lsd_tau_max": cap})
        s = plant.zero_state(1)
        s[0, 3] = 8.0
        plant.init_wheels(s)
        for _ in range(50):
            s, ex = plant.step(s, cmd(0.35, 1.0, 0.0, device), dt=0.01, substeps=1)
        wrl, wrr = float(s[0, 8]), float(s[0, 9])
        tl, tr = float(ex["drive_torque"][0, 2]), float(ex["drive_torque"][0, 3])
        print(f"  kd={kd:5.1f} cap={cap:6.1f}: omega RL={wrl:7.2f} RR={wrr:7.2f} "
              f"d={wrl - wrr:6.2f}  T RL={tl:7.0f} RR={tr:7.0f} "
              f"coupling={float(ex['coupling'][0]):6.0f}")


def probe_aero(preset, device):
    for label, over in (("aero on ", {}),
                        ("drag off", {"aero_Cdx": 0.0, "aero_Clx": 0.0}),
                        ("aero off", {"aero_Cdx": 0.0, "aero_Cdy": 0.0,
                                      "aero_Clx": 0.0, "aero_Cly": 0.0})):
        plant = new_plant(preset, device, overrides=over)
        s = plant.zero_state(1)
        s[0, 3] = 60.0
        plant.init_wheels(s)
        for _ in range(300):
            s, ex = plant.step(s, cmd(0.0, 1.0, 0.0, device), dt=0.01, substeps=1)
        print(f"  {label}: v={float(s[0, 3]) * 3.6:6.1f} km/h  drag={float(ex['aero_fx']):7.0f} N  "
              f"downforce={-float(ex['aero_fz']):7.0f} N  sum Fz={float(ex['fz'][0].sum()):7.0f} N")


def probe_wheel_spin(preset, device):
    print("  throttle step at 10 m/s (rear left wheel):")
    for level in (0.25, 1.0):
        plant = new_plant(preset, device)
        s = plant.zero_state(1)
        s[0, 3] = 10.0
        plant.init_wheels(s)
        for k in range(6):
            s, ex = plant.step(s, cmd(0.0, level, 0.0, device), dt=0.05, substeps=5)
            print(f"    throttle {level:.2f} t={(k + 1) * 0.05 * 5:4.1f}s "
                  f"v={float(s[0, 3]):6.2f} omega={float(s[0, 8]):7.2f} "
                  f"slip={float(ex['sx'][0, 2]):6.3f} Fx={float(ex['fx_wheel'][0, 2]):7.0f}")


def probe_benchmark(preset, device, sizes=(2000, 20000), substeps=5):
    print(f"  throughput ({substeps} physics substeps per control step):")
    for n in sizes:
        plant = new_plant(preset, device, n=n)
        s = plant.zero_state(n)
        s[:, 3] = 10.0
        plant.init_wheels(s)
        command = torch.zeros((n, 3), device=device)
        command[:, 1] = 0.5
        for _ in range(10):
            s, _ = plant.step(s, command, dt=0.01 * substeps, substeps=substeps)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = 50
        for _ in range(reps):
            s, _ = plant.step(s, command, dt=0.01 * substeps, substeps=substeps)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / reps
        print(f"    n={n:6d}: {elapsed * 1e3:6.2f} ms / control step "
              f"({n / elapsed / 1e6:5.2f} M env-steps/s, {elapsed / n * 1e9:6.1f} ns/env-step)")


def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else "fullsize_racecar_ev"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"== FastPlant sanity check: preset={preset} device={device} ==")
    print("[1] straight line acceleration")
    probe_acceleration(preset, device)
    print("[2] braking")
    probe_braking(preset, device)
    print("[3] steady state cornering")
    probe_cornering(preset, device)
    print("[4] lateral acceleration limit")
    probe_lateral_limit(preset, device)
    print("[5] limited slip differential")
    probe_lsd(preset, device)
    print("[6] aerodynamics")
    probe_aero(preset, device)
    print("[7] wheel spin transient")
    probe_wheel_spin(preset, device)
    print("[8] throughput")
    probe_benchmark(preset, device)


if __name__ == "__main__":
    main()
