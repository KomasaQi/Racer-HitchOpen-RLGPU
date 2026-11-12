"""Calibrate the CarSim flat scenario: steering map, Ackermann, wheel radius.

1. Steering: hold the steering wheel at a set of angles with the car at rest and
   read the left/right front wheel angles at the ground (Steer_L1, Steer_R1).
2. Ackermann: compare the measured inner/outer split with the geometric ideal.
3. Wheel radius: roll straight at a small throttle and divide the ground speed by
   the front wheel spin (the front axle is undriven, so it does not slip).
"""
import ctypes
import math
import sys

sys.path.insert(0, r"D:\CarSim2024_1_Full_Database\Extensions\Custom_Py")
import vs_solver  # noqa: E402

SIMFILE = r"D:\carsim_wm_py\flat_mu100\simfile.sim"
IDX = {"AVz": 2, "Steer_L1": 3, "Steer_R1": 4, "AVy_L1": 5, "AVy_R1": 7,
       "Xo": 9, "Yo": 10, "Yaw": 11, "Vx": 12, "Vy": 13, "Beta": 14}

WHEELBASE = 2.650      # from the sprung-mass par (LX_AXLE 2650 mm)
TRACK = 1.720          # front track from the steering par

vs = vs_solver.vs_solver()
dll = ctypes.cdll.LoadLibrary(vs.get_dll_path(SIMFILE))
assert vs.get_api(dll)
if getattr(dll, "vs_set_opt_error_dialog", None) is not None:
    dll.vs_set_opt_error_dialog(0)


def episode(steer_deg, throttle, seconds, sample_every=None):
    cfg = vs.read_configuration(SIMFILE)
    t = cfg["t_start"]
    exp = vs.copy_export_vars(cfg["n_export"])
    imp = [float(steer_deg), float(throttle), 0.0]
    n = int(round(seconds / 0.0005))
    samples = []
    for k in range(n):
        t += 0.0005
        st, exp = vs.integrate_io(t, imp, exp)
        if st != 0:
            break
        if sample_every and k % sample_every == 0:
            samples.append({name: exp[i] for name, i in IDX.items()})
    vs.terminate_run(t)
    return ({name: exp[i] for name, i in IDX.items()}, samples)


def ackermann_pct(left_deg, right_deg):
    """Measured vs ideal Ackermann for a right turn (right wheel is the inner one)."""
    inner = min(abs(left_deg), abs(right_deg))
    outer = max(abs(left_deg), abs(right_deg))
    if outer < 1e-3 or inner >= outer:
        return float("nan")
    # ideal: cot(outer) - cot(inner) = track / wheelbase
    cot_out = 1.0 / math.tan(math.radians(outer))
    cot_in = 1.0 / math.tan(math.radians(inner))
    ideal_delta = TRACK / WHEELBASE
    measured_delta = cot_in - cot_out          # >0 when the inner steers more
    return 100.0 * measured_delta / ideal_delta


print("=== 1. steering map (car at rest, wheels read after 0.4 s) ===")
print("%10s %10s %10s %10s %10s" % ("SW[deg]", "L1[deg]", "R1[deg]", "mean", "ratio"))
rows = []
for sw in (-600, -500, -400, -300, -200, -100, -50, 0, 50, 100, 200, 300, 400, 500, 600):
    st, _ = episode(sw, 0.0, 0.4)
    l1, r1 = st["Steer_L1"], st["Steer_R1"]
    mean = 0.5 * (l1 + r1)
    ratio = abs(sw) / abs(mean) if abs(mean) > 1e-3 else float("nan")
    rows.append((sw, l1, r1, mean))
    print("%10.0f %10.3f %10.3f %10.3f %10.2f" % (sw, l1, r1, mean, ratio))

print("\n=== 2. Ackermann at a right turn (negative SW steers right) ===")
for sw, l1, r1, mean in rows:
    if sw >= 0:
        continue
    pct = ackermann_pct(l1, r1)
    print("  SW %6.0f  L %7.3f  R %7.3f   Ackermann %6.1f %%"
          % (sw, l1, r1, pct))

print("\n=== 3. wheel radius from rolling straight ===")
print("%8s %10s %10s %10s %10s" % ("thr", "Vx[m/s]", "wL1[rad/s]", "R_f[m]", "R_r[m]"))
for thr in (0.10, 0.15, 0.20, 0.30):
    st, _ = episode(0.0, thr, 3.0)
    v = st["Vx"] / 3.6
    w_l = st["AVy_L1"] * 2 * math.pi / 60.0
    w_r = st["AVy_R1"] * 2 * math.pi / 60.0
    print("%8.2f %10.3f %10.3f %10.4f %10.4f"
          % (thr, v, w_l, v / max(w_l, 1e-6), v / max(w_r, 1e-6)))
