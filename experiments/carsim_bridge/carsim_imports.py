"""Do the three import channels actually control the car?"""
import ctypes
import sys

sys.path.insert(0, r"D:\CarSim2024_1_Full_Database\Extensions\Custom_Py")
import vs_solver  # noqa: E402

SIMFILE = sys.argv[1] if len(sys.argv) > 1 else r"D:\carsim_wm_py\flat_mu100\simfile.sim"
IDX = {"Steer_L1": 3, "AVy_L1": 5, "Xo": 9, "Yo": 10, "Yaw": 11,
       "Vx": 12, "Vy": 13, "Beta": 14, "AVz": 2}

vs = vs_solver.vs_solver()
dll = ctypes.cdll.LoadLibrary(vs.get_dll_path(SIMFILE))
assert vs.get_api(dll)
if getattr(dll, "vs_set_opt_error_dialog", None) is not None:
    dll.vs_set_opt_error_dialog(0)


def trial(steer_deg, throttle, seconds=2.0):
    cfg = vs.read_configuration(SIMFILE)
    t = cfg["t_start"]
    exp = vs.copy_export_vars(cfg["n_export"])
    imp = [float(steer_deg), float(throttle), 0.0]
    n = int(round(seconds / 0.0005))
    for _ in range(n):
        t += 0.0005
        st, exp = vs.integrate_io(t, imp, exp)
        if st != 0:
            return {"error": st}
    vs.terminate_run(t)
    return {k: exp[i] for k, i in IDX.items()}


print("%-28s %10s %10s %10s %10s" % ("case", "Vx[km/h]", "SteerL1", "Yaw[deg]", "AVy_L1"))
for steer, thr in ((0.0, 0.0), (0.0, 0.5), (30.0, 0.0), (30.0, 0.4), (-30.0, 0.4)):
    r = trial(steer, thr)
    if "error" in r:
        print("steer %6.1f thr %.2f  ERROR %s" % (steer, thr, r["error"]))
        continue
    print("steer %6.1f thr %.2f  %12.3f %10.3f %10.3f %10.1f"
          % (steer, thr, r["Vx"], r["Steer_L1"], r["Yaw"], r["AVy_L1"]))
