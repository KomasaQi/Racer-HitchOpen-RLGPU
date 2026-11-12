"""Drive one CarSim scenario headlessly and check that it starts from rest.

Runs a short episode at 20 Hz control (5 internal 0.01 s advances per action), then
terminates and re-initializes to see whether the solver can be reset in-process.
"""
import ctypes
import sys
import time

sys.path.insert(0, r"D:\CarSim2024_1_Full_Database\Extensions\Custom_Py")
import vs_solver  # noqa: E402

SIMFILE = r"D:\carsim_wm_py\flat_mu050\simfile.sim"

# export order of the regenerated 36-channel set
CH = ["Ax", "Ay", "AVz", "Steer_L1", "Steer_R1", "AVy_L1", "AVy_L2", "AVy_R1",
      "AVy_R2", "Xo", "Yo", "Yaw", "Vx", "Vy", "Beta", "Fx_WC_L1", "Fx_WC_L2",
      "Fx_WC_R1", "Fx_WC_R2", "Fy_WC_L1", "Fy_WC_L2", "Fy_WC_R1", "Fy_WC_R2",
      "Fz_WC_L1", "Fz_WC_L2", "Fz_WC_R1", "Fz_WC_R2", "Kappa_L1", "Kappa_L2",
      "Kappa_R1", "Kappa_R2", "Alpha_L1", "Alpha_L2", "Alpha_R1", "Alpha_R2",
      "AV_Motor"]


def open_solver(simfile):
    vs = vs_solver.vs_solver()
    dll = ctypes.cdll.LoadLibrary(vs.get_dll_path(simfile))
    assert vs.get_api(dll), "solver API missing"
    if getattr(dll, "vs_set_opt_error_dialog", None) is not None:
        dll.vs_set_opt_error_dialog(0)
    return vs, dll


def run_episode(vs, simfile, throttle, n_steps=60, label=""):
    cfg = vs.read_configuration(simfile)
    n_imp, n_exp = cfg["n_import"], cfg["n_export"]
    t = cfg["t_start"]
    exp = vs.copy_export_vars(n_exp)
    imp = [0.0] * n_imp
    # NOTE: vs_read_configuration() already initializes the solver.  Calling
    # vs_initialize() as well raises an access violation.
    row = []
    for k in range(n_steps):
        imp[0] = 0.0            # steering wheel, deg
        imp[1] = throttle       # throttle 0..1
        imp[2] = 0.0            # brake master cylinder, MPa
        for _ in range(5):      # 5 x 0.01 s = 20 Hz control
            t += 0.01
            status, exp = vs.integrate_io(t, imp, exp)
            if status != 0:
                return None, "solver status %s at t=%.3f" % (status, t)
        if k % 20 == 0:
            row.append((round(t, 2), round(exp[12], 3), round(exp[9], 3),
                        round(exp[10], 3), round(exp[11], 2), round(exp[5], 1)))
    vs.terminate_run(t)
    return row, None


if __name__ == "__main__":
    vs, dll = open_solver(SIMFILE)
    print("export channels:", len(CH))

    for label, thr in (("zero throttle", 0.0), ("30 % throttle", 0.3)):
        t0 = time.time()
        row, err = run_episode(vs, SIMFILE, thr, 60, label)
        dt = time.time() - t0
        print("\n== %s (60 control steps = 3 s, %.2f s wall)" % (label, dt))
        if err:
            print("   ERROR", err)
            continue
        print("   t[s]   Vx[km/h]   Xo[m]    Yo[m]   Yaw[deg]  AVy_L1[rpm]")
        for r in row:
            print("   %-6s %-10s %-8s %-7s %-9s %s" % r)

    # can the same process start a fresh run?
    row2, err2 = run_episode(vs, SIMFILE, 0.0, 20, "second run")
    print("\n== reset check (second run, zero throttle)")
    print("   ", "ERROR " + err2 if err2 else row2)
