"""Dump every export channel so the indices can be matched to names."""
import ctypes
import sys

sys.path.insert(0, r"D:\CarSim2024_1_Full_Database\Extensions\Custom_Py")
import vs_solver  # noqa: E402

SIMFILE = sys.argv[1] if len(sys.argv) > 1 else r"D:\carsim_wm_py\flat_mu050\simfile.sim"
CH = ["Ax", "Ay", "AVz", "Steer_L1", "Steer_R1", "AVy_L1", "AVy_L2", "AVy_R1",
      "AVy_R2", "Xo", "Yo", "Yaw", "Vx", "Vy", "Beta", "Fx_WC_L1", "Fx_WC_L2",
      "Fx_WC_R1", "Fx_WC_R2", "Fy_WC_L1", "Fy_WC_L2", "Fy_WC_R1", "Fy_WC_R2",
      "Fz_WC_L1", "Fz_WC_L2", "Fz_WC_R1", "Fz_WC_R2", "Kappa_L1", "Kappa_L2",
      "Kappa_R1", "Kappa_R2", "Alpha_L1", "Alpha_L2", "Alpha_R1", "Alpha_R2",
      "AV_Motor"]

vs = vs_solver.vs_solver()
dll = ctypes.cdll.LoadLibrary(vs.get_dll_path(SIMFILE))
assert vs.get_api(dll)
if getattr(dll, "vs_set_opt_error_dialog", None) is not None:
    dll.vs_set_opt_error_dialog(0)

cfg = vs.read_configuration(SIMFILE)
n_imp, n_exp = cfg["n_import"], cfg["n_export"]
print("n_import", n_imp, "n_export", n_exp, "t_step", cfg["t_step"])
t = cfg["t_start"]
exp = vs.copy_export_vars(n_exp)
imp = [0.0] * n_imp

snaps = {}
for k in range(4):
    for _ in range(100):          # 100 x 0.01 s = 1 s
        t += 0.01
        st, exp = vs.integrate_io(t, imp, exp)
        if st != 0:
            print("status", st, "t", t)
            break
    snaps[round(t, 2)] = list(exp)
vs.terminate_run(t)

keys = sorted(snaps)
print("\n%-12s" % "channel" + "".join("%14s" % ("t=%.2f" % k) for k in keys))
for i in range(n_exp):
    name = CH[i] if i < len(CH) else "ch%d" % i
    print("%-12s" % name + "".join("%14.4f" % snaps[k][i] for k in keys))
