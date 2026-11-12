"""Probe the CarSim VS API without running a simulation.

Loads carsim_64.dll, suppresses error dialogs, and asks it to read a simfile so we
get the import/export counts and the time step.  Nothing is integrated.
"""
import ctypes
import os
import sys

sys.path.insert(0, r"D:\CarSim2024_1_Full_Database\Extensions\Custom_Py")
import vs_solver  # noqa: E402

SIMFILE = sys.argv[1] if len(sys.argv) > 1 else r"D:\carsim_wm_py\flat_mu050\simfile.sim"

vs = vs_solver.vs_solver()
dll_path = vs.get_dll_path(SIMFILE)
print("simfile  ", SIMFILE)
print("dll      ", dll_path, "exists", os.path.exists(dll_path) if dll_path else None)
if not dll_path or not os.path.exists(dll_path):
    raise SystemExit("solver dll not found")

dll = ctypes.cdll.LoadLibrary(dll_path)
print("loaded   ", bool(dll))
print("api      ", vs.get_api(dll))

# no modal error box: a headless process must not block on a dialog
if getattr(dll, "vs_set_opt_error_dialog", None) is not None:
    dll.vs_set_opt_error_dialog(0)
    print("dialogs  suppressed")

cfg = vs.read_configuration(SIMFILE)
print("config   ", cfg)
