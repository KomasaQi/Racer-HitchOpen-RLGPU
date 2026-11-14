"""CarSim flat-ground environment for the continuous-drift task.

This is the real-world stand-in: the CarSim solver is driven headlessly through the
official VehicleSim API, the car starts from rest on a flat pad, and only the three
import channels (steering wheel angle, throttle, brake pressure) act on it.

Contract with the rest of the repository
----------------------------------------
``s`` uses the canonical 10-state layout, SI units, exactly like the GPU plant:

    0 x         world x, m
    1 y         world y, m  (left positive)
    2 psi       yaw, rad    (counter clockwise from +x)
    3 vx_world  world-frame x velocity, m/s
    4 vy_world  world-frame y velocity, m/s
    5 r         yaw rate, rad/s
    6..9        wheel angular speeds FL, FR, RL, RR, rad/s

``step`` takes ``[steer, throttle, brake]`` in the *pedal* convention of
``ContinuousDriftIWDEnv``: steer is normalised to [-1, 1] of the maximum steering
wheel angle, throttle to [0, 1], brake to [0, 1] of the maximum master-cylinder
pressure.  It advances exactly one control period (default 0.05 s = 20 Hz) by
calling the solver at its own 0.5 ms step.

Notes learned the hard way:
* ``vs_read_configuration`` already initialises the solver -- calling
  ``vs_initialize`` afterwards raises an access violation;
* one ``vs_integrate_io`` call advances ONE 0.5 ms solver step, independent of the
  time argument, so a 0.05 s control step needs 100 calls;
* ``Beta`` is meaningless at standstill (atan2 of two near-zero velocities), so it is
  zeroed below a small speed instead of being trusted.
"""
import ctypes
import os
import sys
import tempfile
import shutil
import json
from pathlib import Path

import numpy as np
import torch

_CARSIM_PY = r"D:\CarSim2024_1_Full_Database\Extensions\Custom_Py"
if _CARSIM_PY not in sys.path:
    sys.path.insert(0, _CARSIM_PY)

# channel order of the 36-channel export set
CHANNELS = [
    "Ax", "Ay", "AVz", "Steer_L1", "Steer_R1",
    "AVy_L1", "AVy_L2", "AVy_R1", "AVy_R2",
    "Xo", "Yo", "Yaw", "Vx", "Vy", "Beta",
    "Fx_WC_L1", "Fx_WC_L2", "Fx_WC_R1", "Fx_WC_R2",
    "Fy_WC_L1", "Fy_WC_L2", "Fy_WC_R1", "Fy_WC_R2",
    "Fz_WC_L1", "Fz_WC_L2", "Fz_WC_R1", "Fz_WC_R2",
    "Kappa_L1", "Kappa_L2", "Kappa_R1", "Kappa_R2",
    "Alpha_L1", "Alpha_L2", "Alpha_R1", "Alpha_R2",
    "AV_Motor",
]
IDX = {name: i for i, name in enumerate(CHANNELS)}

DEFAULT_SCENARIOS = {
    # The original flat_mu scenarios are already open loop: the speed-controller
    # module is present but OPT_SC=0 disables it.  Official-solver probes confirm
    # that throttle, steering and brake imports have authority.
    0.50: r"D:\carsim_wm_py\flat_mu050\simfile.sim",
    0.75: r"D:\carsim_wm_py\flat_mu075\simfile.sim",
    1.00: r"D:\carsim_wm_py\flat_mu100\simfile.sim",
}

# pedal scaling: this vehicle's steering wheel range and brake pressure range
# calibrated (see experiments/carsim_bridge/carsim_calib.py): the steering
# wheel to wheel-angle map measured 13.9:1 near centre and 12.0:1 at full lock,
# so 345 deg of steering wheel is the 25.8 deg (0.45 rad) wheel angle the GPU
# plant trains with -- command +/-1 must mean the same thing on both plants.
MAX_STEER_SW_DEG = 345.0
# CarSim names wheels by side then axle (L1, L2, R1, R2 = FL, RL, FR, RR) while the
# plant contract is FL, FR, RL, RR.  Getting this wrong swaps the front-right and
# rear-left channels, so the "front" speed estimate silently includes a driven rear
# wheel.
CARSIM_WHEEL_SUFFIXES = ("L1", "R1", "L2", "R2")

# the policy commands ROAD-WHEEL radians, bounded by the GPU plant max_steer
MAX_ROAD_WHEEL_RAD = 0.45


def road_wheel_rad_to_steering_wheel_deg(angle):
    """Convert the policy road-wheel angle to the CarSim steering import.

    Measured map: 13.9:1 near centre falling to 12.0:1 at full lock, so
    MAX_STEER_SW_DEG corresponds to the 0.45 rad the GPU plant trains with.
    Every bridge site must call this instead of scaling the angle itself.
    """
    return float(np.clip(float(angle) / MAX_ROAD_WHEEL_RAD, -1.0, 1.0)
                 * MAX_STEER_SW_DEG)
MAX_BRAKE_MPA = 6.0           # agreed with the owner: 6 MPa is the working ceiling


class CarSimFlatEnv:
    """A single-vehicle CarSim environment (one solver instance, n = 1)."""

    def __init__(self, scenario="flat_mu050", simfile=None, device="cpu",
                 dt=0.05, physics_dt=0.01, max_steps=600, seed=None,
                 quiet=False):
        self.device = torch.device(device)
        self.dt = float(dt)
        self.physics_dt = float(physics_dt)
        self.max_steps = int(max_steps)
        self.n = 1
        self.quiet = quiet
        self.scenario = scenario
        if simfile is None:
            key = float(str(scenario).replace("flat_mu", "").replace("0", "0.", 1)) \
                if "mu" in str(scenario) else 1.0
            simfile = DEFAULT_SCENARIOS.get(round(key, 2), DEFAULT_SCENARIOS[1.00])
        self.simfile = simfile
        # Preserve the authoritative input, but never share LastRun output files
        # with the GUI or another evaluator. Solver output is disposable here.
        self._run_directory = tempfile.TemporaryDirectory(prefix="carsim_bridge_", ignore_cleanup_errors=True)
        original_simfile = Path(simfile).read_text(encoding="utf-8")
        output_prefix = str(Path(self._run_directory.name) / "run")
        output_keys = {"FILEBASE": output_prefix,
                       "INPUTARCHIVE": output_prefix + "_all.par",
                       "ECHO": output_prefix + "_echo.par",
                       "FINAL": output_prefix + "_end.par",
                       "LOGFILE": output_prefix + "_log.txt",
                       "ERDFILE": output_prefix + ".vs"}
        lines = []
        for line in original_simfile.splitlines():
            key = line.strip().split(maxsplit=1)[0] if line.strip() else ""
            lines.append(key + " " + output_keys[key] if key in output_keys else line)
        self._runtime_simfile = str(Path(self._run_directory.name) / "simfile.sim")
        Path(self._runtime_simfile).write_text("\n".join(lines) + "\n", encoding="utf-8")
        if seed is not None:
            np.random.seed(seed)

        import vs_solver
        self._vs = vs_solver.vs_solver()
        self._dll = ctypes.cdll.LoadLibrary(self._vs.get_dll_path(self.simfile))
        if not self._vs.get_api(self._dll):
            raise RuntimeError("CarSim solver API incomplete: " + self.simfile)
        if getattr(self._dll, "vs_set_opt_error_dialog", None) is not None:
            self._dll.vs_set_opt_error_dialog(0)

        self._cfg = None
        self._t = 0.0
        self._exp = None
        self._run_open = False
        self._open()

        self.s = torch.zeros((1, 10), device=self.device)
        self.step_count = 0
        self.is_done = torch.zeros(1, dtype=torch.uint8, device=self.device)
        self.solver_dt = float(self._cfg["t_step"])
        self.n_physics = int(round(self.dt / self.solver_dt))
        if self.n_physics < 1 or abs(self.n_physics*self.solver_dt-self.dt) > 1e-9:
            raise ValueError("control period must be an integer multiple of CarSim solver step")
        self._raw = np.zeros(len(CHANNELS))

    # ------------------------------------------------------------------ solver
    def _open(self):
        """Read the run configuration, which also initialises the solver."""
        self._cfg = self._vs.read_configuration(self._runtime_simfile)
        self._run_open = True
        if self._cfg["t_stop"] - self._cfg["t_start"] < self.dt:
            message = self._solver_error()
            self.close()
            raise RuntimeError(f"CarSim run duration is shorter than one control period: {self._cfg}; {message}")
        if self._cfg["n_export"] != len(CHANNELS):
            raise RuntimeError(
                "scenario exports %d channels, the bridge knows %d; regenerate the "
                "simfile or update CHANNELS" % (self._cfg["n_export"], len(CHANNELS)))
        self._exp = self._vs.copy_export_vars(self._cfg["n_export"])
        self._t = self._cfg["t_start"]
        # Xo/Yo locate the vehicle origin; Vx/Vy refer to the instantaneous
        # TOTAL-vehicle CG. Mixing them creates an r*1.4626 m kinematic error.
        # Read the matching CG coordinates through the official read-only API.
        get_ptr = self._dll.vs_get_var_ptr
        get_ptr.argtypes = [ctypes.c_char_p]
        get_ptr.restype = ctypes.POINTER(ctypes.c_double)
        self._cg_ptrs = tuple(get_ptr(key) for key in (b"XCG_TM", b"YCG_TM"))
        if not all(bool(ptr) for ptr in self._cg_ptrs):
            raise RuntimeError("CarSim must expose instantaneous total-CG coordinates")
        self._time_ptr = get_ptr(b"T")
        if not self._time_ptr:
            raise RuntimeError("CarSim must expose its actual integration time")

    def _solver_error(self):
        error = self._dll.vs_get_error_message
        error.argtypes = []
        error.restype = ctypes.c_char_p
        return (error() or b"").decode("utf-8", errors="replace")

    def close(self):
        if self._run_open:
            self._vs.terminate_run(self._t)
            self._run_open = False

    # ------------------------------------------------------------------ episodes
    def reset(self):
        """Put the car back at rest at the scenario origin."""
        self.close()
        self._open()
        self.step_count = 0
        self.is_done.zero_()
        self._raw = np.asarray(self._exp, dtype=np.float64).copy()
        self._update_state(self._raw)
        return self.s

    # ---------------------------------------------------------------- stepping
    def _imports(self, action):
        """Map the normalised pedal action to CarSim import values."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        steer = float(np.clip(a[0], -1.0, 1.0)) * MAX_STEER_SW_DEG
        throttle = float(np.clip(a[1], 0.0, 1.0))
        brake = float(np.clip(a[2], 0.0, 1.0)) * MAX_BRAKE_MPA
        return [steer, throttle, brake]

    def step(self, action):
        imp = self._imports(action)
        status = 0
        for _ in range(self.n_physics):
            self._t += self.solver_dt
            status, self._exp = self._vs.integrate_io(self._t, imp, self._exp)
            if status != 0:
                raise RuntimeError(
                    f"CarSim stopped (status={status}) at requested t={self._t:.6f}, "
                    f"actual t={self._time_ptr[0]:.6f}; {self._solver_error()}; "
                    f"log={self._run_directory.name}")
        actual_t = float(self._time_ptr[0])
        if abs(actual_t-self._t) > self.solver_dt * 1.1:
            raise RuntimeError(f"CarSim time mismatch: requested={self._t}, actual={actual_t}")
        self.step_count += 1
        self._raw = np.asarray(self._exp, dtype=np.float64).copy()
        self._update_state(self._raw)
        if self.step_count >= self.max_steps:
            self.is_done[:] = 2          # full length
        reward = torch.zeros(1, device=self.device)
        return self.s, reward, self.is_done.clone(), {"status": status}

    def save_native_results(self, destination):
        """Preserve the closed solver run for native VehicleSim postprocessing."""
        if self._run_open:
            raise RuntimeError("Close the CarSim run before preserving native results")
        source = Path(self._run_directory.name)
        destination = Path(destination).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.is_file():
                shutil.copy2(item, destination / item.name)
        # The binary companion uses the same basename as the .vs metadata.
        metadata = destination / "run.vs"
        if metadata.exists():
            data = json.loads(metadata.read_text(encoding="utf-8-sig"))
            data["VsChannelGroup"]["Input File"] = str(metadata)
            metadata.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return str(destination)

    def _update_state(self, raw):
        x, y = (float(ptr[0]) for ptr in self._cg_ptrs)
        psi = np.deg2rad(raw[IDX["Yaw"]])
        vx_b = raw[IDX["Vx"]] / 3.6          # km/h -> m/s, body frame
        vy_b = raw[IDX["Vy"]] / 3.6
        r = np.deg2rad(raw[IDX["AVz"]])      # deg/s -> rad/s
        # body -> world (CarSim: x forward, y left, psi counter clockwise)
        cp, sp = np.cos(psi), np.sin(psi)
        vx_w = cp * vx_b - sp * vy_b
        vy_w = sp * vx_b + cp * vy_b
        w = np.array([raw[IDX["AVy_" + k]] for k in CARSIM_WHEEL_SUFFIXES],
                     dtype=np.float64) * (2.0 * np.pi / 60.0)
        self.s[0, 0] = x
        self.s[0, 1] = y
        self.s[0, 2] = psi
        self.s[0, 3] = vx_w
        self.s[0, 4] = vy_w
        self.s[0, 5] = r
        self.s[0, 6:10] = torch.as_tensor(w, device=self.device)

    # ------------------------------------------------------------------- extras
    def obs(self):
        return self.s

    def raw(self):
        """The 36 exported channels as a dict in CarSim units."""
        values = {k: float(self._raw[i]) for k, i in IDX.items()}
        values.update(Xcg_TM=float(self._cg_ptrs[0][0]), Ycg_TM=float(self._cg_ptrs[1][0]))
        return values

    def beta(self):
        """Sideslip in rad, or 0 when the car is too slow for it to mean anything."""
        v = float(np.hypot(self.s[0, 3], self.s[0, 4]))
        if v < 0.5:
            return 0.0
        return float(np.deg2rad(self._raw[IDX["Beta"]]))

    def geometry(self):
        """Wheel positions and radius for compatibility with the GPU plant."""
        import yaml
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "xcar-simulation", "presets.yaml")
        with open(path, encoding="utf-8") as f:
            p = yaml.safe_load(f)["fullsize_racecar_ev"]["parameters"]
        x = torch.tensor([p["lF"], p["lF"], -p["lR"], -p["lR"]])
        y = torch.tensor([p["Tf"]/2, -p["Tf"]/2, p["Tr"]/2, -p["Tr"]/2])
        rad = torch.tensor([p["rF"], p["rF"], p["rR"], p["rR"]])
        return x, y, rad, torch.zeros(4)
