"""Build an optional self-contained CarSim scenario that starts from rest.

The project's normal ``flat_mu*`` scenarios already have ``OPT_SC 0`` and give
the three imports full authority.  This utility is kept only for reproducible
experiments that need a stable local copy of the expanded run.

  1. copies the expanded Run_all.par of a stock scenario,
  2. preserves all parameter blocks (some control blocks also define units/macros),
  3. forces a zero initial speed,
  4. writes a simfile that points at the copied par.

Everything stays inside D:\\carsim_wm_py so the stock database is untouched.
"""
import os
import re
import shutil
import sys

DB = r"D:\CarSim2024_1_Full_Database"
OUT_ROOT = r"D:\carsim_wm_py"

# scenario name -> (source Run directory, friction label)
SCENARIOS = {
    "flat_mu050": "Run_837b2d89-8932-4dae-ac67-7268b98a9959",
    "flat_mu075": "Run_12100cc0-8e59-4364-9b58-d89bf89544dc",
    "flat_mu100": "Run_a1ec4d1b-e3ee-4c01-b272-d1345618d521",
}

# procedure par that carries the driver model / speed controller links
CONTROL_LINK = re.compile(r"^(ENTER_PARSFILE|PARSFILE)\s+Control[\\/]", re.I)
# INSTALL_SPEED_CONTROLLER may be removed for a minimal run, although OPT_SC 0
# already disables it in the source flat scenarios.
CONTROL_BLOCK = re.compile(r"^(INSTALL_DM_PATH_FOLLOWER|INSTALL_DM_IMPORTS|"
                           r"INSTALL_SPEED_CONTROLLER|OPT_DM|LTARG_|CREATE_PATH_OBJ|"
                           r"OPT_STR_BY_TRQ|OPT_SPEED_CONTROLLER)\b")
SPEED_KEYS = re.compile(r"^(SPEED_|VSTART|OPT_INIT_SPEED)\b")


def patch_par(src_par, dst_par):
    with open(src_par, "r", encoding="latin-1") as f:
        lines = f.readlines()
    out, skipped = [], 0
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("*"):
            out.append(line)
            continue
        # Do not remove Control blocks.  OPT_SC 0 / OPT_DM 0 already disable the
        # active controllers, while those parsfiles also define units such as
        # S/M.  Removing the whole block produced a run that parsed the simfile
        # but stopped at T=0 with an undefined-unit error.
        out.append(line)
    text = "".join(out)
    # zero initial speed, short minimum stop time
    text = re.sub(r"(?m)^TSTOP\s+\S+", "TSTOP 100", text)
    text = re.sub(r"(?m)^SSTOP\s+\S+", "SSTOP 100000", text)
    if not re.search(r"(?m)^VSTART\b", text):
        text = text.replace("OPT_INIT_PATH 1", "OPT_INIT_PATH 1\nVSTART 0")
    with open(dst_par, "w", encoding="latin-1") as f:
        f.write(text)
    return skipped


# NOTE: INPUT must use the WORK_DIR macro form, not a bare absolute path --
# vs_solver.read_configuration returns n_export 0 for the bare form, which is why the
# regenerated runs could never be loaded even though PORTS_EXP was present.
SIM_TEMPLATE = """SIMFILE

SET_MACRO $(ROOT_FILE_NAME)$ {root}
SET_MACRO $(OUTPUT_PATH)$ {outdir}\\
SET_MACRO $(WORK_DIR)$ {outdir}\\
SET_MACRO $(OUTPUT_FILE_PREFIX)$ $(WORK_DIR)$Results\\$(ROOT_FILE_NAME)$\\LastRun

FILEBASE $(OUTPUT_FILE_PREFIX)$
INPUT $(WORK_DIR)$Run_all.par
INPUTARCHIVE $(OUTPUT_FILE_PREFIX)$_all.par
ECHO $(OUTPUT_FILE_PREFIX)$_echo.par
FINAL $(OUTPUT_FILE_PREFIX)$_end.par
LOGFILE $(OUTPUT_FILE_PREFIX)$_log.txt
ERDFILE $(OUTPUT_FILE_PREFIX)$.vs
PROGDIR D:\\Program Files (x86)\\CarSim2024.1_Prog\\
DATADIR {db}\\
GUI_REFRESH_V CarSim_RefreshEvent_4948
PRODUCT_ID CarSim
PRODUCT_VER 2024.1
VEHICLE_CODE i_i
EXT_MODEL_STEP 0.00050000
PORTS_IMP 1,3
PORTS_EXP 1,36

DLLFILE D:\\Program Files (x86)\\CarSim2024.1_Prog\\Programs\\solvers\\carsim_64.dll
END
"""


def build(name, run_uuid):
    src = os.path.join(DB, "Results", run_uuid, "Run_all.par")
    outdir = os.path.join(OUT_ROOT, "rl_rest", name)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "Results", "rl_" + name), exist_ok=True)
    par = os.path.join(outdir, "Run_all.par")
    n = patch_par(src, par)
    sim = os.path.join(outdir, "simfile.sim")
    with open(sim, "w", encoding="ascii") as f:
        f.write(SIM_TEMPLATE.format(root="rl_" + name, outdir=outdir, par=par, db=DB))
    print("%-12s removed %2d control lines -> %s" % (name, n, sim))
    return sim


if __name__ == "__main__":
    for name, uuid in SCENARIOS.items():
        build(name, uuid)
