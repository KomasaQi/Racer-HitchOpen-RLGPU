"""Build non-destructive CarSim copies with internal EV channels exported."""
from __future__ import annotations

import os
import re
from pathlib import Path


DB = Path(r"D:\CarSim2024_1_Full_Database")
OUT = Path(r"D:\carsim_wm_py\motor_probe")
SCENARIOS = {
    "flat_mu050": "Run_837b2d89-8932-4dae-ac67-7268b98a9959",
    "flat_mu075": "Run_12100cc0-8e59-4364-9b58-d89bf89544dc",
    "flat_mu100": "Run_a1ec4d1b-e3ee-4c01-b272-d1345618d521",
}
BASE_EXPORTS = 36
MOTOR_EXPORTS = (
    "M_MotCmd", "M_Motor", "PwrDrHEV", "PwrMotDm", "PwrMotor",
    "EffMotor", "PwrBttry", "PwrBtrDs", "SocBttry", "VocBttry",
    "V_Bttry", "A_Bttry", "ResBttry",
)

SIM = """SIMFILE

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
PRODUCT_ID CarSim
PRODUCT_VER 2024.1
VEHICLE_CODE i_i
EXT_MODEL_STEP 0.00050000
PORTS_IMP 1,3
PORTS_EXP 1,{n_export}

DLLFILE D:\\Program Files (x86)\\CarSim2024.1_Prog\\Programs\\solvers\\carsim_64.dll
END
"""


def build(name: str, uuid: str) -> Path:
    src = DB / "Results" / uuid / "Run_all.par"
    out = OUT / name
    result = out / "Results" / ("motor_" + name)
    result.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="latin-1")
    if any(re.search(r"(?mi)^EXPORT\s+" + re.escape(ch) + r"\s*$", text)
           for ch in MOTOR_EXPORTS):
        raise RuntimeError("source unexpectedly already contains motor-probe exports")
    marker = re.search(r"(?mi)^EXPORT\s+AV_Motor\s*$", text)
    if not marker:
        raise RuntimeError("AV_Motor export marker missing in " + str(src))
    extra = "\n" + "\n".join("EXPORT " + ch for ch in MOTOR_EXPORTS)
    text = text[:marker.end()] + extra + text[marker.end():]
    text = re.sub(r"(?m)^TSTOP\s+\S+", "TSTOP 100", text)
    (out / "Run_all.par").write_text(text, encoding="latin-1")
    sim = out / "simfile.sim"
    sim.write_text(SIM.format(root="motor_" + name, outdir=out, db=DB,
                              n_export=BASE_EXPORTS + len(MOTOR_EXPORTS)),
                   encoding="ascii")
    print(sim)
    return sim


if __name__ == "__main__":
    for scenario, run in SCENARIOS.items():
        build(scenario, run)
