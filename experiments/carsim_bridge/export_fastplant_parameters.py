"""Export the actual loaded production preset and key CarSim total-body facts."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "xcar-simulation"))
from fast_plant import FastRacerPlant, TENSOR_KEYS, load_preset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    p = load_preset("fullsize_racecar_ev")
    plant = FastRacerPlant(p, 1, "cpu")
    runtime = {k: float(plant.col(k)[0]) for k in TENSOR_KEYS}
    # Prove YAML -> resolved preset -> runtime tensor, rather than merely listing
    # values that may be ignored by the dynamics.
    for k, v in runtime.items():
        if isinstance(p.get(k), (int, float)):
            assert abs(v - float(p[k])) <= 2e-4 * max(1.0, abs(float(p[k]))), k
    carsim = {
        "source": r"D:\CarSim2024_1_Full_Database\Results\Run_a1ec4d1b-e3ee-4c01-b272-d1345618d521\LastRun_echo.par",
        "m_total_kg": 1461.4, "Iz_total_kgm2": 1693.469182,
        "lF_total_cg_m": 1.46257356, "lR_total_cg_m": 1.18742644,
        "h_total_cg_m": 0.373483988, "track_front_m": 1.720,
        "track_rear_m": 1.600, "wheel_tire_spin_front_kgm2": 1.77,
        "wheel_tire_spin_rear_kgm2": 2.27,
        "effective_radius_front_m": 0.332, "effective_radius_rear_m": 0.347,
        "static_front_axle_N": 6421.708973, "static_rear_axle_N": 7909.729337,
    }
    payload = {"preset": p, "runtime_tensor": runtime, "carsim_totals": carsim}
    with open(os.path.join(args.out, "current_fastplant_parameters.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    rows = [
        ("总质量 m", "kg", p["m"], carsim["m_total_kg"], "M_TL"),
        ("总偏航惯量 Iz", "kg m^2", p["Iz"], carsim["Iz_total_kgm2"], "IZZ_TL"),
        ("质心到前轴 lF", "m", p["lF"], carsim["lF_total_cg_m"], "LX_CG_TL"),
        ("质心到后轴 lR", "m", p["lR"], carsim["lR_total_cg_m"], "LX_AXLE-LX_CG_TL"),
        ("总质心高 h", "m", p["h"], carsim["h_total_cg_m"], "H_CG_TL"),
        ("前/后轮距", "m", "%g / %g" % (p["Tf"], p["Tr"]), "1.720 / 1.600", "L_TRACK"),
        ("前/后轮+胎自转惯量", "kg m^2", "%g / %g" % (p["IwF"], p["IwR"]), "1.77 / 2.27", "ISPIN"),
        ("前/后有效滚动半径", "m", "%g / %g" % (p["rF"], p["rR"]), "0.332 / 0.347", "RRE(1) / RRE(2)"),
        ("轮胎纵向 B/C/D", "-", "%g / %g / %g" % (p["bx"], p["cx"], p["D"]), "三路面纯滑表与动态复放标定", "Fx table compact fit"),
        ("轮胎横向 B/C/D", "-", "%g / %g / %g" % (p["by"], p["cy"], p["D"] * p["mu_y_ratio"]), "三路面纯滑表与动态复放标定", "Fy table compact fit"),
        ("附着-小滑移刚度指数 px/py", "-", "%g / %g" % (p["tire_stiffness_grip_exponent_x"], p["tire_stiffness_grip_exponent_y"]), "0.20 / 0.40（CarSim 短窗 A/B）", "analytic grip-stiffness coupling"),
        ("终传比", "-", p["gear_ratio"], 3.44, "R_GEAR_RD"),
        ("LSD T60 / exponent", "N m / -", "%g / %g" % (p["lsd_torque_ref"], p["lsd_exponent"]), "90.09 / 0.5", "M_DIFF_VISC_TABLE"),
        ("电机/差速器/半轴惯量", "kg m^2", "%g / %g / %g" % (p["motor_inertia"], p["diff_input_inertia"], p["halfshaft_inertia"]), "0.0226 / 0.013 / 0.009", "IMOTOR/IDS/IHS"),
        ("油门机械功率上限", "kW", p["max_power_w"] / 1000.0, 384.0, "PWR_DRV_THROTTLE_COEFFICIENT"),
        ("电池 Voc/Vmin/R", "V / V / ohm", "%g / %g / %g" % (p["battery_voc_v"], p["battery_min_voltage_v"], p["battery_resistance_ohm"]), "936.936 / 720 / 0.12", "VOC_BATTERY/V_BATTERY_MIN/R_BATTERY_DISCHARGE"),
        ("有效电机包络", "-", "25-rpm merged 1D", "motor + efficiency + battery limit", "MMOTOR_MAX/2D_LINEAR"),
        ("空气密度", "kg/m^3", p["rho_air"], 1.206, "D_AIR"),
    ]
    with open(os.path.join(args.out, "CURRENT_FASTPLANT_PARAMETERS.md"), "w", encoding="utf-8") as f:
        f.write("# 当前 FastPlant 标称参数（实际加载值）\n\n")
        f.write("生成时已逐项断言 resolved preset 与 `plant.p` 运行时张量一致。\n\n")
        f.write("| 参数 | 单位 | FastPlant | CarSim/证据 | 来源字段 |\n|---|---:|---:|---:|---|\n")
        for row in rows:
            f.write("| %s | %s | %s | %s | %s |\n" % row)
        f.write("\n完整参数（含轮胎、电机、制动、空气动力和所有张量列）见 `current_fastplant_parameters.json`。\n")


if __name__ == "__main__":
    main()
