"""Extract the expanded CarSim front/rear pure-slip carpets to JSON."""
import argparse
import json
import re


def numeric_row(line):
    try:
        return [float(x.strip()) for x in line.split(",")]
    except ValueError:
        return None


def extract(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    found = {}
    axle = None
    for i, line in enumerate(lines):
        if "#FullDataName Tire: Longitudinal Force" in line or \
                "#FullDataName Tire: Lateral Force" in line:
            axle = "front" if "Front" in line else "rear" if "Rear" in line else None
        token = line.strip().split(" ", 1)[0]
        if token not in ("FX_TIRE_CARPET", "FY_TIRE_CARPET") or axle is None:
            continue
        kind = "fx" if token.startswith("FX") else "fy"
        rows = []
        j = i + 1
        while j < len(lines) and lines[j].strip() != "ENDTABLE":
            row = numeric_row(lines[j].strip())
            if row is not None:
                rows.append(row)
            j += 1
        if not rows:
            raise ValueError("empty %s/%s carpet" % (axle, kind))
        key = "%s_%s" % (axle, kind)
        table = {"slip": [r[0] for r in rows[1:]],
                 "fz": rows[0][1:],
                 "force": [r[1:] for r in rows[1:]],
                 "slip_unit": "ratio" if kind == "fx" else "degree",
                 "force_unit": "N", "fz_unit": "N"}
        if key in found and found[key] != table:
            raise ValueError("non-identical duplicate carpet %s" % key)
        found[key] = table
    expected = {"front_fx", "front_fy", "rear_fx", "rear_fy"}
    if set(found) != expected:
        raise ValueError("found %s, expected %s" % (sorted(found), sorted(expected)))
    return {"source": path, "format": "CarSim expanded pure-slip carpets", "tables": found}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = extract(args.run_all)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    for name, table in sorted(data["tables"].items()):
        peak = max(max(row) for row in table["force"])
        print("%-10s slip=%d loads=%d peak_force=%.1f" %
              (name, len(table["slip"]), len(table["fz"]), peak))


if __name__ == "__main__":
    main()
