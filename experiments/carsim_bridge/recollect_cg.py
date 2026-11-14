"""Recollect CarSim CG coordinates; retain identified FastPlant comparison rows."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.carsim_bridge.system_id_dataset import carsim_trace, DURATIONS, HEADER

p = argparse.ArgumentParser()
p.add_argument("source")
p.add_argument("--out", required=True)
args = p.parse_args()
source, out = Path(args.source), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
manifest = {"position_reference": "instantaneous_total_vehicle_CG", "dt": .05,
            "fast_source": str(source.resolve()), "source_hashes": {}}
for path in source.glob("*__*.csv"):
    scenario, maneuver = path.stem.split("__")
    with path.open(newline="") as f:
        fast = [row for row in csv.DictReader(f) if row["plant"] == "fast"]
    mu = float(scenario.split("mu")[1]) / 100.
    print(scenario, maneuver, flush=True)
    carsim = carsim_trace(scenario, mu, maneuver, DURATIONS[maneuver], .05)
    with (out / path.name).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows([[row[key] for key in HEADER] for row in fast])
        writer.writerows(carsim)
    manifest["source_hashes"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
(out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
