"""Verify every production DR range both covers nominal and changes dynamics."""
import argparse
import ast
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xcar-simulation"))
from fast_plant import FastRacerPlant, load_preset, PARAM_INDEX


def ranges_from_runner():
    tree = ast.parse((ROOT / "run.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "ev_drift":
                    return ast.literal_eval(value)
    raise RuntimeError("production ev_drift range configuration not found")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    ranges = ranges_from_runner()
    nominal = load_preset("fullsize_racecar_ev")
    results = {}
    torch.manual_seed(6)
    base = torch.zeros(32, 10)
    base[:, 3] = torch.linspace(4., 30., 32)
    base[:, 4] = 5. * (torch.rand(32)-.5)
    base[:, 5] = torch.linspace(-1.2, 1.2, 32)
    u = torch.stack((torch.linspace(-.35,.35,32), torch.ones(32), torch.zeros(32)), 1)
    for key, bounds in ranges.items():
        assert key in PARAM_INDEX, key
        plant = FastRacerPlant(nominal, 64, "cpu")
        plant.tyre_scale[:] = .65
        state = base.repeat(2, 1)
        plant.init_wheels(state)
        state[:, 8:10] *= 1.8
        # A torque cap is only identifiable after reaching the cap. Preserve
        # ordinary cases and add a few extreme differential-speed samples.
        state[::4, 8] += 500.
        plant.p[:32, PARAM_INDEX[key]] = bounds[0]
        plant.p[32:, PARAM_INDEX[key]] = bounds[1]
        for _ in range(5):
            state, ex = plant.step(state, u.repeat(2, 1), dt=.01)
        diff = (state[:32] - state[32:]).abs().amax(0)
        centers = [nominal[key]]
        if key == "D":
            centers = [nominal[key] * scale for scale in (.5, .75, 1.)]
        covered = all(bounds[0] <= center <= bounds[1] for center in centers)
        active = float(diff.max()) > 1e-6
        results[key] = {"range": bounds, "nominal": centers, "covered": covered,
                        "active": active, "max_state_delta": float(diff.max())}
    payload = {"parameters": results,
               "all_pass": all(x["covered"] and x["active"] for x in results.values())}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
