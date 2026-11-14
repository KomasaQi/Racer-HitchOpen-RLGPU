"""Measure coupled wheel-solve residuals and time-step convergence."""
import argparse
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xcar-simulation"))
from fast_plant import FastRacerPlant, load_preset


def run(h, n=128, iterations=12, dr=False):
    torch.manual_seed(112)
    plant = FastRacerPlant(load_preset("fullsize_racecar_ev"), n, "cpu")
    plant.audit_wheel_solver = True
    plant.wheel_solver_bisection_iters = iterations
    if dr:
        from dr_sensitivity_audit import ranges_from_runner
        plant.randomize(ranges=ranges_from_runner())
    else:
        plant.tyre_scale[:, 0] = torch.linspace(.5, 1., n)
    state = torch.zeros(n, 10)
    state[:, 3] = 5. + 15 * torch.rand(n)
    state[:, 4] = 10 * (torch.rand(n) - .5)
    state[:, 5] = 2 * (torch.rand(n) - .5)
    plant.init_wheels(state)
    state[:, 8:10] *= 1. + torch.rand(n, 2)
    command = torch.stack((.8 * (torch.rand(n) - .5), torch.rand(n),
                           torch.zeros(n)), dim=1)
    # Include braking and large left/right speed differences.
    command[n//2:, 2] = command[n//2:, 1]
    command[n//2:, 1] = 0.
    residuals = []
    for _ in range(round(.2 / h)):
        state, extras = plant.step(state, command, dt=h)
        residuals.append(plant.last_wheel_residual.abs() / h)
    residual = torch.stack(residuals).flatten()
    return state, {"max_residual_Nm": float(residual.max()),
                   "p95_residual_Nm": float(residual.quantile(.95)),
                   "finite": bool(torch.isfinite(state).all())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--dr", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    runs = [run(h, n=512 if args.dr else 128, iterations=args.iterations, dr=args.dr)
            for h in (.01, .005, .0025)]
    result = {str(h): metrics for h, (_, metrics) in zip((.01, .005, .0025), runs)}
    truth = runs[-1][0]
    for h, (state, _) in zip((.01, .005), runs[:-1]):
        result[str(h)]["state_rmse_vs_2p5ms"] = (state-truth).square().mean(0).sqrt().tolist()
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
