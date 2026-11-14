"""Validate the one-state motor lag against recorded CarSim internal channels."""
import argparse
import json
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    data = pd.read_csv(args.trace)
    results = []
    for q, group in data.groupby("throttle"):
        g = group.sort_values("time")
        target, actual = g.M_MotCmd.to_numpy(), g.M_Motor.to_numpy()
        dt = np.diff(np.r_[0., g.time])
        predicted = []
        state = 0.
        for command, h in zip(target, dt):
            state += -np.expm1(-h / .05) * (command - state)
            predicted.append(state)
        error = np.asarray(predicted) - actual
        results.append({"throttle": float(q),
                        "static_rmse_Nm": float(np.sqrt(np.mean((target-actual)**2))),
                        "lag_50ms_rmse_Nm": float(np.sqrt(np.mean(error**2))),
                        "first_0p5s_lag_rmse_Nm": float(np.sqrt(np.mean(error[g.time.to_numpy() <= .5]**2)))})
    with open(args.output, "w") as f:
        json.dump({"source": args.trace, "tau_s": .05, "results": results}, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
