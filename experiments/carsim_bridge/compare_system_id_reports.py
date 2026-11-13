"""Compare two system-ID aggregate reports without hiding regressions in one mean."""
import argparse
import os

import pandas as pd


def markdown_table(frame):
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for row in frame.itertuples(index=False, name=None):
        cells = [("%.4f" % x) if isinstance(x, float) else str(x) for x in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = pd.read_csv(os.path.join(args.baseline, "families.csv"))
    cand = pd.read_csv(os.path.join(args.candidate, "families.csv"))
    keys = ["split", "scenario", "family"]
    merged = base.merge(cand, on=keys, suffixes=("_baseline", "_candidate"))
    merged["delta"] = merged.nrmse_candidate - merged.nrmse_baseline
    merged["relative_change"] = merged.delta / merged.nrmse_baseline.clip(lower=1e-9)
    os.makedirs(args.out, exist_ok=True)
    merged.to_csv(os.path.join(args.out, "comparison.csv"), index=False)
    rows = []
    for split, group in merged.groupby("split"):
        rows.append({
            "split": split,
            "baseline_mean": group.nrmse_baseline.mean(),
            "candidate_mean": group.nrmse_candidate.mean(),
            "relative_change": group.nrmse_candidate.mean() / group.nrmse_baseline.mean() - 1.0,
            "improved": int((group.delta < -1e-9).sum()),
            "regressed": int((group.delta > 1e-9).sum()),
            "total": len(group),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    with open(os.path.join(args.out, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("# System-ID A/B comparison\n\n")
        f.write(markdown_table(summary) + "\n\n")
        f.write("## Largest improvements\n\n")
        f.write(markdown_table(merged.nsmallest(12, "delta")[
            keys + ["nrmse_baseline", "nrmse_candidate", "delta"]]) + "\n\n")
        f.write("## Largest regressions\n\n")
        f.write(markdown_table(merged.nlargest(12, "delta")[
            keys + ["nrmse_baseline", "nrmse_candidate", "delta"]]) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
