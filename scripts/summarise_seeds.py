"""Summarise the per-seed JSONs written by ``scripts/eval_seeds.sh`` as a Markdown table.

Usage: ``uv run python scripts/summarise_seeds.py <out_prefix>`` (the same prefix as eval_seeds.sh).
"""

from __future__ import annotations

import glob
import json
import re
import sys

import numpy as np


def _fmt(x: float, pct: bool = True) -> str:
    return f"{100 * x:.1f}" if pct else f"{x:.3f}"


def main():
    prefix = sys.argv[1]
    seeds = sorted(
        {
            int(m.group(1))
            for f in glob.glob(f"{prefix}_seed*_chain5.json")
            if (m := re.search(r"_seed(\d+)_chain5\.json$", f))
        }
    )
    rows, agg = [], {k: [] for k in ("chain", "circle", "flip", "hi", "r")}
    for s in seeds:
        chain = json.load(open(f"{prefix}_seed{s}_chain5.json"))
        circle = json.load(open(f"{prefix}_seed{s}_circle5.json"))
        labels = json.load(open(f"{prefix}_seed{s}_labels.json"))
        flips = labels["chatter"]["step_sign_flip_rate"]
        hi = labels["chatter"]["power_share_above_5hz"]
        r = labels["cmd_to_realised"]["r_best_lag"]
        lag = labels["cmd_to_realised"]["best_lag_steps"]
        sat = max(
            max(chain["saturation_ge_0.99"].values()), max(circle["saturation_ge_0.99"].values())
        )
        ci_c = chain["success_wilson95"]
        ci_o = circle["success_wilson95"]
        rows.append(
            f"| {s} | {_fmt(chain['success_rate'])} [{_fmt(ci_c[0])}, {_fmt(ci_c[1])}] "
            f"| {_fmt(circle['success_rate'])} [{_fmt(ci_o[0])}, {_fmt(ci_o[1])}] "
            f"| {_fmt(sat)} "
            f"| {chain['speed_tracking']['corr_episode_speed_target']:.3f} "
            f"| {', '.join(_fmt(flips[a]) for a in ('vx', 'vy', 'vz'))} "
            f"| {', '.join(_fmt(hi[a]) for a in ('vx', 'vy', 'vz'))} "
            f"| {', '.join(f'{r[a]:.3f} ({lag[a]})' for a in ('vx', 'vy', 'vz'))} |"
        )
        agg["chain"].append(chain["success_rate"])
        agg["circle"].append(circle["success_rate"])
        agg["flip"].append(max(flips.values()))
        agg["hi"].append(max(hi.values()))
        agg["r"].append(min(r.values()))
    print(
        "| Seed | 5-gate chains, success % [Wilson 95 %] | 5-gate circles, success % "
        "| max `|a|>=0.99` % | speed r | sign flips % (vx, vy, vz) | power > 5 Hz % (vx, vy, vz) "
        "| eq. (3) r (lag) (vx, vy, vz) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    if seeds:
        print(
            f"\nMean over {len(seeds)} seeds: chains {_fmt(np.mean(agg['chain']))} % "
            f"(min {_fmt(min(agg['chain']))}), circles {_fmt(np.mean(agg['circle']))} % "
            f"(min {_fmt(min(agg['circle']))}); worst sign-flip rate {_fmt(max(agg['flip']))} %, "
            f"worst power above 5 Hz {_fmt(max(agg['hi']))} %, lowest eq. (3) r "
            f"{min(agg['r']):.3f}."
        )


if __name__ == "__main__":
    main()
