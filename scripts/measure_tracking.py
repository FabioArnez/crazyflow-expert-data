"""Open-loop tracking of the velocity-command layer: hover, per-axis steps, label consistency.

Usage: ``uv run python scripts/measure_tracking.py [--grid 480 120 30] [--out results/x.json]``.
Runs on CPU. The numbers in ``docs/crazyflow040_state_control.md`` come from this script.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

from expert_data.tracking import (  # noqa: E402
    AXES,
    label_consistency,
    run_commands,
    smooth_random_commands,
    step_response,
)
from expert_data.velocity import CommandConfig, RateGrid  # noqa: E402

STEPS = {"vx": 1.0, "vy": 1.0, "vz": 0.5, "wz": 1.0}  # step amplitudes (m/s, rad/s)
SIGNALS = {  # per-axis std (m/s, m/s, m/s, rad/s) and low-pass cutoff (Hz) of random commands
    "gentle": ((0.4, 0.4, 0.2, 0.4), 0.5),
    "moderate": ((0.6, 0.6, 0.3, 0.6), 0.5),
    "aggressive": ((0.6, 0.6, 0.3, 0.6), 1.0),
}


def measure(grid: RateGrid, cmd_cfg: CommandConfig, n_random: int = 16) -> dict:
    f = grid.decision_freq
    t_len, t_step = int(5 * f), int(0.5 * f)
    cmds = np.zeros((t_len, 1 + len(AXES), 4), np.float32)
    for i, ax in enumerate(AXES):
        cmds[t_step:, 1 + i, i] = STEPS[ax]
    out = run_commands(cmds, grid, cmd_cfg, start_z=20.0)
    pos = out["pos"]
    result = {
        "grid": [grid.sim_freq, grid.state_freq, grid.decision_freq],
        "command": cmd_cfg.__dict__,
        "hover_5s": {
            "final_offset_m": float(np.linalg.norm(pos[-1, 0] - pos[0, 0])),
            "max_offset_m": float(np.linalg.norm(pos[:, 0] - pos[0, 0], axis=-1).max()),
        },
        "steps": {
            ax: step_response(out["realised"][:, 1 + i, i], t_step, STEPS[ax], grid.dt)
            for i, ax in enumerate(AXES)
        },
        "consistency": {},
    }
    for name, (scale, cutoff) in SIGNALS.items():
        rand = smooth_random_commands(0, int(10 * f), n_random, grid.dt, np.array(scale), cutoff)
        out = run_commands(rand, grid, cmd_cfg, start_z=20.0)
        speed = np.linalg.norm(out["vel"], axis=-1).max(0)
        skip = int(f)  # drop the first second
        r0, lag, rb = label_consistency(rand[skip:], out["realised"][skip:])
        result["consistency"][name] = {
            "std": list(scale),
            "cutoff_hz": cutoff,
            "diverged_worlds": int((speed > 3.0).sum()),
            "n_worlds": n_random,
            "r_lag0": dict(zip(AXES, r0.round(3).tolist(), strict=True)),
            "best_lag_steps": dict(zip(AXES, lag.tolist(), strict=True)),
            "r_best_lag": dict(zip(AXES, rb.round(3).tolist(), strict=True)),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, nargs=3, default=[480, 120, 30])
    parser.add_argument("--anchor", default="integrated", choices=["integrated", "measured"])
    parser.add_argument("--acc-max-xy", type=float, default=CommandConfig.acc_max_xy)
    parser.add_argument("--acc-max-z", type=float, default=CommandConfig.acc_max_z)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    grid = RateGrid(*args.grid)
    cmd_cfg = CommandConfig(
        anchor=args.anchor, acc_max_xy=args.acc_max_xy, acc_max_z=args.acc_max_z
    )
    result = measure(grid, cmd_cfg)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
