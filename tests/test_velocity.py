"""Open-loop checks of the velocity-command layer on the 480/120/30 Hz grid (CPU).

The thresholds are set from the measured numbers in ``results/tracking_480-120-30.json`` with some
margin; ``pytest -s`` prints the values.
"""

import numpy as np
import pytest

from expert_data.tracking import (
    AXES,
    label_consistency,
    run_commands,
    smooth_random_commands,
    step_response,
)
from expert_data.velocity import CommandConfig, RateGrid

GRID = RateGrid()


def test_rate_grid_nests_as_integers():
    assert (GRID.ticks_per_ctrl, GRID.ctrls_per_decision, GRID.ticks_per_decision) == (4, 4, 16)
    alt = RateGrid(500, 100, 25)
    assert (alt.ticks_per_ctrl, alt.ctrls_per_decision, alt.ticks_per_decision) == (5, 4, 20)
    with pytest.raises(ValueError):
        RateGrid(500, 100, 30)  # 30 Hz is not reachable from crazyflow's 500/100 Hz default
    with pytest.raises(ValueError):
        RateGrid(480, 100, 30)


def test_zero_command_holds_hover():
    out = run_commands(np.zeros((90, 2, 4), np.float32), GRID)  # 3 s
    offset = np.linalg.norm(out["pos"] - out["pos"][0], axis=-1)
    print(f"\nhover: max offset {offset.max():.3f} m, final {offset[-1].max():.3f} m")
    # The controller's model mass (29 g vs 32.8 g) leaves a ~3 cm steady sag inside the leash.
    assert offset.max() < 0.06
    assert np.abs(out["realised"][-30:]).max() < 0.02  # at rest after 2 s


def test_measured_anchor_sinks_at_zero_command():
    """Spec eq. (2) taken literally (re-anchored at the measured position) cannot hold altitude."""
    out = run_commands(np.zeros((90, 1, 4), np.float32), GRID, CommandConfig(anchor="measured"))
    sink = out["pos"][0, 0, 2] - out["pos"][-1, 0, 2]
    print(f"\nmeasured anchor: sank {sink:.3f} m in 3 s")
    assert sink > 0.15


@pytest.mark.parametrize("axis,target", [(0, 1.0), (1, 1.0), (2, 0.5), (3, 1.0)])
def test_step_command_is_tracked(axis, target):
    t_len, t_step = 120, 15
    cmds = np.zeros((t_len, 1, 4), np.float32)
    cmds[t_step:, 0, axis] = target
    out = run_commands(cmds, GRID, start_z=5.0)
    m = step_response(out["realised"][:, 0, axis], t_step, target, GRID.dt)
    print(f"\n{AXES[axis]} step {target}: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items()))
    assert abs(m["steady_state_error_rel"]) < 0.05
    assert m["rise_time_s"] < 0.5
    assert m["overshoot_rel"] < 0.5
    others = [i for i in range(4) if i != axis]
    assert np.abs(out["realised"][-30:, 0, others]).max() < 0.1  # no cross-talk at steady state


def _gentle_consistency():
    rand = smooth_random_commands(0, 300, 8, GRID.dt, np.array([0.4, 0.4, 0.2, 0.4]), 0.5)
    out = run_commands(rand, GRID, start_z=20.0)
    return label_consistency(rand[30:], out["realised"][30:])


def test_label_consistency_on_smooth_commands():
    """Spec §3.9 / eq. (3): commanded vs realised, here at the lag the closed loop actually has."""
    r0, lag, r_best = _gentle_consistency()
    print(f"\nr(lag 0) {r0.round(3)}, best lag {lag}, r(best lag) {r_best.round(3)}")
    assert np.all(r_best[:3] >= 0.95)
    assert np.all(lag[:2] <= 5) and lag[2] <= 3 and lag[3] <= 1
    assert r_best[3] >= 0.8  # yaw: tilt couples into the Euler yaw used by eq. (3)


@pytest.mark.xfail(strict=True, reason="xy lag of the Mellinger cascade is ~4 decisions (130 ms)")
def test_spec_lag_criterion_one_step():
    """Spec §3.9 asks for r >= 0.95 at a lag of at most one decision; not met on this airframe."""
    _, lag, r_best = _gentle_consistency()
    assert np.all(lag <= 1) and np.all(r_best >= 0.95)
