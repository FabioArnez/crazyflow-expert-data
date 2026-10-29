"""Open-loop tracking checks of the velocity-command layer (used by the tests and by
``scripts/measure_tracking.py``).

``run_commands`` flies a batch of worlds through a given command sequence (no policy, no gates) and
logs the pose at every decision. The metrics follow spec §3.9: the per-dimension Pearson correlation
between the commanded ``a_t`` and the realised ``bar a_t`` of eq. (3), and the lag (in decision
steps) that maximises their cross-correlation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import Dynamics, load_params

from expert_data.geometry import hover_rpm, quat_from_rpy, yaw_from_quat
from expert_data.velocity import (
    CommandConfig,
    RateGrid,
    VelocityReference,
    make_sim,
    make_velocity_step,
    realised_command,
    reset_drones,
)

AXES = ("vx", "vy", "vz", "wz")


def nominal_hover_rpm(drone: str = "cf2x_L250") -> float:
    params = load_params(Dynamics.first_principles, drone)
    return float(hover_rpm(np.asarray(params["rpm2thrust"]), float(np.asarray(params["mass"]))))


def run_commands(
    commands: np.ndarray,
    grid: RateGrid,
    cmd_cfg: CommandConfig | None = None,
    start_z: float = 1.0,
    start_yaw: float = 0.0,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Fly ``commands`` ``(T, N, 4)`` (heading-frame m/s, rad/s) from hover; one world per column.

    Returns ``pos (T+1, N, 3)``, ``yaw (T+1, N)``, ``vel (T+1, N, 3)`` at every decision and the
    realised command ``realised (T, N, 4)`` of spec eq. (3).
    """
    t_len, n, _ = commands.shape
    sim = make_sim(n, grid, device=device)
    sim.reset()
    step = make_velocity_step(sim.build_step_fn(), grid, cmd_cfg)
    pos0 = jnp.tile(jnp.array([0.0, 0.0, start_z]), (n, 1))
    yaw0 = jnp.full((n,), start_yaw)
    quat0 = quat_from_rpy(jnp.zeros(n), jnp.zeros(n), yaw0)
    data = reset_drones(
        sim.data,
        sim.default_data.controls,
        jnp.ones((n,), dtype=bool),
        pos0,
        quat0,
        jnp.zeros((n, 3)),
        jnp.zeros((n, 3)),
        jnp.full((n, 4), nominal_hover_rpm()),
    )
    ref = VelocityReference.at(pos0, yaw0)

    def body(carry, cmd):
        data, ref = carry
        data, ref = step(data, ref, cmd)
        st = data.states
        return (data, ref), (st.pos[:, 0], yaw_from_quat(st.quat[:, 0]), st.vel[:, 0])

    _, (pos, yaw, vel) = jax.jit(lambda c, x: jax.lax.scan(body, c, x))(
        (data, ref), jnp.asarray(commands, dtype=jnp.float32)
    )
    pos = jnp.concatenate([pos0[None], pos])
    yaw = jnp.concatenate([yaw0[None], yaw])
    vel = jnp.concatenate([jnp.zeros((1, n, 3)), vel])
    realised = realised_command(pos[:-1], pos[1:], yaw[:-1], yaw[1:], grid.dt)
    return {
        "pos": np.asarray(pos),
        "yaw": np.asarray(yaw),
        "vel": np.asarray(vel),
        "realised": np.asarray(realised),
    }


def step_response(
    signal: np.ndarray, t_step: int, target: float, dt: float, settle_window: float = 1.0
) -> dict[str, float]:
    """Rise time (10-90 %), steady-state error and overshoot of a step from 0 to ``target``.

    ``signal[i]`` is the realised value over decision ``i`` (eq. 3); the step is applied from
    decision ``t_step`` on.
    """
    y = signal[t_step:] / target
    above10 = np.flatnonzero(y >= 0.1)
    above90 = np.flatnonzero(y >= 0.9)
    rise = (above90[0] - above10[0]) * dt if above10.size and above90.size else float("nan")
    n_settle = max(1, int(round(settle_window / dt)))
    ss = float(np.mean(signal[-n_settle:]) - target)
    return {
        "rise_time_s": float(rise),
        "delay_to_50pct_s": float(np.flatnonzero(y >= 0.5)[0] * dt) if (y >= 0.5).any() else np.nan,
        "steady_state_error": ss,
        "steady_state_error_rel": ss / target,
        "overshoot_rel": float(max(0.0, y.max() - 1.0)),
    }


def label_consistency(commanded: np.ndarray, realised: np.ndarray, max_lag: int = 10):
    """Per-dimension Pearson r at zero lag and the lag (decisions) maximising the correlation.

    Args:
        commanded, realised: ``(T, ..., 4)``; extra axes are treated as independent series.

    Returns:
        ``(r0 (4,), best_lag (4,), r_best (4,))``; a positive lag means the realised signal trails.
    """
    t_len = commanded.shape[0]
    c = commanded.reshape(t_len, -1, commanded.shape[-1])
    r = realised.reshape(t_len, -1, realised.shape[-1])

    def corr(lag: int) -> np.ndarray:
        a, b = c[: t_len - lag], r[lag:]
        a = a - a.mean(0)
        b = b - b.mean(0)
        num = (a * b).sum(0)
        den = np.sqrt((a * a).sum(0) * (b * b).sum(0)) + 1e-12
        return (num / den).mean(0)  # average over series

    table = np.stack([corr(lag) for lag in range(max_lag + 1)])  # (L, 4)
    return table[0], table.argmax(0), table.max(0)


def masked_label_consistency(
    commanded: np.ndarray, realised: np.ndarray, mask: np.ndarray, max_lag: int = 10
):
    """:func:`label_consistency` for variable-length episodes, pooled over worlds.

    Args:
        commanded, realised: ``(T, N, 4)``.
        mask: ``(T, N)``, True where the decision belongs to the evaluated episode.

    Returns:
        ``(r0 (4,), best_lag (4,), r_best (4,))``; pairs ``(t, t + lag)`` count only if both are
        inside the mask.
    """
    t_len = commanded.shape[0]

    def corr(lag: int) -> np.ndarray:
        m = (mask[: t_len - lag] & mask[lag:])[..., None]
        n = max(int(m.sum()), 1)
        a, b = commanded[: t_len - lag], realised[lag:]
        a = np.where(m, a - (a * m).sum((0, 1)) / n, 0.0)
        b = np.where(m, b - (b * m).sum((0, 1)) / n, 0.0)
        return (a * b).sum((0, 1)) / (np.sqrt((a * a).sum((0, 1)) * (b * b).sum((0, 1))) + 1e-12)

    table = np.stack([corr(lag) for lag in range(max_lag + 1)])
    return table[0], table.argmax(0), table.max(0)


def smooth_random_commands(
    key: int, t_len: int, n: int, dt: float, scale: np.ndarray, cutoff_hz: float = 1.0
) -> np.ndarray:
    """Band-limited random commands ``(t_len, n, 4)``: white noise through a 2nd-order low-pass.

    Approximates what an expert issues (smooth, few-Hz content); ``scale`` is the per-axis standard
    deviation after filtering.
    """
    rng = np.random.default_rng(key)
    x = rng.standard_normal((t_len, n, 4))
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz * dt)
    for _ in range(2):
        y = np.zeros_like(x)
        for i in range(1, t_len):
            y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
        x = y
    x = x / (x[t_len // 10 :].std(axis=(0, 1)) + 1e-9)
    return (x * np.asarray(scale)).astype(np.float32)
