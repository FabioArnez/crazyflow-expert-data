"""Heading-frame velocity commands executed through crazyflow's state controller (Option A).

The action of the expert (and later of the VLA) is spec eq. (1)

    a_t = (v^h_x, v^h_y, v^h_z, omega_z),    v^h = R_psi^T v^W,

a velocity in the heading frame (world frame rotated by yaw only; x forward, y left, z up) plus a
yaw rate, held for one decision period ``Delta = 1 / decision_freq``. crazyflow 0.4.0 has no
velocity mode; ``Control.state`` takes a 16-D full-state setpoint
``[x, y, z, vx, vy, vz, ax, ay, az, qx, qy, qz, qw, wx, wy, wz]`` that drives the Mellinger
position controller (see ``docs/crazyflow040_state_control.md``). A velocity command becomes a
stream of state setpoints through an integrated reference (spec eq. 2). Two variants exist:

``integrated`` (default)
    A reference ``(p_ref, v_ref, psi_ref)`` is carried across decisions and advanced at every
    state-controller tick (4 per decision on the 480/120/30 Hz grid):

        v_ref <- v_ref + clip(R_psi v^h - v_ref, a_max * dt_ctrl)          (rate limit)
        a_ref  = (change of v_ref) / dt_ctrl                               (feed-forward)
        p_ref <- p_ref + v_ref dt_ctrl + a_ref dt_ctrl^2 / 2
        psi_ref <- psi_ref + omega_z dt_ctrl,   body-rate feed-forward (0, 0, omega_z)

    The rate limit keeps the reference physically feasible. The Mellinger controller has no tilt
    limit, so an infeasible reference (a velocity step) produces large position/velocity errors, a
    tilt demand of tens of degrees, motor saturation and, for diagonal errors of ~1 m, a tumble.
    ``a_max`` plays the role of the Crazyflie firmware's roll/pitch limit in its velocity mode.
    The offset ``p_ref - p`` is clipped to ``leash`` metres at every decision (and the yaw offset to
    ``yaw_leash``) so the reference cannot wind up when the drone is blocked or saturated; inside
    the leash, the accumulated position error lets the P term cancel gravity-model and drag errors,
    so a zero command holds position and the realised velocity has no steady-state error.

``measured`` (spec eq. 2 literally)
    ``p_ref = p_t + Delta R_psi v^h`` from the measured position, ``v_ref = R_psi v^h``, one
    setpoint per decision. Every decision discards the accumulated error, so the loop degrades to
    velocity damping with a steady-state error from the controller's mass mismatch (the drone sinks
    at zero command). Kept as the baseline of ``scripts/measure_tracking.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from crazyflow.control import Control
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.sim import functional as F
from crazyflow.sim.data import SimData
from crazyflow.utils import leaf_replace, pytree_replace, world_mask
from flax import struct
from jax import Array

from expert_data.geometry import heading_to_world, quat_from_rpy, wrap_angle, yaw_from_quat

STATE_DIM = 16
ANCHORS = ("integrated", "measured")


@dataclass(frozen=True)
class RateGrid:
    """Physics : state controller : decision rates. They must nest as integers (spec §1.5).

    The attitude and force-torque stages run at ``sim_freq`` (every tick) so that their internal
    ``dt = 1 / freq`` matches the integration step; crazyflow's default of 500 Hz would run every
    tick at 480 Hz physics but with a 4 % wrong ``dt`` in the D and I terms.
    """

    sim_freq: int = 480
    state_freq: int = 120
    decision_freq: int = 30

    def __post_init__(self):
        if self.sim_freq % self.state_freq or self.state_freq % self.decision_freq:
            raise ValueError(
                f"rates must nest as integers, got {self.sim_freq}/{self.state_freq}/"
                f"{self.decision_freq} Hz"
            )

    @property
    def ticks_per_ctrl(self) -> int:
        """Physics ticks per state-controller tick."""
        return self.sim_freq // self.state_freq

    @property
    def ctrls_per_decision(self) -> int:
        """State-controller ticks per decision."""
        return self.state_freq // self.decision_freq

    @property
    def ticks_per_decision(self) -> int:
        return self.sim_freq // self.decision_freq

    @property
    def dt(self) -> float:
        """Decision period ``Delta`` in seconds."""
        return 1.0 / self.decision_freq


@dataclass(frozen=True)
class CommandConfig:
    """How a velocity command is turned into state setpoints (see the module docstring)."""

    anchor: str = "integrated"
    acc_max_xy: float = 2.0  # m/s^2, horizontal rate limit of v_ref (~ 12 deg of tilt)
    acc_max_z: float = 1.0  # m/s^2, vertical rate limit of v_ref (thrust/weight is ~1.5)
    leash: float = 0.3  # m, max |p_ref - p| carried into a decision
    yaw_leash: float = 0.3  # rad, max |psi_ref - psi| carried into a decision

    def __post_init__(self):
        if self.anchor not in ANCHORS:
            raise ValueError(f"anchor must be one of {ANCHORS}, got {self.anchor!r}")


def make_sim(
    n_worlds: int, grid: RateGrid, drone: str = "cf2x_L250", device: str = "cpu", **kwargs
) -> Sim:
    """A crazyflow ``Sim`` in ``Control.state`` mode on the rate grid (first-principles model)."""
    return Sim(
        n_worlds=n_worlds,
        n_drones=1,
        drone=drone,
        dynamics=Dynamics.first_principles,
        control=Control.state,
        freq=grid.sim_freq,
        state_freq=grid.state_freq,
        attitude_freq=grid.sim_freq,
        force_torque_freq=grid.sim_freq,
        device=device,
        **kwargs,
    )


@struct.dataclass
class VelocityReference:
    """Reference carried between decisions (world frame)."""

    pos: Array  # (N, 3)
    vel: Array  # (N, 3) rate-limited reference velocity
    yaw: Array  # (N,)

    @classmethod
    def at(cls, pos: Array, yaw: Array, vel: Array | None = None) -> VelocityReference:
        """Reference of a drone at ``pos`` / ``yaw`` moving with ``vel`` (default: at rest)."""
        return cls(pos=pos, vel=jnp.zeros_like(pos) if vel is None else vel, yaw=yaw)


def state_setpoint(
    pos: Array, vel: Array, yaw: Array, yaw_rate: Array, acc: Array | None = None
) -> Array:
    """Pack a ``Control.state`` setpoint ``(N, 1, 16)`` (level attitude; zero acc by default).

    Only the yaw of the quaternion is used by the controller; ``(0, 0, yaw_rate)`` is the body-rate
    feed-forward handed to the attitude stage.
    """
    zeros = jnp.zeros_like(yaw)
    quat = quat_from_rpy(zeros, zeros, yaw)
    rates = jnp.stack([zeros, zeros, yaw_rate], axis=-1)
    acc = jnp.zeros_like(pos) if acc is None else acc
    return jnp.concatenate([pos, vel, acc, quat, rates], axis=-1)[:, None, :]


def _clip_norm(x: Array, max_norm: float) -> Array:
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x * jnp.minimum(1.0, max_norm / jnp.maximum(norm, 1e-9))


def make_velocity_step(
    step_fn: Callable[..., SimData], grid: RateGrid, cfg: CommandConfig | None = None
) -> Callable[[SimData, VelocityReference, Array], tuple[SimData, VelocityReference]]:
    """Build ``velocity_step(sim, ref, cmd) -> (sim, ref)`` that executes one decision.

    Args:
        step_fn: crazyflow's pure step function, ``sim.build_step_fn()``.
        grid: The rate grid; a decision is ``grid.ctrls_per_decision`` state-controller ticks of
            ``grid.ticks_per_ctrl`` physics ticks each.
        cfg: Reference shaping (anchor, acceleration limits, leashes); default ``CommandConfig()``.

    ``cmd`` is the physical command ``(N, 4)``: heading-frame velocity (m/s) and yaw rate (rad/s).
    The heading frame is the drone's measured yaw at the start of the decision (spec eq. 1).
    """
    cfg = CommandConfig() if cfg is None else cfg
    n_sub, ticks, dt = grid.ctrls_per_decision, grid.ticks_per_ctrl, grid.dt
    h = 1.0 / grid.state_freq
    dv_max_xy, dv_max_z = cfg.acc_max_xy * h, cfg.acc_max_z * h

    def velocity_step(
        sim: SimData, ref: VelocityReference, cmd: Array
    ) -> tuple[SimData, VelocityReference]:
        pos, quat = sim.states.pos[:, 0], sim.states.quat[:, 0]
        yaw = yaw_from_quat(quat)
        v_cmd = heading_to_world(cmd[:, :3], yaw)
        yaw_rate = cmd[:, 3]

        if cfg.anchor == "measured":
            setpoint = state_setpoint(pos + dt * v_cmd, v_cmd, yaw + dt * yaw_rate, yaw_rate)
            sim = step_fn(F.state_control(sim, setpoint), n_steps=grid.ticks_per_decision)
            return sim, VelocityReference(
                pos=pos + dt * v_cmd, vel=v_cmd, yaw=wrap_angle(yaw + dt * yaw_rate)
            )

        p0 = pos + _clip_norm(ref.pos - pos, cfg.leash)
        y0 = yaw + jnp.clip(wrap_angle(ref.yaw - yaw), -cfg.yaw_leash, cfg.yaw_leash)

        def sub_step(carry, _):
            sim, p_ref, v_ref, y_ref = carry
            dv = v_cmd - v_ref
            dv = jnp.concatenate(
                [_clip_norm(dv[:, :2], dv_max_xy), jnp.clip(dv[:, 2:], -dv_max_z, dv_max_z)], -1
            )
            acc = dv / h
            sim = F.state_control(sim, state_setpoint(p_ref, v_ref, y_ref, yaw_rate, acc))
            sim = step_fn(sim, n_steps=ticks)
            p_ref = p_ref + h * v_ref + 0.5 * h * dv
            return (sim, p_ref, v_ref + dv, y_ref + h * yaw_rate), None

        (sim, p_ref, v_ref, y_ref), _ = jax.lax.scan(
            sub_step, (sim, p0, ref.vel, y0), None, length=n_sub
        )
        return sim, VelocityReference(pos=p_ref, vel=v_ref, yaw=wrap_angle(y_ref))

    return velocity_step


def reset_drones(
    sim: SimData,
    default_controls: Any,
    mask: Array,
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    rotor_vel: Array,
) -> SimData:
    """Write initial states into the worlds selected by ``mask`` and clear their controllers.

    crazyflow's own ``reset`` restores every per-world leaf from ``default_data``; here the states
    come from the caller, while the controller memory (integral errors, previous rates, last tick
    ``steps = -1``) is restored from ``default_controls`` (``sim.default_data.controls``) so the
    first state-controller tick happens on the first physics tick of the next decision. A
    hold-position setpoint at the new pose is staged. ``rotor_vel`` should be the exact hover speed
    (Phase-1 lesson: unequal rotor speeds at reset act as a torque impulse).

    Args:
        pos, quat, vel, ang_vel: ``(N, 3)``, ``(N, 4)``, ``(N, 3)``, ``(N, 3)`` initial states.
        rotor_vel: ``(N, 4)`` initial rotor speeds in RPM.
    """
    n = mask.shape[0]
    zeros3 = jnp.zeros((n, 1, 3))
    states = leaf_replace(
        sim.states,
        mask,
        pos=pos[:, None],
        quat=quat[:, None],
        vel=vel[:, None],
        ang_vel=ang_vel[:, None],
        rotor_vel=rotor_vel[:, None],
        force=zeros3,
        torque=zeros3,
    )
    controls = pytree_replace(sim.controls, default_controls, world_mask(sim.controls), mask)
    hold = state_setpoint(pos, jnp.zeros_like(pos), yaw_from_quat(quat), jnp.zeros((n,)))
    state_ctrl = leaf_replace(controls.state, mask, cmd=hold, staged_cmd=hold)
    controls = controls.replace(
        state=state_ctrl,
        rotor_vel=jnp.where(mask[:, None, None], rotor_vel[:, None], controls.rotor_vel),
    )
    core = sim.core.replace(steps=jnp.where(mask[:, None], 0, sim.core.steps))
    return sim.replace(states=states, controls=controls, core=core)


def realised_command(pos0: Array, pos1: Array, yaw0: Array, yaw1: Array, dt: float) -> Array:
    """Spec eq. (3): mean realised heading-frame velocity and yaw rate over one decision.

    ``(R_psi0^T (p1 - p0) / dt, wrap(psi1 - psi0) / dt)``; used as the Option-B consistency check
    of the commanded labels (spec §3.9).
    """
    v_h = heading_to_world(pos1 - pos0, -yaw0) / dt
    return jnp.concatenate([v_h, (wrap_angle(yaw1 - yaw0) / dt)[..., None]], axis=-1)
