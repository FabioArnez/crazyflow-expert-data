"""Quaternion / heading-frame helpers and the geometric gate-passage test.

Everything is pure ``jax.numpy`` and broadcasts over leading batch dimensions. Quaternions are
scalar-last ``xyzw`` (crazyflow's convention) and rotate body -> world. The *heading frame* is the
world frame rotated about z by the yaw angle only (x forward, y left, z up; spec eq. 1).

The passage test implements spec eq. (13) (and eq. 2 of the crazyflow assessment note): a gate with
centre ``g``, unit normal ``n`` (direction of travel), in-plane lateral axis ``u`` and vertical axis
``z`` is passed at decision step ``t`` iff the drone crosses the gate plane in the direction of
``n`` between ``t-1`` and ``t`` and the interpolated crossing point lies inside the opening.
Crossing inside the frame band is a collision, crossing outside the frame is a miss. Both thresholds
account for the drone radius so the test is on the centre of mass.

Ported from crazyflow-gate-ppo (``gate_ppo/geometry.py``); the heading-frame helpers are new.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def quat_rotate(q: Array, v: Array) -> Array:
    """Rotate ``v`` (..., 3) by the unit quaternion ``q`` (..., 4) xyzw: body -> world."""
    q_xyz, w = q[..., :3], q[..., 3:4]
    t = 2.0 * jnp.cross(q_xyz, v)
    return v + w * t + jnp.cross(q_xyz, t)


def quat_rotate_inv(q: Array, v: Array) -> Array:
    """Rotate ``v`` by the conjugate of ``q``: world -> body."""
    q_conj = jnp.concatenate([-q[..., :3], q[..., 3:4]], axis=-1)
    return quat_rotate(q_conj, v)


def quat_from_rpy(roll: Array, pitch: Array, yaw: Array) -> Array:
    """xyzw quaternion from extrinsic x-y-z Euler angles (scipy ``from_euler("xyz")``)."""
    cr, sr = jnp.cos(roll / 2), jnp.sin(roll / 2)
    cp, sp = jnp.cos(pitch / 2), jnp.sin(pitch / 2)
    cy, sy = jnp.cos(yaw / 2), jnp.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return jnp.stack([x, y, z, w], axis=-1)


def yaw_from_quat(q: Array) -> Array:
    """Yaw angle (rad, z of extrinsic x-y-z Euler angles) of an xyzw quaternion ``(..., 4)``."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a: Array) -> Array:
    """Wrap angles to ``[-pi, pi)``."""
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def heading_to_world(v: Array, yaw: Array) -> Array:
    """``R_psi v``: rotate heading-frame vectors ``(..., 3)`` to the world frame."""
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


def world_to_heading(v: Array, yaw: Array) -> Array:
    """``R_psi^T v``: rotate world-frame vectors ``(..., 3)`` to the heading frame."""
    return heading_to_world(v, -yaw)


def gate_frame(yaw: Array) -> tuple[Array, Array]:
    """Unit normal ``n`` (travel direction) and lateral axis ``u`` of an upright gate.

    ``yaw`` is the gate heading (direction of travel through the opening).

    Returns:
        ``(n, u)`` each of shape ``(..., 3)``. The vertical axis is the world z axis.
    """
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    zero = jnp.zeros_like(yaw)
    n = jnp.stack([c, s, zero], axis=-1)
    u = jnp.stack([-s, c, zero], axis=-1)
    return n, u


def gate_crossing(
    prev_pos: Array,
    pos: Array,
    gate_pos: Array,
    gate_yaw: Array,
    half_opening: float,
    frame_thickness: float,
    drone_radius: float,
) -> tuple[Array, Array, Array]:
    """Classify the segment ``prev_pos -> pos`` against the gate plane.

    Args:
        prev_pos: Drone position at the previous control step, ``(..., 3)``.
        pos: Drone position now, ``(..., 3)``.
        gate_pos: Gate centre, ``(..., 3)``.
        gate_yaw: Gate heading, ``(...)``.
        half_opening: Half the inner opening (square gate), metres.
        frame_thickness: Width of the frame bars, metres.
        drone_radius: Radius of the drone's bounding sphere, metres.

    Returns:
        Boolean arrays ``(passed, hit_frame, missed)`` of shape ``(...)``. At most one is True.
    """
    n, u = gate_frame(gate_yaw)
    s_prev = jnp.sum((prev_pos - gate_pos) * n, axis=-1)
    s = jnp.sum((pos - gate_pos) * n, axis=-1)
    crossing = (s_prev < 0.0) & (s >= 0.0)
    denom = jnp.where(crossing, s - s_prev, 1.0)
    alpha = jnp.where(crossing, -s_prev / denom, 0.0)
    p_cross = prev_pos + alpha[..., None] * (pos - prev_pos)
    d = p_cross - gate_pos
    lat_u = jnp.abs(jnp.sum(d * u, axis=-1))
    lat_z = jnp.abs(d[..., 2])
    lat = jnp.maximum(lat_u, lat_z)  # square opening -> Chebyshev distance from the centre
    w_in = half_opening - drone_radius
    w_out = half_opening + frame_thickness + drone_radius
    passed = crossing & (lat < w_in)
    hit = crossing & (lat >= w_in) & (lat <= w_out)
    missed = crossing & (lat > w_out)
    return passed, hit, missed


def hover_rpm(rpm2thrust, mass: float, g: float = 9.81) -> float:
    """Per-motor RPM at which the quadratic thrust curve carries ``mass`` (4 motors).

    ``rpm2thrust`` is crazyflow's ``[c0, c1, c2]`` with ``f = c0 + c1 * rpm + c2 * rpm**2``.
    """
    c0, c1, c2 = (float(c) for c in rpm2thrust)
    thrust = mass * g / 4.0
    disc = c1 * c1 - 4.0 * c2 * (c0 - thrust)
    return (-c1 + disc**0.5) / (2.0 * c2)
