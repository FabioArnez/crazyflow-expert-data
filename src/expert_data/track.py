"""Randomised gate tracks and initial drone states (pure JAX, one world per call; vmap over worlds).

A track is a polyline of ``n_gates`` upright gates. Gate ``i`` sits ``d_i ~ U(spacing)`` metres from
gate ``i-1`` (or from the spawn point) along its own heading ``psi_i = psi_{i-1} + dpsi_i`` with
``dpsi_i ~ U(-yaw_range, yaw_range)``, so a drone that turns by ``dpsi_i`` right after a gate always
arrives at the next one aligned with its normal. Altitude is a clipped random walk in
``[z_min, z_max]`` (the "gate altitude clipped to [0.7, 2.2] m" fix from the discussion log).
``gate_yaw_noise`` rotates each gate normal away from the segment heading (0 = normals aligned
with the path — a regularity the policy will exploit, see README "Generalisation").

A second family, :func:`sample_circle_track`, places the gates on an imperfect circle through the
spawn point so that every turn has the same sign and a fixed magnitude ``2*pi/gates_per_turn``.

Ported unchanged from crazyflow-gate-ppo (``gate_ppo/track.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from expert_data.geometry import quat_from_rpy


def sample_track(
    key: Array,
    n_gates: int,
    yaw_range: Array,
    gate_yaw_noise: float,
    spacing_min: float,
    spacing_max: float,
    dz_max: float,
    z_min: float,
    z_max: float,
    start_z_min: float,
    start_z_max: float,
) -> tuple[Array, Array, Array, Array]:
    """Sample one gate chain.

    Returns:
        ``gates_pos (n_gates, 3)``, ``gates_yaw (n_gates,)``, ``start_pos (3,)``, ``start_yaw ()``.
    """
    k_head, k_z0, k_dyaw, k_d, k_dz, k_gyaw = jax.random.split(key, 6)
    heading0 = jax.random.uniform(k_head, (), minval=-jnp.pi, maxval=jnp.pi)
    z0 = jax.random.uniform(k_z0, (), minval=start_z_min, maxval=start_z_max)
    dyaw = jax.random.uniform(k_dyaw, (n_gates,), minval=-yaw_range, maxval=yaw_range)
    d = jax.random.uniform(k_d, (n_gates,), minval=spacing_min, maxval=spacing_max)
    dz = jax.random.uniform(k_dz, (n_gates,), minval=-dz_max, maxval=dz_max)

    yaws = heading0 + jnp.cumsum(dyaw)
    xy = jnp.cumsum(d[:, None] * jnp.stack([jnp.cos(yaws), jnp.sin(yaws)], axis=-1), axis=0)
    zs = _altitude_walk(z0, dz, z_min, z_max)
    gates_pos = jnp.concatenate([xy, zs[:, None]], axis=-1)
    start_pos = jnp.array([0.0, 0.0, 0.0]).at[2].set(z0)
    # The gate normal equals the segment heading unless gate_yaw_noise decouples them; the
    # chain family's hidden regularity 'normal == direction from the previous gate' is exactly
    # what circular tracks violate.
    gate_yaws = yaws + jax.random.uniform(
        k_gyaw, (n_gates,), minval=-gate_yaw_noise, maxval=gate_yaw_noise
    )
    return gates_pos, gate_yaws, start_pos, heading0


def _altitude_walk(z0: Array, dz: Array, z_min: float, z_max: float) -> Array:
    """Clipped random walk of gate altitudes starting from the spawn altitude."""

    def step(z_prev: Array, dz_i: Array) -> tuple[Array, Array]:
        z = jnp.clip(z_prev + dz_i, z_min, z_max)
        return z, z

    _, zs = jax.lax.scan(step, z0, dz)
    return zs


def sample_circle_track(
    key: Array,
    n_gates: int,
    gates_per_turn_min: Array,
    gates_per_turn_max: Array,
    spacing_min: float,
    spacing_max: float,
    radius_noise: float,
    angle_noise: float,
    yaw_noise: float,
    dz_max: float,
    z_min: float,
    z_max: float,
    start_z_min: float,
    start_z_max: float,
) -> tuple[Array, Array, Array, Array]:
    """Sample one imperfect circular track through the spawn point.

    The drone spawns on the circle heading along its tangent. Consecutive gates are
    ``2*pi/gates_per_turn`` radians apart around the centre, so the heading change per gate has that
    magnitude and a constant sign (clockwise or counter-clockwise, drawn per world). The chord
    between gates is drawn from ``[spacing_min, spacing_max]`` and fixes the radius. Radius, angular
    spacing,    gate yaw and altitude carry noise, hence "imperfect".

    Returns:
        ``gates_pos (n_gates, 3)``, ``gates_yaw (n_gates,)``, ``start_pos (3,)``, ``start_yaw ()``.
    """
    k_head, k_z0, k_sign, k_gpt, k_chord, k_ang, k_rad, k_yaw, k_dz = jax.random.split(key, 9)
    heading0 = jax.random.uniform(k_head, (), minval=-jnp.pi, maxval=jnp.pi)
    z0 = jax.random.uniform(k_z0, (), minval=start_z_min, maxval=start_z_max)
    sign = jnp.where(jax.random.bernoulli(k_sign), 1.0, -1.0)
    gates_per_turn = jax.random.uniform(
        k_gpt, (), minval=gates_per_turn_min, maxval=gates_per_turn_max
    )
    dphi = 2.0 * jnp.pi / gates_per_turn
    chord = jax.random.uniform(k_chord, (), minval=spacing_min, maxval=spacing_max)
    radius = chord / (2.0 * jnp.sin(dphi / 2.0))
    # Centre lies perpendicular to the spawn heading, on the side of the turn.
    centre = radius * jnp.stack(
        [jnp.cos(heading0 + sign * jnp.pi / 2), jnp.sin(heading0 + sign * jnp.pi / 2)]
    )
    phi0 = heading0 - sign * jnp.pi / 2  # polar angle of the spawn point seen from the centre
    steps = dphi * (
        1.0 + jax.random.uniform(k_ang, (n_gates,), minval=-angle_noise, maxval=angle_noise)
    )
    phi = phi0 + sign * jnp.cumsum(steps)
    r = radius * (
        1.0 + jax.random.uniform(k_rad, (n_gates,), minval=-radius_noise, maxval=radius_noise)
    )
    xy = centre + r[:, None] * jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=-1)
    yaws = (
        phi
        + sign * jnp.pi / 2
        + jax.random.uniform(k_yaw, (n_gates,), minval=-yaw_noise, maxval=yaw_noise)
    )
    dz = jax.random.uniform(k_dz, (n_gates,), minval=-dz_max, maxval=dz_max)
    zs = _altitude_walk(z0, dz, z_min, z_max)
    gates_pos = jnp.concatenate([xy, zs[:, None]], axis=-1)
    start_pos = jnp.array([0.0, 0.0, 0.0]).at[2].set(z0)
    return gates_pos, yaws, start_pos, heading0


def sample_start(
    key: Array,
    start_pos: Array,
    start_yaw: Array,
    hover_rpm: float,
    xy_jitter: float,
    yaw_noise: float,
    vel_max: float,
    tilt_max: float,
    ang_vel_max: float,
    rpm_jitter: float,
) -> tuple[Array, Array, Array, Array, Array]:
    """Sample initial drone states for a batch of worlds.

    Args:
        key: PRNG key.
        start_pos: ``(N, 3)`` spawn points from :func:`sample_track`.
        start_yaw: ``(N,)`` spawn headings.
        hover_rpm: Per-motor hover RPM; rotors are pre-spun so the drone does not drop at t=0.

    Returns:
        ``pos (N, 3)``, ``quat (N, 4)``, ``vel (N, 3)``, ``ang_vel (N, 3)``, ``rotor_vel (N, 4)``.
    """
    n = start_pos.shape[0]
    k_xy, k_yaw, k_tilt, k_vel, k_ang, k_rpm = jax.random.split(key, 6)
    xy = start_pos[:, :2] + jax.random.uniform(k_xy, (n, 2), minval=-xy_jitter, maxval=xy_jitter)
    pos = jnp.concatenate([xy, start_pos[:, 2:3]], axis=-1)
    yaw = start_yaw + jax.random.uniform(k_yaw, (n,), minval=-yaw_noise, maxval=yaw_noise)
    tilt = jax.random.uniform(k_tilt, (n, 2), minval=-tilt_max, maxval=tilt_max)
    quat = quat_from_rpy(tilt[:, 0], tilt[:, 1], yaw)
    vel = jax.random.uniform(k_vel, (n, 3), minval=-vel_max, maxval=vel_max)
    ang_vel = jax.random.uniform(k_ang, (n, 3), minval=-ang_vel_max, maxval=ang_vel_max)
    rotor_vel = hover_rpm * (
        1.0 + jax.random.uniform(k_rpm, (n, 4), minval=-rpm_jitter, maxval=rpm_jitter)
    )
    return pos, quat, vel, ang_vel, rotor_vel
