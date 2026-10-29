"""Functional gate-flying environment with heading-frame velocity actions (Option A, batched).

The environment is a pair of pure functions ``reset`` / ``step`` over an :class:`EnvState` pytree
that wraps crazyflow's ``SimData``. One step is one 30 Hz decision: the velocity command is executed
through ``Control.state`` by :mod:`expert_data.velocity` (4 state-controller ticks x 4 physics ticks
on the 480/120/30 Hz grid), gate events are geometric tests in JAX (spec eq. 13; no MJX contact
queries), and worlds auto-reset in the same step so the whole rollout can live inside
``jax.lax.scan``.

Observation (45-D, privileged; vectors in the drone's body frame so the policy is yaw-invariant):
``z (1) | gravity dir (3) | vel (3) | ang_vel (3) | rotor_vel normalised (4) | prev action (4) |
current gate: rel pos (3), normal (3), lateral axis (3) | next gate: same (9) |
gates remaining (1) | command reference: p_ref - p (3), v_ref (3), wrap(psi_ref - psi) (1) |
target speed (1)``. The first 37 entries are the RPM expert's observation; the next 7 expose the
velocity layer's carried reference, which is controller state the policy would otherwise not see.

Target speed. Every episode draws ``v_target ~ U[target_speed_min, target_speed_max]``; the policy
observes it and the reward penalises ``k_speed * (|v| - v_target)^2``. This removes the incentive
to fly at the action bound (the progress reward alone pays for speed) and gives the dataset the
speed diversity that speed words in the instructions need (spec §3.7). ``info["target_speed"]``
carries it for the scenario record.

Action: ``a in [-1, 1]^4`` -> ``(v^h_x, v^h_y, v^h_z, omega_z) = a * (v_max_x, v_max_y, v_max_z,
yaw_rate_max)``. This physical command (``info["cmd"]``) is the demonstration label (Option A):
it is exactly what the VLA will send through the same interface.

Ported from crazyflow-gate-ppo (``gate_ppo/env.py``): track sampling, gate test, reward, autoreset
and domain randomisation are unchanged in substance; the action path, the reset of the controller
state, the per-world parameter batching and the target speed are new.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.sim.data import SimData
from flax import struct
from jax import Array

from expert_data.config import EnvConfig
from expert_data.geometry import (
    gate_crossing,
    gate_frame,
    hover_rpm,
    quat_rotate_inv,
    wrap_angle,
    yaw_from_quat,
)
from expert_data.track import sample_circle_track, sample_start, sample_track
from expert_data.velocity import (
    CommandConfig,
    RateGrid,
    VelocityReference,
    make_sim,
    make_velocity_step,
    reset_drones,
)

OBS_DIM = 45
ACTION_DIM = 4
ROTOR_OBS_SCALE = 0.3  # rotor speed observation = (rpm / hover - 1) / this


@struct.dataclass
class EnvParams:
    """Traced environment parameters: change them between iterations without recompiling."""

    yaw_range: Array
    n_gates_min: Array
    n_gates_max: Array
    dr_mass: Array
    dr_inertia: Array
    dr_motor_gain: Array
    circle_prob: Array
    circle_gates_per_turn_min: Array
    circle_gates_per_turn_max: Array
    target_speed_min: Array
    target_speed_max: Array

    @classmethod
    def from_config(
        cls,
        cfg: EnvConfig,
        yaw_range: float | None = None,
        n_gates_min: int | None = None,
        n_gates_max: int | None = None,
        domain_randomization: bool = True,
        circle_prob: float | None = None,
        gates_per_turn: float | None = None,
        target_speed: float | None = None,
    ) -> EnvParams:
        """Parameters from the config; the keyword arguments fix a value (e.g. for evaluation)."""
        dr = 1.0 if domain_randomization else 0.0
        gpt_min = cfg.circle_gates_per_turn_min if gates_per_turn is None else gates_per_turn
        gpt_max = cfg.circle_gates_per_turn_max if gates_per_turn is None else gates_per_turn
        return cls(
            yaw_range=jnp.float32(cfg.yaw_range if yaw_range is None else yaw_range),
            n_gates_min=jnp.int32(cfg.n_gates_min if n_gates_min is None else n_gates_min),
            n_gates_max=jnp.int32(cfg.n_gates if n_gates_max is None else n_gates_max),
            dr_mass=jnp.float32(cfg.dr_mass * dr),
            dr_inertia=jnp.float32(cfg.dr_inertia * dr),
            dr_motor_gain=jnp.float32(cfg.dr_motor_gain * dr),
            circle_prob=jnp.float32(cfg.circle_prob if circle_prob is None else circle_prob),
            circle_gates_per_turn_min=jnp.float32(gpt_min),
            circle_gates_per_turn_max=jnp.float32(gpt_max),
            target_speed_min=jnp.float32(
                cfg.target_speed_min if target_speed is None else target_speed
            ),
            target_speed_max=jnp.float32(
                cfg.target_speed_max if target_speed is None else target_speed
            ),
        )


@struct.dataclass
class EnvState:
    sim: SimData
    ref: VelocityReference  # carried reference of the velocity-command layer
    gates_pos: Array  # (N, G, 3), G = n_gates + 1 (the last one is a continuation gate)
    gates_yaw: Array  # (N, G)
    n_active: Array  # (N,) gates required for success
    gate_idx: Array  # (N,) index of the current target gate
    prev_pos: Array  # (N, 3)
    prev_action: Array  # (N, 4) normalised
    t: Array  # (N,) decisions since reset
    ep_return: Array  # (N,)
    target_speed: Array  # (N,) m/s, drawn per episode
    rng: Array


class GateVelocityEnv:
    """Batched gate-flying task: holds the crazyflow ``Sim`` and constants; state is a pytree."""

    def __init__(self, cfg: EnvConfig, device: str = "cpu"):
        if not 1 <= cfg.n_gates_min <= cfg.n_gates:
            raise ValueError("need 1 <= n_gates_min <= n_gates")
        if cfg.track_type not in ("chain", "circle", "mixed"):
            raise ValueError(f"unknown track_type {cfg.track_type!r}")
        self.cfg = cfg
        self.grid = RateGrid(cfg.sim_freq, cfg.state_freq, cfg.decision_freq)
        self.cmd_cfg = CommandConfig(
            anchor=cfg.cmd_anchor,
            acc_max_xy=cfg.cmd_acc_max_xy,
            acc_max_z=cfg.cmd_acc_max_z,
            leash=cfg.cmd_leash,
            yaw_leash=cfg.cmd_yaw_leash,
        )
        self.n_envs = cfg.n_envs
        self.n_gates = cfg.n_gates
        self.n_total_gates = cfg.n_gates + 1
        self.sim = make_sim(cfg.n_envs, self.grid, drone=cfg.drone, device=device)
        self.sim.reset()
        self._default_controls = self.sim.default_data.controls
        self._velocity_step = make_velocity_step(self.sim.build_step_fn(), self.grid, self.cmd_cfg)
        params = load_params(Dynamics.first_principles, cfg.drone)
        self.mass0 = float(np.asarray(params["mass"]))
        self.J0 = np.asarray(params["J"], dtype=np.float32)
        self.rpm2thrust0 = np.asarray(params["rpm2thrust"], dtype=np.float32).reshape(1, 3)
        self.hover_rpm = float(hover_rpm(self.rpm2thrust0[0], self.mass0))
        self.action_scale = jnp.array(
            [cfg.v_max_x, cfg.v_max_y, cfg.v_max_z, cfg.yaw_rate_max], dtype=jnp.float32
        )
        self.obs_dim = OBS_DIM
        self.action_dim = ACTION_DIM

    # ----------------------------------------------------------------------------------- state
    def init_state(self, key: Array) -> EnvState:
        """Allocate an ``EnvState`` from the freshly reset ``Sim``; call :meth:`reset` next.

        crazyflow 0.4.0 creates the dynamics parameters without a world axis; they are broadcast
        here to ``(N, 1, ...)`` so that domain randomisation can write one value per world.
        """
        n, g = self.n_envs, self.n_total_gates
        sim = self.sim.data
        p = sim.params
        sim = sim.replace(
            params=p.replace(
                mass=jnp.broadcast_to(p.mass, (n, 1, 1)),
                J=jnp.broadcast_to(p.J, (n, 1, 3, 3)),
                J_inv=jnp.broadcast_to(p.J_inv, (n, 1, 3, 3)),
                rpm2thrust=jnp.broadcast_to(p.rpm2thrust, (n, 1, 1, 3)),
            )
        )
        zeros3 = jnp.zeros((n, 3))
        return EnvState(
            sim=sim,
            ref=VelocityReference.at(zeros3, jnp.zeros((n,))),
            gates_pos=jnp.zeros((n, g, 3)),
            gates_yaw=jnp.zeros((n, g)),
            n_active=jnp.full((n,), self.n_gates, dtype=jnp.int32),
            gate_idx=jnp.zeros((n,), dtype=jnp.int32),
            prev_pos=zeros3,
            prev_action=jnp.zeros((n, ACTION_DIM)),
            t=jnp.zeros((n,), dtype=jnp.int32),
            ep_return=jnp.zeros((n,)),
            target_speed=jnp.zeros((n,)),
            rng=key,
        )

    def reset(
        self, state: EnvState, params: EnvParams, mask: Array | None = None
    ) -> tuple[EnvState, Array]:
        """Re-initialise the worlds selected by ``mask`` (all if None). Returns ``(state, obs)``."""
        mask = jnp.ones((self.n_envs,), dtype=bool) if mask is None else mask
        rng, key = jax.random.split(state.rng)
        state = self._reset_worlds(state.replace(rng=rng), mask, key, params)
        return state, self.observe(state)

    def _sample_tracks(self, key: Array, params: EnvParams):
        cfg, n, g = self.cfg, self.n_envs, self.n_total_gates
        k_track, k_type = jax.random.split(key)
        common = dict(
            n_gates=g,
            spacing_min=cfg.spacing_min,
            spacing_max=cfg.spacing_max,
            dz_max=cfg.dz_max,
            z_min=cfg.z_min,
            z_max=cfg.z_max,
            start_z_min=cfg.start_z_min,
            start_z_max=cfg.start_z_max,
        )
        chain_fn = partial(
            sample_track,
            yaw_range=params.yaw_range,
            gate_yaw_noise=cfg.gate_yaw_noise,
            **common,
        )
        circle_fn = partial(
            sample_circle_track,
            gates_per_turn_min=params.circle_gates_per_turn_min,
            gates_per_turn_max=params.circle_gates_per_turn_max,
            radius_noise=cfg.circle_radius_noise,
            angle_noise=cfg.circle_angle_noise,
            yaw_noise=cfg.circle_yaw_noise,
            **common,
        )
        track_keys = jax.random.split(k_track, n)
        if cfg.track_type == "chain":
            return jax.vmap(chain_fn)(track_keys)
        if cfg.track_type == "circle":
            return jax.vmap(circle_fn)(track_keys)
        # mixed: draw both families, pick per world (Phase-1 lesson: the chain-only sampler made
        # every gate normal point along the path and the expert failed on circles)
        chain, circle = jax.vmap(chain_fn)(track_keys), jax.vmap(circle_fn)(track_keys)
        is_circle = jax.random.bernoulli(k_type, params.circle_prob, (n,))
        return jax.tree.map(
            lambda a, b: jnp.where(is_circle.reshape((n,) + (1,) * (a.ndim - 1)), b, a),
            chain,
            circle,
        )

    def _reset_worlds(
        self, state: EnvState, mask: Array, key: Array, params: EnvParams
    ) -> EnvState:
        cfg, n = self.cfg, self.n_envs
        k_track, k_active, k_start, k_dr, k_speed = jax.random.split(key, 5)
        gates_pos, gates_yaw, start_pos, start_yaw = self._sample_tracks(k_track, params)
        n_active = jax.random.randint(
            k_active, (n,), params.n_gates_min, params.n_gates_max + 1, dtype=jnp.int32
        )
        target_speed = jax.random.uniform(
            k_speed, (n,), minval=params.target_speed_min, maxval=params.target_speed_max
        )
        pos, quat, vel, ang_vel, _ = sample_start(
            k_start,
            start_pos,
            start_yaw,
            hover_rpm=self.hover_rpm,
            xy_jitter=cfg.start_xy_jitter,
            yaw_noise=cfg.start_yaw_noise,
            vel_max=cfg.start_vel_max,
            tilt_max=cfg.start_tilt_max,
            ang_vel_max=cfg.start_ang_vel_max,
            rpm_jitter=0.0,
        )
        # Domain randomisation: per-world mass, inertia and thrust-curve scale ("motor constant").
        k_mass, k_inertia, k_gain = jax.random.split(k_dr, 3)
        mass = self.mass0 * (
            1.0 + jax.random.uniform(k_mass, (n,), minval=-params.dr_mass, maxval=params.dr_mass)
        )
        inertia_scale = 1.0 + jax.random.uniform(
            k_inertia, (n, 1, 3, 1), minval=-params.dr_inertia, maxval=params.dr_inertia
        )
        inertia = jnp.asarray(self.J0)[None, None] * inertia_scale
        gain = 1.0 + jax.random.uniform(
            k_gain, (n,), minval=-params.dr_motor_gain, maxval=params.dr_motor_gain
        )
        rpm2thrust = jnp.asarray(self.rpm2thrust0)[None, None] * gain[:, None, None, None]
        # Pre-spin every rotor exactly at this world's hover speed (Phase-1 lesson: unequal rotor
        # speeds during spin-up act as a torque impulse and tumble the drone).
        hover = hover_rpm(self.rpm2thrust0[0], mass / gain)
        rotor_vel = jnp.broadcast_to(hover[:, None], (n, 4))

        sim = reset_drones(
            state.sim, self._default_controls, mask, pos, quat, vel, ang_vel, rotor_vel
        )

        def sel(new: Array, old: Array) -> Array:
            m = mask.reshape((n,) + (1,) * (new.ndim - 1))
            return jnp.where(m, new, old)

        p = sim.params
        sim = sim.replace(
            params=p.replace(
                mass=sel(mass[:, None, None], p.mass),
                J=sel(inertia, p.J),
                J_inv=sel(jnp.linalg.inv(inertia), p.J_inv),
                rpm2thrust=sel(rpm2thrust, p.rpm2thrust),
            )
        )
        ref = VelocityReference(
            pos=sel(pos, state.ref.pos),
            vel=sel(vel, state.ref.vel),
            yaw=sel(yaw_from_quat(quat), state.ref.yaw),
        )
        return state.replace(
            sim=sim,
            ref=ref,
            gates_pos=sel(gates_pos, state.gates_pos),
            gates_yaw=sel(gates_yaw, state.gates_yaw),
            n_active=sel(n_active, state.n_active),
            gate_idx=sel(jnp.zeros_like(state.gate_idx), state.gate_idx),
            prev_pos=sel(pos, state.prev_pos),
            prev_action=sel(jnp.zeros_like(state.prev_action), state.prev_action),
            t=sel(jnp.zeros_like(state.t), state.t),
            ep_return=sel(jnp.zeros_like(state.ep_return), state.ep_return),
            target_speed=sel(target_speed, state.target_speed),
        )

    # ------------------------------------------------------------------------------ observation
    def observe(self, state: EnvState) -> Array:
        cfg, n = self.cfg, self.n_envs
        st = state.sim.states
        pos, quat, vel, ang_vel = st.pos[:, 0], st.quat[:, 0], st.vel[:, 0], st.ang_vel[:, 0]
        rotor = st.rotor_vel[:, 0]
        ar = jnp.arange(n)
        idx = state.gate_idx
        idx_next = jnp.minimum(idx + 1, self.n_total_gates - 1)
        g0, y0 = state.gates_pos[ar, idx], state.gates_yaw[ar, idx]
        g1, y1 = state.gates_pos[ar, idx_next], state.gates_yaw[ar, idx_next]
        n0, u0 = gate_frame(y0)
        n1, u1 = gate_frame(y1)
        to_body = lambda v: quat_rotate_inv(quat, v)
        gravity_dir = to_body(jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), (n, 3)))
        remaining = jnp.minimum(state.n_active - idx, cfg.obs_remaining_cap)
        remaining = remaining.astype(jnp.float32)[:, None]
        rotor_norm = (rotor / self.hover_rpm - 1.0) / ROTOR_OBS_SCALE
        ref_yaw_err = wrap_angle(state.ref.yaw - yaw_from_quat(quat))[:, None]
        return jnp.concatenate(
            [
                pos[:, 2:3],
                gravity_dir,
                to_body(vel),
                ang_vel,
                rotor_norm,
                state.prev_action,
                to_body(g0 - pos),
                to_body(n0),
                to_body(u0),
                to_body(g1 - pos),
                to_body(n1),
                to_body(u1),
                remaining,
                to_body(state.ref.pos - pos),
                to_body(state.ref.vel),
                ref_yaw_err,
                state.target_speed[:, None],
            ],
            axis=-1,
        )

    # ------------------------------------------------------------------------------------ step
    def max_steps(self, n_active: Array, target_speed: Array) -> Array:
        """Episode budget in decisions; slower target speeds get proportionally more time."""
        cfg = self.cfg
        per_gate = cfg.time_slack * cfg.spacing_max / target_speed
        return (cfg.time_base + n_active * per_gate) * cfg.decision_freq

    def command(self, action: Array) -> Array:
        """Physical command ``(N, 4)`` (m/s, m/s, m/s, rad/s) of a normalised action."""
        return jnp.clip(action, -1.0, 1.0) * self.action_scale

    def step(
        self, state: EnvState, action: Array, params: EnvParams
    ) -> tuple[EnvState, Array, Array, Array, Array, Array, dict[str, Array]]:
        """Advance one decision and auto-reset finished worlds.

        Returns:
            ``(state, obs, reward, terminated, truncated, final_obs, info)``. ``obs`` is the
            observation *after* auto-reset; ``final_obs`` is the pre-reset observation, needed to
            bootstrap the value of truncated episodes. ``info`` carries pre-reset quantities.
        """
        cfg, n = self.cfg, self.n_envs
        a = jnp.clip(action, -1.0, 1.0)
        cmd = a * self.action_scale
        sim, ref = self._velocity_step(state.sim, state.ref, cmd)

        st = sim.states
        pos, quat, vel, ang_vel = st.pos[:, 0], st.quat[:, 0], st.vel[:, 0], st.ang_vel[:, 0]
        ar = jnp.arange(n)
        idx = state.gate_idx
        g, yaw = state.gates_pos[ar, idx], state.gates_yaw[ar, idx]

        passed, hit, missed = gate_crossing(
            state.prev_pos, pos, g, yaw, cfg.half_opening, cfg.frame_thickness, cfg.drone_radius
        )
        d_prev = jnp.linalg.norm(state.prev_pos - g, axis=-1)
        d_new = jnp.linalg.norm(pos - g, axis=-1)
        progress = d_prev - d_new

        rel_b = quat_rotate_inv(quat, g - pos)
        cos_heading = rel_b[:, 0] / (jnp.linalg.norm(rel_b[:, :2], axis=-1) + 1e-6)
        r_yaw = -cfg.k_yaw * (1.0 - cos_heading)
        r_rate = -cfg.k_ang_vel * jnp.sum(ang_vel**2, axis=-1)
        r_act = -cfg.k_action_rate * jnp.sum((a - state.prev_action) ** 2, axis=-1)
        speed = jnp.linalg.norm(vel, axis=-1)
        r_speed = -cfg.k_speed * (speed - state.target_speed) ** 2

        cos_tilt = quat_rotate_inv(quat, jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), (n, 3)))
        tilted = cos_tilt[:, 2] < jnp.cos(cfg.max_tilt)
        floor = pos[:, 2] < cfg.z_floor
        ceiling = pos[:, 2] > cfg.z_ceiling
        far = d_new > cfg.max_gate_dist
        crashed = hit | floor | ceiling | far | tilted
        gate_idx_new = idx + passed.astype(jnp.int32)
        success = passed & (gate_idx_new >= state.n_active)
        terminated = crashed | missed | success

        reward = (
            cfg.k_progress * progress
            + cfg.gate_bonus * passed
            - cfg.crash_penalty * crashed
            - cfg.miss_penalty * missed
            + r_yaw
            + r_rate
            + r_act
            + r_speed
        )
        t_new = state.t + 1
        truncated = (t_new >= self.max_steps(state.n_active, state.target_speed)) & ~terminated
        done = terminated | truncated
        ep_return = state.ep_return + reward

        new_state = state.replace(
            sim=sim,
            ref=ref,
            gate_idx=jnp.minimum(gate_idx_new, self.n_total_gates - 1),
            prev_pos=pos,
            prev_action=a,
            t=t_new,
            ep_return=ep_return,
        )
        final_obs = self.observe(new_state)
        info = {
            "episode_return": ep_return,
            "episode_length": t_new.astype(jnp.float32),
            "gates_passed": gate_idx_new.astype(jnp.float32),
            "success": success.astype(jnp.float32),
            "crashed": crashed.astype(jnp.float32),
            "missed": missed.astype(jnp.float32),
            "speed": speed,
            "target_speed": state.target_speed,
            "cmd": cmd,
            "pos": pos,
            "quat": quat,
            "vel": vel,
        }
        rng, key = jax.random.split(state.rng)
        reset_state = self._reset_worlds(new_state.replace(rng=rng), done, key, params)
        return (
            reset_state,
            self.observe(reset_state),
            reward,
            terminated,
            truncated,
            final_obs,
            info,
        )
