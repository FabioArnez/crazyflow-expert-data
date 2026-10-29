import jax
import jax.numpy as jnp
import numpy as np

from expert_data.config import EnvConfig
from expert_data.env import ACTION_DIM, OBS_DIM, EnvParams, GateVelocityEnv


def make_env(n_envs=4, n_gates=3, n_gates_min=2, **kw):
    cfg = EnvConfig(n_envs=n_envs, n_gates=n_gates, n_gates_min=n_gates_min, **kw)
    env = GateVelocityEnv(cfg, device="cpu")
    params = EnvParams.from_config(cfg)
    state = env.init_state(jax.random.key(0))
    state, obs = env.reset(state, params)
    return env, params, state, obs


def test_reset_shapes_and_track_constraints():
    env, params, state, obs = make_env()
    cfg = env.cfg
    assert obs.shape == (4, OBS_DIM)
    assert bool(jnp.all(jnp.isfinite(obs)))
    gates = np.asarray(state.gates_pos)
    assert gates.shape == (4, cfg.n_gates + 1, 3)
    assert np.all(gates[..., 2] >= cfg.z_min - 1e-6) and np.all(gates[..., 2] <= cfg.z_max + 1e-6)
    n_active = np.asarray(state.n_active)
    assert np.all(n_active >= 2) and np.all(n_active <= 3)
    # Per-world dynamics parameters were randomised; rotors spin exactly at each world's hover.
    mass = np.asarray(state.sim.params.mass[:, 0, 0])
    assert np.all(np.abs(mass / env.mass0 - 1.0) <= cfg.dr_mass + 1e-6)
    assert np.unique(mass).size == 4
    rotor = np.asarray(state.sim.states.rotor_vel[:, 0])
    np.testing.assert_allclose(rotor, rotor[:, :1].repeat(4, axis=1))
    gain = np.asarray(state.sim.params.rpm2thrust[:, 0, 0, 2]) / env.rpm2thrust0[0, 2]
    c = env.rpm2thrust0[0]
    thrust = gain * (c[1] * rotor[:, 0] + c[2] * rotor[:, 0] ** 2) * 4
    np.testing.assert_allclose(thrust, mass * 9.81, rtol=1e-4)
    # The command reference starts at the drone's pose.
    np.testing.assert_allclose(state.ref.pos, state.sim.states.pos[:, 0], atol=1e-6)


def test_zero_command_holds_altitude():
    env, params, state, obs = make_env(start_vel_max=0.0, start_ang_vel_max=0.0)
    step = jax.jit(env.step)
    z0 = np.asarray(state.sim.states.pos[:, 0, 2])
    for _ in range(2 * env.cfg.decision_freq):  # two seconds of zero velocity command
        state, obs, reward, terminated, truncated, final_obs, info = step(
            state, jnp.zeros((4, ACTION_DIM)), params
        )
        assert not bool(jnp.any(terminated | truncated))
    z1 = np.asarray(state.sim.states.pos[:, 0, 2])
    # DR (mass +-10 %, thrust +-5 %) changes the steady sag inside the leash, not the hold.
    assert np.all(np.abs(z1 - z0) < 0.12), (z0, z1)
    assert np.all(np.abs(np.asarray(info["vel"])) < 0.1)


def test_forward_command_moves_along_heading():
    env, params, state, obs = make_env(start_vel_max=0.0, start_ang_vel_max=0.0, start_tilt_max=0.0)
    step = jax.jit(env.step)
    p0 = np.asarray(state.sim.states.pos[:, 0])
    yaw0 = np.asarray(state.ref.yaw)
    action = jnp.zeros((4, ACTION_DIM)).at[:, 0].set(0.5)  # 0.75 m/s forward
    for _ in range(env.cfg.decision_freq):
        state, obs, reward, terminated, truncated, final_obs, info = step(state, action, params)
    np.testing.assert_allclose(info["cmd"][:, 0], 0.5 * env.cfg.v_max_x)
    d = np.asarray(info["pos"]) - p0
    heading = np.stack([np.cos(yaw0), np.sin(yaw0)], -1)
    along = (d[:, :2] * heading).sum(-1)
    across = d[:, 0] * -heading[:, 1] + d[:, 1] * heading[:, 0]
    assert np.all(along > 0.4) and np.all(np.abs(across) < 0.1), (along, across)


def test_step_outputs_and_autoreset():
    env, params, state, obs = make_env()
    step = jax.jit(env.step)
    key = jax.random.key(1)
    any_done = False
    for _ in range(150):
        key, k = jax.random.split(key)
        action = jax.random.uniform(k, (4, ACTION_DIM), minval=-1.0, maxval=1.0)
        action = action.at[:, 2].set(1.0)  # climb into the ceiling so that episodes end
        state, obs, reward, terminated, truncated, final_obs, info = step(state, action, params)
        assert obs.shape == final_obs.shape == (4, OBS_DIM)
        assert bool(jnp.all(jnp.isfinite(obs))) and bool(jnp.all(jnp.isfinite(reward)))
        assert np.all(np.abs(np.asarray(info["cmd"])) <= np.asarray(env.action_scale) + 1e-6)
        done = np.asarray(terminated | truncated)
        any_done |= bool(done.any())
        # Worlds that finished were re-initialised in the same step.
        assert np.all(np.asarray(state.t)[done] == 0)
        assert np.all(np.asarray(state.gate_idx)[done] == 0)
    assert any_done  # 0.75 m/s climb reaches the 3.5 m ceiling within ~4 s
