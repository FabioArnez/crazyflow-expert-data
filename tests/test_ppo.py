import jax
import jax.numpy as jnp
import numpy as np

from expert_data.config import Config, EnvConfig, PPOConfig, RunConfig
from expert_data.env import EnvParams
from expert_data.ppo import RunningStats
from expert_data.train import build


def test_running_stats_matches_numpy():
    stats = RunningStats.create(3)
    rng = np.random.default_rng(0)
    chunks = [rng.normal(size=(50, 3)) * 2 + 1, rng.normal(size=(70, 3)) * 0.5 - 1]
    for c in chunks:
        stats = stats.update(jnp.asarray(c))
    all_x = np.concatenate(chunks)
    np.testing.assert_allclose(np.asarray(stats.mean), all_x.mean(0), atol=1e-3)
    np.testing.assert_allclose(np.asarray(stats.var), all_x.var(0), atol=1e-3)


def test_one_train_iteration_runs():
    cfg = Config(
        env=EnvConfig(n_envs=8, n_gates=2, n_gates_min=1),
        ppo=PPOConfig(n_steps=4, n_epochs=1, n_minibatches=2, total_env_steps=64),
        run=RunConfig(device="cpu", jax_cache=False),
    )
    env, _, train_state, train_iteration, n_iterations, k_env, k_run = build(cfg)
    env_params = EnvParams.from_config(cfg.env)
    env_state = env.init_state(k_env)
    env_state, obs = env.reset(env_state, env_params)
    runner = (train_state, env_state, obs, k_run)
    params_before = jax.tree.map(np.asarray, train_state.params)
    for _ in range(2):
        runner, metrics = train_iteration(runner, env_params)
    metrics = jax.device_get(metrics)
    keys = ("loss/total", "loss/policy", "loss/value", "loss/entropy", "rollout/reward_mean")
    for key in keys + ("rollout/cmd_speed", "rollout/action_saturation"):
        assert np.isfinite(metrics[key]), key
    assert int(runner[0].n_updates) == 2
    changed = jax.tree.leaves(
        jax.tree.map(lambda a, b: bool(np.any(a != np.asarray(b))), params_before, runner[0].params)
    )
    assert any(changed)
