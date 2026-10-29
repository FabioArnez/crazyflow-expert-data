import jax
import jax.numpy as jnp
import numpy as np
import pytest

from expert_data.config import Config, EnvConfig, NetConfig, PPOConfig, RunConfig
from expert_data.env import EnvParams
from expert_data.ppo import (
    RunningStats,
    gaussian_entropy,
    gaussian_log_prob,
    policy_action,
    policy_entropy,
    policy_log_prob,
    tanh_log_det,
)
from expert_data.train import build


def test_tanh_log_prob_is_change_of_variables():
    mean, log_std = jnp.array([0.3, -1.0]), jnp.array([-0.5, 0.2])
    u = jnp.array([[0.1, -2.0], [1.5, 0.4]])
    a = policy_action(u, squash=True)
    # density of a = tanh(u): p_u(atanh a) / prod(1 - a^2)
    expected = gaussian_log_prob(jnp.arctanh(a), mean, log_std) - jnp.sum(jnp.log1p(-(a**2)), -1)
    np.testing.assert_allclose(policy_log_prob(u, mean, log_std, True), expected, rtol=1e-4)
    np.testing.assert_allclose(
        policy_log_prob(u, mean, log_std, False), gaussian_log_prob(u, mean, log_std)
    )


def test_tanh_log_det_is_stable():
    u = jnp.array([[0.0, 5.0, -20.0, 40.0]])
    out = tanh_log_det(u)
    assert bool(jnp.all(jnp.isfinite(out)))
    np.testing.assert_allclose(
        tanh_log_det(u[:, :2]), jnp.sum(jnp.log(1 - jnp.tanh(u[:, :2]) ** 2)), rtol=1e-4
    )


def test_squashed_actions_stay_inside_the_bounds():
    u = jax.random.normal(jax.random.key(0), (1000, 4)) * 2.0
    a = np.asarray(policy_action(u, squash=True))
    assert np.all(np.abs(a) < 1.0)
    assert np.all(np.abs(np.asarray(policy_action(u, squash=False))) <= 1.0)


def test_squashed_entropy_penalises_saturation():
    key = jax.random.key(1)
    log_std = jnp.full((4,), -0.5)
    centred = policy_entropy(key, jnp.zeros((4096, 4)), log_std, True).mean()
    saturated = policy_entropy(key, jnp.full((4096, 4), 3.0), log_std, True).mean()
    assert float(centred) < float(gaussian_entropy(log_std))
    assert float(saturated) < float(centred) - 5.0
    # Monte-Carlo check of the centred estimate against the entropy of tanh(N(0, sigma)).
    u = jnp.exp(log_std[0]) * jax.random.normal(jax.random.key(2), (200_000,))
    h_1d = float(gaussian_entropy(log_std[:1]) + jnp.mean(jnp.log(1 - jnp.tanh(u) ** 2)))
    np.testing.assert_allclose(float(centred), 4 * h_1d, atol=0.05)


def test_running_stats_matches_numpy():
    stats = RunningStats.create(3)
    rng = np.random.default_rng(0)
    chunks = [rng.normal(size=(50, 3)) * 2 + 1, rng.normal(size=(70, 3)) * 0.5 - 1]
    for c in chunks:
        stats = stats.update(jnp.asarray(c))
    all_x = np.concatenate(chunks)
    np.testing.assert_allclose(np.asarray(stats.mean), all_x.mean(0), atol=1e-3)
    np.testing.assert_allclose(np.asarray(stats.var), all_x.var(0), atol=1e-3)


@pytest.mark.parametrize("squash", [True, False])
def test_one_train_iteration_runs(squash):
    cfg = Config(
        env=EnvConfig(n_envs=8, n_gates=2, n_gates_min=1),
        net=NetConfig(squash=squash),
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
    for key in keys + ("rollout/cmd_speed", "rollout/action_saturation", "rollout/saturation_a0"):
        assert np.isfinite(metrics[key]), key
    assert int(runner[0].n_updates) == 2
    changed = jax.tree.leaves(
        jax.tree.map(lambda a, b: bool(np.any(a != np.asarray(b))), params_before, runner[0].params)
    )
    assert any(changed)
