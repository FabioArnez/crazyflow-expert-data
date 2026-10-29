"""PureJaxRL-pattern PPO: rollout, GAE, and the clipped update in one ``jax.jit`` per iteration.

Refinements over the vault's reference harness (its listed open items): proper truncation
bootstrapping (``V(s_T)`` of the pre-reset observation on timeouts), running Welford observation
normalisation carried in the train state, entropy bonus and linear LR annealing against entropy
collapse.

Ported from crazyflow-gate-ppo (``gate_ppo/ppo.py``) unchanged except for the rollout metrics,
which now report the velocity command instead of motor RPM.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct
from jax import Array

from expert_data.config import Config

LOG_2PI = float(jnp.log(2.0 * jnp.pi))


@struct.dataclass
class RunningStats:
    mean: Array
    var: Array
    count: Array

    @classmethod
    def create(cls, dim: int) -> RunningStats:
        return cls(mean=jnp.zeros((dim,)), var=jnp.ones((dim,)), count=jnp.float32(1e-4))

    def update(self, x: Array) -> RunningStats:
        """Welford parallel merge with a batch ``x`` of shape ``(..., dim)``."""
        x = x.reshape(-1, x.shape[-1])
        batch_mean, batch_var = x.mean(0), x.var(0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m2 = (
            self.var * self.count
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / total
        )
        return RunningStats(mean=new_mean, var=m2 / total, count=total)

    def normalize(self, x: Array) -> Array:
        return jnp.clip((x - self.mean) / jnp.sqrt(self.var + 1e-8), -10.0, 10.0)


@struct.dataclass
class TrainState:
    params: Any
    opt_state: Any
    obs_stats: RunningStats
    n_updates: Array


@struct.dataclass
class Transition:
    obs: Array  # raw observation (normalised on the fly with the iteration's stats)
    action: Array
    log_prob: Array
    value: Array
    reward: Array
    terminated: Array
    truncated: Array
    value_final: Array  # V(final_obs), used to bootstrap truncated episodes
    info: dict[str, Array]


def gaussian_log_prob(x: Array, mean: Array, log_std: Array) -> Array:
    z = (x - mean) * jnp.exp(-log_std)
    return jnp.sum(-0.5 * z**2 - log_std - 0.5 * LOG_2PI, axis=-1)


def gaussian_entropy(log_std: Array) -> Array:
    return jnp.sum(log_std + 0.5 * (LOG_2PI + 1.0), axis=-1)


def make_optimizer(cfg: Config, n_iterations: int) -> optax.GradientTransformation:
    n_opt_steps = n_iterations * cfg.ppo.n_epochs * cfg.ppo.n_minibatches
    if cfg.ppo.anneal_lr:
        lr = optax.linear_schedule(cfg.ppo.lr, 0.0, n_opt_steps)
    else:
        lr = cfg.ppo.lr
    return optax.chain(optax.clip_by_global_norm(cfg.ppo.max_grad_norm), optax.adam(lr, eps=1e-5))


def make_train_iteration(env, network, cfg: Config, tx: optax.GradientTransformation):
    """Build the jitted ``train_iteration(runner_state, env_params) -> (runner_state, metrics)``."""
    n_envs, n_steps = cfg.env.n_envs, cfg.ppo.n_steps
    n_epochs, n_minibatches = cfg.ppo.n_epochs, cfg.ppo.n_minibatches
    batch_size = n_envs * n_steps
    if batch_size % n_minibatches != 0:
        raise ValueError("n_envs * n_steps must be divisible by n_minibatches")
    minibatch_size = batch_size // n_minibatches
    gamma, lam, clip_eps = cfg.ppo.gamma, cfg.ppo.gae_lambda, cfg.ppo.clip_eps

    def forward(params, stats: RunningStats, raw_obs: Array):
        obs = stats.normalize(raw_obs) if cfg.ppo.normalize_obs else raw_obs
        return network.apply(params, obs)

    def train_iteration(runner_state, env_params):
        train_state, env_state, last_obs, rng = runner_state

        # ---------------------------------------------------------------- rollout (lax.scan)
        def env_step(carry, _):
            train_state, env_state, last_obs, rng = carry
            rng, key = jax.random.split(rng)
            mean, log_std, value = forward(train_state.params, train_state.obs_stats, last_obs)
            action = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
            log_prob = gaussian_log_prob(action, mean, log_std)
            env_state, obs, reward, terminated, truncated, final_obs, info = env.step(
                env_state, action, env_params
            )
            _, _, value_final = forward(train_state.params, train_state.obs_stats, final_obs)
            transition = Transition(
                last_obs, action, log_prob, value, reward, terminated, truncated, value_final, info
            )
            return (train_state, env_state, obs, rng), transition

        (train_state, env_state, last_obs, rng), traj = jax.lax.scan(
            env_step, (train_state, env_state, last_obs, rng), None, n_steps
        )
        _, _, last_value = forward(train_state.params, train_state.obs_stats, last_obs)

        # -------------------------------------------------------------------------- GAE
        def gae_step(carry, tr: Transition):
            gae, next_value = carry
            done = tr.terminated | tr.truncated
            bootstrap = jnp.where(
                tr.terminated, 0.0, jnp.where(tr.truncated, tr.value_final, next_value)
            )
            delta = tr.reward + gamma * bootstrap - tr.value
            gae = delta + gamma * lam * (1.0 - done.astype(jnp.float32)) * gae
            return (gae, tr.value), gae

        _, advantages = jax.lax.scan(
            gae_step, (jnp.zeros((n_envs,)), last_value), traj, reverse=True
        )
        targets = advantages + traj.value

        # ---------------------------------------------------------------- PPO epochs
        batch = (traj.obs, traj.action, traj.log_prob, traj.value, advantages, targets)
        batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)

        def loss_fn(params, stats, mb):
            obs, action, old_log_prob, old_value, adv, target = mb
            mean, log_std, value = forward(params, stats, obs)
            log_prob = gaussian_log_prob(action, mean, log_std)
            log_ratio = log_prob - old_log_prob
            ratio = jnp.exp(log_ratio)
            if cfg.ppo.normalize_adv:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg_loss = jnp.maximum(-adv * ratio, -adv * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps))
            pg_loss = pg_loss.mean()
            if cfg.ppo.clip_vloss:
                value_clipped = old_value + jnp.clip(value - old_value, -clip_eps, clip_eps)
                v_loss = 0.5 * jnp.maximum((value - target) ** 2, (value_clipped - target) ** 2)
            else:
                v_loss = 0.5 * (value - target) ** 2
            v_loss = v_loss.mean()
            entropy = gaussian_entropy(log_std).mean()
            loss = pg_loss + cfg.ppo.vf_coef * v_loss - cfg.ppo.ent_coef * entropy
            aux = {
                "loss/total": loss,
                "loss/policy": pg_loss,
                "loss/value": v_loss,
                "loss/entropy": entropy,
                "loss/approx_kl": jnp.mean((ratio - 1.0) - log_ratio),
                "loss/clipfrac": jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32)),
            }
            return loss, aux

        def update_minibatch(train_state: TrainState, mb):
            (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                train_state.params, train_state.obs_stats, mb
            )
            updates, opt_state = tx.update(grads, train_state.opt_state, train_state.params)
            params = optax.apply_updates(train_state.params, updates)
            aux["loss/grad_norm"] = optax.global_norm(grads)
            return train_state.replace(params=params, opt_state=opt_state), aux

        def update_epoch(carry, _):
            train_state, rng = carry
            rng, key = jax.random.split(rng)
            perm = jax.random.permutation(key, batch_size)
            minibatches = jax.tree.map(
                lambda x: x[perm].reshape((n_minibatches, minibatch_size) + x.shape[1:]), batch
            )
            train_state, aux = jax.lax.scan(update_minibatch, train_state, minibatches)
            return (train_state, rng), aux

        (train_state, rng), loss_metrics = jax.lax.scan(
            update_epoch, (train_state, rng), None, n_epochs
        )
        if cfg.ppo.normalize_obs:
            train_state = train_state.replace(obs_stats=train_state.obs_stats.update(traj.obs))
        train_state = train_state.replace(n_updates=train_state.n_updates + 1)

        # ------------------------------------------------------------------ metrics
        done = (traj.terminated | traj.truncated).astype(jnp.float32)
        n_done = done.sum()

        def done_mean(x: Array) -> Array:
            return jnp.where(n_done > 0, (x * done).sum() / jnp.maximum(n_done, 1.0), jnp.nan)

        info = traj.info
        target_var = jnp.var(targets)
        metrics = {
            "episode/return": done_mean(info["episode_return"]),
            "episode/length": done_mean(info["episode_length"]),
            "episode/success": done_mean(info["success"]),
            "episode/gates_passed": done_mean(info["gates_passed"]),
            "episode/crash": done_mean(info["crashed"]),
            "episode/miss": done_mean(info["missed"]),
            "episode/timeout": done_mean(traj.truncated.astype(jnp.float32)),
            "episode/count": n_done,
            "rollout/reward_mean": traj.reward.mean(),
            "rollout/speed": info["speed"].mean(),
            "rollout/cmd_speed": jnp.linalg.norm(info["cmd"][..., :3], axis=-1).mean(),
            "rollout/cmd_yaw_rate": jnp.abs(info["cmd"][..., 3]).mean(),
            "rollout/action_saturation": (jnp.abs(traj.action) >= 1.0).mean(),
            "rollout/crash_per_step": info["crashed"].mean(),
            "policy/log_std": train_state.params["params"]["log_std"].mean(),
            "policy/value_mean": traj.value.mean(),
            "policy/explained_variance": 1.0 - jnp.var(targets - traj.value) / (target_var + 1e-8),
        }
        metrics.update(jax.tree.map(jnp.mean, loss_metrics))
        return (train_state, env_state, last_obs, rng), metrics

    return jax.jit(train_iteration)
