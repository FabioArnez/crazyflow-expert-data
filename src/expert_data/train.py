"""Hydra entry point: ``expert-train`` (or ``python -m expert_data.train``).

Python only dispatches one compiled update per iteration, logs the metrics it returns, drives the
yaw-range curriculum, and writes checkpoints. Everything else runs inside ``jax.jit``. Ported from
crazyflow-gate-ppo (``gate_ppo/train.py``).
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # share the GPU with PyTorch later

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402

import hydra  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from hydra.core.config_store import ConfigStore  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from expert_data.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from expert_data.config import Config  # noqa: E402
from expert_data.env import EnvParams, GateVelocityEnv  # noqa: E402
from expert_data.networks import ActorCritic  # noqa: E402
from expert_data.ppo import (  # noqa: E402
    RunningStats,
    TrainState,
    make_optimizer,
    make_train_iteration,
)

cs = ConfigStore.instance()
cs.store(name="base_schema", node=Config)
# Absolute path so the console script works from any cwd (relative paths would be resolved
# package-style by Hydra and require configs/ to be a Python package).
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")

CONSOLE_KEYS = (
    ("episode/return", "ret", "{:7.2f}"),
    ("episode/length", "len", "{:5.0f}"),
    ("episode/success", "succ", "{:.2f}"),
    ("episode/gates_passed", "gates", "{:.2f}"),
    ("episode/crash", "crash", "{:.2f}"),
    ("episode/miss", "miss", "{:.2f}"),
    ("episode/timeout", "tout", "{:.2f}"),
    ("rollout/speed", "v", "{:.2f}"),
    ("rollout/action_saturation", "sat", "{:.3f}"),
    ("loss/entropy", "ent", "{:.2f}"),
    ("loss/approx_kl", "kl", "{:.4f}"),
    ("policy/log_std", "logstd", "{:.2f}"),
)


def build(cfg: Config):
    """Construct env, network, optimiser and the initial runner state."""
    env = GateVelocityEnv(cfg.env, device=cfg.run.device)
    network = ActorCritic(env.action_dim, tuple(cfg.net.hidden), cfg.net.log_std_init)
    iteration_steps = cfg.env.n_envs * cfg.ppo.n_steps
    n_iterations = max(1, math.ceil(cfg.ppo.total_env_steps / iteration_steps))
    tx = make_optimizer(cfg, n_iterations)

    key = jax.random.key(cfg.run.seed)
    key, k_net, k_env, k_run = jax.random.split(key, 4)
    params = network.init(k_net, jnp.zeros((1, env.obs_dim)))
    train_state = TrainState(
        params=params,
        opt_state=tx.init(params),
        obs_stats=RunningStats.create(env.obs_dim),
        n_updates=jnp.int32(0),
    )
    if cfg.run.init_from:
        ckpt = load_checkpoint(cfg.run.init_from)
        params = jax.tree.map(jnp.asarray, ckpt["params"])
        stats = RunningStats(
            mean=jnp.asarray(ckpt["obs_stats"]["mean"]),
            var=jnp.asarray(ckpt["obs_stats"]["var"]),
            count=jnp.float32(ckpt["obs_stats"]["count"]),
        )
        train_state = train_state.replace(params=params, opt_state=tx.init(params), obs_stats=stats)
        print(f"warm-started params + obs stats from {cfg.run.init_from}")

    train_iteration = make_train_iteration(env, network, cfg, tx)
    return env, network, train_state, train_iteration, n_iterations, k_env, k_run


def initial_yaw_range(cfg: Config) -> float:
    if cfg.curriculum.enabled:
        return min(cfg.curriculum.yaw_range_init, cfg.env.yaw_range)
    return cfg.env.yaw_range


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> float:
    config: Config = OmegaConf.to_object(cfg)
    if config.run.device == "cpu":
        jax.config.update("jax_platforms", "cpu")
    if config.run.jax_cache:
        from crazyflow.utils import enable_cache

        enable_cache()
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(cfg))
    print(f"jax backend: {jax.default_backend()} | output dir: {out_dir}")

    env, _, train_state, train_iteration, n_iterations, k_env, k_run = build(config)
    iteration_steps = config.env.n_envs * config.ppo.n_steps
    print(
        f"hover rpm {env.hover_rpm:.0f} | grid {config.env.sim_freq}/{config.env.state_freq}/"
        f"{config.env.decision_freq} Hz | {config.env.n_envs} envs x {config.ppo.n_steps} steps = "
        f"{iteration_steps} samples/update | {n_iterations} updates | "
        f"{n_iterations * iteration_steps / 1e6:.1f}M env steps"
    )

    yaw_range = initial_yaw_range(config)
    env_params = EnvParams.from_config(config.env, yaw_range=yaw_range)
    env_state = env.init_state(k_env)
    env_state, obs = env.reset(env_state, env_params)
    runner_state = (train_state, env_state, obs, k_run)

    wandb_run = None
    if config.run.wandb:
        import wandb

        wandb_run = wandb.init(
            project=config.run.wandb_project, name=config.run.name, config=cfg_dict
        )

    recent_success: deque[float] = deque(maxlen=config.curriculum.window)
    metrics_file = open(out_dir / "metrics.jsonl", "w")
    t_start = time.time()
    last_success = float("nan")
    for it in range(1, n_iterations + 1):
        t0 = time.time()
        runner_state, metrics = train_iteration(runner_state, env_params)
        metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
        dt = time.time() - t0
        env_steps = it * iteration_steps
        metrics.update(
            {
                "iteration": it,
                "env_steps": env_steps,
                "sps": iteration_steps / dt,
                "time": time.time() - t_start,
                "curriculum/yaw_range": yaw_range,
            }
        )
        metrics_file.write(json.dumps(metrics) + "\n")
        metrics_file.flush()
        if wandb_run is not None:
            wandb_run.log(metrics, step=env_steps)

        if not math.isnan(metrics["episode/success"]):
            last_success = metrics["episode/success"]
            recent_success.append(last_success)
        if (
            config.curriculum.enabled
            and len(recent_success) == config.curriculum.window
            and np.mean(recent_success) >= config.curriculum.success_threshold
            and yaw_range < config.env.yaw_range
        ):
            yaw_range = min(yaw_range + config.curriculum.yaw_range_step, config.env.yaw_range)
            env_params = env_params.replace(yaw_range=jnp.float32(yaw_range))
            recent_success.clear()
            print(f"[curriculum] iteration {it}: yaw_range -> {yaw_range:.2f} rad")

        if it % config.run.log_interval == 0 or it == 1 or it == n_iterations:
            parts = [f"it {it:5d}", f"{env_steps / 1e6:6.2f}M", f"{metrics['sps'] / 1e3:6.1f}k sps"]
            for key, label, fmt in CONSOLE_KEYS:
                parts.append(f"{label} {fmt.format(metrics[key])}")
            parts.append(f"yaw {yaw_range:.2f}")
            print(" | ".join(parts), flush=True)
        if it % config.run.ckpt_interval == 0 or it == n_iterations:
            save_checkpoint(
                out_dir / "checkpoint.pkl",
                runner_state[0],
                cfg_dict,
                extra={"iteration": it, "env_steps": env_steps, "yaw_range": yaw_range},
            )
    metrics_file.close()
    if wandb_run is not None:
        wandb_run.finish()
    print(
        f"done in {(time.time() - t_start) / 60:.1f} min; checkpoint: {out_dir / 'checkpoint.pkl'}"
    )
    return last_success


if __name__ == "__main__":
    main()
