"""Separate actor / critic MLPs (Flax linen), following the vault's settled architecture.

2x256 tanh, orthogonal init with sqrt(2) gain on hidden layers, 0.01 on the policy head and 1.0 on
the value head (Andrychowicz et al., 2021), state-independent ``log_std``. Ported unchanged from
crazyflow-gate-ppo (``gate_ppo/networks.py``).
"""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp
from jax import Array

LOG_STD_MIN, LOG_STD_MAX = -5.0, 1.0


class ActorCritic(nn.Module):
    action_dim: int
    hidden: Sequence[int] = (256, 256)
    log_std_init: float = -0.5

    @nn.compact
    def __call__(self, obs: Array) -> tuple[Array, Array, Array]:
        """Returns ``(mean, log_std, value)`` with shapes ``(..., A)``, ``(A,)``, ``(...)``."""
        h = obs
        for i, width in enumerate(self.hidden):
            h = nn.Dense(
                width,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=nn.initializers.zeros,
                name=f"actor_{i}",
            )(h)
            h = nn.tanh(h)
        mean = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros,
            name="actor_mean",
        )(h)
        log_std = self.param(
            "log_std", nn.initializers.constant(self.log_std_init), (self.action_dim,)
        )
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

        v = obs
        for i, width in enumerate(self.hidden):
            v = nn.Dense(
                width,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=nn.initializers.zeros,
                name=f"critic_{i}",
            )(v)
            v = nn.tanh(v)
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
            name="critic_value",
        )(v)
        return mean, log_std, value[..., 0]
