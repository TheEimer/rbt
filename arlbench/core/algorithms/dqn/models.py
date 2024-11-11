"""Q-Networks for DQN."""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return x

class CNNQ(nn.Module):
    """A CNN-based Q-Network for DQN."""

    action_dim: int
    activation: str = "tanh"
    hidden_size: int = 512
    discrete: bool = True

    def _record_activations(self, x, layer):
        if self.is_initializing():
            name = "/".join(layer.scope.path)
            self.layer_names.append(name)
        return IdentityLayer(name=f"{layer.name}_act")(x)

    def setup(self):
        """Initializes the CNN Q-Network."""
        if self.activation == "tanh":
            self.activation_func = nn.tanh
        elif self.activation == "relu":
            self.activation_func = nn.relu
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")

        self.conv1 = nn.Conv(
            features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.conv2 = nn.Conv(
            features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.conv3 = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.dense = nn.Dense(
            features=self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.out_layer = nn.Dense(
            self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )

    def __call__(self, x):
        """Applies the CNN to the input."""
        x = x / 255.0
        x = jnp.transpose(x, (0, 2, 3, 1))
        q = self.conv1(x)
        q = self.activation_func(q)
        q = self._record_activations(q, self.conv1)
        q = self.conv2(q)
        q = self.activation_func(q)
        q = self._record_activations(q, self.conv2)
        q = self.conv3(q)
        q = self.activation_func(q)
        q = self._record_activations(q, self.conv3)
        q = q.reshape((q.shape[0], -1))  # flatten
        q = self.dense(q)
        q = self.activation_func(q)
        q = self._record_activations(q, self.dense)
        q = self.out_layer(q)
        return self._record_activations(q, self.out_layer)


class MLPQ(nn.Module):
    """An MLP-based Q-Network for DQN."""

    action_dim: int
    activation: str = "tanh"
    hidden_size: int = 64
    discrete: bool = True

    def _record_activations(self, x, layer):
        return IdentityLayer(name=f"{layer.name}_act")(x)

    def setup(self):
        """Initializes the MLP Q-Network."""
        if self.activation == "tanh":
            self.activation_func = nn.tanh
        elif self.activation == "relu":
            self.activation_func = nn.relu
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")

        self.dense0 = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.dense1 = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.out_layer = nn.Dense(
            self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )

    @nn.compact
    def __call__(self, x):
        """Applies the MLP to the input."""
        q = self.dense0(x)
        q = self.activation_func(q)
        q = self._record_activations(q, self.dense0)
        q = self.dense1(q)
        q = self.activation_func(q)
        q = self._record_activations(q, self.dense1)
        q = self.out_layer(q)
        return self._record_activations(q, self.out_layer)
