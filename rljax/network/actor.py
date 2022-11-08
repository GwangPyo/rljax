import haiku as hk
import jax.numpy as jnp
from jax import nn
from rljax.network.conditional_flow import FlowSequential, TanhFlowLayer
from rljax.network.base import MLP
from rljax.network.conv import DQNBody
import math
import numpy as np


class DeterministicPolicy(hk.Module):
    """
    Policy for DDPG and TD3.
    """

    def __init__(
        self,
        action_space,
        hidden_units=(256, 256),
        d2rl=False,
    ):
        super(DeterministicPolicy, self).__init__()
        self.action_space = action_space
        self.hidden_units = hidden_units
        self.d2rl = d2rl

    def __call__(self, x):
        return MLP(
            self.action_space.shape[0],
            self.hidden_units,
            hidden_activation=nn.relu,
            output_activation=jnp.tanh,
            d2rl=self.d2rl,
        )(x)


class StateDependentGaussianPolicy(hk.Module):
    """
    Policy for SAC.
    """

    def __init__(
        self,
        action_space,
        hidden_units=(256, 256),
        log_std_min=-20.0,
        log_std_max=2.0,
        clip_log_std=True,
        d2rl=False,
    ):
        super(StateDependentGaussianPolicy, self).__init__()
        self.action_space = action_space
        self.hidden_units = hidden_units
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.clip_log_std = clip_log_std
        self.d2rl = d2rl

    def __call__(self, x):
        x = MLP(
            2 * self.action_space.shape[0],
            self.hidden_units,
            hidden_activation=nn.relu,
            d2rl=self.d2rl,
        )(x)
        mean, log_std = jnp.split(x, 2, axis=1)
        if self.clip_log_std:
            log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        else:
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (jnp.tanh(log_std) + 1.0)
        return mean, log_std


class StateIndependentGaussianPolicy(hk.Module):
    """
    Policy for PPO.
    """

    def __init__(
        self,
        action_space,
        hidden_units=(64, 64),
    ):
        super(StateIndependentGaussianPolicy, self).__init__()
        self.action_space = action_space
        self.hidden_units = hidden_units

    def __call__(self, x):
        mean = MLP(
            self.action_space.shape[0],
            self.hidden_units,
            hidden_activation=jnp.tanh,
            output_scale=0.01,
        )(x)
        log_std = hk.get_parameter("log_std", (1, self.action_space.shape[0]), init=jnp.zeros)
        return mean, log_std


class CategoricalPolicy(hk.Module):
    """
    Policy for SAC-Discrete.
    """

    def __init__(
        self,
        action_space,
        hidden_units=(512,),
    ):
        super(CategoricalPolicy, self).__init__()
        self.action_space = action_space
        self.hidden_units = hidden_units

    def __call__(self, x):
        if len(x.shape) == 4:
            x = DQNBody()(x)
        x = MLP(
            self.action_space.n,
            self.hidden_units,
            hidden_activation=nn.relu,
            output_scale=0.01,
        )(x)
        pi_s = nn.softmax(x, axis=1)
        log_pi_s = jnp.log(pi_s + (pi_s == 0.0) * 1e-8)
        return pi_s, log_pi_s


class RC_DSACPolicy(hk.Module):
    """
    Policy for RC-DSAC.
    """

    def __init__(
        self,
        action_space,
        hidden_units=(256, 256),
        log_std_min=-20.0,
        log_std_max=2.0,
        clip_log_std=True,
        d2rl=False,
        num_cosines: int = 64
    ):
        super(RC_DSACPolicy, self).__init__()
        self.action_space = action_space
        self.hidden_units = hidden_units
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.clip_log_std = clip_log_std
        self.d2rl = d2rl

        self.num_cosines = num_cosines
        self.pi = math.pi * jnp.arange(1, num_cosines + 1, dtype=jnp.float32).reshape(1, 1, num_cosines)

    def __call__(self, x, betas: jnp.ndarray):
        feature = MLP(
            128,
            self.hidden_units,
            hidden_activation=jnp.tanh,
            output_scale=0.01,
        )(x)

        def cosine_embedding(x, betas):
            feature_dim = x.shape[1]
            betas = betas[:, [0]]

            # Calculate features.
            cosine = jnp.cos(jnp.expand_dims(betas, 2) * self.pi).reshape(-1, self.num_cosines)
            cosine_feature = nn.relu(hk.Linear(feature_dim)(cosine)).reshape(-1, feature_dim)

            x = nn.relu((x * cosine_feature))
            # Apply quantile network.
            output = hk.Linear(self.action_space.shape[0] * 2)(MLP(128, self.hidden_units, hidden_activation=jnp.tanh, output_scale=0.01,)(x))
            return output
        outputs = cosine_embedding(feature, betas)
        mean, log_std = jnp.split(outputs, 2, axis=1)
        if self.clip_log_std:
            log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        else:
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (jnp.tanh(log_std) + 1.0)
        return mean, log_std


class FlowPolicy(hk.Module):
    def __init__(self,
                 action_space,
                 net_arch=(16, 16, 16, 16),
                 name='flow_policy',
                 d2rl = False,
                 ):
        super(FlowPolicy, self).__init__(name=name)
        self.action_space = action_space
        self.flow = FlowSequential(
            dim=action_space.shape[-1],
            net_arch=net_arch,
            outlayer=TanhFlowLayer,
        )
        self.d2rl = d2rl

    def __call__(self, observations, noise):

        init, (_, forward, inverse, _) = self.flow(observations, noise)
        return forward(noise, feature=observations)


class FractionAwareFlowPolicy(FlowPolicy):
    def __call__(self, observations, noise, fractions):
        hk.MultiHeadAttention()
        init, (_, forward, inverse, _) = self.flow(observations, noise)
        return forward(noise, feature=observations)



