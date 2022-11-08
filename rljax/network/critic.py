import math

import haiku as hk
import jax.numpy as jnp
import jax.random
import numpy as np
from jax import nn

from rljax.network.base import MLP
from rljax.network.conv import DQNBody
from rljax.network.misc import MonotoneMLP, MonotoneLinear


class ContinuousVFunction(hk.Module):
    """
    Critic for PPO.
    """

    def __init__(
        self,
        num_critics=1,
        hidden_units=(64, 64),
    ):
        super(ContinuousVFunction, self).__init__()
        self.num_critics = num_critics
        self.hidden_units = hidden_units

    def __call__(self, x):
        def _fn(x):
            return MLP(
                1,
                self.hidden_units,
                hidden_activation=jnp.tanh,
            )(x)

        if self.num_critics == 1:
            return _fn(x)
        return [_fn(x) for _ in range(self.num_critics)]


class ContinuousQFunction(hk.Module):
    """
    Critic for DDPG, TD3 and SAC.
    """

    def __init__(
        self,
        num_critics=2,
        hidden_units=(256, 256),
        d2rl=False,
    ):
        super(ContinuousQFunction, self).__init__()
        self.num_critics = num_critics
        self.hidden_units = hidden_units
        self.d2rl = d2rl

    def __call__(self, s, a):
        def _fn(x):
            return MLP(
                1,
                self.hidden_units,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
                d2rl=self.d2rl,
            )(x)

        x = jnp.concatenate([s, a], axis=1)
        # Return list even if num_critics == 1 for simple implementation.
        return [_fn(x) for _ in range(self.num_critics)]


class ContinuousQuantileFunction(hk.Module):
    """
    Critic for TQC.
    """

    def __init__(
        self,
        num_critics=5,
        hidden_units=(512, 512, 512),
        num_quantiles=25,
        d2rl=False,
    ):
        super(ContinuousQuantileFunction, self).__init__()
        self.num_critics = num_critics
        self.hidden_units = hidden_units
        self.num_quantiles = num_quantiles
        self.d2rl = d2rl

    def __call__(self, s, a):
        def _fn(x):
            return MLP(
                self.num_quantiles,
                self.hidden_units,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
                d2rl=self.d2rl,
            )(x)

        x = jnp.concatenate([s, a], axis=1)
        return [_fn(x) for _ in range(self.num_critics)]


class DiscreteQFunction(hk.Module):
    """
    Critic for DQN and SAC-Discrete.
    """

    def __init__(
        self,
        action_space,
        num_critics=1,
        hidden_units=(512,),
        dueling_net=False,
    ):
        super(DiscreteQFunction, self).__init__()
        self.action_space = action_space
        self.num_critics = num_critics
        self.hidden_units = hidden_units
        self.dueling_net = dueling_net

    def __call__(self, x):
        def _fn(x):
            if len(x.shape) == 4:
                x = DQNBody()(x)
            output = MLP(
                self.action_space.n,
                self.hidden_units,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
            )(x)
            if self.dueling_net:
                baseline = MLP(
                    1,
                    self.hidden_units,
                    hidden_activation=nn.relu,
                    hidden_scale=np.sqrt(2),
                )(x)
                return output + baseline - output.mean(axis=1, keepdims=True)
            else:
                return output

        if self.num_critics == 1:
            return _fn(x)
        return [_fn(x) for _ in range(self.num_critics)]


class DiscreteQuantileFunction(hk.Module):
    """
    Critic for QR-DQN.
    """

    def __init__(
        self,
        action_space,
        num_critics=1,
        num_quantiles=200,
        hidden_units=(512,),
        dueling_net=True,
    ):
        super(DiscreteQuantileFunction, self).__init__()
        self.action_space = action_space
        self.num_critics = num_critics
        self.num_quantiles = num_quantiles
        self.hidden_units = hidden_units
        self.dueling_net = dueling_net

    def __call__(self, x):
        def _fn(x):
            if len(x.shape) == 4:
                x = DQNBody()(x)
            output = MLP(
                self.action_space.n * self.num_quantiles,
                self.hidden_units,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
            )(x)
            output = output.reshape(-1, self.num_quantiles, self.action_space.n)
            if self.dueling_net:
                baseline = MLP(
                    self.num_quantiles,
                    self.hidden_units,
                    hidden_activation=nn.relu,
                    hidden_scale=np.sqrt(2),
                )(x)
                baseline = baseline.reshape(-1, self.num_quantiles, 1)
                return output + baseline - output.mean(axis=2, keepdims=True)
            else:
                return output

        if self.num_critics == 1:
            return _fn(x)
        return [_fn(x) for _ in range(self.num_critics)]


class DiscreteImplicitQuantileFunction(hk.Module):
    """
    Critic for IQN and FQF.
    """

    def __init__(
        self,
        action_space,
        num_critics=1,
        num_cosines=64,
        hidden_units=(512,),
        dueling_net=True,
    ):
        super(DiscreteImplicitQuantileFunction, self).__init__()
        self.action_space = action_space
        self.num_critics = num_critics
        self.num_cosines = num_cosines
        self.hidden_units = hidden_units
        self.dueling_net = dueling_net
        self.pi = math.pi * jnp.arange(1, num_cosines + 1, dtype=jnp.float32).reshape(1, 1, num_cosines)

    def __call__(self, x, cum_p):
        def _fn(x, cum_p):
            if len(x.shape) == 4:
                x = DQNBody()(x)

            # NOTE: For IQN and FQF, number of quantiles are variable.
            feature_dim = x.shape[1]
            num_quantiles = cum_p.shape[1]
            # Calculate features.
            cosine = jnp.cos(jnp.expand_dims(cum_p, 2) * self.pi).reshape(-1, self.num_cosines)
            cosine_feature = nn.relu(hk.Linear(feature_dim)(cosine)).reshape(-1, num_quantiles, feature_dim)
            x = (x.reshape(-1, 1, feature_dim) * cosine_feature).reshape(-1, feature_dim)
            # Apply quantile network.
            output = MLP(
                self.action_space.n,
                self.hidden_units,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
            )(x)
            output = output.reshape(-1, num_quantiles, self.action_space.n)
            if self.dueling_net:
                baseline = MLP(
                    1,
                    self.hidden_units,
                    hidden_activation=nn.relu,
                    hidden_scale=np.sqrt(2),
                )(x)
                baseline = baseline.reshape(-1, num_quantiles, 1)
                return output + baseline - output.mean(axis=2, keepdims=True)
            else:
                return output

        if self.num_critics == 1:
            return _fn(x, cum_p)
        return [_fn(x, cum_p) for _ in range(self.num_critics)]


class ContinuousImplicitQuantileFunction(hk.Module):
    """
    Critic for TQC.
    """

    def __init__(
        self,
        num_critics=2,
        hidden_units_features=(256, 256),
        hidden_units_output=(256, 256),
        num_cosines=64,
        d2rl=False,
    ):
        super(ContinuousImplicitQuantileFunction, self).__init__()
        self.num_critics = num_critics
        self.num_cosines = num_cosines
        self.hidden_units_features = hidden_units_features
        self.hidden_units_outputs = hidden_units_output
        self.d2rl = d2rl
        self.pi = math.pi * jnp.arange(1, num_cosines + 1, dtype=jnp.float32).reshape(1, 1, num_cosines)

    def __call__(self, s, a, cum_p):
        def feature_extractor(x):
            return MLP(
                output_dim=self.hidden_units_features[-1],
                hidden_units=self.hidden_units_features,
                hidden_activation=nn.relu,
                d2rl=self.d2rl,
                hidden_scale=np.sqrt(2),
            )(x)

        def cosine_embedding(x, cum_p):
            feature_dim = x.shape[1]
            num_quantiles = cum_p.shape[1]
            # Calculate features.
            cosine = jnp.cos(jnp.expand_dims(cum_p, 2) * self.pi).reshape(-1, self.num_cosines)
            cosine_feature = nn.relu(hk.Linear(feature_dim)(cosine)).reshape(-1, num_quantiles, feature_dim)
            x = (x.reshape(-1, 1, feature_dim) * cosine_feature).reshape(-1, feature_dim)
            # Apply quantile network.
            output = MLP(
                1,
                self.hidden_units_outputs,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
            )(x)
            output = output.reshape(-1, num_quantiles,)
            return output

        x = jnp.concatenate([s, a], axis=-1)
        extracted_features = [feature_extractor(x) for _ in range(self.num_critics)]
        quantiles = [cosine_embedding(z, cum_p) for z in extracted_features]
        return quantiles


class ContinuousMonotoneImplicitQuantileFunction(hk.Module):
    """
    Critic for TQC.
    """

    def __init__(
        self,
        num_critics=2,
        hidden_units_features=(256, 256),
        hidden_units_output=(256, 256),
        num_cosines=16,
        d2rl=False,
    ):
        super(ContinuousMonotoneImplicitQuantileFunction, self).__init__()
        self.num_critics = num_critics
        self.num_cosines = num_cosines
        self.hidden_units_features = hidden_units_features
        self.hidden_units_outputs = hidden_units_output
        self.d2rl = d2rl
        # (1/N pi , 2/N pi, ... , pi)
        # self.pi = math.pi * jnp.arange(1, num_cosines + 1, dtype=jnp.float32).reshape(1, 1, num_cosines)
        arange = jnp.arange(1, (num_cosines // 2), dtype=jnp.float32)
        reci_arange = jnp.arange(1/len(arange), 1, 1/len(arange), dtype=jnp.float32 )
        self.power = jnp.concatenate([arange, reci_arange])
        self.num_cosines = len(self.power)

    def __call__(self, s, a, cum_p):
        def feature_extractor(x):
            return MLP(
                output_dim=self.hidden_units_features[-1],
                hidden_units=self.hidden_units_features,
                hidden_activation=nn.relu,
                d2rl=self.d2rl,
                hidden_scale=np.sqrt(2),
            )(x)

        def cosine_embedding(x, cum_p):
            feature_dim = x.shape[1]
            num_quantiles = cum_p.shape[1]
            # Calculate features.
            # concave
            expanded = jnp.expand_dims(cum_p, 1)
            expanded = jnp.repeat(expanded, axis=1, repeats=self.num_cosines)
            power_embedding = jnp.power(expanded, self.power[None, :, None]).transpose(0, 2, 1)

            tau_feature  = MonotoneLinear(feature_dim)(power_embedding)
            scale_bias = MLP(output_dim=2,
                                hidden_units=self.hidden_units_outputs,
                                hidden_activation=nn.relu,
                                hidden_scale=np.sqrt(2)
                                )(x)
            scale = nn.softplus(scale_bias[..., [0]])
            bias = scale_bias[..., [1]]
            x = (jax.nn.relu(x.reshape(-1, 1, feature_dim)) * tau_feature.reshape(-1, num_quantiles, feature_dim)).reshape(-1, feature_dim)
            # Apply quantile network.
            # x shape (batch, N_quantiles, feature_dim)
            output = MonotoneMLP(
                1,
                self.hidden_units_outputs,
                hidden_activation=nn.tanh,
            )(x)
            # output_shape = (batch, N_quantiles, 1)

            output = output.reshape(-1, num_quantiles,)
            # scale shape = (batch, 1)
            # bias shape = (batch, 1)
            return scale * output + bias

        x = jnp.concatenate([s, a], axis=-1)
        extracted_features = [feature_extractor(x) for _ in range(self.num_critics)]
        quantiles = [cosine_embedding(z, cum_p) for z in extracted_features]

        return quantiles


class RCDSACCritic(hk.Module):
    """
    Critic for TQC.
    """

    def __init__(
            self,
            num_critics=2,
            hidden_units_features=(256, 256),
            num_cosines=64,
            d2rl=False,
    ):
        super(RCDSACCritic, self).__init__()
        self.num_critics = num_critics
        self.num_cosines = num_cosines
        self.hidden_units_features = hidden_units_features
        self.d2rl = d2rl
        self.pi = math.pi * jnp.arange(1, num_cosines + 1, dtype=jnp.float32).reshape(1, 1, num_cosines)

    def __call__(self, s, a, cum_p, betas):
        num_quantiles = cum_p.shape[-1]

        def feature_extractor(x):
            return MLP(
                output_dim=self.hidden_units_features[-1],
                hidden_units=self.hidden_units_features,
                hidden_activation=nn.relu,
                d2rl=self.d2rl,
                hidden_scale=np.sqrt(2),
            )(x)

        def cosines(cum_p):
            # Calculate features.
            cosine = jnp.cos(jnp.expand_dims(cum_p, 2) * self.pi).reshape(cum_p.shape[0], -1, self.num_cosines)
            return cosine

        def cosine_embeddings(cosines_cump, cosines_beta):
            cosines_beta = jnp.repeat(cosines_beta, axis=1, repeats=cosines_cump.shape[1])
            cats = jnp.concatenate((cosines_cump, cosines_beta), axis=-1)
            return nn.relu(MLP(output_dim=128, hidden_units=(128, 128))(cats)).reshape(-1, num_quantiles, 128)

        observation_features = [feature_extractor(s) for _ in range(self.num_critics)]
        action_features = [nn.relu(hk.Linear(output_size=self.hidden_units_features[-1])(a)) for _ in
                           range(self.num_critics)]
        obs_action_features = [hk.Linear(output_size=128)(jnp.concatenate([obs, action], axis=-1))
                               for obs, action in zip(observation_features, action_features)]

        obs_action_features = [z.reshape(-1, 1, 128) for z in obs_action_features]

        cosines_cump = cosines(cum_p)
        cosines_beta = cosines(betas)
        cosine_features = [cosine_embeddings(cosines_cump, cosines_beta) for _ in range(self.num_cosines)]
        final_features = [z * phi for z, phi in zip(obs_action_features, cosine_features)]

        quantiles = [(MLP(1, self.hidden_units_features, nn.relu, hidden_scale=np.sqrt(2))(z)).reshape(-1, num_quantiles) for z in final_features]
        return quantiles


if __name__ == '__main__':
    def fn_critic(s, a, cum_p):
        return ContinuousMonotoneImplicitQuantileFunction(
        )(s, a, cum_p)
    critic = hk.without_apply_rng(hk.transform(fn_critic))
    cum_p = jnp.arange(0, 1, 1 / 32)[None]
    critic_param = critic.init(
        rng=jax.random.PRNGKey(21),
        s=jnp.ones((1, 4)),
        a=jnp.ones((1, 2)),
        cum_p=cum_p
    )
    arr = critic.apply(critic_param, jnp.ones((1, 4)), jnp.ones((1, 2)), cum_p)
    print(arr[0])
    print(arr[1])