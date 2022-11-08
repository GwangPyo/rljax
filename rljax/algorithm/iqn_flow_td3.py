from functools import partial
from typing import List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from rljax.network import ContinuousImplicitQuantileFunction, ContinuousMonotoneImplicitQuantileFunction
from rljax.network.actor import FlowPolicy
from rljax.util import quantile_loss, fake_state, fake_action
from rljax.algorithm import IQNTD3


class IQNFlowTD3(IQNTD3):
    name = 'IQNFlowTD3'

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm=None,
        gamma=0.99,
        nstep=1,
        num_critics=2,
        buffer_size=10 ** 6,
        use_per=False,
        batch_size=256,
        start_steps=10000,
        update_interval=1,
        tau=5e-3,
        fn_actor=None,
        fn_critic=None,
        lr_actor=1e-3,
        lr_critic=1e-3,
        units_actor=(64, 64, 32, 32),
        units_critic_feature=(256, 256),
        units_critic_outputs=(64, 32),
        d2rl=False,
        num_quantiles=25,
        num_quantiles_to_drop=2,
        std=0.1,
        std_target=0.2,
        clip_noise=0.5,
        update_interval_policy=2,
        monotone=False
    ):
        if d2rl:
            self.name += "-D2RL"

        if fn_critic is None:
            if not monotone:
                def fn_critic(s, a, cum_p):
                    return ContinuousImplicitQuantileFunction(
                        num_critics=num_critics,
                        hidden_units_features=units_critic_feature,
                        hidden_units_output=units_critic_outputs,
                        d2rl=d2rl,
                    )(s, a, cum_p)
            else:
                def fn_critic(s, a, cum_p):
                    return ContinuousMonotoneImplicitQuantileFunction(
                        num_critics=num_critics,
                        hidden_units_features=units_critic_feature,
                        hidden_units_output=units_critic_outputs,
                        d2rl=d2rl,
                    )(s, a, cum_p)

        if fn_actor is None:
            def fn_actor(s, z):
                return FlowPolicy(
                    action_space=action_space,
                    d2rl=d2rl,
                    net_arch=units_actor
                )(s, z)

        if not hasattr(self, "fake_args_actor"):
            self.fake_args_actor = (fake_state(state_space), fake_action(action_space))
        if not hasattr(self, "fake_args"):
            self.fake_args_critic = (fake_state(state_space), fake_action(action_space), np.empty((1, num_quantiles), dtype=np.float32))
        self.action_shape = action_space.shape

        super(IQNTD3, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            nstep=nstep,
            num_critics=num_critics,
            buffer_size=buffer_size,
            use_per=use_per,
            batch_size=batch_size,
            start_steps=start_steps,
            update_interval=update_interval,
            tau=tau,
            fn_actor=fn_actor,
            fn_critic=fn_critic,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            std=std,
            std_target=std_target,
            clip_noise=clip_noise,
            update_interval_policy=update_interval_policy,
        )
        self.num_quantiles = num_quantiles
        self.num_quantiles_target = (num_quantiles - num_quantiles_to_drop) * num_critics

        self.use_key_critic = True
        self.use_key_actor = True

    def select_action(self, state):
        normal = jax.random.normal(key=next(self.rng), shape=(state.shape[0], ) + self.action_shape)
        action = self._select_action(self.params_actor, state[None, ...], normal)
        return np.array(action[0])

    @partial(jax.jit, static_argnums=0)
    def _select_action(
            self,
            params_actor: hk.Params,
            state: np.ndarray,
            normal: jnp.ndarray
    ) -> jnp.ndarray:
        mean, _ = self.actor.apply(params_actor, state, normal)
        return mean

    @partial(jax.jit, static_argnums=0)
    def _sample_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> jnp.ndarray:

        normal = jax.random.normal(key=key, shape=(state.shape[0], ) + self.action_shape)
        action, _ = self.actor.apply(params_actor, state, normal)
        return action

    @partial(jax.jit, static_argnums=0)
    def _calculate_value_list(
        self,
        params_critic: hk.Params,
        state: jnp.ndarray,
        action: jnp.ndarray,
        cum_p: jnp.ndarray
    ) -> List[jnp.ndarray]:
        return self.critic.apply(params_critic, state, action, cum_p)

    @partial(jax.jit, static_argnums=0)
    def _calculate_value(
        self,
        params_critic: hk.Params,
        state: jnp.ndarray,
        action: jnp.ndarray,
        cum_p: jnp.ndarray
    ) -> jnp.ndarray:
        return jnp.concatenate(self._calculate_value_list(params_critic, state, action, cum_p), axis=1)

    @partial(jax.jit, static_argnums=0)
    def _loss_critic(
            self,
            params_critic: hk.Params,
            params_critic_target: hk.Params,
            params_actor_target: hk.Params,
            state: jnp.ndarray,
            action: jnp.ndarray,
            reward: jnp.ndarray,
            done: jnp.ndarray,
            next_state: jnp.ndarray,
            weight: jnp.ndarray or List[jnp.ndarray],
            *args,
            **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        next_action = self._sample_action(params_actor_target, next_state, *args, **kwargs)
        random_numbers = jax.random.uniform(shape=(2, next_action.shape[0], self.num_quantiles),
                                   dtype=jnp.float32, *args, **kwargs)
        cum_p = random_numbers[0]
        cum_p_prime = random_numbers[1]

        target = self._calculate_target(params_critic_target, reward, done, next_state, next_action,
                                        cum_p_prime)
        q_list = self._calculate_value_list(params_critic, state, action, cum_p)
        return self._calculate_loss_critic_and_abs_td(q_list, target, weight, cum_p)

    @partial(jax.jit, static_argnums=0)
    def _calculate_target(
        self,
        params_critic_target: hk.Params,
        reward: jnp.ndarray,
        done: jnp.ndarray,
        next_state: jnp.ndarray,
        next_action: jnp.ndarray,
        cum_p_prime: jnp.ndarray
    ) -> jnp.ndarray:
        next_quantile = self._calculate_value(params_critic_target, next_state, next_action, cum_p_prime)
        next_quantile = jnp.sort(next_quantile, axis=-1)[:, : self.num_quantiles_target]
        return jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_quantile)

    @partial(jax.jit, static_argnums=0)
    def _calculate_loss_critic_and_abs_td(
        self,
        quantile_list: List[jnp.ndarray],
        target: jnp.ndarray,
        weight: jnp.ndarray,
        cum_p: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        def _loss(x: jnp.ndarray):
            return quantile_loss(target[:, None, :] - x[:, :, None], cum_p, weight, 'huber')

        def scan_quantile_huber_loss(total, x_i):
            return total + _loss(x_i), total

        quantiles_array = jnp.stack(quantile_list, axis=0)
        loss_critic, abs_td = jax.lax.scan(scan_quantile_huber_loss, init=0., xs=quantiles_array)
        loss_critic /= self.num_critics * self.num_quantiles
        return loss_critic, jax.lax.stop_gradient(abs_td[0])

    @partial(jax.jit, static_argnums=0)
    def _explore(
            self,
            params_actor: hk.Params,
            state: np.ndarray,
            key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        normal = jax.random.normal(shape=(state.shape[0],) + self.action_shape, key=key)
        action, _ = self.actor.apply(params_actor, state, normal)
        return action

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        state: jnp.ndarray,
        *args,
        **kwargs,
    ) -> Tuple[jnp.ndarray, None]:
        normal = jax.random.normal(shape=(state.shape[0],) + self.action_shape, *args, **kwargs)
        action, _ = self.actor.apply(params_actor, state, normal)
        cum_p = jax.random.uniform(shape=(action.shape[0], self.num_quantiles),
                                   dtype=action.dtype,
                                   *args, **kwargs)
        mean_q = self._calculate_value(params_critic, state, action, cum_p).mean()
        return -mean_q, None
