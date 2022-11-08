from functools import partial
from typing import List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from rljax.network import ContinuousImplicitQuantileFunction, ContinuousMonotoneImplicitQuantileFunction, \
    StateDependentGaussianPolicy
from rljax.util import quantile_loss, fake_state, fake_action
from rljax.algorithm import SAC


class IQNSAC(SAC):
    name = 'IQNSAC'

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
        batch_size=100,
        start_steps=10000,
        update_interval=1,
        tau=1e-3,
        fn_actor=None,
        fn_critic=None,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        units_actor=(256, 256),
        units_critic_feature=(256, 256),
        units_critic_outputs=(64, 32),
        log_std_min=-20.0,
        log_std_max=2.0,
        d2rl=False,
        num_quantiles=20,
        num_quantiles_to_drop=4,
        monotone=False,
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

            def fn_actor(s):
                return StateDependentGaussianPolicy(
                    action_space=action_space,
                    hidden_units=units_actor,
                    log_std_min=log_std_min,
                    log_std_max=log_std_max,
                    d2rl=d2rl,
                )(s)
        if not hasattr(self, "fake_args"):
            self.fake_args_critic = (fake_state(state_space), fake_action(action_space), np.empty((1, num_quantiles), dtype=np.float32))

        super(IQNSAC, self).__init__(
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
            lr_alpha=lr_alpha,
        )
        self.num_quantiles = num_quantiles
        self.num_quantiles_target = (num_quantiles - num_quantiles_to_drop) * num_critics

    @partial(jax.jit, static_argnums=0)
    def _calculate_value_list(
        self,
        params_critic: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        cum_p: jnp.ndarray
    ) -> List[jnp.ndarray]:
        return self.critic.apply(params_critic, state, action, cum_p)

    @partial(jax.jit, static_argnums=0)
    def _calculate_value(
        self,
        params_critic: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        cum_p: jnp.ndarray
    ) -> jnp.ndarray:
        return jnp.concatenate(self._calculate_value_list(params_critic, state, action, cum_p), axis=1)

    @partial(jax.jit, static_argnums=0)
    def _loss_critic(
            self,
            params_critic: hk.Params,
            params_critic_target: hk.Params,
            params_actor: hk.Params,
            log_alpha: jnp.ndarray,
            state: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            next_state: np.ndarray,
            weight: np.ndarray or List[jnp.ndarray],
            *args,
            **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        next_action, next_log_pi = self._sample_action(params_actor, next_state, *args, **kwargs)
        randoms = jax.random.uniform(shape=(2, self.batch_size, self.num_quantiles), dtype=jnp.float32, *args, **kwargs)
        cum_p = randoms[0]
        cum_p_prime = randoms[1]

        target = self._calculate_target(params_critic_target, log_alpha, reward, done, next_state, next_action,
                                        next_log_pi, cum_p_prime)
        q_list = self._calculate_value_list(params_critic, state, action, cum_p)
        return self._calculate_loss_critic_and_abs_td(q_list, target, weight, cum_p)

    @partial(jax.jit, static_argnums=0)
    def _calculate_target(
        self,
        params_critic_target: hk.Params,
        log_alpha: jnp.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        next_action: jnp.ndarray,
        next_log_pi: jnp.ndarray,
        cum_p_prime: jnp.ndarray
    ) -> jnp.ndarray:
        next_quantile = self._calculate_value(params_critic_target, next_state, next_action, cum_p_prime)
        next_quantile = self._calculate_sorted_target(next_quantile, self.num_quantiles_target)
        next_quantile -= jnp.exp(log_alpha) * self._calculate_log_pi(next_action, next_log_pi)
        return jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_quantile)

    @staticmethod
    @partial(jax.jit, static_argnums=1)
    def _calculate_sorted_target(quantiles, num_quantile_target):
        return jax.lax.sort(quantiles)[:, :num_quantile_target]

    @partial(jax.jit, static_argnums=0)
    def _calculate_loss_critic_and_abs_td(
        self,
        quantile_list: List[jnp.ndarray],
        target: jnp.ndarray,
        weight: np.ndarray,
        cum_p: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        def _loss(x):
            return quantile_loss(target[:, None, :] - x[:, :, None], cum_p, weight, 'huber')

        def scan_quantile_huber_loss(total, x_i):
            return total + _loss(x_i), total

        quantiles_array = jnp.stack(quantile_list, axis=0)
        loss_critic, abs_td = jax.lax.scan(scan_quantile_huber_loss, init=0., xs=quantiles_array)
        loss_critic /= self.num_critics * self.num_quantiles
        return loss_critic, jax.lax.stop_gradient(loss_critic/self.num_critics)

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        log_alpha: jnp.ndarray,
        state: np.ndarray,
        *args,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        action, log_pi = self._sample_action(params_actor, state, *args, **kwargs)
        cum_p = jax.random.uniform(shape=(action.shape[0], self.num_quantiles), *args, **kwargs)
        mean_q = self._calculate_value(params_critic, state, action, cum_p).mean()
        mean_log_pi = self._calculate_log_pi(action, log_pi).mean()
        return jax.lax.stop_gradient(jnp.exp(log_alpha)) * mean_log_pi - mean_q, jax.lax.stop_gradient(mean_log_pi)
