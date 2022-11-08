from functools import partial
from typing import List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from rljax.network.critic import RCDSACCritic
from rljax.network.actor import RC_DSACPolicy
from rljax.util import quantile_loss, fake_state, fake_action, reparameterize_gaussian_and_tanh
from rljax.algorithm import IQNSAC
from rljax.algorithm.misc.risk_measures import cvar, power, wang


class RCDSAC(IQNSAC):
    name = 'RCDSAC'
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
        tau=1e-3,
        fn_actor=None,
        fn_critic=None,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        units_actor=(256, 256),
        units_critic_feature=(256, 256),
        log_std_min=-20.0,
        log_std_max=2.0,
        d2rl=False,
        num_quantiles=16,
        num_quantiles_to_drop=0,
        confidence_min: float = 0.,
        confidence_max: float = 1.,
        risk_measure: str = "cvar",
        target_confidence: float = 0.5
    ):
        risk_measure_map = {"cvar": cvar, "power": power, "wang": wang}
        try:
            self.risk_measure_sample = risk_measure_map[risk_measure]
        except KeyError:
            raise NotImplementedError(f"risk measure {risk_measure} is not implemented")
        self.confidence_min = confidence_min
        self.confidence_max = confidence_max
        self.sample_confidence = jax.jit(partial(self._sample_confidence, min_=confidence_min, max_=confidence_max))

        self.target_confidence = target_confidence
        if d2rl:
            self.name += "-D2RL"

        if fn_critic is None:

            def fn_critic(s, a, cum_p, betas):
                return RCDSACCritic(
                    num_critics=num_critics,
                    hidden_units_features=units_critic_feature,
                    d2rl=d2rl,
                )(s, a, cum_p, betas)

        if fn_actor is None:

            def fn_actor(s, betas):
                return RC_DSACPolicy(
                    action_space=action_space,
                    hidden_units=units_actor,
                    log_std_min=log_std_min,
                    log_std_max=log_std_max,
                    d2rl=d2rl,
                )(s, betas)
        if not hasattr(self, "fake_args"):
            self.fake_args_critic = (fake_state(state_space), fake_action(action_space), np.empty((1, num_quantiles),dtype=np.float32),
                                     np.empty((1, 1), dtype=np.float32))
            self.fake_args_actor = (fake_state(state_space), np.empty((1, num_quantiles), dtype=np.float32))

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

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _sample_confidence(self, probs: jnp.ndarray, min_: float, max_: float) -> jnp.ndarray:
        scale = (max_ - min_)
        pre = scale * probs + min_
        return pre

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        confidence = self.target_confidence * jnp.ones(shape=(state.shape[0], 1), dtype=jnp.float32)
        mean, _ = self.actor.apply(params_actor, state, confidence)
        return jnp.tanh(mean)

    @partial(jax.jit, static_argnums=0)
    def _select_action_with_target(self, params_actor, state, target_risk, key):
        confidence = target_risk * jnp.ones(shape=(state.shape[0], 1), dtype=jnp.float32)
        mean, log_std = self.actor.apply(params_actor, state, confidence)
        return reparameterize_gaussian_and_tanh(mean, log_std, key, False)

    def select_action_with_target(self, state, target_risk):
        action = self._select_action_with_target(self.params_actor, jnp.asarray(state[None, ...], dtype=jnp.float32),
                                                 target_risk,
                                                 next(self.rng))

        return np.array(action[0])

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        base = jax.random.uniform(shape=(state.shape[0], 1), key=key, dtype=jnp.float32)
        confidence = self.sample_confidence(base)
        mean, log_std = self.actor.apply(params_actor, state, confidence)
        return reparameterize_gaussian_and_tanh(mean, log_std, key, False)

    @partial(jax.jit, static_argnums=0)
    def _sample_action(
            self,
            params_actor: hk.Params,
            state: jnp.ndarray,
            betas: jnp.ndarray,
            key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mean, log_std = self.actor.apply(params_actor, state, betas)

        return reparameterize_gaussian_and_tanh(mean, log_std, key, True)

    @partial(jax.jit, static_argnums=0)
    def _calculate_value_list(
        self,
        params_critic: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        cum_p: jnp.ndarray,
        betas: jnp.ndarray
    ) -> List[jnp.ndarray]:
        return self.critic.apply(params_critic, state, action, cum_p, betas)

    @partial(jax.jit, static_argnums=0)
    def _calculate_value(
        self,
        params_critic: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        cum_p: jnp.ndarray,
        betas: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.concatenate(self._calculate_value_list(params_critic, state, action, cum_p, betas), axis=1)

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
            next_state: jnp.ndarray,
            weight: np.ndarray or List[jnp.ndarray],
            *args,
            **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        base = jax.random.uniform(shape=(self.batch_size, 1), dtype=jnp.float32, *args, **kwargs)
        betas = self.sample_confidence(base)
        randoms = jax.random.uniform(shape=(2, self.batch_size, self.num_quantiles), dtype=jnp.float32, *args, **kwargs)
        cum_p, cum_p_prime = randoms[0], randoms[1]
        next_action, next_log_pi = self._sample_action(params_actor, next_state, betas, *args, **kwargs)
        target = self._calculate_target(params_critic_target, log_alpha, reward, done, next_state, next_action,
                                        next_log_pi, cum_p_prime, betas)
        q_list = self._calculate_value_list(params_critic, state, action, cum_p, betas)
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
        cum_p_prime: jnp.ndarray,
        betas: jnp.ndarray
    ) -> jnp.ndarray:
        next_quantile = self._calculate_value(params_critic_target, next_state, next_action, cum_p_prime,
                                              betas=betas)
        next_quantile = self._calculate_sorted_target(next_quantile, self.num_quantiles_target)
        next_quantile -= jax.lax.exp(log_alpha) * self._calculate_log_pi(next_action, next_log_pi)
        return jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_quantile)

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
        base  = jax.random.uniform(shape=(state.shape[0], 1 + self.num_quantiles), *args, **kwargs)
        base, cum_p = base[:, :-1], base[:, [-1]]
        betas = self.sample_confidence(base)
        cum_p = self.risk_measure_sample(confidence=betas, cum_p=cum_p)

        action, log_pi = self._sample_action(params_actor, state, betas, *args, **kwargs)

        # risk sensitive update over actor
        q_list = self._calculate_value_list(params_critic, state, action, cum_p, betas)
        q_means = jnp.stack(q_list, axis=0).mean(axis=-1).mean(axis=0)

        mean_log_pi = self._calculate_log_pi(action, log_pi)[..., 0]
        return jnp.mean(jax.lax.stop_gradient(jax.lax.exp(log_alpha)) * mean_log_pi - q_means), \
               jax.lax.stop_gradient(mean_log_pi.mean())
