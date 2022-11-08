from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from rljax.algorithm.td3 import TD3
from rljax.network.actor import FlowPolicy
from rljax.util import fake_action, fake_state


class FlowTD3(TD3):
    name = "FlowTD3"

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
        lr_actor=3e-4,
        lr_critic=3e-4,
        units_actor=(64, 64, 32, 32),
        units_critic=(256, 256),
        d2rl=False,
        std=0.1,
        std_target=0.2,
        clip_noise=0.5,
        update_interval_policy=2,
    ):
        if d2rl:
            self.name += "-D2RL"

        if fn_actor is None:
            def fn_actor(s, z):
                return FlowPolicy(
                    action_space=action_space,
                    d2rl=d2rl,
                    net_arch=units_actor
                )(s, z)
        if not hasattr(self, "fake_args_actor"):
            self.fake_args_actor = (fake_state(state_space), fake_action(action_space))
        self.action_shape = action_space.shape

        super(FlowTD3, self).__init__(
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
            units_critic=units_critic,
            clip_noise=clip_noise,
            update_interval_policy=update_interval_policy
        )
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
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        state: jnp.ndarray,
        *args,
        **kwargs,
    ) -> Tuple[jnp.ndarray, None]:

        normal = jax.random.normal(shape=(state.shape[0], ) + self.action_shape, *args, **kwargs)
        action, _ = self.actor.apply(params_actor, state, normal)
        mean_q = self.critic.apply(params_critic, state, action)[0].mean()
        return -mean_q.mean(), None

    @partial(jax.jit, static_argnums=0)
    def _explore(
            self,
            params_actor: hk.Params,
            state: np.ndarray,
            key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        normal = jax.random.normal(shape=(state.shape[0], ) + self.action_shape, key=key)
        action, _ = self.actor.apply(params_actor, state, normal)
        return action

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
