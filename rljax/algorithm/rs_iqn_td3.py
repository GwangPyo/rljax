from functools import partial
from typing import List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from rljax.algorithm import IQNTD3
from rljax.algorithm.misc.risk_measures import cvar, power, wang

risk_measures = {"cvar": cvar, "power": power, "wang": wang}


class RS_IQNTD3(IQNTD3):
    name = 'RS_IQNTD3'
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
        units_actor=(256, 256),
        units_critic_feature=(256, 256),
        units_critic_outputs=(64, 32),
        d2rl=False,
        num_quantiles=25,
        num_quantiles_to_drop=2,
        std=0.1,
        std_target=0.2,
        clip_noise=0.5,
        update_interval_policy=2,
        monotone=False,
        risk_measure='cvar',
        risk_param=0.5,
    ):
        self.risk_measure = risk_measures[risk_measure]
        self.risk_param = risk_param

        super(RS_IQNTD3, self).__init__(
            num_agent_steps,
            state_space,
            action_space,
            seed,
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
            units_actor=units_actor,
            units_critic_feature=units_critic_feature,
            units_critic_outputs=units_critic_outputs,
            d2rl=d2rl,
            num_quantiles=num_quantiles,
            num_quantiles_to_drop=num_quantiles_to_drop,
            std=std,
            std_target=std_target,
            clip_noise=clip_noise,
            update_interval_policy=update_interval_policy,
            monotone=monotone
        )

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        state: jnp.ndarray,
        *args,
        **kwargs,
    ) -> Tuple[jnp.ndarray, None]:
        action = self.actor.apply(params_actor, state)
        cum_p = jax.random.uniform(shape=(action.shape[0], self.num_quantiles),
                                   dtype=action.dtype,
                                   *args, **kwargs)
        cum_p = self.risk_measure(self.risk_param, cum_p)
        mean_q = self._calculate_value(params_critic, state, action, cum_p).mean()
        return -mean_q, None
