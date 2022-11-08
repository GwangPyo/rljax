from functools import partial
from typing import List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from rljax.network import ContinuousImplicitQuantileFunction, ContinuousMonotoneImplicitQuantileFunction, \
    StateDependentGaussianPolicy
from rljax.util import quantile_loss, fake_state, fake_action
from rljax.algorithm import IQNSAC
from rljax.algorithm.misc.risk_measures import cvar, wang, power


class RS_IQNSAC(IQNSAC):
    name = 'RS_IQNSAC'

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
        risk_measure='cvar',
        risk_param=0.5,
    ):
        risk_measures = {"cvar": cvar, "wang": wang, "power": power}
        self.risk_measure = risk_measures[risk_measure]
        self.risk_param = risk_param
        super(RS_IQNSAC, self).__init__(
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
            units_actor=units_actor,
            units_critic_feature=units_critic_feature,
            units_critic_outputs=units_critic_outputs,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            d2rl=d2rl,
            num_quantiles=num_quantiles,
            num_quantiles_to_drop=num_quantiles_to_drop,
            monotone=monotone
        )

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
        cum_p = self.risk_measure(self.risk_param, cum_p)
        mean_q = self._calculate_value(params_critic, state, action, cum_p).mean()
        mean_log_pi = self._calculate_log_pi(action, log_pi).mean()
        return jax.lax.stop_gradient(jnp.exp(log_alpha)) * mean_log_pi - mean_q, jax.lax.stop_gradient(mean_log_pi)
