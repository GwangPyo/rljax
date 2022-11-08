from rljax.network.critic import *
import matplotlib.pyplot as plt
import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial
import optax
from rljax.util.loss import quantile_loss
from rljax.util.optim import optimize
from tqdm import tqdm


def f(t):
    """
    Worst case = -1 (Best among other Threes)
    CVaR 50%   = -1
    Mean       = -1
    """
    return -1.


def g(t):
    """
    Worst case = -2
    CVaR 50%   =  0 (Best among other Threes)
    Mean       =  2
    """
    return 8 * t - 2.


def h(t):
    """
    Worst case = -2.5
    CVaR  50%  = -0.5
    Mean       =  2.5 (Best Among Threes)
    """
    return   6 * ((t + 0.5) ** 2) - 4


def target_f(x, t):
    lt_zero = 1. - jnp.heaviside(x, 0)# jnp.asarray(x <= 0, dtype=jnp.float32)
    reward = lt_zero * (jnp.abs(x) * g(t) + (1. - jnp.abs(x)) * h(t)) + (1. - lt_zero) * ((1. - jnp.abs(x)) * h(t) + jnp.abs(x) * f(t))
    """
 
    # equals to g(x) if actions == -1  and equals to h(x) if actions == 0
    # equals to h(x) if actions == 0, and equals to f(x) if actions == 1
    """
    return reward


class VanillaIQN(object):
    @staticmethod
    def iqn_fn(s, a, probs):
        return ContinuousImplicitQuantileFunction(num_critics=1, )(s, a, probs)

    def __init__(self, lr=3e-4, batch_size=256, num_quantiles=32):
        self.rng = hk.PRNGSequence(jax.random.PRNGKey(42))
        self.f, self.g, self.h, self.target = jax.jit(f), jax.jit(g), jax.jit(h), jax.jit(target_f)
        self.lr = lr
        self.batch_size = batch_size
        self.critic = None
        self.param_critic = None
        self.opt_critic = None
        self.opt_state_critic = None
        self.num_quantiles = num_quantiles
        self.build()

    def build(self):
        self.critic = hk.without_apply_rng(hk.transform(self.iqn_fn))
        obs_placeholder = jnp.ones((1, 1))
        actions_placeholder = jnp.ones((1, 1))
        cum_p_placeholder = jnp.ones((1, self.num_quantiles))
        self.param_critic = self.critic.init(next(self.rng), obs_placeholder, actions_placeholder, cum_p_placeholder)
        opt_init, self.opt_critic = optax.adam(self.lr)
        self.opt_state_critic = opt_init(self.param_critic)

    @partial(jax.jit, static_argnums=(0, 1))
    def make_batch(self, batch_size, key: jax.random.PRNGKey):
        key_1, key_2 = jax.random.split(key, 2)
        x = 2 * jax.random.uniform(key=key_1, shape=(batch_size, 1)) - 1
        s = jnp.zeros_like(x)
        t = jax.random.uniform(key=key_2, shape=(batch_size, 1))
        y = self.target(x, t)
        y = y + jnp.zeros(shape=(batch_size, self.num_quantiles))
        return (s, x), jax.lax.stop_gradient(y)

    @partial(jax.jit, static_argnums=0)
    def _calculate_loss_critic_and_abs_td(
        self,
        quantile_list,
        target: jnp.ndarray,
        weight: np.ndarray,
        cum_p: jnp.ndarray
    ):
        def _loss(x):
            return quantile_loss(target[:, None, :] - x[:, :, None], cum_p, weight, 'huber')

        def scan_quantile_huber_loss(total, x_i):
            return total + _loss(x_i), total

        quantiles_array = jnp.stack(quantile_list, axis=0)
        loss_critic, abs_td = jax.lax.scan(scan_quantile_huber_loss, init=0., xs=quantiles_array)
        loss_critic /= self.num_quantiles
        return loss_critic, None

    def loss(self, params_critic, x_inputs, s_inputs, cum_p, labels):
        y_hat = self.critic.apply(params_critic, s=s_inputs, a=x_inputs, probs=cum_p)
        return self._calculate_loss_critic_and_abs_td(y_hat, labels, weight=jnp.ones((), dtype=jnp.float32),
                                               cum_p=cum_p)

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1,))
    def optimize(
            fn_loss,
            opt,
            opt_state,
            params_to_update: hk.Params,
            *args,
            **kwargs,
    ):
        loss, grad = jax.value_and_grad(fn_loss, has_aux=False)(
            params_to_update,
            *args,
            **kwargs,
        )
        update, opt_state = opt(grad, opt_state)
        params_to_update = optax.apply_updates(params_to_update, update)
        return opt_state, params_to_update, loss

    def train_step(self):
        (s, x), y = self.make_batch(self.batch_size, key=next(self.rng))
        cum_p = jax.random.uniform(shape=(self.batch_size, self.num_quantiles), key=next(self.rng))
        kwargs = {"x_inputs": x, "s_inputs": s, 'cum_p': cum_p, 'labels': y}
        self.opt_state_critic, self.param_critic, loss, _ = optimize(
            fn_loss=self.loss,
            opt=self.opt_critic,
            opt_state=self.opt_state_critic,
            params_to_update=self.param_critic,
            max_grad_norm=None,
            **kwargs
        )
        return loss

    def train(self, steps=10000):
        for _ in tqdm(range(steps)):
            loss = self.train_step()

    def plot(self):
        s = jnp.zeros(shape=(1, 1))
        x = jnp.ones(shape=(1, 1))
        cum_p = jnp.arange(0, 1, 1/2048, dtype=jnp.float32).reshape(1, 2048)

        estimated = self.critic.apply(self.param_critic, s=s, a=x, probs=cum_p)
        estimated = np.asarray(estimated).squeeze()
        plt.plot(estimated, label='estimated_ones')
        plt.plot(self.target(np.ones_like(cum_p).squeeze() + 1e-2, cum_p.squeeze()), label='real_ones')


        estimated_neg_one = self.critic.apply(self.param_critic, s=s, a=-x, probs=cum_p)
        estimated_neg_one = np.asarray(estimated_neg_one).squeeze()

        plt.plot(estimated_neg_one, label='estimated_neg_ones')
        plt.plot(self.target(-np.ones_like(cum_p).squeeze() + 1e-2, cum_p.squeeze()), label='real_neg_ones')

        estimated_zeros = self.critic.apply(self.param_critic, s=s, a=jnp.zeros_like(x), probs=cum_p)
        estimated_zeros = np.asarray(estimated_zeros).squeeze()


        plt.plot(estimated_zeros, label='estimated_zeros')
        plt.plot(self.target(np.zeros_like(cum_p).squeeze() , cum_p.squeeze()), label='real_zeros')

        plt.legend()
        plt.show()


class MonotoneIQN(VanillaIQN):

    @staticmethod
    def iqn_fn(s, a, probs):
        return ContinuousMonotoneImplicitQuantileFunction(num_critics=1, )(s, a, probs)


if __name__ == '__main__':

    model = MonotoneIQN()
    (s, x), y = (model.make_batch(model.batch_size, next(model.rng)))
    print(y.shape)
    model.plot()
    for i in range(100):
        print(i)
        model.train(steps=1000)
        model.plot()

