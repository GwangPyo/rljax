import jax
import haiku as hk
import jax.numpy as jnp
from typing import Sequence, Type, Tuple, Optional
from functools import partial
import numpy as np
import optax

import matplotlib.pyplot as plt

CLAMP_MAX = 20


@jax.jit
def log_prob_normal(x: jnp.ndarray):
    return -0.5 * (jnp.square(x) + jnp.log(2 * np.pi))


class ScaleNet(hk.Module):
    def __init__(self, dim):
        super(ScaleNet, self).__init__()
        self.dim = dim

    def __call__(self, x):
        pre = hk.Linear(self.dim, with_bias=True, w_init=hk.initializers.RandomUniform(-0.05, 0.05))(x)
        return pre.clip(-CLAMP_MAX, 2)


class FlowLayer(hk.Module):
    def __init__(self, dim, features_dim):
        super(FlowLayer, self).__init__()
        self.dim = dim
        self.features_dim = features_dim

    def __call__(self, *args, **kwargs):
        pass


class AffineFlowLayer(FlowLayer):
    def __init__(self, dim, features_dim):
        super().__init__(dim, features_dim)

    def __call__(self, x, feature):
        feature_net = hk.Linear(self.features_dim, )
        scale_net = ScaleNet(self.dim)
        bias_net = hk.Linear(self.dim, with_bias=True, w_init=hk.initializers.RandomUniform(-0.05, 0.05))

        def init(x, feature):
            next_features = jax.nn.relu(feature_net(feature))
            scale, bias = scale_net(feature), bias_net(feature)
            return jnp.exp(scale) * x + bias, next_features

        def compute_param(feature) -> Tuple[Tuple, jnp.ndarray]:
            # do not compute a x + b but, pass params
            next_features = jax.nn.relu(feature_net(feature))
            scale, bias = scale_net(feature), bias_net(feature)
            return (scale, bias), next_features

        def forward_with_params(x, feature):
            next_feature = jax.nn.relu(feature_net(feature))
            log_alpha = scale_net(feature)
            bias = bias_net(feature)
            log_det = log_alpha * jnp.ones_like(x)
            return jnp.exp(log_alpha) * x + bias, next_feature, log_det, (log_alpha, )

        def forward(x, feature):
            next_feature = jax.nn.relu(feature_net(feature))
            log_alpha = scale_net(feature)
            bias = bias_net(feature)
            log_det = log_alpha * jnp.ones_like(x)
            return jnp.exp(log_alpha) * x + bias, next_feature, log_det

        def inverse(x, log_alpha, bias):
            log_det = log_alpha * jnp.ones_like(x)
            scale = jnp.exp(-log_alpha)
            return (x - bias) * scale, log_det

        return init, (compute_param, forward, inverse, forward_with_params)


class PReluFlowLayer(FlowLayer):
    def __init__(self, dim, features_dim,  ):
        super().__init__(dim, features_dim)

    def __call__(self, x, feature):
        feature_net = hk.Linear(self.features_dim, w_init=hk.initializers.RandomUniform(-0.05, 0.05))
        scale_net = ScaleNet(self.dim, )

        def init(x, feature):
            next_features = jax.nn.relu(feature_net(feature))
            prealpha = scale_net(feature)
            prelu_var = jax.lax.select(x >= 0, x, jnp.exp(prealpha) * x)
            return prelu_var, next_features

        def compute_param(feature) -> Tuple[Tuple, jnp.ndarray]:
            # do not compute a x + b but, pass params
            next_features = jax.nn.relu(feature_net(feature))
            prealpha = scale_net(feature)
            return (prealpha, ), next_features

        def forward(x, feature):
            next_features = jax.nn.relu(feature_net(feature))
            log_alpha = scale_net(feature)
            alpha = jnp.exp(log_alpha)
            prelu_var = jax.lax.select(x >= 0, x, alpha * x)
            log_det = jax.lax.select(x >= 0, jnp.zeros_like(x), log_alpha * jnp.ones_like(x))
            return prelu_var, next_features, log_det

        def forward_with_params(x, feature):
            next_features = jax.nn.relu(feature_net(feature))
            log_alpha = scale_net(feature)
            alpha = jnp.exp(log_alpha)
            prelu_var = jax.lax.select(x >= 0, x, alpha * x)
            log_det = jax.lax.select(x >= 0, jnp.zeros_like(x), log_alpha * jnp.ones_like(x))
            return prelu_var, next_features, log_det, (log_alpha, )

        def inverse(y, log_alpha):
            recip_alpha = jnp.exp(-log_alpha)
            log_det = jax.lax.select(y >= 0, jnp.zeros_like(y), log_alpha * jnp.ones_like(y))
            return jax.lax.select(y >= 0, y, y * recip_alpha), log_det

        return init, (compute_param, forward, inverse, forward_with_params)


class SigmoidFlowLayer(hk.Module):
    def __call__(self, x, feature):
        def init(x, feature):
            return jax.nn.sigmoid(x), feature

        def compute_param(feature):
            # empty tuple
            return (), feature

        def forward(x, feature):
            y = jax.nn.sigmoid(x)
            log_det = jnp.log(y) + jnp.log(1 - y)
            log_det = log_det.clip(-CLAMP_MAX, 0.)
            return y, feature, log_det

        def forward_with_params(x, feature):
            y, feature, log_det = forward(x, feature)
            return y, feature, log_det, ()

        def inverse(y, ):
            # clip log y rather than clip y  for numerical stability
            log_y = jnp.log(y).clip(-CLAMP_MAX, 0)
            log_one_minus_y = jnp.log(1. - y).clip(-CLAMP_MAX, 0)
            x = log_y - log_one_minus_y
            log_det = log_y + log_one_minus_y
            return x, log_det
        return init, (compute_param, forward, inverse, forward_with_params)


class TanhFlowLayer(hk.Module):
    def __call__(self, x, feature):
        def init(x, feature):
            return jax.nn.tanh(x), feature

        def compute_param(feature):
            # empty tuple
            return (), feature

        def forward(x, feature):
            y = jax.nn.tanh(x)
            log_det = jnp.log(1 - jnp.square(y) + 1e-6)
            log_det = jnp.clip(log_det, -CLAMP_MAX, 0)
            return y, feature, log_det

        def forward_with_params(x, feature):
            y, feature, log_det = forward(x, feature)
            return y, feature, log_det, ()

        def inverse(y, ):
            x = jax.lax.atanh(jnp.clip(y, -1 + 1e-6, 1 - 1e-6))
            log_det = jnp.log(1. - jnp.square(y) + 1e-6)
            return x, log_det
        return init, (compute_param, forward, inverse, forward_with_params)


class InverseSigmoidLayer(hk.Module):

    def __call__(self, x, feature):
        def init(x, feature):
            x = x.clip(1e-6, 1 - 1e-6)
            y = jnp.log(x).clip(-1000000, 0) - jnp.log(1. - x).clip(-1000000, 0)
            return y, feature

        def compute_param(feature):
            # empty tuple
            return (), feature

        def forward(x, feature):
            log_x = jnp.log(x).clip(-5, 0)
            log_one_minus_x = jnp.log(1. - x).clip(-5, 0)
            y = log_x - log_one_minus_x
            log_det = log_x + log_one_minus_x
            return y, feature, log_det

        def inverse(y, ):
            # clip log y rather than clip y  for numerical stability
            x = jax.nn.sigmoid(y)

            log_det = jnp.log(x.clip(1e-7, 1.-1e-7)) + jnp.log((1. - x).clip(1e-7, 1.-1e-7))
            return x, log_det

        return init, (compute_param, forward, inverse)


class FlowBlock(hk.Module):
    """
    Flow block consists Affine -> PReLU
    """

    def __init__(self, dim, net_arch):
        super(FlowBlock, self).__init__()
        self.dim = dim
        self.features_dim = net_arch
        assert len(net_arch) == 2

    def __call__(self, x, features):
        affine_layer = AffineFlowLayer(self.dim, self.features_dim[0])
        prelu_layer = PReluFlowLayer(self.dim, self.features_dim[1])

        init1, (compute_param1, forward1, inverse1, forward_with_params1) = affine_layer(x, features)
        init2, (compute_param2, forward2, inverse2, forward_with_params2) = prelu_layer(x, features)

        def init(x, feaures):
            y, next_feature = init1(x, feaures)
            return init2(y, next_feature)

        def compute_param(features):
            param1, features_1 = compute_param1(features)
            param2, next_features = compute_param2(features_1)

            return param1 + param2, next_features

        def inverse(y, scale, bias, alpha):
            x2, log_det2 = inverse2(y, alpha)
            x, log_det1 = inverse1(x2, scale, bias)
            return x, log_det2 + log_det1

        def forward(x, features):
            mid, features_1, log_det1 = forward1(x, features)
            y, next_feature, log_det2 = forward2(mid, features_1)
            return y, next_feature, log_det1 + log_det2

        def forward_with_params(x, feature):
            mid, features_1, log_det1, params1 = forward_with_params1(x, feature)
            y, next_feature, log_det2, params2 = forward_with_params2(mid, features_1)
            return y, next_feature, log_det1 + log_det2, params1 + params2

        return init, (compute_param, forward, inverse, forward_with_params)


class FlowSequential(hk.Module):
    def __init__(self, dim, net_arch, outlayer=TanhFlowLayer,
                 name='conditional_flow'):
        super(FlowSequential, self).__init__(name=name)

        assert (len(net_arch) % 2) == 0
        self.dim = dim
        self._net_arch = np.asarray(net_arch, dtype=np.int32)
        self.net_arch = np.split(self._net_arch, len(self._net_arch) //2)
        self.out_layer = outlayer

    def __call__(self, x, feature):
        blocks = [FlowBlock(self.dim, arch) for arch in self.net_arch]
        if self.out_layer:
            blocks.append(self.out_layer())
        inits = []
        compute_params = []
        forwards = []
        inverses = []
        forward_with_params_functions = []

        for block in blocks:
            _init, (_compute_params, _forward, _inverse, _forward_with_params) = block(x, feature)
            inits.append(_init)
            compute_params.append(_compute_params)
            forwards.append(_forward)
            inverses.append(_inverse)
            forward_with_params_functions.append(_forward_with_params)

        def init(x, feature):
            for f in inits:
                x, feature = f(x, feature)
            return x, feature

        def compute_param(feature):
            flow_parameters = []
            for f in compute_params:
                __flow_param, feature = f(feature)
                flow_parameters.append(__flow_param)

            return flow_parameters

        def forward(x, feature):
            log_prob = log_prob_normal(x)
            for f in forwards:
                x, feature, __log_det = f(x, feature)
                log_prob -= __log_det
            return x, log_prob.sum(axis=-1, keepdims=True)

        def forward_with_params(x, feature):
            flow_parameters = ()
            log_prob = log_prob_normal(x)
            for f in forward_with_params_functions:
                x, feature, __log_det, params = f(x, feature)
                log_prob -= __log_det

                flow_parameters = flow_parameters + params
            return x, log_prob.sum(axis=-1, keepdims=True), jnp.concatenate(flow_parameters, axis=-1)

        def inverse(y, param_list):
            log_det = jnp.zeros_like(y)
            for flow_args, f_inv in zip(reversed(param_list), reversed(inverses)):
                y, __log_det = f_inv(y, *flow_args)
                log_det += __log_det
            log_prob = log_prob_normal(y) - log_det
            return y, log_prob.sum(axis=-1, keepdims=True)

        return init, (compute_param, forward, inverse, forward_with_params)


class Tester(object):
    def __init__(self):
        x_placeholder = jnp.ones((1, 1))
        feature_placeholder = jnp.ones((1, 2))

        def factory():
            return FlowSequential(1, [32 for _ in range(4)])(x_placeholder, feature_placeholder)

        f = hk.without_apply_rng(hk.multi_transform(factory))
        self.rng = hk.PRNGSequence(jax.random.PRNGKey(42))
        self.params = f.init(next(self.rng), x_placeholder, feature_placeholder)
        self.flow_param, self.forward, self.inverse, self.forward_with_params = f.apply
        opt_init, self.optim = optax.adam(learning_rate=1e-4, )
        self.opt_state = opt_init(self.params)
        self.batch_size = 512

    @partial(jax.jit, static_argnums=0)
    def make_sample(self, rng: jax.random.PRNGKey):
        param_key, beta_key = jax.random.split(rng, num=2)
        beta_param = jax.random.uniform(shape=(self.batch_size, 2), key=param_key) * 10
        beta = jax.random.beta(a=beta_param[:, [0]], b=beta_param[:, [1]], key=beta_key)
        return beta, beta_param

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

    def train_step(self, rng: jax.random.PRNGKey):
        y, feature = self.make_sample(rng)
        kwargs = {"feature": feature, "y": y, }
        """
        flow_params = self.flow_param(self.params, feature)
        print("inv sigmoid")
        y_prime = np.log(y).clip(-CLAMP_MAX, 0) - jnp.log(1.- y).clip(-CLAMP_MAX, 0)
        print(y_prime)
        print(" log det ")
        print(jnp.log(y).clip(-CLAMP_MAX, 0) +  jnp.log(1. - y).clip(CLAMP_MAX, 0))

        flow_params.pop()
        def inv_block(a,b,c, t):
            t_prime = jax.lax.select(t >= 0, t, jnp.exp(-c) * t)
            t_final = (t - b) * jnp.exp(-a)
            return t_prime, t_final
        print("inv 1 block")
        print("params")
        (a, b, c) = flow_params.pop()
        print(a, b, c)
        print("maximal log det", a + c)

        p, q = inv_block(a, b, c, y_prime)
        print(p)
        print(q)
        y_prime = q
        print("inv 2 block")
        print("params")
        print(a, b, c)
        (a, b, c) = flow_params.pop()
        p, q = inv_block(a, b, c, y_prime)
        print(p)
        print(q)
        y_prime = q

        print("sigmoid")
        print(jax.nn.sigmoid(y_prime))
        """
        self.opt_state, self.params, loss = self.optimize(
            fn_loss=self.loss,
            opt=self.optim,
            opt_state=self.opt_state,
            params_to_update=self.params,
            **kwargs)

        return loss

    def train(self, n_steps: int):
        for _ in range(n_steps):
            rng = next(self.rng)
            loss = self.train_step(rng)
            print(loss)

    @partial(jax.jit, static_argnums=0)
    def loss(self, param: hk.Params, y: jnp.ndarray, feature: jnp.ndarray):
        flow_params = self.flow_param(param, feature)
        z, log_det = self.inverse(param, y, flow_params)
        log_pr_z = log_prob_normal(z)
        log_prob = log_pr_z - log_det
        return -log_prob.sum(axis=-1).mean()

    @partial(jax.jit, static_argnums=0)
    def log_prob(self, param: hk.Params, y: jnp.ndarray, feature: jnp.ndarray):
        flow_params = self.flow_param(param, feature)
        z, log_det = self.inverse(param, y, flow_params)
        log_pr_z = log_prob_normal(z)
        log_prob = log_pr_z - log_det
        return log_prob.sum(axis=-1)

    @staticmethod
    @jax.jit
    def log_prob_normal(x: jnp.ndarray):
        return (-jnp.square(x)/2 - jnp.log(jnp.pi * 2))/2


class Plotter(object):
    def __init__(self, target_a = 0.5, target_b = 0.5):
        self.target_a = target_a
        self.target_b = target_b

    @property
    def sample(self):
        return np.arange(1e-3, 1, 1/1000)

    def plots_original(self, ):
        import scipy
        samples = self.sample
        gamma_func = scipy.special.gamma
        B_ab = gamma_func(self.target_a) * gamma_func(self.target_b) / (gamma_func(self.target_a + self.target_b))
        left = np.power(samples, self.target_a - 1)
        right = np.power(1. - samples, self.target_b - 1)

        pdf = left * right / B_ab
        plt.plot(samples, pdf, label='original')
        # plt.show()

    def plot_sample(self, log_prob_samples):
        prob_sample = np.exp(log_prob_samples)
        plt.plot(self.sample, prob_sample, label='estimated')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    flow_test = Tester()
    plotter = Plotter(2, 5)
    sample = plotter.sample
    features = jnp.array([plotter.target_a, plotter.target_b])
    log_prob = flow_test.log_prob(flow_test.params, plotter.sample[..., None], features[None])
    log_prob = np.array(log_prob)
    print(log_prob.shape)
    plotter.plots_original()
    plotter.plot_sample(log_prob)
    for _ in range(100):
        flow_test.train(10000)
        features = jnp.array([plotter.target_a, plotter.target_b])
        log_prob = flow_test.log_prob(flow_test.params, plotter.sample[..., None], features[None])
        print(log_prob.shape)
        plotter.plots_original()
        plotter.plot_sample(log_prob)