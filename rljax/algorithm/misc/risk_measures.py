import jax.numpy as jnp
import jax
from functools import partial


@jax.jit
def cvar(confidence: float, cum_p: jnp.ndarray):
    return confidence * cum_p


@jax.jit
def cvar_extended(confidence: jnp.ndarray, cum_p: jnp.ndarray) -> jnp.ndarray:
    """
    if confidence < 0:
        return risk sensitive CVaR, and -1 is worst case update
    elif confidence == 0:
        return risk neutral measure
    else, i.e., confidence > 0:
        return risk seeking measure,

    More specifically,
    if confidence <= 0:
        return (1 + confidence) * U([0, 1])
    else:
        return confidence + (1. - confidence) * U([0, 1])
    NOTE: confidence must be in [-1, 1]
    """
    return jax.nn.relu(confidence) + (1. - jnp.abs(confidence)) * cum_p


@jax.jit
def icdf_normal(x: jnp.ndarray):
    """
    the inverse cumulative distribution function of standard normal distribution
    """
    return jnp.sqrt(2) * jax.lax.erf_inv(2 * x - 1)


@jax.jit
def cdf_normal(x: jnp.ndarray):
    """
    the cumulative distribution function of standard normal distribution
    """
    return 0.5 * (1 + jax.lax.erf(x / jnp.sqrt(2)))


@jax.jit
def wang(confidence: jnp.ndarray, cum_p: jnp.ndarray) -> jnp.ndarray:
    """
    Wang's risk measure
    if confidence < 0:
        risk averse
    else:
        risk sensitive
    confidence range is any real number
    """
    return cdf_normal(icdf_normal(cum_p) + confidence)


@partial(jax.jit, static_argnums=1)
def power(confidence: jnp.ndarray, cum_p: jnp.ndarray) -> jnp.ndarray:
    _pow = 1/(1 + jnp.abs(confidence))
    return jnp.where(confidence >= 0, jnp.power(cum_p, _pow), 1. - jnp.power((1 - cum_p), _pow))


