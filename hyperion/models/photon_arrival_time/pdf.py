import numpy as np
import scipy.stats
from jax import numpy as jnp
from jax import value_and_grad, jit


def expon_pdf(x, a):
    """Exponential PDF."""
    return 1 / a * jnp.exp(-x / a)


def make_exp_exp(data, weights):
    """
    Create a two-exponential mixture model pdf.

    This functions returns the likelihood evaluated on data and weights, and
    the likelihood function.

    Parameters:
        data: ndarray
        weights: ndarray

    """

    def func(xs, scale1, scale2, mix):
        lower = jnp.min(jnp.array([scale1, scale2])) * 100
        upper = jnp.max(jnp.array([scale1, scale2])) * 100

        res = jnp.log(mix * expon_pdf(xs, lower) + (1 - mix) * expon_pdf(xs, upper))

        return res

    def obj(scale1, scale2, mix):
        val = -jnp.sum((func(data, scale1, scale2, mix) * weights))
        return val

    return jit(value_and_grad(obj, [0, 1, 2])), func


def make_exp_exp_exp(data, data_weights):
    """
    Create a three-exponential mixture model pdf.

    This functions returns the likelihood evaluated on data and weights, and
    the likelihood function.

    Parameters:
        data: ndarray
        weights: ndarray

    """

    def func(xs, scale1, scale2, scale3, w1, w2):

        scales = jnp.array([scale1, scale2, scale3]) * 100
        scales = jnp.sort(scales)

        weights = (
            jnp.array(
                [jnp.sin(w1) * jnp.cos(w2), jnp.sin(w1) * jnp.sin(w2), jnp.cos(w1)]
            )
            ** 2
        )
        """
        wsum = w1 + w2 + w3
        weights = jnp.array([w1 / wsum, w2 / wsum, w3 / wsum])
        """
        res = jnp.log(
            weights[0] * expon_pdf(xs, scales[0])
            + weights[1] * expon_pdf(xs, scales[1])
            + weights[2] * expon_pdf(xs, scales[2])
        )

        return res

    def obj(scale1, scale2, scale3, w1, w2):
        val = -jnp.sum((func(data, scale1, scale2, scale3, w1, w2) * data_weights))
        return val

    return jit(value_and_grad(obj, [0, 1, 2, 3, 4])), func


def make_gamma_exponential(data, weights):
    """
    Create a gamma-exponential mixture model pdf.

    This functions returns the likelihood evaluated on data and weights, and
    the likelihood function.

    Parameters:
        data: ndarray
        weights: ndarray

    """

    def func(xs, args):
        a, scale, scale2, mix = args
        scale *= 100
        scale2 *= 100

        f1 = np.log(mix) + scipy.stats.gamma.logpdf(xs, a, scale=scale)
        f2 = np.log(1 - mix) + scipy.stats.expon.logpdf(xs, scale=scale2)
        stacked = np.vstack([f1, f2])

        res = scipy.special.logsumexp(stacked, axis=0)
        res[~np.isfinite(res)] = 1e4

        return res

    def obj(args):
        return -(func(data, args) * weights).sum()

    return obj, func
