import numpy as np
import scipy.stats


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
