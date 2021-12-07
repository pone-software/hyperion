"""Utility functions."""
import numpy as np
import scipy.stats
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline

from .constants import Constants


def calc_tres(
    t: np.ndarray, det_radius: float, det_dist: float, c_medium: float
) -> np.ndarray:
    """
    Calculate time residual.

    The time residual is calculated by subtracting the expected (geometric)
    time a photon takes to travel det_dist-det_radius from the measured arrival time

    Parameters:
        t:
    """
    return t - ((det_dist - det_radius) / c_medium)


def cherenkov_ang_dist(costheta: np.ndarray, n_ph: float = 1.35) -> np.ndarray:
    """
    Angular distribution of cherenkov photons for EM cascades.

    Taken from https://arxiv.org/pdf/1210.5140.pdf
    """
    # params for e-

    a = Constants.CherenkovLightYield.AngDist.a
    b = Constants.CherenkovLightYield.AngDist.b
    c = Constants.CherenkovLightYield.AngDist.c
    cos_theta_c = 1 / n_ph
    d = Constants.CherenkovLightYield.AngDist.d
    return a * np.exp(b * np.abs(costheta - cos_theta_c) ** c) + d


ANG_DIST_INT = quad(cherenkov_ang_dist, -1, 1)[0]


def calculate_min_number_steps(
    ref_index_func,
    scattering_length_function,
    det_dist,
    max_tres,
    wavelength,
    p_threshold,
):
    """
    Calculate the minimum number of steps required to satistfy p < p_thresh.

    For a given refractive index function, scattering length function, distance to detector
    and maximum time_residual, calculate how many propagation steps have to be performed such that the probability
    of a photon having propagated for less time than the maximum time residual is less then `p_threshold`.

    Parameters:
        scattering_length_function: function
            function that returns scattering length as function of wavelength
        ref_index_func: function
            function that returns the refractive index as function of wavelength
        det_dist: float
            distance from emitter to detector
        max_tres: float
            maximum time residual
        wavelength: float
            photon wavelength
        p_threshold: float
            probability threshold
    """
    c_medium_f = lambda wl: Constants.BaseConstants.c_vac / ref_index_func(  # noqa E731
        wl
    )

    t_geo = det_dist / (c_medium_f(wavelength) / 1e9) + max_tres
    func = (
        lambda step: scipy.stats.gamma.cdf(
            t_geo * (c_medium_f(wavelength) / 1e9),
            step,
            scale=scattering_length_function(wavelength),
        )
        - 0.01
    )
    lim = scipy.optimize.brentq(func, 2, 100)

    return int(np.ceil(lim))


def make_cascadia_abs_len_func(sca_len_func):
    att_lengths = np.asarray([[365, 10.4], [400, 14.6], [450, 27.7], [585, 7.1]])
    spl = UnivariateSpline(att_lengths[:, 0], np.log(att_lengths[:, 1]), k=2, s=0.01)

    def abs_len(wavelength):
        return 1 / (1 / np.exp(spl(wavelength)) - 1 / sca_len_func(wavelength))

    return abs_len
