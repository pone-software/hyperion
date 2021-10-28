"""Utility functions."""
import numpy as np
from scipy.integrate import quad
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
