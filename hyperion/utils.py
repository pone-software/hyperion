import numpy as np
from scipy.integrate import quad
from .constants import Constants


def calc_tres(t, det_radius, det_dist, c_medium):
    """
    Calculate time residual.

    The time residual is calculated by subtracting the expected (geometric)
    time a photon takes to travel det_dist-det_radius from the measured arrival time
    """
    return t - ((det_dist - det_radius) / c_medium)


def cherenkov_ang_dist(costheta, n_ph=1.35):
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
