"""Script for fitting time residual distributions."""
import scipy.optimize
import numpy as np
from tqdm import tqdm, trange
import pickle
from argparse import ArgumentParser
from jax.config import config

config.update("jax_enable_x64", True)

from hyperion.models.photon_arrival_time.pdf import (
    make_exp_exp_exp,
    make_obj_func,
    fb5_mle,
)
from hyperion.utils import cherenkov_ang_dist, ANG_DIST_INT, calc_tres


det_radius = 0.21
n_ph = 1.35
sca_len = 100
c_medium = 0.3 / n_ph
abs_len = 30


def make_data(t, w, det_dist, thr=2):
    """Truncate data below threshold."""
    tres = calc_tres(t, det_radius, det_dist, c_medium)
    mask = tres > thr

    return tres[mask] - thr, w[mask], 1 - (w[mask].sum() / w.sum())


def wrap_obj_func(f):
    """Wrap an objective function by unpacking parameters."""

    def _f(*pars):
        if len(pars) == 1:
            pars = pars[0]
        res = f(*pars)
        return np.array(res[0], order="F"), np.array(res[1], order="F")

    return _f


def fit(obj):
    """
    Fit the objective function.

    The fit is repeated 5 times with varying seeds
    """

    best_res = None
    for _ in range(5):

        seed = np.random.uniform(0, 1, size=5)
        seed[3:] *= np.pi / 2

        res = scipy.optimize.fmin_l_bfgs_b(
            obj,
            seed,
            bounds=(
                (1e-3, None),
                (1e-3, None),
                (1e-3, None),
                (0, np.pi / 2),
                (0, np.pi / 2),
            ),
            factr=100,
            approx_grad=False,
        )
        if res[2]["warnflag"] == 2:
            continue
        if (best_res is None) or res[1] < best_res[1]:
            best_res = res
    return best_res


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-i", "--infile", required=True, dest="infile")
    parser.add_argument("-o", "--outfile", required=True, dest="outfile")
    args = parser.parse_args()

    det_ph = pickle.load(open(args.infile, "rb"))

    fit_results = []
    rstate = np.random.RandomState(0)

    pdf = make_exp_exp_exp()

    for i in trange(len(det_ph)):
        thetas = np.arccos(rstate.uniform(-1, 1, 100))
        thetas = np.concatenate(
            [
                thetas,
                [
                    np.arccos(1 / 1.35) - 0.01,
                    np.arccos(1 / 1.35),
                    np.arccos(1 / 1.35) + 0.01,
                ],
            ]
        )
        det_dist, isec_times, ph_thetas, stepss, isec_poss = det_ph[i]
        weights = np.exp(-isec_times * c_medium / abs_len)
        for theta in tqdm(thetas, total=len(thetas), leave=False):
            c_weight = cherenkov_ang_dist(np.cos(ph_thetas - theta)) / ANG_DIST_INT * 2
            t, w, ucf = make_data(isec_times, weights * c_weight, det_dist, thr=2)
            obj = make_obj_func(pdf, t, w, 5)
            best_res = fit(wrap_obj_func(obj))

            if best_res is None:
                print(f"Couldn't fit {i}, {theta}")
                continue
            totw = weights * c_weight
            nph_total = totw.sum()

            # Fit arrival positions with FB5
            isec_poss[:, [2, 0]] = isec_poss[:, [0, 2]]
            fb5_pars = fb5_mle(
                isec_poss[:100000], (weights * c_weight)[:100000]
            )  # use at most 100k data points

            fit_results.append(
                {
                    "input": [theta, det_dist],
                    "output_tres": list(best_res[0]) + [ucf, nph_total],
                    "output_arrv_pos": fb5_pars,
                }
            )
    pickle.dump(fit_results, open(args.outfile, "wb"))
