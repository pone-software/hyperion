"""Script for fitting time residual distributions."""
import json
import pickle
from argparse import ArgumentParser

import numpy as np
import scipy.optimize
from jax.config import config
from tqdm import tqdm, trange

config.update("jax_enable_x64", True)

from hyperion.models.photon_arrival_time.pdf import fb5_mle  # noqa: E402
from hyperion.models.photon_arrival_time.pdf import make_exp_exp_exp, make_obj_func
from hyperion.utils import ANG_DIST_INT, calc_tres, cherenkov_ang_dist  # noqa: E402


def make_data(t, w, det_dist, det_radius, c_medium, thr=2):
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
    parser.add_argument(
        "-r", "--det-radius", type=float, required=True, dest="det_radius", default=None
    )
    parser.add_argument(
        "-m",
        "--medium-file",
        type=str,
        required=True,
        dest="medium",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        required=True,
        dest="seed",
    )
    parser.add_argument(
        "--n-thetas",
        type=int,
        default=100,
        dest="n_thetas",
    )
    args = parser.parse_args()

    medium = json.load(open(args.medium))
    c_medium = 0.299792458 / medium["n_ph"]

    det_ph = pickle.load(open(args.infile, "rb"))

    fit_results = []
    rstate = np.random.RandomState(args.seed)

    pdf = make_exp_exp_exp()

    for i in trange(len(det_ph)):
        thetas = np.arccos(rstate.uniform(-1, 1, args.n_thetas))
        thetas = np.concatenate(
            [
                thetas,
                [
                    np.arccos(1 / medium["n_ph"]) - 0.01,
                    np.arccos(1 / medium["n_ph"]),
                    np.arccos(1 / medium["n_ph"]) + 0.01,
                ],
            ]
        )

        sim_data = det_ph[i]
        det_dist = sim_data["dist"]
        isec_times = sim_data["times_det"]
        ph_thetas = sim_data["emission_angles"]
        stepss = sim_data["photon_steps"]
        isec_poss = sim_data["positions_det"]
        nphotons_sim = sim_data["nphotons_sim"]

        weights = np.exp(-isec_times * c_medium / medium["abs_len"])
        for theta in tqdm(thetas, total=len(thetas), leave=False):
            c_weight = cherenkov_ang_dist(np.cos(ph_thetas - theta)) / ANG_DIST_INT * 2
            t, w, ucf = make_data(
                isec_times,
                weights * c_weight,
                det_dist,
                det_radius=args.det_radius,
                c_medium=c_medium,
                thr=2,
            )
            obj = make_obj_func(pdf, t, w, 5)
            best_res = fit(wrap_obj_func(obj))

            if best_res is None:
                print(f"Couldn't fit {i}, {theta}")
                continue
            totw = weights * c_weight
            nph_total = totw.sum()

            detected_fraction = nph_total / nphotons_sim

            # Calculate photon arrival coordinates relative to module center
            det_center = np.asarray([0, 0, det_dist])
            rel = isec_poss - det_center
            rel = rel / np.linalg.norm(rel, axis=1)[:, np.newaxis]

            # Fit arrival positions with FB5
            rel[:, [2, 0]] = rel[:, [0, 2]]
            idx = np.random.choice(
                np.arange(rel.shape[0]),
                size=min(rel.shape[0], 100000),
                replace=False,
            )
            try:
                fb5_pars = fb5_mle(rel[idx], totw[idx])  # use at most 100k data points
            except AssertionError:
                print(isec_poss[idx])
                continue

            fit_results.append(
                {
                    "input": [theta, det_dist],
                    "output_tres": list(best_res[0]) + [ucf, detected_fraction],
                    "output_arrv_pos": fb5_pars,
                }
            )
    pickle.dump(fit_results, open(args.outfile, "wb"))
