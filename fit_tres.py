"""Script for fitting time residual distributions."""
import pickle
from argparse import ArgumentParser

import numpy as np
import scipy.optimize
from jax.config import config
from tqdm import tqdm, trange

config.update("jax_enable_x64", True)

from hyperion.models.photon_arrival_time.pdf import fb5_mle  # noqa: E402
from hyperion.models.photon_arrival_time.pdf import (  # noqa: E402
    make_exp_exp_exp,
    make_obj_func,
)
from hyperion.utils import (  # noqa: E402
    calc_tres,
    cherenkov_ang_dist,
    cherenkov_ang_dist_int,
    make_cascadia_abs_len_func,
)
from hyperion.propagate import (  # noqa: E402
    cascadia_ref_index_func,
    sca_len_func_antares,
)
from hyperion.constants import Constants  # noqa: E402


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

    det_ph = pickle.load(open(args.infile, "rb"))

    fit_results = []
    rstate = np.random.RandomState(args.seed)

    pdf = make_exp_exp_exp()

    ref_index_func = cascadia_ref_index_func

    def c_medium_f(wl):
        """Speed of light in medium for wl (nm)."""
        return Constants.BaseConstants.c_vac / cascadia_ref_index_func(wl)

    cherenkov_ang = np.arccos(1.0 / ref_index_func(420))
    abs_len = make_cascadia_abs_len_func(sca_len_func_antares)

    for i in trange(len(det_ph)):
        thetas = np.arccos(rstate.uniform(-1, 1, args.n_thetas))
        thetas = np.concatenate(
            [
                thetas,
                [
                    cherenkov_ang - 0.01,
                    cherenkov_ang,
                    cherenkov_ang + 0.01,
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
        wavelengths = sim_data["wavelengths"]

        prop_dist = isec_times * c_medium_f(wavelengths) / 1e9
        abs_weight = np.exp(-prop_dist / abs_len(wavelengths))

        for theta in tqdm(thetas, total=len(thetas), leave=False):

            c_weight = (
                cherenkov_ang_dist(
                    np.cos(ph_thetas - theta), n_ph=ref_index_func(wavelengths)
                )
                / cherenkov_ang_dist_int(ref_index_func(wavelengths), -1, 1)
                * 2
            )
            t, w, ucf = make_data(
                isec_times,
                abs_weight * c_weight,
                det_dist,
                det_radius=args.det_radius,
                c_medium=c_medium_f(700) / 1e9,  # use 700nm as reference
                thr=2,
            )
            obj = make_obj_func(pdf, t, w, 5)
            best_res = fit(wrap_obj_func(obj))

            if best_res is None:
                print(f"Couldn't fit {i}, {theta}")
                continue
            totw = abs_weight * c_weight
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
