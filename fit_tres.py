import scipy.optimize
import numpy as np
from tqdm import tqdm, trange
import pickle
from argparse import ArgumentParser

from hyperion.models.photon_arrival_time.pdf import make_gamma_exponential
from hyperion.utils import cherenkov_ang_dist, ANG_DIST_INT

parser = ArgumentParser()
parser.add_argument("-i", "--infile", required=True, dest="infile")
args = parser.parse_args()


n_ph = 1.35
sca_len = 100
c_medium = 0.3 / n_ph
abs_len = 30
r = 0.21

det_ph = pickle.load(open(args.infile, "rb"))


def make_data(t, w, det_dist, thr=2):
    tgeo = t - ((det_dist - r) / c_medium)
    mask = tgeo > thr

    return tgeo[mask] - thr, w[mask], 1 - (w[mask].sum() / w.sum())


def fit(t, w):
    obj, lhfunc = make_gamma_exponential(t, w)

    best_res = None
    for _ in range(8):
        seed = np.random.uniform(0, 1, size=(4,))
        # seed = np.random.uniform(0, 1, size=(6,))

        res = scipy.optimize.fmin_l_bfgs_b(
            obj,
            seed,
            epsilon=0.001,
            bounds=((1e-3, 1), (0.5, None), (1e-3, 0.5), (1e-6, 1 - 1e-6)),
            factr=100,
            approx_grad=True,
        )
        if res[2]["warnflag"] == 2:
            continue
        if (best_res is None) or res[1] < best_res[1]:
            best_res = res
    return best_res, lhfunc


fit_results = []
rstate = np.random.RandomState(0)
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
        best_res, func = fit(t, w)

        if best_res is None:
            print(f"Couldn't fit {i}, {theta}")
            continue
        totw = weights * c_weight
        surv_frac = totw.sum() / (1e7 * 300)

        fit_results.append(
            {"input": [theta, det_dist], "output": list(best_res[0]) + [ucf, surv_frac]}
        )
pickle.dump(fit_results, open("arrival_time_pdf_pars.pickle", "wb"))
