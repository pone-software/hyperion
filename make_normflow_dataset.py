import os
import pickle
from argparse import ArgumentParser

import numpy as np
from scipy.stats import qmc
import scipy.stats

from hyperion.constants import Constants
from hyperion.medium import cascadia_ref_index_func, sca_len_func_antares
from hyperion.pmt.pmt import make_calc_wl_acceptance_weight
from hyperion.utils import (
    calc_tres,
    cherenkov_ang_dist,
    cherenkov_ang_dist_int,
    make_cascadia_abs_len_func,
)

ref_index_func = cascadia_ref_index_func
abs_len = make_cascadia_abs_len_func(sca_len_func_antares)
path_to_wl_file = os.path.join(os.path.dirname(__file__), "data/PMTAcc.csv")
wl_acc = make_calc_wl_acceptance_weight(path_to_wl_file)


def c_medium_f(wl):
    """Speed of light in medium for wl (nm)."""
    return Constants.BaseConstants.c_vac / cascadia_ref_index_func(wl)


def make_dataset(files, seed, tt=4, tts=1.45):

    rstate = np.random.RandomState(seed)

    all_times = []
    all_dists = []
    all_angls = []
    all_nphotons_frac = []

    for file in files:
        data = pickle.load(open(file, "rb"))
        sampler = qmc.Sobol(d=1, scramble=True, seed=rstate)

        # npick = min(dlim, len(data[0]["times_det"]))
        # ixs = rstate.choice(np.arange(len(data[0]["times_det"])), size=npick, replace=False)
        sim_data = data[0]
        det_dist = sim_data["dist"]
        isec_times = sim_data["times_det"]
        ph_thetas = sim_data["emission_angles"]
        # stepss = sim_data["photon_steps"]
        # isec_poss = sim_data["positions_det"]
        nphotons_sim = sim_data["nphotons_sim"]
        wavelengths = sim_data["wavelengths"]

        prop_dist = isec_times * c_medium_f(wavelengths) / 1e9
        abs_weight = np.exp(-prop_dist / abs_len(wavelengths))
        wl_weight = wl_acc(wavelengths, 0.27)
        tres = calc_tres(isec_times, 0.21, det_dist, c_medium_f(700) / 1e9)

        costhetas = 2 * sampler.random_base2(m=6) - 1

        obs_angs = np.arccos(costhetas)

        for obs_ang in obs_angs:
            c_weight = (
                cherenkov_ang_dist(
                    np.cos(ph_thetas - obs_ang), n_ph=ref_index_func(wavelengths)
                )
                / cherenkov_ang_dist_int(ref_index_func(wavelengths), -1, 1)
                * 2
            )

            tot_weight = abs_weight * c_weight * wl_weight

            sum_w = tot_weight.sum()

            ixs = np.arange(len(tres))
            surv_ph = rstate.choice(
                ixs, p=tot_weight / sum_w, size=rstate.poisson(sum_w)
            )

            times = tres[surv_ph]

            if tts > 0:

                a = tt ** 2 / tts ** 2
                b = tts ** 2 / tt
                pdf = scipy.stats.gamma(a, scale=b)

                dt = pdf.rvs(size=len(surv_ph), random_state=rstate) - tt
                times += dt
            all_times.append(times)
            all_dists.append(np.ones_like(surv_ph) * det_dist)
            all_angls.append(np.ones_like(surv_ph) * obs_ang)

            all_nphotons_frac.append([det_dist, obs_ang, sum_w / nphotons_sim])

    return (
        np.vstack(
            [
                np.concatenate(all_times),
                # np.concatenate(all_weights),
                np.concatenate(all_dists),
                np.concatenate(all_angls),
            ]
        ),
        all_nphotons_frac,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--infile", dest="infile", required=True)
    parser.add_argument("-o", "--outfile", dest="outfile", required=True)
    parser.add_argument("--tts", dest="tts", default=0, type=float)
    parser.add_argument("-s", "--seed", dest="seed", default=0, type=int)
    args = parser.parse_args()
    data = make_dataset([args.infile], seed=args.seed, tts=args.tts)
    pickle.dump(data, open(args.outfile, "wb"))


if __name__ == "__main__":
    main()
