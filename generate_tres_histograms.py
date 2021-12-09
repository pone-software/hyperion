import pickle
import os
from argparse import ArgumentParser

import numpy as np
import scipy.stats
from hyperion.utils import (
    calc_tres,
    cherenkov_ang_dist,
    cherenkov_ang_dist_int,
    make_cascadia_abs_len_func,
)
from hyperion.pmt.pmt import make_calc_wl_acceptance_weight
from hyperion.propagate import cascadia_ref_index_func, sca_len_func_antares
from hyperion.constants import Constants

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
    parser.add_argument(
        "--tts",
        type=float,
        default=2,
        dest="tts",
    )
    args = parser.parse_args()

    det_ph = pickle.load(open(args.infile, "rb"))
    hists = []
    inp_data = []

    binning = np.linspace(-30, 500, 530)

    ref_index_func = cascadia_ref_index_func
    abs_len = make_cascadia_abs_len_func(sca_len_func_antares)
    path_to_wl_file = os.path.join(os.path.dirname(__file__), "data/DOMEfficiency.dat")
    wl_acc = make_calc_wl_acceptance_weight(path_to_wl_file)

    def c_medium_f(wl):
        """Speed of light in medium for wl (nm)."""
        return Constants.BaseConstants.c_vac / cascadia_ref_index_func(wl)

    for i in range(len(det_ph)):
        sim_data = det_ph[i]
        det_dist = sim_data["dist"]
        isec_times = sim_data["times_det"]
        ph_thetas = sim_data["emission_angles"]
        stepss = sim_data["photon_steps"]
        isec_poss = sim_data["positions_det"]
        nphotons_sim = sim_data["nphotons_sim"]
        wavelengths = sim_data["wavelengths"]

        rstate = np.random.RandomState(args.seed)

        obs_angs = np.arccos(rstate.uniform(-1, 1, size=args.n_thetas))
        prop_dist = isec_times * c_medium_f(wavelengths) / 1e9
        abs_weight = np.exp(-prop_dist / abs_len(wavelengths))
        wl_weight = wl_acc(wavelengths, 0.28)

        # For time residual use 700nm as reference
        tres = calc_tres(isec_times, args.det_radius, det_dist, c_medium_f(700) / 1e9)

        """
        if args.tts > 0:
            tres += rstate.normal(0, scale=args.tts, size=tres.shape[0])
        """
        for obs in obs_angs:
            c_weight = (
                cherenkov_ang_dist(
                    np.cos(ph_thetas - obs), n_ph=ref_index_func(wavelengths)
                )
                / cherenkov_ang_dist_int(ref_index_func(wavelengths), -1, 1)
                * 2
            )
            tot_weight = abs_weight * c_weight * wl_weight / nphotons_sim

            if args.tts > 0:
                split_len = int(1e5)
                splits = int(np.ceil(len(tres) / split_len))
                eval_cdf = 0
                for nsplit in range(splits):
                    this_slice = slice(nsplit * split_len, (nsplit + 1) * split_len)
                    dist = scipy.stats.norm(tres[this_slice], args.tts)
                    eval_cdf += (
                        dist.cdf(binning[:, np.newaxis]) * tot_weight[this_slice]
                    ).sum(axis=1)
                hist = np.diff(eval_cdf)
            else:

                hist, _ = np.histogram(tres, weights=tot_weight, bins=binning)

            hists.append(hist)
            inp_data.append([obs, det_dist])

    pickle.dump([inp_data, hists], open(args.outfile, "wb"))
