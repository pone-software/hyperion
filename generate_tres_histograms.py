import json
import pickle
from argparse import ArgumentParser

import numpy as np
import scipy.stats
from hyperion.utils import ANG_DIST_INT, calc_tres, cherenkov_ang_dist

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
    parser.add_argument(
        "--tts",
        type=float,
        default=2,
        dest="tts",
    )
    args = parser.parse_args()

    medium = json.load(open(args.medium))
    c_medium = 0.299792458 / medium["n_ph"]

    det_ph = pickle.load(open(args.infile, "rb"))
    hists = []
    inp_data = []

    binning = np.linspace(-30, 500, 530)

    for i in range(len(det_ph)):
        sim_data = det_ph[i]
        det_dist = sim_data["dist"]
        isec_times = sim_data["times_det"]
        ph_thetas = sim_data["emission_angles"]
        stepss = sim_data["photon_steps"]
        isec_poss = sim_data["positions_det"]
        nphotons_sim = sim_data["nphotons_sim"]

        rstate = np.random.RandomState(args.seed)

        obs_angs = np.arccos(rstate.uniform(-1, 1, size=args.n_thetas))
        tres = calc_tres(isec_times, 0.21, det_dist, c_medium)

        """
        if args.tts > 0:
            tres += rstate.normal(0, scale=args.tts, size=tres.shape[0])
        """

        weights = np.exp(-isec_times * c_medium / medium["abs_len"])

        for obs in obs_angs:
            c_weight = cherenkov_ang_dist(np.cos(ph_thetas - obs)) / ANG_DIST_INT * 2
            tot_weight = weights * c_weight / nphotons_sim

            if args.tts > 0:
                split_len = 1e6
                splits = np.ceil(len(tres) / split_len)

                eval_cdf = 0
                for nsplit in range(splits):
                    this_slice = slice(nsplit * split_len, (nsplit + 1) * split_len)
                    dist = scipy.stats.norm(tres[this_slice], args.tts)
                    eval_cdf += (
                        dist.cdf(binning[:, np.newaxis]) * tot_weight[this_slice]
                    ).sum(axis=1)
                hist = eval_cdf.diff()
            else:

                hist, _ = np.histogram(tres, weights=tot_weight, bins=binning)
            hists.append(hist)
            inp_data.append([obs, det_dist])

    pickle.dump([inp_data, hists], open(args.outfile, "wb"))
