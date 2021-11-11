import numpy as np
import pickle
import json
from argparse import ArgumentParser
from hyperion.utils import cherenkov_ang_dist, ANG_DIST_INT, calc_tres


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
    det_dist, isec_times, ph_thetas, stepss, isec_poss = det_ph[0]

    rstate = np.random.RandomState(args.seed)

    obs_angs = np.arccos(rstate.uniform(-1, 1, size=args.n_thetas))
    tres = calc_tres(isec_times, 0.21, det_dist, c_medium)
    weights = np.exp(-isec_times * c_medium / medium["abs_len"])

    hists = []
    inp_data = []

    for obs in obs_angs:
        c_weight = cherenkov_ang_dist(np.cos(ph_thetas - obs)) / ANG_DIST_INT * 2
        tot_weight = weights * c_weight
        hist, _ = np.histogram(tres, weights=tot_weight, bins=np.linspace(0, 500, 500))
        hists.append(hist)
        inp_data.append([obs, det_dist])

    pickle.dump([inp_data, hists], open(args.outfile, "wb"))
