import pickle
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", "--infiles", dest="infiles", nargs="+", required=True)
parser.add_argument("--outfile_times", dest="outfile_data", required=True)
parser.add_argument("--outfile_counts", dest="outfile_counts", required=True)
parser.add_argument("-s", "--seed", dest="seed", type=int, required=True)
args = parser.parse_args()

rstate = np.random.RandomState(args.seed)

data = []
all_photon_counts = []
for f in args.infiles:
    indata, photon_counts = pickle.load(open(f, "rb"))
    size = int(min(5e6, indata.shape[1]))
    choice = rstate.choice(np.arange(indata.shape[1]), size=size, replace=False)
    data.append(indata[:, choice])
    all_photon_counts.append(np.concatenate(photon_counts))
data = np.hstack(data)

np.savez(args.outfile_data, data)
pickle.dump(np.concatenate(all_photon_counts), open(args.outfile_counts, "wb"))
