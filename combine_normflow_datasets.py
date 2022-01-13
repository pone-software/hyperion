import pickle
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", "--infiles", dest="infiles", nargs="+", required=True)
parser.add_argument("-o", "--outfile", dest="outfile", required=True)
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
    all_photon_counts.append(np.stack(photon_counts, axis=1))
data = np.hstack(data)
print(all_photon_counts[0].shape)
all_photon_counts = np.stack(all_photon_counts, axis=1)
print(all_photon_counts.shape)
np.savez(args.outfile, times=data, counts=all_photon_counts)
