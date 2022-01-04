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
for f in args.infiles:
    indata =  pickle.load(open(f, "rb"))
    size = int(min(5E6, indata.shape[1]))
    choice = rstate.choice(np.arange(indata.shape[1]), size=size, replace=False)
    data.append(indata[:, choice])
data = np.hstack(data)

np.savez(args.outfile, data)
