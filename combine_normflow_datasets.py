import pickle
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", "--infiles", dest="infiles", nargs="+", required=True)
parser.add_argument("-o", "--outfile", dest="outfile", required=True)
args = parser.parse_args()

data = []
for f in args.infiles:
    data.append(pickle.load(open(f, "rb")))
data = np.hstack(data)
pickle.dump(data, open(args.outfile, "wb"))
