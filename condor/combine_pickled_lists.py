import pickle
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", dest="infiles", nargs="+", required=True)
parser.add_argument("-o", dest="outfile", required=True)

args = parser.parse_args()

data = []
for f in args.infiles:
    with open(f, "rb") as hdl:
        data += pickle.load(hdl)
pickle.dump(data, open(args.outfile, "wb"))
