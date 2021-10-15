"""Generate photons for a range of emitter-reciever distances."""
import jax.numpy as jnp
from jax import jit, vmap
from hyperion.propagate import (
    photon_sphere_intersection,
    mixed_hg_rayleigh,
    make_step_function,
    collect_hits,
)
from tqdm import tqdm
from scipy.stats import qmc
import numpy as np
import pickle
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-o", "--outfile", required=True, dest="outfile")
parser.add_argument("-s", "--seed", type=int, required=False, dest="seed", default=0)
parser.add_argument("-d", "--dist", type=float, required=False, dest="dist", default=None)
args = parser.parse_args()

outfile = open(args.outfile, "wb")
outfile.close()
sca_len = 100
c_medium = 0.3 / 1.35
abs_len = 30
r = 0.21
emitter_x = jnp.array([0, 0, 0.0])
emitter_t = 0.0
emitter_dir = jnp.array([0, 0, 1.0])

if args.dist is None:
    sampler = qmc.Sobol(d=1, scramble=False)
    sample = sampler.random_base2(m=1)
    dists = (10 ** qmc.scale(sample, -1, np.log10(500))).squeeze()
else:
    dists = [args.dist]

training_data = []
all_data = []
for det_dist in tqdm(dists, total=len(dists), disable=True):
    det_pos = jnp.array([0, 0, det_dist])

    fun = make_step_function(
        1 / sca_len,
        c_medium,
        det_pos,
        r,
        abs_len * 15,
        emitter_x,
        emitter_t,
        mode="uniform",
        intersection_f=photon_sphere_intersection,
        scattering_function=mixed_hg_rayleigh,
    )
    make_n_steps = jit(vmap(fun, in_axes=[0]))

    isec_times, ph_thetas, stepss, isec_poss = collect_hits(make_n_steps, 1e7, 300, args.seed)
    all_data.append([det_dist, isec_times, ph_thetas, stepss, isec_poss])


with open(args.outfile, "wb") as outfile:
    pickle.dump(all_data, outfile)
