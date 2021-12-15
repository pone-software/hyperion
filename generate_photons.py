"""Generate photons for a range of emitter-reciever distances."""
from jax.config import config

config.update("jax_enable_x64", True)

import pickle
import jax
import jax.numpy as jnp
import numpy as np

from argparse import ArgumentParser
from scipy.stats import qmc
from tqdm import tqdm
from jax import jit, vmap

from hyperion.propagate import (
    collect_hits,
    initialize_direction_isotropic,
    make_cherenkov_spectral_sampling_func,
    make_fixed_pos_time_initializer,
    make_loop_for_n_steps,
    make_photon_sphere_intersection_func,
    make_photon_trajectory_fun,
    make_step_function,
)

from hyperion.medium import (
    mixed_hg_rayleigh_antares,
    cascadia_ref_index_func,
    sca_len_func_antares,
)
from hyperion.utils import calculate_min_number_steps


parser = ArgumentParser()
parser.add_argument("-o", "--outfile", required=True, dest="outfile")
parser.add_argument("-s", "--seed", type=int, required=False, dest="seed", default=0)
parser.add_argument(
    "-d", "--dist", type=float, required=False, dest="dist", default=None
)
parser.add_argument(
    "--photons_per_batch", type=float, required=False, dest="ph_per_batch", default=1e7
)
parser.add_argument(
    "--n_photon_batches", type=int, required=False, dest="n_ph_batches", default=5000
)
parser.add_argument(
    "--max_dist", type=float, required=False, dest="max_dist", default=500
)
parser.add_argument(
    "--min_dist", type=float, required=False, dest="min_dist", default=1
)
parser.add_argument(
    "-r", "--det-radius", type=float, required=True, dest="det_radius", default=None
)
args = parser.parse_args()

if jax.default_backend() == "cpu":
    raise RuntimeError("Running on CPU. Bailing...")


outfile = open(args.outfile, "wb")
outfile.close()

emitter_x = jnp.array([0, 0, 0.0])
emitter_t = 0.0

wavelength_init = make_cherenkov_spectral_sampling_func(
    [300, 700], cascadia_ref_index_func
)
photon_init = make_fixed_pos_time_initializer(
    emitter_x, emitter_t, initialize_direction_isotropic, wavelength_init
)

if args.dist is None:
    # Use quasi-random numbers to select random distances with optimal coverage
    # in log space.
    sampler = qmc.Sobol(d=1, scramble=False)
    sample = sampler.random_base2(m=1)
    dists = (
        10 ** qmc.scale(sample, np.log10(args.min_dist), np.log10(args.max_dist))
    ).squeeze()
else:
    dists = [args.dist]

training_data = []
all_data = []
for det_dist in tqdm(dists, total=len(dists), disable=True):
    det_pos = jnp.array([0, 0, det_dist])

    intersection_f = make_photon_sphere_intersection_func(det_pos, args.det_radius)

    step_fun = make_step_function(
        intersection_f=intersection_f,
        scattering_function=mixed_hg_rayleigh_antares,
        scattering_length_function=sca_len_func_antares,
        ref_index_func=cascadia_ref_index_func,
    )

    n_steps = calculate_min_number_steps(
        cascadia_ref_index_func, sca_len_func_antares, det_dist, 500, 300, 0.01
    )

    loop_func = make_loop_for_n_steps(n_steps)

    trajec_fun = make_photon_trajectory_fun(
        step_fun,
        photon_init,
        loop_func=loop_func,
    )
    trajec_fun_v = jit(vmap(trajec_fun, in_axes=[0]))

    times, arrival_angles, emission_angles, steps, positions, sims_cnt, wavelengths = collect_hits(
        trajec_fun_v, args.ph_per_batch, args.n_ph_batches, args.seed
    )

    all_data.append(
        {
            "dist": det_dist,
            "times_det": times,
            "arrival_angles":, arrival_angles,
            "emission_angles": emission_angles,
            "photon_steps": steps,
            "positions_det": positions,
            "nphotons_sim": sims_cnt * args.ph_per_batch,
            "wavelengths": wavelengths,
        }
    )


with open(args.outfile, "wb") as outfile:
    pickle.dump(all_data, outfile)
