"""Example script showing how to propagate photons."""
import json

import jax.numpy as jnp
from hyperion.constants import Constants
from hyperion.propagate import (
    cascadia_ref_index_func,
    initialize_direction_isotropic,
    make_cherenkov_spectral_sampling_func,
    make_fixed_pos_time_initializer,
    make_loop_until_isec_or_maxtime,
    make_photon_sphere_intersection_func,
    make_photon_trajectory_fun,
    make_step_function,
    mixed_hg_rayleigh_antares,
    sca_len_func_antares,
    wl_mono_400nm_init,
)
from jax import jit, random, vmap

# Define emitter position & emission time
emitter_x = jnp.array([0, 0, 0.0])
emitter_t = 0.0

det_dist = 5.0
# Define detector position & sensor radius (sensor is modelled as a sphere)
det_pos = jnp.array([0, 0, det_dist])
det_radius = 0.21

# Isotropic photon emission
emission_dir_init = initialize_direction_isotropic

# Sample wavelength from cherenkov spectrum in (300, 700) nm
wavelength_init = make_cherenkov_spectral_sampling_func(
    [300, 700], cascadia_ref_index_func
)

photon_init = make_fixed_pos_time_initializer(
    emitter_x, emitter_t, emission_dir_init, wavelength_init
)

# We want to calculate intersections with a sphere
intersection_f = make_photon_sphere_intersection_func(det_pos, det_radius)

# Use an approximation to the ANTARES scattering function
scattering_function = mixed_hg_rayleigh_antares

# Use the ANTARES wavelength dependent scattering length
scattering_length_function = sca_len_func_antares

# Use the ANTARES wavelength dependent refractive index with cascadia basin properties
ref_index_func = cascadia_ref_index_func

# Create a step function, which will propagate the photon until the next scattering site.
# If the intersection function returns True, only propagate to intersection position.
step_fun = make_step_function(
    intersection_f=intersection_f,
    scattering_function=scattering_function,
    scattering_length_function=scattering_length_function,
    ref_index_func=ref_index_func,
)


# Get speed of light in medium at 400nm
c_medium = Constants.BaseConstants.c_vac / cascadia_ref_index_func(400)

# Set the maximum time a photon is tracked. Here max time residual=700ns
# Note that the runtime scales with the number of scattering steps.
# For small scattering lengths, the step size can become very small
max_time = jnp.linalg.norm(emitter_x - det_pos) / c_medium + 700

# Propagation is run until either photon intersects or max_time is reached
loop_func = make_loop_until_isec_or_maxtime(max_time)

# make a trajectory function
trajec_fun = make_photon_trajectory_fun(
    step_fun,
    photon_init,
    loop_func=loop_func,
)

# Create a compiled and vectorized trajectory function
trajec_fun_v = jit(vmap(trajec_fun, in_axes=[0]))


key = random.PRNGKey(0)

isec_times = []
ph_thetas = []
stepss = []
nphotons = int(1e7)
isec_poss = []

photon_state = trajec_fun_v(random.split(key, num=nphotons))
