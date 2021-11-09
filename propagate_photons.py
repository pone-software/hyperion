"""Example script showing how to propagate photons."""
import jax.numpy as jnp
from jax import jit, vmap, random
from hyperion.propagate import (
    photon_sphere_intersection,
    mixed_hg_rayleigh_antares,
    make_step_function,
    make_photon_trajectory_fun,
)
import json

# Load medium properties
medium = json.load(open("resources/medium.json"))
c_medium = 0.299792458 / medium["n_ph"]

# Define emitter position & emission time
emitter_x = jnp.array([0, 0, 0.0])
emitter_t = 0.0

# Define detector position & sensor radius (sensor is modelles as a sphere)
det_pos = jnp.array([0, 0, 100.0])
det_radius = 0.21

# The emission direction is not used when photons are emitted isotropically
emitter_dir = jnp.array([0, 0, 1.0])

# Make a step function

step_fun = make_step_function(
    1 / medium["sca_len"],
    c_medium,
    det_pos,
    det_radius,
    intersection_f=photon_sphere_intersection,
    scattering_function=mixed_hg_rayleigh_antares,
)

# Set the maximum time a photon is tracked. Here max time residual=700ns
max_time = jnp.linalg.norm(emitter_x - det_pos) / c_medium + 700


# make a trajectory function
trajec_fun = make_photon_trajectory_fun(
    step_fun,
    emitter_x,
    emitter_t,
    max_time,
    emission_mode="uniform",
    stepping_mode="until_intersect",
)

trajec_fun_v = jit(vmap(trajec_fun, in_axes=[0]))


key = random.PRNGKey(0)

isec_times = []
ph_thetas = []
stepss = []
nphotons = int(1e7)
isec_poss = []

(
    positions,
    times,
    directions,
    has_intersected,
    intersection_position,
    steps,
) = trajec_fun_v(random.split(key, num=nphotons))
