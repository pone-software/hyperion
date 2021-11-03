import jax.numpy as jnp
from jax import jit, vmap, random
from hyperion.propagate import (
    photon_sphere_intersection,
    mixed_hg_rayleigh,
    make_step_function,
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
fun = make_step_function(
    1 / medium["sca_len"],
    c_medium,
    det_pos,
    det_radius,
    medium["abs_len"] * 15,  # maximum propagation distance for a photon
    emitter_x,
    emitter_t,
    mode="uniform",
    intersection_f=photon_sphere_intersection,
    scattering_function=mixed_hg_rayleigh,
)
make_n_steps = jit(vmap(fun, in_axes=[0]))

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
) = make_n_steps(random.split(key, num=nphotons))
