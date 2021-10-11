import jax.numpy as jnp
from jax import jit, vmap
from hyperion.propagate import (
    photon_sphere_intersection,
    mixed_hg_rayleigh,
    make_step_function,
    collect_hits,
)

sca_len = 100
c_medium = 0.3 / 1.35
abs_len = 30
r = 0.21
det_pos = jnp.array([0, 0, 100])
emitter_x = jnp.array([0, 0, 0.0])
emitter_t = 0.0

fun = make_step_function(
    1 / sca_len,
    c_medium,
    det_pos,
    r,
    abs_len * 10,
    emitter_x,
    emitter_t,
    mode="uniform",
    intersection_f=photon_sphere_intersection,
    scattering_function=mixed_hg_rayleigh,
)
make_n_steps = jit(vmap(fun, in_axes=[0]))
isec_times, ph_thetas, stepss, isec_poss = collect_hits(make_n_steps, 1e7, 100)
