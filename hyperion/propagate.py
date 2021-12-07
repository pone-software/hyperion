"""Implementation of the photon propagation code."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.lax import cond, fori_loop, while_loop
from scipy.integrate import quad

from .constants import Constants


def sph_to_cart(theta, phi=0, r=1):
    """Transform spherical to cartesian coordinates."""
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)

    return jnp.array([x, y, z])


def make_photon_sphere_intersection_func(target_x, target_r):
    """
    Make a function that calculates the intersection of a photon path with a sphere.

    Parameters:
        target_x: float[3]
        target_r: float
    """

    @functools.partial(
        jax.profiler.annotate_function, name="photon_sphere_intersection"
    )
    def photon_sphere_intersection(photon_x, photon_p, step_size):
        """
        Calculate intersection.

        Given a photon origin, a photon direction, a step size, a target location and a target radius,
        calculate whether the photon intersects the target and the intrsection point.

        Parameters:
            photon_x: float[3]
            photon_p: float[3]
            step_size: float

        Returns:
            tuple(bool, float[3])
                True and intersection position if intersected.
        """
        p_normed = photon_p  # assume normed

        a = jnp.dot(p_normed, (photon_x - target_x))
        b = a ** 2 - (jnp.linalg.norm(photon_x - target_x) ** 2 - target_r ** 2)
        # Distance of of the intersection point along the line
        d = -a - jnp.sqrt(b)

        isected = (b >= 0) & (d > 0) & (d < step_size)

        # need to check intersection here, otherwise nan-gradients (sqrt(b) if b < 0)
        result = cond(
            isected,
            lambda _: (True, photon_x + d * p_normed),
            lambda _: (False, jnp.ones(3) * 1e8),
            0,
        )

        return result

    return photon_sphere_intersection


def photon_plane_intersection(photon_x, photon_p, target_x, target_r, step_size):
    """
    Intersection of line and plane.

    iven a photon origin, a photon direction, a step size, a target location and a target radius,
    calculate whether the photon intersects the target and the intersection point.

    Parameters:
        photon_x: float[3]
        photon_p: float[3]
        target_x: float[3]
        target_r: float
        step_size: float

    Returns:
        tuple(bool, float[3])
            True and intersection position if intersected.
    """
    # assume plane normal vector is e_z

    plane_normal = jnp.array([0, 0, 1])

    p_n = jnp.dot(photon_p, plane_normal)
    d = jnp.dot((target_x[2] - photon_x[2]), plane_normal) / p_n
    isec_p = photon_x + d * photon_p
    result = cond(
        (p_n != 0)
        & (d > 0)
        & (d <= step_size)
        & (jnp.all(jnp.abs((isec_p - target_x)[:2]) < target_r)),
        lambda _: (True, isec_p),
        lambda _: (False, jnp.ones(3) * 1e8),
        None,
    )
    return result


@functools.partial(
    jax.profiler.annotate_function, name="henyey_greenstein_scattering_angle"
)
def henyey_greenstein_scattering_angle(key, g=0.9):
    """Henyey-Greenstein scattering in one plane."""
    eta = random.uniform(key)
    costheta = (
        1 / (2 * g) * (1 + g ** 2 - ((1 - g ** 2) / (1 + g * (2 * eta - 1))) ** 2)
    )
    return jnp.arccos(costheta)


def rayleigh_scattering_angle(key):
    """Rayleigh scattering. Adapted from clsim."""
    b = 0.835
    p = 1.0 / 0.835

    q = (b + 3.0) * ((random.uniform(key)) - 0.5) / b
    d = q * q + p * p * p

    u1 = -q + jnp.sqrt(d)
    u = jnp.cbrt(jnp.abs(u1)) * jnp.sign(u1)

    v1 = -q - jnp.sqrt(d)
    v = jnp.cbrt(jnp.abs(v1)) * jnp.sign(v1)

    return jnp.arccos(jax.lax.clamp(-1.0, u + v, 1.0))


def liu_scattering_angle(key, g=0.95):
    """
    Simplified liu scattering.

    https://arxiv.org/pdf/1301.5361.pdf
    """
    beta = (1 - g) / (1 + g)
    xi = random.uniform(key)
    costheta = 2 * xi ** beta - 1
    return jnp.arccos(costheta)


def make_mixed_scattering_func(f1, f2, ratio):
    """
    Create a mixture model with two sampling functions.

    Paramaters:
        f1, f2: functions
            Sampling functions taking one argument (random key)
        ratio: float
            Fraction of samples drawn from f1
    """

    def _f(keys):
        k1, k2 = keys
        is_f1 = random.uniform(k1) < ratio

        return cond(is_f1, f1, f2, k2)

    return _f


"""Mix of HG and Rayleigh. Distribution similar to ANTARES Petzold+Rayleigh."""
mixed_hg_rayleigh_antares = make_mixed_scattering_func(
    rayleigh_scattering_angle,
    lambda k: henyey_greenstein_scattering_angle(k, 0.97),
    0.15,
)

"""Mix of HG and Liu. IceCube"""
mixed_hg_liu_icecube = make_mixed_scattering_func(
    lambda k: liu_scattering_angle(k, 0.95),
    lambda k: henyey_greenstein_scattering_angle(k, 0.95),
    0.35,
)


def make_wl_dep_sca_len_func(vol_conc_small_part, vol_conc_large_part):
    """Make a function that calculates the scattering length based on particle concentrations."""

    @functools.partial(jax.profiler.annotate_function, name="sca_len")
    def sca_len(wavelength):
        ref_wlen = 550  # nm
        x = ref_wlen / wavelength

        sca_coeff = (
            0.0017 * jnp.power(x, 4.3)
            + 1.34 * vol_conc_small_part * jnp.power(x, 1.7)
            + 0.312 * vol_conc_large_part * jnp.power(x, 0.3)
        )

        return 1 / sca_coeff

    return sca_len


sca_len_func_antares = make_wl_dep_sca_len_func(0.0075e-6, 0.0075e-6)


def make_ref_index_func(salinity, temperature, pressure):
    """
    Make function that returns refractive index as function of wavelength.

    Parameters:
        salinity: float
            Salinity in parts per thousand
        temperature: float
            Temperature in C
        pressure: float
            Pressure in bar
    """
    n0 = 1.31405
    n1 = 1.45e-5
    n2 = 1.779e-4
    n3 = 1.05e-6
    n4 = 1.6e-8
    n5 = 2.02e-6
    n6 = 15.868
    n7 = 0.01155
    n8 = 0.00423
    n9 = 4382
    n10 = 1.1455e6

    a01 = (
        n0
        + (n2 - n3 * temperature + n4 * temperature * temperature) * salinity
        - n5 * temperature * temperature
        + n1 * pressure
    )
    a2 = n6 + n7 * salinity - n8 * temperature
    a3 = -n9
    a4 = n10

    @functools.partial(jax.profiler.annotate_function, name="ref_index")
    def ref_index_func(wavelength):

        x = 1 / wavelength
        return a01 + x * (a2 + x * (a3 + x * a4))

    return ref_index_func


antares_ref_index_func = make_ref_index_func(
    pressure=215.82225 / 1.01325, temperature=13.1, salinity=38.44
)

cascadia_ref_index_func = make_ref_index_func(
    pressure=269.44088 / 1.01325, temperature=1.8, salinity=34.82
)


def frank_tamm(wavelength, ref_index_func):
    """Frank Tamm Formula."""
    return (
        4
        * np.pi ** 2
        * Constants.BaseConstants.e ** 2
        / (
            Constants.BaseConstants.h
            * Constants.BaseConstants.c_vac
            * (wavelength / 1e9) ** 2
        )
        * (1 - 1 / ref_index_func(wavelength) ** 2)
    )


def make_cherenkov_spectral_sampling_func(wl_range, ref_index_func):
    """
    Make a sampling function that samples from the Frank-Tamm formula in a given wavelength range.

    Parameters:
        wl_range: Tuple
            lower and upper wavelength range
        ref_index_func: function
            Function returning wavelength dependent refractive index

    """
    wls = np.linspace(wl_range[0], wl_range[1], 1000)

    integral = lambda upper: quad(  # noqa E731
        functools.partial(frank_tamm, ref_index_func=ref_index_func), wl_range[0], upper
    )[0]
    norm = integral(wl_range[-1])
    poly_pars = np.polyfit(np.vectorize(integral)(wls) / norm, wls, 10)

    @functools.partial(jax.profiler.annotate_function, name="frank_tamm")
    def sampling_func(rng_key):
        uni = random.uniform(rng_key)
        return jnp.polyval(poly_pars, uni)

    return sampling_func


def calc_new_direction(keys, old_dir, scattering_function):
    """
    Calculate new direction after sampling a scattering angle.

    Scattering is calculated in a reference frame local
    to the photon (e_z) and then rotated back to the global coordinate system.
    """

    theta = scattering_function(keys[0:2])
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    phi = random.uniform(keys[2], minval=0, maxval=2 * np.pi)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    px, py, pz = old_dir

    is_para_z = jnp.abs(pz) == 1

    new_dir = cond(
        is_para_z,
        lambda _: jnp.array(
            [
                sin_theta * cos_phi,
                jnp.sign(pz) * sin_theta * sin_phi,
                jnp.sign(pz) * cos_theta,
            ]
        ),
        lambda _: jnp.array(
            [
                (px * cos_theta)
                + (
                    (sin_theta * (px * pz * cos_phi - py * sin_phi))
                    / (jnp.sqrt(1.0 - pz ** 2))
                ),
                (py * cos_theta)
                + (
                    (sin_theta * (py * pz * cos_phi + px * sin_phi))
                    / (jnp.sqrt(1.0 - pz ** 2))
                ),
                (pz * cos_theta) - (sin_theta * cos_phi * jnp.sqrt(1.0 - pz ** 2)),
            ]
        ),
        None,
    )

    # Need this for numerical stability?
    new_dir = new_dir / jnp.linalg.norm(new_dir)

    return new_dir


def make_step_function(
    intersection_f,
    scattering_function,
    scattering_length_function,
    ref_index_func,
):
    """
    Make a photon step function object.

    Returns a function f(photon_state, key) that performs a photon step
    and returns the new photon state.

    Parameters:
        intersection_f: function
            function used to calculate the intersection
        scattering_function: function
            rng function drawing angles from scattering function
        scattering_length_function: function
            function that returns scattering length as function of wavelength
        ref_index_func: function
            function that returns the refractive index as function of wavelength
    """

    @functools.partial(jax.profiler.annotate_function, name="step_function")
    def step(photon_state, rng_key):
        """Single photon step."""
        pos = photon_state["pos"]
        dir = photon_state["dir"]
        time = photon_state["time"]
        isec = photon_state["isec"]
        stepcnt = photon_state["stepcnt"]
        wavelength = photon_state["wavelength"]

        # TODO: make this more flexible
        keys = random.split(rng_key, 5)

        sca_coeff = 1 / scattering_length_function(wavelength)
        c_medium = (
            Constants.BaseConstants.c_vac * 1e-9 / ref_index_func(wavelength)
        )  # m/ns

        eta = random.uniform(keys[0])
        step_size = -jnp.log(eta) / sca_coeff

        dstep = step_size * dir
        new_pos = pos + dstep
        new_time = time + step_size / c_medium

        # Calculate intersection
        isec, isec_pos = intersection_f(
            pos,
            dir,
            step_size,
        )

        isec_time = time + jnp.linalg.norm(pos - isec_pos) / c_medium

        # If intersected, set position to intersection position
        new_pos = cond(
            isec, lambda args: args[0], lambda args: args[1], (isec_pos, new_pos)
        )

        # If intersected set time to intersection time
        new_time = cond(
            isec,
            lambda args: args[0],
            lambda args: args[1],
            (isec_time, new_time),
        )

        # If intersected, keep previous direction
        new_dir = cond(
            isec,
            lambda args: args[1],
            lambda args: calc_new_direction(args[0], args[1], scattering_function),
            (keys[1:4], dir),
        )

        stepcnt = cond(isec, lambda s: s, lambda s: s + 1, stepcnt)

        new_photon_state = {
            "pos": new_pos,
            "dir": new_dir,
            "time": new_time,
            "isec": isec,
            "stepcnt": stepcnt,
            "wavelength": wavelength,
        }

        return new_photon_state, keys[4]

    return step


def unpack_args(f):
    """Wrap a function by unpacking a single argument tuple."""

    def _f(args):
        return f(*args)

    return _f


@functools.partial(
    jax.profiler.annotate_function, name="initialize_direction_isotropic"
)
def initialize_direction_isotropic(rng_key):
    """Draw direction uniformly on a sphere."""
    k1, k2 = random.split(rng_key, 2)
    theta = jnp.arccos(random.uniform(k1, minval=-1, maxval=1))
    phi = random.uniform(k2, minval=0, maxval=2 * np.pi)
    direction = sph_to_cart(theta, phi, r=1)

    return direction


def initialize_direction_laser(rng_key):
    """Return e_z."""
    direction = jnp.array([0.0, 0.0, 1.0])
    return direction


def make_monochromatic_initializer(wavelength):
    """Make a monochromatic initializer function."""

    def initialize_monochromatic(rng_key):
        return wavelength

    return initialize_monochromatic


wl_mono_400nm_init = make_monochromatic_initializer(400)


def make_loop_until_isec_or_maxtime(max_time):
    """Make function that will call the step_function until either the photon intersetcs or max_time is reached."""

    def loop_until_isec_or_maxtime(step_function, initial_photon_state, rng_key):
        final_photon_state, _ = while_loop(
            lambda args: (args[0]["isec"] == False)  # noqa: E712
            & (args[0]["time"] < max_time),
            unpack_args(step_function),
            (initial_photon_state, rng_key),
        )
        return final_photon_state

    return loop_until_isec_or_maxtime


def make_loop_for_n_steps(n_steps):
    """Make function that calls step_function n_steps times."""

    def loop_for_nsteps(step_function, initial_photon_state, rng_key):
        final_photon_state, _ = fori_loop(
            0,
            n_steps,
            lambda i, args: unpack_args(step_function)(args),
            (initial_photon_state, rng_key),
        )
        return final_photon_state

    return loop_for_nsteps


def make_fixed_pos_time_initializer(
    initial_pos, initial_time, dir_init, wavelength_init
):
    """
    Initialize with fixed position and time and sample for direction and wavelength.

    initial_pos: float[3]
        Position vector of the emitter
    initial_time: float
        Emitter time
    dir_init: function
        Emission direction initializer
    wavelength_init: function
        wavelength initializer
    """

    def init(rng_key):
        k1, k2 = random.split(rng_key, 2)

        # Set initial photon state
        initial_photon_state = {
            "pos": initial_pos,
            "dir": dir_init(k1),
            "time": initial_time,
            "isec": False,
            "stepcnt": 0,
            "wavelength": wavelength_init(k2),
        }
        return initial_photon_state

    return init


def make_photon_trajectory_fun(
    step_function,
    photon_init_function,
    loop_func,
):
    """
    Make a photon trajectory function.

    This function calls the photon step function multiple times until
    some termination condition is reached (defined by `stepping_mode`)

    step_function: function
        Function that updates the photon state

    photon_init_function: function
        Function that returns the initial photon state

    loop_func: function
        Looping function that calls the photon step function.
    """

    def make_steps(key):
        """
        Make a function that steps a photon until it either intersects or  max length is reached.

        Parameters:
            key: PRNGKey

        Returns:
            photon_state: dict
                Final photon state
        """
        k1, k2 = random.split(key, 2)

        # Set initial photon state
        initial_photon_state = photon_init_function(k1)

        final_photon_state = loop_func(step_function, initial_photon_state, k2)

        return final_photon_state

    return make_steps


def collect_hits(traj_func, nphotons, nsims, seed=0, sim_limit=1e7):
    """Run photon prop multiple times and collect hits."""
    key = random.PRNGKey(seed)
    isec_times = []
    ph_thetas = []
    stepss = []
    nphotons = int(nphotons)
    isec_poss = []
    wavelengths = []

    total_detected_photons = 0
    sims_cnt = 0

    for i in range(nsims):
        key, subkey = random.split(key)
        photon_state = traj_func(random.split(key, num=nphotons))

        isecs = photon_state["isec"]

        isec_times.append(np.asarray(photon_state["time"][isecs]))
        stepss.append(np.asarray(photon_state["stepcnt"][isecs]))
        ph_thetas.append(np.asarray(jnp.arccos(photon_state["dir"][isecs, 2])))
        isec_poss.append(np.asarray(photon_state["pos"][isecs]))
        wavelengths.append(np.asarray(photon_state["wavelength"][isecs]))

        sims_cnt = i
        total_detected_photons += jnp.sum(isecs)
        if sim_limit is not None and total_detected_photons > sim_limit:
            break

    isec_times = np.concatenate(isec_times)
    ph_thetas = np.concatenate(ph_thetas)
    stepss = np.concatenate(stepss)
    isec_poss = np.vstack(isec_poss)
    wavelengths = np.concatenate(wavelengths)

    return isec_times, ph_thetas, stepss, isec_poss, sims_cnt, wavelengths
