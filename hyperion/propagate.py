"""Implementation of the photon propagation code."""
import jax.numpy as jnp
from jax import random
from jax.lax import cond, while_loop
import numpy as np
import jax


def sph_to_cart(theta, phi=0, r=1):
    """Transform spherical to cartesian coordinates."""
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)

    return jnp.array([x, y, z])


def photon_sphere_intersection(photon_x, photon_p, target_x, target_r, step_size):
    """
    Intersection of a line with a sphere.

    Given a photon origin, a photon direction, a step size, a target location and a target radius,
    calculate whether the photon intersects the target and the intrsection point.

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

    key, subkey = random.split(key)
    q = (b + 3.0) * ((random.uniform(subkey)) - 0.5) / b
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

    def _f(key):
        k1, k2 = random.split(key)
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


def calc_new_direction(key, old_dir, scattering_function):
    """
    Calculate new direction after sampling a scattering angle.

    Scattering is calculated in a reference frame local
    to the photon (e_z) and then rotated back to the global coordinate system.
    """
    k1, k2 = random.split(key)

    theta = scattering_function(k1)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    phi = random.uniform(k2, minval=0, maxval=2 * np.pi)
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
    sca_coeff,
    c_medium,
    target_x,
    target_r,
    intersection_f=photon_sphere_intersection,
    scattering_function=mixed_hg_rayleigh_antares,
):
    """
    Make a photon step function object.

    Parameters:
        sca_coeff: float
            Scattering coefficient
        c_medium: float
            Speed of light in medium
        target_x: float[3]
            Position vector of the target
        target_r: float
            Extent of the target (radius for spherical targets)
        intersection_f: function
            function used to calculate the intersection
        scattering_function:
            rng function drawing angles from scattering function
    """

    def step(pos, dir, time, key, isec, isec_pos, stepcnt):
        """Single photon step."""
        k1, k2 = random.split(key)

        eta = random.uniform(k1)
        step_size = -jnp.log(eta) / sca_coeff

        dstep = step_size * dir
        new_pos = pos + dstep
        new_time = time + step_size / c_medium

        # Calculate intersection
        isec, isec_pos = intersection_f(
            pos,
            dir,
            target_x,
            target_r,
            step_size,
        )

        isec_time = time + jnp.linalg.norm(pos - isec_pos) / c_medium

        new_pos = cond(
            isec, lambda args: args[0], lambda args: args[1], (isec_pos, new_pos)
        )

        new_time = cond(
            isec,
            lambda args: args[0],
            lambda args: args[1],
            (isec_time, new_time),
        )

        new_dir = cond(
            isec,
            lambda args: args[1],
            lambda args: calc_new_direction(args[0], args[1], scattering_function),
            (k1, dir),
        )

        stepcnt = cond(isec, lambda s: s, lambda s: s + 1, stepcnt)

        return new_pos, new_dir, new_time, k2, isec, isec_pos, stepcnt

    return step


def unpack_args(f):
    """Wrap a function by unpacking a single argument tuple."""

    def _f(args):
        return f(*args)

    return _f


def make_photon_trajectory_fun(
    step_function,
    emitter_x,
    emitter_t,
    max_time,
    emission_mode="uniform",
    stepping_mode="until_intersect",
):
    """
    Make a photon trajectory function.

    This function calls the photon step function multiple times until
    some termination condition is reached (defined by `stepping_mode`)

    max_dist: float
        Maximum time a photon is tracked
    emitter_x: float[3]
        Position vector of the emitter
    emitter_t: float
        Emitter time
    emission_mode: "uniform" or "laser"
        Emission mode
    stepping_mode: "until_intersect"
        Stepping mode
    """
    if stepping_mode == "until_intersect":

        def make_n_steps_until_intersect(key):
            """
            Make a function that steps a photon until it either intersects or max length is reached.

            Parameters:
                key: PRNGKey

            Returns:
                position: float[3]
                    Final photon position
                time: float
                    Final photon time
                direction: float[3]
                    Initial photon direction
            """
            if emission_mode == "uniform":
                k1, k2, k3 = random.split(key, 3)
                theta = jnp.arccos(random.uniform(k1, minval=-1, maxval=1))
                phi = random.uniform(k2, minval=0, maxval=2 * np.pi)
                direction = sph_to_cart(theta, phi, r=1)
            elif emission_mode == "laser":
                direction = jnp.array([0.0, 0.0, 1.0])
                key, k3 = random.split(key)

            position = jnp.array(emitter_x)
            time = emitter_t
            stepcnt = 0
            position, _, time, _, isec, isec_pos, stepcnt = while_loop(
                lambda args: (args[4] == False) & (args[2] < max_time),  # noqa: E712
                unpack_args(step_function),
                (position, direction, time, k3, False, position, stepcnt),
            )

            return position, time, direction, isec, isec_pos, stepcnt

        return make_n_steps_until_intersect
    else:
        raise NotImplementedError(f"Stepping mode {stepping_mode} not implemented")


def collect_hits(traj_func, nphotons, nsims, seed=0, sim_limit=50e6):
    """Run photon prop multiple times and collect hits."""
    key = random.PRNGKey(seed)
    isec_times = []
    ph_thetas = []
    stepss = []
    nphotons = int(nphotons)
    isec_poss = []

    total_detected_photons = 0
    sims_cnt = 0

    for i in range(nsims):
        key, subkey = random.split(key)
        positions, times, directions, isecs, isec_pos, steps = traj_func(
            random.split(key, num=nphotons)
        )
        isec_times.append(np.asarray(times[isecs]))
        stepss.append(np.asarray(steps[isecs]))
        ph_thetas.append(np.asarray(jnp.arccos(directions[isecs, 2])))
        isec_poss.append(np.asarray(isec_pos[isecs]))

        sims_cnt = i
        total_detected_photons += jnp.sum(isecs)
        if sim_limit is not None and total_detected_photons > sim_limit:
            break

    isec_times = np.concatenate(isec_times)
    ph_thetas = np.concatenate(ph_thetas)
    stepss = np.concatenate(stepss)
    isec_poss = np.vstack(isec_poss)

    return isec_times, ph_thetas, stepss, isec_poss, sims_cnt
