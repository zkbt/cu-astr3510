import numpy as np


def exoplanet_transit(
    t,
    rp=0.1,
    per=3.0,
    t0=0.0,
    a=10.0,
    inc=90.0,
    ecc=0.0,
    w=0.0,
    u=[0.2, 0.2],
    *args,
    **kwargs,
):
    """
    One dimensional transit model with quadratic
    limb-darkening, using the `exoplanet-core` package.
    Parameters
    ----------
    t : array
        The times at which to evaluate the model.
    rp : float, array
        The radius of the planet in stellar radii
    per : float
        The period of the planet, in days.
    t0 : float
        Mid-transit time of the transit, in days.
    a : float, array
        The semi-major axis of the orbit, in stellar radii.
    inc : float, array
        The inclination of the orbit, in degrees.
    ecc: float, array
        The eccentricity of the orbit, unitless.
    w : float, array
        The argument of periastron, in degrees.
    u : array
        The quadratic limb-darkening coefficients.
    Returns
    -------
    monochromatic_flux : array
        The flux evaluated at each time.
    cached_inputs : dict
        A kludge to store any intermediate variables
        needed to speed up computation (= none for
        `exoplanet` transits).
    """

    try:
        from exoplanet_core import kepler, quad_limbdark_light_curve
    except ImportError:
        warnings.warn(
            f"""
        You're trying to produce a transit model using `exoplanet_core`,
        but it looks like you don't have `exoplanet_core` installed
        in the environment from which you're running `chromatic`.
        Please either install it with `pip install exoplanet-core`
        (see https://github.com/exoplanet-dev/exoplanet-core for details)
        or use the `.inject_transit(..., method='trapezoid')`
        option instead.
        """
        )

    # these two handy functions were adapted from the Exoplanet code:
    def warp_times(times, t_0, _pad=True):
        if _pad:
            return np.pad(t, (0, len(times)), "constant", constant_values=1) - t_0
        return times - t_0

    def get_true_anomaly(times, t_0, t_ref, n_per, e, _pad=True):
        M = (warp_times(times, t_0, _pad=_pad) - t_ref) * n_per
        if e == 0:
            return np.sin(M), np.cos(M)
        sin_f, cos_f = kepler(M, e + np.zeros(len(M)))
        return sin_f, cos_f

    n = 2 * np.pi / per
    opsw = 1 + np.sin(w)
    E0 = 2 * np.arctan2(
        np.sqrt(1 - ecc) * np.cos(w),
        np.sqrt(1 + ecc) * opsw,
    )
    M0 = E0 - ecc * np.sin(E0)
    t_periastron = t0 - M0 / n
    tref = t_periastron - t0

    # calculate the true anomaly as a function of time:
    sinf, cosf = get_true_anomaly(t, t0, tref, n, ecc, _pad=False)
    cosi = np.cos(inc * np.pi / 180)  # convert inclination to radians
    sini = np.sin(inc * np.pi / 180)  # convert inclination to radians
    w_rad = w * np.pi / 180  # convert omega to radians
    cos_w_plus_f = (np.cos(w_rad) * cosf) - (np.sin(w_rad) * sinf)  # cos(f+w)
    sin_w_plus_f = (np.sin(w_rad) * cosf) + (np.cos(w_rad) * sinf)  # sin(f+w)

    # x,y,z equations from Winn (2010):
    r = (a * (1 - (ecc**2))) / (1 + ecc * cosf)
    x = -r * cos_w_plus_f
    y = -r * sin_w_plus_f * cosi
    z = r * sin_w_plus_f * sini
    r_sky = np.sqrt(x**2 + y**2)

    # use exoplanet_core functions to extract light curve:
    # u_arr = np.array(u)
    flux = quad_limbdark_light_curve(u[0], u[1], r_sky, rp)
    # we only want the lightcurve where z > 0 (planet is between us and star)
    flux[z < 0] = 0
    return 1 + flux
