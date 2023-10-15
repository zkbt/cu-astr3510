from .imports import *
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astropy.table import QTable
from astropy.visualization import quantity_support

# set a high row limit to allow lots of stars in crowded fields
Gaia.ROW_LIMIT = 50000
Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source"

# Gaia filter transformations from
# https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
terms = {}
terms["G-r_sloan"] = [-0.12879, 0.24662, -0.027464, -0.049465]
terms["G-i_sloan"] = [-0.29676, 0.64728, -0.10141]
terms["G-g_sloan"] = [0.13518, -0.46245, -0.25171, 0.021349]
terms["G-V_johnsoncousins"] = [-0.01760, -0.006860, -0.1732]
terms["G-R_johnsoncousins"] = [-0.003226, 0.3833, -0.1345]
terms["G-I_johnsoncousins"] = [0.02085, 0.7419, -0.09631]

uncertainties = {}
uncertainties["G-r_sloan"] = 0.066739
uncertainties["G-i_sloan"] = 0.98957
uncertainties["G-g_sloan"] = 0.16497
uncertainties["G-V_johnsoncousins"] = 0.0458
uncertainties["G-R_johnsoncousins"] = 0.048
uncertainties["G-I_johnsoncousins"] = 0.049


def estimate_other_filters(table):
    """
    Take a table of simple Gaia photometry
    from `get_gaia_data` and use
    color transformations to estimate the
    magnitudes in other filters.

    Data from:
    https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html

    Parameters
    ----------
    table : QTable
        Table with Gaia photometry (see `get_gaia_data`).

    Returns
    -------
    table : QTable
        Table updated with estimated filter magnitudes.
    """

    # define the inputs for color transformation functions
    G = table["G_gaia_mag"]
    BP_minus_RP = table["BP_gaia_mag"] - table["RP_gaia_mag"]

    # loop through filters
    for color, coefficients in terms.items():

        # calculate the polynomial
        G_minus_x = (
            np.sum([c * BP_minus_RP**i for i, c in enumerate(coefficients)], 0)
            * u.mag
        )

        # store in table
        filter_name = color.split("-")[-1]
        table[f"{filter_name}_mag"] = G - G_minus_x
    return table


def get_gaia(center, radius=6 * u.arcmin):
    """
    Get photometry and basic data based on Gaia EDR3.

    Use Gaia Early Data Release 3 to download data
    for all stars within a particular radius of a
    particular center. Gaia is an all-sky space survey;
    basically any star you can see with a moderate
    aperture ground-based telescope has been measured
    by Gaia, so it can provide a useful reference.
    Gaia positions, motions, and photometry will be
    included, along with magnitudes in other filters
    estimated with `estimate_other_filters` via
    Gaia color transformations.

    Parameters
    ----------
    center : SkyCoord
        An astropy SkyCoord object indicating the
        right ascension and declination of the center.
        This center can be created in a few ways:
        ```
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        center = SkyCoord(ra=123.45*u.deg, dec=67.89*u.deg)
        center = SkyCoord(ra=123.45*u.deg, dec=67.89*u.deg)
        other_center = SkyCoord.from_name('Vega')
        ```
    radius : astropy.units.Quantity
        The angular radius around which the star to
        search for photometry. Default is 6 arcminutes.

    Returns
    -------
    table : astropy.table.Table
        An astropy table containing the results,
        with columns for different coordinates
        or filters, and rows for different stars.
    """

    # get the data from the archive
    job = Gaia.cone_search_async(
        center,
        radius,
        columns=[
            "ra",
            "dec",
            "phot_g_mean_mag",
            "phot_rp_mean_mag",
            "phot_bp_mean_mag",
            "parallax",
            "pmra",
            "pmdec",
        ],
    )
    results = job.get_results()

    # tidy up the table and convert to quantities
    table = QTable(results)
    table.rename_columns(
        ["phot_g_mean_mag", "phot_rp_mean_mag", "phot_bp_mean_mag", "dist"],
        ["G_gaia_mag", "RP_gaia_mag", "BP_gaia_mag", "distance_from_center"],
    )

    # add unit to the distance from the field center
    table["distance_from_center"].unit = u.deg

    # populate with other estimated filter magnitudes
    table = estimate_other_filters(table)

    # keep track of center and radius
    table.meta["center"] = center
    table.meta["radius"] = radius

    # return the table
    return table


def plot_gaia(
    table,
    filter="G_gaia",
    faintest_magnitude_to_show=20,
    faintest_magnitude_to_label=16,
    size_of_zero_magnitude=100,
    unit=u.arcmin,
):
    """
    Plot a finder chart using results from `get_gaia_data`.

    Use the table of positions and photometry returned by
    the `get_gaia_data` function to plot a finder chart
    with symbol sizes representing the brightness of the
    stars in a particular filter.

    Parameters
    ----------
    filter : str
        The filter to use for setting the size of the points.
        Options are "G_gaia", "RP_gaia", "BP_gaia", "g_sloan",
        "r_sloan", "i_sloan", "V_johnsoncousins", "R_johnsoncousins",
        "I_johnsoncousins". Default is "G_gaia".
    faintest_magnitude_to_show : float
        What's the faintest star to show? Default is 20.
    faintest_magnitude_to_label : float
        What's the faintest magnitude to which we should
        add a numerical label? Default is 16.
    size_of_zero_magnitude : float
        What should the size of a zeroth magnitude star be?
        Default is 100.
    unit : Unit
        What unit should be used for labels? Default is u.arcmin.
    """

    # extract the center and size of the field
    center = table.meta["center"]
    radius = table.meta["radius"]

    # find offsets relative to the center
    dra = ((table["ra"] - center.ra) * np.cos(table["dec"])).to(unit)
    ddec = (table["dec"] - center.dec).to(unit)

    # set the sizes of the points
    mag = table[f"{filter}_mag"].to_value("mag")
    size_normalization = size_of_zero_magnitude / faintest_magnitude_to_show**2
    marker_size = (
        np.maximum(faintest_magnitude_to_show - mag, 0) ** 2 * size_normalization
    )

    # handle astropy units better
    with quantity_support():

        # plot the stars
        plt.scatter(dra, ddec, s=marker_size, color="black")
        plt.xlabel(
            f"$\Delta$(Right Ascension) [{unit}] relative to {center.ra.to_string(u.hour, format='latex', precision=2)}"
        )
        plt.ylabel(
            f"$\Delta$(Declination) [{unit}] relative to {center.dec.to_string(u.deg, format='latex', precision=2)}"
        )

        # add labels
        filter_label = filter.split("_")[0]
        to_label = np.nonzero(mag < faintest_magnitude_to_label)[0]
        for i in to_label:
            plt.text(
                dra[i],
                ddec[i],
                f"  {filter_label}={mag[i]:.2f}",
                ha="left",
                va="center",
                fontsize=5,
            )

        # add a grid
        plt.grid(color="gray", alpha=0.2)

        # plot a circle for the edge of the field
        circle = plt.Circle(
            [0, 0], radius, fill=False, color="gray", linewidth=2, alpha=0.2
        )
        plt.gca().add_patch(circle)

        # set the axis limits
        plt.xlim(radius, -radius)
        plt.ylim(-radius, radius)
        plt.axis("scaled")
