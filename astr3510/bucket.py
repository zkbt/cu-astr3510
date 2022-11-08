from .imports import *
from astropy.uncertainty import uniform, normal, poisson
from astropy.visualization import quantity_support
from scipy import stats


def catch_photons_in_bucket(
    rate=1 * u.photon / u.s / u.m**2,
    diameter=1.0 * u.m,
    time=100 * u.s,
    visualize=True,
    figure=None,
):
    """
    Simulate a telescope catching photons,
    for a given photon flux (photons/s/m**2),
    telescope diameter (m), and exposure time (s).

    Parameters
    ----------
    rate: Quantity
        The rate at which photons are coming towards us,
        expressed in units of (photons/s/m**2).

    diameter: Quantity
        The effective diameter of a circular telescope,
        expressed in units of (m).

    time: Quantity
        The exposure duration, expressed in units of (s).

    Returns
    -------
    N: int
        The number of photons that land in the telescope
    """

    # what's the radius of the telescope
    radius = diameter / 2.0

    if visualize:
        with quantity_support():
            # make a rectangle that at least includes the telescope
            square = np.maximum(1.3 * diameter, 1.0 * u.m)

            # what's the expected total number of photons
            total_area = square * square
            area = np.pi * radius**2
            # create a rectangle of randomly located photons
            total_expectation = (total_area * time * rate).to_value(u.photon)
            expectation = (area * time * rate).to_value(u.photon)

            kw = dict(
                marker=".",
                markeredgecolor="none",
                alpha=0.5,
                linewidth=0,
                markersize=10,
            )
            if figure is None:
                figure = plt.figure(figsize=(10, 4), dpi=300)
            gs = plt.GridSpec(2, 2)
            plt.subplot(gs[:, 0])
            collected_color = "royalblue"

            Ntotal = poisson(total_expectation, n_samples=1).distribution[0]
            if Ntotal > 0:
                x = uniform(
                    lower=-square / 2, upper=square / 2, n_samples=Ntotal
                ).distribution
                y = uniform(
                    lower=-square / 2, upper=square / 2, n_samples=Ntotal
                ).distribution

                # determine which of these photons landed in the telescope
                incircle = (x**2 + y**2) < radius**2
                outofcircle = ~incircle
                N = np.sum(incircle)

                plt.plot(x[incircle], y[incircle], color=collected_color, **kw)
                plt.plot(x[outofcircle], y[outofcircle], color="black", **kw)
            else:
                N = 0
            # draw a circle
            theta = np.linspace(0, 2 * np.pi, 1000)
            plt.plot(
                radius * np.sin(theta),
                radius * np.cos(theta),
                linewidth=4,
                color=collected_color,
            )

            # label the radius of the circle
            plt.text(
                0,
                -radius * 1.2,
                f"{diameter.to_string(format='latex_inline', precision=4)}",
                ha="center",
                va="top",
                fontweight="bold",
                fontsize=10,
                color=collected_color,
            )
            plt.plot(
                radius * np.array([-1, 1]),
                -radius * 1.1 * np.ones(2),
                color=collected_color,
                linewidth=4,
            )

            # add a title
            plt.title(f"{N} photons")

            # set the aspect ratio of the plot to 1:1
            plt.axis("equal")
            plt.xlim(-square / 2, square / 2)
            plt.ylim(-square / 2, square / 2)

            # get rid of the square around the plot
            plt.axis("off")

            plt.subplot(gs[0, 1])
            kw = dict(format="latex_inline")
            plt.text(
                0,
                0,
                f"Photon Rate\n={rate.to_string(precision=4, **kw)}\n"
                + f"Telescope Diameter\n={diameter.to_string(precision=4, **kw)}\n"
                + f"Telescope Area\n={area.to_string(precision=4, **kw)}\n"
                + f"Exposure Time\n={time.to_string(precision=4, **kw)}\n"
                + f"Expected # of Photons\n={(expectation*u.photon).to_string(precision=4, **kw)}",
                transform=plt.gca().transAxes,
                ha="left",
                va="bottom",
            )
            plt.axis("off")
            plt.subplot(gs[1, 1])
            N_axis = np.arange(0, expectation + 5 * np.sqrt(expectation))
            PDF = stats.poisson(expectation).pmf(N_axis)
            plt.plot(N_axis, PDF, drawstyle="steps-mid", color="black")
            plt.axvline(N, color=collected_color, linewidth=4)
            plt.ylim(0, None)
            plt.xlabel("# of Photons")
            plt.ylabel("P(# of photons)")
    else:

        # if we don't need to make plot, just draw a Poisson number
        area = np.pi * radius**2
        expectation = (area * time * rate).to_value(u.photon)
        N = poisson(expectation, n_samples=1).distribution[0]

    return N
