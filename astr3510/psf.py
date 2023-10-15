from .imports import *

from scipy.special import jv


from astropy.visualization import *


class PSF:
    def plot(
        self,
        theta_range=1 * u.arcsec,
        N=1001,
        unit=u.arcsecond,
        stretches=dict(linear=LinearStretch(), log=LogStretch()),
    ):

        with quantity_support():
            span = theta_range.to_value(unit)
            theta = np.linspace(-span, span, N) * unit
            theta_x2d, theta_y2d = np.meshgrid(theta, theta)
            theta_2d = np.sqrt(theta_x2d**2 + theta_y2d**2)

            fi, ax = plt.subplots(
                2,
                len(stretches),
                figsize=(8, 8),
                dpi=600,
                gridspec_kw=dict(height_ratios=[1, 1]),
                constrained_layout=True,
                sharex=True,
            )
            for i, scale in enumerate(stretches):

                if scale == "log":
                    image = np.log10(self(theta_2d))
                    vmin = -6
                    vmax = 0
                    ymin = 1e-6
                    image[np.isfinite(image) == False] = -6
                else:
                    image = self(theta_2d)
                    ymin = 0
                    vmin = 0
                    vmax = 1

                plt.sca(ax[0, i])
                plt.plot(theta, self(theta))
                plt.ylabel("What does a star look like?")
                plt.yscale(scale)
                plt.ylim(ymin, 1)
                plt.sca(ax[1, i])

                if scale == "log":
                    label = r"log$_{10}$(PSF)"
                else:
                    label = "PSF"
                plt.imshow(
                    image,
                    vmin=vmin,
                    vmax=vmax,
                    extent=[-span, span, -span, span],
                )
                plt.xlabel(rf"$\theta_x$ ({unit.to_string('latex_inline')})")
                plt.ylabel(rf"$\theta_y$ ({unit.to_string('latex_inline')})")
                plt.colorbar(label=label, orientation="horizontal")


class airy(PSF):
    def __init__(self, wavelength=0.5 * u.micron, aperture_radius=0.25 * u.m):
        """NOT NORMALIZED PROPERLY?!?!"""
        self.wavelength = wavelength
        self.aperture_radius = aperture_radius

    def __call__(self, theta):
        wavenumber = 2 * np.pi / self.wavelength
        x = (wavenumber * self.aperture_radius * np.sin(theta)).decompose()
        return ((2 * jv(1, x) / x) ** 2).decompose()


class seeing(PSF):
    def __init__(self, fwhm=1 * u.arcsecond):
        self.fwhm = fwhm

    def __call__(self, theta):
        sigma = self.fwhm / 2.354
        return 1 / (2 * np.pi * sigma**2) * np.exp(-0.5 * theta**2 / sigma**2)

    def __repr__(self):
        return f'{self.fwhm}" seeing'


class stellar_disk(PSF):
    def __init__(self, angular_radius=(1 * u.Rsun) / (10 * u.pc) * u.arcsecond):
        self.angular_radius = angular_radius

    def stellar_disk(theta):
        return (theta <= self.angular_radius) * 1.0
