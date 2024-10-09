from ..imports import *

from scipy.special import jv
from astropy.visualization import *
from scipy.fft import fft2, fftfreq, fftshift
from scipy.signal import convolve2d

# from photutils.profiles import RadialProfile
import warnings


class Aperture:
    def __init__(self, **kw):
        self.make_aperture(**kw)
        self.make_psf(**kw)

    def __repr__(self):
        return self.__class__.__name__

    def label(self):
        return f"{self}-{self.wavelength}".replace(" ", "")

    def make_psf(self, wavelength=0.5 * u.micron, dtheta=None, supersample=5, **kw):

        self.wavelength = wavelength

        dx = np.median(np.diff(self.x))
        dy = np.median(np.diff(self.y))

        skip = 1
        if dtheta is None:
            N_y, N_x = np.array(np.shape(self.aperture)) * supersample
        else:
            N_x = int(wavelength / dx / dtheta.to_value("radian"))
            N_y = int(wavelength / dy / dtheta.to_value("radian"))
            if (N_x < 3) or (N_y < 3):
                print(
                    f"""
                The large pixel size of dtheta={dtheta} is 
                making it almost impossible to do the FFT. 
                Consider remaking your aperture with a 
                larger N=, so you'll still have resolution.
                """
                )
            if (N_x < len(self.x)) or (N_y < len(self.y)):
                skip = int(np.round(np.maximum(len(self.x) / N_x, len(self.y) / N_y)))
                N_x *= skip
                N_y *= skip
                print(
                    f"""
                Be careful! dtheta={dtheta} is so big the FFT is weird!
                Inflating by a factor of {skip}x.
                """
                )
            if N_x % 2 == 0:
                N_x += 1
            if N_y % 2 == 0:
                N_y += 1

        k_x = fftshift(fftfreq(N_x, dx))[::skip]
        k_y = fftshift(fftfreq(N_y, dy))[::skip]

        self.theta_x = (k_x * wavelength * u.radian).to("arcsec")
        self.theta_y = (k_y * wavelength * u.radian).to("arcsec")
        self.theta_x2d, self.theta_y2d = np.meshgrid(self.theta_x, self.theta_y)

        full_resolution_psf = np.abs(fftshift(fft2(self.aperture, (N_y, N_x)))) ** 2
        self.psf = convolve2d(full_resolution_psf, np.ones((skip, skip)), mode="same")[
            ::skip, ::skip
        ]

    def imshow_aperture_and_psf(self, scale="log", save=True, zoom=1):
        fi, ax = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
        self.imshow_aperture(
            ax=ax[0],
            save=False,
        )
        self.imshow_psf(ax=ax[1], save=False, zoom=zoom)
        plt.savefig(f"aperture+psf-imshow-{self.label()}.pdf")

    def display_psf(
        self, theta_range=10 * u.arcsec, stretches=["linear", "log"], save=True
    ):

        with quantity_support():
            span = theta_range.value
            unit = theta_range.unit

            fi, ax = plt.subplots(
                2,
                len(stretches),
                figsize=(8, 7),
                gridspec_kw=dict(height_ratios=[0.5, 1]),
                constrained_layout=True,
                sharex=True,
            )
            for i, scale in enumerate(stretches):

                self.plot_psf(
                    ax=ax[0, i], theta_range=theta_range, scale=scale, save=False
                )
                plt.xlabel("")

                self.imshow_psf(ax=ax[1, i], scale=scale, save=False, colorbar=False)
                if scale == "log":
                    label = r"log$_{10}$(PSF)"
                else:
                    label = "PSF"
                # if i > 0:
                #    ax[1, 0].get_shared_y_axes().join(ax[1, 0], ax[1, i])
                plt.colorbar(ax=ax[:, i], label=label, orientation="horizontal")
            if save:
                plt.savefig(f"aperture+psf-everything-{self.label()}.pdf")

    def imshow_aperture(self, ax=None, save=True, colorbar=True, **kw):
        with quantity_support():
            if ax is None:
                fi, ax = plt.subplots(1, 1, constrained_layout=True)
            plt.sca(ax)
            imshow_kw = dict(
                cmap="gray", extent=[min(self.x), max(self.x), min(self.y), max(self.y)]
            )
            imshow_kw.update(**kw)
            plt.imshow(self.aperture, **imshow_kw)
            plt.xlabel(
                r"$\sf x$ at Telescope Aperture"
                + f' ({self.x.unit.to_string("latex_inline")})'
            )
            plt.ylabel(
                r"$\sf y$ at Telescope Aperture"
                + f' ({self.y.unit.to_string("latex_inline")})'
            )
            if colorbar:
                plt.colorbar()

            if save:
                plt.savefig(f"aperture-imshow-{self.label()}.pdf")

    def imshow_psf(self, ax=None, scale="log", save=True, colorbar=True, zoom=1, **kw):
        with quantity_support():
            if ax is None:
                fi, ax = plt.subplots(1, 1, constrained_layout=True)
            plt.sca(ax)

            plt.cla()
            if scale == "log":
                image = np.log10(self.psf / np.nanmax(self.psf))
                vmin = -10
                vmax = 0
                image[np.isfinite(image) == False] = -6
            else:
                image = self.psf / np.nanmax(self.psf)
                vmin = 0
                vmax = 1
            imshow_kw = dict(
                cmap="gray",
                extent=[
                    min(self.theta_x.value),
                    max(self.theta_x.value),
                    min(self.theta_y.value),
                    max(self.theta_y.value),
                ],
                vmin=vmin,
                vmax=vmax,
            )
            imshow_kw.update(**kw)
            plt.imshow(image, **imshow_kw)
            plt.xlabel(
                r"$\sf \theta_x$ at Focal Plane"
                + f' ({self.theta_x.unit.to_string("latex_inline")})'
            )
            plt.ylabel(
                r"$\sf \theta_y$ at Focal Plane"
                + f' ({self.theta_y.unit.to_string("latex_inline")})'
            )

            plt.xlim(min(self.theta_x.value) / zoom, max(self.theta_x.value) / zoom)
            plt.ylim(min(self.theta_y.value) / zoom, max(self.theta_y.value) / zoom)

            if colorbar:
                if scale == "log":
                    label = r"log$_{10}$(PSF)"
                else:
                    label = "PSF"
                plt.colorbar(label=label)
            if save:
                plt.savefig(f"psf-imshow-{self.label()}.pdf")

    def plot_psf(self, ax=None, theta_range=10 * u.arcsec, scale="log", save=False):
        with quantity_support():
            if ax is None:
                fi, ax = plt.subplots(1, 1, constrained_layout=True)
            plt.sca(ax)

            N_x, N_y = len(self.theta_x), len(self.theta_y)
            i_x = int(np.interp(0 * u.arcsec, self.theta_x, np.arange(N_x)))
            i_y = int(np.interp(0 * u.arcsec, self.theta_y, np.arange(N_y)))

            if scale == "log":
                ymin = 1e-10
                ymax = 1
            else:
                ymin = 0
                ymax = 1

            # dtheta = np.median(np.diff(self.theta_x))
            # edge_radii = np.arange(np.min([N_x/2, N_y/2, theta_range/dtheta]))
            # radial_profile = RadialProfile(self.psf, [i_x, i_y], edge_radii)
            # plt.plot(radial_profile.radius, radial_profile.profile)

            psf = self.psf / np.nanmax(self.psf)
            # r = (self.theta_x2d**2 + self.theta_y2d**2).flatten()
            # i = np.argsort(r)
            # plt.plot(r[i], psf.flatten()[i], **kw)
            # plt.plot(-r[i], psf.flatten()[i], **kw)
            kw = dict(color="black")
            plt.plot(self.theta_x, psf[i_y, :], **kw)
            plt.yscale(scale)
            plt.ylim(ymin, None)
            plt.xlabel(
                r"$\sf \theta$ at Focal Plane"
                + f' ({self.theta_x.unit.to_string("latex_inline")})'
            )
            plt.ylabel("Relative Intensity")
            if save:
                plt.savefig(f"psf-slice-{self.label()}.pdf")

    def make_visualizations(self):
        self.imshow_aperture()
        self.imshow_psf()
        self.imshow_aperture_and_psf()
        self.display_psf()


example_apertures_directory = os.path.join(astr3510_directory, "psfs", "apertures")
default_aperture_image_filename = os.path.join(
    example_apertures_directory, "leavitt.jpg"
)


class BitmapAperture(Aperture):
    def __repr__(self):
        return f"{self.__class__.__name__}-{self.diameter}".replace(" ", "")

    def make_aperture(
        self, filename=default_aperture_image_filename, diameter=1 * u.m, **kw
    ):
        image = plt.imread(filename)
        if len(np.shape(image)) == 3:
            image = np.sum(image, -1)
        image = image / np.nanmax(image)
        rows, columns = np.shape(image)

        # set up a blank image
        self.diameter = diameter
        radius = diameter / 2
        aspect_ratio = columns / rows
        self.x = np.linspace(-radius, radius, columns) * aspect_ratio
        self.y = np.linspace(-radius, radius, rows)
        self.x2d, self.y2d = np.meshgrid(self.x, self.y)
        self.aperture = image


class JWST(BitmapAperture):
    def __repr__(self):
        return "JWST"

    def make_aperture(self, **kw):
        BitmapAperture.make_aperture(
            self,
            filename=os.path.join(example_apertures_directory, "jwst.jpg"),
            diameter=6.5 * u.m,
            **kw,
        )


class BahtinovAperture(BitmapAperture):
    def make_aperture(self, **kw):
        BitmapAperture.make_aperture(
            self,
            filename=os.path.join(example_apertures_directory, "bahtinov.jpg"),
            **kw,
        )


class ShapeAperture(Aperture):
    def make_aperture(self, radius=1 * u.m, N=201, buffer=1.1, **kw):

        # set up a blank image
        self.x = np.linspace(-radius, radius, N) * buffer
        self.y = np.linspace(-radius, radius, N) * buffer
        self.x2d, self.y2d = np.meshgrid(self.x, self.y)
        self.aperture = np.zeros((N, N))


class RectangleAperture(ShapeAperture):
    def __repr__(self):
        return f"Rectangle-{self.width}x{self.height}".replace(" ", "")

    def make_aperture(self, width=1 * u.m, height=1 * u.m, **kw):
        self.width = width
        self.height = height
        ShapeAperture.make_aperture(self, radius=np.maximum(width, height) / 2, **kw)
        r = np.sqrt(self.x2d**2 + self.y2d**2)
        self.aperture[
            (np.abs(self.x2d) < width / 2) * (np.abs(self.y2d) < height / 2)
        ] = 1


class CircleAperture(ShapeAperture):
    def __repr__(self):
        return f"Circle-{self.diameter}".replace(" ", "")

    def make_aperture(self, diameter=1 * u.m, **kw):
        self.diameter = diameter
        ShapeAperture.make_aperture(self, radius=diameter / 2, **kw)
        r = np.sqrt(self.x2d**2 + self.y2d**2)
        self.aperture[r < diameter / 2] = 1


class ObscuredCircleAperture(CircleAperture):
    def __repr__(self):
        return f"ObscuredCircle-diameter={self.diameter}-obscuration={self.obscuration_diameter}-spiders={self.spiders}".replace(
            " ", ""
        )

    def make_aperture(
        self,
        diameter=1 * u.m,
        obscuration_diameter=0.5 * u.m,
        spiders=4,
        spider_width=5 * u.cm,
        **kw,
    ):
        CircleAperture.make_aperture(self, diameter=diameter, **kw)
        self.obscuration_diameter = obscuration_diameter
        self.spiders = spiders
        self.spider_width = spider_width

        # substract central obscuration
        self.subtract_circle(diameter=obscuration_diameter)

        # subtract spiders
        self.subtract_spiders(spiders=spiders, spider_width=spider_width)

    def subtract_circle(self, diameter=0.5 * u.m, x_center=0 * u.m, y_center=0 * u.m):
        r = np.sqrt((self.x2d - x_center) ** 2 + (self.y2d - y_center) ** 2)
        self.aperture[r < diameter / 2] = 0

    def subtract_spiders(self, spiders=4, spider_width=5 * u.cm):
        r = np.sqrt((self.x2d) ** 2 + (self.y2d) ** 2)
        theta = np.arctan2(self.y2d, self.x2d)
        for spider_theta in np.linspace(0, 2 * np.pi, spiders + 1)[:-1] * u.radian:
            distance_from_spider = np.abs(np.sin(theta - spider_theta)) * r
            in_right_direction = np.cos(theta - spider_theta) > 0
            is_spider = (distance_from_spider < spider_width / 2) * in_right_direction
            self.aperture[is_spider] = 0


class Spitzer(ObscuredCircleAperture):
    def __repr__(self):
        return "Spitzer"

    def make_aperture(self, **kw):
        ObscuredCircleAperture.make_aperture(
            self,
            diameter=0.85 * u.m,
            obscuration_diameter=0.12 * u.m,
            spiders=3,
            spider_width=2 * u.cm,
            **kw,
        )


class Hubble(ObscuredCircleAperture):
    def __repr__(self):
        return "Hubble"

    def make_aperture(self, **kw):
        ObscuredCircleAperture.make_aperture(
            self,
            diameter=2.4 * u.m,
            obscuration_diameter=0.8 * u.m,
            spiders=4,
            spider_width=2 * u.cm,
            **kw,
        )

        # add weird mirror circles
        blob_distance_from_center = 0.9 * self.diameter / 2
        blob_diameter = 0.2 * u.m
        for blob_theta in np.array([30, 150, 270]) * np.pi / 180:
            self.subtract_circle(
                diameter=blob_diameter,
                x_center=blob_distance_from_center * np.cos(blob_theta),
                y_center=blob_distance_from_center * np.sin(blob_theta),
            )


class Leto(ObscuredCircleAperture):
    def __repr__(self):
        return "Leto"

    def make_aperture(self, **kw):
        ObscuredCircleAperture.make_aperture(
            self,
            diameter=0.61 * u.m,
            obscuration_diameter=0.2 * u.m,
            spiders=4,
            spider_width=2 * u.cm,
            **kw,
        )


class Artemis(ObscuredCircleAperture):
    def __repr__(self):
        return "Artemis"

    def make_aperture(self, **kw):
        ObscuredCircleAperture.make_aperture(
            self,
            diameter=0.508 * u.m,
            obscuration_diameter=0.198 * u.m,
            spiders=4,
            spider_width=2 * u.cm,
            **kw,
        )


class Airy(CircleAperture):
    """An analytic airy pattern!"""

    def airy(self, theta, wavelength=0.5 * u.micron, diameter=1 * u.m):
        k = 2 * np.pi / wavelength
        x = (k * diameter / 2 * np.sin(theta)).decompose()
        return ((2 * jv(1, x) / x) ** 2).decompose()

    def make_psf(self, wavelength=0.5 * u.micron, supersample=20, **kw):

        self.wavelength = wavelength

        N_y, N_x = np.array(np.shape(self.aperture)) * supersample
        dx = np.median(np.diff(self.x))
        dy = np.median(np.diff(self.y))
        k_x = fftshift(fftfreq(N_x, dx))
        k_y = fftshift(fftfreq(N_y, dy))

        self.theta_x = np.arcsin(k_x * wavelength).to("arcsec")
        self.theta_y = np.arcsin(k_y * wavelength).to("arcsec")
        self.theta_x2d, self.theta_y2d = np.meshgrid(self.theta_x, self.theta_y)

        theta = np.sqrt(self.theta_x2d**2 + self.theta_y2d**2)
        a = self.airy(theta, wavelength=self.wavelength, diameter=self.diameter)
        self.psf = a


apertures_to_test = [
    CircleAperture,
    ObscuredCircleAperture,
    RectangleAperture,
    JWST,
    BahtinovAperture,
    Hubble,
    Spitzer,
    Leto,
    Artemis,
]


"""class PSF:
    

"""


"""class seeing(PSF):
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
"""
