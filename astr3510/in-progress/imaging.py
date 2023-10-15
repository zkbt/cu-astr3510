"""
Tools for creating simulated images with noise,
for conceptual signal-to-noise demonstrations.
"""
from .imports import *
from .psf import *
from chromatic.rainbows.visualizations.utilities import (
    get_animation_writer_and_displayer,
)
from IPython.display import display
from photutils.aperture import (
    CircularAperture,
    aperture_photometry,
    ApertureStats,
    CircularAnnulus,
)


def to_photons(x):
    if "electron" in x.unit.to_string():
        return x * u.photon / u.electron
    elif "photon" in x.unit.to_string():
        return x


class Image:
    keys = ["shape", "gain", "pixel_scale", "psf"]

    def __init__(
        self,
        components=[],
        shape=(30, 30),
        gain=2 * u.photon / u.adu,
        pixel_scale=0.5 * u.arcsec / u.pixel,
        psf=seeing(fwhm=2.0 * u.arcsec),
    ):
        self.shape = shape
        self.gain = gain
        self.pixel_scale = pixel_scale
        self.psf = psf
        self.components = components
        for c in self.components:
            c.image = self

    def describe_as_string(self):
        s = ""  # f"{self.__class__.__name__}:\n"
        for k in self.keys:
            s += f"{k}={getattr(self, k)}\n"
        return s

    def describe_everything_as_string(self):
        return "".join([c.describe_as_string() for c in [self] + self.components])

    def do_photometry(
        self,
        positions=[(15, 15)],
        radius=3,
        sky_subtract=True,
        radius_sky_inner=6,
        radius_sky_outer=10,
        visualize=True,
        **kwargs,
    ):
        self.aperture = CircularAperture(positions=positions, r=radius)

        self.photometry_results = aperture_photometry(
            data=self.images["data"],
            apertures=self.aperture,
            error=self.images["uncertainty"],
        )

        flux = self.photometry_results["aperture_sum"].data[0]
        uncertainty = self.photometry_results["aperture_sum_err"].data[0]
        self.aperture_photometry = aperture_photometry(
            data=self.images["data"],
            apertures=self.aperture,
            error=self.images["uncertainty"],
        )
        print(f"{self.aperture.area:.1f} pixel = ⭕️ aperture area")
        print(f"{flux:>10.1f} photons = ⭐️ measured fluence in aperture")
        flux_model = aperture_photometry(
            data=self.images["model"],
            apertures=self.aperture,
        )["aperture_sum"].data[0]

        flux_variance = aperture_photometry(
            data=self.images["uncertainty"] ** 2,
            apertures=self.aperture,
        )["aperture_sum"].data[0]

        if sky_subtract:
            self.sky_aperture = CircularAnnulus(
                positions, r_in=radius_sky_inner, r_out=radius_sky_outer
            )
            measured_flux_from_sky = (
                ApertureStats(self.images["data"], self.sky_aperture).median[0]
                * self.aperture.area
            )
            flux -= measured_flux_from_sky
            model_flux_from_sky = (
                ApertureStats(self.images["model"], self.sky_aperture).median[0]
                * self.aperture.area
            )
            flux_model -= model_flux_from_sky
            print(
                f"{measured_flux_from_sky:>10.1f} photons = 🌫 sky fluence in aperture"
            )
            print(
                f"{flux:>10.1f} photons = (⭐️ measured) - (🌫 sky) fluence in aperture"
            )

        print(f"{uncertainty:>10.1f} photons = 🃏 uncertainty on fluence in aperture")
        aperture_kw = dict(color="white", linewidth=2)
        sky_kw = dict(color="silver", linewidth=2, linestyle="--")

        text_kw = dict(
            va="top",
            ha="center",
            color="white",
        )

        if visualize:
            try:
                for a in self._plotted_apertures:
                    a[0].set_visible(False)
            except AttributeError:
                self._plotted_apertures = []
            # label the values
            plt.sca(self.imshowed["data"].axes)
            self._plotted_apertures.append(self.aperture.plot(**aperture_kw))
            if sky_subtract:
                self._plotted_apertures.append(self.sky_aperture.plot(**sky_kw))

            # plt.text(
            #    positions[0][0],
            #    positions[0][1] + 1.2 * radius,
            plt.title(
                f"measured flux in aperture:\n{flux:.1f}$\pm${uncertainty:.1f} {self.unit}"
            )
            #    **text_kw,
            # )

            plt.sca(self.imshowed["model"].axes)
            self._plotted_apertures.append(self.aperture.plot(**aperture_kw))
            if sky_subtract:
                self._plotted_apertures.append(self.sky_aperture.plot(**sky_kw))
            plt.title(
                f"model flux in aperture:\n{flux_model:.1f} {self.unit}",
            )

            plt.sca(self.imshowed["uncertainty"].axes)
            self._plotted_apertures.append(self.aperture.plot(**aperture_kw))
            if sky_subtract:
                self._plotted_apertures.append(self.sky_aperture.plot(**sky_kw))

            plt.title(
                f"uncertainty in aperture:\n{np.sqrt(flux_variance):.1f} {self.unit}"
            )

        return flux, flux_model, uncertainty

    def plot_photometry_with_radius(self):
        flux, model, uncertainty = [], [], []
        radii = np.arange(0.2, 15, 0.2)
        for r in radii:
            fl, mo, un = self.do_photometry(
                radius=r, visualize=False, sky_subtract=False
            )
            flux.append(fl)
            model.append(mo)
            uncertainty.append(un)
        flux, model, uncertainty = (
            np.array(flux),
            np.array(model),
            np.array(uncertainty),
        )
        fi, ax = plt.subplots(
            2, 1, dpi=300, figsize=(6, 4), sharex=True, constrained_layout=True
        )

        fwhm_arcsec = self.psf.fwhm
        fwhm_pixels = fwhm_arcsec / self.pixel_scale

        def plot_fwhm():
            alpha = 0.5
            plt.axvline(fwhm_pixels.value, color="gray", alpha=alpha)
            plt.text(
                fwhm_pixels.value,
                0,
                f"  FWHM ({fwhm_arcsec} = {fwhm_pixels})\n",
                color="gray",
                alpha=alpha,
            )

        plt.sca(ax[0])
        plt.errorbar(radii, flux, uncertainty, label="measured")
        plt.plot(radii, model, label="model")
        plt.ylabel("Photons in Aperture")
        plot_fwhm()

        plt.legend(frameon=False)
        plt.ylim(0, None)
        plt.sca(ax[1])
        plt.plot(radii, flux / uncertainty, label="measured")
        plt.plot(radii, model / uncertainty, label="model")
        plt.xlabel("Aperture Radius (pixels)")
        plt.ylabel("S/N (flux/uncertainty)")
        plt.ylim(0, None)
        plot_fwhm()

        plt.legend(frameon=False)

    def __repr__(self):
        return f"Image {self.shape} = {'+'.join([str(x) for x in self.components])}"

    def mu(self, t=1 * u.s):
        return (
            np.sum(
                np.array(
                    [
                        getattr(x, f"model_mean_{self.type}")(t).to_value(self.unit)
                        for x in self.components
                    ],
                    dtype="object",
                ),
                axis=0,
            )
            * self.unit
        )

    def sigma(self, t=1 * u.s):
        return np.sqrt(
            np.sum(
                np.array(
                    [
                        getattr(x, f"model_sigma_{self.type}")(t).to_value(self.unit)
                        ** 2
                        for x in self.components
                    ],
                    dtype="object",
                ),
                axis=0,
            )
            * self.unit**2
        )

    def imshow_realization(
        self,
        t=1 * u.s,
        ax=None,
        filename="animation.mp4",
        photometry=False,
        photometry_positions=[(15, 15)],
        photometry_radius=3,
        **kwargs,
    ):
        if ax is None:
            fi, ax = plt.subplots(1, 1, figsize=(4, 2.5), dpi=300)

        time = np.max(t)
        i = self.realization(t=time)
        e = self.sigma(t=time)
        imshowed = plt.imshow(i, **kwargs)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(label=self.unit.to_string("latex_inline"))

        if photometry:
            self.do_photometry(
                data=i.to_value(self.unit),
                uncertainty=e.to_value(self.unit),
                positions=photometry_positions,
                radius=photometry_radius,
            )

        if np.size(t) > 1:
            writer, displayer = get_animation_writer_and_displayer(filename)
            with writer.saving(fi, filename, fi.get_dpi()):
                for time in t:
                    i = self.realization(t=time)
                    imshowed.set_data(i)
                    writer.grab_frame()
            try:
                display(displayer(filename, embed=True))
            except TypeError:
                display(displayer(filename))

    def make_exposure(
        self,
        t=1 * u.s,
        visualize=True,
        ax=None,
        photometry=False,
        photometry_positions=[(15, 15)],
        photometry_radius=3,
        cmap="gray",
        **kwargs,
    ):
        time = np.max(t)
        model = self.mu(t=time)
        uncertainty = self.sigma(t=time)
        data = self.realization(t=time)

        images = dict(
            data=data.to_value(self.unit),
            model=model.to_value(self.unit),
            uncertainty=uncertainty.to_value(self.unit),
        )
        if ax is None:
            fi, ax = plt.subplots(
                1,
                3 + 1,
                figsize=(10, 4),
                dpi=300,
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )

        imshowed = {}
        for a, i, label in zip(
            ax,
            [model, uncertainty, data],
            [
                f"model",
                f"uncertainty",
                f"data",
            ],
        ):
            plt.sca(a)
            imshowed[label] = plt.imshow(i, cmap=cmap, **kwargs)
            plt.xticks([])
            plt.yticks([])
            plt.ylabel(label.capitalize())
            plt.colorbar(
                orientation="horizontal", label=self.unit.to_string("latex_inline")
            )

            if label == "data":
                if photometry:
                    self.do_photometry(
                        data=images["data"],
                        uncertainty=images["uncertainty"],
                        positions=photometry_positions,
                        radius=photometry_radius,
                    )

        plt.sca(ax[-1])
        texted = plt.text(
            0.1,
            0.0,
            f"exposure_time={time:.1f}\n" + self.describe_everything_as_string(),
            transform=plt.gca().transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            clip_on=False,
        )
        plt.axis("off")

        self.images = images
        self.imshowed = imshowed

    def imshow_with_models(
        self,
        t=1 * u.s,
        ax=None,
        filename="animation.mp4",
        photometry=False,
        photometry_positions=[(15, 15)],
        photometry_radius=3,
        **kwargs,
    ):
        time = np.max(t)
        model = self.mu(t=time)
        sigma = self.sigma(t=time)
        data = self.realization(t=time)

        if ax is None:
            fi, ax = plt.subplots(
                1,
                3 + 1,
                figsize=(13, 2.5),
                dpi=300,
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )

        imshowed = {}
        for a, i, label in zip(
            ax,
            [data, model, sigma],
            [
                f"data",
                f"model",
                f"uncertainty",
            ],
        ):
            plt.sca(a)
            imshowed[label] = plt.imshow(i, **kwargs)
            plt.xticks([])
            plt.yticks([])
            plt.ylabel(label.capitalize())
            plt.colorbar(label=self.unit.to_string("latex_inline"))

            if label == "data":
                if photometry:
                    self.do_photometry(
                        data=data.to_value(u.adu),
                        uncertainty=sigma.to_value(u.adu),
                        positions=photometry_positions,
                        radius=photometry_radius,
                    )

        plt.sca(ax[-1])
        texted = plt.text(
            0.1,
            0.9,
            f"exposure_time={time:.1f}\n" + self.describe_everything_as_string(),
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=9,
            clip_on=False,
        )
        plt.axis("off")

        return imshowed

        if np.size(t) > 1:
            writer, displayer = get_animation_writer_and_displayer(filename)
            with writer.saving(fi, filename, fi.get_dpi()):
                for time in t:
                    model = self.model_mean_uncalibrated(t=time)
                    sigma = self.model_sigma_uncalibrated(t=time)
                    data = self.realization(t=time)
                    for a, i, label in zip(
                        ax,
                        [data, model, sigma],
                        [
                            "Data (ADU)",
                            "Model ($\mu$, ADU)",
                            "Noise ($\sigma$, ADU)",
                        ],
                    ):
                        imshowed[label].set_data(i)
                    texted.set_text(
                        f"exposure_time={time:.1f}\n" + self.describe_as_string()
                    )
                    writer.grab_frame()
            try:
                display(displayer(filename, embed=True))
            except TypeError:
                display(displayer(filename))

    def imshow_components_with_models(self, **kwargs):
        rows = 4
        cols = len(self.components) + 2
        fi, ax = plt.subplots(
            rows,
            cols,
            figsize=(3 * cols, 2.5 * rows),
            dpi=300,
            sharex=True,
            sharey=True,
            constrained_layout=True,
            gridspec_kw=dict(width_ratios=[1, 0.25] + [1] * len(self.components)),
        )
        for col, component in enumerate(self.components):
            component.imshow_with_models(ax=ax[:, col + 2], **kwargs)
            for a in ax[:, col + 2]:
                a.set_title(component.__class__.__name__)

        self.imshow_with_models(ax=ax[:, 0], **kwargs)
        for a in ax[:, 1]:
            plt.sca(a)
            plt.text(
                0.5,
                0.5,
                "=",
                ha="left",
                va="center",
                fontsize=40,
                transform=a.transAxes,
            )
            plt.axis("off")
        for a in ax[:, 0]:
            a.set_title("$E_{light}$")

    def realization(self, *args, **kwargs):
        mu = self.mu(*args, **kwargs).to_value(self.unit)
        sigma = self.sigma(*args, **kwargs).to_value(self.unit)
        return np.random.normal(mu, sigma) * self.unit


class CalibratedImage(Image):
    unit = u.photon
    type = "calibrated"

    """def mu(self, *args, **kwargs):
        return self.model_mean_calibrated(*args, **kwargs)

    def sigma(self, *args, **kwargs):
        return self.model_sigma_calibrated(*args, **kwargs)"""


class UncalibratedImage(Image):
    unit = u.adu
    type = "uncalibrated"

    """def mu(self, *args, **kwargs):
        return self.model_mean_uncalibrated(*args, **kwargs)

    def sigma(self, *args, **kwargs):
        return self.model_sigma_uncalibrated(*args, **kwargs)"""


class ImageComponent(Image):
    """
    An image Component, with a mean model and an uncertainty.
    """

    keys = []

    def describe_everything_as_string(self):
        return self.describe_as_string()

    def __init__(self, *args, **kwargs):
        Image.__init__(self, *args, **kwargs)

    def model_mean_calibrated(self, *args, **kwargs):
        return np.ones(self.image.shape) * u.photon

    def model_sigma_calibrated(self, *args, **kwargs):
        return np.zeros(self.image.shape) * u.photon

    def model_mean_uncalibrated(self, *args, **kwargs):
        return self.model_mean_calibrated(*args, **kwargs) / self.image.gain

    def model_sigma_uncalibrated(self, *args, **kwargs):
        return self.model_sigma_calibrated(*args, **kwargs) / self.image.gain

    def model_variance_photons(self, *args, **kwargs):
        return self.model_sigma_calibrated(*args, **kwargs) ** 2

    def model_variance_adu(self, *args, **kwargs):
        return self.model_sigma_uncalibrated(*args, **kwargs) ** 2

    def __repr__(self):
        keywords = [f"{k}={getattr(self, k)}" for k in self.keys]
        return f"{self.__class__.__name__}({', '.join(keywords)})"


class Bias(ImageComponent):
    keys = ["bias_level", "read_noise"]

    def __init__(self, bias_level=1000 * u.adu, read_noise=10 * u.electron, **kwargs):
        self.bias_level = bias_level
        self.read_noise = read_noise

        ImageComponent.__init__(self, **kwargs)

    def model_mean_calibrated(self, *args, **kwargs):
        return np.zeros(self.image.shape) * u.photon

    def model_sigma_calibrated(self, *args, **kwargs):
        if self.read_noise.unit == u.adu:
            sigma = self.read_noise * self.image.gain
        else:
            sigma = to_photons(self.read_noise)
        return np.ones(self.image.shape) * sigma

    def model_mean_uncalibrated(self, *args, **kwargs):
        return np.ones(self.image.shape) * self.bias_level

    def model_sigma_uncalibrated(self, *args, **kwargs):
        if self.read_noise.unit == u.adu:
            sigma = self.read_noise
        else:
            sigma = to_photons(self.read_noise) / self.image.gain
        return np.ones(self.image.shape) * sigma


class Dark(ImageComponent):
    keys = ["average_dark_rate"]

    def __init__(self, average_dark_rate=1 * u.photon / u.s, **kwargs):
        self.average_dark_rate = average_dark_rate
        ImageComponent.__init__(self, **kwargs)

    def initialize_dark(self):
        self._dark_rate = (
            np.random.exponential(
                self.average_dark_rate.to_value(u.photon / u.s), self.image.shape
            )
            * u.photon
            / u.s
        )

    @property
    def dark_rate(self):
        try:
            return self._dark_rate
        except AttributeError:
            self.initialize_dark()
            return self._dark_rate

    def model_mean_calibrated(self, t=1 * u.s):
        return np.zeros(self.image.shape) * self.image.unit

    def model_mean_uncalibrated(self, t=1 * u.s):
        return self.dark_rate * t / self.image.gain

    def model_sigma_calibrated(self, t=1 * u.s):
        photons = self.dark_rate * t
        return np.sqrt(photons * u.photon)


class Sky(ImageComponent):
    keys = ["sky_brightness"]

    def __init__(self, sky_brightness=3 * u.photon / u.s, **kwargs):
        self.sky_brightness = sky_brightness
        ImageComponent.__init__(self, **kwargs)

    def model_mean_calibrated(self, t=1 * u.s):
        return self.sky_brightness * np.ones(self.image.shape) * t

    def model_sigma_calibrated(self, t=1 * u.s):
        photons = self.sky_brightness * np.ones(self.image.shape) * t
        return np.sqrt(photons * u.photon)


class Star(ImageComponent):
    keys = ["positions", "brightness"]

    def __init__(
        self,
        positions=[(15, 15)],
        brightness=[1000] * u.photon / u.s,
        **kwargs,
    ):

        self.positions = np.atleast_2d(positions)
        self.brightness = np.atleast_1d(brightness)
        ImageComponent.__init__(self, **kwargs)

    @property
    def starlight_rate(self):
        try:
            return self._starlight_rate
        except AttributeError:
            self.initialize_stars()
            return self._starlight_rate

    def initialize_stars(self):

        x, y = np.meshgrid(
            np.arange(self.image.shape[1]), np.arange(self.image.shape[0])
        )
        N = len(self.positions)
        for i in range(N):
            x_center, y_center = self.positions[i]
            theta = (
                np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
                * u.pixel
                * self.image.pixel_scale
            )
            pixel_area = (self.image.pixel_scale * u.pixel) ** 2
            this_star = self.brightness[i] * self.image.psf(theta) * pixel_area
            try:
                stars += this_star
            except NameError:
                stars = this_star
        self._starlight_rate = stars

    def model_mean_calibrated(self, t=1 * u.s):
        return self.starlight_rate * t

    def model_sigma_calibrated(self, t=1 * u.s):
        photons = self.starlight_rate * t
        return np.sqrt(photons * u.photon)
