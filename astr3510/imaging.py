from .imports import *
from .psf import *
from chromatic.rainbows.visualizations.utilities import (
    get_animation_writer_and_displayer,
)
from IPython.display import display
from photutils.aperture import CircularAperture, aperture_photometry


class Image:
    keys = ["shape", "gain", "pixel_scale", "psf"]

    def describe_as_string(self):
        s = ""  # f"{self.__class__.__name__}:\n"
        for k in self.keys:
            s += f"{k}={getattr(self, k)}\n"
        return s

    def describe_everything_as_string(self):
        return "".join([c.describe_as_string() for c in self.components])

    def do_photometry(self, data, error, positions=[(15, 15)], radius=3, **kwargs):
        self.aperture = CircularAperture(positions=positions, r=radius)
        self.aperture_photometry = aperture_photometry(
            data=data, apertures=self.aperture, error=error
        )
        flux = self.aperture_photometry["aperture_sum"].data[0]
        err = self.aperture_photometry["aperture_sum_err"].data[0]
        self.aperture.plot(color="white", linewidth=2)
        plt.text(
            positions[0][0],
            positions[0][1] + 1.2 * radius,
            f"{flux:.1f}$\pm${err:.1f}",
            va="top",
            ha="center",
            color="white",
        )

    def __init__(
        self,
        shape=(30, 30),
        gain=2 * u.electron / u.adu,
        pixel_scale=0.5 * u.arcsec / u.pixel,
        psf=seeing(fwhm=2.0 * u.arcsec),
    ):
        self.unit = u.adu
        self.shape = shape
        self.gain = gain
        self.pixel_scale = pixel_scale
        self.psf = psf
        self.components = []

    def __add__(self, other):
        other.image = self
        self.components.append(other)
        return self

    def __repr__(self):
        return f"Image {self.shape} = {'+'.join([str(x) for x in self.components])}"

    def model_mean_adu(self, t=1 * u.s):
        return (
            np.sum(
                np.array(
                    [x.model_mean_adu(t).to_value(self.unit) for x in self.components],
                    dtype="object",
                ),
                axis=0,
            )
            * self.unit
        )

    def model_variance_adu(self, t=1 * u.s):
        return (
            np.sum(
                np.array(
                    [
                        x.model_variance_adu(t).to_value(self.unit**2)
                        for x in self.components
                    ],
                    dtype="object",
                ),
                axis=0,
            )
            * self.unit**2
        )

    def model_sigma_adu(self, *args, **kwargs):
        return np.sqrt(self.model_variance_adu(*args, **kwargs))

    def realization(self, *args, **kwargs):
        mu = self.model_mean_adu(*args, **kwargs).to_value(self.unit)
        sigma = self.model_sigma_adu(*args, **kwargs).to_value(self.unit)
        return np.random.normal(mu, sigma) * self.unit

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
        e = self.model_sigma_adu(t=time)
        imshowed = plt.imshow(i, **kwargs)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()

        if photometry:
            self.do_photometry(
                data=i.to_value(u.adu),
                error=e.to_value(u.adu),
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
        model = self.model_mean_adu(t=time)
        sigma = self.model_sigma_adu(t=time)
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
                "Data (ADU)",
                "Model ($\mu$, ADU)",
                "Noise ($\sigma$, ADU)",
            ],
        ):
            plt.sca(a)
            imshowed[label] = plt.imshow(i, **kwargs)
            plt.xticks([])
            plt.yticks([])
            plt.ylabel(label)
            plt.colorbar()

            if label == "Data (ADU)":
                if photometry:
                    self.do_photometry(
                        data=data.to_value(u.adu),
                        error=sigma.to_value(u.adu),
                        positions=photometry_positions,
                        radius=photometry_radius,
                    )

        plt.sca(ax[-1])
        texted = plt.text(
            0.1,
            0.9,
            f"exposure_time={time:.1f}\n" + self.describe_as_string(),
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=9,
            clip_on=False,
        )
        plt.axis("off")

        if np.size(t) > 1:
            writer, displayer = get_animation_writer_and_displayer(filename)
            with writer.saving(fi, filename, fi.get_dpi()):
                for time in t:
                    model = self.model_mean_adu(t=time)
                    sigma = self.model_sigma_adu(t=time)
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


Empty = Image
# i = Image() + Bias() + Dark() + Sky() + Stars()


class ImageComponent(Image):
    """
    An image Component, with a mean model and an uncertainty.
    """

    keys = []

    def __init__(self, *args, **kwargs):
        Image.__init__(self, *args, **kwargs)

    def model_mean_adu(self, *args, **kwargs):
        return np.ones(self.image.shape) * self.image.unit

    def model_sigma_adu(self, *args, **kwargs):
        return np.zeros(self.image.shape) * self.image.unit

    def model_variance_adu(self, *args, **kwargs):
        return self.model_sigma_adu(*args, **kwargs) ** 2

    def __repr__(self):
        keywords = [f"{k}={getattr(self, k)}" for k in self.keys]
        return f"{self.__class__.__name__}({', '.join(keywords)})"


class Bias(ImageComponent):
    keys = ["bias_level", "read_noise"]

    def __init__(self, bias_level=1000 * u.adu, read_noise=5 * u.adu, **kwargs):
        self.bias_level = bias_level
        self.read_noise = read_noise
        ImageComponent.__init__(self, **kwargs)

    def model_mean_adu(self, *args, **kwargs):
        return np.ones(self.image.shape) * self.bias_level

    def model_sigma_adu(self, *args, **kwargs):
        return np.ones(self.image.shape) * self.read_noise


class Dark(ImageComponent):
    keys = ["average_dark_rate"]

    def __init__(self, average_dark_rate=1 * u.electron / u.s, **kwargs):
        self.average_dark_rate = average_dark_rate
        ImageComponent.__init__(self, **kwargs)

    def initialize_dark(self):
        self._dark_rate = (
            np.random.exponential(
                self.average_dark_rate.to_value(u.electron / u.s), self.image.shape
            )
            * u.electron
            / u.s
        )

    @property
    def dark_rate(self):
        try:
            return self._dark_rate
        except AttributeError:
            self.initialize_dark()
            return self._dark_rate

    def model_mean_adu(self, t=1 * u.s):
        return self.dark_rate * t / self.image.gain

    def model_sigma_adu(self, t=1 * u.s):
        electrons = self.dark_rate * t
        return np.sqrt(electrons * u.electron) / self.gain


class Sky(ImageComponent):
    keys = ["sky_brightness"]

    def __init__(self, sky_brightness=3 * u.electron / u.s, **kwargs):
        self.sky_brightness = sky_brightness
        ImageComponent.__init__(self, **kwargs)

    def model_mean_adu(self, t=1 * u.s):
        return self.sky_brightness * np.ones(self.image.shape) * t / self.image.gain

    def model_sigma_adu(self, t=1 * u.s):
        electrons = self.sky_brightness * np.ones(self.image.shape) * t
        return np.sqrt(electrons * u.electron) / self.image.gain


class Stars(ImageComponent):
    keys = ["positions", "brightness"]

    def __init__(
        self,
        positions=[(15, 15)],
        brightness=[1000] * u.electron / u.s,
        **kwargs,
    ):

        ImageComponent.__init__(self, **kwargs)

    @property
    def starlight_rate(self):
        try:
            return self._starlight_rate
        except AttributeError:
            self.initialize_stars()
            return self._starlight_rate

    def initialize_stars(
        self, positions=[(15, 15)], brightness=[1000] * u.electron / u.s
    ):

        self.positions = np.atleast_2d(positions)
        self.brightness = np.atleast_1d(brightness)
        x, y = np.meshgrid(
            np.arange(self.image.shape[1]), np.arange(self.image.shape[0])
        )
        N = len(positions)
        for i in range(N):
            x_center, y_center = positions[i]
            theta = (
                np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
                * u.pixel
                * self.image.pixel_scale
            )
            pixel_area = (self.image.pixel_scale * u.pixel) ** 2
            this_star = brightness[i] * self.image.psf(theta) * pixel_area
            try:
                stars += this_star
            except NameError:
                stars = this_star
        self._starlight_rate = stars

    def model_mean_adu(self, t=1 * u.s):
        return self.starlight_rate * t / self.image.gain

    def model_sigma_adu(self, t=1 * u.s):
        electrons = self.starlight_rate * t
        return np.sqrt(electrons * u.electron) / self.image.gain
