"""
Some draft tools for CCD calibrations.
(You should probably ignore these;
check out `imaging.py` instead!)
"""
from .imports import *
from .psf import *


class Camera:
    def __init__(
        self,
        shape=(30, 50),
        bias_level=1000 * u.adu,
        average_dark_rate=1 * u.electron / u.s,
        gain=2 * u.electron / u.adu,
        read_noise=5 * u.adu,
        star_positions=[[12, 20]],
        star_brightnesses=[100] * u.electron / u.s,
        pixel_scale=0.7 * u.arcsec / u.pixel,
        sky_brightness=3 * u.electron / u.s,
        psf=seeing(fwhm=2.0 * u.arcsec),
    ):
        self.shape = shape
        self.bias_level = bias_level
        self.average_dark_rate = average_dark_rate
        self.initialize_dark()
        self.gain = gain
        self.read_noise = read_noise

        self.initialize_photons(
            star_positions=star_positions,
            star_brightnesses=star_brightnesses,
            sky_brightness=sky_brightness,
            pixel_scale=pixel_scale,
            psf=psf,
        )

    def initialize_photons(
        self,
        star_positions=[[12, 20]],
        star_brightnesses=[100] * u.electron / u.s,
        pixel_scale=0.7 * u.arcsec / u.pixel,
        sky_brightness=3 * u.electron / u.s,
        psf=seeing(fwhm=2.0 * u.arcsec),
    ):
        self.star_positions = np.atleast_2d(star_positions)
        self.star_brightnesses = star_brightnesses
        self.sky_brightness = sky_brightness
        self.pixel_scale = pixel_scale
        self.psf = psf
        x, y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        N = len(star_positions)
        for i in range(N):
            x_center, y_center = star_positions[i]
            theta = (
                np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
                * u.pixel
                * self.pixel_scale
            )
            pixel_area = (self.pixel_scale * u.pixel) ** 2
            this_star = star_brightnesses[i] * self.psf(theta) * pixel_area
            try:
                stars += this_star
            except NameError:
                stars = this_star
        self.star_photon_rate = stars
        self.sky_photon_rate = sky_brightness
        self.photon_rate = stars + sky_brightness

    def P_light_model(self, exposure_time=1 * u.s):
        return self.photon_rate * exposure_time

    def P_light_noise_model(self, exposure_time=1 * u.s):
        return self.make_Elight_noise_model(exposure_time) * self.gain

    def P_light(self, exposure_time=1 * u.s):
        mu = self.P_light_model(exposure_time)
        sigma = self.P_light_noise_model(exposure_time)
        return np.random.normal(mu, sigma)

    def initialize_dark(self):
        self.dark_rate = (
            np.random.exponential(
                self.average_dark_rate.to_value(u.electron / u.s), self.shape
            )
            * u.electron
            / u.s
        )

    def make_bias_model(self, *args, **kwargs):
        return np.ones(self.shape) * self.bias_level

    def make_dark_model(self, exposure_time=1 * u.s):
        return self.dark_rate * exposure_time / self.gain + self.make_bias_model()

    def make_light_model(self, exposure_time=1 * u.s):
        return (
            self.photon_rate / self.gain * exposure_time
            + self.dark_rate * exposure_time / self.gain
            + self.make_bias_model()
        )

    def make_bias_noise_model(self, *args, **kwargs):
        return np.ones(self.shape) * self.read_noise

    def make_dark_noise_model(self, exposure_time=1 * u.s):
        photons = self.dark_rate * exposure_time
        return np.sqrt(photons * u.electron) / self.gain

    def make_light_noise_model(self, exposure_time=1 * u.s):
        photons = self.photon_rate * exposure_time
        return np.sqrt(photons * u.electron) / self.gain

    def make_Elight_noise_model(self, exposure_time=1 * u.s):
        return np.sqrt(
            self.make_bias_noise_model() ** 2
            + self.make_dark_noise_model(exposure_time) ** 2
            + self.make_light_noise_model(exposure_time) ** 2
        )

    def make_bias_noise_instance(self, *args, **kwargs):
        return (
            np.random.normal(
                0, self.make_bias_noise_model(*args, **kwargs).to_value(u.adu)
            )
            * u.adu
        )

    def make_dark_noise_instance(self, *args, **kwargs):
        return (
            np.random.normal(
                0, self.make_dark_noise_model(*args, **kwargs).to_value(u.adu)
            )
            * u.adu
        )

    def make_light_noise_instance(self, *args, **kwargs):
        return (
            np.random.normal(
                0, self.make_light_noise_model(*args, **kwargs).to_value(u.adu)
            )
            * u.adu
        )

    def create_images(self, exposure_time=1 * u.s, include_noise=False):
        b = self.make_bias_model()
        d = self.make_dark_model(exposure_time=exposure_time)
        l = self.make_light_model(exposure_time=exposure_time)

        if include_noise:
            b_noise = self.make_bias_noise_instance(exposure_time=exposure_time)
            b += b_noise
            d_noise = self.make_dark_noise_instance(exposure_time=exposure_time)
            d += d_noise + b_noise
            l_noise = self.make_light_noise_instance(exposure_time=exposure_time)
            l += l_noise + d_noise + b_noise
        return b, d, l

    def imshow_exposures(
        self, exposure_time=1 * u.s, include_noise=False, vmin=None, vmax=None
    ):

        b, d, l = self.create_images(
            exposure_time=exposure_time, include_noise=include_noise
        )
        if vmin is None:
            vmin = np.min([b, d, l])
        if vmax is None:
            vmax = np.max([b, d, l])

        fi, ax = plt.subplots(
            1,
            4,
            figsize=(8, 3),
            dpi=600,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        for i, x, t in zip(
            range(3), [b, d, l], ["bias", "dark + bias", "light + dark + bias"]
        ):
            imshowed = ax[i].imshow(x, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax[i].set_title(t)
            plt.colorbar(
                imshowed,
                orientation="horizontal",
                ax=ax[i],
                label="Pixel Brightness (ADU)",
            )
        for a in ax:
            a.set_xticks(np.arange(-0.5, self.shape[1], 1))
            a.set_yticks(np.arange(-0.5, self.shape[0], 1))
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.grid(color="black", linewidth=1)

        a = ax[-1]
        plt.sca(a)
        a.axis("off")
        if include_noise:
            noise_string = f"""
with noise of...
{np.mean(self.make_bias_noise_model(exposure_time=exposure_time)):.1f} read noise
{np.mean(self.make_dark_noise_model(exposure_time=exposure_time)):.1f} dark noise on average
{np.max(self.make_light_noise_model(exposure_time=exposure_time)):.1f} photon noise at peak
..included"""

        else:
            noise_string = "with no noise included"
        s = f"""
exposure time = {exposure_time}

(rows, columns) = {self.shape}
gain = {self.gain}
average bias level = {self.bias_level}
average dark rate = {self.average_dark_rate}
brightest star = {np.max(self.star_brightnesses)}

{noise_string}"""
        plt.text(0, 0, s, transform=a.transAxes, ha="left", va="bottom", fontsize=6)

    def plot_pixels_with_exposure_time(
        self,
        exposure_times=np.linspace(0, 10) * u.s,
        include_noise=False,
        vmin=None,
        vmax=None,
    ):

        N = len(exposure_times)
        rows, cols = self.shape
        b_cube = np.zeros((N, rows, cols))
        d_cube = np.zeros((N, rows, cols))
        l_cube = np.zeros((N, rows, cols))
        for i, e in enumerate(exposure_times):
            b_cube[i, :, :], d_cube[i, :, :], l_cube[i, :, :] = self.create_images(
                exposure_time=e, include_noise=include_noise
            )

        for title in ["bias", "bias + dark", "bias + dark + light"]:
            fi, ax = plt.subplots(
                self.shape[0],
                self.shape[1],
                constrained_layout=True,
                dpi=600,
                sharex=True,
                sharey=True,
                figsize=(8, 6),
            )
            for i in range(rows):
                for j in range(cols):
                    ax[i, j].plot(
                        exposure_times,
                        b_cube[:, i, j],
                        label="bias",
                        linewidth=2,
                        alpha=0.5,
                        linestyle=":",
                    )
                    plt.ylim(0.95 * np.min(b_cube), 1.05 * np.max(b_cube))
                    if "dark" in title:
                        ax[i, j].plot(
                            exposure_times,
                            d_cube[:, i, j],
                            label="dark",
                            linewidth=2,
                            alpha=0.5,
                            linestyle="--",
                        )
                        plt.ylim(0.95 * np.min(b_cube), 1.05 * np.max(d_cube))
                    if "light" in title:
                        ax[i, j].plot(
                            exposure_times,
                            l_cube[:, i, j],
                            label="light",
                            linewidth=2,
                            alpha=0.5,
                            linestyle="-",
                        )
                        plt.ylim(0.95 * np.min(b_cube), 1.05 * np.max(l_cube))
            fi.suptitle(title)
            fi.supylabel("Pixel Brightness (ADU)")
            fi.supxlabel("Exposure Time (s)")
            plt.sca(ax[0, -1])
            plt.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
