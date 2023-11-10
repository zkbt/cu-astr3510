from ..imports import *
from ..gaia import get_gaia
from .apertures import *
from astropy.coordinates import SkyCoord 
from astropy import wcs
from astropy.io import fits
import astropy.units as u

from astropy.convolution import AiryDisk2DKernel, convolve, convolve_fft, Moffat2DKernel, CustomKernel
from astropy.visualization import imshow_norm, ManualInterval, MinMaxInterval, SqrtStretch, LogStretch


# FIXME, synthesize this with imaging.py
class SimulatedImage:
    def __init__(self, 
                 name='16 Cyg',
                 fov = 2*u.arcminute, 
                 pixel_size = 0.1*u.arcsecond, 
                 ):
        
        self.fov = fov 
        self.pixel_size = pixel_size 
        self.N = int((fov/pixel_size).decompose())

        self.setup_stars(name=name)
        self.setup_wcs()
        self.setup_pixels()

    def setup_stars(self, name):    
        self.name = name 
        self.stars = get_gaia(name, radius=self.fov)

    def setup_wcs(self):
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [self.N/2, self.N/2]
        center = self.stars.meta['center']
        w.wcs.cdelt = [self.pixel_size.to_value('deg'), self.pixel_size.to_value('deg')]
        w.wcs.crval = [center.ra.deg, center.dec.deg]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
       
        self.wcs = w 
        self.header = w.to_header()

    def setup_pixels(self):
        '''
        Set up some pxiels, with stars as single pixels. 
        '''
        # calculate pixel locations for stars
        coordinates = SkyCoord(ra=self.stars['ra'], dec=self.stars['dec'])
        x, y = self.wcs.world_to_pixel(coordinates)
        mag = self.stars[f"G_gaia_mag"].to_value("mag")


        image = np.zeros((self.N,self.N))
        flux_zero_mag = 1e10
        star_flux = flux_zero_mag*10**(-0.4*mag)

        # make single-pixel stars
        row = y.astype(int)
        col = x.astype(int)
        ok = (row >= 0) * (row < self.N) * (col >= 0) * (col < self.N)
        image[row[ok], col[ok]] = star_flux[ok]

        self.image_single_pixels = image 

    def simulate(self, aperture=CircleAperture, read_noise=1, **psf_kw):
        psf_kw['dtheta'] = self.pixel_size
        self.aperture = aperture(**psf_kw)
        self.kernel = CustomKernel(self.aperture.psf)
        self.image_model = convolve_fft(self.image_single_pixels, self.kernel)
        noise = np.sqrt(self.image_model + read_noise**2)
        self.image = np.random.normal(self.image_model, noise)
        filename = 'test.fit'
        f = fits.ImageHDU(self.image, header=self.header)
        f.header['object'] = (self.name, 'where is the field centered?')
        #f.header['simulate'] = (title, 'what PSF was simulated?')
        f.header['scale'] = (self.pixel_size.to_value('arcsecond'), 'pixel scale in units of arcseconds/pixel')
        f.writeto(filename, overwrite=True)
        self.display()


    # FIXME!
    def display(self, title='Impossibly Good', kernel=None, vmin=1e-10, vmax=1e-3):
        
        extent = np.array([-self.N/2, self.N/2, -self.N/2, self.N/2])*self.pixel_size.to_value('arcsec')

        plt.figure()
        im, norm = imshow_norm(self.image, origin='lower',
                            interval=ManualInterval(vmin, vmax), 
                            stretch=LogStretch(), extent=extent)
        plt.xlabel('x [arcsec]')
        plt.ylabel('y [arcsec]')
        plt.colorbar()
        #plt.savefig(f'{title}.pdf')
