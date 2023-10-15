from astr3510.psfs import * 
from astr3510.tests.setup_test import *

def test_many_psfs():
    for a in apertures_to_test:
        a().imshow_aperture_and_psf()
        plt.savefig(os.path.join(test_directory, f'psf-test-{a.__name__}.pdf'))

def test_against_airy(wavelength=0.5*u.um, diameter=1*u.m):

    def plot_diffraction_limit():
        t = np.linspace(0, 2*np.pi)
        r = np.arcsin(1.22*wavelength/diameter).to('arcsec')
        plt.plot(r*np.sin(t), r*np.cos(t), color='white', linestyle='--')

    fi, ax = plt.subplots(1, 2, figsize=(8,3), constrained_layout=True, sharex=True, sharey=True)

    plt.sca(ax[0])
    c = CircleAperture(diameter=diameter, wavelength=wavelength)
    c.imshow_psf()
    plot_diffraction_limit()
    plt.colorbar()

    plt.sca(ax[1])
    a = Airy(diameter=diameter, wavelength=wavelength)
    a.imshow_psf()
    plot_diffraction_limit()        
    plt.colorbar()

    plt.savefig(os.path.join(test_directory, 'psf-test-fft+airy.pdf'))

