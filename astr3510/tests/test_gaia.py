from astr3510 import *


def test_gaia():
    plot_gaia(get_gaia(SkyCoord.from_name("GJ1214"), radius=1 * u.arcmin))
