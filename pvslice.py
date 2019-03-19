
from pvextractor import Path
from pvextractor import extract_pv_slice
from astropy import units as u
from astropy.coordinates import FK5
import numpy as np

ra = [278.3947241,278.3928173,278.3896708,278.3882779,278.387925,278.3874497,278.3881945,278.3888875]
dec= [-8.644493589,-8.645027916,-8.646443333,-8.648562055,-8.649796945,-8.651193056,-8.65423175,-8.655833333]

for i in range(len(ra)-1):
	dra  = ra[i+1] - ra[i]
	ddec = dec[i+1] - dec[i]
	sepa = np.sqrt((dra*np.cos(dec[i]/180.*np.pi))**2 + (ddec)**2)
	angle = np.arctan(ddec / (dra*np.cos(dec[i]/180.*np.pi))) / np.pi * 180.
	sepa *= 3600.
	sepa_pix = sepa/0.8
	sepa_pc = sepa/3600./180.*np.pi*4600.
	print "Separation between points: %.2f arcsec or %.2f pixels" % (sepa, sepa_pix)
	print "Separation between points: %.2f pc" % (sepa_pc)
	print "Position angles of segments: %.2f deg" % (90 - angle)

g = FK5(ra * u.deg, dec * u.deg)

path0 = Path(g, width=5 * u.arcsec)

slice0 = extract_pv_slice('18308_11_line.fits', path0) 

slice0.writeto('I18308_11_slice.fits',overwrite=True)

slice1 = extract_pv_slice('18308_22_line.fits', path0) 

slice1.writeto('I18308_22_slice.fits',overwrite=True)
