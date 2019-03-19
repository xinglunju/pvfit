
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc, rcParams
from matplotlib.font_manager import fontManager, FontProperties
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text',usetex=True)
rcParams['text.latex.preamble'] = [
    r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
	r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
	r'\usepackage{helvet}',    # set the normal font here
	r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
	r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]
rcParams.update({'font.size': 20})
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

#import matplotlib
#matplotlib.use('TkAgg')
import aplpy
from astropy.io import fits

def readfits(inputim):
	img = fits.open(inputim)
	dat = img[0].data
	hdr = img[0].header

	## The axes loaded by pyfits are in the reversed order as in the header.
	#inaxis = hdr['NAXIS']
	## Images produced by CASA may have an addtional axis for Stokes parameters
	## which is often useless.
	#if naxis == 3:
	#	data = np.swapaxes(dat[0],0,1)
	## Otherwise just swap the first and the second axis.
	#elif naxis == 2:
	#	data = np.swapaxes(dat,0,1)

	return dat, hdr
	img.close()

data1, hdr1 = readfits('I18308_11_slice.fits')
data2, hdr2 = readfits('I18308_22_slice.fits')

dist = 4.6 # kpc

data1 = data1[::-1,]
hdr1['CDELT2'] = -hdr1['CDELT2'] / 1e3
hdr1['CRPIX2'] = hdr1['NAXIS2']
hdr1['CRVAL2'] = hdr1['CRVAL2'] / 1e3
hdr1['CDELT1'] = (hdr1['CDELT1'] / 180. * np.pi * dist * 1e3, '[pc] Coordinate increment at reference poin')
hdr1['CRVAL1'] = (0.0, '[pc] Coordinate value at reference point')
hdr1['CUNIT1'] = 'pc'
fits.writeto('I18308_11_slice_flip.fits',data1,hdr1,overwrite=True)

data2 = data2[::-1,]
hdr2['CDELT2'] = -hdr2['CDELT2'] / 1e3
hdr2['CRPIX2'] = hdr2['NAXIS2']
hdr2['CRVAL2'] = hdr2['CRVAL2'] / 1e3
hdr2['CDELT1'] = (hdr2['CDELT1'] / 180. * np.pi * dist * 1e3, '[pc] Coordinate increment at reference poin')
hdr2['CRVAL1'] = (0.0, '[pc] Coordinate value at reference point')
hdr2['CUNIT1'] = 'pc'
fits.writeto('I18308_22_slice_flip.fits',data2,hdr2,overwrite=True)

fig = plt.figure(figsize=(6, 8))

f1 = aplpy.FITSFigure('I18308_11_slice_flip.fits',figure=fig,subplot=[0.12,0.545,0.86,0.43])
f1.show_colorscale(vmin=0,cmap='gist_heat_r',aspect='auto')
f1.recenter(0.6, 76, width=1.2, height=32.72)
f1.show_contour('I18308_11_slice_flip.fits',levels=[0.005,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085],colors='green',linewidths=0.8)
f1.axis_labels.set_ytext(r'Vlsr (km s$^{-1}$)')
f1.axis_labels.set_ypad(0)

f1.add_label(0.05,0.92,'I18308',relative=True,horizontalalignment='left',color='black',size='large',weight=1000)
f1.add_label(0.05,0.82,r'NH$_3$ (1,1) P-V',relative=True,horizontalalignment='left',color='black',fontsize='large',weight='black')
f1.ticks.set_color('black')
f1.ticks.set_minor_frequency(4)

f2 = aplpy.FITSFigure('I18308_22_slice_flip.fits',figure=fig,subplot=[0.12,0.115,0.86,0.43])
f2.show_colorscale(vmin=0,cmap='gist_heat_r',aspect='auto')
f2.recenter(0.6, 82, width=1.2, height=32.72)
f2.show_contour('I18308_22_slice_flip.fits',levels=[0.005,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085],colors='green',linewidths=0.8)
f2.axis_labels.set_ytext(r'Vlsr (km s$^{-1}$)')
f2.axis_labels.set_xtext('Offset (pc)')
f2.axis_labels.set_ypad(0)

f2.add_label(0.05,0.9,r'NH$_3$ (2,2) P-V',relative=True,horizontalalignment='left',color='black',fontsize='large',weight='black')
f2.ticks.set_color('black')
f2.ticks.set_minor_frequency(4)

f1.hide_xaxis_label()
f1.hide_xtick_labels()

fig.canvas.draw()
#fig.set_rasterized(True)
fig.savefig('I18308_pvdiagram.pdf')
