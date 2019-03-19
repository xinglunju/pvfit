
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from scipy import optimize

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
rcParams.update({'font.size': 14})

import matplotlib.ticker as ticker

# d is in unit of pixels
# pixel size is 0.8 arcsec
d, t, dt = np.loadtxt('pvcut_fit_I18308.txt', usecols=(0,5,6), unpack=True)

dist = 4.6 # kpc
d = d * 0.8 / 3600. / 180. * np.pi * dist * 1e3

# Fit d with two segments of linear relations
# Use np.piecewise to find the turning point
def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
errfunc = lambda p, x, y, err: (y - piecewise_linear(x, p[0], p[1], p[2], p[3])) / err
pinit = [0.6, 76.1, -1.7, 2.8]
out = optimize.leastsq(errfunc, pinit, args=(d, t, dt))
pfinal = out[0]
print "Turning offset, turning Vlsr, 1st gradient, 2nd gradient"
print pfinal
t_fit = piecewise_linear(d, *pfinal)

t_res = t - t_fit

# Now plot:
#fig = plt.figure(211, figsize=(7, 6))
fig = plt.figure(figsize=(7, 6))
gs1 = gs.GridSpec(2, 1)
gs1.update(wspace=0.0, hspace=0.0)
plt.subplots_adjust(wspace = 0, hspace = 0)

# The first subplot: raw data pv
#raw_pv = fig.add_subplot(211)
raw_pv = fig.add_subplot(gs1[0])

raw_pv.plot(d,t,label=r'NH$_3$ Vlsr',color='blue')
raw_pv.plot(d,t_fit,label='Linear fit',color='k')

raw_pv.fill_between(d, t+dt, t-dt, facecolor='grey', interpolate=True)

raw_pv.set_xlim(0,1.195)
raw_pv.set_ylim(74.8,77.6)
raw_pv.set_ylabel(r'Vlsr (km s$^{-1}$)')
raw_pv.tick_params(which='both',top='on',right='on',direction='in')
raw_pv.legend(loc=9)
raw_pv.text(0.02,77.3,'I18308',color='black',fontsize='large',weight='black')
raw_pv.set_xticklabels(())
raw_pv.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))

# The second subplot: global gradient subtracted pv
processed_pv = fig.add_subplot(gs1[1])

processed_pv.plot(d,t_res,label=r'Residual NH$_3$ Vlsr',color='blue')
processed_pv.fill_between(d, t_res+dt, t_res-dt, facecolor='grey', interpolate=True)
# Offset of dense cores
xcoords = np.array([8.82,24.2,35.57,41.34,47.97,62.03])
xcoords = xcoords * 0.8 / 3600. / 180. * np.pi * dist * 1e3
xerr = np.array([.0257,.0354,.0388,.0393,.06,.041])
for xc in xcoords:
	raw_pv.axvline(x=xc,linestyle='--',color='r')
	processed_pv.axvline(x=xc,linestyle='--',color='r')
# Vlsr of dense cores
ycoords = np.array([76.15,76.38,75.11,75.25,75.5,76.6])
yerr =    np.array([0.17, 1.1,  0.31, 0.11, 0.20,0.09])
#for xc, yc in zip(xcoords,ycoords):
#raw_pv.plot(xcoords, ycoords, 'bx', markersize=12, markeredgewidth=3)
raw_pv.errorbar(xcoords, ycoords, xerr=xerr, yerr=yerr, fmt='r.', zorder=100, elinewidth=3)
corelist = ['c4','c8','c5','c7','c2','c1']
ycoords = [-0.24] * 6
for core,xc,yc in zip(corelist,xcoords,ycoords):
	processed_pv.text(xc,yc,core,color='black',fontsize='large',weight='black')

processed_pv.set_xlim(0,1.195)
processed_pv.set_ylim(-0.28,0.21)
processed_pv.set_xlabel('Offset (pc)')
processed_pv.set_ylabel(r'Residual Vlsr (km s$^{-1}$)')
processed_pv.tick_params(which='both',top='on',right='on',direction='in')
processed_pv.legend(loc=9)
processed_pv.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))

plt.tight_layout()
plt.show()

#fig.set_rasterized(True)
fig.savefig('pv_I18308.pdf')
