
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from astropy import constants as const
from astropy import units as u

kb = const.k_B.cgs.value
mp = const.m_p.cgs.value

from matplotlib import rc, rcParams
from matplotlib.font_manager import fontManager, FontProperties
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text',usetex=True)
rcParams['text.latex.preamble'] = [r'\usepackage{siunitx}', r'\sisetup{detect-all}', r'\usepackage{helvet}', r'\usepackage{sansmath}', r'\sansmath']
rcParams.update({'font.size': 14})

import matplotlib.ticker as ticker

# d is in unit of pixels
# pixel size is 0.8 arcsec
d, trot, drot, sig, dsig = np.loadtxt('pvcut_fit_I18308.txt', usecols=(0,1,2,3,4), unpack=True)

dist = 4.6 # kpc
d = d * 0.8 / 3600. / 180. * np.pi * dist * 1e3

# Now plot:
fig = plt.figure(figsize=(7, 6))
gs1 = gs.GridSpec(2, 1)
gs1.update(wspace=0.0, hspace=0.0)
plt.subplots_adjust(wspace = 0, hspace = 0)

# The first subplot: Trot along filaments
pltrot = fig.add_subplot(gs1[0])

pltrot.plot(d,trot,color='blue')
pltrot.fill_between(d, trot+drot, trot-drot, facecolor='grey', interpolate=True)

xcoords = np.array([8.82,24.2,35.57,41.34,47.97,62.03])
xcoords = xcoords * 0.8 / 3600. / 180. * np.pi * dist * 1e3
for xc in xcoords:
	pltrot.axvline(x=xc,linestyle='--',color='r')

pltrot.set_xlim(0,1.195)
pltrot.set_ylabel(r'$T_\mathrm{rot}$ (K)')
pltrot.tick_params(which='both',top='on',right='on',direction='in')
#pltrot.legend(loc=9)
pltrot.text(0.02,24,'I18308',color='black',fontsize='large',weight='black')
pltrot.set_xticklabels(())
pltrot.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
pltrot.yaxis.set_minor_locator(ticker.MultipleLocator(1))

# The second subplot: sigmav along filaments
plsigma = fig.add_subplot(gs1[1])

sig = np.sqrt(sig**2 - (0.618/2.355)**2)
sig = np.sqrt((sig*1e5)**2 - kb*trot/17.0/mp) / 1e5
plsigma.plot(d,sig,color='blue',label=r'Non-thermal')
plsigma.fill_between(d, sig+dsig, sig-dsig, facecolor='grey', interpolate=True)

# Plot thermal linewidth
thermalv = np.sqrt(kb*trot/2.33/mp) / 1e5
plsigma.plot(d,thermalv,color='darkorange',label=r'Thermal')

for xc in xcoords:
	plsigma.axvline(x=xc,linestyle='--',color='r')

corelist = ['c4','c8','c5','c7','c2','c1']
ycoords = [0.1] * 6
for core,xc,yc in zip(corelist,xcoords,ycoords):
	plsigma.text(xc,yc,core,color='black',fontsize='large',weight='black')

plsigma.set_xlim(0,1.195)
plsigma.set_ylim(0,0.75)
plsigma.set_xlabel('Offset (pc)')
plsigma.set_ylabel(r'$\sigma_\mathrm{nth}$ (km s$^{-1}$)')
plsigma.tick_params(which='both',top='on',right='on',direction='in')
plsigma.legend(loc=9)
plsigma.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
plsigma.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

plt.tight_layout()
plt.draw()

fig.savefig('filparameter_I18308.pdf')

plt.close()
