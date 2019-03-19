import os, time
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants, units
from lmfit import minimize, Parameters, report_fit
from astropy.io import fits

start = time.clock()
print 'Start the timer...'

# Define some useful constants first:
c = constants.c.cgs.value # Speed of light (cm/s)
k_B = constants.k_B.cgs.value # Boltzmann coefficient (erg/K)
h = constants.h.cgs.value # Planck constant (erg*s)

def nh3_read(fitsfile):
	try:
		img = fits.open(fitsfile)
	except IOError:
		return False
	meta = img[0].data
	hdr  = img[0].header
	if hdr['naxis'] == 4:
		meta = meta[0,:,:,:]
		hdr.remove('naxis4')
		hdr.remove('crpix4')
		hdr.remove('cdelt4')
		hdr.remove('crval4')
		hdr.remove('ctype4')
		hdr['naxis'] = 3
		print 'Extra axis removed. NH3 file sucessfully loaded.'
	elif hdr['naxis'] <= 3:
		print 'NH3 file sucessfully loaded.'
	else:
		print 'The file is unlikely a cube. Find the correct one please!'
	return meta,hdr
	img.close()

def nh3_load_axes(header):
	"""
	nh3_load_axes(header)
	Load the axes based on the header.
	"""
	# Y axis, which is Vlsr
	v0 = header['crval2'] / 1e3
	v0pix = int(header['crpix2'])
	vaxis = onevpix * (np.arange(naxisy)+1-v0pix) + v0
	return vaxis

def nh3_init():
	"""
	nh3_init()
	Initialize the parameters. No input keywords.
	"""
	nh3_info = {}
	nh3_info['E'] = [23.4, 64.9, 125., 202., 298., 412.] # Upper level energy (Kelvin)
	nh3_info['frest'] = [23.6944955, 23.7226333, 23.8701292, \
	                     24.1394163, 24.5329887, 25.0560250] # Rest frequency (GHz)
	nh3_info['b0'] = 298.117 # 
	nh3_info['c0'] = 186.726 # Rotational constants (GHz)
	nh3_info['mu'] = 1.468e-18 # Permanet dipole moment (esu*cm)
	nh3_info['gI'] = [2./8, 2./8, 4./8, 2./8, 2./8, 4./8]
	nh3_info['Ri'] = [[0.5,5./36,1./9], [56./135+25./108+3./20,7./135,1./20], \
					[0.8935,0,0], [0.935,0,0], [0.956,0,0], [0.969,0,0]] # Relative strength
	nh3_info['vsep'] = [[7.522,19.532], [16.210,26.163], 21.49, 24.23, 25.92, 26.94]
	# Velocity separations between hyperfine lines (km/s)

	return nh3_info

def gauss_tau(axis,p):
	"""
	Genenerate a Gaussian model given an axis and a set of parameters.
	p: [T, Ntot, vlsr, sigmav, J, hyper, ff]
	hyper = 0 -- main
	hyper =-1 -- left inner satellite
	hyper = 1 -- right inner satellite
	hyper =-2 -- left outer satellite
	hyper = 2 -- right outer satellite
	"""
	T = p[0]; Ntot = p[1]; vlsr = p[2]; sigmav = p[3]; J = p[4]; hyper = p[5]; ff = p[6]
	K = J
	gk = 2

	if hyper > 0:
		vlsr = vlsr + nh3_info['vsep'][J-1][hyper-1]
	elif hyper < 0:
		vlsr = vlsr - nh3_info['vsep'][J-1][abs(hyper)-1]

	phijk = 1/np.sqrt(2 * np.pi) / (sigmav * 1e5) * np.exp(-0.5 * (axis - vlsr)**2 / sigmav**2)
	Ri = nh3_info['Ri'][J-1][abs(hyper)]
	Ajk = (64 * np.pi**4 * (nh3_info['frest'][J-1] * 1e9)**3 * nh3_info['mu']**2 / 3 / h / c**3) * (K**2) / (J * (J + 1)) * Ri
	gjk = (2*J + 1) * gk * nh3_info['gI'][J-1]
	b0 = nh3_info['b0']
	c0 = nh3_info['c0']
	Q = 168.7*np.sqrt(T**3/b0**2/c0)
	Njk = Ntot * (gjk / Q) * np.exp(-1.0 * nh3_info['E'][J-1] / T)

	tau = (h * c**3 * Njk * Ajk) / (8 * np.pi * nh3_info['frest'][J-1]**2 * 1e18 * k_B * T) * phijk
	f = (T-2.73) * ff * (1 - np.exp(-1.0 * tau))

	return f

def tau(Ntot, sigmav, T, J, hyper):
	"""
	Calculate tau
	"""
	K = J; gk = 2
	phi = 1/np.sqrt(2 * np.pi) / (sigmav * 1e5)
	Ri = nh3_info['Ri'][J-1][abs(hyper)]
	Ajk = (64 * np.pi**4 * (nh3_info['frest'][J-1] * 1e9)**3 * nh3_info['mu']**2 / 3 / h / c**3) * (K**2) / (J * (J + 1)) * Ri
	gjk = (2*J + 1) * gk * nh3_info['gI'][J-1]
	b0 = nh3_info['b0']
	c0 = nh3_info['c0']
	Q = 168.7*np.sqrt(T**3/b0**2/c0)
	Njk = Ntot * (gjk / Q) * np.exp(-1.0 * nh3_info['E'][J-1] / T)
	tau = (h * c**3 * Njk * Ajk) / (8 * np.pi * nh3_info['frest'][J-1]**2 * 1e18 * k_B * T) * phi

	return tau

def model_11(params, vaxis, spec):
	"""
	Model hyperfine components of NH3 (1,1) and (2,2)
	then subtract data.
	"""
	T = params['T'].value
	Ntot = params['Ntot'].value
	vlsr = params['vlsr'].value
	sigmav = params['sigmav'].value
	ff = params['ff'].value

	vlsr22 = vlsr + 35.

	model = gauss_tau(vaxis,[T,Ntot,vlsr,sigmav,1,0,ff]) + \
			gauss_tau(vaxis,[T,Ntot,vlsr,sigmav,1,-1,ff]) + \
			gauss_tau(vaxis,[T,Ntot,vlsr,sigmav,1,1,ff]) + \
			gauss_tau(vaxis,[T,Ntot,vlsr22,sigmav,2,0,ff]) + \
			gauss_tau(vaxis,[T,Ntot,vlsr22,sigmav,2,1,ff])

	return model - spec

def model_11_2c(params, vaxis, spec):
	"""
	Model for two velocity components
	"""
	temp = model_11(params, vaxis, spec)
	T2 = params['T2'].value
	Ntot2 = params['Ntot2'].value
	vlsr2 = params['vlsr2'].value
	sigmav2 = params['sigmav2'].value
	ff2 = params['ff2'].value

	vlsr222 = vlsr2 + 35.

	model = gauss_tau(vaxis,[T2,Ntot2,vlsr2,sigmav2,1,0,ff2]) + \
			gauss_tau(vaxis,[T2,Ntot2,vlsr2,sigmav2,1,-1,ff2]) + \
			gauss_tau(vaxis,[T2,Ntot2,vlsr2,sigmav2,1,1,ff2]) + \
			gauss_tau(vaxis,[T2,Ntot2,vlsr222,sigmav2,2,0,ff2]) + \
			gauss_tau(vaxis,[T2,Ntot2,vlsr222,sigmav2,2,1,ff2])
	
	return temp + model

clickvalue = []
def onclick(event):
	print 'The Vlsr you select: %f' % event.xdata
	clickvalue.append(event.xdata)

def fit_spec(spec1, spec2, vaxis1, vaxis2, cutoff=0.009, varyv=2, interactive=True, mode='single'):
	"""
	fit_spec(spec1, spec2, spec3, spec4, spec5)
	Fit the five NH3 spectra simultaneously, derive best-fitted temperature and column density.
	"""
	if interactive:
		plt.ion()
		f = plt.figure(figsize=(8,6))
		ax = f.add_subplot(111)

	spec1 = spec1[5:-5]
	spec2 = spec2[5:-5]
	cutoff = cutoff
	vaxis1 = vaxis1[5:-5]
	vaxis2 = vaxis2[5:-5]
	spec = np.concatenate((spec1, spec2))
	vaxis = np.concatenate((vaxis1, vaxis2+35.0))

	Trot = 0; sigmav = 0; vpeak = 0
	dTrot = 0; dsigmav = 0; dvpeak = 0
	Trot2 = 0; sigmav2 = 0; vpeak2 = 0
	dTrot2 = 0; dsigmav2 = 0; dvpeak2 = 0

	unsatisfied = True
	while unsatisfied:
		if interactive:
			f.clear()
			plt.plot(vaxis, spec, 'k-', label='Spectrum')
			plt.tick_params(top='on')
			cutoff_line = [cutoff] * len(vaxis)
			cutoff_line_minus = [-1.0*cutoff] * len(vaxis)
			plt.plot(vaxis, cutoff_line, 'r--')
			plt.plot(vaxis, cutoff_line_minus, 'r--')
			plt.xlabel(r'$V_\mathrm{lsr}$ (km s$^{-1}$)', fontsize=15, labelpad=0)
			plt.ylabel(r'$T_\mathrm{B}$ (K)', fontsize=15)
			plt.text(-0.02, 1.05, sourcename, transform=ax.transAxes, color='r', fontsize=15)
			plt.tight_layout()
			#plt.ylim([-1,16])
			#clickvalue = []
			if mode == 'single':
				#cid = f.canvas.mpl_connect('button_press_event', onclick)
				#raw_input('Click on the plot to select Vlsr...')
				#print clickvalue
				#if len(clickvalue) >= 1:
				#	print 'Please select at least one velocity! The last one will be used.'
				#	vlsr1 = clickvalue[-1]
				#elif len(clickvalue) == 0:
				#	vlsr1 = -999
				print 'Input one velocity manually:'
				manualv = raw_input()
				manualv = manualv.split()
				if len(manualv) == 1:
					vlsr1 = np.float_(manualv)
				else:
					print 'Invalid input...'
				print 'The Vlsr is %0.2f km/s' % vlsr1
				#raw_input('Press any key to start fitting...')
				#f.canvas.mpl_disconnect(cid)
				vlsr2 = -999
			elif mode == 'double':
				#cid = f.canvas.mpl_connect('button_press_event', onclick)
				#raw_input('Click on the plot to select Vlsrs...')
				#print clickvalue
				#if len(clickvalue) >= 2:
				#	print 'Please select at least two velocities! The last two will be used.'
				#	vlsr1,vlsr2 = clickvalue[-2],clickvalue[-1]
				#elif len(clickvalue) == 1:
				#	vlsr1 = clickvalue[-1]
				#	vlsr2 = -9999
				#elif len(clickvalue) == 0:
				#	vlsr1,vlsr2 = -9999,-9999
				print 'Input two velocities manually:'
				manualv = raw_input()
				manualv = manualv.split()
				if len(manualv) == 2:
					vlsr1,vlsr2 = np.float_(manualv)
				else:
					print 'Invalid input...'
				print 'The two Vlsrs are %0.2f km/s and %0.2f km/s.' % (vlsr1,vlsr2)
				#raw_input('Press any key to start fitting...')
				#f.canvas.mpl_disconnect(cid)
			else:
				vlsr1,vlsr2 = -999,-999
		else:
			if mode == 'single':
				if spec_low.max() >= cutoff:
					vlsr1 = __xcorrelate__(spec_low, vaxis_low)
					if vlsr1 <=82 or vlsr1 >=92:
						vlsr1 = -999
					if spec_low[np.abs(vaxis_low - vlsr1).argmin()] <= cutoff:
						vlsr1 = -999
					if spec_low[np.abs(vaxis_low - vlsr1 + 7.47385).argmin()] <= cutoff and spec_low[np.abs(vaxis_low - vlsr1 - 7.56923).argmin()] <= cutoff:
						vlsr1 = -999
					if spec_upp[np.abs(vaxis_low - vlsr1).argmin()] <= cutoff:
						vlsr1 = -999
				else:
					vlsr1 = -999
				vlsr2 = -999
			elif mode == 'double':
				vlsr1,vlsr2 = 86.0,88.0
			else:
				vlsr1,vlsr2 = -999,-999

		# Add 5 parameters
		params = Parameters()
		if vlsr1 != -999:
			params.add('Ntot', value=1e16, min=1e13, max=1e18)
			params.add('T', value=30, min=5, max=100)
			params.add('sigmav', value=0.5, min=0.1, max=5.0)
			if varyv > 0:
				params.add('vlsr', value=vlsr1, min=vlsr1-varyv*onevpix, max=vlsr1+varyv*onevpix)
			elif varyv == 0:
				params.add('vlsr', value=vlsr1, vary=False)
			params.add('ff', value=0.5, min=0, max=1.0)
		# another 5 parameters for the second component
		if vlsr2 != -999:
			params.add('Ntot2', value=1e16, min=1e13, max=1e18)
			params.add('T2', value=30, min=5, max=100)
			params.add('sigmav2', value=0.5, min=0.1, max=5.0)
			if varyv > 0:
				params.add('vlsr2', value=vlsr2, min=vlsr2-varyv*onevpix, max=vlsr2+varyv*onevpix)
			elif varyv == 0:
				params.add('vlsr2', value=vlsr2, vary=False)
			params.add('ff2', value=0.5, min=0, max=1.0)

		# Run the non-linear minimization
		if vlsr1 != -999 and vlsr2 != -999:
			result = minimize(model_11_2c, params, args=(vaxis, spec))
		elif vlsr1 != -999 or vlsr2 != -999:
			result = minimize(model_11, params, args=(vaxis, spec))
		else:
			unsatisfied = False
			continue
		
		final = spec + result.residual
		#report_fit(params)

		if vlsr2 == -999:
			plt.text(-0.02, 0.95, r'$V_\mathrm{lsr}$=%.2f($\pm$%.2f) km/s' % (result.params['vlsr'].value,result.params['vlsr'].stderr), transform=ax.transAxes, color='r', fontsize=15)
		if vlsr2 != -999:
			plt.text(-0.02, 0.95, r'$V_\mathrm{lsr}$=%.2f($\pm$%.2f) km/s' % (result.params['vlsr'].value,result.params['vlsr'].stderr), transform=ax.transAxes, color='m', fontsize=15)
			plt.text(0.35, 0.70, r'$V_\mathrm{lsr}$=%.2f($\pm$%.2f) km/s' % (result.params['vlsr2'].value,result.params['vlsr2'].stderr), transform=ax.transAxes, color='c', fontsize=15)

		if interactive:
			plt.plot(vaxis, final, 'r', label='Best-fit model')
			if vlsr1 != -999 and vlsr2 != -999:
				final1 = model_11(result.params, vaxis, spec) + spec
				final2 = final - final1
				plt.plot(vaxis, final1, 'm--', label='1st component', linewidth=2)
				plt.plot(vaxis, final2, 'c--', label='2nd component', linewidth=2)
				plt.text(-0.02, 0.90, r'$T_\mathrm{rot}$=%.2f($\pm$%.2f) K' % (result.params['T'].value,result.params['T'].stderr), transform=ax.transAxes, color='m', fontsize=15)
				plt.text(0.35, 0.65, r'$T_\mathrm{rot}$=%.2f($\pm$%.2f) K' % (result.params['T2'].value,result.params['T2'].stderr), transform=ax.transAxes, color='c', fontsize=15)
				plt.text(-0.02, 0.85, r'$N$(NH$_3$)=%.2e($\pm$%.2e) cm$^{-2}$' % (result.params['Ntot'].value,result.params['Ntot'].stderr), transform=ax.transAxes, color='m', fontsize=15)
				plt.text(0.35, 0.60, r'$N$(NH$_3$)=%.2e($\pm$%.2e) cm$^{-2}$' % (result.params['Ntot2'].value,result.params['Ntot2'].stderr), transform=ax.transAxes, color='c', fontsize=15)
				plt.text(-0.02, 0.80, r'$\sigma_v$=%.2f($\pm$%.2f) km/s' % (result.params['sigmav'].value,result.params['sigmav'].stderr), transform=ax.transAxes, color='m', fontsize=15)
				plt.text(0.35, 0.55, r'$\sigma_v$=%.2f($\pm$%.2f) km/s' % (result.params['sigmav2'].value,result.params['sigmav2'].stderr), transform=ax.transAxes, color='c', fontsize=15)
				plt.text(-0.02, 0.75, r'Filling factor=%.2f($\pm$%.2f)' % (result.params['ff'].value,result.params['ff'].stderr), transform=ax.transAxes, color='m', fontsize=15)
				plt.text(0.35, 0.50, r'Filling factor=%.2f($\pm$%.2f)' % (result.params['ff2'].value,result.params['ff2'].stderr), transform=ax.transAxes, color='c', fontsize=15)
				plt.text(-0.02, 0.68, r'$\tau$(1,1,m)=%.1f' % (tau(result.params['Ntot'].value,result.params['sigmav'].value,result.params['T'].value,1,0)), transform=ax.transAxes, color='m', fontsize=15)
				plt.text(0.35, 0.43, r'$\tau$(1,1,m)=%.1f' % (tau(result.params['Ntot2'].value,result.params['sigmav2'].value,result.params['T2'].value,1,0)), transform=ax.transAxes, color='c', fontsize=15)
			elif vlsr1 != -999 or vlsr2 != -999:
				plt.text(-0.02, 0.90, r'$T_\mathrm{rot}$=%.2f($\pm$%.2f) K' % (result.params['T'].value,result.params['T'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(-0.02, 0.85, r'$N$(NH$_3$)=%.2e($\pm$%.2e) cm$^{-2}$' % (result.params['Ntot'].value,result.params['Ntot'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(-0.02, 0.80, r'$\sigma_v$=%.2f($\pm$%.2f) km/s' % (result.params['sigmav'].value,result.params['sigmav'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(-0.02, 0.75, r'Filling factor=%.2f($\pm$%.2f)' % (result.params['ff'].value,result.params['ff'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(-0.02, 0.68, r'$\tau$(1,1,m)=%.1f' % (tau(result.params['Ntot'].value,result.params['sigmav'].value,result.params['T'].value,1,0)), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(-0.02, 0.63, r'$\tau$(1,1,s)=%.1f' % (tau(result.params['Ntot'].value,result.params['sigmav'].value,result.params['T'].value,1,1)), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(-0.02, 0.58, r'$\tau$(2,2,m)=%.1f' % (tau(result.params['Ntot'].value,result.params['sigmav'].value,result.params['T'].value,2,0)), transform=ax.transAxes, color='r', fontsize=15)
			plt.legend(fontsize=15,loc='upper right')
			#plt.show()
			plt.draw()
			print 'Is the fitting ok? y/n'
			yn = raw_input()
			if yn == 'y':
				unsatisfied = False 
				#currentT = time.strftime("%Y-%m-%d_%H:%M:%S")
				plt.savefig('NH3_fitting_no%i_test.png' % i)
				Trot = result.params['T'].value
				dTrot = result.params['T'].stderr
				sigmav = result.params['sigmav'].value
				dsigmav = result.params['sigmav'].stderr
				vpeak = result.params['vlsr'].value
				dvpeak = result.params['vlsr'].stderr
				if vlsr2 != -999:
					Trot2 = result.params['T2'].value
					dTrot2 = result.params['T2'].stderr
					sigmav2 = result.params['sigmav2'].value
					dsigmav2 = result.params['sigmav2'].stderr
					vpeak2 = result.params['vlsr2'].value
					dvpeak2 = result.params['vlsr2'].stderr
				plt.close()
			else:
				unsatisfied = True
			#raw_input('Press any key to continue...')
			f.clear()
		else:
			unsatisfied = False
	
	return Trot, dTrot, sigmav, dsigmav, vpeak, dvpeak, Trot2, dTrot2, sigmav2, dsigmav2, vpeak2, dvpeak2

##############################################################################


nh3_info = nh3_init()

# Load the data
# X axis is spatial offset
# Y axis is Vlsr
# But in pyfits they are reversed
data1, hdr1 = nh3_read('I18308_11_slice.fits')
data2, hdr2 = nh3_read('I18308_22_slice.fits')

sourcename = 'I18308'

onevpix = hdr1['cdelt2'] / 1e3 # km/s
naxisx = hdr1['naxis1']
naxisy = hdr1['naxis2']

vaxis_11 = nh3_load_axes(hdr1)
vaxis_22 = nh3_load_axes(hdr2)
vaxis_11 = vaxis_11[::-1]
vaxis_22 = vaxis_22[::-1]

bmaj1 = 5.10; bmin1 = 3.22
bmaj2 = 5.10; bmin2 = 3.21

# 1sigma rms = 0.003 Jy/beam
cf= 1.224e6 * 0.0005 * 3 / (nh3_info['frest'][1])**2 / (bmaj1 * bmin1)

# Open a text file
target = open('pvcut_fit_'+sourcename+'.txt', 'a')

#for i in range(naxisx):
#for i in [0,1,2,3,4,5]:
for i in range(naxisx)[56:]:
	print "Fitting spectrum no. %i" % i
	spec1 = data1[:,i]
	spec1 = spec1[::-1]
	spec2 = data2[:,i]
	spec2 = spec2[::-1]
	spec1 = 1.224e6 * spec1 / (nh3_info['frest'][0])**2 / (bmaj1 * bmin1)
	spec2 = 1.224e6 * spec2 / (nh3_info['frest'][1])**2 / (bmaj2 * bmin2)
	t,dt,s,ds,v,dv,t2,dt2,s2,ds2,v2,dv2= fit_spec(spec1, spec2, vaxis_11, vaxis_22, cutoff=cf, varyv=1, interactive=True, mode='single')
	target.write("%i\t%.2f %.2f %.2f %.2f %.2f %.2f\t%.2f %.2f %.2f %.2f %.2f %.2f\n" % (i,t,dt,s,ds,v,dv,t2,dt2,s2,ds2,v2,dv2))

target.close()

elapsed = (time.clock() - start)
print 'Stop the timer...'
print 'Time used: %0.0f seconds, or %0.1f minutes.' % (elapsed, elapsed/60.)

# tricky ones
# 13, 14, 15
