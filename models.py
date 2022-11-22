import numpy as np
import math as m
import lazyfit.utility as utility
import types

import scipy.special

pi = np.pi

inf = float('inf')

###########################################
# generic functions
###########################################

def peak_finder(x, y):
	'''generic function for finding the biggest peak in (x,y) data.
	Returns a tupple of peak amplitude, position, FWHM and background'''

	B = np.min(y)  # background estimate
	A = np.max(y) - B  # peak amplitude
	x0 = x[np.argmax(y)]  # position estimate

	# the hard part, estimating the FWHM.
	# this we do by finding the value where the y drops by 0.5
	# note, we assume that x is in increasing order!
	filter_under_half = (y - B) < A / 2  # filter where background corrected signal is below 50% of peak
	a = x[np.max(
		np.argwhere(filter_under_half * (x < x0)))]  # x value of the first point to the left of the peak going below
	b = x[np.min(
		np.argwhere(filter_under_half * (x > x0)))]  # x value of the first point to the right of the peak going below
	FWHM = np.abs(b - a)  # full width half maximum

	return [A, x0, FWHM, B]


###########################################
# lorentzian
###########################################


def func_lorentz(x, A, x0, FWHM, B):
	"""lorentzian + B"""
	return A/(1+(x-x0)**2/(FWHM/2)**2) + B

def guess_lorentz(x, y):
	return peak_finder(x,y)


def bounds_lorentz(x, y):
	# assume peak to be withing x data, define FWHM to be positive
	lb = [-inf, np.min(x), 0, -inf]
	ub = [inf, np.max(x), inf, inf]

	return (lb, ub)

lorentz = types.SimpleNamespace()
lorentz.f = func_lorentz
lorentz.guess = guess_lorentz
lorentz.string = 'A/(1+(x-x0)^2/(FWHM/2)^2)+B'
lorentz.tex = r'$\frac{A}{1+(x-x0)^2/(FWHM/2)^2}+B$'
lorentz.bounds = bounds_lorentz


###########################################
# exponential decay
###########################################

def func_exp(x, A, Gamma, B):
	"""exp decay"""
	return A*np.exp(-x*Gamma) + B

def guess_exp(x,y):
	B = np.min(y)
	A = np.max(y)-B

	# find 1/e time. This must occour after the maximum. For this reason, we define a new set of xvalues. This is need incase the 
	try:
		filt = x>x[np.argmax(y)]
		x1 = x[filt]
		y1 = y[filt]

		e_time = x1[np.min(np.argwhere(y1-B<A*np.exp(-1)))]
		Gamma = 1/e_time
	except Exception:
		Gamma = 0

	return [A, Gamma, B]

def bounds_exp(x,y):
	lb = [0,0,-inf]
	ub = [inf, inf, inf]
	return lb,ub

exp = types.SimpleNamespace()
exp.f = func_exp
exp.guess = guess_exp
exp.string = 'A*exp(-x*Gamma) + B'
exp.bounds = bounds_exp


###########################################
# exponential decay convolved  with gaussian response
###########################################

def func_convexp(x, A, Gamma, B,  x0, s):
	"""exp decay convolved with gaussian with standard deviation s and mean x0"""

	if s == 0:
		return (x>=x0) * A * np.exp(-Gamma*(x-x0)) + B

	peakval = np.exp(0.5*Gamma**2*s**2)*scipy.special.erfc(Gamma*s/np.sqrt(2))
	return (A/peakval)*np.exp(-Gamma*(x-x0)+0.5*Gamma**2*s**2)*scipy.special.erfc((-(x-x0)/s+Gamma*s)/np.sqrt(2)) + B

def guess_convexp(x, y):

	x0 = x[np.argmax(y)]
	A = np.max(y)
	B = np.min(y)

	# find 1/e time. This must occour after the maximum. For this reason, we define a new set of xvalues. This is need incase the
	try:
		filt = x > x0
		x1 = x[filt]
		y1 = y[filt]

		e_time = x1[np.min(np.argwhere(y1 - B < A * np.exp(-1)))]
		Gamma = 1 / e_time
	except Exception:
		Gamma = 0

	# assume that the instrument response correspondsto 10% of the decay time
	s = 0.1/Gamma

	return [A, Gamma, B, x0, s]

def bounds_convexp(x,y):
	lb = [0, 0, -inf, -inf, 0]
	ub = [inf, inf, inf, inf, inf]
	return lb,ub

convexp = types.SimpleNamespace()
convexp.f = func_convexp
convexp.guess = guess_convexp
convexp.string = 'A*exp(-x*Gamma) conv N(x;0,s) + B'
convexp.bounds = bounds_convexp


###########################################
# Sinussoidal
###########################################


def func_sin(x, A, f, phi, B):
	"""sinussoidal oscillation"""
	return A*np.sin(x*f*2*pi+phi)+B

def guess_sin(x,y):
	B = np.mean(y)
	A = np.sqrt(2)*np.std(y)
	f, _ = utility.get_main_fourier_component(x,y)

	# it turns out that a more robust phi estimate can be constructed by trying 8 different values, and calculating the overlap with the data
	phi_list = np.arange(0, 2 * pi, pi / 4)
	overlap = np.zeros(8)
	for i, phi in enumerate(phi_list):
		overlap[i] = np.sum((y - B) * func_sin(x, 1, f, phi, 0))
	phi = phi_list[np.argmax(overlap)]

	return [A, f, phi, B]

def bounds_sin(x, y):

	lb = [0,0,0,-inf]
	up = [inf, inf, 2*pi, inf]

	return (lb,up)

sin = types.SimpleNamespace()
sin.f = func_sin
sin.guess = guess_sin
sin.string = 'A*sin(x*f*2pi+phi)+B'
sin.bounds = bounds_sin


###########################################
# Ramsey
###########################################


def func_ramsey(x, A, f, phi, B, T2s, alpha):
	"""Ramsey oscillation"""
	return A*np.sin(x*f*2*pi+phi)*np.exp(-(x/T2s)**alpha)+B

def guess_ramsey(x,y):
	
	# use the same guess as sin and set T2s too be half the x range and alpha t0
	guess = guess_sin(x,y) + [np.max(x)/2, 2]

	return guess

def bounds_ramsey(x, y):

	lb = [0, 0, 0, -inf, 0, 0]
	up = [inf, inf, 2*pi, inf, inf, inf]

	return (lb,up)

ramsey = types.SimpleNamespace()
ramsey.f = func_ramsey
ramsey.guess = guess_ramsey
ramsey.string = 'A*sin(x*f*2pi+phi)*exp(-(x/T2s)^alpha)+B'
ramsey.bounds = bounds_ramsey

###########################################
# two level saturation
###########################################


def func_twolvlsat(x, Psat, Imax):
	"""Two level saturation.
	Note that Imax is the intensity for x->inf"""

	if x is 0: # check if we are inputting a single python float, return 0 to avoid divide by zero error
		return 0

	with np.errstate(divide='ignore'): # ignore numpy divide by zero error
		return Imax/(1+Psat/x)

def guess_twolvlsat(x,y):
	
	Imax = np.max(y)
	Psat = x[np.min(np.argwhere(y>Imax*0.5))] # take Psat as first point where y goes over 1/2 of max

	return [Psat, Imax]

def bounds_twolvlsat(x, y):

	lb = [0, 0]
	ub = [inf, inf]

	return (lb,ub)

twolvlsat = types.SimpleNamespace()
twolvlsat.f = func_twolvlsat
twolvlsat.guess = guess_twolvlsat
twolvlsat.string = 'Imax*/(1+Psat/x)'
twolvlsat.bounds = bounds_twolvlsat

###########################################
# Rabi
###########################################


def func_rabi(x, A, x_pi, B):
	"""Rabi oscillation"""
	return A*np.sin((x/x_pi)*pi/2)**2+B

def guess_rabi(x,y):
	
	# use the same guess as sin and set T2s too be half the x range
	B = np.min(y)
	A = np.max(y)-B
	#x_pi = x[np.argmax(y)] # guess that the first pi is the hightest... this may fail
	x_pi = 0.5 / utility.get_main_fourier_component(x, y)[0]
	return [A, x_pi, B]

def bounds_rabi(x, y):

	lb = [0, 0, -inf]
	up = [inf, inf, inf]

	return (lb,up)

rabi = types.SimpleNamespace()
rabi.f = func_rabi
rabi.guess = guess_rabi
rabi.string = 'A*sin((x/x_pi)*pi/2)^2+B'
rabi.bounds = bounds_rabi


###########################################
# Unnormalised gaussian
###########################################

def func_gaussian(x, A, x0, s, B):
	"""Gaussian + B"""
	return A * np.exp(-(x-x0)**2/(2*s**2)) + B

def guess_gaussian(x, y):
	A, x0, FWHM, B =  peak_finder(x, y)
	return [A, x0, FWHM/2.35, B]  # convert FWHM to standard deviation


def bounds_gaussian(x, y):
	# assume peak to be withing x data, define sigma to be positive
	lb = [-inf, np.min(x), 0, -inf]
	ub = [inf, np.max(x), inf, inf]
	return lb, ub


gaussian = types.SimpleNamespace()
gaussian.f = func_gaussian
gaussian.guess = guess_gaussian
gaussian.string = 'A*exp(-(x-x0)^2/(2*s^2))+B'
gaussian.tex = r'$Ae^{-(x-x_0)^2/(2s^2)}+B$'
gaussian.bounds = bounds_gaussian

###########################################
# Normalised gaussian
###########################################

def func_normgaussian(x, A, x0, s, B):
	"""normalised gaussian + B"""
	return A * np.exp(-(x-x0)**2/(2*s**2))/(np.sqrt(2*pi)*s) + B

def guess_normgaussian(x, y):
	A, x0, FWHM, B =  peak_finder(x, y)
	s = FWHM/2.35
	A *= np.sqrt(2*pi)*s
	return [A, x0, s, B]  # convert FWHM to standard deviation


def bounds_normgaussian(x, y):
	# assume peak to be withing x data, define sigma to be positive
	lb = [0, np.min(x), 0, -inf]
	ub = [inf, np.max(x), inf, inf]
	return lb, ub


normgaussian = types.SimpleNamespace()
normgaussian.f = func_normgaussian
normgaussian.guess = guess_normgaussian
normgaussian.string = 'A*Norm(x;x0,s)+B'
normgaussian.tex = r'$\frac{Ae^{-(x-x_0)^2/(2s^2)}}{\sqrt{2\pi}s}+B$'
normgaussian.bounds = bounds_normgaussian

###########################################
# linear
###########################################

def func_lin(x, A, B):
	"""normalised gaussian + B"""
	return A * x + B

def guess_lin(x, y):
	# this can be done analytically! See page 100 of Statistics by R. J. Barlow
	A = (np.mean(x*y)-np.mean(x)*np.mean(y))/(np.mean(x**2)-np.mean(x)**2)
	B = np.mean(y) - A*np.mean(x)
	return [A,B]

def bounds_lin(x, y):
	# assume peak to be withing x data, define sigma to be positive
	lb = [-inf, -inf]
	ub = [inf, inf]
	return lb, ub


lin = types.SimpleNamespace()
lin.f = func_lin
lin.guess = guess_lin
lin.string = 'A*x+B'
lin.tex = r'$Ax+B'
lin.bounds = bounds_lin

###########################################
# Voigt
###########################################


def  func_voigt(x, A, x0, L, G, B):
	'''voigt with lorentzian FWHM L and Gaussian FWHM G and background b'''

	# note that the scipy function takes a gaussian standard deviation and a lorentzian half width half maximmum.
	# as we specify FWHM for both distributions we need to convert appropriately
	# we also divide with the function evaluated at zero detuning to ensure an amplitude of 1 at resonance
	return A*scipy.special.voigt_profile((x-x0), utility.FWHM_to_sigma(G), L/2)\
		   /scipy.special.voigt_profile(0, utility.FWHM_to_sigma(G), L/2) \
		   + B


def guess_voigt(x,y):
	# do basic peak detection
	A, x0, FWHM, B = peak_finder(x, y)

	# we will assume that L=G, substite into the equation for effective Voigt linewidth (see utility.get_Voigt_FWHM() ) and solve for L

	L = FWHM/(0.5346+np.sqrt(1+0.2166))

	return [A, x0, L, L, B ]


def bounds_voigt(x, y):
	lb = [-inf, np.min(x), 0, 0, -inf]
	ub = [inf, np.max(x), inf, inf, inf]
	return lb, ub

voigt = types.SimpleNamespace()
voigt.f = func_voigt
voigt.guess = guess_voigt
voigt.string = 'A*Voigt(x;x0,L,G)+B'
#voigt.tex = r'$Ax+B'
voigt.bounds = bounds_voigt


###########################################
# logistic
###########################################

def func_logistic(x, A, B, x0, k,):
	"""logistic + background"""
	return B + A/(1+np.exp(-(x-x0)*k))

def guess_logistic(x, y):
	B = np.min(y)
	A = np.max(y) - np.min(y)
	x0 = x[np.argmin(np.abs(y-B-A/2))] # find where y is at the mid value
	x10 = x[np.argmin(np.abs(y-B-A*0.1))]# 10% percentile
	x90 = x[np.argmin(np.abs(y - B - A * 0.9))]  # 90% percentile
	k = 5/(x90-x10)

	return [A,B, x0, k]

def bounds_logistic(x, y):
	# assume peak to be withing x data, define FWHM to be positive
	lb = [0, -inf, np.min(x), 0]
	ub = [inf, inf, np.max(x), inf]

	return (lb, ub)

logistic = types.SimpleNamespace()
logistic.f = func_logistic
logistic.guess = guess_logistic
logistic.string = 'A/(1+exp(-(x-x0)*k))+B'
#logistic.tex = r'$\frac{A}{1+(x-x0)^2/(FWHM/2)^2}+B$'
logistic.bounds = bounds_logistic


###########################################
# logistic pulse
###########################################

def func_logpulse(x, A, B, x0, x1, k0, k1):
	"""the logistic functions multiplied together with opposite directions"""
	return B + A*func_logistic(x, 1, 0, x0, k0)*func_logistic(x, 1, 0, x1, -k1) # second logpulse should be descending

def guess_logpulse(x, y):
	B = np.min(y)
	A = np.max(y) - np.min(y)
	x0 = x[np.min(np.argwhere(y>B+A/2))] # find first x where y above half max
	x1 = x[np.max(np.argwhere(y>B+A/2))] # find last x where y above half max
	k = 5/(x1-x0)*10 # just assume for now that the risetime 10% of FWHM. This should be good enough for initial condition
	return [A, B, x0, x1, k, k]

def bounds_logpulse(x, y):
	# assume peak to be withing x data, define FWHM to be positive
	lb = [0, -inf, np.min(x), np.min(x), 0, 0]
	ub = [inf, inf, np.max(x), np.max(x), inf, inf]

	return (lb, ub)

logpulse = types.SimpleNamespace()
logpulse.f = func_logpulse
logpulse.guess = guess_logpulse
logpulse.string = 'A/((1+exp(-(x-x0)k0))(1+exp(-(x+x0)k0)) + B'
#logpulse.tex = r'$\frac{A}{1+(x-x0)^2/(FWHM/2)^2}+B$'
logpulse.bounds = bounds_logpulse



