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

def func_exp(x, amp, Gamma, bg):
	"""exp decay"""
	return amp*np.exp(-x*Gamma) + bg

def guess_exp(x,y):
	bg = np.min(y)
	amp = np.max(y)-bg

	# find 1/e time. This must occour after the maximum. For this reason, we define a new set of xvalues. This is need incase the 
	try:
		filt = x>x[np.argmax(y)]
		x1 = x[filt]
		y1 = y[filt]

		e_time = x1[np.min(np.argwhere(y1-bg<amp*np.exp(-1)))]
		Gamma = 1/e_time
	except Exception:
		Gamma = 0

	return [amp, Gamma, bg]

def bounds_exp(x,y):
	lb = [0,0,-inf]
	ub = [inf, inf, inf]
	return lb,ub

exp = types.SimpleNamespace()
exp.f = func_exp
exp.guess = guess_exp
exp.string = 'amp*exp(-x*Gamma) + bg'
exp.bounds = bounds_exp


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


def func_ramsey(x, A, f, phi, B, T2s):
	"""Ramsey oscillation"""
	return A*np.sin(x*f*2*pi+phi)*np.exp(-x**2/T2s**2)+B

def guess_ramsey(x,y):
	
	# use the same guess as sin and set T2s too be half the x range
	guess = guess_sin(x,y)
	guess.append(np.max(x)/2)

	return guess

def bounds_ramsey(x, y):

	lb = [0, 0, 0, -inf, 0]
	up = [inf, inf, 2*pi, inf, inf]

	return (lb,up)

ramsey = types.SimpleNamespace()
ramsey.f = func_ramsey
ramsey.guess = guess_ramsey
ramsey.string = 'A*sin(x*f*2pi+phi)*exp(-x^2/T2s^2)+B'
ramsey.bounds = bounds_ramsey

###########################################
# Ramsey with free exponent
###########################################


def func_freeramsey(x, A, f, phi, B, T2s, alpha):
	"""Ramsey oscillation"""
	return A*np.sin(x*f*2*pi+phi)*np.exp(-(x/T2s)**alpha)+B

def guess_freeramsey(x,y):
	
	# use the same guess as sin and set T2s too be half the x range
	guess = guess_sin(x,y)
	guess.append(np.max(x)/2)
	guess.append(1.5)

	return guess

def bounds_freeramsey(x, y):

	lb = [0, 0, 0, -inf, 0,0]
	up = [inf, inf, 2*pi, inf, inf, inf]

	return (lb,up)

freeramsey = types.SimpleNamespace()
freeramsey.f = func_freeramsey
freeramsey.guess = guess_freeramsey
freeramsey.string = 'A*sin(x*f*2pi+phi)*exp(-(x/T2s)^alpha)+B'
freeramsey.bounds = bounds_freeramsey


###########################################
# two level saturation
###########################################


def func_twolvlsat(x, Psat, Imax):
	"""Two level saturation.
	Note that Imax is the intensity for x->inf"""
	return 2*Imax*x/Psat/(1+2*x/Psat)

def guess_twolvlsat(x,y):
	
	Imax = np.max(y)
	Psat = x[np.min(np.argwhere(y>Imax*0.66))] # take Psat as first point where y goes over 2/3 of max

	return [Psat, Imax]

def bounds_twolvlsat(x, y):

	lb = [0, 0]
	ub = [inf, inf]

	return (lb,ub)

twolvlsat = types.SimpleNamespace()
twolvlsat.f = func_twolvlsat
twolvlsat.guess = guess_twolvlsat
twolvlsat.string = '2*Imax*x/Psat/(1+2*x/Psat)'
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
	x_pi = x[np.argmax(y)] # guess that the first pi is the hightest... this may fail

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
	# use barlow
	pass

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
	return A*scipy.special.voigt_profile((x-x0), utility.sigma_to_FWHM(G), L*2) + B
	pass


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