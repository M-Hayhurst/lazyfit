import numpy as np
import math as m
import FitUtility
import types
pi = np.pi

inf = float('inf')


###########################################
# lorentzian
###########################################


def func_lorentz(x, A, x0, FWHM, B):
	"""lorentzian + B"""
	return A/(1+(x-x0)**2/(FWHM/2)**2) + B

def guess_lorentz(x, y):

	B = np.min(y)		# background estimate
	A = np.max(y) - B 	# peak amplitude
	x0 = x[np.argmax(y)]

	# the hard part, guess the FWHM. 
	# this we do by finding the value where the y drops by 0.5
	filter_over_half = (y-B)>A/2
	a = x[np.min(np.argwhere(filter_over_half))] # x value of first element over threshold
	b = x[np.max(np.argwhere(filter_over_half))] # x value of last element over threshold
	FWHM = np.abs(b-a)
	return [A, x0, FWHM, B]

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
	f, phi = FitUtility.get_main_fourier_component(x,y)

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