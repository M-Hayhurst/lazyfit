from scipy.fft import fft, fftfreq
import numpy as np
import math as m

EPSILON = 1E-20 # tiny offset applied to fit limits when fixing a paramter

def significant_round(x, n):
    '''round the float x to n significant digits'''
    return round(x, n-1-int(m.floor(m.log10(abs(x)))))

def format_error(y, dy, version=2):
	"""return a nice string representation of a number and its error
	Parameters:
	y			value
	dy			error
	version 	formatting style (default value = 2)

	Returns:
	string

	version 1 formats errors as (y ± dy) exponent
	version 2 formats errors as y(dy) exponent
	version 2 is more compact.

	By convention, errors are reported with two significant digits
	"""

	# this is surprisingly hard to implement

	dy = np.abs(dy) # ensure  error is positive

	# process error
	dy_rounded = significant_round(dy, 2) # round to two significant digits
	dy_exponent = int(m.floor(np.log10(dy))) # get order of magnitude

	# process value
	# first, determine how many digits we want. This depends on the magnitude of the error
	y_exponent = int(m.floor(np.log10(np.abs(y))))  # value order of magnitude
	y_n_digits = y_exponent - dy_exponent +1 # order of magnitude diff from y to dy
	sign_buffer = ' ' if y>0 else '' # add a blank space if positive number so a - does not shift to coumns
	y_str = '%.*f' % (y_n_digits, y / 10 ** y_exponent) # this automatically performs the correct rounding of y

	if version == 1: #format as (y ± dy) exponent
		dy_str = '%.*f' % (y_n_digits, dy_rounded / 10 ** y_exponent)
		return '('+sign_buffer+ y_str + '±' + dy_str + ')e'+str(y_exponent)

	elif version == 2: # format as y(dy) exponent
		dy_str = str(round(dy_rounded*10**(-dy_exponent+1)))[0:2]
		return sign_buffer + y_str +'(' +dy_str + ')e'+str(y_exponent)

def clean_data(x, y, dy=None):
	"""remove any NaN or inf in the data and sort the data to ensure monotoneously increasing x values
	Returns:
	x values, y values, numbers of removed points
	"""

	filt_x = np.logical_and(np.isnan(x) == 0, np.isinf(x) == 0)
	filt_y = np.logical_and(np.isnan(y) == 0, np.isinf(y) == 0)

	filt = np.logical_and(filt_x, filt_y)
	n_bad = np.sum(filt == 0)

	x,y = x[filt], y[filt]
	if type(dy) is np.ndarray:
		dy = dy[filt]

	# finally, sort the arrays, probably redundant most of the time, but the fit guesses assume monotonusly increasing x
	p = x.argsort() # get sorted indices for x
	x = x[p]
	y = y[p]
	if type(dy) is np.ndarray:
		dy = dy[p]

	return x,y,dy, n_bad


def get_main_fourier_component(t, y, ignore_dc=True):
	"""take the fourier transform of a timesignal and return the dominant fourier component and phase"""

	N = len(t)
	yf = fft(y)
	if ignore_dc:
		yf[0] = 0  # remove DC component from FFT
	dt = t[1]-t[0]  # time step size, assume equal
	xf = fftfreq(N, dt)[:N//2] # construct xaxis

	max_ind = np.argmax(np.abs(yf))
	max_freq = xf[max_ind]
	max_phase = np.angle(yf[max_ind]) % (2*np.pi)  # fold into range 0..2pi to conform with fit limits

	return max_freq, max_phase


def get_voigt_FWHM(G, L):
	'''returns the FWHM of a Voigt distribution given its Gassian FWHM G and lorentzian FWHM L'''

	# equation taken from https://en.wikipedia.org/wiki/Voigt_profile#The_width_of_the_Voigt_profile
	return 0.5346*L + np.sqrt(0.2166*L**2+G**2)

def get_voigt_FWHM_err(G, L, G_err, L_err, LG_cov):
	'''returns the error on the FWHM of a Voigt distribution given its Gassian FWHM G and lorentzian FWHM L'''

	a = 0.5346 # first coeffecient in FWHM formular, see https://en.wikipedia.org/wiki/Voigt_profile#The_width_of_the_Voigt_profile
	b = 0.2166

	# calculate variance using error propagation
	var = (a+(b*L**2+G**2)**-0.5*b*L)**2 * L_err**2
	+ ( (b*L**2+G**2)**-0.5*G)**2 * G_err**2
	+ 2 * ( (b*L**2+G**2)**-1.5*G*b*L)**2 * LG_cov 

	return var**0.5

def sigma_to_FWHM(s):
	'''calculate FWHM of gaussian given its standard deviation'''
	return s*2.3548200450309493  # 2*sqrt(2*log(2))

def FWHM_to_sigma(s):
	'''the inverse of sigma_to_FWHM()'''
	return s/2.3548200450309493  # 2*sqrt(2*log(2))

def get_logistic_risetime(k, low=0.1, high=0.9):
	'''get the risetime of a logistic function.

	Args:
		k		logistic rate, eg. estimated by the logistic or logpulse fitmodels
		low	low value of normalised logistic where we start measuring the rise, defaults to 0.1
		high	high value of normalised logistic where we stop measuring the rise, defaults to 0.9

	If the optional paramters low and high are not specified, the function will return the 10-90 rise time.
	'''

	return (1/k)* (np.log(1/low-1) - np.log(1/high-1)) # see https://condellpark.com/kd/sars_files/LogisticRisetime.pdf for deviration, or do it yourself. It is a nice problem.

def get_logistic_risetime_error(k, k_error, low=0.1, high=0.9):
	'''get the uncertainty of the risetime of a logistic function.

	Args:
		k		logistic rate, eg. estimated by the logistic or logpulse fitmodels
		low	low value of normalised logistic where we start measuring the rise, defaults to 0.1
		high	high value of normalised logistic where we stop measuring the rise, defaults to 0.9

	If the optional paramters low and high are not specified, the function will return the 10-90 rise time.
	'''

	return (k_error/k**2)* (np.log(1/low-1) - np.log(1/high-1)) # see https://condellpark.com/kd/sars_files/LogisticRisetime.pdf for deviration, or do it yourself. It is a nice problem.
