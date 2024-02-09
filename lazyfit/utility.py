from scipy.fft import fft, fftfreq
import numpy as np
import math as m

EPSILON = 1E-20 # tiny offset applied to fit limits when fixing a paramter

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

	By convention, errors are a single digit unless the first digit is 1.
	"""

	# this is surprisingly hard, and I should probably just have found a library that does this
	if np.isinf(dy):
		return '%g.3±inf' % y
	elif np.isnan(dy):
		return '%g.3±nan' % y
	elif dy == 0:
		return '%g.3' % y

	dy = np.abs(dy) # ensure  error is positive

	y_exponent = int(m.floor(np.log10(np.abs(y))))  # order of magnitude
	dy_exponent = int(m.floor(np.log10(dy)))
	exp_diff = y_exponent - dy_exponent  # order of magnitude diff from y to dy

	# determinte how many digits we want on the error.
	# We only want two digits if the first digit is 1, eg. 12.34+-0.41 should become 12.34(4), but 12.34+-0.12 should be 12.34(12)
	dy_2digits = int(str(round(dy*10**(-dy_exponent+1)))[0:2]) # first two digits of error as int

	if dy_2digits<20: # keep two
		n_dy_digits = 2
		dy_str = str(dy_2digits)
	elif dy_2digits<95: # python rounds to nearest even number, so 9.5 gets rounded up
		dy_str = str(round(dy_2digits/10)) # round to single digit
		n_dy_digits = 1
	else: # 96,97,98,99 need to be rounded up
		dy_str = '1'
		n_dy_digits = 1
		dy_exponent += 1

	# determine significant digits on y
	y_digits = exp_diff + n_dy_digits - 1 # number of digits on y after decimal. Add one to match the 2 error digits
	sign_buffer = ' ' if y>0 else '' # add a blank space if positive number so a - does not shift to coumns
	y_str = '%.*f' % (y_digits, y / 10 ** y_exponent)

	if version == 1:
		#format as (y ± dy) exponent
		return '('+sign_buffer+ y_str + '±' + dy_str + ')e'+str(y_exponent)
	
	elif version == 2:
		# format as y(dy) exponent
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

	# equation taken from https://en.wikipedia.org/wiki/Voigt_profile
	return 0.5346*L + np.sqrt(0.2166*L**2+G**2)

def sigma_to_FWHM(s):
	'''calculate FWHM of gaussian given its standard deviation'''
	return s*2.3548200450309493  # 2*sqrt(2*log(2))

def FWHM_to_sigma(s):
	'''the inverse of sigma_to_FWHM()'''
	return s/2.3548200450309493  # 2*sqrt(2*log(2))

def get_logistic_risetime(k, ylow=0.1, yhigh=0.9):
	'''get the risetime of a logistic function.

	Args:
		k		logistic rate, eg. estimated by the logistic or logpulse fitmodels
		ylow	low value of normalised logistic where we start measuring the rise, defaults to 0.1
		yhigh	high value of normalised logistic where we stop measuring the rise, defaults to 0.9

	If the optional paramters ylow and yhigh are not specified, the function will return the 10-90 rise time.
	'''

	return (1/k)* (np.log(1/ylow-1) - np.log(1/yhigh-1)) # see https://condellpark.com/kd/sars_files/LogisticRisetime.pdf for deviration, or do it yourself. It is a nice problem.

