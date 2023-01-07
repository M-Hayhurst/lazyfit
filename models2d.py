import numpy as np
import math as m
import lazyfit.utility as utility
import types
import scipy.stats
import lazyfit.models as models
import inspect

pi = np.pi
inf = float('inf')

###########################################
#### create class for fit models
###########################################

class LazyFitModel2d:
	'''Class for containg fitmodels. '''

	def __init__(self, name, f, guess=None, bounds=None, math_string=None, description=None ):
		self.name = name # string containing model name
		self.f = f # function for evaluating model
		self.string = math_string # string with math expression
		self.guess = guess # function for getting guess
		self.bounds = bounds # function for getting bounds
		self.description = description # additional description

	def get_param_names(self):
		'''return dictionary with names of fit parameters'''
		return inspect.getargspec(self.f).args[2::]  # get fit function arguments but without 'x' and 'y' which is always first arguments

	def __repr__(self):
		'''show some usefull information when printing the model object in the terminal'''
		return f'<LazyFitModel2d "{self.name}". Fit parameters: {self.get_param_names()}.'

###########################################
# Two dimensional Gaussian with covariance term
###########################################

def _func_gaussian2d(x, y, A, x0, y0, sx, sy, p, B):
	"""Bivariate gaussian plus background

	Parameters:
	x	xdata
	y	ydata
	A	amplitude
	x0 	mean in x
	y0 	mean in y
	sx 	x standard deviation
	sy 	y standard deviation
	p	linear correlation coeffecient
	B	background"""

	cov = np.array([[sx**2, sx*sy*p],[sx*sy*p, sy**2]]) # covariance matrix
	xy = np.array([x.ravel(),y.ravel()]).transpose()

	res =  A * scipy.stats.multivariate_normal.pdf(xy, mean=[x0,y0], cov=cov)/scipy.stats.multivariate_normal.pdf((0,0), mean=[0,0], cov=cov) + B
	return np.resize(res, x.shape) # imitate size of x

def _guess_gaussian2d(x, y, z):
	# TODO use some built in function

	B = np.min(z)
	A = np.max(z) - B

	# estimate mean and std of x
	x_projec = np.sum(z, axis=0) # project onto x axis
	[_, x0, x_FWHM, _] = models.peak_finder(x, x_projec)

	# do the same for y
	y_projec = np.sum(z, axis=1)
	[_, y0, y_FWHM, _] = models.peak_finder(y, y_projec)

	return [A, x0, y0, x_FWHM/2.35, y_FWHM/2.35, 0, B]  # convert FWHM to standard deviation

def _bounds_gaussian2d(x, y, z):
	# assume peak to be withing x data, define sigma to be positive
	lb = [-inf, np.min(x), np.min(y), 0, 0, -1, -inf ]
	ub = [inf, np.max(x), np.max(y), inf, inf, 1, inf]
	return lb, ub

gaussian2d = LazyFitModel2d('gaussian2d', _func_gaussian2d, _guess_gaussian2d, _bounds_gaussian2d, 'Bivariate gaussian')
