import numpy as np
import math as m
import lazyfit.utility as utility
import types
import scipy.stats
import lazyfit.models as models

pi = np.pi

inf = float('inf')

###########################################
# Two dimensional Gaussian with covariance term
###########################################

def func_gaussian2d(x, y, A, x0, y0, sx, sy, p, B):
	"""Bivariate gaussian plus background
	Parameters:
	x	1D array of x values
	y	2D array of y values
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

def guess_gaussian2d(x, y, z):
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

def bounds_gaussian2d(x, y, z):
	# assume peak to be withing x data, define sigma to be positive
	lb = [-inf, np.min(x), np.min(y), 0, 0, -1, -inf ]
	ub = [inf, np.max(x), np.max(y), inf, inf, 1, inf]
	return lb, ub


gaussian2d = types.SimpleNamespace()
gaussian2d.f = func_gaussian2d
gaussian2d.guess = guess_gaussian2d
gaussian2d.string = 'Bivariate gaussian'
gaussian2d.bounds = bounds_gaussian2d
