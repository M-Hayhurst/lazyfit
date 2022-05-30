import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import types
import math as m
import FitUtility
import FitModels
pi = np.pi


def format_error(y, dy):
    '''return a nice string representation of a number and its error'''
    if np.isinf(dy):
    	return '%g.3(inf)'%y
    elif np.isnan(dy):
    	return '%g.3(nan)'%y

    dy = np.abs(dy)

    y_exponent = int(m.floor(np.log10(np.abs(y)))) # order of magnitude
    dy_exponent = int(m.floor(np.log10(dy)))
    
    # choose how many digits we want on the y
    exp_diff = y_exponent-dy_exponent # order of magnitude diff from y to dy
    dy_str = ('%.20g'%(dy*10**(exp_diff+2))).replace('.','') # string representation without any decimal
    flag1 = str(dy)[0]=='1' # first digit is 1
    y_precision = exp_diff + int(flag1) # number of digits on y after decimal 
    
    # TODO: Make error round corrently

    s = ('%.*f' % (y_precision, y/10**y_exponent) + '(' +
        dy_str[0:(1+int(flag1))] +
        ')e' + str(y_exponent) )
    return s


def clean_data(x,y):
	'''remove any NaN or inf in the data'''

	filt_x = np.logical_and(np.isnan(x)==0, np.isinf(x)==0)
	filt_y = np.logical_and(np.isnan(y)==0, np.isinf(y)==0)

	filt = np.logical_and(filt_x, filt_y)
	n_bad = np.sum(filt == 0)

	return (x[filt], y[filt], n_bad)

class Wrapper:
	def __init__(self, fittype, x, y, dy=None, guess=None, bounds=None, fix=None, verbose=False):

		self.fittype = fittype
		self.verbose = verbose
		self.fix = fix

		# clean data
		self.x, self.y, n_bad = clean_data(x, y)
		if n_bad >0:
			print(f'Warning: Removing {n_bad} cases of NaN or Inf from data to enable fitting')
		
		# errors
		if dy is None:
			self.has_dy = False
		else:
			if isinstance(dy, float) or isinstance(dy, int):
				# assume this is a constant error
				self.dy = np.ones(len(self.x))*dy
			else:
				self.dy = dy
			self.has_dy = True 


		# find fit model
		try:
			self.model = getattr(FitModels, fittype)
		except AttributeError:
			raise Exception(f'No fit model named "{fittype}"')

		# extrac stuff from fit model
		self.f = self.model.f
		self.fitvars = self.f.__code__.co_varnames[1::]
		self.n_DOF = len(self.x) - len(self.fitvars)

		# generate guess
		if guess is None:
			self.guess = self.model.guess(self.x, self.y)
		else:
			self.guess = guess 

		# generate bounds
		if bounds is None:
			if hasattr(self.model, 'bounds'):
				self.bounds = self.model.bounds(self.x, self.y)
			else:
				self.bounds = None # TODO this could give problems with the fix functionality
		else:
			self.bounds = bounds

		# fix necessary varibles if neceassary
		if fix is not None:
			for key, val in fix:
				# find the index of the key in the fitting params
				ind = np.argwhere([key == k for k in self.fitvars])

				# go to this index, fix bounds and guess to the value
				self.bounds[0][ind] = val
				self.bounds[1][ind] = val 
				self.guess[ind] = val 


	def fit(self, options={}):
		self.fit_res = scipy.optimize.curve_fit(self.f, self.x, self.y, self.guess, bounds=self.bounds, **options)
		self.params = self.fit_res[0]
		self.COVB = self.fit_res[1]
		self.errors = np.sqrt(np.diag(self.COVB))

	def get_chi2(self):
		return np.sum( (self.y-self.predict(self.x))**2/self.dy**2)

	def predict(self, x):
		return self.f(x, *self.params)


	def plot(self, N=200, print_params = True, plot_guess = False, use_log = False, plot_residuals = False, figsize=None, fmt='o'):
		
		fig = plt.figure(figsize=figsize)

		if plot_residuals:
			ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
		else:
			ax1 = plt.gca()

		if self.has_dy:
			plt.errorbar(self.x, self.y, self.dy, fmt=fmt, label='Data')
		else:
			plt.plot(self.x, self.y, fmt, label='Data')

		xdummy = np.linspace(np.min(self.x), np.max(self.x), N)
		plt.plot(xdummy, self.predict(xdummy), label='Fit')

		if plot_guess:
			plt.plot(xdummy, self.f(xdummy, *self.guess), label='Guess')

		if use_log:
			plt.yscale('log')

		# make legend
		ax = plt.gca()
		ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

		# plot residuals in subplot
		if plot_residuals:
			ax2 = plt.subplot2grid((3,1), (2,0))
			if self.has_dy:
				plt.errorbar(self.x, self.y-self.predict(self.x), self.dy, fmt='o')
			else:
				plt.plot(self.x, self.y-self.predict(self.x))

			plt.ylabel('Residual')
			plt.grid(axis='y')


		# print list of fitting parameters
		if print_params:
			# print model name
			summary = f'Model: {self.fittype}\n'

			# print the math expression, either with plain text or latex
			#if hasattr(self.model, 'tex'):
			#	summary += self.model.tex +'\n'
			#else:
			summary += self.model.string +'\n'

			# print fit paramters
			for i in range(len(self.fitvars)):
				#summary += '\n%s: %2.2g±%2.2g'%(self.fitvars[i].ljust(5), self.params[i], self.errors[i]) 
				if self.fix and self.fitvars[i] in self.fix:
					summary += '\n'+self.fitvars[i].ljust(5)+' (FIXED)'
				else:
					summary += '\n'+self.fitvars[i].ljust(5) +': ' + format_error(self.params[i], self.errors[i]) 

			# print degrees of freedom
			summary += f'\n\nN DOF: {self.n_DOF}'

			# print chi2 estimate if data has errors
			if self.has_dy:
				chi2 = self.get_chi2()
				summary += '\nchi2    = %2.4g'%chi2
				summary += '\nchi2red = %2.4f'%(chi2/self.n_DOF)

			y_cord = 0.6 if plot_residuals else 0.7
			plt.text(1.02, y_cord, summary, transform=ax1.transAxes, horizontalalignment='left', fontfamily='monospace', verticalalignment='top')

		# return fig handle
		return fig

		
	def plot_guess(self, N=200):
		if self.has_dy:
			print(self.x.shape)
			print(self.y.shape)
			print(self.dy.shape)
			plt.errorbar(self.x, self.y, self.dy, fmt='o', label='Data')
		else:
			plt.plot(self.x, self.y, 'o', label='Data')

		xdummy = np.linspace(np.min(self.x), np.max(self.x), N)
		plt.plot(xdummy, self.f(xdummy, *self.guess), label='Guess')
