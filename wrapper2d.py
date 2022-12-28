import lazyfit.models2d as models2d
import numpy as np
pi = np.pi
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import lazyfit.utility as utility
import inspect


def fit2d(fittype, x, y, z, dz=None, guess=None, bounds=None, fix={}, verbose=False, options={}):
    """Provides a short cut to creating a Wrapper object and calling fit()"""
    f = Wrapper2d(fittype, x, y, z, dz, guess, bounds, fix, verbose)
    f.fit(options)
    return f

epsilon = 1E-20


class Wrapper2d:
    def __init__(self, fittype, x, y, z, dz=None, guess=None, bounds=None, fix={}, verbose=False):

        self.fittype = fittype
        self.verbose = verbose
        self.fix = fix

        # clean data
        self.x, self.y, self.z, = x, y, z

        # errors
        if dz is None:
            self.has_dz = False
        else:
            if isinstance(dz, float) or isinstance(dz, int):
                # assume this is a constant error
                self.dz = np.ones((x.size, y.size)) * dz
            else:
                self.dz = dz
            self.has_dz = True

        # find fit model
        try:
            self.model = getattr(models2d, fittype)
        except AttributeError:
            raise Exception(f'No fit model named "{fittype}"')

        # extrac stuff from fit model
        self.f = self.model.f
        self.fitvars = inspect.getargspec(self.f).args[
                       2::]  # get fit function arguments but without 'x' and 'y' which are always the first two arguments
        self.n_DOF = self.z.size - len(self.fitvars)

        # generate guess
        if guess is None:
            self.guess = self.model.guess(self.x, self.y, self.z)
        else:
            self.guess = guess

        # generate bounds
        if bounds is None:
            if hasattr(self.model, 'bounds'):
                self.bounds = self.model.bounds(self.x, self.y, self.z)
            else:
                self.bounds = None  # TODO this could give problems with the fix functionality
        else:
            self.bounds = bounds

        # fix varibles if neceassary
        for key, val in fix.items():
            # find the index of the key in the fitting params
            ind = self.fitvars.index(key)

            # go to this index, fix bounds and guess to the value
            self.bounds[0][ind] = val * 0.999999 - epsilon
            self.bounds[1][ind] = val * 1.000001 + epsilon
            self.guess[ind] = val

    def f_wrapper(self, *args):
        '''Wrapper for the fitting function. Does two things:
        1) unwraps the first argument into x and y vabibles
        2) returns a linearised output'''

        x, y = args[0] # unwrap first two arguments
        return self.f(x,y, *args[1::]).ravel()

    def fit(self, options={}):
        # create mesh grid versions of x and y, see eg. https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
        x_mesh, y_mesh = np.meshgrid(self.x, self.y)

        # do the actual fit
        self.fit_res = scipy.optimize.curve_fit(self.f_wrapper, (x_mesh, y_mesh), self.z.ravel(), self.guess, bounds=self.bounds, **options)

        # save parameters and errors as arrays
        self.params = self.fit_res[0]
        self.COVB = self.fit_res[1]
        self.errors = np.sqrt(np.diag(self.COVB))

        # also save fit paramters and errors as dicts
        self.params_dict, self.errors_dict = {}, {}
        for i, var in enumerate(self.fitvars):
            self.params_dict[var] = self.params[i]
            self.errors_dict[var] = self.errors[i]

    def predict(self, x, y):
        x_mesh, y_mesh = np.meshgrid(x, y)
        return self.f(x_mesh, y_mesh, *self.params)

    def get_guess(self, x, y):
        x_mesh, y_mesh = np.meshgrid(x, y)
        return self.f(x_mesh, y_mesh, *self.guess)

    def get_chi2(self):
        return np.sum((self.z - self.predict(self.x, self.y)) ** 2 / self.dz ** 2)

    def get_pval(self):
        '''Return probability of null hypothesis given chisquared sum, and degrees of freedom.
        Only works if you supplied errors to the fitting routine'''
        return scipy.stats.distributions.chi2.sf(self.get_chi2(), self.n_DOF)

    def plot(self, print_params=True, plot_guess=False, logz=False, plot_residuals=True, figsize=(12,5), fmt='o'):

        # this is harder for 2D, the strategy is to crete 3 windows:
        # 1) data
        # 2) fit
        # 3) difference, ie. error
        # 4) guess (optional)

        fig = plt.figure(figsize=figsize)

        subplot_size = (1, 2+int(plot_residuals)+int(plot_guess))
        i_subplot = 0
        axes = []

        # plot data
        axes.append(plt.subplot2grid(subplot_size, (0, i_subplot)))
        plt.pcolormesh(self.x, self.y, self.z)
        plt.title('Data')
        plt.colorbar()
        i_subplot += 1

        # Fit guess
        if plot_guess:
            axes.append(plt.subplot2grid(subplot_size, (0, i_subplot)))
            plt.pcolormesh(self.x, self.y, self.get_guess(self.x, self.y))
            plt.title('Fit guess')
            plt.colorbar()
            i_subplot += 1

        #plot fit
        axes.append(plt.subplot2grid(subplot_size, (0, i_subplot)))
        plt.pcolormesh(self.x, self.y, self.predict(self.x, self.y))
        plt.title('Fit')
        plt.colorbar()
        i_subplot += 1

        # Difference
        if plot_residuals:
            axes.append(plt.subplot2grid(subplot_size, (0, i_subplot)))
            plt.pcolormesh(self.x, self.y,  self.z-self.predict(self.x, self.y), cmap='seismic')
            plt.title('Data - Fit')
            plt.colorbar()
            i_subplot += 1



        # guess

        # print list of fitting parameters
        if print_params:
            # print model name
            summary = f'Model: {self.fittype}\n'

            # print the math expression, either with plain text or latex
            # if hasattr(self.model, 'tex'):
            #	summary += self.model.tex +'\n'
            # else:
            summary += self.model.string + '\n'

            # print fit paramters
            ljust_len = max([len(s) for s in self.fitvars])  # find the longs fit variable name

            for i in range(len(self.fitvars)):
                if self.fix and self.fitvars[i] in self.fix:
                    summary += '\n' + self.fitvars[i].ljust(ljust_len) + ': %.3g (FIXED)' % self.guess[i]
                else:
                    summary += '\n' + self.fitvars[i].ljust(ljust_len) + ': ' + utility.format_error(self.params[i],
                                                                                                     self.errors[i])

            # print degrees of freedom
            summary += f'\n\nN DOF: {self.n_DOF}'

            # print chi2 estimate if data has errors
            if self.has_dz:
                chi2 = self.get_chi2()
                summary += '\nchi2    = %2.4g' % chi2
                summary += '\nchi2red = %2.4f' % (chi2 / self.n_DOF)
                summary += '\np = %2.4f' % self.get_pval()

            y_cord = 1
            plt.text(1.4, y_cord, summary, transform=axes[-1].transAxes, horizontalalignment='left', fontfamily='monospace',
                     verticalalignment='top')

        # return fig handle
        return fig

    def plot_guess(self, N=200):

        fig = plt.figure()
        plt.pcolormesh(self.x, self.y, self.get_guess(self.x, self.y))
        plt.title('Fit guess')
        plt.colorbar()

        return fig