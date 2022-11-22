import lazyfit.models as models
import numpy as np
pi = np.pi
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import lazyfit.utility as utility
import inspect


def fit(fittype, x, y, dy=None, guess=None, bounds=None, fix={}, verbose=False, options={}):
    """Provides a short cut to creating a Wrapper object and calling fit()"""
    f = Wrapper(fittype, x, y, dy, guess, bounds, fix, verbose)
    f.fit(options)
    return f


epsilon = 1E-20


class Wrapper:
    def __init__(self, fittype, x, y, dy=None, guess=None, bounds=None, fix={}, verbose=False):

        self.fittype = fittype
        self.verbose = verbose
        self.fix = fix

        # clean data
        self.x, self.y, n_bad = utility.clean_data(x, y)
        if n_bad > 0:
            print(f'Warning: Removing {n_bad} cases of NaN or Inf from data to enable fitting')

        # errors
        if dy is None:
            self.has_dy = False
        else:
            if isinstance(dy, float) or isinstance(dy, int):
                # assume this is a constant error
                self.dy = np.ones(len(self.x)) * dy
            else:
                self.dy = dy
            self.has_dy = True

        # find fit model
        try:
            self.model = getattr(models, fittype)
        except AttributeError:
            raise Exception(f'No fit model named "{fittype}"')

        # extrac stuff from fit model
        self.f = self.model.f
        self.fitvars = inspect.getargspec(self.f).args[
                       1::]  # get fit function arguments but without 'x' which is always first argument
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

    def fit(self, options={}):
        # do the actual fit
        self.fit_res = scipy.optimize.curve_fit(self.f, self.x, self.y, self.guess, bounds=self.bounds, **options)

        # save parameters and errors as arrays
        self.params = self.fit_res[0]
        self.COVB = self.fit_res[1]
        self.errors = np.sqrt(np.diag(self.COVB))

        # also save fit paramters and errors as dicts
        self.params_dict, self.errors_dict = {}, {}
        for i, var in enumerate(self.fitvars):
            self.params_dict[var] = self.params[i]
            self.errors_dict[var] = self.errors[i]

    def get_chi2(self):
        return np.sum((self.y - self.predict(self.x)) ** 2 / self.dy ** 2)

    def predict(self, x):
        return self.f(x, *self.params)

    def get_pval(self):
        '''Return probability of null hypothesis given chisquared sum, and degrees of freedom.
        Only works if you supplied errors to the fitting routine'''
        return scipy.stats.distributions.chi2.sf(self.get_chi2(), self.n_DOF)

    def plot(self, N=200, print_params=True, plot_guess=False, logy=False, plot_residuals=False, figsize=None, fmt='o'):

        fig = plt.figure(figsize=figsize)

        if plot_residuals:
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        else:
            ax1 = plt.gca()

        if self.has_dy:
            ax1.errorbar(self.x, self.y, self.dy, fmt=fmt, label='Data', zorder=0)
        else:
            ax1.plot(self.x, self.y, fmt, label='Data', zorder=0)

        xdummy = np.linspace(np.min(self.x), np.max(self.x), N)  # create finer xaxis for fit and guess
        ax1.plot(xdummy, self.predict(xdummy), label='Fit', zorder=10)  # plot fig

        if plot_guess:
            ax1.plot(xdummy, self.f(xdummy, *self.guess), label='Guess', zorder=5)  # plot guess

        if logy:
            ax1.set_yscale('log')

        # make legend
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

        # plot residuals in subplot
        if plot_residuals:
            ax2 = plt.subplot2grid((3, 1), (2, 0))
            if self.has_dy:
                ax2.errorbar(self.x, self.y - self.predict(self.x), self.dy, fmt='o')
            else:
                ax2.plot(self.x, self.y - self.predict(self.x))

            plt.ylabel('Residual')
            plt.grid(axis='y')

            # get rid of the tick labels on the top xaxis
            plt.setp(ax1.get_xticklabels(), visible=False)

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
            if self.has_dy:
                chi2 = self.get_chi2()
                summary += '\nchi2    = %2.4g' % chi2
                summary += '\nchi2red = %2.4f' % (chi2 / self.n_DOF)
                summary += '\np = %2.4f' % self.get_pval()

            y_cord = 0.6 if plot_residuals else 0.7
            plt.text(1.02, y_cord, summary, transform=ax1.transAxes, horizontalalignment='left', fontfamily='monospace',
                     verticalalignment='top')

        # return fig handle
        return fig

    def plot_guess(self, N=200):

        fig = plt.figure()

        if self.has_dy:
            plt.errorbar(self.x, self.y, self.dy, fmt='o', label='Data')
        else:
            plt.plot(self.x, self.y, 'o', label='Data')

        xdummy = np.linspace(np.min(self.x), np.max(self.x), N)
        plt.plot(xdummy, self.f(xdummy, *self.guess), label='Guess')

        return fig