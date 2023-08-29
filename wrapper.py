import lazyfit.models as models
import numpy as np
pi = np.pi
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import lazyfit.utility as utility
import types
import textwrap

def fit(fittype, x, y, dy=None, guess=None, bounds=None, fix={}, verbose=False, options={}):
    """Provides a short cut to creating a Wrapper object and calling fit()"""
    f = Wrapper(fittype, x, y, dy, guess, bounds, fix, verbose)
    try:
        f.fit(options)
    except Exception as e:
        print('Could not perform fit. Try plotting the data and fit guess with .plot_guess(). Stack trace:')
        raise e
    return f


class Wrapper:
    def __init__(self, fittype, x, y, dy=None, guess=None, bounds=None, fix={}, verbose=False):

        self.fittype = fittype
        self.verbose = verbose
        self.fix = fix

        # check dimensions of input data
        if x.size != y.size:
            raise Exception(
                f'Dimensions of data to not match. Wrapper got x {x.shape}, y {y.shape}. x and y must have same length')

        # clean data
        self.x, self.y, n_bad = utility.clean_data(x, y)
        if n_bad > 0:
            print(f'Warning: Removing {n_bad} cases of NaN or Inf from data to enable fitting')

        # errors
        if dy is None:
            self.has_dy = False
            self.dy = dy
        else:
            if isinstance(dy, float) or isinstance(dy, int):
                # assume this is a constant error
                self.dy = np.ones(len(self.x)) * dy
            else:
                self.dy = dy
            self.has_dy = True

        # find fit model
        if type(fittype) is models.LazyFitModel: # a fitmodel saved in a simplenamespace was passed
            self.model = fittype
        elif type(fittype) is str:
            try:
                self.model = getattr(models, fittype)
            except AttributeError:
                raise Exception(f'No fit model named "{fittype}"')
        else:
            raise Exception('Invalid fit model. Must either be a lazyfit.models.FitModel object or a string referring to a built-in model.')

        # extrac stuff from fit model
        self.f = self.model.f
        self.fitvars = self.model.get_param_names()
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
            self.bounds[0][ind] = val * 0.999999 - utility.EPSILON
            self.bounds[1][ind] = val * 1.000001 + utility.EPSILON
            self.guess[ind] = val

    def fit(self, options={}):
        # do the actual fit
        self.fit_res = scipy.optimize.curve_fit(self.f, self.x, self.y, self.guess, self.dy, bounds=self.bounds, **options)

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

    def plot(self, N=200, print_params=True, plot_guess=False, logy=False, logx=False, plot_residuals=False, figsize=(8,4), marker='o',
             xlabel='', ylabel=''):
        '''Plot the data and the fit.
        
        Kwargs:
        N               (int, 200) Number of points used to draw fit and guess curves
        print_params    (bool, True) Print the fit parameters to the right of the plot
        plot_guess      (bool, False) Plot the initial guess
        logy            (bool, False) Plot with logarithmic y axis
        logx            (bool, False) Plot with logarithmic x axis
        plot_residuals  (bool, False) Insert plot below showing fit residuals
        figsize         (tupple), figure size
        marker             (sting, 'o') marker used for plotting data
        xlabel          (string) x-axis label
        ylabel          (string) y-axis label
        '''

        fig = plt.figure(figsize=figsize)

        # determine plot layout. Use a 3x3 grid. Plot is first 2 colums and all rows in the absence of residuals plot or first two rows in case of residual plot
        if plot_residuals:
            ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
        else:
            ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)

        # plot data
        if self.has_dy:
            ax1.errorbar(self.x, self.y, self.dy, label='Data', zorder=0, color='tab:blue', fmt=marker)
        else:
            ax1.plot(self.x, self.y, marker, label='Data', zorder=0, color='tab:blue')

        # plot fit and guess
        xdummy = np.linspace(np.min(self.x), np.max(self.x), N)  # create finer xaxis for fit and guess
        ax1.plot(xdummy, self.predict(xdummy), label='Fit', zorder=10, color='tab:red')  # plot fig
        if plot_guess:
            ax1.plot(xdummy, self.f(xdummy, *self.guess), label='Guess', zorder=5, color='tab:green')  # plot guess

        # formatting of plot
        if logy:
            ax1.set_yscale('log')
        if logx:
            ax1.set_xscale('log')

        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        plt.ylabel(ylabel)
        if not plot_residuals:
            plt.xlabel(xlabel)

        # plot residuals in subplot
        if plot_residuals:
            ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
            if self.has_dy:
                ax2.errorbar(self.x, self.y - self.predict(self.x), self.dy, label='Data', zorder=0, color='tab:blue', fmt=marker)
            else:
                ax2.plot(self.x, self.y - self.predict(self.x), marker, label='Data', zorder=0, color='tab:blue')

            plt.ylabel('Residual')
            plt.grid(axis='y')
            plt.xlabel(xlabel)

            # get rid of the tick labels on the top xaxis
            plt.setp(ax1.get_xticklabels(), visible=False)

        # print list of fitting parameters
        if print_params:
            ax3 = plt.subplot2grid((3, 3), (0, 2), rowspan=3)
            ax3.axis('off') # remove axis

            # print model name
            summary = f'Model: {self.model.name}\n'

            #print the math expression
            if hasattr(self.model, 'string'):
                summary += textwrap.fill(self.model.string, width=24) + '\n' # wrap the math expression to avoid the figure getting too wide

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

            # wrap the text

            plt.text(-0.1, 0.75, summary, transform=ax3.transAxes, horizontalalignment='left', fontfamily='monospace',
                     verticalalignment='top', wrap=True)
            
        # TODO. Set background colour for saving figure, make sure it always looks nice.
        # TODO: text folding, how?

        plt.subplots_adjust(left=0.08, right=1, top=0.92, bottom=0.08) # reduce the margins, mainly for saving

        # return fig handle
        return fig

    def plot_guess(self, N=200):
        '''Plot data and initial guess'''

        fig = plt.figure()

        if self.has_dy:
            plt.errorbar(self.x, self.y, self.dy, fmt='o', label='Data', color='tab:blue')
        else:
            plt.plot(self.x, self.y, 'o', label='Data')

        xdummy = np.linspace(np.min(self.x), np.max(self.x), N)
        plt.plot(xdummy, self.f(xdummy, *self.guess), label='Guess', color='tab:green')
        # TODO: Legend???

        return fig