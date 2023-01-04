# LazyFit
Lazyfit provides a convenient wrapper class for the Scipy curve fitting function.
This allows fitting, plotting and statistical tests to be performed from a single Python object.

For mundane fitting tasks, Lazyfit can yield a fit and plot with just two lines of code:

![Alt text](screenshots/Sine_example.PNG?raw=true "Title")

Lazyfit contains a library of common fit models encountered in the physical sciences. 
These models automatically estimate good starting points and bounds for the fit parameters, thus eliminating manual estimation.

Finally, lazyfit also contains a wrapper for fitting two-dimensional data, which removes the tedious process of converting data between matrices and 1d-arrays.

# Documentation
Lazyfit is documented in a number of Jupyter notebooks.
- The features of the lazyfit wrapper are covered in [example_notebooks/feature_overview.ipynb](example_notebooks/feature_overview.ipynb)
- A catalogue of built-in fit modes is given in [example_notebooks/model_catalog.ipynb](example_notebooks/model_catalog.ipynb)
- Fitting of 2-dimensional data is covered in [example_notebooks/2d_fitting.ipynb](example_notebooks/2d_fitting.ipynb)

# Key features
- Built-in plotting with printing of model parameters, errors, chi2 values and p-values.
- Fit model library with automatic estimation of initial parameter values.
- Fix fit parameters by passing a dictionary 
- Easy access to fit parameters and errors formatted as either arrays or dictionaries.
- Multiple keyword-based plotting features including residual plotting and logarithmic axes.
- Fit wrapper for 2d data

# Development notes
This project is heavily tailored towards my own fitting needs encountered as a physicist working within experimental quantum optics.
For this reason, the model library probably omits many fit models which are essential to other disciplines.
Fortunately, it is easy to add your own fit models as explained in the feature overview.

Currently, 2d fitting is at an experimental stage.