# LazyFit
Lazyfit provides a convenient wrapper for Scipy curve fitting.
This allows fitting, plotting, and statistical tests using a single Python object.

For mundane tasks, Lazyfit can yield a fit and plot with just two lines of code:

![Alt text](screenshots/Sine_example.PNG?raw=true "Title")

Additionally, Lazyfit contains a growing library of common fit models encountered in the physical sciences. These models automatically estimate good starting points and bounds for the fit parameters, thus eliminating manual estimation.

Finally, Lazyfit also contains a wrapper for fitting two-dimensional data, which removes the tedious process of converting data between matrices and 1d-arrays.

# Documentation
Lazyfit is documented across several Jupyter notebooks.
- The features of the Lazyfit wrapper are covered in [example_notebooks/feature_overview.ipynb](example_notebooks/feature_overview.ipynb)
- A catalogue of built-in fit modes is given in [example_notebooks/model_catalogue.ipynb](example_notebooks/model_catalogue.ipynb)
- Fitting of 2-dimensional data is covered in [example_notebooks/2d_fitting.ipynb](example_notebooks/2d_fitting.ipynb)

# Key features
- Easy plotting of data, model parameters, errors, chi^2 values and p-values.
- Fit model library with automatic estimation of initial parameter values.
- Access fit parameters and errors using either arrays or dictionaries.
- Lock the values of fit parameters by passing a dictionary.
- Multiple keyword-based plotting features including residual plotting and logarithmic axes.
- Fit wrapper for 2d data.

# Installation
You can pip install this library using "pip install git+https://github.com/M-Hayhurst/lazyfit@release". This requires git to be installed on your computer.
Alternatively, switch to the "release" branch, download the contents of the git, and run "python setup.py install".

# Development notes
This project is heavily tailored towards my fitting needs as a physicist working within experimental quantum optics. For this reason, the model library probably omits many fit models essential to other disciplines. Fortunately, it is easy to add custom fit models as explained in the feature overview.

Currently, 2d fitting is at an experimental stage.