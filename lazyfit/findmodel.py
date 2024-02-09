# extra helper function to make it easier to access the built-models
# we will achieve this by remove upper case letters from the user input and using a look-up dictionary
# eg. the inputs "sin" and "sine" will both be valid.
import re
import lazyfit.models as models

def find_model(model_name):
    '''tried to find an internal fit model.
    Returns the fit model if it exists, otherwise returns None
    '''

    # remove special characters and convert to lower case
    model_name = re.sub('[\(\)\{\}<>\_\-]', '', model_name)
    model_name = model_name.lower()

    # see if it exists
    if hasattr(models, model_name):
        return getattr(models, model_name)
    if model_name in abbreviations:
        return getattr(models, abbreviations[model_name])
    return None


abbreviations = {
    'lorentzian' : 'lorentz',
    'exponential': 'exp',
    'biexp': 'biexponential',
    'convolvedexponential': 'convexp',
    'sin': 'sine',
    'dampsin' : 'dampsine', # lol
    'dampedsin': 'dampsine',
    'dampedsine': 'dampsine',
    '2lvlsat':'twolvlsat',
    'linear': 'lin',
    'poly1': 'lin',
    'quad': 'quadratic',
    'logisticpulse':'logpulse',
    'dualgauss': 'dualgaussian',
    't1':'T1' # here we need to conver back to uppser case
}

