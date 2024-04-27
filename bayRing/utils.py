import numpy as np

#FIXME: implement fixed params reading for all models
def get_param_override(fixed_params, x, name):
    """
        Function returning either a sample or the fixed value for the parameter considered.
        ---------------
        
        Returns x[name], unless it is over-ridden by
        value in the fixed_params dictionary.
        
    """
    if name in fixed_params: return fixed_params[name]
    else:                    return x[name]

def find_longest_name_length(names):
    
    """
    
    Find the length of the longest name in a list of names.

    Parameters
    ----------

    names : list
        List of names.

    Returns
    -------

    longest_name_length : int
        Length of the longest name in the list.
    
    """

    longest_name_length = 0
    for key in names:
        if(len(key)>longest_name_length): longest_name_length = len(key)

    return longest_name_length

def minimisation_compatibility_check(parameters):

    if not(parameters['template']=='Kerr'): raise ValueError("Minimization algorithm only works for Kerr parameters['template']." )
    if not(parameters['QQNM_modes']==None): raise ValueError("Minimization algorithm does not work with QQNM modes.")
    if not(parameters['tail']==0)         : raise ValueError("Minimization algorithm does not work with Kerr tail." )

    return

# Function taken from watpy (https://git.tpi.uni-jena.de/core/watpy).

def diff1(xp, yp, pad=True):

    """
    Computes the first derivative of y(x) using centered 2nd order
    accurate finite-differencing

    This function returns an array of yp.shape[0]-2 elements

    NOTE: the data needs not to be equally spaced
    """
    
    dyp = [(yp[i+1] - yp[i-1])/(xp[i+1] - xp[i-1]) \
            for i in range(1, xp.shape[0]-1)]
    dyp = np.array(dyp)

    if pad==True:
        dyp = np.insert(dyp, 0, dyp[0])
        dyp = np.append(dyp, dyp[-1])

    return dyp