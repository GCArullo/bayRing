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
