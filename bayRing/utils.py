import io, numpy as np, pkg_resources, os, re, tarfile, warnings

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

def set_prefix(warning_message=True):
    
    """
        Set the prefix path for the data files.

        Parameters
        ----------

        warning_message : bool
            If True, a warning message is printed if the environment variable is not set.

        Returns
        -------

        prefix : str
            Path to the data files.

    """
    
    # Check environment
    try:
        prefix = os.path.join(os.environ['BAYRING_PREFIX'])
    except KeyError:
        prefix = ''
        if(warning_message):
            warnings.warn("The requested functionality requires data not included in the package. Please set a $BAYRING_PREFIX variable which contains the path to such data. This can be done by setting 'export BAYRING_PREFIX= yourpath' in your ~/.bashrc file. Typically, BAYRING_PREFIX contains the path to the clone of the repository containing the source code.")
    return prefix
    
def filter_dict_by_key(a, target_key):

    """

    Filter a dictionary by a specific key, returning a new dictionary with the specified key and its corresponding values.

    Parameters
    ----------

    a : dict
        Dictionary to be filtered.
    target_key : str
        Key to filter the dictionary by.

    Returns
    -------

    filtered : dict
        Filtered dictionary containing the specified key and its corresponding values.

    """

    filtered = {}
    for category in ['linear', 'quadratic']:
        subdict = a.get(category, {})
        if target_key in subdict: filtered[category] = {target_key: subdict[target_key]}
        else                    : filtered[category] = {}

    return filtered

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

def read_psi4_RIT_format(tar_gz_path, asc_name):

    """

    Read the content of an .asc file inside a tar.gz file, and convert it to a dictionary of numpy arrays, assuming the format of RIT.

    Parameters
    ----------

    tar_gz_path : str
        Path to the tar.gz file containing the .asc file.

    Returns
    -------

    asc_files_data : dict
        Dictionary containing the content of the .asc file.

    """

    # Initialize a dictionary to store the content of .asc files
    asc_files_data = {}

    # Open the tar.gz file and read .asc files directly from it
    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.asc'):
                # Read the .asc file content
                file = tar.extractfile(member)
                if file is not None:
                    file_content = file.read()
                    # Use io.StringIO to convert the file content to a file-like object
                    file_like_object = io.StringIO(file_content.decode('utf-8'))
                    
                    # Skip the first three lines
                    for _ in range(3):
                        file_like_object.readline()
                    
                    # Read the header (fourth line)
                    header = file_like_object.readline().strip().split()
                    
                    # Remove the strings before the symbol ":"
                    header = [col.split(':')[-1] for col in header]

                    # Read the rest of the content into a numpy array
                    data = np.loadtxt(file_like_object)
                    
                    # Ensure the data is 2D
                    if data.ndim == 1:
                        data = data[np.newaxis, :]
                    
                    # Transpose the numpy array to invert its shape
                    data = data.T
                    
                    # Check if the number of headers matches the number of data columns
                    if len(header) != data.shape[0]:
                        print(f"Warning: Header length {len(header)} does not match data columns {data.shape[0]} for file {member.name}.")
                        continue
                    
                    # Create a dictionary with column names as keys and corresponding columns as numpy arrays
                    file_data_dict = {header[i]: data[i] for i in range(len(header))}

                    if not member.name.startswith('ExtrapPsi4'):
                        cleaned_name = re.sub(r'.*ExtrapPsi4', 'ExtrapPsi4', member.name)
                        asc_files_data[cleaned_name] = file_data_dict
                    else:
                        asc_files_data[member.name] = file_data_dict

    try   : output = asc_files_data[asc_name]
    except: 
        # A few simulation had an inconsistent naming convention
        asc_name_parts = asc_name.split('/')
        asc_name_parts[0] = asc_name_parts[0].replace('ecc', '')
        asc_name_new = asc_name_parts[0] + '/' + asc_name_parts[1]
        output = asc_files_data[asc_name_new]

    return output