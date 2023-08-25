import numpy as np, warnings

import qnm
import pyRing.utils as pyRing_utils

twopi = 2.*np.pi
qnm.download_data()

def read_quad_modes(QQNM_modes, l_NR, m):  

    """

    Read the quadratic modes from QQNM_modes string.

    Parameters
    ----------

    QQNM_modes : string
        String containing the quadratic modes.
    l_NR : int
        l-mode of the NR waveform.
    m : int
        m-mode of the NR waveform.

    Returns
    -------

    quad_modes : dict
        Dictionary containing the quadratic modes.

    """

    quad_modes_list   = QQNM_modes[0].split(',')

    quad_modes        = {'sum': [], 'diff': []}
    for i in range(len(quad_modes_list)):

        term_i_parent = quad_modes_list[i].split('x')[0]
        mode_i_child1 = quad_modes_list[i].split('x')[1]
        mode_i_child2 = quad_modes_list[i].split('x')[2]

        term = term_i_parent[0]
        if('-' in mode_i_child1): l1, m1, n1 = int(mode_i_child1[0]), -int(mode_i_child1[2]), int(mode_i_child1[3])
        else                    : l1, m1, n1 = int(mode_i_child1[0]),  int(mode_i_child1[1]), int(mode_i_child1[2])
        if('-' in mode_i_child2): l2, m2, n2 = int(mode_i_child2[0]), -int(mode_i_child2[2]), int(mode_i_child2[3])
        else                    : l2, m2, n2 = int(mode_i_child2[0]),  int(mode_i_child2[1]), int(mode_i_child2[2])

        # Impose angular selection rules.
        if not(np.abs(m)==np.abs(m1)+np.abs(m2)): warnings.warn("\n* You are attempting to model quadratic mode coupling with m indices that violate |m|=|m1|+|m2|. Are you sure about your model?\n")
        if((l_NR>l1+l2) or (l_NR<l1-l2))        : warnings.warn("\n* Angular selection rules require the l1-l2 <= l <= l1+l2, but the values l={},l1={}, l2={} were passed. Are you sure about your model?\n".format(l_NR,l1,l2))

        # The overtone index is dummy for quadratic modes.
        if(term=='P'):
            quad_modes['sum'].append(
                (
                    (l_NR,   m,      0), 
                    (l1,     m1,    n1),
                    (l2,     m2,    n2),
                )
            )
        elif(term=='M'):
            quad_modes['diff'].append(
                (
                    (l_NR,   m,      0), 
                    (l1,     m1,    n1),
                    (l2,     m2,    n2),
                )
            )
        else: raise ValueError("* Incorrect format of quadratic mode pased: {} (example of expected format: `--l-NR 4 --m 4 --QQNM-modes Px220x321,Px220x2-20`)".format(quad_modes_list[i]))

    return quad_modes

def construct_full_modes(linear_modes, quad_modes):

    """

    Construct the full list of modes from the linear and quadratic modes.

    Parameters
    ----------

    modes : list
        List of linear modes.
    quad_modes : dict   
        Dictionary containing the quadratic modes.

    Returns
    -------

    modes_full : list
        List of all modes.

    """

    modes_full = []
    for linear_mode in linear_modes: modes_full.append(linear_mode)
    if quad_modes is not None:
        for quad_term in quad_modes.keys():
            for mode in quad_modes[quad_term]: 
                modes_full.append(mode[0])
                modes_full.append(mode[1])
                modes_full.append(mode[2])

    # Remove duplicates.
    modes_full = list(dict.fromkeys(modes_full))

    return modes_full 

def read_linear_modes(modes_input, l_NR, m):

    """

    Read the linear modes from the modes_input string.

    Parameters
    ----------

    modes_input : string
        String containing the linear modes.
    l_NR : int
        l-mode of the NR waveform.
    m : int
        m-mode of the NR waveform.

    Returns
    -------

    modes : list
        List of linear modes.
    modes_string : string
        String containing the linear modes.

    """

    modes_list   = modes_input.split(',')
    modes        = []
    for i in range(len(modes_list)):
        if('-' in modes_list[i]): l_ring,m_ring,n = int(modes_list[i][0]), -int(modes_list[i][2]), int(modes_list[i][3])
        else                    : l_ring,m_ring,n = int(modes_list[i][0]),  int(modes_list[i][1]), int(modes_list[i][2])
        modes.append((l_ring,m_ring,n))

    return modes

def read_Kerr_modes(modes_input, QQNM_modes, charge, l_NR, m, NR_metadata):

    """

    Read the linear and quadratic modes from the modes_input string.

    Parameters
    ----------

    modes_input : string
        String containing the linear modes.

    QQNM_modes : string
        String containing the quadratic modes.

    charge : bool
        If True, the Kerr metric is used.

    l_NR : int
        l-mode of the NR waveform.

    m : int
        m-mode of the NR waveform.

    Returns
    -------

    modes_full : list
        List of all modes.

    modes_string : string
        String containing the linear modes.

    quad_modes_string : string
        String containing the quadratic modes.

    """

    if(modes_input is None): raise ValueError("You are using the Kerr template, but you did not specify any QNM modes to be fitted. Please, specify them with the --QNM-modes option.")

    # Linear modes            
    linear_modes = read_linear_modes(modes_input, l_NR, m)

    # Quadratic modes
    if(QQNM_modes is not None): quad_modes = read_quad_modes(QQNM_modes, l_NR, m)
    else                      : quad_modes = None

    modes_full = construct_full_modes(linear_modes, quad_modes)

    qnm_cached = {}
    for (l_ring,m_ring,n) in modes_full:

        if(charge): 
            interpolate_freq, interpolate_tau = pyRing_utils.qnm_interpolate_KN(2, l_ring, m_ring, n)
            freq = (interpolate_freq(NR_metadata['af'], NR_metadata['qf']) / NR_metadata['Mf']) * (1./twopi)
            tau  = -1./(interpolate_tau(NR_metadata['af'], NR_metadata['qf'])) * NR_metadata['Mf']
            qnm_cached[(2, l_ring, m_ring, n)] = {'f': freq, 'tau': tau}

        else      : 

            omega, _, _ = qnm.modes_cache(s=-2,l=l_ring,m=m_ring,n=n)(a=NR_metadata['af'])
            freq        = (np.real(omega) / NR_metadata['Mf']) * (1./twopi)
            tau         = -1./(np.imag(omega)) * NR_metadata['Mf']
            qnm_cached[(2, l_ring, m_ring, n)] = {'f': freq, 'tau': tau}

    return linear_modes, quad_modes, qnm_cached

# FIXME: this is a repetition of the function above, and should be eliminated.
def construct_single_QNM_cached_dict(l, m, n, charge, NR_metadata):

    """

    Read the modes from the modes_input string.

    Parameters
    ----------

    N_modes : int
        Number of modes to be fitted.

    l_NR : int
        l-mode of the NR waveform.

    m : int
        m-mode of the NR waveform.

    charge : bool
        If True, the charged Kerr-Newman spectrum is computed.

    Returns
    -------

    modes : list
        List of damped-sinusoid modes.

    modes_string : string
        String containing the damped-sinusoid modes.

    qnm_cached : dict
        Dictionary containing the values of the QNM frequencies and damping times.

    """ 

    # Load the fundamental mode, to be used for post-processing comparisons.
    qnm_cached = {}
    if(charge): 
        interpolate_freq, interpolate_tau = pyRing_utils.qnm_interpolate_KN(2, l, m, n)
        freq = (interpolate_freq(NR_metadata['af'], NR_metadata['qf']) / NR_metadata['Mf']) * (1./twopi)
        tau  = -1./(interpolate_tau(NR_metadata['af'], NR_metadata['qf'])) * NR_metadata['Mf']
        qnm_cached[(2, l, m, n)] = {'f': freq, 'tau': tau}

    else      :

        omega, _, _ = qnm.modes_cache(s=-2,l=l,m=m,n=n)(a=NR_metadata['af'])
        freq        = (np.real(omega) / NR_metadata['Mf']) * (1./twopi)
        tau         = -1./(np.imag(omega)) * NR_metadata['Mf']
        qnm_cached[(2, l, m, n)] = {'f': freq, 'tau': tau}

    return qnm_cached

def read_tail_modes(modes_input):

    """

    Read the linear modes at which will be added a tail.

    Parameters
    ----------

    modes_input : string
        String containing the linear modes to which a tail is added.

    Returns
    -------

    modes : list
        List of the (l,m) linear modes with n=0 to which the tail is added.

    """

    modes_list   = modes_input.split(',')
    modes        = []
    for i in range(len(modes_list)):
        if('-' in modes_list[i]): l_ring,m_ring = int(modes_list[i][0]), -int(modes_list[i][2])
        else                    : l_ring,m_ring = int(modes_list[i][0]),  int(modes_list[i][1])
        modes.append((l_ring,m_ring))

    return modes