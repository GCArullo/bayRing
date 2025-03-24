# Standard python packages
import corner, h5py, matplotlib.pyplot as plt, numpy as np, os, qnm, scipy.linalg as sl, seaborn as sns, shutil, numba
from scipy.interpolate           import interp1d

# GW-packages
from pycbc.psd                   import from_txt
from pycbc.types.timeseries      import TimeSeries
from pycbc.psd                   import aLIGOZeroDetHighPower
from pycbc.types.frequencyseries import FrequencySeries
from pycbc.filter                import sigma, overlap as compute_FD_overlap, overlap_cplx as compute_FD_overlap_cplx, match as compute_FD_match, matched_filter_core, matched_filter
import lal
from lal.antenna                 import AntennaResponse

# Package internal imports
import bayRing.utils             as utils
import bayRing.waveform_utils    as waveform_utils

# Costants
twopi = 2.*np.pi

# Color palette
colbBlue   = "#4477AA"
colbRed    = "#EE6677"
colbGreen  = "#228833"
colbYellow = "#CCBB44"
colbCyan   = "#66CCEE"
colbPurple = "#AA3377"
colbGray   = "#BBBBBB"

# Conversions
C_mt=(lal.MSUN_SI * lal.G_SI) / (lal.C_SI**3) #s, converts a mass expressed in solar masses into a time in seconds
C_md=(lal.MSUN_SI * lal.G_SI)/(1e6*lal.PC_SI*lal.C_SI**2) #adimensional, converts a mass expressed in solar masses to a distance in Megaparsec

def read_results_object_from_previous_inference(parameters):

    if(parameters['Inference']['method'] == 'Minimization'):

        utils.minimisation_compatibility_check(parameters)

        results_object_tmp = np.genfromtxt(os.path.join( parameters['I/O']['outdir'],'Algorithm/Minimization_results.txt'), names=True, deletechars="")
        results_object = {}
        for key in results_object_tmp.dtype.names:
            print(results_object_tmp[key])
            results_object[key] = results_object_tmp[key]

    elif(parameters['Inference']['method'] == 'Nested-sampler'):

        if(parameters['Inference']['sampler'] == 'cpnest'):
            try   :   results_object = np.genfromtxt(os.path.join( parameters['I/O']['outdir'],'Algorithm/posterior.dat'), names=True, deletechars="", delimiter = ",")
            except:   results_object = np.genfromtxt(os.path.join( parameters['I/O']['outdir'],'Algorithm/posterior.dat'), names=True, deletechars="")
        elif(parameters['Inference']['sampler'] == 'raynest'):
            filename        = os.path.join( parameters['I/O']['outdir'],'Algorithm/raynest.h5')
            h5_file         = h5py.File(filename,'r')
            results_object  = h5_file['combined'].get('posterior_samples')

    else: raise ValueError('Method {} not recognised.'.format(parameters['Inference']['method']))

    return results_object

def print_point_estimate(results_object, names, method):

    """

    Print the point estimates of the results of a minimization or a nested sampling algorithm.

    Parameters
    ----------

    results_object : object
        Object containing the results of the minimization or nested sampling algorithm.

    names : list
        List of the names of the parameters.

    method : str
        Method used to obtain the results from which the point estimates will be drawn. Can be either 'Minimization' or 'Nested-sampler'.

    Returns
    -------

    Nothing, but prints the point estimates.

    """

    if(method=='Minimization'):
        longest_name_length = utils.find_longest_name_length(results_object.keys())
        for key in results_object.keys(): print('{} : {:.12f}'.format(key.ljust(longest_name_length), results_object[key]))
    else:
        longest_name_length = utils.find_longest_name_length(names)
        for key in names:
            median      = np.median(results_object[key])
            lower_bound = median-np.percentile(results_object[key], 5)
            upper_bound = np.percentile(results_object[key], 95)-median
            print('{} : {:.12f} + {:.12f} - {:.12f}'.format(key.ljust(longest_name_length), median, upper_bound, lower_bound))

    return

def save_results_minimization(results, outdir):

    """

    Save the results of a minimization algorithm.

    Parameters
    ----------

    results : dict
        Dictionary containing the results of the minimization algorithm.

    outdir : str
        Output directory.

    Returns
    -------

    Nothing, but saves the results of the minimization algorithm in a text file.

    """

    len_params    = len(results.values())
    header_string = ''
    for key in results.keys(): header_string += '{}\t'.format(key)
    
    results_output = np.array([np.array(list(results.values())[i]) for i in range(len_params)]).reshape((1,len_params))
    np.savetxt(os.path.join(outdir,'Algorithm','Minimization_Results.txt'), results_output, header = header_string)
    
    return

def store_and_print_amp_phi(amp_name, phi_name, t0, omega, tau, results_object, longest_name_length, outdir):

    """

    Store and print the amplitude and phase of the inferred mode when defined at t0.

    Parameters
    ----------

    amp_name : str
        Name of the amplitude parameter.
    
    phi_name : str
        Name of the phase parameter.

    t0 : float
        Time at which the amplitude and phase are defined.

    omega : float   
        Frequency of the mode.

    tau : float
        Damping time of the mode.

    results_object : dict
        Dictionary containing the results of the inference algorithm.

    longest_name_length : int
        Length of the longest parameter name.

    outdir : str
        Output directory.

    Returns
    -------

    Nothing, but stores and prints the amplitude and phase of the inferred mode.

    """

    exp_tau_factor   = np.exp(t0/tau)
    sum_omega_factor = t0 * omega

    amp_median =  np.exp(np.median(    results_object[amp_name]    )) *   exp_tau_factor
    amp_lower  =  np.exp(np.percentile(results_object[amp_name],  5)) *   exp_tau_factor
    amp_upper  =  np.exp(np.percentile(results_object[amp_name], 95)) *   exp_tau_factor
    phi_median =        (np.median(    results_object[phi_name]    )  - sum_omega_factor)%(2*np.pi)
    phi_lower  =        (np.percentile(results_object[phi_name],  5)  - sum_omega_factor)%(2*np.pi)
    phi_upper  =        (np.percentile(results_object[phi_name], 95)  - sum_omega_factor)%(2*np.pi)

    amp_lower_err  = amp_median - amp_lower
    amp_upper_err  = amp_upper  - amp_median
    phi_lower_err  = phi_median - phi_lower
    phi_upper_err  = phi_upper  - phi_median

    print('{} : {:.12f} + {:.12f} - {:.12f}'.format(amp_name.split('ln_')[-1].ljust(longest_name_length), amp_median, amp_upper_err, amp_lower_err))
    print('{} : {:.12f} + {:.12f} - {:.12f}'.format(phi_name.ljust(longest_name_length), phi_median, phi_upper_err, phi_lower_err))

    outFile_amp = open(os.path.join(outdir,'Peak_quantities/amps_tpeak.txt'), 'a')
    outFile_amp.write('{}\t{}\t{}\t{}\n'.format(amp_name.ljust(longest_name_length), amp_median, amp_lower, amp_upper))
    outFile_amp.close()
    outFile_phi = open(os.path.join(outdir,'Peak_quantities/phis_tpeak.txt'), 'a')
    outFile_phi.write('{}\t{}\t{}\t{}\n'.format(phi_name.ljust(longest_name_length), phi_median, phi_lower, phi_upper))
    outFile_phi.close()

    return

def post_process_amplitudes(t0, results_object, NR_metadata, qnm_cached, modes, quad_modes, outdir):

    """

    Post-process the amplitudes and phases of the inferred modes.

    Parameters
    ----------

    t0 : float
        Time at which the amplitude and phase are defined.

    results_object : dict
        Dictionary containing the results of the inference algorithm.

    NR_metadata : dict
        Dictionary containing the metadata of the NR simulation.

    qnm_interpolants : dict
        Dictionary containing the interpolants of the QNM frequencies and damping times.

    modes : list
        List of the modes to be inferred.

    quad_modes : list
        List of the quadrupole modes to be inferred.

    outdir : str
        Output directory.

    Returns
    -------

    Nothing, but stores and prints the amplitude and phase of the inferred mode.

    """

    print('\n* Amplitudes and phases at t_peak:\n')

    outFile_amp = open(os.path.join(outdir,'Peak_quantities/amps_tpeak.txt'), 'w')
    outFile_phi = open(os.path.join(outdir,'Peak_quantities/phis_tpeak.txt'), 'w')
    outFile_amp.write('#name\tmedian\tlower\tupper\n')
    outFile_phi.write('#name\tmedian\tlower\tupper\n')
    outFile_amp.close()
    outFile_phi.close()

    Mf = NR_metadata['Mf']
    af = NR_metadata['af']
    
    if 'qf' in NR_metadata.keys(): qf = NR_metadata['qf']
    else                         : qf = None

    if (quad_modes is not None): longest_name_length = len('phi_diff_x-yz_x-yz_x-yz')
    else                       : longest_name_length = len('phi_x-yz')

    for (l_x, m_x, n_x) in modes:

        amp_name = 'ln_A_{}{}{}'.format(l_x, m_x, n_x)
        phi_name =  'phi_{}{}{}'.format(l_x, m_x, n_x)

        omega, tau = qnm_cached[(2,l_x,m_x,n_x)]['f'] * twopi, qnm_cached[(2,l_x,m_x,n_x)]['tau']

        store_and_print_amp_phi(amp_name, phi_name, t0, omega, tau, results_object, longest_name_length, outdir)

    if(quad_modes is not None):
        for quad_term in quad_modes.keys():
            for ((l,m,n),(l1,m1,n1),(l2,m2,n2)) in quad_modes[quad_term]:

                quad_string = '{}_{}{}{}_{}{}{}_{}{}{}'.format(quad_term, l,m,n, l1,m1,n1, l2,m2,n2)
                amp_name = 'ln_A_{}'.format(quad_string)
                phi_name = 'phi_{}'.format(quad_string)

                omega1, tau1 = qnm_cached[(2,l1, m1, n1)]['f'] * twopi, qnm_cached[(2,l1, m1, n1)]['tau']
                omega2, tau2 = qnm_cached[(2,l2, m2, n2)]['f'] * twopi, qnm_cached[(2,l2, m2, n2)]['tau']

                tau   = (tau1 * tau2)/(tau1 + tau2)
                if  (quad_term=='sum' ): omega = omega1 + omega2
                elif(quad_term=='diff'): omega = omega1 - omega2

                store_and_print_amp_phi(amp_name, phi_name, t0, omega, tau, results_object, longest_name_length, outdir)

    return 

def l2norm_residual_vs_nr(results_object, inference_model, NR_sim, outdir):
    
    """

    Compare the residual of the fit with the NR error.

    Find the peak time of the amplitude.

    Parameters
    ----------

    results_object : dict
        Dictionary containing the results of the inference algorithm.
    
    inference_model : Nested sampler object
        Nested sampler object. 

    NR_sim : NR_sim
        NR simulation object.

    outdir : str
        output directory

    Returns
    ---------

    Nothing, but prints and stores in a file the L2 norm of residuals and NR_error.

    """

    NR_err_r, NR_err_i = np.real(NR_sim.NR_cpx_err_cut), np.imag(NR_sim.NR_cpx_err_cut)
    NR_r, NR_i         = np.real(NR_sim.NR_cpx_cut)     , np.imag(NR_sim.NR_cpx_cut)
    t_cut = NR_sim.t_NR_cut

    models_re_list = [np.real(np.array(inference_model.model(p))) for p in results_object]
    models_im_list = [np.imag(np.array(inference_model.model(p))) for p in results_object]

    wf_r = np.percentile(np.array(models_re_list),[50], axis=0)[0]
    wf_i = np.percentile(np.array(models_im_list),[50], axis=0)[0]

    l2_NR       = np.trapz(np.sqrt(      NR_err_r** 2 +       NR_err_i** 2), t_cut)
    l2_residual = np.trapz(np.sqrt((NR_r - wf_r) ** 2 + (NR_i - wf_i) ** 2), t_cut)

    print(f'\n* L2 norm of residual is {l2_residual}')
    print(f'\n* L2 norm of NR error is {l2_NR}\n')

    outFile_L2_errors = open(os.path.join(outdir,'Algorithm/L2_errors.txt'), 'w')
    outFile_L2_errors.write('# L2 norm of residual is \n')
    outFile_L2_errors.write(f'{l2_residual} \n')
    outFile_L2_errors.write('# L2 norm of NR error is \n')
    outFile_L2_errors.write(f'{l2_NR} \n')

    return 

def init_plotting():

    """
    
    Function to set the default plotting parameters.

    Parameters
    ----------
    None

    Returns
    -------
    Nothing, but sets the default plotting parameters.
    
    """
    
    plt.rcParams['figure.max_open_warning'] = 0
    
    plt.rcParams['mathtext.fontset']  = 'stix'
    plt.rcParams['font.family']       = 'STIXGeneral'

    plt.rcParams['font.size']         = 14
    plt.rcParams['axes.linewidth']    = 1
    plt.rcParams['axes.labelsize']    = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize']    = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize']   = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize']   = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize']   = plt.rcParams['font.size']
    plt.rcParams['xtick.major.size']  = 3
    plt.rcParams['xtick.minor.size']  = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size']  = 3
    plt.rcParams['ytick.minor.size']  = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    
    plt.rcParams['legend.frameon']             = False
    plt.rcParams['legend.loc']                 = 'center left'
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    
    return

def compare_with_GR_QNMs(results_object, qnm_cached, NR_sim, outdir):

    l,m              = NR_sim.l, NR_sim.m
    f_samples        = results_object['f_0']
    f_rd_fundamental = qnm_cached[(2,l,m,0)]['f']

    plt.figure()
    sns.histplot(f_samples       , color="darkred", fill=True , alpha=0.9, label='EFT fund mode')
    plt.axvline(f_rd_fundamental, color='black', linestyle='--', lw=2.2,  label='GR fund mode')
    plt.xlabel(r'$f_{fund}$')
    plt.ylabel(r'$p(f_{fund})$')
    plt.legend(loc='best')
    plt.savefig(os.path.join(outdir,'Plots/Results/f_fundamental.pdf'), bbox_inches='tight')

    return

def compute_FD_optimal_SNR(asd_file, h, n, f_min, f_max):

    # Ensure PSD matches the waveform's `delta_f`
    delta_f = 2*f_max/n
    psd     = from_txt(
                        filename        = asd_file,
                        length          = n,
                        delta_f         = delta_f,
                        low_freq_cutoff = f_min,
                        is_asd_file     = True
                    )

    h_tilde = h.to_frequencyseries(delta_f=delta_f)
    fd_snr  = sigma(h_tilde, psd=psd, low_frequency_cutoff=f_min)

    return fd_snr

@numba.njit
def fast_interpolation(x, xp, fp):
    """Numba-accelerated linear interpolation."""
    return np.interp(x, xp, fp)

def interpolate_waveform(t_start_g, t_end_g, M, wf_lNR, acf):

    """
    Interpolates the waveform to match the length of the autocovariance function (ACF).
    
    Parameters
    ----------
    - t_start_g (float) : Start time in geometrical units.
    - t_end_g   (float) : End time in geometrical units.
    - M (float)         : Mass of the system.
    - wf_lNR (array)    : The original NR waveform data.
    - acf (array)       : The autocovariance function (defines new length).

    Returns
    -------
    - wf_int (array) Interpolated waveform with the same length as `acf`.
    """

    # Compute start and end time in physical units
    t_start = t_start_g * C_mt * M
    t_end   = t_end_g * C_mt * M

    # Generate time arrays
    t_array = np.linspace(t_start, t_end, len(wf_lNR))  # Original waveform time
    t_int   = np.linspace(t_start, t_end, len(acf))       # Target interpolation time

    # Use Numba-optimized interpolation
    wf_int = fast_interpolation(t_int, t_array, wf_lNR)

    return wf_int

def convert_asd_to_pycbc_psd(asd_file, delta_f):

    """
    Load an ASD file, compute the PSD, and convert it to a PyCBC FrequencySeries.

    Parameters
    ----------

    asd_file (str): Path to the ASD file (two columns: frequency, ASD value)

    Returns
    -------

    pycbc.types.FrequencySeries: The computed PSD as a FrequencySeries object.
    """
    
    # Load ASD data from file
    data       = np.loadtxt(asd_file)
    asd_values = data[:, 1]   # Second column: ASD values
    
    # Compute PSD by squaring ASD values
    psd_values = asd_values ** 2
    
    print(f"Loaded ASD file: {asd_file}, PSD length: {len(psd_values)}")

    # Convert to PyCBC FrequencySeries
    psd = FrequencySeries(psd_values, delta_f=delta_f)
    
    return psd

def clear_directory(directory_path):

    """
    Clears all files inside a directory without deleting the directory itself.

    Parameters
    ----------
        directory_path (str): Path to the directory to be cleared.

    Returns
    -------
        None
    """

    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path): os.unlink(file_path)     # Delete files and symlinks
                elif os.path.isdir(file_path)                            : shutil.rmtree(file_path) # Recursively delete folders
            except Exception as e: print(f"Failed to delete {file_path}: {e}")

    else:
        os.makedirs(directory_path, exist_ok=True)

def truncate_and_interpolate_acf(t_ACF, ACF_smoothed, M, t_start_g, t_end_g, t_NR_s, print_truncation_info):

    """
    Truncate and interpolate the Autocorrelation Function (ACF) based on time constraints.

    Parameters
    ----------
        ACF_smoothed (np.ndarray): the original smoothed ACF.
        t_ACF (np.ndarray): the time axis associated to ACF_smoothed.
        t_start (float): Start time for analysis [geometric units].
        t_end (float): End time for analysis [geometric units].
        t_NR_s (np.ndarray): NR time array in seconds, starting at 0, and ending at t_end-t_start.
        N_sim (int): The number of points for the interpolated ACF.

    Returns
    -------
        np.ndarray: The new array corresponding to the interpolated ACF on the NR time array.
    """

    # First, we take only the first half of the ACF, which is the one associated to positive frequencies
    half_index        = len(ACF_smoothed) // 2
    t_ACF_half        = t_ACF[:half_index]
    ACF_smoothed_half = ACF_smoothed[:half_index]

    # Compute the truncation point (T_RD = t_end - t_start)
    T_RD  = (t_end_g - t_start_g) * C_mt * M
    index = np.argmin(np.abs(t_ACF_half - T_RD))

    # Truncate the ACF to ringdown analysis (See https://arxiv.org/abs/2107.05609 for discussion on truncation)
    ACF_truncated   = ACF_smoothed_half[:index+1]
    t_ACF_truncated = t_ACF_half[:index+1]

    # Perform linear interpolation
    interpolator = interp1d(t_ACF_truncated, ACF_truncated, kind='linear', fill_value="extrapolate")
    ACF_trunc    = interpolator(t_NR_s)

    if print_truncation_info:

        print("Truncation info:")
        print("ACF time array expr. in [s] (full): ", t_ACF)
        print("ACF time array expr. in [s] (first half, associated to positive frequencies): ", t_ACF_half)
        print("Truncated ACF time array expr. in [s] : ", t_ACF_truncated)
        print("Truncated waveform time array expr. in geometrical units : ", t_NR_s/(M*C_mt))
    
    return ACF_trunc

def mismatch_sanity_checks(NR_sim, results, inference_model, outdir, method, acf, M, dL, t_start_g, t_end_g, window_size_DX, window_size_SX, k):

    """
    Performs sanity checks for mismatch computation.

    Parameters
    ----------

    NR_sim : NR_sim
        NR simulation object.

    results : dict
        Dictionary containing the results object.

    inference_model : inference_model
        Nested sampling model object.

    outdir : string
        Output directory.

    method : string
        Method used to fit the waveform.

    acf : array
        Autocovariance function of the noise (expressed in seconds).

    M : float
        Mass of the remnant (expressed in solar masses).

    dL: float
        Luminosity distance of the source with respect to the observer (expressed in Megaparsec).

    Returns
    -------
    
    Nothing, only creates sanity plots.
    """

    # outdir
    sanity_checks_dir = os.path.join(outdir, 'Algorithm', 'Sanity_Checks')

    # create folder
    os.makedirs(sanity_checks_dir, exist_ok=True)

    #start and end times of the analysis [s]
    t_start = t_start_g * C_mt * M
    t_end   = t_end_g * C_mt * M
    t_trunc = np.linspace(t_start, t_end, len(NR_sim.t_NR_cut))

    # Calculate scaled NR waveform components
    NR_r = NR_sim.NR_r_cut * (C_md * M) / dL
    NR_i = NR_sim.NR_i_cut * (C_md * M) / dL

    # Initialize lists to store waveform components
    models_re_list = []
    models_im_list = []

    if method == 'Nested-sampler':
        models_re_list = [np.real(np.array(inference_model.model(p))) for p in results]
        models_im_list = [np.imag(np.array(inference_model.model(p))) for p in results]

    wf_r_quantiles = {}
    wf_i_quantiles = {}

    for perc in [5, 50, 95]:
        wf_r_quantiles[perc] = np.percentile(np.array(models_re_list), [perc], axis=0)[0] * (C_md * M) / dL
        wf_i_quantiles[perc] = np.percentile(np.array(models_im_list), [perc], axis=0)[0] * (C_md * M) / dL

    # Compute whitened NR components
    whiten_NR_r = sl.solve_toeplitz(acf, NR_r, check_finite=False)
    whiten_NR_i = sl.solve_toeplitz(acf, NR_i, check_finite=False)

    # Compute whitened waveform quantiles
    wf_r_whitened = {perc: sl.solve_toeplitz(acf, wf_r_quantiles[perc], check_finite=False) for perc in [5, 50, 95]}
    wf_i_whitened = {perc: sl.solve_toeplitz(acf, wf_i_quantiles[perc], check_finite=False) for perc in [5, 50, 95]}

    # Create Toeplitz matrix from acf and compute its inverse
    acf_toeplitz     = sl.toeplitz(acf)
    acf_toeplitz_inv = np.linalg.inv(acf_toeplitz)

    # Apply whitening using the Toeplitz inverse matrix
    toeplitz_whitened_NR_r = np.dot(acf_toeplitz_inv, NR_r)
    toeplitz_whitened_NR_i = np.dot(acf_toeplitz_inv, NR_i)
    wf_r_toeplitz_whitened = {perc: np.dot(acf_toeplitz_inv, wf_r_quantiles[perc]) for perc in [5, 50, 95]}
    wf_i_toeplitz_whitened = {perc: np.dot(acf_toeplitz_inv, wf_i_quantiles[perc]) for perc in [5, 50, 95]}

    # Generate plot for real components (No Whitening)
    plt.figure(figsize=(10, 6))
    plt.plot(t_trunc, NR_r, label='NR_r', color='blue', linewidth=1.5)
    plt.plot(t_trunc, wf_r_quantiles[50], label='50% CI', linestyle='-', color=colbRed)
    plt.title('Real Component Comparison (No Whitening)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    filename = f"Real_Component_No_Whitening_M_{M}_dL_{dL}_M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}.pdf"
    plt.savefig(os.path.join(sanity_checks_dir, filename))
    plt.close()

    # Generate plot for imaginary components (No Whitening)
    plt.figure(figsize=(10, 6))
    plt.plot(t_trunc, NR_i, label='NR_i', color=colbBlue, linewidth=1.5)
    plt.plot(t_trunc, wf_i_quantiles[50], label='50% CI', linestyle='-', color=colbRed)
    plt.title('Imaginary Component Comparison (No Whitening)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    #plt.xlim(1.1285,1.13)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    filename = f"Imaginary_Component_No_Whitening_M_{M}_dL_{dL}_M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}.pdf"
    plt.savefig(os.path.join(sanity_checks_dir, filename))
    plt.close()

    # Generate plot for real components (Whitening with solve_toeplitz)
    plt.figure(figsize=(10, 6))
    plt.plot(t_trunc, whiten_NR_r, label='NR_r (whitened)', color=colbBlue, linewidth=1.5)
    plt.plot(t_trunc, wf_r_whitened[50], label='50% CI (whitened)', linestyle='-', color=colbRed)
    plt.title('Real Component Comparison (Whitened with solve_toeplitz)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude (Whitened)')
    #plt.xlim(1.1285,1.13)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    filename = f"Real_Component_Whitened_M_{M}_dL_{dL}_M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}.pdf"
    plt.savefig(os.path.join(sanity_checks_dir, filename))
    plt.close()

    # Generate plot for imaginary components (Whitening with solve_toeplitz)
    plt.figure(figsize=(10, 6))
    plt.plot(t_trunc, whiten_NR_i, label='NR_i (whitened)', color=colbBlue, linewidth=1.5)
    plt.plot(t_trunc, wf_i_whitened[50], label='50% CI (whitened)', linestyle='-', color=colbRed)
    plt.title('Imaginary Component Comparison (Whitened with solve_toeplitz)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude (Whitened)')
    #plt.xlim(1.1285,1.13)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    filename = f"Imaginary_Component_Whitened_M_{M}_dL_{dL}_M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}.pdf"
    plt.savefig(os.path.join(sanity_checks_dir, filename))
    plt.close()

    # Generate plot for real components (Toeplitz Whitening)
    plt.figure(figsize=(10, 6))
    plt.plot(t_trunc, toeplitz_whitened_NR_r, label='NR_r (Toeplitz whitened)', color=colbBlue, linewidth=1.5)
    plt.plot(t_trunc, wf_r_toeplitz_whitened[50], label='50% CI (Toeplitz whitened)', linestyle='-', color=colbRed)
    plt.title('Real Component Comparison (Toeplitz Whitening)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude (Toeplitz Whitened)')
    #plt.xlim(1.1285,1.13)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    filename = f"Real_Component_Toeplitz_Whitening_M_{M}_dL_{dL}_M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}.pdf"
    plt.savefig(os.path.join(sanity_checks_dir, filename))
    plt.close()

    # Generate plot for imaginary components (Toeplitz Whitening)
    plt.figure(figsize=(10, 6))
    plt.plot(t_trunc, toeplitz_whitened_NR_i, label='NR_i (Toeplitz whitened)', color=colbBlue, linewidth=1.5)
    plt.plot(t_trunc, wf_i_toeplitz_whitened[50], label='50% CI (Toeplitz whitened)', linestyle='-', color=colbRed)
    plt.title('Imaginary Component Comparison (Toeplitz Whitening)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude (Toeplitz Whitened)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    filename = f"Imaginary_Component_Toeplitz_Whitening_M_{M}_dL_{dL}_M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}.pdf"
    plt.savefig(os.path.join(sanity_checks_dir, filename))
    plt.close()

    print("Plots saved to:", os.path.join(outdir, 'Algorithm'))

    return

def compute_mismatch_check_TD_FD(NR_sim, results, inference_model, outdir, method, acf, N_FFT, M, dL, t_start_g, t_end_g, f_min, f_max, asd_file, window_size, k, compare_TD_FD, sanity_check_mm):
   
    """
    OLD VERSION. Compute the mismatch of the model with respect to NR simulations.
    """

    # File paths for saving results
    mismatch_filename = f"Mismatch_M_{M}_dL_{dL}_t_s_{round(t_start_g,1)}M_w_{round(window_size,1)}_k_{round(k,2)}_NFFT_{N_FFT}.txt"
    mismatch_filename_fd = f"Mismatch_M_{M}_dL_{dL}_t_s_{round(t_start_g,1)}M_w_{round(window_size,1)}_k_{round(k,2)}_NFFT_{N_FFT}_FD.txt"
    outFile_path = os.path.join(outdir, 'Algorithm', mismatch_filename)
    outFile_path_fd = os.path.join(outdir, 'Algorithm', mismatch_filename_fd)

    with open(outFile_path, 'w') as outFile_mismatch, open(outFile_path_fd, 'w') as outFile_mismatch_fd:
        outFile_mismatch.write('#CI\tStrain_data\tMismatch\n')
        outFile_mismatch_fd.write('#CI\tStrain_data\tFD_Mismatch\n')

    # Extract NR waveform components
    NR_r = NR_sim.NR_r_cut * (C_md * M) / dL
    NR_i = NR_sim.NR_i_cut * (C_md * M) / dL
    NR_dict = {'real': NR_r, 'imaginary': NR_i}

    for NR_quant, NR_data in NR_dict.items():
        try:
            NR_int = interpolate_waveform(t_start_g, t_end_g, M, wf_lNR=NR_data, acf=acf)
            whiten_whiten_h_NR = sl.solve_toeplitz(acf, NR_int, check_finite=False)
            h_NR_h_NR_sqrt = np.sqrt(abs(np.dot(NR_int, whiten_whiten_h_NR)))

        except Exception as e:
            print(f"Error in NR scalar product for {NR_quant}: {e}")
            continue
        
        # Load waveform template
        if method == 'Nested-sampler':
            models_re_list = [np.real(np.array(inference_model.model(p))) for p in results]
            models_im_list = [np.imag(np.array(inference_model.model(p))) for p in results]

        for perc in [5, 50, 95]:
            try:
                wf_r = np.percentile(np.array(models_re_list), [perc], axis=0)[0]
                wf_i = np.percentile(np.array(models_im_list), [perc], axis=0)[0]

                wf_r *= (C_md * M) / dL
                wf_i *= (C_md * M) / dL
                wf_quant = {'real': wf_r, 'imaginary': wf_i}

                wf_int = interpolate_waveform(t_start_g, t_end_g, M, wf_lNR=wf_quant[NR_quant], acf=acf)
                whiten_whiten_h_wf = sl.solve_toeplitz(acf, wf_int, check_finite=False)
                h_wf_h_wf_sqrt = np.sqrt(abs(np.dot(wf_int, whiten_whiten_h_wf)))
                h_wf_h_NR = np.dot(wf_int, whiten_whiten_h_NR)

                TD_match = h_wf_h_NR / (h_NR_h_NR_sqrt * h_wf_h_wf_sqrt)
                TD_mismatch = 1 - TD_match

                with open(outFile_path, 'a') as outFile_mismatch:
                    outFile_mismatch.write(f'{perc}\t{NR_quant}\t{TD_mismatch}\n')

                if compare_TD_FD:
                    psd = convert_asd_to_pycbc_psd(asd_file, f_min, f_max, delta_f=2*f_max/len(acf))
                    h_TS = TimeSeries(wf_int, delta_t=1/(2*f_max))
                    NR_TS = TimeSeries(NR_int, delta_t=1/(2*f_max))

                    FD_match_m = float(compute_FD_match(h_TS, NR_TS, psd=psd, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)[0])
                    FD_mismatch = 1 - FD_match_m

                    with open(outFile_path_fd, 'a') as outFile_mismatch_fd:
                        outFile_mismatch_fd.write(f'{perc}\t{NR_quant}\t{FD_mismatch}\n')

            except Exception as e:
                print(f"Error processing mismatch for {perc}% CI and {NR_quant}: {e}")
                continue

    return

def compute_mismatch_hplus_hcross(NR_sim, results, inference_model, outdir, method, acf, N_FFT, M, dL, t_start_g, f_min, f_max, asd_file, window_size_DX, window_size_SX, k, saturation_DX, saturation_SX, mismatch_print_flag, compare_TD_FD):
   
    """
    Compute the mismatch of the model with respect to NR simulations.
    """

    print("\nProcessing mismatch computation for plus and cross polarizations.\n")

    # File paths for saving results
    mismatch_filename = f"Mismatch_M_{M}_dL_{dL}_t_s_{round(t_start_g,1)}M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}_satDX_{round(saturation_DX,1)}_satSD_{round(saturation_SX,1)}_NFFT_{N_FFT}.txt"
    outFile_path      = os.path.join(outdir, 'Algorithm', mismatch_filename)
    
    with open(outFile_path, 'w') as outFile_mismatch:
        outFile_mismatch.write('#CI\tStrain_data\tMismatch\n')

    # Extract NR waveform components (physical units)
    NR_r    = NR_sim.NR_r_cut * (C_md * M) / dL
    NR_i    = NR_sim.NR_i_cut * (C_md * M) / dL
    NR_dict = {'real': NR_r, 'imaginary': NR_i}

    for NR_quant, NR_data in NR_dict.items():
        try:

            # Compute <NR|NR>
            whiten_whiten_h_NR = sl.solve_toeplitz(acf, NR_data, check_finite=False)
            h_NR_h_NR_sqrt     = np.sqrt(abs(np.dot(NR_data, whiten_whiten_h_NR)))

            if mismatch_print_flag: print(f"<NR|NR>**0.5={h_NR_h_NR_sqrt:.1f}")

        except Exception as e:
            print(f"Error in NR scalar product for {NR_quant}: {e}")
            continue
        
        # Load waveform template
        if method == 'Nested-sampler':
            models_re_list = [np.real(np.array(inference_model.model(p))) for p in results]
            models_im_list = [np.imag(np.array(inference_model.model(p))) for p in results]

        for perc in [5, 50, 95]:
            try:

                # Extract waveform (geometric units)
                wf_r = np.percentile(np.array(models_re_list), [perc], axis=0)[0]
                wf_i = np.percentile(np.array(models_im_list), [perc], axis=0)[0]

                # Convert to physical units
                wf_r    *= (C_md * M) / dL
                wf_i    *= (C_md * M) / dL
                wf_quant = {'real': wf_r, 'imaginary': wf_i}

                # Compute scalar products with h_wf
                whiten_whiten_h_wf = sl.solve_toeplitz(acf, wf_quant[NR_quant], check_finite=False)
                h_wf_h_wf_sqrt     = np.sqrt(abs(np.dot(wf_quant[NR_quant], whiten_whiten_h_wf)))
                h_wf_h_NR          = np.dot(wf_quant[NR_quant], whiten_whiten_h_NR)

                # Match and mismatch computations
                TD_match    = h_wf_h_NR / (h_NR_h_NR_sqrt * h_wf_h_wf_sqrt)
                TD_mismatch = 1 - TD_match

                if mismatch_print_flag==1:
                    print(f"<h|h>**0.5={h_wf_h_wf_sqrt:.1f}")
                    print(f"<h|NR>={h_wf_h_NR:.1f}")
                    print(f"Time domain mismatch={TD_mismatch}")

                with open(outFile_path, 'a') as outFile_mismatch:
                    outFile_mismatch.write(f'{perc}\t{NR_quant}\t{TD_mismatch}\n')

                if compare_TD_FD:

                    mismatch_filename_fd = f"Mismatch_M_{M}_dL_{dL}_t_s_{round(t_start_g,1)}M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}_NFFT_{N_FFT}.txt"
                    outFile_path_fd      = os.path.join(outdir, 'Algorithm', mismatch_filename_fd)

                    with open(outFile_path_fd, 'w') as outFile_mismatch_fd:
                        outFile_mismatch_fd.write('#CI\tStrain_data\tFD_Mismatch\n')

                    psd   = convert_asd_to_pycbc_psd(asd_file, f_min, f_max, delta_f=2*f_max/len(acf))
                    h_TS  = TimeSeries(wf_quant[NR_quant], delta_t=1/(2*f_max))
                    NR_TS = TimeSeries(NR_data, delta_t=1/(2*f_max))

                    FD_match_m = float(compute_FD_match(h_TS, NR_TS, psd=psd, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)[0])
                    FD_mismatch = 1 - FD_match_m

                    if mismatch_print_flag==1:
                        print(f"Time domain mismatch={TD_mismatch}")
                        print(f"Frequency domain mismatch={TD_mismatch}")

                    with open(outFile_path_fd, 'a') as outFile_mismatch_fd:
                        outFile_mismatch_fd.write(f'{perc}\t{NR_quant}\t{FD_mismatch}\n')

            except Exception as e:
                print(f"Error processing mismatch for {perc}% CI and {NR_quant}: {e}")
                continue

    return

def compute_mismatch_htot(NR_sim, results, inference_model, outdir, method, acf, N_FFT, M, dL, ra, dec, psi, t_start_g, window_size_DX, window_size_SX, k, saturation_DX, saturation_SX):
    
    """
    Compute the mismatch of the model with respect to NR simulations.
    
    """
    print("\nProcessing mismatch computation for the total signal.\n")
    # File paths for saving results
    mismatch_filename = f"Mismatch_h_tot_M_{M}_dL_{dL}_t_s_{round(t_start_g,1)}M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}_satDX_{round(saturation_DX,1)}_satSD_{round(saturation_SX,1)}_NFFT_{N_FFT}.txt"
    outFile_path      = os.path.join(outdir, 'Algorithm', mismatch_filename)
    
    with open(outFile_path, 'w') as outFile_mismatch:
        outFile_mismatch.write('#CI\tStrain_data\tMismatch\n')

    # Extract NR waveform components (physical units)
    NR_r = NR_sim.NR_r_cut * (C_md * M) / dL
    NR_i = NR_sim.NR_i_cut * (C_md * M) / dL

    # Compute polarizations
    resp            = AntennaResponse('H1', ra=ra, dec=dec, psi=psi, tensor=True, times=1126259462.43)
    F_plus, F_cross = resp.plus, resp.cross
    NR_data         = F_plus * NR_r + F_cross * NR_i

    # Compute <NR|NR>
    whiten_whiten_h_NR = sl.solve_toeplitz(acf, NR_data, check_finite=False)
    h_NR_h_NR_sqrt     = np.sqrt(abs(np.dot(NR_data, whiten_whiten_h_NR)))

    # Load waveform template
    if method == 'Nested-sampler':
        models_re_list = [np.real(np.array(inference_model.model(p))) for p in results]
        models_im_list = [np.imag(np.array(inference_model.model(p))) for p in results]

    for perc in [5, 50, 95]:
            
            # Extract waveform (geometric units)
            wf_r = np.percentile(np.array(models_re_list), [perc], axis=0)[0]
            wf_i = np.percentile(np.array(models_im_list), [perc], axis=0)[0]

            # Convert to physical units
            wf_r *= (C_md * M) / dL
            wf_i *= (C_md * M) / dL
            wf    = F_plus * wf_r + F_cross * wf_i

            # Compute scalar products with h_wf
            whiten_whiten_h_wf = sl.solve_toeplitz(acf, wf, check_finite=False)
            h_wf_h_wf_sqrt     = np.sqrt(abs(np.dot(wf, whiten_whiten_h_wf)))
            h_wf_h_NR          = np.dot(wf, whiten_whiten_h_NR)

            # Match/mismatch computations
            TD_match    = h_wf_h_NR / (h_NR_h_NR_sqrt * h_wf_h_wf_sqrt)
            TD_mismatch = 1 - TD_match

            with open(outFile_path, 'a') as outFile_mismatch:
                outFile_mismatch.write(f'{perc}\t{TD_mismatch}\n')
        
    return 

def compute_optimal_SNR(NR_sim, results, inference_model, outdir, method, acf, N_FFT, M, dL, t_start_g, t_end_g, f_min, f_max, asd_file, window_size_DX, window_size_SX, k, saturation_DX, saturation_SX, compare_TD_FD):
    """
    Compute the optimal SNR of the model waveform.
    """
    print("\nProcessing optimal SNR computation for plus and cross polarizations.\n")

    # File paths for saving results
    optimal_SNR_filename = f"Optimal_SNR_M_{M}_dL_{dL}_t_s_{round(t_start_g,1)}M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}_satDX_{round(saturation_DX,1)}_satSD_{round(saturation_SX,1)}_NFFT_{N_FFT}.txt"
    optimal_SNR_filename_fd = f"Optimal_SNR_M_{M}_dL_{dL}_t_s_{round(t_start_g,1)}M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}_NFFT_{N_FFT}_FD.txt"
    outFile_path = os.path.join(outdir, 'Algorithm', optimal_SNR_filename)
    outFile_path_fd = os.path.join(outdir, 'Algorithm', optimal_SNR_filename_fd)

    with open(outFile_path, 'w') as outFile_SNR, open(outFile_path_fd, 'w') as outFile_SNR_fd:
        outFile_SNR.write('#CI\tStrain_data\tOptimal_SNR\n')
        outFile_SNR_fd.write('#CI\tStrain_data\tOptimal_SNR_FD\n')

    NR_r = NR_sim.NR_r_cut * (C_md * M) / dL
    NR_i = NR_sim.NR_i_cut * (C_md * M) / dL
    NR_dict = {'real': NR_r, 'imaginary': NR_i}

    for NR_quant, NR_data in NR_dict.items():

        for perc in [5, 50, 95]:
            try:
                wf_r = np.percentile([np.real(np.array(inference_model.model(p))) for p in results], [perc], axis=0)[0]
                wf_i = np.percentile([np.imag(np.array(inference_model.model(p))) for p in results], [perc], axis=0)[0]

                wf_r *= (C_md * M) / dL
                wf_i *= (C_md * M) / dL
                wf_int = interpolate_waveform(t_start_g, t_end_g, M, wf_lNR=wf_r if NR_quant == "real" else wf_i, acf=acf)

                optimal_SNR_TD = np.sqrt(abs(np.dot(wf_int, sl.solve_toeplitz(acf, wf_int, check_finite=False))))

                with open(outFile_path, 'a') as outFile_SNR:
                    outFile_SNR.write(f'{perc}\t{NR_quant}\t{optimal_SNR_TD}\n')

                if compare_TD_FD:
                    h_TS = TimeSeries(wf_int, delta_t=1/(2*f_max))
                    optimal_SNR_FD = compute_FD_optimal_SNR(asd_file, h_TS, len(acf), f_min, f_max)

                    print("Optimal TD SNR: ", optimal_SNR_TD)
                    print("Optimal FD SNR: ", optimal_SNR_FD)

                    with open(outFile_path_fd, 'a') as outFile_SNR_fd:
                        outFile_SNR_fd.write(f'{perc}\t{NR_quant}\t{optimal_SNR_FD}\n')

            except Exception as e:
                print(f"Error processing optimal SNR for {perc}% CI and {NR_quant}: {e}")
                continue

def plot_NR_vs_model(NR_sim, template, metadata, results, inference_model, outdir, method, tail_flag):

    """

    Plot the NR waveform against the model waveform.

    Parameters
    ----------

    NR_sim : NR_sim
        NR simulation object.

    template : template
        Template object.

    metadata : dict
        Dictionary containing the metadata.

    results : dict
        Dictionary containing the results object.

    inference_model : inference_model
        Nested sampling model object.

    outdir : string
        Output directory.

    method : string
        Method used to fit the waveform.

    Returns
    -------

    Nothing, but plots the simulation/model comparison and saves the figure.

    """

    init_plotting()
    
    #take NR elements
    NR_r, NR_i, NR_r_err, NR_i_err, NR_amp, NR_f, t_NR, t_peak                                                = NR_sim.NR_r, NR_sim.NR_i, np.real(NR_sim.NR_err_cmplx), np.imag(NR_sim.NR_err_cmplx), NR_sim.NR_amp, NR_sim.NR_freq, NR_sim.t_NR, NR_sim.t_peak
    t_cut, tM_start, tM_end, NR_r_cut, NR_i_cut, NR_r_err_cut, NR_i_err_cut, NR_amp_cut, NR_phi_cut, NR_f_cut = NR_sim.t_NR_cut, NR_sim.tM_start, NR_sim.tM_end, NR_sim.NR_r_cut, NR_sim.NR_i_cut, np.real(NR_sim.NR_cpx_err_cut), np.imag(NR_sim.NR_cpx_err_cut), NR_sim.NR_amp_cut, NR_sim.NR_phi_cut, NR_sim.NR_freq_cut

    wf_data_type = NR_sim.waveform_type

    l,m = NR_sim.l, NR_sim.m

    f_rd_fundamental    = template.qnm_cached[(2,l,m,0)]['f']
    tau_rd_fundamental  = template.qnm_cached[(2,l,m,0)]['tau']

    plot_overtones_flag = 0
    f_rd_overtones      = {}
    for n in [1,3,7,9]: 
        omega_n, _, _     = qnm.modes_cache(s=-2,l=l,m=m,n=n)(a=np.abs(metadata['af']))
        f_rd_overtones[n] = (np.real(omega_n) / metadata['Mf']) * (1./twopi)

    try:
        m1, m2, chi1, chi2 = metadata['m1'], metadata['m2'], metadata['chi1'], metadata['chi2'],
        f_peak             = utils.F_mrg_Nagar(m1, m2, chi1, chi2, geom=1)
    except:
        f_peak             = None

    # get the amplitude at the time close to the peak
    amp_peak = NR_amp[np.argmin(np.abs(t_NR - t_peak))]

    lw_small        = 0.5
    lw_medium       = 1.2
    lw_std          = 1.8
    lw_large        = 2.2

    color_NR        = 'k'
    color_model     = '#cc0033'
    color_t_start   = 'mediumseagreen' #'#990066', '#cc0033', '#ff0000'
    color_t_peak    = 'royalblue'
    color_f_overt   = 'darkorange'

    alpha_std       = 1.0
    alpha_med       = 0.8

    ls_t            = '--'
    ls_f            = '--'

    if(tail_flag) :
        fontsize_legend = 20
        fontsize_labels = 25
        color_f_ring    = 'royalblue'
    else:
        fontsize_legend = 18
        fontsize_labels = 23
        color_f_ring    = 'forestgreen'

    if(not(tail_flag) and not(wf_data_type=='psi4') and (NR_sim.NR_catalog=='SXS' or NR_sim.NR_catalog=='RIT')): tM_end = 80
    if(wf_data_type=='psi4'): 
        tM_end = 120
        label_data = '\psi_{4,%s%s}'%(l,m)
    else:
        label_data = 'h_{%s%s}'%(l,m)

    ########################
    # Waveforms comparison #
    ########################

    if(tail_flag):
        f   = plt.figure(figsize=(8,12))
        ax2 = plt.subplot(2,1,1)
        ax4 = plt.subplot(2,1,2)
        
        rescale = 1.4
    else:
        f   = plt.figure(figsize=(12,8))
        ax1 = plt.subplot(2,2,1)
        ax2 = plt.subplot(2,2,2)
        ax3 = plt.subplot(2,2,3)
        ax4 = plt.subplot(2,2,4)
  
        ax1.set_xlim([-10, tM_end])
        ax3.set_xlim(ax1.get_xlim())

        rescale = 1.0

    ax2.set_xlim(-10, tM_end)
    ax4.set_xlim(ax2.get_xlim())

    # Plot NR data

    if not(tail_flag):
        ax1.plot(t_NR - t_peak, NR_r,                                                      c=color_NR,      lw=lw_std,    alpha=alpha_std, ls='-' )
        ax1.axvline(tM_start,                                                              c=color_t_start, lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax1.axvline(0.0, label=r'$t_{\rm peak}$',                                          c=color_t_peak,  lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax1.set_ylabel(r'$\mathrm{Re[%s]}$'%(label_data), fontsize=fontsize_labels)

        ax3.plot(t_NR - t_peak, NR_i,                                                      c=color_NR,      lw=lw_std,    alpha=alpha_std, ls='-' )
        ax3.axvline(tM_start, label=r'$t_{\rm start} = t_{\rm peak} \, + %d \mathrm{M}$'%tM_start, c=color_t_start, lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax3.axvline(0.0,                                                                   c=color_t_peak,  lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax3.set_ylabel(r'$\mathrm{Im[%s]}$'%(label_data), fontsize=fontsize_labels)
        ax3.set_xlabel(r'$t - t_{peak} \, [\mathrm{M}]$', fontsize=fontsize_labels)

    ax2.semilogy(t_NR - t_peak, NR_amp*np.e**((t_NR- t_peak)/tau_rd_fundamental), label=r'$\mathrm{NR}$',                            c=color_NR,      lw=lw_std,    alpha=alpha_std, ls='-' )
    ax2.axvline(tM_start,                                                                  c=color_t_start, lw=lw_std,    alpha=alpha_std, ls=ls_t)
    if(not(tail_flag)): ax2.axvline(0.0,                                                   c=color_t_peak,  lw=lw_std,    alpha=alpha_std, ls=ls_t)
    if(not(tail_flag) and (NR_sim.NR_catalog=='SXS' or NR_sim.NR_catalog=='RIT')): ax2.set_ylim([1e-1*amp_peak, 10*amp_peak])
    elif(  tail_flag  and (NR_sim.NR_catalog=='SXS' or NR_sim.NR_catalog=='RIT')): ax2.set_ylim([2*1e-4, 2*np.max(NR_amp)])
    ax2.set_xlabel(r'$\mathrm{t - t_{peak} \, [M}]$', fontsize=fontsize_labels)

    ax4.plot(t_NR - t_peak, NR_f,                                                          c=color_NR,      lw=lw_std,     alpha=alpha_std, ls='-' )
    ax4.axhline(f_rd_fundamental, label=r'$\mathit{f_{%d%d0}}$'%(l,m),                     c=color_f_ring,  lw=lw_std,     alpha=alpha_std, ls=ls_f)
    if(plot_overtones_flag):
        for n in [1,3,9]: 
            if(n==1): leg = r'$\mathit{f_{%d%dn}}$'%(l,m)
            else    : leg = None
            ax4.axhline(f_rd_overtones[n], label=leg,         c=color_f_overt, lw=lw_std*0.4, alpha=alpha_std, ls=ls_f)

    if(tail_flag): 
        ax4.axhline(0.0,      label=r'$\mathit{f_{\rm tail}}$',                            c=color_model,   lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax4.axvline(tM_start, label=r'$\mathrm{t_{start} = t_{peak} \, + %d M}$'%tM_start, c=color_t_start, lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax4.axvline(0.0,                                                                   c=color_t_peak,  lw=lw_std,    alpha=alpha_std, ls=ls_t)
    else         : 
        ax4.axvline(0.0,                                                                   c=color_t_peak,  lw=lw_std,    alpha=alpha_std, ls=ls_t)
    ax4.set_xlabel(r'$t - t_{peak} \, [\mathrm{M}]$'    , fontsize=fontsize_labels)

    # Find the index of zero
    t_peak_idx = np.argmin(np.abs(t_NR - t_peak))
    
    if not(tail_flag):
        try   : ax4.set_ylim([-1.5*NR_f[t_peak_idx], 3.5*NR_f[t_peak_idx]])
        except: pass
    else:
        ax4.set_ylim([-0.08, 0.28])

    if not(inference_model==None):

        # Plot waveform reconstruction
        if(method=='Nested-sampler'):
            models_re_list = [np.real(np.array(inference_model.model(p))) for p in results]
            models_im_list = [np.imag(np.array(inference_model.model(p))) for p in results]

        for perc in [50, 5, 95]:

            if(method=='Nested-sampler'):
                wf_r = np.percentile(np.array(models_re_list),[perc], axis=0)[0]
                wf_i = np.percentile(np.array(models_im_list),[perc], axis=0)[0]
            else:
                wf_r = np.real(np.array(inference_model.model(results)))
                wf_i = np.imag(np.array(inference_model.model(results)))

            wf_amp, wf_phi = waveform_utils.amp_phase_from_re_im(wf_r, wf_i)
            wf_f           = np.gradient(wf_phi, t_cut)/(twopi)
            
            if(perc==50):
                if not(tail_flag):
                    ax1.plot(t_cut - t_peak, wf_r,                                               c=color_model, lw=lw_large*rescale, alpha=alpha_std, ls='-' )
                    ax3.plot(t_cut - t_peak, wf_i,                                               c=color_model, lw=lw_large*rescale, alpha=alpha_std, ls='-' )
                ax2.semilogy(t_cut - t_peak, wf_amp, label=r'$\mathrm{%s}$'%(template.wf_model), c=color_model, lw=lw_large*rescale, alpha=alpha_std, ls='-' )
                ax4.plot(    t_cut - t_peak, wf_f,                                               c=color_model, lw=lw_large*rescale, alpha=alpha_std, ls='-' )
            else:
                if not(tail_flag):
                    ax1.plot(t_cut - t_peak, wf_r,                                               c=color_model, lw=lw_std,           alpha=alpha_med, ls='--' )
                    ax3.plot(t_cut - t_peak, wf_i,                                               c=color_model, lw=lw_std,           alpha=alpha_med, ls='--' )
                ax2.semilogy(t_cut - t_peak, wf_amp,                                             c=color_model, lw=lw_std,           alpha=alpha_med, ls='--' )
                ax4.plot(    t_cut - t_peak, wf_f,                                               c=color_model, lw=lw_std,           alpha=alpha_med, ls='--' )


        if(tail_flag):
            # for name_x in results.names:
            #     if ('ln_A_tail' in name_x):
            #         results[name_x] = np.log(1e-32)
            try   : results['ln_A_tail_22'] = np.log(1e-32)
            except: pass
            try   : results['ln_A_tail_32'] = np.log(1e-32)
            except: pass

            # Plot QNM waveform reconstruction
            if(method=='Nested-sampler'):
                models_re_list = [np.real(np.array(inference_model.model(p))) for p in results]
                models_im_list = [np.imag(np.array(inference_model.model(p))) for p in results]
            
            for perc in [50, 5, 95]:

                if(method=='Nested-sampler'):
                    wf_r = np.percentile(np.array(models_re_list),[perc], axis=0)[0]
                    wf_i = np.percentile(np.array(models_im_list),[perc], axis=0)[0]
                else:
                    wf_r = np.real(np.array(inference_model.model(results)))
                    wf_i = np.imag(np.array(inference_model.model(results)))

                wf_amp, wf_phi = waveform_utils.amp_phase_from_re_im(wf_r, wf_i)
                wf_f           = np.gradient(wf_phi, t_cut)/(twopi)
                
                if(perc==50):
                    ax2.semilogy(t_cut - t_peak, wf_amp, label=r'$\mathrm{%s \,\, QNMs}$'%(template.wf_model), c='royalblue', lw=lw_large*1.4, alpha=alpha_std, ls='-' )
                    ax4.plot(    t_cut - t_peak, wf_f,                                                         c='royalblue', lw=lw_large*1.4, alpha=alpha_std, ls='-' )
                else:
                    ax2.semilogy(t_cut - t_peak, wf_amp,                                                       c='royalblue', lw=lw_std,       alpha=alpha_med, ls='--' )
                    ax4.plot(    t_cut - t_peak, wf_f,                                                         c='royalblue', lw=lw_std,       alpha=alpha_med, ls='--' )

                if(method=='Minimization'): break

    if not(tail_flag):
        ax1.set_ylabel(r'$\mathit{Re(%s)}$'%(label_data)                           , fontsize=fontsize_labels*rescale)
        ax3.set_ylabel(r'$\mathit{Im(%s)}$'%(label_data)                           , fontsize=fontsize_labels*rescale)
    ax2.set_ylabel(    r'$\mathit{A_{%d%d}(t)} \cdot e^{t/\tau_{%d%d0}}$'%(l,m,l,m), fontsize=fontsize_labels*rescale)
    ax4.set_ylabel(    r'$\mathit{f_{%d%d}\,(t)}$'%(l,m)                           , fontsize=fontsize_labels*rescale)

    plt.rcParams['legend.frameon'] = True

    ax2.legend(    loc='best', fontsize=fontsize_legend, shadow=True)
    ax4.legend(    loc='best', fontsize=fontsize_legend, shadow=True)

    if not(tail_flag): 
        ax1.legend(loc='best', fontsize=fontsize_legend, shadow=True)
        ax3.legend(loc='best', fontsize=fontsize_legend, shadow=True)
        ax1.set_xlim(ax3.get_xlim())
        ax1.set_xticklabels([])
        plt.suptitle('{}-{}'.format(NR_sim.NR_catalog, NR_sim.NR_ID), size=28)

    ax2.set_xlim(ax4.get_xlim())
    ax2.set_xticklabels([])
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.subplots_adjust(hspace=0, wspace=0.27)
    if(tail_flag): leg_name_tail = '_tail'
    else         : leg_name_tail = ''
    plt.savefig(os.path.join(outdir, f'Plots/Comparisons/Waveform_reconstruction{leg_name_tail}.pdf'), bbox_inches='tight')

    if (tail_flag): plt.rcParams['legend.frameon'] = False

    if (inference_model==None): return

    ############################
    # Residuals reconstruction #
    ############################

    f   = plt.figure(figsize=(12,8))
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)
    ax3 = plt.subplot(2,2,3)
    ax4 = plt.subplot(2,2,4)

    ax1.set_xlim([tM_start, tM_end])
    ax2.set_xlim(ax1.get_xlim())
    ax3.set_xlim(ax1.get_xlim())
    ax4.set_xlim(ax1.get_xlim())

    ax1.errorbar(t_cut - t_peak, np.zeros(len(NR_r_cut)), yerr=np.array(NR_r_err_cut), label=r'$\mathrm{NR error}$', c=color_NR, lw=lw_small, alpha=alpha_std, ls='-', capsize=0.15)
    ax3.errorbar(t_cut - t_peak, np.zeros(len(NR_i_cut)), yerr=np.array(NR_i_err_cut),                               c=color_NR, lw=lw_small, alpha=alpha_std, ls='-', capsize=0.15)

    for perc in [50, 5, 95]:
    
        if(method=='Nested-sampler'):
            wf_r = np.percentile(np.array(models_re_list),[perc], axis=0)[0]
            wf_i = np.percentile(np.array(models_im_list),[perc], axis=0)[0]
        else:
            wf_r = np.real(np.array(inference_model.model(results)))
            wf_i = np.imag(np.array(inference_model.model(results)))

        if(perc==50):
            ax1.plot(t_cut - t_peak, wf_r   - NR_r_cut  ,                                                  c=color_model, lw=lw_large, alpha=alpha_std, ls='-' )
            ax2.plot(t_cut - t_peak, wf_amp - NR_amp_cut,                                                  c=color_model, lw=lw_large, alpha=alpha_std, ls='-' )
            ax3.plot(t_cut - t_peak, wf_i   - NR_i_cut  , label=r'$\mathrm{%s - NR}$'%(template.wf_model), c=color_model, lw=lw_large, alpha=alpha_std, ls='-' )
            ax4.plot(t_cut - t_peak, wf_f   - NR_f_cut  ,                                                  c=color_model, lw=lw_large, alpha=alpha_std, ls='-' )
        else:
            ax1.plot(t_cut - t_peak, wf_r   - NR_r_cut  ,                                                  c=color_model, lw=lw_std, alpha=alpha_med, ls='--')
            ax2.plot(t_cut - t_peak, wf_amp - NR_amp_cut,                                                  c=color_model, lw=lw_std, alpha=alpha_med, ls='--')
            ax3.plot(t_cut - t_peak, wf_i   - NR_i_cut  ,                                                  c=color_model, lw=lw_std, alpha=alpha_med, ls='--')
            ax4.plot(t_cut - t_peak, wf_f   - NR_f_cut  ,                                                  c=color_model, lw=lw_std, alpha=alpha_med, ls='--')
            
        if(method=='Minimization'): break

    ax1.legend(loc='best', fontsize=fontsize_legend)
    ax3.legend(loc='best', fontsize=fontsize_legend)

    ax1.set_ylabel(r'$\mathit{Re(%s)}$'%(label_data), fontsize=fontsize_labels)
    ax2.set_ylabel(r'$\mathit{A(t)}$'               , fontsize=fontsize_labels)
    ax3.set_ylabel(r'$\mathit{Im(%s)}$'%(label_data), fontsize=fontsize_labels)
    ax4.set_ylabel(r'$\mathit{f\,(t)}$'             , fontsize=fontsize_labels)

    ax3.set_xlabel(r'$t - t_{peak} \, [\mathrm{M}]$', fontsize=fontsize_labels)
    ax4.set_xlabel(r'$t - t_{peak} \, [\mathrm{M}]$', fontsize=fontsize_labels)

    ax3.set_xlim(ax1.get_xlim())  
    ax4.set_xlim(ax2.get_xlim())
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    plt.suptitle('{}-{} residuals'.format(NR_sim.NR_catalog, NR_sim.NR_ID), size=28)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.subplots_adjust(hspace=0, wspace=0.3)

    plt.savefig(os.path.join(outdir, 'Plots/Comparisons/Residuals_reconstruction.pdf'), bbox_inches='tight')

    # Decay rate
    if(tail_flag):

        plt.figure(figsize=(6,6))

        log_t_NR         = np.log(t_NR  - t_peak)
        log_t_cut        = np.log(t_cut - t_peak)

        log_A_NR         = np.log(NR_amp)
        dlog_A_NR_dlog_t = utils.diff1(log_t_NR, log_A_NR)

        models_re_list = [np.real(np.array(inference_model.model(p))) for p in results]
        models_im_list = [np.imag(np.array(inference_model.model(p))) for p in results]
        
        for perc in [50, 5, 95]:
            wf_r = np.percentile(np.array(models_re_list),[perc], axis=0)[0]
            wf_i = np.percentile(np.array(models_im_list),[perc], axis=0)[0]

            wf_amp, _ = waveform_utils.amp_phase_from_re_im(wf_r, wf_i)

            log_A_wf         = np.log(wf_amp)
            dlog_A_wf_dlog_t = utils.diff1(log_t_cut, log_A_wf)

            plt.plot(t_cut - t_peak, dlog_A_wf_dlog_t, c=color_model, lw=lw_std, alpha=alpha_med, ls='-')

        plt.axhline(0.0, c='k', ls='--', lw=0.7)
        plt.axhline(-1.0, c='mediumseagreen', ls='--',  label='Okuzumi+',  lw=1.2)
        plt.axhline(-1.3, c='crimson',        ls='--', label='Albanesi+', lw=1.7)
        plt.plot(    t_NR  - t_peak, dlog_A_NR_dlog_t, c=color_NR,    lw=lw_std, alpha=alpha_med, ls='-')
        plt.xlim([75, 100])
        plt.ylim([-3.4, 1.5])
        plt.xlabel(r'$t - t_{peak} \, [\mathrm{M}]$',                   fontsize=fontsize_labels)
        plt.ylabel(r'$\mathrm{p}$', fontsize=fontsize_labels)
        plt.legend(loc='best', fontsize=fontsize_labels*0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'Plots/Comparisons/Decay_rate.pdf'), bbox_inches='tight')

    return

def plot_fancy_residual(NR_sim, template, metadata, results, inference_model, outdir, method):

    """

    Plot the residuals vs the NR error in a single figure.

    Parameters
    ----------

    NR_sim : NR_sim
        NR simulation object.

    template : template
        Template object.

    metadata : dict
        Dictionary containing the metadata.

    results : dict
        Dictionary containing the results object.

    inference_model : inference_model
        Nested sampling model object.

    outdir : string
        Output directory.

    method : string
        Method used to fit the waveform.

    Returns
    -------

    Nothing.

    """
    plt.rcParams["mathtext.fontset"]  = "stix"
    plt.rcParams["font.family"]       = "STIXGeneral"
    plt.rcParams["font.size"]         = 14
    plt.rcParams["legend.fontsize"]   = 12
    plt.rcParams["xtick.labelsize"]   = 10
    plt.rcParams["ytick.labelsize"]   = 10
    plt.rcParams["xtick.major.size"]  = 3
    plt.rcParams["xtick.minor.size"]  = 3
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["xtick.minor.width"] = 1
    plt.rcParams["ytick.major.size"]  = 3
    plt.rcParams["ytick.minor.size"]  = 3
    plt.rcParams["ytick.major.width"] = 1
    plt.rcParams["ytick.minor.width"] = 1

    t_peak =  NR_sim.t_peak
    t_cut, tM_start, tM_end, NR_r_cut, NR_i_cut, NR_r_err_cut, NR_i_err_cut = NR_sim.t_NR_cut, NR_sim.tM_start, NR_sim.tM_end, NR_sim.NR_r_cut, NR_sim.NR_i_cut, np.real(NR_sim.NR_cpx_err_cut), np.imag(NR_sim.NR_cpx_err_cut)

    l,m = NR_sim.l, NR_sim.m

    lw_small = 0.1
    lw_std   = 1.0
    lw_large = 1.5

    if not(inference_model==None):

        # Plot waveform reconstruction
        if(method=='Nested-sampler'):
            models_re_list = [np.real(np.array(inference_model.model(p))) for p in results]
            models_im_list = [np.imag(np.array(inference_model.model(p))) for p in results]

    ###########################
    # Waveform reconstruction #
    ###########################

    f, [ax1, ax2]   = plt.subplots(nrows = 2, ncols = 1, figsize=(4,7))

    ax1.set_xlim([tM_start, tM_end])
    ax2.set_xlim(ax1.get_xlim())

    ax1.fill_between(t_cut - t_peak, -np.array(NR_r_err_cut), np.array(NR_r_err_cut), color = 'dimgray', alpha = 0.2, label = r'$\mathrm{Numerical Error} $')
    ax2.fill_between(t_cut - t_peak, -np.array(NR_i_err_cut), np.array(NR_i_err_cut), color = 'dimgray', alpha = 0.2, label = r'$\mathrm{Numerical Error} $')

    wf_r_m = np.percentile(np.array(models_re_list), [5], axis=0)[0]
    wf_r   = np.percentile(np.array(models_re_list),[50], axis=0)[0]
    wf_r_P = np.percentile(np.array(models_re_list),[95], axis=0)[0]

    wf_i_m = np.percentile(np.array(models_im_list), [5], axis=0)[0]
    wf_i   = np.percentile(np.array(models_im_list),[50], axis=0)[0]
    wf_i_P = np.percentile(np.array(models_im_list),[95], axis=0)[0]

    ax1.plot(t_cut - t_peak, wf_r - NR_r_cut, label=r'$\mathrm{Residual}$', c='firebrick', lw=lw_std)
    ax2.plot(t_cut - t_peak, wf_i - NR_i_cut, c='firebrick', lw=lw_std)

    ax1.fill_between(t_cut - t_peak, wf_r_m - NR_r_cut, wf_r_P - NR_r_cut, color = 'firebrick', alpha = 0.2)
    ax2.fill_between(t_cut - t_peak, wf_i_m - NR_i_cut, wf_i_P - NR_i_cut, color = 'firebrick', alpha = 0.2)

    ax1.legend(loc='best')
    if(NR_sim.catalog == 'Teukolsky'):
        ax1.set_ylabel(r'$\Re(\Psi_{4,%i%i})$'%(l,m))
        ax2.set_ylabel(r'$\Im(\Psi_{4,%i%i})$'%(l,m))
    else:
        ax1.set_ylabel(r'$\Re(h_{%i%i})$'%(l,m))
        ax2.set_ylabel(r'$\Im(h_{%i%i})$'%(l,m))
    ax2.set_xlabel(r'$t/M$')
    for ax in [ax1, ax2]: ax.set_xlim([tM_start, 80])
    plt.suptitle('{}-{} residuals'.format(NR_sim.NR_catalog, NR_sim.NR_ID), size=28)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(os.path.join(outdir, 'Plots/Comparisons/Fancy_Residuals.pdf'), bbox_inches='tight')

    return

def plot_fancy_reconstruction(NR_sim, template, metadata, results, inference_model, outdir, method):

    """

    Plot the NR waveform and its reconstruction

    Parameters
    ----------

    NR_sim : NR_sim
        NR simulation object.

    template : template
        Template object.

    metadata : dict
        Dictionary containing the metadata.

    results : dict
        Dictionary containing the results object.

    inference_model : inference_model
        Nested sampling model object.

    outdir : string
        Output directory.

    method : string
        Method used to fit the waveform.

    Returns
    -------

    Nothing.

    """
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["font.size"] = 14
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["xtick.minor.width"] = 1
    plt.rcParams["ytick.major.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3
    plt.rcParams["ytick.major.width"] = 1
    plt.rcParams["ytick.minor.width"] = 1

    NR_r, NR_i, NR_r_err, NR_i_err, NR_amp, NR_f, t_NR, t_peak                                                = NR_sim.NR_r, NR_sim.NR_i, np.real(NR_sim.NR_err_cmplx), np.imag(NR_sim.NR_err_cmplx), NR_sim.NR_amp, NR_sim.NR_freq, NR_sim.t_NR, NR_sim.t_peak
    t_cut, tM_start, tM_end, NR_r_cut, NR_i_cut, NR_r_err_cut, NR_i_err_cut, NR_amp_cut, NR_phi_cut, NR_f_cut = NR_sim.t_NR_cut, NR_sim.tM_start, NR_sim.tM_end, NR_sim.NR_r_cut, NR_sim.NR_i_cut, np.real(NR_sim.NR_cpx_err_cut), np.imag(NR_sim.NR_cpx_err_cut), NR_sim.NR_amp_cut, NR_sim.NR_phi_cut, NR_sim.NR_freq_cut

    l,m = NR_sim.l, NR_sim.m

    f_rd_fundamental = template.qnm_cached[(2,l,m,0)]['f']

    try:
        m1, m2, chi1, chi2 = metadata['m1'], metadata['m2'], metadata['chi1'], metadata['chi2'],
        f_peak             = utils.F_mrg_Nagar(m1, m2, chi1, chi2) * (G_SI*C_SI**(-3))
    except:
        f_peak             = None

    lw_small = 0.1
    lw_std   = 1.0
    lw_large = 1.2

    if not(inference_model==None):

        # Plot waveform reconstruction
        if(method=='Nested-sampler'):
            models_re_list = [np.real(np.array(inference_model.model(p))) for p in results]
            models_im_list = [np.imag(np.array(inference_model.model(p))) for p in results]

    f, [ax1, ax2]   = plt.subplots(nrows = 2, ncols = 1, figsize=(4,7))

    ax1.set_xlim([-10, tM_end])
    ax2.set_xlim(ax1.get_xlim())

    ax1.plot(t_NR - t_peak, NR_r, color = 'black', label = r'$\mathrm{Numerical Data} $', lw = lw_large)
    ax2.plot(t_NR - t_peak, NR_i, color = 'black', lw = lw_large)
    
    wf_r =  np.percentile(np.array(models_re_list),[50], axis=0)[0]
    wf_i =  np.percentile(np.array(models_im_list),[50], axis=0)[0]

    wf_r_m = np.percentile(np.array(models_re_list),[5], axis=0)[0]
    wf_r_P = np.percentile(np.array(models_re_list),[95], axis=0)[0]

    wf_i_m = np.percentile(np.array(models_im_list),[5], axis=0)[0]
    wf_i_P = np.percentile(np.array(models_im_list),[95], axis=0)[0]

    ax1.plot(t_cut - t_peak, wf_r, label=r'$\mathrm{Reconstruction}$', c='firebrick', lw=lw_std)
    ax2.plot(t_cut - t_peak, wf_i, c='firebrick', lw=lw_std)

    ax1.fill_between(t_cut - t_peak, wf_r_m , wf_r_P, color = 'firebrick', alpha = 0.2)
    ax2.fill_between(t_cut - t_peak, wf_i_m , wf_i_P, color = 'firebrick', alpha = 0.2)

    for ax in [ax1, ax2]:
        ax.axvline(tM_start,           label=r'$t_{\rm start} = t_{\rm peak} + $' + rf'${np.round(tM_start)}M$', c='darkgreen', lw=lw_large, alpha=1.0, ls=':' )
        ax.axvline(0.0,                label=r'$t_{\rm peak}$',                                               c='k',         lw=lw_large, alpha=1.0, ls='--')

    ax1.legend(loc='best', fontsize= 11)
    ax1.set_ylabel(r'$\Re(\Psi_4)$')
    ax2.set_ylabel(r'$\Im(\Psi_4)$')
    ax2.set_xlabel(r'$t/M$')
    for ax in [ax1, ax2]:   ax.set_xlim([-10, 80])
    # plt.suptitle('SXS:BBH:{} (residuals)'.format(NR_sim.NR_ID), size=24)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(os.path.join(outdir, 'Plots/Comparisons/Fancy_Reconstruction.pdf'), bbox_inches='tight')

    return

def global_corner(x, names, output, truths=None):

    """
    
    Create a corner plot of all parameters.
    
    Parameters
    ----------

    x       : dictionary    
        Dictionary of parameters.
    names   : list
        List of parameter names.
    output  : string
        Output directory.

    Returns
    -------

    Nothing, but saves a corner plot to the output directory.

    """

    samples = []
    for xy in names: samples.append(np.array(x[xy]))
    samples = np.transpose(samples)
    mask    = [i for i in range(samples.shape[-1]) if not all(samples[:,i]==samples[0,i]) ]

    fig = plt.figure(figsize=(10,10))
    C   = corner.corner(samples[:,mask],
                        quantiles     = [0.05, 0.5, 0.95],
                        labels        = names,
                        color         = 'darkred',
                        show_titles   = True,
                        title_kwargs  = {"fontsize": 12},
                        use_math_text = True,
                        truths = truths
                        )
    plt.savefig(os.path.join(output, 'Plots', 'Results', 'corner.png'), bbox_inches='tight')

    return

def plot_multiple_psd(psd_data, f_min, f_max, outdir, direction, window):
    """
    Plot multiple smoothed PSD curves in function of frequency.

    Parameters:
        psd_data (dict): A dictionary where keys are labels (str) and values are PSD arrays (np.ndarray).
        f_min (float): Minimum frequency.
        f_max (float): Maximum frequency.
        outdir (str): Output directory for saving the plot.
        direction (str): 'below' or 'above' to distinguish between smoothing directions.
        window (float): The smoothing window size.

    Returns:
        None
    """
    try:
        # Determine subfolder based on direction
        subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
        save_path = os.path.join(outdir, "Algorithm", subfolder)
        os.makedirs(save_path, exist_ok=True)

        # Set x-axis range based on direction
        if direction == "below":
            x_min, x_max = f_min/2, f_min + window
        elif direction == "above":
            x_min, x_max = f_max - window, f_max
        elif direction == "below-and-above":
            x_min, x_max = f_min, f_max

        # Create the plot
        plt.figure(figsize=(12, 8))

        for label, PSD_smoothed in psd_data.items():
            freq = np.linspace(0, f_max, len(PSD_smoothed))
            plt.plot(freq, PSD_smoothed, label=label, linestyle="dotted", linewidth=1.5)

        # Add labels, title, and grid
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD [Hz^-1]")
        plt.title(f"Smoothed PSD for Various Parameters ({direction.capitalize()})")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(x_min, x_max)
        #plt.legend()
        plt.grid(True)

        # Save the plot
        filename = "Multiple_Smoothed_PSD.png"
        path = os.path.join(save_path, filename)
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"Failed to generate smoothed PSD plot ({direction}): {e}")

def plot_psd_and_acf(psd_data, acf_data, asd_filepath, f_min, f_max, outdir, direction):
    """
    Plot multiple smoothed PSD and ACF curves in a single figure with two subplots.

    Parameters:
        psd_data (dict): A dictionary where keys are labels (str) and values are PSD arrays (np.ndarray).
        acf_data (dict): A dictionary where keys are labels (str) and values are ACF arrays (np.ndarray).
        f_min (float): Minimum frequency.
        f_max (float): Maximum frequency.
        t_start_g (float): Start time for ACF plot.
        t_end_g (float): End time for ACF plot.
        outdir (str): Output directory for saving the plot.
        direction (str): 'below', 'above', or 'below-and-above' to distinguish between smoothing directions.
        window (float): The smoothing window size.

    Returns:
        None
    """
    try:

        # Load ASD file and convert it to PSD
        freq_file, asd_file = np.loadtxt(asd_filepath, unpack=True)
        psd_file            = asd_file**2 

        # Determine subfolder based on smoothing direction
        subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
        save_path = os.path.join(outdir, "Algorithm", subfolder)

        # Clear the directory if it exists, otherwise create it
        os.makedirs(save_path, exist_ok=True)

        # Create a single figure with two subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 12))

        # ------------------ Plot PSD ------------------ #
        axs[0].plot(freq_file, psd_file, label="No window application", linewidth=1.2, linestyle='--', color="black")
        for i, (label, PSD_smoothed) in enumerate(psd_data.items()):
            freq = np.linspace(0, f_max, len(PSD_smoothed))
            alpha = max(0.3, 1 - (i * 0.15))  # Decrease opacity for different curves
            axs[0].plot(freq, PSD_smoothed, label=label, linewidth=2, color=colbBlue, alpha=alpha)

        axs[0].set_xlabel("Frequency [Hz]")
        axs[0].set_ylabel("PSD [Hz^-1]")
        axs[0].set_title(f"Smoothed PSD ({direction.capitalize()})")
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        axs[0].grid(True)

        # ------------------ Plot ACF ------------------
        # Duration time
        dt = 1/(2*f_max)

        for i, (label, ACF_smoothed) in enumerate(acf_data.items()):
            N_FFT = len(ACF_smoothed)
            T = N_FFT*dt
            t_array = np.linspace(0, T, N_FFT)
            alpha = max(0.3, 1 - (i * 0.15))  # Decrease opacity for different curves
            axs[1].plot(t_array, ACF_smoothed, label=label, linewidth=2, color=colbBlue, alpha=alpha)

        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("ACF")
        axs[1].set_title(f"Smoothed ACF ({direction.capitalize()})")
        axs[1].grid(True)

        # Center the legend inside the plot
        axs[1].legend(loc="upper center")

        # Adjust layout and save the plot
        plt.tight_layout()
        filename = "PSD_and_ACF_Smoothed.pdf"
        path = os.path.join(save_path, filename)
        plt.savefig(path)
        plt.close(fig)

    except Exception as e:
        print(f"Failed to generate smoothed PSD and ACF plots ({direction}): {e}")

def plot_psd_near_fmin_fmax(psd_data, f_min, f_max, window_size_DX, window_size_SX, outdir, direction):
    """
    Plot PSD curves near f_min and f_max in a single figure with two side-by-side subplots.
    
    Parameters:
        psd_data (dict): Dictionary where keys are labels (str) and values are PSD arrays (np.ndarray).
        f_min (float): Minimum frequency.
        f_max (float): Maximum frequency.
        window_size (float): The smoothing window size.
        outdir (str): Output directory for saving the plot.
        direction (str): 'below', 'above', or 'below-and-above' to distinguish between smoothing directions.

    Returns:
        None
    """
    try:
        # Define fixed colors
        colbBlue = "#4477AA"  # Base color for PSD
        colbRed = "#EE6677"   # Alternative color if needed

        # Determine subfolder based on smoothing direction
        subfolder = {
            "below": "Left_smoothing",
            "above": "Right_smoothing",
            "below-and-above": "Both_edges_smoothing"
        }.get(direction, "Unknown_smoothing")

        save_path = os.path.join(outdir, "Algorithm", subfolder)
        os.makedirs(save_path, exist_ok=True)

        # Set x-axis limits for zoomed regions
        x_min1, x_max1 = f_min * 0.9, (f_min + window_size_DX)  # Zoom near f_min
        x_min2, x_max2 = (f_max - window_size_SX), f_max  # Zoom near f_max

        # Create figure with two side-by-side subplots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # ------------------ Plot PSD near f_min and f_max ------------------
        for i, (label, PSD_smoothed) in enumerate(psd_data.items()):
            freq = np.linspace(0, f_max, len(PSD_smoothed))  # Generate frequency axis
            alpha = max(0.3, 1 - (i * 0.15))  # Decrease opacity for different curves
            
            # Identify indices for zoomed regions
            idx_min1, idx_max1 = np.argmin(np.abs(freq - x_min1)), np.argmin(np.abs(freq - x_max1))
            idx_min2, idx_max2 = np.argmin(np.abs(freq - x_min2)), np.argmin(np.abs(freq - x_max2))

            # y-axis limits dynamically
            y_min1, y_max1 = PSD_smoothed[idx_min1], PSD_smoothed[idx_max1]
            y_min2, y_max2 = PSD_smoothed[idx_min2], PSD_smoothed[idx_max2]

            # Re-order y-lims
            y_min1, y_max1 = min(y_min1, y_max1), max(y_min1, y_max1)
            y_min2, y_max2 = min(y_min2, y_max2), max(y_min2, y_max2)

            # Plot for both subplots with alpha variation
            axs[0].plot(freq, PSD_smoothed, label=label, linewidth=1.5, color=colbBlue, alpha=alpha)
            axs[1].plot(freq, PSD_smoothed, label=label, linewidth=1.5, color=colbRed, alpha=alpha)

        # Adjust subplot 1 (f_min)
        axs[0].set_xlabel("Frequency [Hz]")
        axs[0].set_ylabel("PSD [Hz^-1]")
        axs[0].set_title(f"Smoothed PSD near f_min ({direction.capitalize()})")
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        axs[0].set_xlim(x_min1 * 0.99, x_max1 * 1.01)
        axs[0].set_ylim(y_min1 * 0.5, y_max1 * 2)
        axs[0].grid(True)

        # Adjust subplot 2 (f_max)
        axs[1].set_xlabel("Frequency [Hz]")
        axs[1].set_ylabel("PSD [Hz^-1]")
        axs[1].set_title(f"Smoothed PSD near f_max ({direction.capitalize()})")
        axs[1].set_yscale("log")
        axs[1].set_xlim(x_min2 * 0.9, x_max2 *1.01)
        axs[1].set_ylim(y_min2 * 0.02, y_max2 * 1.1)
        #axs[1].set_ylim(1e-46, 1e-41)
        axs[1].grid(True)

        # TO IMPROVE: fix the limits of the axes.

        # Adjust layout and save the plot
        plt.tight_layout()
        filename = "PSD_Near_fmin_fmax.pdf"
        path = os.path.join(save_path, filename)
        plt.savefig(path)
        plt.close(fig)
        print(f"\nSaved PSD plots near f_min and f_max to {path}.\n")

    except Exception as e:
        print(f"Failed to generate PSD plots near f_min and f_max ({direction}): {e}")

def plot_acf_interpolated(t_array, t_trunc, ACF_smoothed, truncated_acf, outdir, window_size_DX, window_size_SX, k, saturation_DX, saturation_SX, direction):

    # Determine subfolder based on direction
    subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
    save_path = os.path.join(outdir, "Algorithm", subfolder)
    os.makedirs(save_path, exist_ok=True)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot acf interpolated
    plt.plot(t_array,ACF_smoothed,label="Original ACF", color=colbBlue)
    plt.plot(t_trunc,truncated_acf,label="Truncated ACF",linestyle="dotted", color=colbRed)
    plt.legend()
    plt.xlabel("t [s]")
    plt.xlim(t_trunc[0],t_trunc[-1]*1.5)
    plt.ylim(min(ACF_smoothed)*0.8, max(ACF_smoothed)*1.2)

    # Save the plot
    filename = f"Truncated_ACF_wDX={round(window_size_DX,1)}Hz_wSX={round(window_size_SX,1)}Hz_k={round(k,3)}_sat_DX={round(saturation_DX,0)}_sat_SX={round(saturation_SX,0)}.pdf"
    path = os.path.join(save_path, filename)
    plt.savefig(path)
    plt.close()

def plot_mismatch_by_window_DX(mismatch_data, outdir, direction, M, dL, N_fft):
    """
    Plot mismatch for real and imaginary components against window_size_DX for fixed other parameters.
    """
    components = ['real', 'imaginary']
    percentiles = [50]
    subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
    save_path = os.path.join(outdir, "Algorithm", subfolder)
    os.makedirs(save_path, exist_ok=True)
    
    for N_FFT in N_fft:
        for k, saturation_DX, saturation_SX, window_size_SX in sorted(set((k, sDX, sSX, wsx) for (_, wsx, k, sDX, sSX) in mismatch_data.keys())):
            for component in components:
                plt.figure(figsize=(12, 8))
                for perc in percentiles:
                    window_vals, mismatch_vals = [], []
                    for (wDX, wsx, k_val, sDX, sSX), data in mismatch_data.items():
                        if k_val == k and sDX == saturation_DX and sSX == saturation_SX and wsx == window_size_SX:
                            window_vals.append(wDX)
                            mismatch_vals.append(data[component][perc])
                    plt.plot(window_vals, mismatch_vals, label=f"{perc}% CI", marker='o')
                plt.xlabel("wDX [Hz]", fontsize=26)
                plt.ylabel("Mismatch", fontsize=26)
                plt.xticks(fontsize=22)
                plt.yticks(fontsize=22)
                plt.grid(True)
                filename = f"Mismatch_M={M}_dL={dL}_k={round(k,0)}_satDX={saturation_DX:.2e}_satSX={saturation_SX:.2e}_wsx={window_size_SX}_direction={direction}_NFFT_{N_FFT}.pdf"
                plt.savefig(os.path.join(save_path, filename))
                plt.close()

def plot_mismatch_by_window_SX(mismatch_data, outdir, direction, M, dL, N_fft):
    """
    Plot mismatch for real and imaginary components against window_size_SX for fixed other parameters.
    """
    components = ['real', 'imaginary']
    percentiles = [50]
    subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
    save_path = os.path.join(outdir, "Algorithm", subfolder)
    os.makedirs(save_path, exist_ok=True)
    
    for N_FFT in N_fft:
        for k, saturation_DX, saturation_SX, window_size_DX in sorted(set((k, sDX, sSX, wdx) for (wdx, _, k, sDX, sSX) in mismatch_data.keys())):
            for component in components:
                plt.figure(figsize=(12, 8))
                for perc in percentiles:
                    window_vals, mismatch_vals = [], []
                    for (wdx, wSX, k_val, sDX, sSX), data in mismatch_data.items():
                        if k_val == k and sDX == saturation_DX and sSX == saturation_SX and wdx == window_size_DX:
                            window_vals.append(wSX)
                            mismatch_vals.append(data[component][perc])
                    plt.plot(window_vals, mismatch_vals, label=f"{perc}% CI", marker='o')
                plt.xlabel("wSX [Hz]", fontsize=26)
                plt.ylabel("Mismatch", fontsize=26)
                plt.xticks(fontsize=22)
                plt.yticks(fontsize=22)
                plt.grid(True)
                filename = f"Mismatch_M={M}_dL={dL}_k={round(k,0)}_satDX={saturation_DX:.2e}_satSX={saturation_SX:.2e}_wdx={window_size_DX}_direction={direction}_NFFT_{N_FFT}.pdf"
                plt.savefig(os.path.join(save_path, filename))
                plt.close()

def plot_optimal_SNR_by_window_DX(optimal_SNR_data, outdir, direction, M, dL, N_fft):
    """
    Plot optimal SNR for real and imaginary components against window_size_DX for fixed k and saturations.
    """
    components = ['real', 'imaginary']
    percentiles = [50]
    subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
    save_path = os.path.join(outdir, "Algorithm", subfolder)
    os.makedirs(save_path, exist_ok=True)
    
    for N_FFT in N_fft:
        for k, saturation_DX, saturation_SX, window_size_SX in sorted(set((k, sDX, sSX, wsx) for (_, wsx, k, sDX, sSX) in optimal_SNR_data.keys())):
            for component in components:
                plt.figure(figsize=(12, 8))
                for perc in percentiles:
                    window_vals, snr_vals = [], []
                    for (wDX, wsx, k_val, sDX, sSX), data in optimal_SNR_data.items():
                        if k_val == k and sDX == saturation_DX and sSX == saturation_SX and wsx == window_size_SX:
                            window_vals.append(wDX)
                            snr_vals.append(data[component][perc])
                    plt.plot(window_vals, snr_vals, label=f"{perc}% CI", marker='o')
                plt.xlabel("wDX [Hz]", fontsize=26)
                plt.ylabel("Optimal SNR", fontsize=26)
                plt.xticks(fontsize=22)
                plt.yticks(fontsize=22)
                plt.grid(True)
                filename = f"Optimal_SNR_M={M}_dL={dL}_k={round(k,0)}_satDX={saturation_DX:.2e}_satSX={saturation_SX:.2e}_wsx={window_size_SX}_direction={direction}_NFFT_{N_FFT}.pdf"
                plt.savefig(os.path.join(save_path, filename))
                plt.close()

def plot_optimal_SNR_by_window_SX(optimal_SNR_data, outdir, direction, M, dL, N_fft):
    """
    Plot optimal SNR for real and imaginary components against window_size_SX for fixed k and saturations.
    """
    components = ['real', 'imaginary']
    percentiles = [50]
    subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
    save_path = os.path.join(outdir, "Algorithm", subfolder)
    os.makedirs(save_path, exist_ok=True)
    
    for N_FFT in N_fft:
        for k, saturation_DX, saturation_SX, window_size_DX in sorted(set((k, sDX, sSX, wdx) for (wdx, _, k, sDX, sSX) in optimal_SNR_data.keys())):
            for component in components:
                plt.figure(figsize=(12, 8))
                for perc in percentiles:
                    window_vals, snr_vals = [], []
                    for (wdx, wSX, k_val, sDX, sSX), data in optimal_SNR_data.items():
                        if k_val == k and sDX == saturation_DX and sSX == saturation_SX and wdx == window_size_DX:
                            window_vals.append(wSX)
                            snr_vals.append(data[component][perc])
                    plt.plot(window_vals, snr_vals, label=f"{perc}% CI", marker='o')
                plt.xlabel("wSX [Hz]", fontsize=26)
                plt.ylabel("Optimal SNR", fontsize=26)
                plt.xticks(fontsize=22)
                plt.yticks(fontsize=22)
                plt.grid(True)
                filename = f"Optimal_SNR_M={M}_dL={dL}_k={round(k,0)}_satDX={saturation_DX:.2e}_satSX={saturation_SX:.2e}_wdx={window_size_DX}_direction={direction}_NFFT_{N_FFT}.pdf"
                plt.savefig(os.path.join(save_path, filename))
                plt.close()

def plot_condition_number_by_window_DX(condition_numbers, outdir, direction, M, dL, N_fft):
    """
    Plot condition number against window_size_DX.
    """
    subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
    save_path = os.path.join(outdir, "Algorithm", subfolder)
    os.makedirs(save_path, exist_ok=True)
    
    for N_FFT in N_fft:
        plt.figure(figsize=(12, 8))
        window_vals, cond_vals = [], []
        for (wDX, _, k, sDX, sSX), cond_number in condition_numbers.items():
            window_vals.append(wDX)
            cond_vals.append(cond_number)
        plt.plot(window_vals, cond_vals, marker='o')
        plt.xlabel("wDX [Hz]", fontsize=26)
        plt.ylabel("Condition Number", fontsize=26)
        plt.xscale("log")
        plt.yscale('log')
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid(True)
        filename = f"Condition_Number_M={M}_dL={dL}_direction={direction}_NFFT_{N_FFT}.pdf"
        plt.savefig(os.path.join(save_path, filename))
        plt.close()

def plot_condition_number_by_window_SX(condition_numbers, outdir, direction, M, dL, N_fft):
    """
    Plot condition number against window_size_SX.
    """
    subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
    save_path = os.path.join(outdir, "Algorithm", subfolder)
    os.makedirs(save_path, exist_ok=True)
    
    for N_FFT in N_fft:
        plt.figure(figsize=(12, 8))
        window_vals, cond_vals = [], []
        for (_, wSX, k, sDX, sSX), cond_number in condition_numbers.items():
            window_vals.append(wSX)
            cond_vals.append(cond_number)
        plt.plot(window_vals, cond_vals, marker='o')
        plt.xlabel("wSX [Hz]", fontsize=26)
        plt.ylabel("Condition Number", fontsize=26)
        plt.xscale("log")
        plt.yscale('log')
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid(True)
        filename = f"Condition_Number_M={M}_dL={dL}_direction={direction}_NFFT_{N_FFT}.pdf"
        plt.savefig(os.path.join(save_path, filename))
        plt.close()

def plot_mismatch_by_k(mismatch_data, outdir, direction, M, dL, N_fft):
    """
    Plot mismatch for real and imaginary components by varying k, keeping window sizes and saturations fixed.
    """
    components = ['real', 'imaginary']
    percentiles = [50]
    subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
    save_path = os.path.join(outdir, "Algorithm", subfolder)
    os.makedirs(save_path, exist_ok=True)
    
    window_sizes_DX = sorted(set(wdx for wdx, _, _, _, _ in mismatch_data.keys()))
    window_sizes_SX = sorted(set(wsx for _, wsx, _, _, _ in mismatch_data.keys()))
    saturation_DX_values = sorted(set(s for _, _, _, s, _ in mismatch_data.keys()))
    saturation_SX_values = sorted(set(s for _, _, _, _, s in mismatch_data.keys()))
    
    for N_FFT in N_fft:
        for window_size_DX in window_sizes_DX:
            for window_size_SX in window_sizes_SX:
                for saturation_DX in saturation_DX_values:
                    for saturation_SX in saturation_SX_values:
                        for component in components:
                            plt.figure(figsize=(12, 8))
                            for perc in percentiles:
                                k_vals, mismatch_vals = [], []
                                for (wdx, wsx, k_val, sDX, sSX), data in mismatch_data.items():
                                    if wdx == window_size_DX and wsx == window_size_SX and sDX == saturation_DX and sSX == saturation_SX:
                                        k_vals.append(k_val)
                                        mismatch_vals.append(data[component][perc])
                            plt.plot(k_vals, mismatch_vals, label=f"{perc}% CI", marker='o')
                            plt.xlabel("k", fontsize=26)
                            plt.ylabel("Mismatch", fontsize=26)
                            plt.xticks(fontsize=22)
                            plt.yticks(fontsize=22)
                            plt.legend()
                            plt.xscale("log")
                            plt.grid(True)
                            filename = f"Mismatch_M={M}M0_dL={dL}Mpc_{component}_wDX={round(window_size_DX,1)}_wSX={round(window_size_SX,1)}_satDX={saturation_DX:.2e}_satSX={saturation_SX:.2e}_direction={direction}_NFFT_{round(N_FFT,0)}.pdf"
                            plt.savefig(os.path.join(save_path, filename))
                            plt.close()

def plot_optimal_SNR_by_k(optimal_SNR_data, outdir, direction, M, dL, N_fft):
    """
    Plot optimal SNR for real and imaginary components by varying k, keeping window sizes and saturations fixed.
    """
    components = ['real', 'imaginary']
    percentiles = [50]
    subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
    save_path = os.path.join(outdir, "Algorithm", subfolder)
    os.makedirs(save_path, exist_ok=True)
    
    window_sizes_DX = sorted(set(wdx for wdx, _, _, _, _ in optimal_SNR_data.keys()))
    window_sizes_SX = sorted(set(wsx for _, wsx, _, _, _ in optimal_SNR_data.keys()))
    saturation_DX_values = sorted(set(s for _, _, _, s, _ in optimal_SNR_data.keys()))
    saturation_SX_values = sorted(set(s for _, _, _, _, s in optimal_SNR_data.keys()))
    
    for N_FFT in N_fft:
        for window_size_DX in window_sizes_DX:
            for window_size_SX in window_sizes_SX:
                for saturation_DX in saturation_DX_values:
                    for saturation_SX in saturation_SX_values:
                        for component in components:
                            plt.figure(figsize=(12, 8))
                            for perc in percentiles:
                                k_vals, snr_vals = [], []
                                for (wdx, wsx, k_val, sDX, sSX), data in optimal_SNR_data.items():
                                    if wdx == window_size_DX and wsx == window_size_SX and sDX == saturation_DX and sSX == saturation_SX:
                                        k_vals.append(k_val)
                                        snr_vals.append(data[component][perc])
                            plt.plot(k_vals, snr_vals, label=f"{perc}% CI", marker='o')
                            plt.xlabel("k", fontsize=26)
                            plt.ylabel("Optimal SNR", fontsize=26)
                            plt.xticks(fontsize=22)
                            plt.yticks(fontsize=22)
                            plt.legend()
                            plt.xscale("log")
                            plt.grid(True)
                            filename = f"Optimal_SNR_M={M}M0_dL={dL}Mpc_{component}_wDX={round(window_size_DX,1)}_wSX={round(window_size_SX,1)}_satDX={saturation_DX:.2e}_satSX={saturation_SX:.2e}_direction={direction}_NFFT_{round(N_FFT,0)}.pdf"
                            plt.savefig(os.path.join(save_path, filename))
                            plt.close()

def plot_condition_number_by_k(condition_numbers, outdir, direction, M, dL, N_fft):
    """
    Plot the condition number of the ACF Toeplitz matrix by varying k, keeping window sizes, saturation_DX, and saturation_SX fixed.
    """
    subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
    save_path = os.path.join(outdir, "Algorithm", subfolder)
    os.makedirs(save_path, exist_ok=True)
    
    window_sizes_DX = sorted(set(wdx for wdx, _, _, _, _ in condition_numbers.keys()))
    window_sizes_SX = sorted(set(wsx for _, wsx, _, _, _ in condition_numbers.keys()))
    saturation_DX_values = sorted(set(s for _, _, _, s, _ in condition_numbers.keys()))
    saturation_SX_values = sorted(set(s for _, _, _, _, s in condition_numbers.keys()))
    
    for N_FFT in N_fft:
        for window_size_DX in window_sizes_DX:
            for window_size_SX in window_sizes_SX:
                for saturation_DX in saturation_DX_values:
                    for saturation_SX in saturation_SX_values:
                        plt.figure(figsize=(12, 8))
                        k_vals, cond_vals = [], []
                        for (wdx, wsx, k_val, sDX, sSX), cond_number in condition_numbers.items():
                            if wdx == window_size_DX and wsx == window_size_SX and sDX == saturation_DX and sSX == saturation_SX:
                                k_vals.append(k_val)
                                cond_vals.append(cond_number)
                        plt.plot(k_vals, cond_vals, marker='o')
                        plt.xlabel("k", fontsize=26)
                        plt.ylabel("Condition Number", fontsize=26)
                        plt.xticks(fontsize=22)
                        plt.yticks(fontsize=22)
                        plt.xscale("log")
                        #plt.yscale('log')
                        plt.grid(True)
                        filename = f"Condition_Number_M={M}M0_dL={dL}Mpc_wDX={round(window_size_DX,1)}_wSX={round(window_size_SX,1)}_satDX={saturation_DX:.2e}_satSX={saturation_SX:.2e}_direction={direction}_NFFT_{round(N_FFT,0)}.pdf"
                        plt.savefig(os.path.join(save_path, filename))
                        plt.close()

def plot_mismatch_by_saturation_DX(mismatch_data, outdir, direction, M, dL, N_fft):
    """
    Plot mismatch for real and imaginary components by varying saturation_DX, keeping window sizes, k, and saturation_SX fixed.
    """
    components = ['real', 'imaginary']
    percentiles = [50]
    subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
    save_path = os.path.join(outdir, "Algorithm", subfolder)
    os.makedirs(save_path, exist_ok=True)
    
    window_sizes_DX = sorted(set(wdx for wdx, _, _, _, _ in mismatch_data.keys()))
    window_sizes_SX = sorted(set(wsx for _, wsx, _, _, _ in mismatch_data.keys()))
    k_values = sorted(set(k for _, _, k, _, _ in mismatch_data.keys()))
    saturation_SX_values = sorted(set(s for _, _, _, _, s in mismatch_data.keys()))
    
    for N_FFT in N_fft:
        for window_size_DX in window_sizes_DX:
            for window_size_SX in window_sizes_SX:
                for k in k_values:
                    for saturation_SX in saturation_SX_values:
                        for component in components:
                            plt.figure(figsize=(12, 8))
                            for perc in percentiles:
                                sat_DX_vals, mismatch_vals = [], []
                                for (wdx, wsx, k_val, sDX, sSX), data in mismatch_data.items():
                                    if wdx == window_size_DX and wsx == window_size_SX and k_val == k and sSX == saturation_SX:
                                        sat_DX_vals.append(sDX)
                                        mismatch_vals.append(data[component][perc])
                            plt.plot(sat_DX_vals, mismatch_vals, label=f"{perc}% CI", marker='o')
                            plt.xlabel("Saturation DX", fontsize=26)
                            plt.ylabel("Mismatch", fontsize=26)
                            plt.xticks(fontsize=22)
                            plt.yticks(fontsize=22)
                            plt.legend()
                            plt.xscale('log')
                            plt.grid(True)
                            filename = f"Mismatch_M={M}M0_dL={dL}Mpc_{component}_wDX={round(window_size_DX,1)}_wSX={round(window_size_SX,1)}_k={round(k,0)}_satSX={saturation_SX:.2e}_direction={direction}_NFFT_{round(N_FFT,0)}.pdf"
                            plt.savefig(os.path.join(save_path, filename))
                            plt.close()

def plot_optimal_SNR_by_saturation_DX(optimal_SNR_data, outdir, direction, M, dL, N_fft):
    """
    Plot optimal SNR for real and imaginary components by varying saturation_DX, keeping window sizes, k, and saturation_SX fixed.
    """
    components = ['real', 'imaginary']
    percentiles = [50]
    subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
    save_path = os.path.join(outdir, "Algorithm", subfolder)
    os.makedirs(save_path, exist_ok=True)
    
    window_sizes_DX = sorted(set(wdx for wdx, _, _, _, _ in optimal_SNR_data.keys()))
    window_sizes_SX = sorted(set(wsx for _, wsx, _, _, _ in optimal_SNR_data.keys()))
    k_values = sorted(set(k for _, _, k, _, _ in optimal_SNR_data.keys()))
    saturation_SX_values = sorted(set(s for _, _, _, _, s in optimal_SNR_data.keys()))
    
    for N_FFT in N_fft:
        for window_size_DX in window_sizes_DX:
            for window_size_SX in window_sizes_SX:
                for k in k_values:
                    for saturation_SX in saturation_SX_values:
                        for component in components:
                            plt.figure(figsize=(12, 8))
                            for perc in percentiles:
                                sat_DX_vals, snr_vals = [], []
                                for (wdx, wsx, k_val, sDX, sSX), data in optimal_SNR_data.items():
                                    if wdx == window_size_DX and wsx == window_size_SX and k_val == k and sSX == saturation_SX:
                                        sat_DX_vals.append(sDX)
                                        snr_vals.append(data[component][perc])
                            plt.plot(sat_DX_vals, snr_vals, label=f"{perc}% CI", marker='o')
                            plt.xlabel("Saturation DX", fontsize=26)
                            plt.ylabel("Optimal SNR", fontsize=26)
                            plt.xticks(fontsize=22)
                            plt.yticks(fontsize=22)
                            plt.legend()
                            plt.xscale('log')
                            plt.grid(True)
                            filename = f"Optimal_SNR_M={M}M0_dL={dL}Mpc_wDX={round(window_size_DX,1)}_wSX={round(window_size_SX,1)}_k={round(k,0)}_satSX={saturation_SX:.2e}_direction={direction}_NFFT_{round(N_FFT,0)}.pdf"
                            plt.savefig(os.path.join(save_path, filename))
                            plt.close()

def plot_condition_number_by_saturation_DX(condition_numbers, outdir, direction, M, dL, N_fft):
    """
    Plot the condition number of the ACF Toeplitz matrix by varying saturation_DX, keeping window sizes, k, and saturation_SX fixed.
    """
    subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
    save_path = os.path.join(outdir, "Algorithm", subfolder)
    os.makedirs(save_path, exist_ok=True)
    
    window_sizes_DX = sorted(set(wdx for wdx, _, _, _, _ in condition_numbers.keys()))
    window_sizes_SX = sorted(set(wsx for _, wsx, _, _, _ in condition_numbers.keys()))
    k_values = sorted(set(k for _, _, k, _, _ in condition_numbers.keys()))
    saturation_SX_values = sorted(set(s for _, _, _, _, s in condition_numbers.keys()))
    
    for N_FFT in N_fft:
        for window_size_DX in window_sizes_DX:
            for window_size_SX in window_sizes_SX:
                for k in k_values:
                    for saturation_SX in saturation_SX_values:
                        plt.figure(figsize=(12, 8))
                        sat_DX_vals, cond_vals = [], []
                        for (wdx, wsx, k_val, sDX, sSX), cond_number in condition_numbers.items():
                            if wdx == window_size_DX and wsx == window_size_SX and k_val == k and sSX == saturation_SX:
                                sat_DX_vals.append(sDX)
                                cond_vals.append(cond_number)
                        plt.plot(sat_DX_vals, cond_vals, marker='o')
                        plt.xlabel("Saturation DX", fontsize=26)
                        plt.ylabel("Condition Number", fontsize=26)
                        plt.xticks(fontsize=22)
                        plt.yticks(fontsize=22)
                        plt.xscale('log')
                        plt.yscale('log')
                        plt.grid(True)
                        filename = f"Condition_Number_M={M}M0_dL={dL}Mpc_wDX={round(window_size_DX,1)}_wSX={round(window_size_SX,1)}_k={round(k,0)}_satSX={saturation_SX:.2e}_direction={direction}_NFFT_{round(N_FFT,0)}.pdf"
                        plt.savefig(os.path.join(save_path, filename))
                        plt.close()

def plot_mismatch_by_saturation_SX(mismatch_data, outdir, direction, M, dL, N_fft):
    """
    Plot mismatch for real and imaginary components by varying saturation_SX, keeping window sizes, k, and saturation_DX fixed.
    """
    components = ['real', 'imaginary']
    percentiles = [50]
    subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
    save_path = os.path.join(outdir, "Algorithm", subfolder)
    os.makedirs(save_path, exist_ok=True)
    
    window_sizes_DX = sorted(set(wdx for wdx, _, _, _, _ in mismatch_data.keys()))
    window_sizes_SX = sorted(set(wsx for _, wsx, _, _, _ in mismatch_data.keys()))
    k_values = sorted(set(k for _, _, k, _, _ in mismatch_data.keys()))
    saturation_DX_values = sorted(set(s for _, _, _, s, _ in mismatch_data.keys()))
    
    for N_FFT in N_fft:
        for window_size_DX in window_sizes_DX:
            for window_size_SX in window_sizes_SX:
                for k in k_values:
                    for saturation_DX in saturation_DX_values:
                        for component in components:
                            plt.figure(figsize=(12, 8))
                            for perc in percentiles:
                                sat_SX_vals, mismatch_vals = [], []
                                for (wdx, wsx, k_val, sDX, sSX), data in mismatch_data.items():
                                    if wdx == window_size_DX and wsx == window_size_SX and k_val == k and sDX == saturation_DX:
                                        sat_SX_vals.append(sSX)
                                        mismatch_vals.append(data[component][perc])
                            plt.plot(sat_SX_vals, mismatch_vals, label=f"{perc}% CI", marker='o')
                            plt.xlabel("Saturation SX", fontsize=26)
                            plt.ylabel("Mismatch", fontsize=26)
                            plt.xticks(fontsize=22)
                            plt.yticks(fontsize=22)
                            plt.xscale('log')
                            plt.legend()
                            plt.grid(True)
                            filename = f"Mismatch_M={M}M0_dL={dL}Mpc_{component}_wDX={round(window_size_DX,1)}_wSX={round(window_size_SX,1)}_k={round(k,0)}_satDX={saturation_DX:.2e}_direction={direction}_NFFT_{round(N_FFT,0)}.pdf"
                            plt.savefig(os.path.join(save_path, filename))
                            plt.close()

def plot_optimal_SNR_by_saturation_SX(optimal_SNR_data, outdir, direction, M, dL, N_fft):
    """
    Plot optimal SNR for real and imaginary components by varying saturation_SX, keeping window sizes, k, and saturation_DX fixed.
    """
    components = ['real', 'imaginary']
    percentiles = [50]
    subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
    save_path = os.path.join(outdir, "Algorithm", subfolder)
    os.makedirs(save_path, exist_ok=True)
    
    window_sizes_DX = sorted(set(wdx for wdx, _, _, _, _ in optimal_SNR_data.keys()))
    window_sizes_SX = sorted(set(wsx for _, wsx, _, _, _ in optimal_SNR_data.keys()))
    k_values = sorted(set(k for _, _, k, _, _ in optimal_SNR_data.keys()))
    saturation_DX_values = sorted(set(s for _, _, _, s, _ in optimal_SNR_data.keys()))
    
    for N_FFT in N_fft:
        for window_size_DX in window_sizes_DX:
            for window_size_SX in window_sizes_SX:
                for k in k_values:
                    for saturation_DX in saturation_DX_values:
                        for component in components:
                            plt.figure(figsize=(12, 8))
                            for perc in percentiles:
                                sat_SX_vals, snr_vals = [], []
                                for (wdx, wsx, k_val, sDX, sSX), data in optimal_SNR_data.items():
                                    if wdx == window_size_DX and wsx == window_size_SX and k_val == k and sDX == saturation_DX:
                                        sat_SX_vals.append(sSX)
                                        snr_vals.append(data[component][perc])
                            plt.plot(sat_SX_vals, snr_vals, label=f"{perc}% CI", marker='o')
                            plt.xlabel("Saturation SX", fontsize=26)
                            plt.ylabel("Optimal SNR", fontsize=26)
                            plt.xticks(fontsize=22)
                            plt.yticks(fontsize=22)
                            plt.xscale('log')
                            plt.legend()
                            plt.grid(True)
                            filename = f"Optimal_SNR_M={M}M0_dL={dL}Mpc_wDX={round(window_size_DX,1)}_wSX={round(window_size_SX,1)}_k={round(k,0)}_satDX={saturation_DX:.2e}_direction={direction}_NFFT_{round(N_FFT,0)}.pdf"
                            plt.savefig(os.path.join(save_path, filename))
                            plt.close()

def plot_condition_number_by_saturation_SX(condition_numbers, outdir, direction, M, dL, N_fft):
    """
    Plot the condition number of the ACF Toeplitz matrix by varying saturation_SX, keeping window sizes, k, and saturation_DX fixed.
    """
    subfolder = "Left_smoothing" if direction == "below" else "Right_smoothing" if direction == "above" else "Both_edges_smoothing"
    save_path = os.path.join(outdir, "Algorithm", subfolder)
    os.makedirs(save_path, exist_ok=True)
    
    window_sizes_DX = sorted(set(wdx for wdx, _, _, _, _ in condition_numbers.keys()))
    window_sizes_SX = sorted(set(wsx for _, wsx, _, _, _ in condition_numbers.keys()))
    k_values = sorted(set(k for _, _, k, _, _ in condition_numbers.keys()))
    saturation_DX_values = sorted(set(s for _, _, _, s, _ in condition_numbers.keys()))
    
    for N_FFT in N_fft:
        for window_size_DX in window_sizes_DX:
            for window_size_SX in window_sizes_SX:
                for k in k_values:
                    for saturation_DX in saturation_DX_values:
                        plt.figure(figsize=(12, 8))
                        sat_SX_vals, cond_vals = [], []
                        for (wdx, wsx, k_val, sDX, sSX), cond_number in condition_numbers.items():
                            if wdx == window_size_DX and wsx == window_size_SX and k_val == k and sDX == saturation_DX:
                                sat_SX_vals.append(sSX)
                                cond_vals.append(cond_number)
                        plt.plot(sat_SX_vals, cond_vals, marker='o')
                        plt.xlabel("Saturation SX", fontsize=26)
                        plt.ylabel("Condition Number", fontsize=26)
                        plt.xticks(fontsize=22)
                        plt.yticks(fontsize=22)
                        plt.xscale('log')
                        plt.yscale('log')
                        plt.grid(True)
                        filename = f"Condition_Number_M={M}M0_dL={dL}Mpc_wDX={round(window_size_DX,1)}_wSX={round(window_size_SX,1)}_k={round(k,0)}_satDX={saturation_DX:.2e}_direction={direction}_NFFT_{round(N_FFT,0)}.pdf"
                        plt.savefig(os.path.join(save_path, filename))
                        plt.close()

def plot_condition_numbers(outdir, condition_numbers, thresholds=(1e3, 1e6)):

    """
    Plot the condition numbers of the ACF Toeplitz matrix as a function of window size for different k values,
    including shaded regions to indicate conditioning quality.
    
    Parameters:
        condition_numbers (dict): Dictionary with keys as (window_size, k) and values as condition numbers.
        outdir (str): Directory to save the plot.
        thresholds (tuple): Thresholds for marking the zones (well-conditioned, moderately, poorly).
                            Default: (1e3, 1e6).
    """

    # Ensure the input is a dictionary
    if not isinstance(condition_numbers, dict):
        raise ValueError("condition_numbers must be a dictionary.")

    # Extract unique values of k and sorted window sizes
    ks = sorted(set(k for _, k in condition_numbers.keys()))
    window_sizes = sorted(set(ws for ws, _ in condition_numbers.keys()))

    # Define the regions
    low_threshold, high_threshold = thresholds

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Add shaded regions
    plt.axhspan(0, low_threshold, color='green', alpha=0.1, label="Well-Conditioned")
    plt.axhspan(low_threshold, high_threshold, color='yellow', alpha=0.1, label="Moderately Conditioned")
    plt.axhspan(high_threshold, 10 * high_threshold, color='red', alpha=0.1, label="Poorly Conditioned")

    for k in ks:
        # Extract condition numbers for each window_size at the current k
        cond_numbers = [condition_numbers[(ws, k)] for ws in window_sizes]
        plt.plot(window_sizes, cond_numbers, label=f"k = {k}", linestyle="dotted")

    # Configure plot
    plt.xlabel("Window Size")
    plt.ylabel("Condition Number")
    plt.title("Condition Number of ACF Toeplitz Matrix vs Window Size")
    plt.yscale("log")
    plt.legend(title="Steepness (k)", loc="upper left")
    plt.grid(True)

    # Save plot to file
    os.makedirs(outdir, exist_ok=True)
    plot_file_path = os.path.join(outdir, "Algorithm/Condition_Numbers_Plot.png")
    plt.savefig(plot_file_path)
    print(f"Condition number plot saved to {plot_file_path}")

def plot_mismatch_optimal_SNR_condition_number_window_parameters(mismatch_data, optimal_SNR_data, condition_numbers, outdir, direction, M, dL, N_FFT):
    """
    Plots mismatch, optimal SNR, and condition number data if the corresponding x-axis variable has dim > 1.
    """

    # Define plotting functions with their corresponding x-axis variables
    plot_functions = {
        "mismatch": {
            "plot_mismatch_by_window_DX": "window_DX",
            "plot_mismatch_by_window_SX": "window_SX",
            "plot_mismatch_by_k": "k",
            "plot_mismatch_by_saturation_DX": "saturation_DX",
            "plot_mismatch_by_saturation_SX": "saturation_SX",
        },
        "optimal_SNR": {
            "plot_optimal_SNR_by_window_DX": "window_DX",
            "plot_optimal_SNR_by_window_SX": "window_SX",
            "plot_optimal_SNR_by_k": "k",
            "plot_optimal_SNR_by_saturation_DX": "saturation_DX",
            "plot_optimal_SNR_by_saturation_SX": "saturation_SX",
        },
        "condition_numbers": {
            "plot_condition_number_by_window_DX": "window_DX",
            "plot_condition_number_by_window_SX": "window_SX",
            "plot_condition_number_by_k": "k",
            "plot_condition_number_by_saturation_DX": "saturation_DX",
            "plot_condition_number_by_saturation_SX": "saturation_SX",
        }
    }

    datasets = {
        "mismatch": mismatch_data,
        "optimal_SNR": optimal_SNR_data,
        "condition_numbers": condition_numbers
    }

    for category, functions in plot_functions.items():
        data_dict = datasets[category]

        for func_name, x_key in functions.items():
            extracted_values = set()
            for key in data_dict.keys():
                param_mapping = {
                    "window_DX": key[0],
                    "window_SX": key[1],
                    "k": key[2],
                    "saturation_DX": key[3],
                    "saturation_SX": key[4],
                }
                if x_key in param_mapping:
                    extracted_values.add(param_mapping[x_key])

            # Only call the function if there are multiple unique values for the x-axis variable
            if len(extracted_values) > 1:
                plot_function = globals().get(func_name, None)
                if plot_function:
                    plot_function(data_dict, outdir, direction, M, dL, N_FFT)
                else:
                    print(f"Warning: Function {func_name} not found in global scope.")
