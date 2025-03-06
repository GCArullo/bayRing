import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize    import fmin, minimize as min
from scipy.signal      import find_peaks
import os

# Constants
twopi = 2.*np.pi
c=2.99792458*10**8 #m/s
G=6.67259*1e-11 #N*m^2/kg
M_s=1.9885*10**30 #solar masses
Mpc = 3.0857*1e22 #Mpc in meters

# Conversions
C_mt=(M_s*G)/(c**3) #s, converts a mass expressed in solar masses into a time in seconds
C_md=(M_s*G)/(Mpc*c**2) #adimensional, converts a mass expressed in solar masses to a distance in Megaparsec

def amp_phase_from_re_im(h_re, h_im):

    """

    Compute the amplitude and phase of a waveform from its real and imaginary parts.

    Parameters
    ----------

    h_re : array
        Real part of the waveform.
    h_im : array
        Imaginary part of the waveform.

    Returns
    -------

    amp : array
        Amplitude of the waveform.
    phase : array   
        Phase of the waveform.

    """

    amp   = np.sqrt(h_re**2 + h_im**2)
    # Assuming the minus convention on the phase.
    phase = np.unwrap(np.angle(h_re - 1j*h_im))
    
    return amp, phase

def mismatch_waveforms(deltaT_deltaPhi, time, amp1, amp2, phase1, phase2, t1, t2):

    """

    Compute the mismatch between two waveforms.

    Parameters
    ----------

    deltaT_deltaPhi : array
        Array containing the time and phase shift between the two waveforms.
    time : array
        Time array for the waveforms.
    amp1 : array
        Amplitude of the first waveform.
    amp2 : array
        Amplitude of the second waveform.
    phase1 : array
        Phase of the first waveform.
    phase2 : array
        Phase of the second waveform.
    t1 : float
        Initial time for the mismatch computation.
    t2 : float
        Final time for the mismatch computation.

    Returns
    -------

    mismatch : float
        Mismatch between the two waveforms.

    """
    
    # Unpack the parameters to be optimized.
    deltaT, deltaPhi = deltaT_deltaPhi[0],deltaT_deltaPhi[1]

    # Cut the time array.
    mask_t   = np.logical_and(time>t1,time<t2)
    t_masked = time[mask_t]

    # Compute the norm of the two waveforms.
    norm1 = np.sum(np.abs(amp1(t_masked))**2)
    norm2 = np.sum(np.abs(amp2(t_masked-deltaT))**2)

    # Compute the numerator for the mismatch.
    num = np.real(np.sum(amp1(t_masked)*amp2(t_masked-deltaT)*np.exp(-1j*(phase1(t_masked) - phase2(t_masked-deltaT) - deltaPhi))))
    return 1.-num/np.sqrt(norm1*norm2)

def align_waveforms_with_mismatch(t_NR, NR_amp, NR_phi, t_2, NR_r_2, NR_i_2, t_min_mismatch, t_max_mismatch):

    """

    Align two waveforms using the mismatch between them.

    Parameters
    ----------

    t_NR : array
        Time array for the first NR waveform.
    NR_amp : array
        Amplitude of the first NR waveform.
    NR_phi : array
        Phase of the first NR waveform.
    t_2 : array
        Time array for the second NR waveform.
    NR_r_2 : array
        Real part of the second NR waveform.
    NR_i_2 : array
        Imaginary part of the second NR waveform.
    t_min_mismatch : float
        Initial time for the mismatch computation.
    t_max_mismatch : float
        Final time for the mismatch computation.

    Returns
    -------

    rough_deltaPhi_estimate_2 : float
        Rough estimate of the phase shift between the two waveforms.
    deltaT_estimate_2 : float
        Rough estimate of the time shift between the two waveforms.
    mismatch : float
        Mismatch between the two waveforms.

    """

    NR_amp_interp, NR_phi_interp     = interp1d(t_NR, NR_amp, fill_value=0.0, bounds_error=False), interp1d(t_NR, NR_phi, fill_value=0.0, bounds_error=False)

    # Amplitude and phase decomposition for NR simulation with different resolutions/extrapolation orders.
    NR_amp_2, NR_phi_2               = amp_phase_from_re_im(NR_r_2, NR_i_2)
    NR_amp_2_interp, NR_phi_2_interp = interp1d(t_2, NR_amp_2, fill_value=0.0, bounds_error=False), interp1d(t_2, NR_phi_2, fill_value=0.0, bounds_error=False)
    
    # Initial guess (used in the minimisation algorithm) of dephasing between different resolutions/extrapolation orders. Will use 0 for deltaT
    rough_deltaPhi_estimate_2        = NR_phi_interp(t_min_mismatch) - NR_phi_2_interp(t_min_mismatch)
    # Get deltaT and deltaPhi for alignment by minimising the mismatch.
    # THE DOCUMENTATION OF THIS FUNCTION SUCKS SO BADLY. To get the actual value of the mismatch, need to add: `full_output=True, disp=False)[1]` at the end of the line.
    deltaT_2, deltaPhi_2             = fmin(mismatch_waveforms, np.array([0.,rough_deltaPhi_estimate_2]), args=(t_NR, NR_amp_interp, NR_amp_2_interp, NR_phi_interp, NR_phi_2_interp, t_min_mismatch, t_max_mismatch), ftol=1e-15)
    
    # Align the waveforms.
    NR_cmplx_2_aligned               = NR_amp_2_interp(t_NR-deltaT_2) * np.exp(1j*(NR_phi_2_interp(t_NR-deltaT_2) + deltaPhi_2))
    NR_r_2_aligned, NR_i_2_aligned   = np.real(NR_cmplx_2_aligned), -np.imag(NR_cmplx_2_aligned)

    return NR_r_2_aligned, NR_i_2_aligned

def align_waveforms_at_peak(t_NR, NR_amp, NR_phi, t_res, NR_r_res, NR_i_res):

    """

    Align two waveforms at their peaks.

    Parameters
    ----------

    t_NR : array
        Time array for the first NR waveform.
    NR_amp : array
        Amplitude of the first NR waveform.
    NR_phi : array
        Phase of the first NR waveform.
    t_res : array
        Time array for the second NR waveform with a different resolution.
    NR_r_res : array
        Real part of the second NR waveform with a different resolution.
    NR_i_res : array
        Imaginary part of the second NR waveform with a different resolution.

    Returns
    -------

    NR_r_res_interp_aligned_time_phase : array
        Real part of the second NR waveform with a different resolution, aligned in time and phase with the first NR waveform.
    NR_i_res_interp_aligned_time_phase : array
        Imaginary part of the second NR waveform with a different resolution, aligned in time and phase with the first NR waveform.

    """

    # Different resolutions have different time axes, thus we need to reinterpolate before comparing.
    NR_r_res_interp, NR_i_res_interp     = interp1d(t_res, NR_r_res, fill_value=0.0, bounds_error=False)(t_NR), interp1d(t_res, NR_i_res, fill_value=0.0, bounds_error=False)(t_NR)
    NR_amp_res_interp, NR_phi_res_interp = amp_phase_from_re_im(NR_r_res_interp, NR_i_res_interp)

    # Align time axes
    idx_max_NR_amp                                                 = np.argmax(NR_amp)
    idx_max_NR_amp_res_interp                                      = np.argmax(NR_amp_res_interp)
    idx_roll_amp                                                   = idx_max_NR_amp - idx_max_NR_amp_res_interp
    NR_r_res_interp_aligned_time                                   = np.roll(NR_r_res_interp, idx_roll_amp)
    NR_i_res_interp_aligned_time                                   = np.roll(NR_i_res_interp, idx_roll_amp)
    NR_amp_res_interp_aligned_time, NR_phi_res_interp_aligned_time = amp_phase_from_re_im(NR_r_res_interp_aligned_time, NR_i_res_interp_aligned_time)

    # Align phases at peak
    NR_phi_res_interp_aligned_time_phase = NR_phi_res_interp_aligned_time - NR_phi_res_interp_aligned_time[idx_max_NR_amp] + NR_phi[idx_max_NR_amp]
    h                                    = NR_amp_res_interp_aligned_time * np.exp(1j*NR_phi_res_interp_aligned_time_phase)
    NR_r_res_interp_aligned_time_phase   = np.real( h)
    # This minus sign is connected to the minus sign in the phase.
    NR_i_res_interp_aligned_time_phase   = np.imag(-h)

    return NR_r_res_interp_aligned_time_phase, NR_i_res_interp_aligned_time_phase

def find_peak_time(t, A, ecc):

    """
    
    t : array
        Array containing the time samples.

    A : array
        Array containing the amplitude samples.

    ecc : float
        Eccentricity of the NR simulation.

    Returns
    -------

    t_peak : float
        Peak time of the amplitude.

    """

    # Find the peak time.
    if((ecc> 1e-3) and (ecc < 0.89)):
        try:
            peak_height_threshold = np.max(A) * 0.5
            peaks, _              = find_peaks(A, height=peak_height_threshold)
            merger_idx            = peaks[-1] # the merger is the last peak of A
            t_peak                = t[merger_idx]
            print('* Using the last relative maximum of the amplitude to define the peak of the waveform.')
        except:
            print('* No peaks found, using the maximum of the amplitude to define the peak of the waveform.')
            t_peak                = t[np.argmax(A)]
    else:
        print('* Using the maximum of the amplitude to define the peak of the waveform.')
        t_peak                    = t[np.argmax(A)]

    return t_peak

def compute_condition_number(acf):
    """
    Compute the condition number of a Toeplitz matrix derived from the ACF.

    Parameters:
        acf (np.ndarray): Autocorrelation function (ACF).

    Returns:
        float: Condition number of the Toeplitz matrix.
    """
    from scipy.linalg import toeplitz
    toeplitz_matrix = toeplitz(acf)
    return np.linalg.cond(toeplitz_matrix)

def extract_NR_params(NR_sim, M):
    """
    Extracts key time-related parameters from an NR_sim object.

    Parameters:
        NR_sim (object): An object containing the attributes:
            - t_peak
            - t_NR_cut (list or array)
            - NR_r_cut (list or array)

    Returns:
        tuple: (t_start_g, t_end_g, t_NR_s, NR_length)
    """
    t_peak = NR_sim.t_peak
    t_NR_cut = NR_sim.t_NR_cut
    t_start_g, t_end_g = t_NR_cut[0] - t_peak, t_NR_cut[-1] - t_peak
    NR_length = len(NR_sim.NR_r_cut)
    t_NR_s = (t_NR_cut - t_peak - t_start_g) * M * C_mt

    print("\nEstimated starting and end times and NR simulation lenght")
    print("t_start={0:.1f}M, t_end={1:.1f}M, NR_length={2:.0f}\n".format(t_start_g, t_end_g, NR_length))

    return t_start_g, t_end_g, t_NR_s, NR_length

def extract_GW_parameters(parameters):
    """Extract GW parameters for mismatch computation."""
    return (
        parameters['Mismatch']['M'], parameters['Mismatch']['dL'],
        parameters['Mismatch']['ra'], parameters['Mismatch']['dec'], parameters['Mismatch']['psi']
    )

def extract_and_compute_psd_parameters(asd_path, psd_dict):
    """
    Load the PSD file, extract key frequency parameters, and compute window properties.
    
    Parameters:
        asd_path (str): Path to the ASD file.
        f_sample (float): Sampling frequency in Hz.
        psd (dict): Dictionary containing window properties.
    
    Returns:
        tuple: (f_min, f_max, dt, df, N_psd, n_iterations, window_sizes, steepness_values, saturation_DX_values, saturation_SX_values)
            - f_min (float): Minimum frequency in Hz.
            - f_max (float): Maximum frequency in Hz.
            - dt (float): Time resolution (1 / 2*f_max).
            - df (float): Frequency resolution (minimum difference between consecutive frequencies).
            - N_points (int): Number of PSD points (f_sample / df).
            - n_iterations (int): Number of FFT iterations from psd dictionary.
            - window_sizes (list): Smoothed window sizes.
            - steepness_values (list): Logarithmic values for steepness.
            - saturation_DX_values (list): Logarithmic values for saturation_DX.
            - saturation_SX_values (list): Logarithmic values for saturation_SX.
    """
    try:
        # Load the frequency data from the ASD file
        freq_file, _ = np.loadtxt(asd_path, unpack=True)
        
        # Extract PSD parameters
        f_min, f_max = np.min(freq_file), np.max(freq_file)
        f_sample = 2 * f_max
        dt = 1 / f_sample
        df = np.min(np.diff(freq_file))
        N_points = int(f_sample / df)
        n_FFT_points = psd_dict['n_FFT_points']
        
        # Print extracted values
        print("\nASD file parameters")
        print(f"f_min={f_min:.0f}Hz, f_max={f_max:.0f}Hz, dt={dt:.6f}s, df={df:.4f}Hz, N_points={N_points}\n")
        
        # Compute window properties
        window_sizes = np.linspace(psd_dict['window'], psd_dict['window_max'], psd_dict['n_window']).tolist()
        steepness_values = np.logspace(np.log10(psd_dict['steepness']), np.log10(psd_dict['steepness_max']), psd_dict['n_steepness']).tolist()
        saturation_DX_values = np.logspace(np.log10(psd_dict['saturation_DX']), np.log10(psd_dict['saturation_DX_max']), psd_dict['n_saturation_DX']).tolist()
        saturation_SX_values = np.logspace(np.log10(psd_dict['saturation_SX']), np.log10(psd_dict['saturation_SX_max']), psd_dict['n_saturation_SX']).tolist()
        
        return f_min, f_max, dt, df, N_points, n_FFT_points, window_sizes, steepness_values, saturation_DX_values, saturation_SX_values
    
    except Exception as e:
        print(f"Error processing PSD file {asd_path}: {e}")
        return None, None, None, None, None, None, None, None, None, None

def acf_from_asd(asd_filepath, f_min, f_max, N_points):

    ''' 
        OLD VERSION
        Compute the autocovariance function (ACF) from a given amplitude spectral density (ASD),
        given a frequency range and the number of points of the corresponding time array. 
        Note that this function was rewritten with the smoothing implementation at the edges of the PSD.

        Parameters
        ----------
        
        f_min : float
            Minimum frequency to consider.
        f_max : float
            Maximum frequency to consider.
        n_points : int
            Number of points of the time array.
        asd_filepath : str
            Path to the ASD file.
        
        Returns
        -------

        ACF : array
            Autocovariance function.

     '''
    
    # Frequency axis construction
    s_rate = f_max * 2
    dt     = 1./s_rate
    df     = s_rate/N_points
    f      = np.fft.rfftfreq(N_points, d=dt)

    # Load ASD file and convert it to PSD
    freq_file, asd_file = np.loadtxt(asd_filepath, unpack=True)
    psd_file            = asd_file**2                          

    # Interpolate the PSD to the desired frequency range
    psd_file  =  psd_file[freq_file > f_min]
    freq_file = freq_file[freq_file > f_min]
    psd_file  =  psd_file[freq_file < f_max]
    freq_file = freq_file[freq_file < f_max]

    print('\n\n\nFIXME: edges should go to the same constant when taking the FT\n\n\n')
    PSD       = np.interp(f, freq_file, psd_file)

    # Compute the ACF. We are using the one-sided PSD, thus it is twice the Fourier transform of the autocorrelation function (see eq. 7.15 of Maggiore Vol.1). We take the real part just to convert the complex output of fft to a real numpy float. The imaginary part if already 0 when coming out of the fft.
    ACF = 0.5 * np.real(np.fft.irfft(PSD*df)) * N_points 

    return ACF

def apply_smoothing(frequencies, values, f_anchor_l, f_anchor_h, saturation_DX, saturation_SX, k, window_size, direction):
    """
    Apply smoothing saturation to specified frequency ranges.

    Parameters:
        frequencies (np.ndarray): Array of frequency values.
        values (np.ndarray): Array of corresponding values (e.g., PSD values).
        f_anchor_l (float): Low anchor frequency for smoothing (for 'below' or 'below-and-above').
        f_anchor_h (float): High anchor frequency for smoothing (for 'above' or 'below-and-above').
        saturation (float): Saturation value for the target smoothing.
        k (float): Smoothing steepness parameter (controls the exponential decay).
        window_size (float): Smoothing window size around anchor frequencies.
        direction (str): Direction of smoothing ('below', 'above', or 'below-and-above').

    Returns:
        np.ndarray: Smoothed values. If `window_size` is 0, returns the original values.
    """

    # Return early if window_size is 0 to avoid unnecessary computations
    if window_size == 0:
        return values

    def smoothing_function(frequencies, values, f_anchor, window_size, target_value, k, is_above):
        """
        Function to apply smoothing to a specified frequency range (left or right).

        Parameters:
            frequencies (np.ndarray): Frequency array.
            values (np.ndarray): Corresponding values array.
            f_anchor (float): Anchor frequency.
            window_size (float): Smoothing window size.
            target_value (float): Target value for the smoothed region.
            k (float): Smoothing steepness parameter.
            is_above (bool): If True, applies smoothing from above; otherwise, from below.

        Returns:
            np.ndarray: Smoothed values for the selected range.
        """
        if is_above:
            smooth_range = frequencies[(frequencies >= f_anchor - window_size) & (frequencies <= f_anchor)]
            indices = np.where((frequencies >= f_anchor - window_size) & (frequencies <= f_anchor))
            smoothing_factor = 1 - np.exp(-(smooth_range - (f_anchor - window_size)) * k)
            s_norm = 1 - np.exp(- (window_size) * k)
        else:
            smooth_range = frequencies[(frequencies >= f_anchor) & (frequencies <= f_anchor + window_size)]
            indices = np.where((frequencies >= f_anchor) & (frequencies <= f_anchor + window_size))
            smoothing_factor = 1 - np.exp((smooth_range - (f_anchor + window_size)) * k)
            s_norm = 1 - np.exp(- (window_size) * k)

        # Apply the smoothing formula
        values[indices] = values[indices] * (1 - smoothing_factor) + target_value * smoothing_factor / s_norm
        return values

    # Apply smoothing for 'below', 'above', or 'below-and-above'
    if direction == 'below':
        target_value_l = saturation_DX*values[0]
        values = smoothing_function(frequencies, values, f_anchor_l, window_size, target_value_l, k, is_above=False)

    elif direction == 'above':
        target_value_h = saturation_SX*values[-1]
        values = smoothing_function(frequencies, values, f_anchor_h, window_size, target_value_h, k, is_above=True)

    elif direction == 'below-and-above':
        target_value_l = saturation_DX*values[0]
        target_value_h = saturation_SX*values[-1]
        # Left smoothing
        values = smoothing_function(frequencies, values, f_anchor_l, window_size, target_value_l, k, is_above=False)
        # Right smoothing
        values = smoothing_function(frequencies, values, f_anchor_h, window_size, target_value_h, k, is_above=True)

    else:
        raise ValueError("Invalid direction. Choose between 'below', 'above', or 'below-and-above'.")

    return values

def apply_C1(frequencies, values, f_start, window_size):
    """
    Applies moving average in order to transform the acf to C1.

    Parameters:
    - frequencies: Array of frequency values.
    - values: Array of corresponding PSD values.
    - f_start: Start of the range where the transition begins.
    - output_file: Path to save the output data.

    Returns:
    - Modified values after applying the transition.
    """

    # Define the range for concavity control
    C1_window=window_size*2
    f_up=f_start+C1_window
    f_down=f_start-C1_window
    indices = np.where((frequencies >= f_down) & (frequencies <= f_up))

    # Initialize the transition array
    transition = values[indices]

    N=17
    for n in range(0,N,1):

        # Apply mild concavity by averaging three consecutive elements
        len_indices=indices[0][:-2]
        for idx in len_indices:  # Exclude the last two indices to avoid out-of-bounds
            transition[np.where(indices[0] == idx)] = (
                values[idx] + values[idx+1] + values[idx+2]) / 3    

        # Replace values in the specified range with the transition array
        values[indices] = transition

    return values

def acf_from_asd_with_smoothing(asd_path, f_min, f_max, N_points, window_size, k, saturation_DX, saturation_SX, direction, C1_flag):
    """
    Compute the ACF from the ASD with smoothing applied.

    Parameters:
        asd_path (str): Path to the ASD file.
        f_min (float): Minimum frequency.
        f_max (float): Maximum frequency.
        N_points (int): Number of frequency points.
        window_size (float): Smoothing window size.
        k (float): Smoothing steepness parameter.
        saturation (float): Saturation value for smoothing target.
        direction (str): 'below' for smoothing near f_min, 'above' for smoothing near f_max.

    Returns:
        np.ndarray: Smoothed PSD and ACF.
    """

    # Load ASD file and convert to PSD
    freq_file, asd_file = np.loadtxt(asd_path, unpack=True)
    psd_file = asd_file ** 2

    # Sampling settings
    s_rate = 2 * f_max
    dt = 1. / s_rate
    df = s_rate / N_points
    f = np.fft.rfftfreq(N_points, d=dt)

    # Filter and interpolate PSD within (f_min, f_max)
    mask = (freq_file > f_min) & (freq_file < f_max)
    freq_file = freq_file[mask]
    psd_file = psd_file[mask]
    PSD_band = np.interp(f, freq_file, psd_file)

    # Apply smoothing
    smoothed_PSD = PSD_band.copy()
    smoothed_PSD = apply_smoothing(f, smoothed_PSD, f_min, f_max, saturation_DX, saturation_SX, k, window_size, direction)

    # Extend PSD for f < f_min
    f_below_min = f[f < f_min]
    if len(f_below_min) > 0:
        PSD_below_min = np.full_like(f_below_min, saturation_DX*smoothed_PSD[0])
        smoothed_PSD[:len(f_below_min)] = PSD_below_min

    #-----------------------------------------------------C^1 fixing------------------------------------------------------------#

    if C1_flag==True:
        
        if direction=='below':

            # Apply the C1 fixing near f_min + window_low
            f_transition_start = f_min + window_size
            smoothed_PSD = apply_C1(f, smoothed_PSD, f_transition_start, window_size)

        elif direction=='above':

            # Apply the C1 fixing near f_max - window_low
            f_transition_start = f_max - window_size
            smoothed_PSD = apply_C1(f, smoothed_PSD, f_transition_start, window_size)

        elif direction=='below-and-above':

            # Apply the C1 fixing near f_min + window_low
            f_transition_start = f_min + window_size
            smoothed_PSD = apply_C1(f, smoothed_PSD, f_transition_start, window_size)

            # Apply the C1 fixing near f_max - window_low
            f_transition_start = f_max - window_size
            smoothed_PSD = apply_C1(f, smoothed_PSD, f_transition_start, window_size)

    # ACF for smoothed PSD
    ACF_smoothed = 0.5 * np.real(np.fft.irfft(smoothed_PSD * df)) * N_points

    return smoothed_PSD, ACF_smoothed
