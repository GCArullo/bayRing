import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize    import fmin, minimize as min
from scipy.signal      import find_peaks
import os

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


def acf_from_asd(asd_filepath, f_min, f_max, N_points):

    ''' 
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

def apply_smoothing(frequencies, values, f_anchor_l, f_anchor_h, multiplier_factor, k, window_size, direction):
    """
    Apply smoothing saturation to specified frequency ranges.

    Parameters:
        frequencies (np.ndarray): Array of frequency values.
        values (np.ndarray): Array of corresponding values (e.g., PSD values).
        f_anchor_l (float): Low anchor frequency for smoothing (for 'below' or 'below-and-above').
        f_anchor_h (float): High anchor frequency for smoothing (for 'above' or 'below-and-above').
        multiplier_factor (float): Multiplier factor for the target smoothing value.
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
        target_value_l = values[0] * multiplier_factor
        values = smoothing_function(frequencies, values, f_anchor_l, window_size, target_value_l, k, is_above=False)

    elif direction == 'above':
        target_value_h = values[-1] * multiplier_factor
        values = smoothing_function(frequencies, values, f_anchor_h, window_size, target_value_h, k, is_above=True)

    elif direction == 'below-and-above':
        target_value_l = values[0] * multiplier_factor
        target_value_h = values[-1] * multiplier_factor
        # Left smoothing
        values = smoothing_function(frequencies, values, f_anchor_l, window_size, target_value_l, k, is_above=False)
        # Right smoothing
        values = smoothing_function(frequencies, values, f_anchor_h, window_size, target_value_h, k, is_above=True)

    else:
        raise ValueError("Invalid direction. Choose between 'below', 'above', or 'below-and-above'.")

    return values

def apply_C1(frequencies, values, f_start):
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

    # Define the range for concavity control -> C1_window is fixed to 4Hz
    C1_window=4
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

def acf_from_asd_with_smoothing(asd_path, f_min, f_max, N_points, window_size, k, multiplier_factor, direction, C1_flag):
    """
    Compute the ACF from the ASD with smoothing applied.

    Parameters:
        asd_path (str): Path to the ASD file.
        f_min (float): Minimum frequency.
        f_max (float): Maximum frequency.
        N_points (int): Number of frequency points.
        window_size (float): Smoothing window size.
        k (float): Smoothing steepness parameter.
        multiplier_factor (float): Multiplier factor for smoothing.
        direction (str): 'below' for smoothing near f_min, 'above' for smoothing near f_max.

    Returns:
        np.ndarray: Smoothed PSD and ACF.
    """

    # Load ASD file and convert to PSD
    freq_file, asd_file = np.loadtxt(asd_path, unpack=True)
    psd_file = asd_file ** 2

    # Sampling settings
    s_rate = f_max * 2
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
    smoothed_PSD = apply_smoothing(f, smoothed_PSD, f_min, f_max, multiplier_factor, k, window_size, direction)

    # Extend PSD for f < f_min
    f_below_min = f[f < f_min]
    if len(f_below_min) > 0:
        PSD_below_min = np.full_like(f_below_min, PSD_band[0] * multiplier_factor)
        smoothed_PSD[:len(f_below_min)] = PSD_below_min

    #-----------------------------------------------------C^1 fixing------------------------------------------------------------#

    if C1_flag==True:
        
        if direction=='below':

            # Apply the C1 fixing near f_min + window_low
            f_transition_start = f_min + window_size
            smoothed_PSD = apply_C1(f, smoothed_PSD, f_transition_start)

        elif direction=='above':

            # Apply the C1 fixing near f_max - window_low
            f_transition_start = f_max - window_size
            smoothed_PSD = apply_C1(f, smoothed_PSD, f_transition_start)

        elif direction=='below-and-above':

            # Apply the C1 fixing near f_min + window_low
            f_transition_start = f_min + window_size
            smoothed_PSD = apply_C1(f, smoothed_PSD, f_transition_start)

            # Apply the C1 fixing near f_max - window_low
            f_transition_start = f_max - window_size
            smoothed_PSD = apply_C1(f, smoothed_PSD, f_transition_start)

    # ACF for smoothed PSD
    ACF_smoothed = 0.5 * np.real(np.fft.irfft(smoothed_PSD * df)) * N_points

    return smoothed_PSD, ACF_smoothed
