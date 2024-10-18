import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize    import fmin, minimize as min
from scipy.signal      import find_peaks

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

def acf_from_asd(asd_filepath, f_min, f_max, N_points):

    ''' 
        Compute the autocovariance function (ACF) from a given amplitude spectral density (ASD),
        given a frequency range and the number of points of the corresponding time array. 
        
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