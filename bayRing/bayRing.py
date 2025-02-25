#! /usr/bin/env python

# Standard python packages
import matplotlib.pyplot as plt, numpy as np, os, time, traceback
from scipy.interpolate import interp1d, CubicSpline
from optparse       import OptionParser
try:                import configparser
except ImportError: import ConfigParser as configparser
import cpnest, cpnest.model

# Package internal imports
import bayRing.NR_waveforms       as NR_waveforms
import bayRing.postprocess        as postprocess
import bayRing.initialise         as initialise
import bayRing.QNM_utils          as QNM_utils
import bayRing.inference          as inference
import bayRing.template_waveforms as template_waveforms
import bayRing.waveform_utils     as wf_utils

from pyRing.utils import print_section

import scipy.linalg as sl

#constants
twopi = 2.*np.pi
c=2.99792458*10**8 #m/s
G=6.67259*1e-11
M_s=1.9885*10**30 #solar masses
C_mt=(M_s*G)/c**3 #s, converts a mass expressed in solar masses into a time in seconds

if __name__=='__main__':
    main()

def main():

    # ==================================================#
    # Initialize execution and read configuration file. #
    # ==================================================#

    # Print ascii art.
    try   : print("\u001b[\u001b[38;5;39m{}\u001b[0m".format(initialise.my_art))
    except: pass
    print(initialise.__ascii_art__)
    
    # Initialise and read config.
    execution_time = time.time()
    parser         = OptionParser(initialise.usage)
    parser.add_option('--config-file', type='string', metavar = 'config_file', default = None)
    (opts,args)    = parser.parse_args()
    config_file    = opts.config_file

    if not config_file:
        parser.print_help()
        parser.error('Please specify a config file.')
    if not os.path.exists(config_file): parser.error('Config file {} not found.'.format(config_file))
    Config = configparser.ConfigParser()
    Config.read(config_file)

    print_section('Input parameters')
    print(('* Reading config file : `{}`.'.format(config_file)))
    print( '* With sections       : {}.\n'.format(str(Config.sections())))
    print( '* I\'ll be running with the following values:\n')

    # ===================================================#
    # Read input parameters from the configuration file. #
    # ===================================================#

    parameters = initialise.read_config(Config)

    # =================#
    # Set output dirs. #
    # =================#

    initialise.set_output(parameters['I/O']['outdir'], parameters['I/O']['screen-output'], parameters['Inference']['method'], config_file, parameters['I/O']['run-type'])

    # ==============#
    # Load NR data. #
    # ==============#

    print_section('NR data loading')
    parameters['Injection-data']['modes-list'] = NR_waveforms.read_fake_NR(parameters['NR-data']['catalog'], parameters['Injection-data']['modes'])

    #NR simulation object
    NR_sim      = NR_waveforms.NR_simulation(parameters['NR-data']['catalog']                       , 
                                             parameters['NR-data']['ID']                            , 
                                             parameters['NR-data']['res-level']                     , 
                                             parameters['NR-data']['extrap-order']                  , 
                                             parameters['NR-data']['pert-order']                    , 
                                             parameters['NR-data']['dir']                           , 
                                             parameters['NR-data']['properties-file']               ,
                                             parameters['Injection-data']['modes-list']             , 
                                             parameters['Injection-data']['times']                  , 
                                             parameters['Injection-data']['noise']                  , 
                                             parameters['Injection-data']['tail']                   , 
                                             parameters['NR-data']['l-NR']                          , 
                                             parameters['NR-data']['m']                             , 
                                             parameters['I/O']['outdir']                            ,
                                             waveform_type  = parameters['NR-data']['waveform-type'], 
                                             download       = parameters['NR-data']['download']     , 
                                             NR_error       = parameters['NR-data']['error']        , 
                                             tM_start       = parameters['Inference']['t-start']    , 
                                             tM_end         = parameters['Inference']['t-end']      , 
                                             t_delay_scd    = parameters['Inference']['dt-scd']     , 
                                             t_peak_22      = parameters['NR-data']['t-peak-22']    ,
                                             t_min_mismatch = parameters['NR-data']['error-t-min']  , 
                                             t_max_mismatch = parameters['NR-data']['error-t-max']  )

    error       = NR_sim.NR_cpx_err_cut
    NR_metadata = NR_waveforms.read_NR_metadata(NR_sim, parameters['NR-data']['catalog'])

    print_section('Simulation metadata')
    for key in NR_metadata.keys(): print('{}: {}'.format(key.ljust(len('omg_peak_22')), NR_metadata[key]))

    # =================#
    # Load Kerr modes. #
    # =================#
    
    Kerr_modes, Kerr_quad_modes, qnm_cached = QNM_utils.read_Kerr_modes(parameters['Model']['QNM-modes'], parameters['Model']['QQNM-modes'], parameters['Model']['charge'], parameters['NR-data']['l-NR'], parameters['NR-data']['m'], NR_metadata)
    Kerr_tail_modes                         = QNM_utils.read_tail_modes(parameters['Model']['Kerr-tail-modes'])

    # ============#
    # Load model. #
    # ============#

    wf_model = template_waveforms.WaveformModel(NR_sim.t_NR_cut                                                            , 
                                                NR_sim.t_min                                                               , 
                                                NR_sim.t_peak                                                              ,
                                                parameters['Model']['template']                                            , 
                                                parameters['Model']['N-DS-modes']                                          , 
                                                Kerr_modes                                                                 , 
                                                NR_metadata                                                                , 
                                                qnm_cached                                                                 , 
                                                parameters['NR-data']['l-NR']                                              , 
                                                parameters['NR-data']['m']                                                 , 
                                                tail                      = parameters['Model']['Kerr-tail']                   ,
                                                tail_modes                = Kerr_tail_modes                                    ,     
                                                quadratic_modes           = Kerr_quad_modes                                    , 
                                                const_params              = parameters['NR-data']['add-const']                 , 
                                                KerrBinary_version        = parameters['Model']['KerrBinary-version']              ,
                                                KerrBinary_amp_nc_version = parameters['Model']['KerrBinary-amplitudes-nc-version'],
                                                TEOB_NR_fit               = parameters['Model']['TEOB-NR-fit']                 ,
                                                TEOB_template             = parameters['Model']['TEOB-template']               ,
                                                )

    # ===============#
    # Set inference. #
    # ===============#

    if(  parameters['Inference']['sampler']=='raynest'): 
        import raynest, raynest.model
        InferenceModel = inference.Dynamic_InferenceModel(raynest.model.Model)
    elif(parameters['Inference']['sampler']=='cpnest' ): 
        InferenceModel = inference.Dynamic_InferenceModel( cpnest.model.Model)
    else                                               : 
        raise ValueError("Unknown sampler.")

    inference_model = InferenceModel(NR_sim.NR_cpx_cut                                    , 
                                     error                                                , 
                                     wf_model                                             , 
                                     Config                                               ,
                                     parameters['Inference']['method']                    , 
                                     parameters['Inference']['min-method']                , 
                                     likelihood_kind=parameters['Inference']['likelihood'])

    tail_flag = wf_model.wf_model=='Kerr' and wf_model.tail==1
    # Plot and terminate execution if plotting only.
    if(parameters['I/O']['run-type']=='plot-NR-only'): 
        postprocess.plot_NR_vs_model(NR_sim, wf_model, NR_metadata, None, None, parameters['I/O']['outdir'], None, tail_flag)
        # In case a tail run is selected, do plots also without tail format
        if(tail_flag): postprocess.plot_NR_vs_model(NR_sim, wf_model, NR_metadata, None, None, parameters['I/O']['outdir'], None, False)
        print('\n* NR-only plotting run-type selected. Exiting.\n')
        exit()

    print_section('Inference')

    #==============================#
    # Inference execution section. #
    #==============================#
    
    if(  parameters['I/O']['run-type']=='full'           ): results_object = inference.run_inference(parameters, inference_model)
    elif(parameters['I/O']['run-type']=='post-processing'): results_object = postprocess.read_results_object_from_previous_inference(parameters)
    else                                                  : raise Exception("Unknown run type selected. Exiting.")
        
    #=========================#
    # Postprocessing section. #
    #=========================#

    print_section('Post-processing')

    """
    #print info
    print('results_object: ', results_object)
    print('results_object len: ', len(results_object))
    print('inference_model: ', inference_model)
    """

    print('\n* Note: except for free damped sinusoids fits, quantities are quoted at the selected peak time.\n')
    postprocess.print_point_estimate(results_object, inference_model.access_names(), parameters['Inference']['method'])
    postprocess.l2norm_residual_vs_nr(results_object, inference_model, NR_sim, parameters['I/O']['outdir'])

    # postprocess.plot_fancy_residual(NR_sim, wf_model, NR_metadata, results_object, inference_model, parameters['I/O']['outdir'], parameters['Inference']['method'])
    # postprocess.plot_fancy_reconstruction(NR_sim, wf_model, NR_metadata, results_object, inference_model, parameters['I/O']['outdir'], method)

    # Not needed now that we define everything directly at the peak.
    # if(parameters['Model']['template']=='Kerr'): postprocess.post_process_amplitudes(parameters['Inference']['t-start'], results_object, NR_metadata, qnm_cached, Kerr_modes, Kerr_quad_modes, parameters['I/O']['outdir'])
    if(parameters['NR-data']['catalog']=='C2EFT' and 'Damped-sinusoids' in parameters['Model']['template']): postprocess.compare_with_GR_QNMs(results_object, qnm_cached, NR_sim, parameters['I/O']['outdir'])

    if(parameters['I/O']['run-type']=='full'):
    
        if(parameters['Inference']['method']=='Nested-sampler'):
            os.system('mv {dir}/Algorithm/posterior*.pdf {dir}/Plots/Results/.'.format(dir = parameters['I/O']['outdir']))
            if(  parameters['Inference']['sampler']=='raynest'): os.system('mv {dir}/Algorithm/*trace.png   {dir}/Plots/Chains/.'.format(dir = parameters['I/O']['outdir']))
            elif(parameters['Inference']['sampler']=='cpnest' ): os.system('mv {dir}/Algorithm/nschain*.pdf {dir}/Plots/Chains/.'.format(dir = parameters['I/O']['outdir']))
            
        execution_time = (time.time() - execution_time)/60.0
        print('\nExecution time (min): {:.2f}\n'.format(execution_time))


    try   : 
        postprocess.plot_NR_vs_model(NR_sim, wf_model, NR_metadata, results_object, inference_model, parameters['I/O']['outdir'], parameters['Inference']['method'], tail_flag)
        # In case a tail run is selected, do plots also without tail format
        if(tail_flag): postprocess.plot_NR_vs_model(NR_sim, wf_model, NR_metadata, results_object, inference_model, parameters['I/O']['outdir'], parameters['Inference']['method'], False    )
    except Exception as e:
        print(f"Waveform reconstruction plot failed with error: {e}")
        traceback.print_exc()    

    #===============================#
    # Mismatch computation section. #
    #===============================#

    # Initialize dictionaries
    psd_data = {}
    acf_data = {}
    mismatch_data = {}
    optimal_SNR_data = {}

    # Mass and distance (for the mismatch)
    M = parameters['Mismatch']['M']
    dL = parameters['Mismatch']['dL']

    # Extract t-peak, t_start and t_end
    t_peak = NR_sim.t_peak
    t_NR_cut = NR_sim.t_NR_cut
    t_start_g, t_end_g = t_NR_cut[0]-t_peak, t_NR_cut[-1]-t_peak

    print("t_start_g={0:.4f}M, t_end_g={1:.4f}M".format(t_start_g, t_end_g))

    # Compute start and end time in physical units
    t_start = t_start_g * C_mt * M
    t_end = t_end_g * C_mt * M

    # PSD loading and extrema
    psd = parameters['PSD-settings']
    asd_path = psd['asd-path']
    f_sample = 2048*2
    f_min, f_max, dt, df = postprocess.extract_psd_parameters(asd_path, f_sample)

    # Window properties for f_min and f_max
    window_min, window_max, n_window = psd['window_min'], psd['window_max'], psd['n_window']
    steepness_min, steepness_max, n_steepness = psd['steepness_min'], psd['steepness_max'], psd['n_steepness']
    saturation_min, saturation_max, n_saturation = psd['saturation_min'], psd['saturation_max'], psd['n_saturation']

    # Smoothed window parameters
    window_sizes = np.linspace(window_min, window_max, n_window).tolist()
    steepness_values = np.logspace(np.log10(steepness_min), np.log10(steepness_max), n_steepness).tolist()
    saturation_values = np.logspace(np.log10(saturation_min), np.log10(saturation_max), n_saturation).tolist()
    
    # Flags (to change)
    flags = parameters['Flags']
    check_TD_FD = False
    C1_choice = True
    sanity_check_mm = False
    print(check_TD_FD, C1_choice, sanity_check_mm)

    # Number of points for PSD/ACF
    N_sim = len(NR_sim.NR_r_cut)
    N_psd = int(2*f_max/df)
    n_iterations = psd['n_FFT_points']

    # Choose if iterate or not on N_FFT
    if n_iterations==1:
        N_FFT = [N_psd]
    else:    
        N_FFT = np.linspace(N_sim, N_psd, n_iterations)

    # Define the directory path
    smoothing_paths = ["Left_smoothing", "Right_smoothing", "Both_edges_smoothing"]
    for smoothing_path in smoothing_paths:
        algorithm_dir = os.path.join(parameters['I/O']['outdir'], "Algorithm", smoothing_path)

        # Clear it before plotting
        postprocess.clear_directory(algorithm_dir)

    #"""
    # Iterate over the number of FFT points
    for N_fft in N_FFT:

        #convert to integer
        N_fft=int(N_fft)

        # Iterate over the smoothing parameters and compute ACF
        for window_size, k, saturation in [(w, s, t) for w in window_sizes for s in steepness_values for t in saturation_values]:
                
            # Consistency check on starting time, end time and f_min 
            if (t_end-t_start)>1/(f_min+window_size):
                print("Please provide (t_end-t_start)<1/(f_min+window_size).")
                exit()

            try:
                
                # Compute ACF with smoothing
                PSD_smoothed, ACF_smoothed = wf_utils.acf_from_asd_with_smoothing(
                    asd_path,
                    f_min, f_max,
                    N_fft,
                    window_size=window_size,
                    k=k,
                    saturation=saturation,
                    direction=psd['direction'],
                    C1_flag=C1_choice
                )

                # Store smoothed PSD/ACF data in dictionaries
                psd_data[f"window size={round(window_size,1)}Hz, k={k}, {psd['direction']}, N_FFT={N_fft}"] = PSD_smoothed
                acf_data[f"window size={round(window_size,1)}Hz, k={k}, {psd['direction']}, N_FFT={N_fft}"] = ACF_smoothed

                #-------------------------------------------------- Mismatch computation -------------------------------------------------------#

                # Compute total duration and time array
                T = N_fft * dt
                t_ACF = np.linspace(0, T, len(ACF_smoothed))

                # Truncate ACF to ringdown analysis lenght
                t_trunc, ACF_trunc = postprocess.truncate_and_interpolate_acf(t_ACF, ACF_smoothed, t_start, t_end, N_sim)

                # Call compute_mismatch with the subsampled smoothed ACF
                postprocess.compute_mismatch(
                    NR_sim, 
                    results_object, 
                    inference_model, 
                    parameters['I/O']['outdir'],
                    parameters['Inference']['method'], 
                    ACF_trunc, N_fft,
                    M, dL,
                    t_start_g, t_end_g,
                    f_min, f_max,
                    asd_path,
                    window_size, k,
                    check_TD_FD,
                    sanity_check_mm
                )

                # Read mismatch results from file
                mismatch_filename = f"Mismatch_M_{M}_dL_{dL}_t_s_{t_start_g}M_w_{round(window_size,1)}_k_{round(k,2)}_NFFT_{N_fft}.txt"
                mismatch_file = os.path.join(parameters['I/O']['outdir'], 'Algorithm', mismatch_filename)

                with open(mismatch_file, 'r') as f:
                    lines = f.readlines()[1:]  # Skip the header

                # Store mismatch results in mismatch_data (consider only real and imaginary for simplicity)
                mismatch_data[(window_size, k, saturation)] = {'real': {}, 'imaginary': {}}
                for line in lines:
                    perc, component, mismatch = line.strip().split('\t')
                    perc = int(perc)
                    mismatch_data[(window_size, k, saturation)][component][perc] = float(mismatch)

                #-------------------------------------------------- optimal SNR computation -------------------------------------------------------#

                # Plot ACF
                postprocess.plot_acf_interpolated(t_ACF, t_trunc, 
                                                  ACF_smoothed, ACF_trunc, 
                                                  parameters['I/O']['outdir'], 
                                                  window_size, k,
                                                  saturation,
                                                  psd['direction'])

                # Call compute_mismatch with the subsampled smoothed ACF
                postprocess.compute_optimal_SNR(
                    NR_sim, 
                    results_object, 
                    inference_model, 
                    parameters['I/O']['outdir'],
                    parameters['Inference']['method'], 
                    ACF_trunc,
                    N_fft,
                    M, dL,
                    t_start_g, t_end_g,
                    f_min, f_max,
                    asd_path,
                    window_size, k,
                    check_TD_FD
                )

                # Read optimal SNR results from file
                optimal_SNR_filename = f"Optimal_SNR_M_{M}_dL_{dL}_t_s_{t_start_g}M_w_{round(window_size,1)}_k_{round(k,2)}_NFFT_{N_fft}.txt"
                optimal_SNR_file = os.path.join(parameters['I/O']['outdir'], 'Algorithm', optimal_SNR_filename)
                with open(optimal_SNR_file, 'r') as f:
                    lines = f.readlines()[1:]  # Skip the header

                # Store mismatch results in optimal_SNR_data (consider only real and imaginary for simplicity)
                optimal_SNR_data[(window_size, k, saturation)] = {'real': {}, 'imaginary': {}}
                for line in lines:
                    perc, component, optimal_SNR = line.strip().split('\t')
                    perc = int(perc)
                    optimal_SNR_data[(window_size, k, saturation)][component][perc] = float(optimal_SNR)

            except Exception as e:
                print(f"Optimal SNR computation failed for window_size={window_size}, k={k}: {e}")

    #----------------------------------------------------------------------------------- Postprocessing --------------------------------------------------------------------------------------------------------------------------#

    # Postprocess plots
    postprocess.plot_psd_near_fmin_fmax(psd_data, f_min, f_max, window_size, parameters['I/O']['outdir'], psd['direction'])
    postprocess.plot_psd_and_acf(psd_data, acf_data, f_min, f_max, t_start, t_end, parameters['I/O']['outdir'], psd['direction'], window_size)
    postprocess.plot_mismatch_by_window(mismatch_data, parameters['I/O']['outdir'], psd['direction'], M, dL, N_FFT)
    postprocess.plot_optimal_SNR_by_window(optimal_SNR_data, parameters['I/O']['outdir'], psd['direction'], M, dL, N_FFT)
    postprocess.plot_mismatch_by_k(mismatch_data, parameters['I/O']['outdir'], psd['direction'], M, dL, N_FFT)
    postprocess.plot_mismatch_by_saturation(mismatch_data, parameters['I/O']['outdir'], psd['direction'], M, dL, N_FFT)
    postprocess.plot_optimal_SNR_by_k(optimal_SNR_data, parameters['I/O']['outdir'], psd['direction'], M, dL, N_FFT)
    postprocess.plot_optimal_SNR_by_saturation(optimal_SNR_data, parameters['I/O']['outdir'], psd['direction'], M, dL, N_FFT)


    #"""

    # Attempt to generate the global corner plot
    try:
        postprocess.global_corner(results_object, inference_model.names, parameters['I/O']['outdir'])
    except Exception as e:
        print(f"Corner plot failed with error: {e}")
        traceback.print_exc()

    # Show plots if the option is enabled
    if parameters['I/O']['show-plots']:
        plt.show()
