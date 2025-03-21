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
import lal

# Constants
twopi = 2.*np.pi

# Conversions
C_mt=(lal.MSUN_SI * lal.G_SI) / (lal.C_SI**3) #s, converts a mass expressed in solar masses into a time in seconds
C_md=(lal.MSUN_SI * lal.G_SI)/(1e6*lal.PC_SI*lal.C_SI**2) #adimensional, converts a mass expressed in solar masses to a distance in Megaparsec

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
    psd_data, acf_data, mismatch_data, optimal_SNR_data, condition_numbers = {}, {}, {}, {}, {}

    # Assign GW parameters
    M, dL, ra, dec, psi = wf_utils.extract_GW_parameters(parameters)
    t_start_g_true = parameters['Inference']['t-start']

    # Extract estimated t-peak, t_start and t_end
    t_start_g, t_end_g, t_NR_s, NR_length = wf_utils.extract_NR_params(NR_sim, M)

    # Convert estimated start and end time in seconds
    t_start, t_end = t_start_g * C_mt * M, t_end_g * C_mt * M

    # Load PSD parameters
    f_min, f_max, dt, _, N_points, n_FFT_points, asd_path, n_iterations_C1, window_sizes_DX, window_sizes_SX, steepness_values, saturation_DX_values, saturation_SX_values, direction = wf_utils.extract_and_compute_psd_parameters(parameters['Mismatch-PSD-settings'])

    # Load flags
    compare_TD_FD, clear_directory, C1_flag, mismatch_print_flag, mismatch_section_plot_flag = wf_utils.extract_flags(parameters['Flags'])

    # Choose if iterate or not on N_FFT
    N_FFT = [N_points] if n_FFT_points == 1 else list(map(int, np.logspace(np.log10(NR_length), np.log10(2*N_points), n_FFT_points)))

    # Choose if cleaning directories or not
    if clear_directory == 1:

        # Define the directory path
        smoothing_paths = ["Left_smoothing", "Right_smoothing", "Both_edges_smoothing"]
        for smoothing_path in smoothing_paths:
            algorithm_dir = os.path.join(parameters['I/O']['outdir'], "Algorithm", smoothing_path)

            postprocess.clear_directory(algorithm_dir) 

    # Iterate over the number of FFT points
    for N_fft in N_FFT:

        # Iterate over the smoothing parameters and compute ACF
        for window_size_DX, window_size_SX, k, saturation_DX, saturation_SX in [(wdx, wsx, s, tdx, tsx) for wdx in window_sizes_DX for wsx in window_sizes_SX for s in steepness_values for tdx in saturation_DX_values for tsx in saturation_SX_values]:
                
            # Consistency check on starting time, end time and f_min 
            if (t_end-t_start)>1/(f_min+window_size_DX) and direction!='above':
                print("Please provide (t_end-t_start) < 1/(f_min+window_size_DX).")
                print("Forbidden frequency:",f_min+window_size_DX)
                exit()

            try:

                # Print window parameters
                print(f"\n\nSelected window parameters: w_DX={round(window_size_DX,1)}Hz, w_SX={round(window_size_SX,1)}Hz, k={round(k,1)}, saturation_DX={round(saturation_DX,1)}, saturation_SX={round(saturation_SX,1)}, N_FFT={N_fft}\n")


                # Compute PSD and ACF with smoothing at PSD edges
                PSD_smoothed, ACF_smoothed = wf_utils.acf_from_asd_with_smoothing(
                    asd_path,
                    f_min, f_max,
                    N_fft,
                    window_size_DX=window_size_DX,
                    window_size_SX=window_size_SX,
                    k=k,
                    saturation_DX=saturation_DX,
                    saturation_SX=saturation_SX,
                    direction=direction,
                    C1_flag=C1_flag,
                    n_iterations_C1=n_iterations_C1
                )

                # Store smoothed PSD/ACF data in dictionaries
                psd_data[f"wDX={round(window_size_DX,1)}Hz, wSX={round(window_size_SX,1)}Hz, k={round(k,0)}, satDX={round(saturation_DX,1)}, satSX={round(saturation_SX,1)}, N_FFT={N_fft}"] = PSD_smoothed
                acf_data[f"wDX={round(window_size_DX,1)}Hz, wSX={round(window_size_SX,1)}Hz, k={round(k,0)}, satDX={round(saturation_DX,1)}, satSX={round(saturation_SX,1)}, N_FFT={N_fft}"] = ACF_smoothed

                #-------------------------------------------------- Mismatch computation -------------------------------------------------------#

                # Truncate ACF to ringdown analysis lenght
                t_ACF = np.linspace(0, N_fft*dt, len(ACF_smoothed))
                ACF_truncated_NR = postprocess.truncate_and_interpolate_acf(t_ACF, ACF_smoothed, M, t_start_g, t_end_g, t_NR_s, mismatch_print_flag)

                # Store condition number values in dictionary
                condition_numbers[(window_size_DX, window_size_SX, k, saturation_DX, saturation_SX)] = wf_utils.compute_condition_number(ACF_truncated_NR)

                # Compute mismatch for hplus, hcross
                postprocess.compute_mismatch_hplus_hcross(
                    NR_sim, 
                    results_object, 
                    inference_model, 
                    parameters['I/O']['outdir'],
                    parameters['Inference']['method'], 
                    ACF_truncated_NR, N_fft,
                    M, dL,
                    t_start_g_true,
                    f_min, f_max,
                    asd_path,
                    window_size_DX, window_size_SX, k,
                    mismatch_print_flag,
                    compare_TD_FD
                )

                # Compute mismatch for htot
                postprocess.compute_mismatch_htot(
                    NR_sim, 
                    results_object, 
                    inference_model, 
                    parameters['I/O']['outdir'],
                    parameters['Inference']['method'], 
                    ACF_truncated_NR, N_fft,
                    M, dL, ra, dec, psi,
                    t_start_g_true,
                    window_size_DX, window_size_SX, k
                )

                # Plot mismatch sanity checks

                # Read mismatch results from file
                mismatch_filename = f"Mismatch_M_{M}_dL_{dL}_t_s_{round(t_start_g_true,1)}M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}_NFFT_{N_fft}.txt"
                mismatch_file = os.path.join(parameters['I/O']['outdir'], 'Algorithm', mismatch_filename)

                with open(mismatch_file, 'r') as f:
                    lines = f.readlines()[1:]  # Skip the header

                # Store mismatch results in a dictionry
                mismatch_data[(window_size_DX, window_size_SX, k, saturation_DX, saturation_SX)] = {'real': {}, 'imaginary': {}}
                for line in lines:
                    perc, component, mismatch = line.strip().split('\t')
                    perc = int(perc)
                    mismatch_data[(window_size_DX, window_size_SX, k, saturation_DX, saturation_SX)][component][perc] = float(mismatch)

                #-------------------------------------------------- optimal SNR computation -------------------------------------------------------#

                if mismatch_section_plot_flag==1:

                    # Plot truncated ACF
                    postprocess.plot_acf_interpolated(t_ACF, t_NR_s, ACF_smoothed, ACF_truncated_NR, parameters['I/O']['outdir'], window_size_DX, window_size_SX, k, saturation_DX, saturation_SX, direction)
                
                # Call compute_mismatch with the subsampled smoothed ACF
                postprocess.compute_optimal_SNR(
                    NR_sim, 
                    results_object, 
                    inference_model, 
                    parameters['I/O']['outdir'],
                    parameters['Inference']['method'], 
                    ACF_truncated_NR,
                    N_fft,
                    M, dL,
                    t_start_g, t_end_g,
                    f_min, f_max,
                    asd_path,
                    window_size_DX, window_size_SX, k,
                    compare_TD_FD
                )

                if mismatch_section_plot_flag==1:
                    postprocess.mismatch_sanity_checks(NR_sim, 
                                                       results_object, 
                                                       inference_model, 
                                                       parameters['I/O']['outdir'],
                                                       parameters['Inference']['method'],  
                                                       ACF_truncated_NR,
                                                       M, dL, t_start_g, t_end_g, window_size_DX, window_size_SX, k)

                # Read optimal SNR results from file
                optimal_SNR_filename = f"Optimal_SNR_M_{M}_dL_{dL}_t_s_{round(t_start_g,1)}M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}_NFFT_{N_fft}.txt"
                optimal_SNR_file = os.path.join(parameters['I/O']['outdir'], 'Algorithm', optimal_SNR_filename)
                with open(optimal_SNR_file, 'r') as f:
                    lines = f.readlines()[1:]  # Skip the header

                # Store mismatch results in a dictionary
                optimal_SNR_data[(window_size_DX, window_size_SX, k, saturation_DX, saturation_SX)] = {'real': {}, 'imaginary': {}}
                for line in lines:
                    perc, component, optimal_SNR = line.strip().split('\t')
                    perc = int(perc)
                    optimal_SNR_data[(window_size_DX, window_size_SX, k, saturation_DX, saturation_SX)][component][perc] = float(optimal_SNR)

            except Exception as e:
                print(f"Optimal SNR computation failed for window_sizes=({window_size_DX, window_size_SX})Hz, k={k}, saturations={(saturation_DX, saturation_SX)}: {e}")

    #----------------------------------------------------------------------------------- Postprocessing --------------------------------------------------------------------------------------------------------------------------#

    if mismatch_section_plot_flag==1:

        # Postprocess plots
        postprocess.plot_psd_and_acf(psd_data, acf_data, asd_path, f_min, f_max, parameters['I/O']['outdir'], direction)
        postprocess.plot_mismatch_optimal_SNR_condition_number_window_parameters(mismatch_data, optimal_SNR_data, condition_numbers, parameters['I/O']['outdir'], direction, M, dL, N_FFT)

    # Attempt to generate the global corner plot
    try:
        postprocess.global_corner(results_object, inference_model.names, parameters['I/O']['outdir'])
    except Exception as e:
        print(f"Corner plot failed with error: {e}")
        traceback.print_exc()

    # Show plots if the option is enabled
    if parameters['I/O']['show-plots']:
        plt.show()
