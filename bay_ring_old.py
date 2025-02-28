#! /usr/bin/env python

# Standard python packages
import matplotlib.pyplot as plt, numpy as np, os, time, traceback
from scipy.interpolate import interp1d
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

    #----------------------------------------------------- Smoothing and Mismatch computation -----------------------------------------------------------------------------------------#

    # Initialize dictionaries to store ASD, ACF, and mismatch data
    psd_data = {}
    acf_data = {}
    mismatch_data = {}
    multiplier_factor=1e3 #this is an example

    # Determine smoothing parameters (to do: add below-above)
    if parameters['Mismatch']['direction'] == 'below':
        window_sizes = np.arange(0, 1.5, 0.25).tolist()
        steepness_values = [1, 3]
    elif parameters['Mismatch']['direction'] == 'above':
        window_sizes = np.arange(0, 1.5, 0.25).tolist()
        steepness_values = [1, 3]
    else:
        raise ValueError("Invalid direction. Choose 'below' or 'above'.")

    # Iterate over the smoothing parameters and compute ACF
    for window_size, k in [(w, s) for w in window_sizes for s in steepness_values]:
        print(f"Calculating ACF with smoothing: window_size={window_size}, k={k}, direction={parameters['Mismatch']['direction']}")
        try:
            # Compute ACF with smoothing
            smoothed_N_points = int(1e5)
            PSD_smoothed, ACF_smoothed = wf_utils.acf_from_asd_with_smoothing(
                parameters['Mismatch']['asd-path'],
                parameters['Mismatch']['f-min'],
                parameters['Mismatch']['f-max'],
                smoothed_N_points,
                window_size=window_size,
                k=k,
                multiplier_factor=multiplier_factor,
                direction=parameters['Mismatch']['direction']
            )

            # Store smoothed PSD/ACF data in dictionaries
            psd_data[f"window_{window_size}_k_{k}_{parameters['Mismatch']['direction']}"] = PSD_smoothed
            acf_data[f"window_{window_size}_k_{k}_{parameters['Mismatch']['direction']}"] = ACF_smoothed

            # Time array
            dt = 1.0 / (2 * parameters['Mismatch']['f-max'])
            t_start, t_end = parameters['Inference']['t-start'], parameters['Inference']['t-end'] #in M units
            t_start *= C_mt
            t_end *= C_mt
            dt *= C_mt #convert into seconds

            # Create a time array corresponding to the ACF_smoothed
            N_points = len(ACF_smoothed)
            t_array = np.arange(0, N_points * dt, dt)

            # Print info
            print("Time [M]:", t_array)
            print("Time [s]:", t_array * C_mt)

            # Ensure t_array matches the length of ACF_smoothed if off by one due to floating point arithmetic
            if len(t_array) > len(ACF_smoothed):
                t_array = t_array[:len(ACF_smoothed)]

            # Create an interpolation function for the ACF using interp1d
            acf_interpolated_func = interp1d(t_array, ACF_smoothed, kind='cubic', fill_value="extrapolate")

            # Generate the truncated time array t_trunc between t_start and t_end with the same number of points as NR_sim.NR_r_cut
            num_points = len(NR_sim.NR_r_cut)
            t_trunc = np.linspace(t_start, t_end, num_points)

            # Interpolate ACF on the t_trunc array
            truncated_acf = acf_interpolated_func(t_trunc)

            # Print information about the truncated ACF
            print("Truncated time array (t_trunc):", t_trunc)
            print("Truncated ACF dimensions to match NR_r_cut:", len(truncated_acf))

            # Plot original ACF
            #plt.plot(t_array, ACF_smoothed, label='Original ACF', color='blue', linestyle='--')

            # Plot truncated ACF
            plt.figure(figsize=(8,6))
            plt.plot(t_trunc, truncated_acf, label='Truncated ACF', color='red')

            plt.xlabel('Time [s]')
            plt.ylabel('ACF')
            plt.title('Autocorrelation Function (ACF) - Truncated')
            plt.legend()
            plt.grid(True)
            path_acf=os.path.join(parameters['I/O']['outdir'], "Algorithm/Autocorrelation_truncated.png")
            plt.savefig(path_acf)


            """
            # Call compute_mismatch with the subsampled smoothed ACF
            postprocess.compute_mismatch(
                NR_sim, results_object, inference_model, parameters['I/O']['outdir'],
                parameters['Inference']['method'], sub_ACF_smoothed
            )

            # Read mismatch results from file
            mismatch_file = os.path.join(parameters['I/O']['outdir'], 'Algorithm/Mismatch.txt')
            with open(mismatch_file, 'r') as f:
                lines = f.readlines()[1:]  # Skip the header

            # Store mismatch results in mismatch_data (consider only real and imaginary fro simplicity)
            mismatch_data[(window_size, k)] = {'real': {}, 'imaginary': {}}
            for line in lines:
                perc, component, mismatch = line.strip().split('\t')
                perc = int(perc)
                mismatch_data[(window_size, k)][component][perc] = float(mismatch)
        
            """

        except Exception as e:
            print(f"Mismatch computation failed for window_size={window_size}, k={k}: {e}")
        
    #----------------------------------------------------------------------------------- Postprocessing --------------------------------------------------------------------------------------------------------------------------#
 
    # Plot all ACF curves
    postprocess.plot_multiple_psd(psd_data, parameters['Mismatch']['f-min'], parameters['Mismatch']['f-max'], parameters['I/O']['outdir'], parameters['Mismatch']['direction'], window_size)
    postprocess.plot_multiple_acf_with_smoothing(acf_data, dt, parameters['I/O']['outdir'], parameters['Mismatch']['direction'])
    postprocess.plot_mismatch_by_window(mismatch_data, parameters['I/O']['outdir'], parameters['Mismatch']['direction'])

    # Attempt to generate the global corner plot
    try:
        postprocess.global_corner(results_object, inference_model.names, parameters['I/O']['outdir'])
    except Exception as e:
        print(f"Corner plot failed with error: {e}")
        traceback.print_exc()

    # Show plots if the option is enabled
    if parameters['I/O']['show-plots']:
        plt.show()
