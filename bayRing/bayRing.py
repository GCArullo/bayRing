#! /usr/bin/env python

# Standard python packages
import matplotlib.pyplot as plt, numpy as np, os, time, traceback
from scipy.interpolate import interp1d, CubicSpline
from optparse       import OptionParser
try:                import configparser
except ImportError: import ConfigParser as configparser

# GW-specific imports
import pyRing.utils as pyRing_utils
import lal
import cpnest, cpnest.model

# Package internal imports
import bayRing.NR_waveforms       as NR_waveforms
import bayRing.postprocess        as postprocess
import bayRing.initialise         as initialise
import bayRing.QNM_utils          as QNM_utils
import bayRing.inference          as inference
import bayRing.template_waveforms as template_waveforms
import bayRing.utils              as utils
import bayRing.waveform_utils     as wf_utils

# Constants
twopi = 2.*np.pi

# Conversions
C_mt=(lal.MSUN_SI * lal.G_SI) / (lal.C_SI**3) #s, converts a mass expressed in solar masses into a time in seconds
C_md=(lal.MSUN_SI * lal.G_SI)/(1e6*lal.PC_SI*lal.C_SI**2) #adimensional, converts a mass expressed in solar masses to a distance in Megaparsec

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

    pyRing_utils.print_section('Input parameters')
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

    pyRing_utils.print_section('NR data loading')
    parameters['Injection-data']['modes-list'] = NR_waveforms.read_fake_NR(parameters['NR-data']['catalog'], parameters['Injection-data']['modes'])
    for optional_path in ['properties-file', 'fits-file']:
        parameters['NR-data'][optional_path] = utils.normalize_optional_path(parameters['NR-data'][optional_path])

    #NR simulation object
    NR_sim      = NR_waveforms.NR_simulation(parameters['NR-data']['catalog']                       , 
                                             parameters['NR-data']['ID']                            , 
                                             parameters['NR-data']['res-level']                     , 
                                             parameters['NR-data']['extrap-order']                  ,
                                             parameters['NR-data']['pert-order']                    , 
                                             parameters['NR-data']['dir']                           , 
                                             parameters['NR-data']['properties-file']               ,
                                             parameters['NR-data']['fits-file']                     ,
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
    pyRing_utils.print_section('Simulation metadata')
    for key in NR_metadata.keys(): print('{}: {}'.format(key.ljust(len('omg_peak_22')), NR_metadata[key]))

    if parameters['NR-data']['fits-file']:

        import pandas as pd

        fit_data     = pd.read_csv(parameters['NR-data']['fits-file'])
        fit_metadata = fit_data.iloc[0].to_dict()
        pyRing_utils.print_subsection('Fits metadata')

        for key in fit_metadata.keys(): print('{}: {}'.format(key.ljust(len('fit_type')), fit_metadata[key]))
    else:
        fit_metadata = None    

    # =================#
    # Load Kerr modes. #
    # =================#
    
    cache_negative_m_qnms = (
        parameters['Model']['template'] == 'KerrBinary'
        and parameters['Model']['KerrBinary-version'] == 'Cheung2023'
        and NR_metadata['af'] < 0.0
    )
    Kerr_modes, Kerr_quad_modes, qnm_cached = QNM_utils.read_Kerr_modes(
        parameters['Model']['QNM-modes'],
        parameters['Model']['QQNM-modes'],
        parameters['Model']['charge'],
        parameters['NR-data']['l-NR'],
        parameters['NR-data']['m'],
        NR_metadata,
        cache_negative_m_qnms=cache_negative_m_qnms,
    )
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
                                                fit_metadata                                                               ,  
                                                qnm_cached                                                                 , 
                                                parameters['NR-data']['l-NR']                                              , 
                                                parameters['NR-data']['m']                                                 , 
                                                tail                      = parameters['Model']['Kerr-tail']                       ,
                                                tail_modes                = Kerr_tail_modes                                        ,     
                                                quadratic_modes           = Kerr_quad_modes                                        , 
                                                const_params              = parameters['NR-data']['add-const']                     , 
                                                KerrBinary_version        = parameters['Model']['KerrBinary-version']              ,
                                                KerrBinary_amp_nc_version = parameters['Model']['KerrBinary-amplitudes-nc-version'],
                                                TEOB_template             = parameters['Model']['TEOB-template']                   ,
                                                TEOB_global_fit           = parameters['Model']['TEOB-global-fit']                 ,
                                                TEOB_merger_data          = parameters['Model']['TEOB-merger-data']                ,
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
        postprocess.plot_fancy_reconstruction(NR_sim, wf_model, NR_metadata, None, None, parameters['I/O']['outdir'], None, tail_flag, parameters['I/O']['extract-damping-time-flag'])
        # In case a tail run is selected, do plots also without tail format
        if(tail_flag): postprocess.plot_fancy_reconstruction(NR_sim, wf_model, NR_metadata, None, None, parameters['I/O']['outdir'], None, False, parameters['I/O']['extract-damping-time-flag'])
        print('\n* NR-only plotting run-type selected. Exiting.\n')
        exit()

    pyRing_utils.print_section('Inference')

    #==============================#
    # Inference execution section. #
    #==============================#
    
    if(  parameters['I/O']['run-type']=='full'           ): results_object = inference.run_inference(parameters, inference_model)
    elif(parameters['I/O']['run-type']=='post-processing'): results_object = postprocess.read_results_object_from_previous_inference(parameters)
    else                                                  : raise Exception("Unknown run type selected: {}. Exiting.".format(parameters['I/O']['run-type']))

    if parameters['I/O']['run-type']=='full':
        import pickle
        model_samples = [np.array(inference_model.model(p)) for p in postprocess.waveform_parameter_samples(results_object, parameters['Inference']['method'])]
        with open(os.path.join(parameters['I/O']['outdir'], 'NR_sim.pkl'), 'wb') as f:
            pickle.dump([NR_sim, model_samples, wf_model], f)
        
    #=========================#
    # Postprocessing section. #
    #=========================#

    pyRing_utils.print_section('Post-processing')

    pyRing_utils.print_subsection('Parameters estimates')
    print('* Note: except for free damped sinusoids fits, quantities are quoted at the selected peak time.\n')
    postprocess.print_point_estimate(results_object, inference_model.access_names(), parameters['Inference']['method'])

    pyRing_utils.print_subsection('Waveform metrics')
    postprocess.l2norm_residual_vs_nr(results_object, inference_model, NR_sim, parameters['I/O']['outdir'])

    # Not needed now that we define everything directly at the peak.
    # if(parameters['Model']['template']=='Kerr'): postprocess.post_process_amplitudes(parameters['Inference']['t-start'], results_object, NR_metadata, qnm_cached, Kerr_modes, Kerr_quad_modes, parameters['I/O']['outdir'])
    if(parameters['NR-data']['catalog']=='C2EFT' and 'Damped-sinusoids' in parameters['Model']['template']): postprocess.compare_with_GR_QNMs(results_object, qnm_cached, NR_sim, parameters['I/O']['outdir'])

    if(parameters['I/O']['run-type']=='full'):
    
        if(parameters['Inference']['method']=='Nested-sampler'):
            os.system('mv {dir}/Algorithm/posterior*.pdf {dir}/Plots/Results/.'.format(dir = parameters['I/O']['outdir']))
            if(  parameters['Inference']['sampler']=='raynest'): os.system('mv {dir}/Algorithm/*trace.png   {dir}/Plots/Chains/.'.format(dir = parameters['I/O']['outdir']))
            elif(parameters['Inference']['sampler']=='cpnest' ): os.system('mv {dir}/Algorithm/nschain*.pdf {dir}/Plots/Chains/.'.format(dir = parameters['I/O']['outdir']))
            
        execution_time = (time.time() - execution_time)/60.0
        print('* Execution time (min): {:.2f}\n'.format(execution_time))

    try   : 
        postprocess.plot_fancy_reconstruction(NR_sim, wf_model, NR_metadata, results_object, inference_model, parameters['I/O']['outdir'], parameters['Inference']['method'], tail_flag, parameters['I/O']['extract-damping-time-flag'])
        postprocess.plot_fancy_residual(      NR_sim, wf_model, NR_metadata, results_object, inference_model, parameters['I/O']['outdir'], parameters['Inference']['method'], tail_flag)
        # In case a tail run is selected, do plots also without tail format
        if(tail_flag):
            postprocess.plot_fancy_reconstruction(NR_sim, wf_model, NR_metadata, results_object, inference_model, parameters['I/O']['outdir'], parameters['Inference']['method'], False, parameters['I/O']['extract-damping-time-flag'])
            postprocess.plot_fancy_residual(      NR_sim, wf_model, NR_metadata, results_object, inference_model, parameters['I/O']['outdir'], parameters['Inference']['method'], False)
    except Exception as e:
        print(f"Waveform reconstruction plot failed with error: {e}")
        traceback.print_exc()    

    pyRing_utils.print_subsection(f'Mismatch and SNR computation')
    postprocess.run_mismatch_and_SNR_computation(NR_sim, results_object, inference_model, parameters, wf_utils)

    # Attempt to generate the global corner plot
    if(parameters['Inference']['method']=='Nested-sampler'):
        try:
            postprocess.global_corner(results_object, inference_model.names, parameters['I/O']['outdir'])
        except Exception as e:
            print(f"Corner plot failed with error: {e}")
            traceback.print_exc()

    # Show plots if the option is enabled
    if parameters['I/O']['show-plots']:
        plt.show()

if __name__=='__main__':
    main()
