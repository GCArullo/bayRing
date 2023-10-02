#! /usr/bin/env python

# Standard python packages
import matplotlib.pyplot as plt, numpy as np, os, time
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
from pyRing.utils           import print_section

twopi = 2.*np.pi

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

    NR_sim      = NR_waveforms.NR_simulation(parameters['NR-data']['catalog']                     , 
                                             parameters['NR-data']['ID']                          , 
                                             parameters['NR-data']['res-level']                   , 
                                             parameters['NR-data']['extrap-order']                , 
                                             parameters['NR-data']['pert-order']                  , 
                                             parameters['NR-data']['dir']                         , 
                                             parameters['Injection-data']['modes-list']           , 
                                             parameters['Injection-data']['times']                , 
                                             parameters['Injection-data']['noise']                , 
                                             parameters['Injection-data']['tail']                 , 
                                             parameters['NR-data']['l-NR']                        , 
                                             parameters['NR-data']['m']                           , 
                                             parameters['I/O']['outdir']                          , 
                                             download       = parameters['NR-data']['download']   , 
                                             NR_error       = parameters['NR-data']['error']      ,
                                             tM_start       = parameters['Inference']['t-start']  , 
                                             tM_end         = parameters['Inference']['t-end']    , 
                                             t_delay_scd    = parameters['Inference']['dt-scd']   , 
                                             t_min_mismatch = parameters['NR-data']['error-t-min'], 
                                             t_max_mismatch = parameters['NR-data']['error-t-max'])
    
#tM_start       = parameters['Inference']['t-start']   ,
    error       = NR_sim.NR_cpx_err_cut
    NR_metadata = NR_waveforms.read_NR_metadata(NR_sim, parameters['NR-data']['catalog'])

    # =================#
    # Load Kerr modes. #
    # =================#
    
    Kerr_modes, Kerr_quad_modes, qnm_cached = QNM_utils.read_Kerr_modes(parameters['Model']['QNM-modes'], parameters['Model']['QQNM-modes'], parameters['Model']['charge'], parameters['NR-data']['l-NR'], parameters['NR-data']['m'], NR_metadata)
    Kerr_tail_modes                         = QNM_utils.read_tail_modes(parameters['Model']['Kerr-tail-modes'])

    # ============#
    # Load model. #
    # ============#

    wf_model = template_waveforms.WaveformModel(NR_sim.t_NR_cut                                       , 
                                                NR_sim.t_min                                          , 
                                                parameters['Model']['template']                       , 
                                                parameters['Model']['N-DS-modes']                     , 
                                                Kerr_modes                                            , 
                                                NR_metadata                                           , 
                                                qnm_cached                                            , 
                                                parameters['NR-data']['l-NR']                         , 
                                                parameters['NR-data']['m']                            , 
                                                tail              = parameters['Model']['DS-tail']    ,
                                                tail_modes        = Kerr_tail_modes                   ,     
                                                quadratic_modes   = Kerr_quad_modes                   , 
                                                const_params      = parameters['NR-data']['add-const'], 
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

    # Plot and terminate execution if plotting only.
    if(parameters['I/O']['run-type']=='plot-NR-only'): 
        postprocess.plot_NR_vs_model(NR_sim, wf_model, NR_metadata, None, None, parameters['I/O']['outdir'], None)
        print('\n* NR-only plotting run-type selected. Exiting.\n')
        exit()

    if(parameters['Model']['fixed-waveform']): results_object = {'dummy_x': np.array([0.0])}
    else:

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

        print('\n* Note: quantities are quoted at the start time of the fit (i.e. t_peak + t_0 [M]).\n')
        postprocess.print_point_estimate(results_object, inference_model.access_names(), parameters['Inference']['method'])
        # postprocess.plot_fancy_residual(NR_sim, wf_model, NR_metadata, results_object, inference_model, parameters['I/O']['outdir'], parameters['Inference']['method'])
        # postprocess.plot_fancy_reconstruction(NR_sim, wf_model, NR_metadata, results_object, inference_model, parameters['I/O']['outdir'], method)
        postprocess.l2norm_residual_vs_nr(results_object, inference_model, NR_sim, parameters['I/O']['outdir'])

        if('Kerr' in parameters['Model']['template'] ): postprocess.post_process_amplitudes(parameters['Inference']['t-start'], results_object, NR_metadata, qnm_cached, Kerr_modes, Kerr_quad_modes, parameters['I/O']['outdir'])
        if(parameters['NR-data']['catalog']=='C2EFT' and 'Damped-sinusoids' in parameters['Model']['template']): postprocess.compare_with_GR_QNMs(results_object, qnm_cached, NR_sim, parameters['I/O']['outdir'])

        if(parameters['I/O']['run-type']=='full'):
        
            if(parameters['Inference']['method']=='Nested-sampler'):
                os.system('mv {dir}/Algorithm/posterior*.pdf {dir}/Plots/Results/.'.format(dir = parameters['I/O']['outdir']))
                os.system('mv {dir}/Algorithm/nschain*.pdf {dir}/Plots/Chains/.'.format(dir = parameters['I/O']['outdir']))
                
            execution_time = (time.time() - execution_time)/60.0
            print('\nExecution time (min): {:.2f}\n'.format(execution_time))

    postprocess.plot_NR_vs_model(NR_sim, wf_model, NR_metadata, results_object, inference_model, parameters['I/O']['outdir'], parameters['Inference']['method'])
    postprocess.global_corner(results_object, inference_model.names, parameters['I/O']['outdir'])
    if(parameters['I/O']['show-plots']): plt.show()