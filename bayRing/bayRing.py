#! /usr/bin/env python

# Standard python packages
import concurrent.futures
import copy, matplotlib.pyplot as plt, numpy as np, os, time, traceback
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

def _prepare_start_time_parameters(base_parameters, base_outdir, t_start, start_index, n_start_times,
                                   parallel_start_time=False, nr_mode=None, mode_index=1, n_modes=1):

    parameters = copy.deepcopy(base_parameters)
    multi_start = n_start_times > 1
    multi_mode  = n_modes > 1
    active_outdir = base_outdir

    if(nr_mode is not None):
        l_nr, m_nr = nr_mode
        parameters['NR-data']['l-NR'] = l_nr
        parameters['NR-data']['m']    = m_nr
        parameters['I/O']['mode-output'] = multi_mode
        parameters['I/O']['mode-index']  = mode_index
        parameters['I/O']['n-modes']     = n_modes
        parameters['I/O']['mode-label']  = initialise.format_nr_mode_label(l_nr, m_nr)
        if(parameters['Model']['template'] == 'Damped-sinusoids'):
            parameters['Model']['QNM-modes'] = '{}{}0'.format(l_nr, m_nr)
        if(multi_mode):
            active_outdir = initialise.nr_mode_output_dir(base_outdir, nr_mode)
    else:
        parameters['I/O']['mode-output'] = False

    parameters['Inference']['t-start'] = t_start
    parameters['I/O']['base-outdir']   = base_outdir
    parameters['I/O']['start-time-output'] = multi_start
    parameters['I/O']['start-time-parallel'] = parallel_start_time and (multi_start or multi_mode)
    parameters['I/O']['start-time-index'] = start_index
    parameters['I/O']['n-start-times']    = n_start_times

    if(multi_start):
        parameters['I/O']['outdir'] = initialise.start_time_output_dir(active_outdir, t_start)
    elif(multi_mode):
        parameters['I/O']['outdir'] = active_outdir

    return parameters

def _run_single_start(Config, parameters, config_file):

    execution_time = time.time()

    # =================#
    # Set output dirs. #
    # =================#

    start_time_output = parameters['I/O'].get('start-time-output', False)
    mode_output = parameters['I/O'].get('mode-output', False)
    scan_output = start_time_output or mode_output
    parallel_start_time = parameters['I/O'].get('start-time-parallel', False)
    initialise.set_output(parameters['I/O']['outdir'],
                          parameters['I/O']['screen-output'],
                          parameters['Inference']['method'],
                          config_file,
                          parameters['I/O']['run-type'],
                          shared_files=not(scan_output),
                          redirect_streams=not(scan_output) or parallel_start_time)

    if(mode_output):
        pyRing_utils.print_section('NR-mode fit {}/{}'.format(parameters['I/O']['mode-index'], parameters['I/O']['n-modes']))
        print('* NR mode          : ({}, {})'.format(parameters['NR-data']['l-NR'], parameters['NR-data']['m']))
        print('* Output directory : `{}`.\n'.format(parameters['I/O']['outdir']))
    if(start_time_output):
        pyRing_utils.print_section('Start-time fit {}/{}'.format(parameters['I/O']['start-time-index'], parameters['I/O']['n-start-times']))
        print('* t-start [M]      : {}'.format(parameters['Inference']['t-start']))
        print('* Output directory : `{}`.\n'.format(parameters['I/O']['outdir']))

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
                                                N_ds_tails                = parameters['Model']['N-DS-tails']                      ,
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
    # Plot and terminate the active start-time run if plotting only.
    if(parameters['I/O']['run-type']=='plot-NR-only'):
        postprocess.plot_fancy_reconstruction(NR_sim, wf_model, NR_metadata, None, None, parameters['I/O']['outdir'], None, tail_flag, parameters['I/O']['extract-damping-time-flag'])
        # In case a tail run is selected, do plots also without tail format
        if(tail_flag): postprocess.plot_fancy_reconstruction(NR_sim, wf_model, NR_metadata, None, None, parameters['I/O']['outdir'], None, False, parameters['I/O']['extract-damping-time-flag'])
        print('\n* NR-only plotting run-type selected. Exiting this start-time run.\n')
        return

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

    return

def _run_start_time_job(config_file, parameters):

    Config = configparser.ConfigParser()
    Config.read(config_file)
    _run_single_start(Config, parameters, config_file)

    return

def _run_scan_jobs_parallel(config_file, run_parameters_list, scan_workers):

    errors = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=scan_workers) as executor:
        future_to_start_time = {
            executor.submit(_run_start_time_job, config_file, run_parameters): (
                run_parameters['I/O'].get('mode-index', 1),
                run_parameters['NR-data']['l-NR'],
                run_parameters['NR-data']['m'],
                run_parameters['I/O']['start-time-index'],
                run_parameters['Inference']['t-start'],
                run_parameters['I/O']['outdir'],
            )
            for run_parameters in run_parameters_list
        }

        for future in concurrent.futures.as_completed(future_to_start_time):
            mode_index, l_nr, m_nr, start_index, t_start, outdir = future_to_start_time[future]
            try:
                future.result()
            except Exception:
                errors.append((mode_index, l_nr, m_nr, start_index, t_start, outdir, traceback.format_exc()))

    if(errors):
        for mode_index, l_nr, m_nr, start_index, t_start, outdir, formatted_traceback in errors:
            print('* Scan fit failed for mode ({}, {}) and t-start = {} M.'.format(l_nr, m_nr, t_start))
            print('* Output directory: `{}`.'.format(outdir))
            print(formatted_traceback)
        raise RuntimeError('{} fit(s) failed during the parallel scan.'.format(len(errors)))

    return

def _run_start_times_parallel(config_file, run_parameters_list, start_time_workers):

    _run_scan_jobs_parallel(config_file, run_parameters_list, start_time_workers)

    return

def main():

    # ==================================================#
    # Initialize execution and read configuration file. #
    # ==================================================#

    # Print ascii art.
    try   : print("\u001b[\u001b[38;5;39m{}\u001b[0m".format(initialise.my_art))
    except: pass
    print(initialise.__ascii_art__)

    # Initialise and read config.
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

    parameters       = initialise.read_config(Config)
    start_times      = initialise.get_start_time_values(parameters)
    nr_modes         = initialise.get_nr_mode_values(parameters)
    base_outdir      = parameters['I/O']['outdir']
    n_start_times    = len(start_times)
    n_modes          = len(nr_modes)
    multi_start_time = n_start_times > 1
    multi_mode       = n_modes > 1
    start_time_workers = min(parameters['Inference']['n-start-time-workers'], n_start_times)
    mode_workers       = min(parameters['Inference']['n-mode-workers'], n_modes)
    scan_workers       = min(max(start_time_workers, mode_workers), n_start_times*n_modes)
    parallel_scan      = (multi_start_time or multi_mode) and scan_workers > 1

    if(multi_start_time or multi_mode):
        pyRing_utils.print_section('Start-time scan')
        if(multi_start_time):
            print('* Repeating the fit for {} start times: {}'.format(n_start_times, start_times))
        if(multi_mode):
            print('* Repeating the fit for {} NR modes: {}'.format(n_modes, nr_modes))
        if(parallel_scan):
            print('* Running up to {} scan fits in parallel.'.format(scan_workers))
        print('* Base output directory: `{}`.\n'.format(base_outdir))
        initialise.set_shared_output(base_outdir, parameters['I/O']['screen-output'], config_file, parameters['I/O']['run-type'])

    run_parameters_list = [
        _prepare_start_time_parameters(
            parameters, base_outdir, t_start, start_index, n_start_times,
            parallel_scan, nr_mode, mode_index, n_modes
        )
        for mode_index, nr_mode in enumerate(nr_modes, start=1)
        for start_index, t_start in enumerate(start_times, start=1)
    ]

    if(parallel_scan):
        _run_scan_jobs_parallel(config_file, run_parameters_list, scan_workers)
    else:
        for run_parameters in run_parameters_list:
            _run_single_start(Config, run_parameters, config_file)

    if(multi_mode and parameters['Flags']['compute_hm_mismatch']):
        postprocess.run_higher_mode_mismatch_scan(run_parameters_list, parameters)

    # Show plots if the option is enabled
    if parameters['I/O']['show-plots']:
        plt.show()

if __name__=='__main__':
    main()
