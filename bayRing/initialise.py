import ast, json, os, sys
try:                import configparser
except ImportError: import ConfigParser as configparser

import bayRing.QNM_utils as QNM_utils
import pyRing.utils    as pyRing_utils
from pyRing.initialise import store_git_info

def set_output(outdir, screen_output, method, config_file, run_type):

    """

    Set the output directory and the output to the screen.

    Parameters
    ----------

    outdir : str
        Output directory.

    screen_output : bool
        If True, the output is printed on the screen.

    method : str
        Method used to obtain the results with which the results will be obtained. Can be either 'Minimization' or 'Nested-sampler'.
    
    Returns
    -------

    Nothing, but creates the output directory and sets the output to screen.

    """
        
    if not os.path.exists(outdir):                                     os.makedirs(outdir)
    if not os.path.exists(os.path.join(outdir,'Algorithm')):           os.makedirs(os.path.join(outdir,'Algorithm'))
    if not os.path.exists(os.path.join(outdir,'Peak_quantities')):     os.makedirs(os.path.join(outdir,'Peak_quantities'))
    if(method=='Nested-sampler'):
        if not os.path.exists(os.path.join(outdir,'Plots','Chains')):  os.makedirs(os.path.join(outdir,'Plots','Chains'))
    if not os.path.exists(os.path.join(outdir,'Plots')):               os.makedirs(os.path.join(outdir,'Plots'))
    if not os.path.exists(os.path.join(outdir,'Plots','Results')):     os.makedirs(os.path.join(outdir,'Plots','Results'))
    if not os.path.exists(os.path.join(outdir,'Plots','Comparisons')): os.makedirs(os.path.join(outdir,'Plots','Comparisons'))

    if not(screen_output):
        sys.stdout = open(os.path.join(outdir,'stdout_bayRing.txt'), 'w')
        sys.stderr = open(os.path.join(outdir,'stderr_bayRing.txt'), 'w')

    store_git_info(outdir)

    try:
        if (run_type=='full'):
            os.system('cp {} {}/.'.format(config_file, outdir))
    except: pass

    return

def read_config(Config):

    """

    Read the configuration file.

    Parameters
    ----------

    Config : configparser.ConfigParser
        ConfigParser object.
    config_file : str
        Configuration file.
    
    Returns
    -------

    parameters : dict
        Dictionary with the input parameters.

    """

    # Dictionary containing the default values of the parameters
    parameters={

        'I/O': 
        {
        'run-type'         : 'full',
        'screen-output'    : 0,
        'show-plots'       : 0,
        'outdir'           : './',
        },

        'NR-data':
        {
        'download'         : 1,
        'dir'              : '',
        'catalog'          : 'SXS',
        'ID'               : '0305',
        'extrap-order'     : 2,
        'res-level'        : -1,
        'res-nx'           : 0,   
        'res-nl'           : 0,  
        'pert-order'       : 'lin', 
        'l-NR'             : 2,
        'm'                : 2,
        'error'            : 'align-with-mismatch-res-only',
        'error-t-min'      : 4e-3,
        'error-t-max'      : 3e-1,
        'add-const'        : '0.0,0.0',
        'properties-file'  : '',
        't-peak-22'        : 0.0,
        'waveform-type'    : 'strain',
        },

        'Injection-data':
        {
        'modes'            : '220,221,320',
        'times'            : 'from-SXS-NR',
        'noise'            : None,
        'tail'             : 0.0,
        },

        'Model':
        {
        'template'                         : 'Kerr'       ,
        'N-DS-modes'                       : 1            ,
        'QNM-modes'                        : '220,221,320',
        'QQNM-modes'                       : ''           ,
        'Kerr-tail'                        : 0            ,
        'Kerr-tail-modes'                  : '22'         ,
        'KerrBinary-version'               : 'London2018' ,
        'KerrBinary-amplitudes-nc-version' : ''           ,
        'TEOB-NR-fit'                      : 0            ,
        'TEOB-template'                    : 'qc'         ,
        },

        'Inference':
        {
        'method'           : 'Nested-sampler',
        'likelihood'       : 'gaussian'      ,
        'sampler'          : 'cpnest'        ,
        'nlive'            : 256             ,
        'maxmcmc'          : 256             ,
        'seed'             : 1234            ,
        'nnest'            : 1               ,
        'nensemble'        : 1               ,

        't-start'          : 20.0 ,
        't-end'            : 140.0,
        'dt-scd'           : 0.0  ,
        
        'min-method'       : 'lm',
        'min-iter-min'     : 1   ,
        'min-iter-max'     : 1000,
        },

        'Mismatch-PSD-settings':
        {
        'asd-path'              : ''     ,
        'direction'             : 'below',
        'window_DX'             : 0.8    ,
        'window_DX_max'         : 10.0   ,
        'window_SX'             : 0.8    ,
        'window_SX_max'         : 10.0   ,
        'n_window_DX'           : 1      ,
        'n_window_SX'           : 1      ,
        'steepness'             : 7.     ,
        'steepness_max'         : 200.   ,
        'n_steepness'           : 1      ,
        'saturation_DX'         : 1.      ,
        'saturation_DX_max'     : 5.      ,
        'n_saturation_DX'       : 1      ,
        'saturation_SX'         : 1.      ,
        'saturation_SX_max'     : 5.      ,
        'n_saturation_SX'       : 1      ,
        'n_FFT_points'          : 1      ,
        'n_iterations_C1'       : 1      
        },

        'Mismatch-GW-parameters':
        {
        'M'                    : 60     ,
        'dL'                   : 410    ,
        'ra'                   : 1.375  ,
        'dec'                  : -0.2108,
        'psi'                  : 2.659
        },

        'Flags': 
        {
        'C1_flag'                      : 1,
        'clear_directory'              : 1,
        'compare_TD_FD'                : 0,
        'mismatch_print_flag'          : 0,
        'mismatch_section_plot_flag'   : 0,
        }

    }

    #General input read.
    for parameters_section in parameters.keys():

        pyRing_utils.print_subsection(f'[{parameters_section}]')

        try:
            for key in parameters[parameters_section].keys():
                keytype = type(parameters[parameters_section][key])
                try                                                     : parameters[parameters_section][key] = keytype(Config.get(parameters_section, key))
                except (KeyError, configparser.NoOptionError, TypeError): pass

                # Other reading options
                # if   ('ds-modes'        in key): parameters[parameters_section][key] = json.loads(      Config.get(parameters_section, f'{key}')) # dict
                # elif ('quadratic-modes' in key): parameters[parameters_section][key] = eval(            Config.get(parameters_section, f'{key}')) # dict of lists
                # elif ('Kerr-tail-modes' in key): parameters[parameters_section][key] = eval(            Config.get(parameters_section, f'{key}')) # list
                # elif ('mode'            in key): parameters[parameters_section][key] = ast.literal_eval(Config.get(parameters_section,    key  )) # lists
                    
                print("{name} : {value}".format(name=key.ljust(max_len_keyword), value=parameters[parameters_section][key]))
        except (KeyError, configparser.NoSectionError, configparser.NoOptionError, TypeError): pass

    # Cleanup specific parameters formatting
    if(parameters['Inference']['sampler'] == 'raynest'):
        print('Nnest + nensemble: ', parameters['Inference']['nnest'] + parameters['Inference']['nensemble'])
        if parameters['Inference']['nensemble'] < parameters['Inference']['nnest']: raise ValueError(f"Invalid parallelization options: input nensemble ( =  {parameters['Inference']['nensemble']}) cannot be smaller than input nnest ( = {parameters['Inference']['nnest']} ). ")

    # For Teukolsky, map the different resolution levels to their values of nx_, nl_.
    if(parameters['NR-data']['res-nx'] != 0 and parameters['NR-data']['res-nl'] != 0): parameters['NR-data']['res-level'] = "nx_"+str(parameters['NR-data']['res-nx'])+"_nl_"+str(parameters['NR-data']['res-nl'])
    if(parameters['NR-data']['error']=='from-SXS-NR'):
        if not(parameters['Injection-data']['times']=='from-SXS-NR'):
            raise ValueError("When the error is taken from the corresponding SXS simulation, the times must be taken from the simulation as well.")
    
    if (parameters['Inference']['method']=='Minimization'): raise ValueError("Minimization is still a work in progress and is not supported yet. Please use the `Nested-sampler` method.")

    if not(parameters['Inference']['method']=='Nested-sampler'):

        parameters['Inference']['nlive']   = None
        parameters['Inference']['maxmcmc'] = None
        parameters['Inference']['nGuess']  = {'A' : parameters['Inference']['nGuess-A'], 'phi' : parameters['Inference']['nGuess-phi']}

    if(parameters['NR-data']['catalog'] == 'cbhdb' or parameters['NR-data']['catalog'] == 'charged_raw'): parameters['Model']['charge'] = 1
    else                                                                                                : parameters['Model']['charge'] = 0

    if not(parameters['NR-data']['add-const']==None): parameters['NR-data']['add-const'] = [float(value) for value in parameters['NR-data']['add-const'].split(',')]

    if ((parameters['Model']['template']=='KerrBinary' or parameters['Model']['template']=='TEOBPM') and not(parameters['NR-data']['l-NR']==2 and parameters['NR-data']['m']==2) and parameters['NR-data']['t-peak-22']==0.0): raise ValueError("The time of the peak of the 22 mode must be provided for the KerrBinary and TEOBPM models when fitting the HMs, to correctly rescale the NR-calibrated quantities.")

    if  (parameters['Model']['template']=='Damped-sinusoids'): 
        parameters['Model']['QNM-modes'] = '{}{}0'.format(parameters['NR-data']['l-NR'], parameters['NR-data']['m']) 
    elif(parameters['Model']['template']=='KerrBinary'          ): 
        if  (parameters['Model']['KerrBinary-version']=='London2018'): 
            parameters['Model']['QNM-modes'] = '220,221,210,330,331,320,440,430,2-20,2-21,2-10,3-30,3-31,3-20,4-40,4-30'
            if not(parameters['NR-data']['l-NR']==2 or parameters['NR-data']['l-NR']==3 or parameters['NR-data']['l-NR']==4): raise ValueError("The KerrBinary-London template is only available for l=2,3,4")
        elif(parameters['Model']['KerrBinary-version']=='Cheung2023'): 
            parameters['Model']['QNM-modes'] = '220,221,210,211,330,331,320,440,430,550,2-20,2-10'
            if not(parameters['NR-data']['l-NR']==2 or parameters['NR-data']['l-NR']==3 or parameters['NR-data']['l-NR']==4 or parameters['NR-data']['l-NR']==5): raise ValueError("The KerrBinary-Cheung template is only available for l=2,3,4,5")
        elif  (parameters['Model']['KerrBinary-version']=='noncircular'): 
            parameters['Model']['QNM-modes'] = '220,210,330'
            if not(parameters['NR-data']['l-NR']==2 or parameters['NR-data']['l-NR']==3 or parameters['NR-data']['l-NR']==4): raise ValueError("The KerrBinary-noncircular template is only available for l=2,3")  
    elif(parameters['Model']['template']=='TEOBPM'      ):
        parameters['Model']['QNM-modes'] = '220,221,210,211,330,331,320,321,310,311,440,441,430,431,420,421,410,411,550,551'
        if not(parameters['NR-data']['l-NR']==2 or parameters['NR-data']['l-NR']==3 or parameters['NR-data']['l-NR']==4  or parameters['NR-data']['l-NR']==5): raise ValueError("The TEOBPM template is only available for l=2,3,4,5")

    print('\n\n\nFIXME: print updated vars\n\n\n')

    return parameters

#Description of the package. Printed on stdout if --help option is given.
usage="""\n\n %prog --config-file config.ini\n
Inference package targeting ringdown modeling of numerical relativity waveforms.

Options syntax: default values (which also implies the variable type) and sections of the configuration file where each parameter should be passed are declared below.
By convention, booleans are represented by the integers [0,1].
To use default values, do not include the parameter in the configuration file: empty fields are interpreted as empty strings.
A dot is present at the end of each description line and is not to be intended as part of the default value.                                                                                                                                                                         default=None)

    *************************************************
    * Parameters to be passed to the [I/O] section. *
    *************************************************

        run-type         Type of run. Available options: ['full', 'post-processing', 'plot-NR-only'].                        Default: 'full'.
        screen-output    Boolean to divert stdout and stderr to files or to screen.                                          Default: 0.
        show-plots       Boolean to show results plots.                                                                      Default: 0.
        outdir           Path of the output directory.                                                                       Default: './'.

    *****************************************************
    * Parameters to be passed to the [NR-data] section. *
    *****************************************************

        download         Boolean to ask for the download of the requested SXS NR simulation.                                 Default 1.
        dir              Absolute path of NR local data.                                                                     Default: ''.
        catalog          NR catalog used. Available options: ['SXS', 'RIT', 'RWZ-env', 'Teukolsky', 'cbhdb', 'charged_raw', 'fake_NR']. Default: 'SXS'.
        ID               Simulation ID to be considered. Example for SXS: 0305. Example for Teukolsky: \
                         `a_0.7_A_0.141_w_1.4_ingoing_ang_15`.                                                               Default: 0305.
        extrap-order     Extrapolation order of the `SXS` simulations. Smaller N is better for ringdown \
              (data.black-holes.org/waveforms/index.html). Available options: ['2', '3', '4'].                               Default: 2.
        res-level        Resolution level of the simulation. For `SXS`: -1 selects the maximum available resolution. \
              Available values for Teukosly data: [1,...,9] (lowest to highest). Fixes `res-nx` and `res-nl`.                Default: -1.
        res-nx           Number of collocation points in the radial direction [only for Teukolsky data]. \
            Overwrites `res-level`.                                                                                          Default: 0. 
        res-nl           Number of collocation points in the angular direction [only for Teukolsky data]. \
            Overwrites `res-level`.                                                                                          Default: 0.
        pert-order       Perturbation order to consider in Teukolsky data. Available options: ['lin', 'scd'].                Default: `lin`.
        l-NR             Polar NR spherical index to be fitted, possibly different than QNM ones, \
            since mixing between different l happens.                                                                        Default: 2.
        m                Angular spherical index to be fitted (same for IMR and QNMs), since only modes with same m do mix.  Default: 2.
        error            Method to compute the NR error. Available options for `SXS`: \
                         ['constant-X', 'align-with-mismatch-all', 'align-with-mismatch-res-only', 'align-at-peak'], \
                         for `Teukolsky`: ['constant-X', 'resolution'] where X is the constant value selected by the user, \
                         for `RIT`: ['constant-X']. For 'fake_NR': ['gaussian-X', 'from-SXS-NR'] where X is the standard \
                         deviation of the Gaussian distribution of the noise.                                                Default: 'align-with-mismatch-res-only'.
        error-t-min      Lower time to be used in the computation of the NR error with the 'align-with-mismatch' option, expressed as minus the percentace of the peak time.     Default: 4e-3.
        error-t-max      Upper time to be used in the computation of the NR error with the 'align-with-mismatch' option, expressed as minus the percentace of the peak time.     Default: 3e-1.
        add-const        Parameter of the complex constant to be added to the fit template. Required to account for spurious \
                         effects in simulations. Example format: '--add-const A,phi'.                                        Default: '0.0,0.0'.
        properties-file  Path to the file containing additional properties of the NR simulation in `.csv` format. \
                         Follows the conventions of: `github.com/GCArullo/noncircular_BBH_fits/tree/main/Parameters_to_fit.  Default: ''.
        t-peak-22        Time of the peak of the 22 mode. Used as reference time in KerrBinary model. Must be passed when \
                         fitting HMs with KerrBinary.                                                                        Default: 0.0.                         
        waveform-type    Type of waveform to be used. Available options: ['strain', 'psi4'].                                 Default: 'strain'.

    ************************************************************
    * Parameters to be passed to the [Injection-data] section. *
    ************************************************************
        
        modes            Modes that will be included in the generated QNMs strain. Example: '220,221'.                       Default: '220,221,320'.
        times            Mode to choose the times at which to compute the NR strain. Options: ['from-metadata', \
                         'from-SXS-NR']. If the error is taken from the SXS simulation, the times must be taken \
                         from the SXS sim as well.                                                                           Default: 'from-SXS-NR'.
        noise            Noise injection option. If None, the noise is not added to the simulated Kerr QNMs data; \
            if '1', the noise is added to the data. Options: None, '1'.                                                      Default: None.
        tail             Option to add the tail to the simulated Kerr QNMs data; if '1', the tail is added to the data. \
            Options: None, '1'.                                                                                              Default: None.

    ***************************************************
    * Parameters to be passed to the [Model] section. *
    ***************************************************
    
        template                         Fitting template. Available options: ['Damped-sinusoids', 'Kerr', 'Kerr-Damped-sinusoids',\
              'KerrBinary', 'TEOBPM'].                                                                                                                                  Default: 'Kerr'.
        N-DS-modes                       Number of free modes in the ringdown model if 'Damped-sinusoids' in template. Otherwise, ignored.                                Default: 1.
        QNM-modes                        List of modes of the ringdown model, if 'Kerr' in template. Otherwise, ignored. \
                                         Example format: '220,221,320'.                                                                                                   Default: '220,221,320'.
        QQNM-modes                       List of quadratic modes of the ringdown model if 'Kerr' in template. Otherwise, ignored. \
                                         Example format: '--QQNM-modes ``Px220x321,Px220x221', i.e. (child_term x parent1 x parent2), \
                                         where the child mode is assumed to be equal to the selected (l_NR,m) multipole and child_term=P,M \
                                         (parent frequencies sum or difference).                                                                                          Default: ''.
        Kerr-tail                        Boolean to add a tail factor to the Kerr template.                                                                               Default: 0.
        Kerr-tail-modes                  Modes to which a tail will be added in the fitting template. Example format: '22,32'.                                            Default: '22'.
        KerrBinary-version               Option to select the version of the KerrBinary model to be used. Available options: ['London2018', 'Cheung2023', 'noncircular'].     Default: 'London2018'.
        KerrBinary-amplitudes-nc-version Option to select the version of the KerrBinary model amplitudes noncircular correction fit to be used. Format: `X-Y`, \ 
                                         where each entry selects a noncircular variable to be used for the noncircular fit, among ['bmrg','Emrg', 'Jmrg', 'Mf', 'af']. \
                                         Can also pass a single variable instead of two, but not less than one or more than two.                                          Default: ''.

        TEOB-NR-fit                      Boolean to fit also for NR calibration coefficients within TEOB model, otherwise, use default fits.                              Default: 0.
        TEOB-template                    TEOB template to be used. Available options: ['qc', 'nc']. The 'qc' version is defined in  \
                                         arXiv:1904.09550, arXiv:2001.09082, while the 'nc' in II.C of arXiv:2305.19336.                                                  Default: 'qc'.

    *******************************************************
    * Parameters to be passed to the [Inference] section. *
    *******************************************************

        For more information about the sampling algorithm, see the respective samplers documentation.

        method           Inference method to be used. Available options: ['Nested-sampler', 'Minimization'].                 Default: 'Nested-sampler'.
        
        t-start          Start time of the fit and reference time of amplitudes [M units]. \
            Relative to complex strain amplitude peak time.                                                                  Default: 20.
        t-end            End time of the fit and reference time of amplitudes [M units]. \
            Relative to complex strain amplitude peak time.                                                                  Default: 140.
        dt-scd           Positive delay between the complex strain amplitude peak time of (child) second order modes \
                         and (parent) linear modes. Used to define linear amplitudes at the same time of secondary ones.     Default: 0.0.

        ***************************************
        * Nested-sampler specific parameters. *
        ***************************************

        likelihood       Likelihood type to be used. Available options: ['gaussian', 'laplace'].                             Default: 'gaussian'.
        sampler          Which sampler to use. Available options: ['cpnest', 'raynest'].                                     Default: 'cpnest'.
        nlive            Number of live points to be used for the sampling.                                                  Default: 256.
        maxmcmc          Number of maximum Markov Chain Monte Carlo steps to be used during the sampling.                    Default: 256.
        seed             Seed for the random initialisation of the sampler.                                                  Default: 1234.
        nnest            Number of nested samplers to run in parallel ('massively-parallel' branch only).                    Default: 1.
        nensemble        Total number of ensemble processes running. nensemble = nnest * N_ev, where N_ev is the number \
                         of live points being substituted at each NS step. Requires N_ev << nlive. \
                         Also n_cpu = nnest+nensemble.                                                                       Default: 1.

        *************************************
        * Minimization specific parameters. *
        *************************************  

            The minimization:

                - is bounded within the selected prior bounds;
                - is seeded by a starting value, which can be either set by the user, or will be randomly selected within \
                  the prior bounds. In the latter case, a user-given number of seeds will be used and the best one will
                  be kept to initialize the main minimization loop;
                - is forced to run for a minimum number of iterations, and to stop after a maximum number of iterations; 
        
            min-method       Method to be used in the scipy.least_squares() function. Available options: ['lm', 'None'].         Default: 'lm'.
            min-iter-min     Minimum number of iterations for the minimization algorithm.                                        Default: 1.
            min-iter-max     Maximum number of iterations for the minimization algorithm.                                        Default: 1000.
            n-random-seeds   Number of random seeds to be used to initialize the minimization.                                   Default: 1.

        
    ****************************************************
    * Parameters to be passed to the [Priors] section. *
    ****************************************************   

        Parameters names and default bounds for all available models are documented in the `read_default_bounds` function of the `inference.py` module.
        
        Prior default bounds can be changed by adding 'param-min=value' or 'param-max=value' to this section, where `param` is the name of the parameter under consideration.

        User-controlled starting values for the minimization can be set by adding`'param-start=value` to the [Priors] section, where `param` is the name of the parameter under consideration. User-defined starting values overrun the `seeding` option for that parameter.
        
    *******************************************************************
    * Parameters to be passed to the [Mismatch-PSD-settings] section. *
    *******************************************************************  
        asd-path            Path to the ASD file. Default: https://dcc.ligo.org/ligo-t1800044/public.
        direction           Where to apply the smoothing in the PSD before doing the FFT. If below, it applies to low frequencies, if above to high frequencies, if below-and-above on both. Default: below.
        n_FFT_points        Number of iterations for values of the points that are used to compute the PSD. Default: 1.
        n_iterations_C1     Number of iteriations for the C1 algorithm. Default: 1.
        window_DX           Minimum window size for smoothing on the right side. Default: 0.8.
        window_DX_max       Maximum window size for smoothing on the right side. Default: 10.
        n_window_DX         Number of steps for the right-side windowing. Default: 1.
        window_SX           Minimum window size for smoothing on the left side. Default: 0.8.
        window_SX_max       Maximum window size for smoothing on the left side. Default: 10.
        n_window_SX         Number of steps for the left-side windowing. Default: 1.
        steepness           Minimum steepness parameter for smoothing. Default: 7.
        steepness_max       Maximum steepness parameter for smoothing. Default: 200.
        n_steepness         Number of steps in the steepness parameter range. Default: 1.
        saturation_DX       Minimum saturation value for the right-side windowing. Default: 1.0.
        saturation_DX_max   Maximum saturation value for the right-side windowing. Default: 5.0.
        n_saturation_DX     Number of steps for right-side saturation values. Default: 1.
        saturation_SX       Minimum saturation value for the left-side windowing. Default: 1.0.
        saturation_SX_max   Maximum saturation value for the left-side windowing. Default: 5.0.
        n_saturation_SX     Number of steps for left-side saturation values. Default: 1.

    *************************************************
    * Parameters to be passed in the Flags section. *
    *************************************************

        C1_flag              Enables or disables C1 fixing on the PSD after smoothing application.
                             - 1: Enable C1 iterations.
                             - 0: Disable C1 iterations.
                             Default: 1.

        clear_directory      Controls whether the output directory for the smoothing section is cleared before the run.
                             - 1: Clear the directory before execution.
                             - 0: Keep existing files.
                             Default: 1.

        mismatch_print_flag  Determines whether to print mismatch information (e.g. the scalar products involved in the mismatch).
                             - 1: Print mismatch values.
                             - 0: Do not print mismatch values.
                             Default: 0.

        mismatch_section_plot_flag  
                             Determines whether to plot sanity check plots regarding the mismatch section (for instance, the windowed PSD vs the original one).
                             - 1: Generate and save mismatch section plots.
                             - 0: Do not generate plots.
                             Default: 0.

        compare_TD_FD        Enables comparison between Time Domain (TD) and Frequency Domain (FD) mismatches.
                             - 1: Compute and compare both TD and FD mismatches.
                             - 0: Skip comparison.
                             Default: 0.

    

    ******************************************************
    * Parameters to be passed to the [Mismatch-GW-parameters] section. *
    ******************************************************
        M                The mass of the remnant (in solar masses).                                 Default: 60.
        dL               The luminosity distance of the source with respect to the observer.        Default: 410.
        ra               Right ascension (in radiants).                                             Default: 1.375.
        dec              Declination (in radiants).                                                 Default: -0.2108.
        psi              Polarization angle (in radiants).                                          Default: 2.659.
        
"""
                                                     
try:
    import art
    my_art = art.text2art("            Launching     bayRing") # Return ASCII text (default font)
except: print("* Warning: The `art` package could not be imported. Please consider installing it locally for best visual renditions using `pip install art`.")

__ascii_art__ = """\n\n \u001b[\u001b[38;5;39m
                                         @.
                                        &  @
                                        @  ,
                                        (
                                                       *
                                            &            @
                                       #    @        @
                                       @             .    ,
                                       *    .             @
                                                     @
                                                     ,    &
                                      (     #             @           @
                                      *     @                       @   @
                                      *     &       /
                                            .       @      #       @     @          *
*   @  %       *       @       &     @                     %                      @    &          *    @     &    @     @
                                                    *      *              @      @      @     @
                                             &                    @                        %
                                                                 .&        @   @
                                                   .        @                &
                                             @                   @
                                                   @
                                             *               @  @
                                                   .            &
                                                              %&
                                              *
                                              .
                                              @    @
                                              
                                               @  .
                                               /
                                                 @
\u001b[0m"""

max_len_keyword = len('KerrBinary-amplitudes-nc-version')
