import ast, json, math, os, re, shutil, subprocess, sys
try:                import configparser
except ImportError: import ConfigParser as configparser

import pyRing.utils    as pyRing_utils
from pyRing.initialise import store_git_info

def _clean_float(value):

    value = float(value)
    if(abs(value) < 1e-12): value = 0.0
    return float("{:.12g}".format(value))

def parse_start_time_values(raw_value):

    """

    Parse the `t-start` configuration value.

    Accepted forms are:
    - a scalar value, e.g. `30.0`;
    - a comma-separated or Python-style list/tuple, e.g. `20,30,40`;
    - an inclusive colon range, e.g. `20:40:5`.

    """

    raw_value = str(raw_value).strip()
    if(raw_value == ''):
        raise ValueError("The `t-start` option cannot be empty.")

    if(':' in raw_value):
        range_values = [value.strip() for value in raw_value.split(':')]
        if(len(range_values) != 3):
            raise ValueError("Invalid `t-start` range `{}`. Use `start:stop:step`.".format(raw_value))

        start, stop, step = [_clean_float(value) for value in range_values]
        if(step == 0.0):
            raise ValueError("Invalid `t-start` range `{}`. The step cannot be zero.".format(raw_value))
        if((stop - start)*step < 0.0):
            raise ValueError("Invalid `t-start` range `{}`. The step sign must move from start to stop.".format(raw_value))

        values  = []
        current = start
        tol     = abs(step)*1e-10 + 1e-12
        if(step > 0.0):
            while(current <= stop + tol):
                values.append(_clean_float(current))
                current = current + step
        else:
            while(current >= stop - tol):
                values.append(_clean_float(current))
                current = current + step
    else:
        try:
            literal_value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            literal_value = raw_value

        if(isinstance(literal_value, (list, tuple))):
            values = [_clean_float(value) for value in literal_value]
        else:
            values = [_clean_float(literal_value)]

    if(len(values) == 0):
        raise ValueError("The `t-start` option must provide at least one value.")
    if(len(set(values)) != len(values)):
        raise ValueError("The `t-start` option contains duplicate values: {}.".format(values))

    return values

def _clean_int(value):

    value = float(value)
    if not(value.is_integer()):
        raise ValueError("Expected an integer value, got `{}`.".format(value))

    return int(value)

def _safe_numeric_value(raw_value):

    node = ast.parse(str(raw_value).strip(), mode='eval')
    allowed_names = {'pi': math.pi}
    allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div)
    allowed_unaryops = (ast.UAdd, ast.USub)

    def evaluate(subnode):
        if isinstance(subnode, ast.Expression):
            return evaluate(subnode.body)
        if isinstance(subnode, ast.Constant) and isinstance(subnode.value, (int, float)):
            return float(subnode.value)
        if isinstance(subnode, ast.Name) and subnode.id in allowed_names:
            return allowed_names[subnode.id]
        if isinstance(subnode, ast.UnaryOp) and isinstance(subnode.op, allowed_unaryops):
            value = evaluate(subnode.operand)
            return value if isinstance(subnode.op, ast.UAdd) else -value
        if isinstance(subnode, ast.BinOp) and isinstance(subnode.op, allowed_binops):
            left, right = evaluate(subnode.left), evaluate(subnode.right)
            if isinstance(subnode.op, ast.Add): return left + right
            if isinstance(subnode.op, ast.Sub): return left - right
            if isinstance(subnode.op, ast.Mult): return left * right
            if isinstance(subnode.op, ast.Div): return left / right
        raise ValueError("Invalid numeric expression `{}`.".format(raw_value))

    return evaluate(node)

def parse_angle_values(raw_value, option_name='inclination'):

    raw_value = str(raw_value).strip()
    if(raw_value == ''):
        raise ValueError("The {} option cannot be empty.".format(option_name))

    if(':' in raw_value):
        range_values = [value.strip() for value in raw_value.split(':')]
        if(len(range_values) != 3):
            raise ValueError("Invalid {} range `{}`. Use `start:stop:step`.".format(option_name, raw_value))

        start, stop, step = [_safe_numeric_value(value) for value in range_values]
        values = parse_start_time_values("{}:{}:{}".format(start, stop, step))
    else:
        try:
            literal_value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            literal_value = raw_value

        if(isinstance(literal_value, (list, tuple))):
            values = [_clean_float(_safe_numeric_value(value)) for value in literal_value]
        else:
            values = [_clean_float(_safe_numeric_value(value)) for value in str(raw_value).split(',')]

    if(len(values) == 0):
        raise ValueError("The {} option must provide at least one value.".format(option_name))
    if(len(set(values)) != len(values)):
        raise ValueError("The {} option contains duplicate values: {}.".format(option_name, values))

    return values

def parse_int_values(raw_value, option_name):

    raw_value = str(raw_value).strip()
    if(raw_value == ''):
        raise ValueError("The `{}` option cannot be empty.".format(option_name))

    if(':' in raw_value):
        values = [_clean_int(value) for value in parse_start_time_values(raw_value)]
    else:
        try:
            literal_value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            literal_value = raw_value

        if(isinstance(literal_value, (list, tuple))):
            values = [_clean_int(value) for value in literal_value]
        else:
            values = [_clean_int(value) for value in str(raw_value).split(',')]

    if(len(values) == 0):
        raise ValueError("The `{}` option must provide at least one value.".format(option_name))

    return values

def _parse_compact_nr_mode(token):

    match = re.fullmatch(r'([1-9]\d*)([+-]?\d+)', token.strip())
    if(match is None):
        raise ValueError("Invalid compact NR mode `{}`. Use e.g. `22`, `3-3`, or Python pairs like `(2, 2)`.".format(token))

    return (int(match.group(1)), int(match.group(2)))

def _normalise_nr_mode_pairs(raw_modes):

    modes = []
    for mode in raw_modes:
        if(isinstance(mode, str)):
            modes.append(_parse_compact_nr_mode(mode))
        elif(isinstance(mode, (list, tuple)) and len(mode) == 2):
            modes.append((_clean_int(mode[0]), _clean_int(mode[1])))
        else:
            raise ValueError("Invalid NR mode `{}`. Use `(l, m)` pairs.".format(mode))

    return modes

def parse_nr_mode_values(raw_value):

    raw_value = str(raw_value).strip()
    if(raw_value == ''):
        raise ValueError("The `NR-modes` option cannot be empty.")

    try:
        literal_value = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        literal_value = raw_value

    if(isinstance(literal_value, (list, tuple))):
        if(len(literal_value) == 2 and all(isinstance(value, (int, float)) for value in literal_value)):
            modes = _normalise_nr_mode_pairs([literal_value])
        else:
            modes = _normalise_nr_mode_pairs(literal_value)
    else:
        modes = _normalise_nr_mode_pairs([token for token in str(raw_value).split(',') if token.strip()])

    return validate_nr_mode_values(modes)

def nr_mode_values_from_l_m(l_values, m_values):

    if(len(l_values) == 1 and len(m_values) > 1):
        modes = [(l_values[0], m_value) for m_value in m_values]
    elif(len(m_values) == 1 and len(l_values) > 1):
        modes = [(l_value, m_values[0]) for l_value in l_values]
    elif(len(l_values) == len(m_values)):
        modes = list(zip(l_values, m_values))
    else:
        raise ValueError("When `NR-modes` is not set, `l-NR` and `m` must be scalars, equal-length lists, or one scalar plus one list.")

    return validate_nr_mode_values(modes)

def validate_nr_mode_values(modes):

    if(len(modes) == 0):
        raise ValueError("At least one NR mode must be provided.")

    normalised_modes = []
    for l_value, m_value in modes:
        l_value, m_value = int(l_value), int(m_value)
        if(l_value < 2):
            raise ValueError("Invalid NR mode ({}, {}). The spherical index l must be at least 2.".format(l_value, m_value))
        if(abs(m_value) > l_value):
            raise ValueError("Invalid NR mode ({}, {}). The condition |m| <= l is required.".format(l_value, m_value))
        normalised_modes.append((l_value, m_value))

    if(len(set(normalised_modes)) != len(normalised_modes)):
        raise ValueError("The NR mode list contains duplicate values: {}.".format(normalised_modes))

    return normalised_modes

def format_start_time_label(t_start):

    label = "{:.12g}".format(float(t_start))
    label = label.replace('-', 'm').replace('+', '').replace('.', 'p')

    return "t_start_{}M".format(label)

def start_time_output_dir(base_outdir, t_start):

    return os.path.join(base_outdir, format_start_time_label(t_start))

def format_nr_mode_label(l_value, m_value):

    m_label = str(int(m_value)).replace('-', 'm')

    return "mode_l{}_m{}".format(int(l_value), m_label)

def nr_mode_output_dir(base_outdir, nr_mode):

    return os.path.join(base_outdir, format_nr_mode_label(*nr_mode))

def get_start_time_values(parameters):

    return list(parameters['Inference'].get('t-start-list', [parameters['Inference']['t-start']]))

def get_nr_mode_values(parameters):

    return list(parameters['NR-data'].get('NR-mode-list', [(parameters['NR-data']['l-NR'], parameters['NR-data']['m'])]))

def _copy_config_to_output(config_file, outdir, run_type):

    try:
        if(run_type == 'full' and config_file is not None):
            shutil.copy2(config_file, outdir)
    except: pass

    return

def _redirect_output(outdir, screen_output):

    if not(screen_output):
        sys.stdout = open(os.path.join(outdir,'stdout_bayRing.txt'), 'w')
        sys.stderr = open(os.path.join(outdir,'stderr_bayRing.txt'), 'w')

    return

def _is_git_repository():

    try:
        return subprocess.call(['git', 'rev-parse', '--is-inside-work-tree'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    except OSError:
        return False

def _store_git_info(outdir):

    if not(_is_git_repository()):
        print("The current directory is not a git repository. Git info will not be stored.")
        return

    store_git_info(outdir)

    return

def set_shared_output(outdir, screen_output, config_file, run_type):

    """

    Set output files that are shared by all start times in a scan.

    """

    if not os.path.exists(outdir): os.makedirs(outdir)

    _redirect_output(outdir, screen_output)
    _store_git_info(outdir)
    _copy_config_to_output(config_file, outdir, run_type)

    return

def set_output(outdir, screen_output, method, config_file, run_type, shared_files=True, redirect_streams=True):

    """

    Set the output directory and the output to the screen.

    Parameters
    ----------

    outdir : str
        Output directory.

    screen_output : bool
        If True, the output is printed on the screen.

    method : str
        Method used to obtain the results with which the results will be obtained.
    
    Returns
    -------

    Nothing, but creates the output directory and sets the output to screen.

    """
        
    if not os.path.exists(outdir):                                     os.makedirs(outdir)
    if not os.path.exists(os.path.join(outdir,'Algorithm')):           os.makedirs(os.path.join(outdir,'Algorithm'))
    if not os.path.exists(os.path.join(outdir,'Algorithm/Mismatch')):  os.makedirs(os.path.join(outdir,'Algorithm/Mismatch'))
    if not os.path.exists(os.path.join(outdir,'Peak_quantities')):     os.makedirs(os.path.join(outdir,'Peak_quantities'))
    if(method=='Nested-sampler'):
        if not os.path.exists(os.path.join(outdir,'Plots','Chains')):  os.makedirs(os.path.join(outdir,'Plots','Chains'))
    if not os.path.exists(os.path.join(outdir,'Plots')):               os.makedirs(os.path.join(outdir,'Plots'))
    if not os.path.exists(os.path.join(outdir,'Plots','Results')):     os.makedirs(os.path.join(outdir,'Plots','Results'))
    if not os.path.exists(os.path.join(outdir,'Plots','Comparisons')): os.makedirs(os.path.join(outdir,'Plots','Comparisons'))

    if(redirect_streams):
        _redirect_output(outdir, screen_output)

    if(shared_files):
        _store_git_info(outdir)
        _copy_config_to_output(config_file, outdir, run_type)

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
        'run-type'                 : 'full',
        'screen-output'            : 0,
        'show-plots'               : 0,
        'extract-damping-time-flag': 1,
        'outdir'                   : './',
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
        'NR-modes'         : '',
        'error'            : 'align-with-mismatch-res-only',
        'error-t-min'      : 3e-1,
        'error-t-max'      : 4e-3,
        'add-const'        : '0.0,0.0',
        'properties-file'  : '',
        'fits-file'        : '',
        't-peak-22'        : 0.0,
        'waveform-type'    : 'strain',
        },

        'Injection-data':
        {
        'modes'            : '220,221,320',
        'times'            : 'from-SXS-NR',
        'noise'            : None,
        'tail'             : 0.0,
        'parameters'       : '',
        },

        'Model':
        {
        'template'                         : 'Kerr'       ,
        'N-DS-modes'                       : 1            ,
        'N-DS-tails'                       : 0            ,
        'QNM-modes'                        : '220,221,320',
        'QQNM-modes'                       : ''           ,
        'Kerr-tail'                        : 0            ,
        'Kerr-tail-modes'                  : '22'         ,
        'KerrBinary-version'               : 'London2018' ,
        'KerrBinary-final-state-nc-version': ''           ,
        'KerrBinary-amplitudes-nc-version' : ''           ,
        'TEOB-template'                    : 'HypTan'     ,
        'TEOB-global-fit'                  : 1            ,
        'TEOB-merger-data'                 : 0            ,
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
        'n-start-time-workers': 1            ,
        'n-mode-workers'   : 1               ,

        't-start'          : 20.0 ,
        't-end'            : 140.0,
        'dt-scd'           : 0.0  ,
        
        'min-method'       : 'trf',
        'min-iter-max'     : 1000,
        'n-random-seeds'   : 16  ,
        'linear-inversion-eigenvalue-tol': 1e-10,
        },

        'Mismatch-PSD-settings':
        {
        'asd-path'              : ''               ,
        'obs-time'              : 0.               ,
        'direction'             : 'below-and-above',
        'window_DX'             : 0.8              ,
        'window_DX_max'         : 10.0             ,
        'window_SX'             : 0.8              ,
        'window_SX_max'         : 10.0             ,
        'n_window_DX'           : 1                ,
        'n_window_SX'           : 1                ,
        'steepness'             : 7.               ,
        'steepness_max'         : 200.             ,
        'n_steepness'           : 1                ,
        'saturation_DX'         : 1.               ,
        'saturation_DX_max'     : 5.               ,
        'n_saturation_DX'       : 1                ,
        'saturation_SX'         : 1.               ,
        'saturation_SX_max'     : 5.               ,
        'n_saturation_SX'       : 1                ,
        'n_FFT_points'          : 1                ,
        'n_iterations_C1'       : 1      
        },

        'Mismatch-GW-parameters':
        {
        'M'                    : 60     ,
        'dL'                   : 410    ,
        'ra'                   : 1.375  ,
        'dec'                  : -0.2108,
        'psi'                  : 2.659  ,
        'azimuth'              : 0.0    ,
        'inclination'          : '0:pi:pi/4',
        'polarisation'         : '0:3*pi/4:pi/4',
        'hm-include-negative-m': 1
        },

        'Flags': 
        {
        'apply_window'                 : 1,
        'C1_flag'                      : 1,
        'clear_directory'              : 1,
        'compare_TD_FD'                : 0,
        'mismatch_print_flag'          : 0,
        'mismatch_section_plot_flag'   : 0,
        'compute_hm_mismatch'          : 1,
        }

    }
    if Config.has_option('Injection-data', 'Kerr-parameters'):
        raise ValueError(
            "[Injection-data] Kerr-parameters is no longer supported. "
            "Use [Injection-data] parameters with the current parameter names."
        )

    #General input read.
    for parameters_section in parameters.keys():

        pyRing_utils.print_subsection(f'[{parameters_section}]')

        try:
            for key in list(parameters[parameters_section].keys()):
                keytype = type(parameters[parameters_section][key])
                try:
                    if(parameters_section == 'Mismatch-GW-parameters' and key == 'polarisation' and Config.has_option(parameters_section, 'polarization') and not(Config.has_option(parameters_section, key))):
                        raw_value = Config.get(parameters_section, 'polarization')
                    else:
                        raw_value = Config.get(parameters_section, key)
                    if(parameters_section == 'Inference' and key == 't-start'):
                        t_start_values = parse_start_time_values(raw_value)
                        parameters[parameters_section][key] = t_start_values[0]
                        parameters[parameters_section]['t-start-list'] = t_start_values
                    elif(parameters_section == 'NR-data' and key in ['l-NR', 'm']):
                        mode_index_values = parse_int_values(raw_value, key)
                        parameters[parameters_section][key] = mode_index_values[0]
                        parameters[parameters_section]['{}-list'.format(key)] = mode_index_values
                    elif(parameters_section == 'NR-data' and key == 'NR-modes'):
                        if(str(raw_value).strip() != ''):
                            nr_mode_values = parse_nr_mode_values(raw_value)
                            parameters[parameters_section][key] = raw_value
                            parameters[parameters_section]['NR-mode-list'] = nr_mode_values
                    elif(parameters_section == 'Mismatch-GW-parameters' and key in ['inclination', 'polarisation']):
                        angle_values = parse_angle_values(raw_value, key)
                        parameters[parameters_section][key] = raw_value
                        parameters[parameters_section]['{}-list'.format(key)] = angle_values
                    else:
                        parameters[parameters_section][key] = keytype(raw_value)
                except (KeyError, configparser.NoOptionError, TypeError):
                    pass

                # Other reading options
                # if   ('ds-modes'        in key): parameters[parameters_section][key] = json.loads(      Config.get(parameters_section, f'{key}')) # dict
                # elif ('quadratic-modes' in key): parameters[parameters_section][key] = eval(            Config.get(parameters_section, f'{key}')) # dict of lists
                # elif ('Kerr-tail-modes' in key): parameters[parameters_section][key] = eval(            Config.get(parameters_section, f'{key}')) # list
                # elif ('mode'            in key): parameters[parameters_section][key] = ast.literal_eval(Config.get(parameters_section,    key  )) # lists
                    
                print_value = parameters[parameters_section][key]
                if(parameters_section == 'Inference' and key == 't-start'):
                    print_value = parameters[parameters_section].get('t-start-list', [parameters[parameters_section][key]])
                    if(len(print_value) == 1): print_value = print_value[0]
                if(parameters_section == 'NR-data' and key in ['l-NR', 'm']):
                    print_value = parameters[parameters_section].get('{}-list'.format(key), [parameters[parameters_section][key]])
                    if(len(print_value) == 1): print_value = print_value[0]
                if(parameters_section == 'NR-data' and key == 'NR-modes'):
                    print_value = parameters[parameters_section].get('NR-mode-list', parameters[parameters_section][key])
                if(parameters_section == 'Mismatch-GW-parameters' and key in ['inclination', 'polarisation']):
                    print_value = parameters[parameters_section].get('{}-list'.format(key), parameters[parameters_section][key])
                    if(isinstance(print_value, list) and len(print_value) == 1): print_value = print_value[0]
                print("{name} : {value}".format(name=key.ljust(max_len_keyword), value=print_value))
        except (KeyError, configparser.NoSectionError, configparser.NoOptionError, TypeError): pass

    if('t-start-list' not in parameters['Inference']):
        parameters['Inference']['t-start-list'] = [float(parameters['Inference']['t-start'])]
    if(parameters['Inference']['n-start-time-workers'] < 1):
        raise ValueError("Invalid start-time parallelization option: `n-start-time-workers` must be at least 1.")
    if(parameters['Inference']['n-mode-workers'] < 1):
        raise ValueError("Invalid mode parallelization option: `n-mode-workers` must be at least 1.")

    if('NR-mode-list' not in parameters['NR-data']):
        l_values = parameters['NR-data'].get('l-NR-list', [parameters['NR-data']['l-NR']])
        m_values = parameters['NR-data'].get('m-list', [parameters['NR-data']['m']])
        parameters['NR-data']['NR-mode-list'] = nr_mode_values_from_l_m(l_values, m_values)
    parameters['NR-data']['l-NR'], parameters['NR-data']['m'] = parameters['NR-data']['NR-mode-list'][0]

    if('inclination-list' not in parameters['Mismatch-GW-parameters']):
        parameters['Mismatch-GW-parameters']['inclination-list'] = parse_angle_values(parameters['Mismatch-GW-parameters']['inclination'], 'inclination')
    if(Config.has_option('Mismatch-GW-parameters', 'polarization') and not(Config.has_option('Mismatch-GW-parameters', 'polarisation'))):
        parameters['Mismatch-GW-parameters']['polarisation'] = Config.get('Mismatch-GW-parameters', 'polarization')
        parameters['Mismatch-GW-parameters']['polarisation-list'] = parse_angle_values(parameters['Mismatch-GW-parameters']['polarisation'], 'polarisation')
    if('polarisation-list' not in parameters['Mismatch-GW-parameters']):
        parameters['Mismatch-GW-parameters']['polarisation-list'] = parse_angle_values(parameters['Mismatch-GW-parameters']['polarisation'], 'polarisation')

    # Cleanup specific parameters formatting
    if(parameters['Inference']['sampler'] == 'raynest'):
        print('Nnest + nensemble: ', parameters['Inference']['nnest'] + parameters['Inference']['nensemble'])
        if parameters['Inference']['nensemble'] < parameters['Inference']['nnest']: raise ValueError(f"Invalid parallelization options: input nensemble ( =  {parameters['Inference']['nensemble']}) cannot be smaller than input nnest ( = {parameters['Inference']['nnest']} ). ")

    # For Teukolsky, map the different resolution levels to their values of nx_, nl_.
    if(parameters['NR-data']['res-nx'] != 0 and parameters['NR-data']['res-nl'] != 0): parameters['NR-data']['res-level'] = "nx_"+str(parameters['NR-data']['res-nx'])+"_nl_"+str(parameters['NR-data']['res-nl'])
    if(parameters['NR-data']['error']=='from-SXS-NR'):
        if not(parameters['Injection-data']['times']=='from-SXS-NR'):
            raise ValueError("When the error is taken from the corresponding SXS simulation, the times must be taken from the simulation as well.")
    
    if(parameters['Inference']['method'] in ['Minimization', 'Linear-inversion']):

        parameters['Inference']['nlive']   = None
        parameters['Inference']['maxmcmc'] = None

    elif not(parameters['Inference']['method']=='Nested-sampler'):

        raise ValueError("Unknown inference method: {}.".format(parameters['Inference']['method']))

    if(parameters['NR-data']['catalog'] == 'cbhdb' or parameters['NR-data']['catalog'] == 'charged_raw'): parameters['Model']['charge'] = 1
    else                                                                                                : parameters['Model']['charge'] = 0

    if not(parameters['NR-data']['add-const']==None): parameters['NR-data']['add-const'] = [float(value) for value in parameters['NR-data']['add-const'].split(',')]
    injection_parameters = parameters['Injection-data']['parameters']
    if not(injection_parameters==''):
        parameters['Injection-data']['parameters'] = ast.literal_eval(injection_parameters)
    else:
        parameters['Injection-data']['parameters'] = None

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
        
        elif  (parameters['Model']['KerrBinary-version']=='Carullo2024'):
            parameters['Model']['QNM-modes'] = '220,210,330'
            if not(parameters['NR-data']['l-NR']==2 or parameters['NR-data']['l-NR']==3 or parameters['NR-data']['l-NR']==4): raise ValueError("The KerrBinary-Carullo2024 template is only available for l=2,3")
    
    elif(parameters['Model']['template']=='TEOBPM'      ):
        parameters['Model']['QNM-modes']     = '220,221,210,211,330,331,320,321,310,311,440,441,430,431,420,421,410,411,550,551'
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

        run-type                    Type of run. Available options: ['full', 'post-processing', 'plot-NR-only'].                        Default: 'full'.
        
        screen-output               Boolean to divert stdout and stderr to files or to screen.                                          Default: 0.
        
        show-plots                  Boolean to show results plots.                                                                      Default: 0.
        
        extract-damping-time-flag   Flag to extract the damping time from the amplitude when plotting the waveform comparison.          Default: 1.

        outdir                      Path of the output directory.                                                                       Default: './'.

    *****************************************************
    * Parameters to be passed to the [NR-data] section. *
    *****************************************************

        download                Boolean to ask for the download of the requested SXS NR simulation.                                 Default 1.
        
        dir                     Absolute path of NR local data.                                                                     Default: ''.
        
        catalog                 NR catalog used. Available options: ['SXS', 'RIT', 'RWZ-env', 'Teukolsky', 'cbhdb', 'charged_raw', 'injections']. Default: 'SXS'.
        
        ID                      Simulation ID to be considered. Example for SXS: 0305. Example for Teukolsky: \
                                `a_0.7_A_0.141_w_1.4_ingoing_ang_15`.                                                               Default: 0305.
        
        extrap-order            Extrapolation order of the `SXS` simulations. Smaller N is better for ringdown \
                     (data.black-holes.org/waveforms/index.html). Available options: ['2', '3', '4'].                               Default: 2.
        
        res-level               Resolution level of the simulation. For `SXS`: -1 selects the maximum available resolution. \
                     Available values for Teukosly data: [1,...,9] (lowest to highest). Fixes `res-nx` and `res-nl`.                Default: -1.
        
        res-nx                  Number of collocation points in the radial direction [only for Teukolsky data]. \
                   Overwrites `res-level`.                                                                                          Default: 0. 
        
        res-nl                  Number of collocation points in the angular direction [only for Teukolsky data]. \
                   Overwrites `res-level`.                                                                                          Default: 0.
        
        pert-order              Perturbation order to consider in Teukolsky data. Available options: ['lin', 'scd'].                Default: `lin`.
        
        l-NR                    Polar NR spherical index to be fitted, possibly different than QNM ones, \
                   since mixing between different l happens. Can be a scalar or a list paired with `m`.                             Default: 2.
        
        m                       Angular spherical index to be fitted (same for IMR and QNMs), since only modes with same m do mix. \
                                Can be a scalar or a list paired with `l-NR`.                                                       Default: 2.

        NR-modes                Optional list of NR `(l,m)` modes to fit in one invocation. Accepts Python pairs such as \
                                `[(2,2),(3,3)]` or compact tokens such as `22,33,4-4`. Overrides list values passed to \
                                `l-NR` and `m` when non-empty.                                                                      Default: ''.
        
        error                   Method to compute the NR error. Available options for `SXS`: \
                                ['constant-X', 'align-with-mismatch-all', 'align-with-mismatch-res-only', 'align-at-peak'], \
                                for `Teukolsky`: ['constant-X', 'resolution'] where X is the constant value selected by the user, \
                                for `RIT`: ['constant-X', 'late-time-const-error']. For 'injections': ['gaussian-X', 'from-SXS-NR'] where X is the standard \
                                deviation of the Gaussian distribution of the noise.                                                Default: 'align-with-mismatch-res-only'.
        
        error-t-min             Lower time to be used in the computation of the NR error with the 'align-with-mismatch' option, expressed as minus the percentace of the peak time. Example: t_min_mm = t_peak * (1-`error-t-min`). Default: 3e-1.
        
        error-t-max             Upper time to be used in the computation of the NR error with the 'align-with-mismatch' option, expressed as minus the percentace of the peak time. Example: t_max_mm = t_peak * (1-`error-t-max`). Default: 4e-3.
        
        add-const               Parameter of the complex constant to be added to the fit template. Required to account for spurious \
                                effects in simulations. Example format: '--add-const A,phi'.                                        Default: '0.0,0.0'.
        
        properties-file         Path to the file containing additional properties of the NR simulation in `.csv` format. \
                                Follows the conventions of: `github.com/GCArullo/noncircular_BBH_fits/tree/main/Parameters_to_fit.  Default: ''.

        fits-file               Path to the file containing the fits of the NR simulation. Used for 'RatExp' template global fits.  Default: ''.
        
        t-peak-22               Time of the peak of the 22 mode. Used as reference time in KerrBinary model. Must be passed when \
                                fitting HMs with KerrBinary.                                                                        Default: 0.0.                         
        
        waveform-type           Type of waveform to be used. Available options: ['strain', 'psi4'].                                 Default: 'strain'.

    ************************************************************
    * Parameters to be passed to the [Injection-data] section. *
    ************************************************************
        
        modes            Modes that will be included in the generated QNMs strain. Example: '220,221'.                       Default: '220,221,320'.
        
        times            Mode to choose the times at which to compute the NR strain. Options: ['from-metadata', \
                         'from-SXS-NR']. If the error is taken from the SXS simulation, the times must be taken \
                         from the SXS sim as well.                                                                           Default: 'from-SXS-NR'.
        
        noise            Noise injection option. If None, the noise is not added to the injection data; \
            if '1', the noise is added to the data. Options: None, '1'.                                                      Default: None.
        
        tail             Option to add the Kerr tail to the injection data; if '1', the tail is added to the data. \
            Options: None, '1'.                                                                                              Default: None.
        parameters       Dictionary used to generate a template injection from config values when catalog='injections'. \
                         Required keys are: `t_start`, `t_end`, `dt` and either `q` or both `m1`, `m2`. \
                         Kerr-like templates also require `Mf`, `af`; NR-informed templates compute those from \
                         binary parameters. Waveform keys match the selected template parameter names, e.g. \
                         `ln_A_220`, `phi_220`, `f_0`, `tau_0`, `phi`, `phi_mrg_22`.                                       Default: ''.

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

        KerrBinary-version               Option to select the version of the KerrBinary model to be used. Available options: ['London2018', 'Cheung2023', 'Carullo2024'].     Default: 'London2018'.

        KerrBinary-final-state-nc-version Option to select the version of the KerrBinary model final-state noncircular correction fit. Format: `X-Y`, \
                                         where each entry selects a noncircular variable to be used for the noncircular fit, among ['Emrg', 'Jmrg']. \
                                         Required only for Carullo2024 template injections.                                                                         Default: ''.

        KerrBinary-amplitudes-nc-version Option to select the version of the KerrBinary model amplitudes noncircular correction fit to be used. Format: `X-Y`, \ 
                                         where each entry selects a noncircular variable to be used for the noncircular fit, among ['bmrg','Emrg', 'Jmrg', 'Mf', 'af']. \
                                         Can also pass a single variable instead of two, but not less than one or more than two.                                          Default: ''.
        
        TEOB-template                    TEOB template to be used. Available options: ['HypTan', 'RatExp']. The 'HypTan' version is defined in  \
                                         arXiv:1904.09550, arXiv:2001.09082, while the 'RatExp' in II.C of arXiv:2305.19336. 
                                         
                                         Additionally, if 'RatExp' template is selected, the TEOB-merger-data flag has to be set to 1, and NR merger data has \
                                         to be provided in [NR-data][properties-file].                                                                                    Default: 'HypTan'.

        N-DS-tails                       Number of free tails in the ringdown model if 'Damped-sinusoids' in template. Otherwise, ignored.                                Default: 0.

        TEOB-global-fit                  Boolean to use the NR-calibrated global fits of the TEOB model. 
                                         If 1: 
                                            - For 'HypTan' template, this selects the internally coded quasi-circular fits in pyRing.
                                            - For 'RatExp' template, fits-file containing global fit coefficients have to be provided in [NR-data][fits-file].
                                         If 0: Runs local fits for the amplitude and phase coefficients.                                                                  Default: 1.

        TEOB-merger-data                 Boolean flag to switch between using the values of the amplitude and frequency at the peak of the modes as given \ 
                                         by the NR merger data (TEOB-merger-data = 1, to be provided in [NR-data][properties-file]) or by the quasi-circular fits \
                                         (TEOB-merger-data = 0).                                                                                                          Default: 0.

    *******************************************************
    * Parameters to be passed to the [Inference] section. *
    *******************************************************

        For more information about the sampling algorithm, see the respective samplers documentation.

        method           Inference method to be used. Available options: ['Nested-sampler', 'Minimization', 'Linear-inversion']. Default: 'Nested-sampler'.
        
        t-start          Start time of the fit and reference time of amplitudes [M units]. \
            Relative to complex strain amplitude peak time. Can be a scalar, a comma/list of values \
            such as `20,30,40`, or a colon range `start:stop:step` such as `20:40:5`. \
            When multiple start times are supplied, bayRing repeats the fit in one process and stores \
            each run under `outdir/t_start_<value>M/`.                                                                        Default: 20.
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

        n-start-time-workers
                         Number of start-time fits to run in parallel when `t-start` supplies multiple values. \
                         Each fit is run in a separate process and keeps its products under its \
                         `outdir/t_start_<value>M/` directory. This is in addition to sampler-level \
                         parallelism set by options such as `nnest` and `nensemble`.                                           Default: 1.

        n-mode-workers
                         Number of NR-mode fits to run in parallel when multiple `(l,m)` modes are supplied. \
                         Each fit is run in a separate process and keeps its products under its \
                         `outdir/mode_l<l>_m<m>/` directory, with start-time subdirectories below it when needed.               Default: 1.

        *****************************************
        * Point-estimate specific parameters.   *
        *****************************************  

            The minimization:

                - is bounded within the selected prior bounds;
                - is seeded by a starting value, which can be either set by the user, or will be randomly selected within \
                  the prior bounds. In the latter case, a user-given number of seeds will be used and the best one will
                  be kept as the point estimate;
                - is stopped after a maximum number of function evaluations per seed; 
        
            min-method       Method to be used in the scipy.least_squares() function. Available options: ['trf', 'dogbox']. Default: 'trf'.
            
            min-iter-max     Maximum number of iterations for the minimization algorithm.                                        Default: 1000.
            
            n-random-seeds   Number of random seeds to be used to initialize the minimization.                                   Default: 16.

            The linear inversion:

                - solves directly for Kerr QNM, quadratic-mode, and fixed-exponent tail complex amplitudes;
                - requires each tail exponent p_tail_* to be fixed, since tail exponents are nonlinear;

            linear-inversion-eigenvalue-tol
                             Absolute floor applied to Fisher-matrix eigenvalues in the Kerr linear inversion.                    Default: 1e-10.

        
    ****************************************************
    * Parameters to be passed to the [Priors] section. *
    ****************************************************   

        Parameters names and default bounds for all available models are documented in the `read_default_bounds` function of the `inference.py` module.
        
        Prior default bounds can be changed by adding 'param-min=value' or 'param-max=value' to this section, where `param` is the name of the parameter under consideration.

        Parameters can be fixed by adding 'fix-param=value' to this section, where `param` is the name of the parameter under consideration.

        User-controlled starting values for the minimization can be set by adding`'param-start=value` to the [Priors] section, where `param` is the name of the parameter under consideration. User-defined starting values overrun the `seeding` option for that parameter.
        
    *******************************************************************
    * Parameters to be passed to the [Mismatch-PSD-settings] section. *
    *******************************************************************  

        asd-path            Path to the ASD file. Default: https://dcc.ligo.org/ligo-t1800044/public.

        obs-time            Time of observation [s]. If not provided, default is computed as T=1/df, where df is the minimum frequency resolution in the PSD frequency array.
        
        direction           Where to apply the smoothing in the PSD before doing the FFT. If 'below', it applies to low frequencies, if 'above' to high frequencies, if 'below-and-above' on both. Default: 'below-and-above'.
        
        n_FFT_points        Number of iterations for values of the points that are used to compute the PSD. Default: 1.
        
        n_iterations_C1     Number of iteriations for algorithm that transforms functions to their C^1 versions. Default: 1.
        
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

        apply_window                Choose wheter to apply window at the edges of the PSD or not. 
                                    Default: 1.

        C1_flag                     Enables or disables C1 fixing on the PSD after smoothing application.
                                    Default: 1.

        clear_directory             Controls whether the output directory for the smoothing section is cleared before the run.
                                    Default: 1.

        compare_TD_FD               Enables comparison between Time Domain (TD) and Frequency Domain (FD) mismatches.
                                    Default: 0.                     

        mismatch_print_flag         Determines whether to print mismatch information (e.g. the scalar products involved in the mismatch).
                                    Default: 0.

        mismatch_section_plot_flag  
                                    Determines whether to plot sanity check plots regarding the mismatch section (for instance, the windowed PSD vs the original one).
                                    Default: 0.    

        compute_hm_mismatch         Boolean to compute detector-projected summed-higher-mode mismatch diagnostics after a multi-mode scan.
                                    Default: 1.

    ********************************************************************
    * Parameters to be passed to the [Mismatch-GW-parameters] section. *
    ********************************************************************
        
        M                The mass of the remnant (in solar masses).                                 Default: 60.
        
        dL               The luminosity distance of the source with respect to the observer.        Default: 410.
        
        ra               Right ascension (in radiants).                                             Default: 1.375.
        
        dec              Declination (in radiants).                                                 Default: -0.2108.
        
        psi              Polarization angle (in radiants) used by fixed-polarisation diagnostics.    Default: 2.659.

        azimuth          Source-frame azimuthal phase entering the spin-weighted spherical harmonic \
                         recomposition of multiple NR modes.                                        Default: 0.0.

        inclination      Inclination values used for summed-higher-mode mismatch diagnostics. Accepts a scalar, \
                         comma/list values, or an inclusive range `start:stop:step`; expressions using `pi` are \
                         accepted.                                                                  Default: `0:pi:pi/4`.

        polarisation     Polarisation-angle values used for summed-higher-mode mismatch diagnostics. The reported \
                         summed-HM mismatch is marginalised over these samples by retaining the minimum mismatch. \
                         Pass a scalar to compute at one fixed polarisation. Expressions using `pi` are accepted. \
                                                                                                      Default: `0:3*pi/4:pi/4`.

        hm-include-negative-m
                         Boolean to include non-precessing negative-m counterparts via \
                         h_{l,-m}=(-1)^l h^*_{lm} when they are not explicitly fitted.              Default: 1.

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

max_len_keyword = max(
    len('KerrBinary-amplitudes-nc-version'),
    len('KerrBinary-final-state-nc-version'),
)
