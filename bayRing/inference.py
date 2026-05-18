import importlib, itertools as it, math, numpy as np, os, pandas as pd, traceback
from scipy.optimize  import least_squares as l_s

try:                import configparser
except ImportError: import ConfigParser as configparser

from cpnest.nest2pos import draw_posterior, compute_weights
import cpnest, cpnest.model
import pyRing.utils      as pyRing_utils
import bayRing.postprocess as postprocess
import bayRing.template_waveforms as template_waveforms
import bayRing.utils       as utils

twopi                  = 2.*np.pi
max_parameter_name_len = len('ln_A_tail_22')
linear_inversion_methods = ['Linear-inversion']
point_estimate_methods   = ['Minimization'] + linear_inversion_methods

# CPNest workers using the spawn start method must pickle the inference model.
# Register the factory-built classes as module globals so pickle can import them.
_DYNAMIC_INFERENCE_MODEL_PREFIX  = 'DynamicInferenceModel_'
_DYNAMIC_INFERENCE_MODEL_CLASSES = {}

def _regularized_symmetric_inverse(matrix, eigenvalue_tol=None):

    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5*(matrix + matrix.T)

    eigvals, eigvecs = np.linalg.eigh(matrix)

    if eigenvalue_tol is None:
        max_eig = np.max(np.abs(eigvals)) if len(eigvals)>0 else 0.0
        eigenvalue_tol = np.finfo(float).eps * max(matrix.shape) * max(max_eig, 1.0)

    eigvals_regularized = np.maximum(eigvals, eigenvalue_tol)
    inverse = np.dot(eigvecs/eigvals_regularized, eigvecs.T)

    return inverse, eigvals, eigvals_regularized

def _errors_from_covariance(names, covariance):

    diagonal = np.diag(covariance)
    errors   = np.sqrt(np.maximum(diagonal, 0.0))

    return dict(zip(names, errors))

def estimate_least_squares_parameter_errors(names, least_squares_result, eigenvalue_tol=None):

    """

    Estimate one-sigma parameter errors from a scipy least_squares result.

    The minimization residuals are already weighted by the configured data
    errors, so the local covariance estimate is the inverse weighted Fisher
    matrix J^T J. This is the nonlinear analogue of the linear-inversion
    covariance used below.

    """

    if not(hasattr(least_squares_result, 'jac')):
        raise ValueError("Cannot estimate minimization errors because the least_squares result has no Jacobian.")

    jacobian = np.asarray(least_squares_result.jac, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[1] != len(names):
        raise ValueError(
            "Cannot estimate minimization errors from a Jacobian with shape {} for {} parameters.".format(
                jacobian.shape, len(names)
            )
        )

    fisher = np.dot(jacobian.T, jacobian)
    covariance, eigvals, eigvals_regularized = _regularized_symmetric_inverse(fisher, eigenvalue_tol=eigenvalue_tol)
    errors = _errors_from_covariance(names, covariance)

    return errors, covariance, eigvals, eigvals_regularized

def _base_class_descriptor(base):

    return '{}:{}'.format(base.__module__, base.__qualname__)

def _dynamic_inference_model_class_name(base):

    return _DYNAMIC_INFERENCE_MODEL_PREFIX + _base_class_descriptor(base).encode('utf-8').hex()

def _resolve_base_class(descriptor):

    module_name, qualname = descriptor.split(':', 1)
    base = importlib.import_module(module_name)
    for attr in qualname.split('.'):
        base = getattr(base, attr)

    return base

def _register_dynamic_inference_model(base, model_class):

    class_name = _dynamic_inference_model_class_name(base)

    model_class.__name__    = class_name
    model_class.__qualname__ = class_name
    model_class.__module__  = __name__

    globals()[class_name] = model_class
    _DYNAMIC_INFERENCE_MODEL_CLASSES[class_name] = model_class

    return model_class

def __getattr__(name):

    if name.startswith(_DYNAMIC_INFERENCE_MODEL_PREFIX):
        try:
            descriptor = bytes.fromhex(name[len(_DYNAMIC_INFERENCE_MODEL_PREFIX):]).decode('utf-8')
            base       = _resolve_base_class(descriptor)
        except Exception as exc:
            raise AttributeError("module '{}' has no attribute '{}'".format(__name__, name)) from exc

        return Dynamic_InferenceModel(base)

    raise AttributeError("module '{}' has no attribute '{}'".format(__name__, name))

def read_parameter_bounds(Config, configparser, basename, fullname, default_bounds):
    
    single_bounds = [0.0,0.0]
    
    try:                                                                                                     single_bounds[0] = Config.getfloat("Priors", fullname+'-min')
    except (KeyError, configparser.NoOptionError, configparser.NoSectionError, configparser.NoSectionError): single_bounds[0] = default_bounds[basename][0]
    try:                                                                                                     single_bounds[1] = Config.getfloat("Priors", fullname+'-max')
    except (KeyError, configparser.NoOptionError, configparser.NoSectionError, configparser.NoSectionError): single_bounds[1] = default_bounds[basename][1]

    print(('{} : [{}, {}]'.format(fullname.ljust(max_parameter_name_len), single_bounds[0], single_bounds[1])))

    return single_bounds

def is_linear_inversion_method(method):

    return str(method) in linear_inversion_methods

def is_point_estimate_method(method):

    return str(method) in point_estimate_methods

def read_parameter_start_minimization(Config, configparser, fullname, bounds, nseeds=1, rng=None):
    
    
    try:                                                                                                     start_value = Config.getfloat("Priors", fullname+'-start')
    except (KeyError, configparser.NoOptionError, configparser.NoSectionError, configparser.NoSectionError): 
        if rng is None:
            rng = np.random.default_rng()
        start_values = rng.uniform(bounds[0], bounds[1], int(nseeds))
        if int(nseeds)>1:
            start_values[0] = 0.5*(bounds[0] + bounds[1])
        start_value  = start_values[0] if int(nseeds)==1 else start_values

    print(('{} : {}'.format(fullname.ljust(max_parameter_name_len), start_value)))

    return start_value

def store_evidence_to_file(parameters, Evidence):

    """

    Function to store the evidence to a file.

    Parameters
    ----------

    parameters: dict
        Dictionary containing the input parameters.

    Evidence: float
        Evidence of the model.

    Returns
    -------

    Nothing, but it stores the evidence to a file.

    """
    
    outFile_evidence = open(os.path.join( parameters['I/O']['outdir'],'Algorithm/Evidence.txt'), 'w')
    outFile_evidence.write('logZ\n')
    outFile_evidence.write('{}'.format(Evidence))
    outFile_evidence.close()

    return

def read_default_bounds(wf_model, TEOB_template=''):

    default_bounds_DS        = {'ln_A': [-20.0, 5.0]            ,
                                'phi' : [0.0, twopi]            ,
                                'f'   : [-2.0/twopi,2.0/twopi]  ,
                                'tau' : [1,50]                  }

    default_bounds_DS_tail   = {'ln_A_tail': [-20.0, 5.0]       ,
                                'phi_tail' : [0.0, twopi]       ,
                                'p_tail'   : [-10.0,  3.0]      }

    default_bounds_Kerr      = {'ln_A': [-20.0, 5.0]            ,
                                'phi' : [0.0, twopi]            }

    default_bounds_Kerr_tail = {'ln_A_tail': [-20.0, 5.0]       ,
                                'phi_tail' : [0.0, twopi]       ,
                                'p_tail'   : [-20.0,  20.0]     }

    default_bounds_TEOBPM    = {'phi_mrg': [0.0  , twopi]       ,
                                'c3A'    : [-10.0, 10.0 ]       ,
                                'c3p'    : [-10.0, 10.0 ]       ,
                                }
    if not(TEOB_template=='SEOBNRv5'):
        default_bounds_TEOBPM['c4p'] = [-10.0, 10.0]
    if(TEOB_template=='SEOBNRv5'):
        default_bounds_TEOBPM['c2A'] = [1.0e-4, 5.0]
        default_bounds_TEOBPM['c2p'] = [1.0e-4, 5.0]
    elif not(TEOB_template=='HypTan'):
        default_bounds_TEOBPM['c2A']          = [-10.0, 10.0]
        default_bounds_TEOBPM['c2p']          = [-10.0, 10.0]

    if(  wf_model=='Damped-sinusoids'     ): default_bounds = default_bounds_DS
    elif(wf_model=='Damped-sinusoids-tail'): default_bounds = default_bounds_DS_tail
    elif(wf_model=='Kerr'                 ): default_bounds = default_bounds_Kerr
    elif(wf_model=='Kerr-tail'            ): default_bounds = default_bounds_Kerr_tail
    elif(wf_model=='KerrBinary'           ): default_bounds = {'phi': [0.0, twopi]}
    elif(wf_model=='TEOBPM'               ): default_bounds = default_bounds_TEOBPM

    return default_bounds

def teobpm_optional_reference_bounds():

    return {
        'DeltaT'              : [0.0, 50.0],
        'A_peak_over_nu'      : [1.0e-8, 1.0],
        'omg_peak'           : [1.0e-4, 1.0],
        'A_peakdot_over_nu'   : [-1.0, 1.0],
        'A_ref_over_nu'       : [1.0e-8, 1.0],
        'A_refdot_over_nu'    : [-1.0, 1.0],
        'A_refdotdot_over_nu' : [-1.0, 1.0],
        'omg_ref'            : [1.0e-4, 1.0],
        'c2A'                : [1.0e-4, 20.0],
        'c2p'                : [1.0e-4, 20.0],
    }

def teobpm_quadratic_44_window_bounds():

    return {
        'quad44_window_delay'    : [-10.0, 50.0],
        'quad44_window_width'    : [1.0e-3, 80.0],
        'quad44_window_steepness': [0.1, 10.0],
    }

def has_teobpm_optional_prior(Config, fullname):

    return (
        Config.has_option('Priors', 'fix-'+fullname) or
        Config.has_option('Priors', fullname+'-min') or
        Config.has_option('Priors', fullname+'-max')
    )

def _railing_header_and_bounds(inference_model):

    header = ''
    bounds = []
    for (i,param) in enumerate(inference_model.names):
        bounds.append(inference_model.bounds[i])
        header +='{par}_low\t{par}_up\t'.format(par=param)

    return header, bounds

def _save_railing_parameters(outdir, railing_parameters, header, suffix=''):

    filename = 'Parameters_prior_railing{}.txt'.format(suffix)
    np.savetxt(os.path.join(outdir, 'Algorithm', filename), np.column_stack(railing_parameters), fmt= "%d", header=header)

    return

def _has_posterior_samples(results_object, names):

    for param in names:
        try:
            values = results_object[param]
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        if(np.isscalar(values)):
            continue
        try:
            if(len(values) > 1):
                return True
        except TypeError:
            try:
                if(values.size > 1):
                    return True
            except AttributeError:
                pass

    return False

def _scalar_result_value(value):

    if(np.isscalar(value)):
        return float(value)

    try:
        return float(value.item())
    except (AttributeError, TypeError, ValueError):
        pass

    values = list(value)
    if(len(values) != 1):
        raise ValueError("Expected a scalar point estimate, got {} values.".format(len(values)))

    return float(values[0])

def _is_finite(value):

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def _point_estimate_railing_flags(value, bounds, tolerance):

    low_bound, high_bound = bounds
    prior_width = high_bound - low_bound
    if not(_is_finite(value)) or prior_width < 0.0:
        return False, False

    tolerance_width = abs(prior_width)*tolerance/100.0
    low_rail  = value <= low_bound  + tolerance_width
    high_rail = value >= high_bound - tolerance_width

    return low_rail, high_rail

def point_estimate_railing_check(results_object, inference_model, outdir, tolerance=2.0):

    """

    Check whether point-estimate parameters sit close to prior boundaries.

    For each parameter, the lower or upper flag is set when the point estimate
    falls within ``tolerance`` percent of the corresponding prior edge.

    """

    try:
        print('\n* Checking point-estimate parameters for railing...')
        railing_parameters = []
        header, bounds = _railing_header_and_bounds(inference_model)

        for (i,param) in enumerate(inference_model.names):
            value = _scalar_result_value(results_object[param])
            low_rail, high_rail = _point_estimate_railing_flags(value, bounds[i], tolerance)
            if(low_rail):
                railing_parameters.append(1)
                print('{}'.format(param.ljust(15)), 'is within {:.1f}% of the lower prior bound.'.format(tolerance))
            else:
                railing_parameters.append(0)
            if(high_rail):
                railing_parameters.append(1)
                print('{}'.format(param.ljust(15)), 'is within {:.1f}% of the upper prior bound.'.format(tolerance))
            else:
                railing_parameters.append(0)

        _save_railing_parameters(outdir, railing_parameters, header)
    except:
        print("\n* Warning: Point-estimate prior railing file generation failed with error: {}.".format(traceback.print_exc()))

    return

def railing_check(results_object, inference_model, outdir, nlive=None, seed=None, tolerance=2.0, check_chains=True):

    """
    
    Function to check if the posterior samples are railing against the prior bounds.

    Parameters
    ----------

    results_object: cpnest.results.Results object
        Results object from the cpnest run.

    inference_model: cpnest.model.Model object
        Inference object for the inference run.

    outdir: str
        Output directory for the inference run.

    nlive: int
        Number of live points used for the inference run.

    tolerance: float, optional
        Tolerance [%] for the railing check. Default is 2%.

    Returns
    -------

    Nothing, but saves the railing check results to a file in the output directory and print the information to the screen. For each of the (lower, upper) bounds, 1 indicates railing, 0 indicates no railing.
    
    """

    try:
        print('\n* Checking for railing...')
        railing_parameters  = []
        header, bounds = _railing_header_and_bounds(inference_model)
        for (i,param) in enumerate(inference_model.names):
            Prior_bins = np.linspace(bounds[i][0], bounds[i][-1], 100)
            low_rail, high_rail = pyRing_utils.railing_check(samples=results_object[param], prior_bins=Prior_bins, tolerance=tolerance)
            if(low_rail):
                railing_parameters.append(1)
                print('{}'.format(param.ljust(15)), 'is railing against the lower prior bound.')
            else:
                railing_parameters.append(0)
            if(high_rail):
                railing_parameters.append(1)
                print('{}'.format(param.ljust(15)), 'is railing against the upper prior bound.')
            else:
                railing_parameters.append(0)
        _save_railing_parameters(outdir, railing_parameters, header)
        
        if check_chains and nlive is not None and seed is not None and np.sum(railing_parameters) > 0:
            print('\n* Identifying chain with railing...')
            try   : chains = [np.genfromtxt(os.path.join(outdir, f'Algorithm/chain_{nlive}_{seed_x}.txt'), names = True, deletechars="") for seed_x in [0,1,2,3]]
            except: chains = [np.genfromtxt(os.path.join(outdir, f'Algorithm/chain_{nlive}_{seed}.txt'), names = True, deletechars="")]
            for chain_number,chain in enumerate(chains):
                log_evs, log_wts    = compute_weights(chain['logL'], nlive)
                weighted_post       = draw_posterior(chain, log_wts)
                railing_parameters_chain = []
                header_chain = ''
                for (i,param) in enumerate(inference_model.names):
                    if railing_parameters[2*i] == 1 or railing_parameters[2*i+1] == 1: 
                        Prior_bins = np.linspace(bounds[i][0], bounds[i][-1], 100)
                        low_rail, high_rail = pyRing_utils.railing_check(samples=weighted_post[param], prior_bins=Prior_bins, tolerance=2.0)
                        header_chain +='{par}_low\t{par}_up\t'.format(par=param)
                        if(low_rail):
                            railing_parameters_chain.append(1)
                            print('{}'.format(param.ljust(15)), f'is railing against the lower prior bound for the chain {chain_number}.')
                        else:
                            railing_parameters_chain.append(0)
                        if(high_rail):
                            railing_parameters_chain.append(1)
                            print('{}'.format(param.ljust(15)), f'is railing against the upper prior bound for the chain {chain_number} .')
                        else:
                            railing_parameters_chain.append(0)
                    else:
                        continue
                _save_railing_parameters(outdir, railing_parameters_chain, header_chain, suffix='_{}'.format(chain_number))
    except:
        print("\n* Warning: Prior railing file generation failed with error: {}.".format(traceback.print_exc()))

    return

def minimization_railing_check(results_object, inference_model, outdir, tolerance=2.0):

    if(_has_posterior_samples(results_object, inference_model.names)):
        railing_check(results_object, inference_model, outdir, tolerance=tolerance, check_chains=False)
    else:
        point_estimate_railing_check(results_object, inference_model, outdir, tolerance=tolerance)

    return

def UNUSED_build_a_grid(self, x_max, x_min, delta_x, n_grid):

    x_tmp = []

    x_tt  = [[0 for i in range(len(x_max))]]

    i_tmp = 1
    j_tmp = 0
    while not(i_tmp == j_tmp):
        i_tmp = j_tmp
        for j in range(len(x_tt)):
            for i in range(len(x_max)):
                if x_tt[j][i]==0:
                    x_tmp = [x_tt[j][k] for k in range(len(x_max))]
                    x_tmp[i]+=1
                    
                    if not(x_tmp in x_tt):
                        j_tmp += 1
                        x_tt.append(x_tmp)

    i_tmp = 1
    j_tmp = 0
    while not(i_tmp == j_tmp):
        i_tmp = j_tmp
        for i in range(len(x_tt)):
            for j in range(len(x_tt)):
            
                x_tmp = [x_tt[i][k] + x_tt[j][k] for k in range(len(x_max))]
                
                if not(x_tmp in x_tt) and not(False in [x_tmp[i] < n_grid+1 for i in range(len(x_max))]):
                    j_tmp += 1
                    x_tt.append(x_tmp)

    y_tt = []

    for i in range(len(x_tt)):

        y_i_tt = []
        
        for j in range(len(x_max)):
            tmp = x_min[j] + x_tt[i][j]*delta_x[j]
            if tmp <= x_max[j]: y_i_tt.append(tmp)
            else              : y_i_tt.append(x_max[j])
            
        y_tt.append(y_i_tt)
        
    return y_tt

def Dynamic_InferenceModel(base):

    class_name = _dynamic_inference_model_class_name(base)
    if class_name in _DYNAMIC_INFERENCE_MODEL_CLASSES:
        return _DYNAMIC_INFERENCE_MODEL_CLASSES[class_name]

    class InferenceModel(base):

        """
        
        Inference model for the ringdown waveform.

        Parameters
        ----------

        data: array
            The data to be fitted.

        error: array
            The error on the data.

        wf_model: object
            The waveform model.

        ln_A_bounds: list
            The bounds on the amplitudes.

        method: string
            The method to be used for the fit.

        min_method: string
            The method to be used for the minimization.

        likelihood_kind: string
            The kind of likelihood to be used.
        
        """

        def __init__(self, data, error, wf_model, Config, method, min_method, likelihood_kind='gaussian'):

            self.data          = data
            self.error         = error
            self.wf_model      = wf_model
            self.kind          = likelihood_kind
            self.Kerr_modes    = self.wf_model.Kerr_modes
            self.N_ds_modes    = self.wf_model.N_ds_modes
            self.N_ds_tails    = getattr(self.wf_model, 'N_ds_tails', 0)
            self.TEOB_template = self.wf_model.TEOB_template
            self.TEOB_global_fit = self.wf_model.TEOB_global_fit
            self.TEOB_merger_data = self.wf_model.TEOB_merger_data 
            self.TEOB_mode_mixing = getattr(self.wf_model, 'TEOB_mode_mixing', 0)
            self.TEOB_counter_rotating = getattr(self.wf_model, 'TEOB_counter_rotating', 0)
            self.TEOB_quadratic_44 = getattr(self.wf_model, 'TEOB_quadratic_44', 0)
            self.TEOB_quadratic_44_window_end = float(getattr(self.wf_model, 'TEOB_quadratic_44_window_end', -1.0))
            self.TEOB_tapered_overtone_44 = getattr(self.wf_model, 'TEOB_tapered_overtone_44', 0)
            self.quadratic_modes = getattr(self.wf_model, 'quadratic_modes', None)
            self.min_method    = min_method
            self.Config        = Config

            self.names          = []
            self.bounds         = []
            self.fixed_params   = {}
            self.min_start_pars = {}

            pyRing_utils.print_section(f'{self.wf_model.wf_model} model')

            if(self.wf_model.wf_model=='Kerr'):
                
                self.tail            = self.wf_model.tail
                self.quadratic_modes = self.wf_model.quadratic_modes
                self.tail_modes      = self.wf_model.tail_modes

                default_bounds = read_default_bounds(self.wf_model.wf_model)   
                for (l_ring, m_ring, n) in self.Kerr_modes:
                    for name in default_bounds.keys():
                        fullname      = '{}_{}{}{}'.format(name, l_ring, m_ring, n)
                        try:
                            self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                        except(configparser.NoOptionError):
                            single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds)
                            self.names.append(fullname)
                            self.bounds.append(single_bounds)

                if(self.quadratic_modes is not None):

                    for quad_term in self.quadratic_modes:
                        for ((l,m,n),(l1,m1,n1),(l2,m2,n2)) in self.quadratic_modes[quad_term]:
                            for name in default_bounds.keys():

                                fullname      = '{}_{}_{}{}{}_{}{}{}_{}{}{}'.format(name, quad_term, l,m,n, l1,m1,n1, l2,m2,n2)
                                try:
                                    self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                                except(configparser.NoOptionError):
                                    single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds)
                                    self.names.append(fullname)
                                    self.bounds.append(single_bounds)

                if(self.tail):
                    default_bounds_tail = read_default_bounds(self.wf_model.wf_model+'-tail')   
                    for (l_ring, m_ring) in self.tail_modes:
                        for name in default_bounds_tail.keys():

                            fullname      = '{}_{}{}'.format(name, l_ring, m_ring)
                            try:
                                self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                            except(configparser.NoOptionError):
                                single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds_tail)
                                self.names.append(fullname)
                                self.bounds.append(single_bounds)
        
            elif(self.wf_model.wf_model=='Damped-sinusoids'):
                           
                default_bounds = read_default_bounds(self.wf_model.wf_model)
                for i,name in it.product(list(range(self.N_ds_modes)),default_bounds.keys()):

                    fullname      = '{}_{}'.format(name, i)
                    try:
                        self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                    except(configparser.NoOptionError):
                        single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds)
                        self.names.append(fullname)
                        self.bounds.append(single_bounds)

                default_bounds_DS_tail = read_default_bounds(self.wf_model.wf_model+'-tail')
                for i,name in it.product(list(range(self.N_ds_tails)),default_bounds_DS_tail.keys()):

                    fullname      = '{}_{}'.format(name,i)
                    try:
                        self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                    except(configparser.NoOptionError):
                        single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds_DS_tail)
                        self.names.append(fullname)
                        self.bounds.append(single_bounds)

            elif(self.wf_model.wf_model=='Kerr-Damped-sinusoids'):

                self.tail            = self.wf_model.tail
                self.quadratic_modes = self.wf_model.quadratic_modes
                self.tail_modes      = self.wf_model.tail_modes

                default_bounds_Kerr = read_default_bounds('Kerr')   
                for (l_ring, m_ring, n) in self.Kerr_modes:
                    for name in default_bounds_Kerr.keys():

                        fullname      = '{}_{}{}{}'.format(name, l_ring, m_ring, n)
                        try:
                            self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                        except(configparser.NoOptionError):
                            single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds_Kerr)
                            self.names.append(fullname)
                            self.bounds.append(single_bounds)

                if(self.quadratic_modes is not None):

                    for quad_term in self.quadratic_modes:
                        for ((l,m,n),(l1,m1,n1),(l2,m2,n2)) in self.quadratic_modes[quad_term]:
                            for name in default_bounds_Kerr.keys():

                                fullname      = '{}_{}_{}{}{}_{}{}{}_{}{}{}'.format(name, quad_term, l,m,n, l1,m1,n1, l2,m2,n2)
                                try:
                                    self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                                except(configparser.NoOptionError):
                                    single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds_Kerr)
                                    self.names.append(fullname)
                                    self.bounds.append(single_bounds)

                if(self.tail):
                    default_bounds_tail = read_default_bounds('Kerr-tail')   
                    for (l_ring, m_ring) in self.tail_modes:
                        for name in default_bounds_tail.keys():

                            fullname      = '{}_{}{}'.format(name, l_ring, m_ring)
                            try:
                                self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                            except(configparser.NoOptionError):
                                single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds_tail)
                                self.names.append(fullname)
                                self.bounds.append(single_bounds)

                default_bounds_DS = read_default_bounds('Damped-sinusoids')
                for i,name in it.product(list(range(self.N_ds_modes)),default_bounds_DS.keys()):

                    fullname      = '{}_{}'.format(name, i)
                    try:
                        self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                    except(configparser.NoOptionError):
                        single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds_DS)
                        self.names.append(fullname)
                        self.bounds.append(single_bounds)

            elif(self.wf_model.wf_model=='KerrBinary'):

                default_bounds = read_default_bounds(self.wf_model.wf_model)   
                for name in default_bounds.keys():
                    try:
                        self.fixed_params[name] = self.Config.getfloat("Priors",'fix-'+name)
                    except(configparser.NoOptionError):
                        single_bounds = read_parameter_bounds(Config, configparser, name, name, default_bounds)
                        self.names.append(name)
                        self.bounds.append(single_bounds)
 
            elif(self.wf_model.wf_model=='TEOBPM'):

                default_bounds_TEOBPM = read_default_bounds(self.wf_model.wf_model, TEOB_template=self.TEOB_template)   
                for name in default_bounds_TEOBPM.keys():
                    if self.TEOB_global_fit and name != 'phi_mrg':
                        continue
                    fullname = '{}_{}{}'.format(name, self.wf_model.l_NR, self.wf_model.m_NR)
                    try:
                        self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                    except(configparser.NoOptionError):
                        single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds_TEOBPM)
                        self.names.append(fullname)
                        self.bounds.append(single_bounds)

                if not(self.TEOB_global_fit):
                    optional_bounds_TEOBPM = teobpm_optional_reference_bounds()
                    for name in optional_bounds_TEOBPM.keys():
                        if name in default_bounds_TEOBPM:
                            continue
                        fullname = '{}_{}{}'.format(name, self.wf_model.l_NR, self.wf_model.m_NR)
                        if has_teobpm_optional_prior(self.Config, fullname):
                            try:
                                self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                            except(configparser.NoOptionError):
                                single_bounds = read_parameter_bounds(Config, configparser, name, fullname, optional_bounds_TEOBPM)
                                self.names.append(fullname)
                                self.bounds.append(single_bounds)

                if self.TEOB_mode_mixing:
                    requested_mode = (self.wf_model.l_NR, self.wf_model.m_NR)
                    parent_mode = template_waveforms.TEOB_MODE_MIXING_PARENTS.get(requested_mode)
                    if parent_mode is not None:
                        for name in default_bounds_TEOBPM.keys():
                            if self.TEOB_global_fit and name != 'phi_mrg':
                                continue
                            fullname = '{}_{}{}'.format(name, parent_mode[0], parent_mode[1])
                            try:
                                self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                            except(configparser.NoOptionError):
                                pass
                        if not(self.TEOB_global_fit):
                            optional_bounds_TEOBPM = teobpm_optional_reference_bounds()
                            for name in optional_bounds_TEOBPM.keys():
                                if name in default_bounds_TEOBPM:
                                    continue
                                fullname = '{}_{}{}'.format(name, parent_mode[0], parent_mode[1])
                                try:
                                    self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                                except(configparser.NoOptionError):
                                    pass

                if self.TEOB_counter_rotating:
                    default_bounds_counter = {
                        'ln_A_counter_scale': [-8.0, 0.0],
                        'phi_mrg_counter'   : [0.0, twopi],
                        'c3A_counter'       : default_bounds_TEOBPM['c3A'],
                        'c3p_counter'       : default_bounds_TEOBPM['c3p'],
                    }
                    if 'c4p' in default_bounds_TEOBPM:
                        default_bounds_counter['c4p_counter'] = default_bounds_TEOBPM['c4p']
                    if not(self.TEOB_template=='HypTan'):
                        default_bounds_counter['c2A_counter'] = default_bounds_TEOBPM['c2A']
                        default_bounds_counter['c2p_counter'] = default_bounds_TEOBPM['c2p']
                    counter_mode_label = '{}{}'.format(self.wf_model.l_NR, -self.wf_model.m_NR)
                    for name in default_bounds_counter.keys():
                        fullname = '{}_{}'.format(name, counter_mode_label)
                        try:
                            self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                        except(configparser.NoOptionError):
                            single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds_counter)
                            self.names.append(fullname)
                            self.bounds.append(single_bounds)
                    if(self.TEOB_template=='HypTan'):
                        optional_bounds_counter = {
                            'c2A_counter': teobpm_optional_reference_bounds()['c2A'],
                            'c2p_counter': teobpm_optional_reference_bounds()['c2p'],
                        }
                        for name in optional_bounds_counter.keys():
                            fullname = '{}_{}'.format(name, counter_mode_label)
                            if has_teobpm_optional_prior(self.Config, fullname):
                                try:
                                    self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                                except(configparser.NoOptionError):
                                    single_bounds = read_parameter_bounds(Config, configparser, name, fullname, optional_bounds_counter)
                                    self.names.append(fullname)
                                    self.bounds.append(single_bounds)

                if self.quadratic_modes is not None:
                    default_bounds_quadratic = read_default_bounds('Kerr')
                    for quad_term in self.quadratic_modes:
                        for modes in self.quadratic_modes[quad_term]:
                            label = '{}_{}{}{}_{}{}{}_{}{}{}'.format(
                                quad_term,
                                modes[0][0], modes[0][1], modes[0][2],
                                modes[1][0], modes[1][1], modes[1][2],
                                modes[2][0], modes[2][1], modes[2][2],
                            )
                            for name in default_bounds_quadratic.keys():
                                fullname = '{}_{}'.format(name, label)
                                try:
                                    self.fixed_params[fullname] = self.Config.getfloat("Priors", 'fix-'+fullname)
                                except(configparser.NoOptionError):
                                    single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds_quadratic)
                                    self.names.append(fullname)
                                    self.bounds.append(single_bounds)

                if self.TEOB_quadratic_44:
                    default_bounds_quadratic_44_window = teobpm_quadratic_44_window_bounds()
                    for name in default_bounds_quadratic_44_window.keys():
                        if name == 'quad44_window_width' and self.TEOB_quadratic_44_window_end >= 0.0:
                            if has_teobpm_optional_prior(self.Config, name):
                                raise ValueError(
                                    "quad44_window_width cannot be fixed or sampled when "
                                    "TEOB-quadratic-44-window-end is enabled."
                                )
                            continue
                        try:
                            self.fixed_params[name] = self.Config.getfloat("Priors",'fix-'+name)
                        except(configparser.NoOptionError):
                            if has_teobpm_optional_prior(self.Config, name):
                                single_bounds = read_parameter_bounds(Config, configparser, name, name, default_bounds_quadratic_44_window)
                                self.names.append(name)
                                self.bounds.append(single_bounds)

                    parent_mode_label = '22'
                    for name in default_bounds_TEOBPM.keys():
                        fullname = '{}_{}'.format(name, parent_mode_label)
                        try:
                            self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                        except(configparser.NoOptionError):
                            pass
                    optional_bounds_TEOBPM = teobpm_optional_reference_bounds()
                    for name in optional_bounds_TEOBPM.keys():
                        if name in default_bounds_TEOBPM:
                            continue
                        fullname = '{}_{}'.format(name, parent_mode_label)
                        try:
                            self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                        except(configparser.NoOptionError):
                            pass

                if self.TEOB_tapered_overtone_44:
                    default_bounds_tapered_overtone_44 = {
                        'ln_A_tapered_441': [-20.0, 5.0],
                        'phi_tapered_441' : [0.0, twopi],
                    }
                    for name in default_bounds_tapered_overtone_44.keys():
                        try:
                            self.fixed_params[name] = self.Config.getfloat("Priors",'fix-'+name)
                        except(configparser.NoOptionError):
                            single_bounds = read_parameter_bounds(Config, configparser, name, name, default_bounds_tapered_overtone_44)
                            self.names.append(name)
                            self.bounds.append(single_bounds)

            else:
                raise ValueError("Unknown template selected: {}".format(self.wf_model.wf_model))

            pyRing_utils.print_subsection('Fixed')
            pyRing_utils.print_fixed_parameters(self.fixed_params)
            
        def access_names(self):

            """
            
            Returns the names of the parameters.

            Parameters
            ----------

            None

            Returns
            -------

            names: list
                The names of the parameters.
            
            """
            
            return self.names

        def access_bounds(self):

            """

            Returns the bounds of the parameters.

            Parameters
            ----------

            None    

            Returns
            -------

            bounds: list
                The bounds of the parameters.

            """

            return self.bounds

        def model(self, x):

            """

            Returns the model.

            Parameters
            ----------

            x: array
                The parameters of the model.

            Returns
            -------

            fit_model: array
                The model to be used in the fit.

            """
            
            fit_model = self.wf_model.waveform(x, self.fixed_params)
            
            return fit_model

        def log_likelihood(self,x):

            """

            Returns the log-likelihood.

            Parameters
            ----------

            x: array
                The parameters of the model.

            Returns
            -------

            lh: float
                The log-likelihood.

            """
            
            if(self.kind=='gaussian'):
                err = 1e-16
                lh_r = -0.5 * np.sum(((np.real(self.data)-np.real(self.model(x)))/(np.real(self.error)+err))**2)
                lh_i = -0.5 * np.sum(((np.imag(self.data)-np.imag(self.model(x)))/(np.imag(self.error)+err))**2)
            #WARNING: needs testing!
            elif(self.kind=='laplace'):
                lh_r = -0.5 * np.sum(np.abs((np.real(self.data)-np.real(self.model(x)))/np.real(self.error)))
                lh_i = -0.5 * np.sum(np.abs((np.imag(self.data)-np.imag(self.model(x)))/np.imag(self.error)))

            return lh_r + lh_i
        
        def log_likelihood_ToMin(self,x):
        
            x_dict  = dict(zip(self.names, x))
            try:
                model = self.model(x_dict)
            except (FloatingPointError, OverflowError, TypeError, ValueError):
                return np.full(2*len(self.data), 1.0e30)
            if not np.all(np.isfinite(model)):
                return np.full(2*len(self.data), 1.0e30)
            err     = 1e-16
            res_r   = (np.real(self.data)-np.real(model))/(np.real(self.error)+err)
            res_i   = (np.imag(self.data)-np.imag(model))/(np.imag(self.error)+err)
            fun_min = np.concatenate((res_r, res_i))
            constraint_residuals = self.minimization_constraint_residuals(x_dict)
            if len(constraint_residuals)>0:
                fun_min = np.concatenate((fun_min, constraint_residuals))

            return fun_min

        def minimization_constraint_residuals(self, x):

            """

            Return penalty residuals for non-rectangular priors used by the nested sampler.

            `least_squares` enforces the rectangular parameter bounds directly; these residuals
            make the minimization respect the additional ordering constraints implemented in
            `log_prior`.

            """

            penalty_scale = 1e6
            residuals = []

            if('Damped-sinusoids' in self.wf_model.wf_model):
                # Order the frequencies per given polarisation (same as m1>m2 in LAL).
                for i in range(1, self.wf_model.N_ds_modes):
                    try:
                        f_i      = utils.get_param_override(self.fixed_params, x, 'f_{}'.format(i  ))
                        f_prev_i = utils.get_param_override(self.fixed_params, x, 'f_{}'.format(i-1))
                        violation = f_prev_i - f_i
                        if violation > 0.0: residuals.append(penalty_scale*violation)
                    except(KeyError):
                        pass

            if(('Kerr' in self.wf_model.wf_model) and self.wf_model.tail==1):
                tail_modes = getattr(self, 'tail_modes', self.wf_model.tail_modes)
                for (l_ring, m_ring) in tail_modes:
                    try:
                        p_tail_i    = utils.get_param_override(self.fixed_params, x, 'p_tail_{}{}'.format(l_ring            , m_ring            ))
                        p_tail_base = utils.get_param_override(self.fixed_params, x, 'p_tail_{}{}'.format(self.wf_model.l_NR, self.wf_model.m_NR))
                        violation   = p_tail_base - p_tail_i
                        if violation > 0.0: residuals.append(penalty_scale*violation)
                    except(KeyError):
                        pass

            return np.array(residuals)
        
        def log_prior(self,x):

            """

            Returns the log-prior. Impose a flat prior on all parameters and frequency ordering for damped-sinusoids.

            Parameters
            ----------

            x: array
                The parameters of the model.

            Returns
            -------

            -np.inf: float
                If the parameters are out of bounds.
            0.0: float
                If the parameters are in bounds, i.e. by default a flat prior on all parameters.

            """

            if not self.in_bounds(x): return -np.inf

            if('Damped-sinusoids' in self.wf_model.wf_model):
                # Order the frequencies per given polarisation (same as m1>m2 in LAL).
                for i in range(self.wf_model.N_ds_modes):
                    try:
                        f_1 = utils.get_param_override(self.fixed_params,x,'f_{}'.format(i  ))
                        f_2 = utils.get_param_override(self.fixed_params,x,'f_{}'.format(i-1))
                        if (f_1 < f_2): return -np.inf
                    except(KeyError):
                        pass

            # In the case of Kerr tails, order the tails by exponent
            if(('Kerr' in self.wf_model.wf_model) and self.wf_model.tail==1):
                tail_modes = getattr(self, 'tail_modes', self.wf_model.tail_modes)
                for (l_ring, m_ring) in tail_modes:
                    # FIXME: temporarily valid only for two modes. Eventually do it for an arbitrary number of modes.
                    p_tail_1 = utils.get_param_override(self.fixed_params,x,'p_tail_{}{}'.format(l_ring            , m_ring            ))
                    p_tail_2 = utils.get_param_override(self.fixed_params,x,'p_tail_{}{}'.format(self.wf_model.l_NR, self.wf_model.m_NR))

                    if (p_tail_1 < p_tail_2): return -np.inf

            return 0.0
    
    return _register_dynamic_inference_model(base, InferenceModel)
            
        
class Minimization_Algorithm():
      
    def __init__(self, inference_model, parameters):

        self.inference_model = inference_model
        self.bounds          = inference_model.access_bounds()
        self.names           = inference_model.access_names()

        self.iter_max        = parameters['Inference']['min-iter-max']
        self.n_random_seeds  = max(1, parameters['Inference']['n-random-seeds'])
        self.rng             = np.random.default_rng(parameters['Inference']['seed'])
        self.min_method      = self._least_squares_method(inference_model.min_method)

        # Convert bounds to a format compatible with `least_squares` arguments
        self.bounds_minim    = (np.array([self.bounds[i][0] for i in range(len(self.bounds))]), np.array([self.bounds[i][1] for i in range(len(self.bounds))]))

        self.start_values = self._initial_points()
        self.best_result  = None
        self.errors       = {}
        self.covariance   = None

    def _least_squares_method(self, requested_method):

        if requested_method is None or str(requested_method).lower() in ['', 'none']:
            requested_method = 'trf'

        requested_method = str(requested_method).lower()
        if requested_method not in ['trf', 'dogbox']:
            raise ValueError("Unknown minimization method: {}. Available options are: ['trf', 'dogbox'].".format(requested_method))

        return requested_method

    def _initial_points(self):

        print('\n* Minimization starting values:')

        start_columns = []
        for i,name_x in enumerate(self.names):
            start_values = read_parameter_start_minimization(self.inference_model.Config, configparser, name_x, self.bounds[i], self.n_random_seeds, rng=self.rng)
            start_values = np.asarray(start_values, dtype=float)

            if start_values.ndim == 0:
                start_values = np.full(self.n_random_seeds, float(start_values))
            else:
                start_values = np.array(start_values, dtype=float)

            start_columns.append(start_values)

        return np.column_stack(start_columns)

    def fun(self, x):
    
        function_to_minimize = self.inference_model.log_likelihood_ToMin(x)
    
        return function_to_minimize

    def minimize_likelihood(self):

        best_result = None

        for i,x0 in enumerate(self.start_values):
            result = l_s(self.fun,
                         x0,
                         bounds   = self.bounds_minim,
                         method   = self.min_method,
                         max_nfev = self.iter_max)

            print("* Minimization seed {}/{}: cost = {:.12e}, nfev = {}, success = {}".format(i+1, self.n_random_seeds, result.cost, result.nfev, result.success))

            if best_result is None or result.cost < best_result.cost:
                best_result = result

        if best_result is None:
            raise RuntimeError("Minimization failed before producing a result.")

        print("\n* Best minimization cost: {:.12e}".format(best_result.cost))
        if not(best_result.success):
            print("* Warning: best minimization result did not satisfy scipy's convergence criterion: {}".format(best_result.message))

        self.best_result = best_result
        self._estimate_parameter_errors()

        return best_result.x

    def _estimate_parameter_errors(self):

        try:
            self.errors, self.covariance, eigvals, eigvals_regularized = estimate_least_squares_parameter_errors(self.names, self.best_result)
            print("* Minimized Fisher eigenvalue range before regularization: [{:.12e}, {:.12e}]".format(np.min(eigvals), np.max(eigvals)))
            print("* Minimized Fisher condition after regularization: {:.12e}".format(np.max(eigvals_regularized)/np.min(eigvals_regularized)))
        except Exception as exc:
            self.errors     = dict((name, np.nan) for name in self.names)
            self.covariance = None
            print("* Warning: minimization error estimate failed: {}".format(exc))

class KerrLinearInversion_Algorithm():
      
    def __init__(self, inference_model, parameters):

        self.inference_model = inference_model
        self.names           = inference_model.access_names()
        self.name_set        = set(self.names)
        self.fixed_params    = inference_model.fixed_params
        self.eigenvalue_tol  = parameters['Inference']['linear-inversion-eigenvalue-tol']
        self.errors          = {}
        self.covariance      = None

        self._validate_model()
        self.solve_components, self.fixed_components = self._classify_linear_components()

        if len(self.solve_components)==0:
            raise ValueError("Linear inversion needs at least one free Kerr complex-amplitude pair.")

        solved_names = set()
        for component in self.solve_components:
            solved_names.add(component['ln_A_name'])
            solved_names.add(component['phi_name'])

        if self.name_set != solved_names:
            unresolved_names = sorted(self.name_set - solved_names)
            raise ValueError(
                "Linear inversion cannot solve non-linear or unsupported free parameters: {}. "
                "Fix these parameters in [Priors] or use Minimization.".format(', '.join(unresolved_names))
            )

    def _validate_model(self):

        if not(self.inference_model.wf_model.wf_model=='Kerr'):
            raise ValueError("Linear inversion is currently implemented only for the Kerr template.")

        if self.inference_model.kind != 'gaussian':
            raise ValueError("Linear inversion is available only with the gaussian likelihood.")

    def _component_status(self, ln_A_name, phi_name):

        ln_A_free = ln_A_name in self.name_set
        phi_free  = phi_name  in self.name_set

        if ln_A_free != phi_free:
            raise ValueError(
                "Linear inversion requires both `{}` and `{}` to be either free or fixed.".format(ln_A_name, phi_name)
            )

        return 'free' if ln_A_free else 'fixed'

    def _fixed_complex_amplitude(self, ln_A_name, phi_name):

        ln_A_value = self.fixed_params[ln_A_name]
        phi_value  = self.fixed_params[phi_name]

        return np.exp(ln_A_value) * np.exp(1j*phi_value)

    def _quadratic_name(self, quad_term, modes):

        (l, m, n), (l1, m1, n1), (l2, m2, n2) = modes

        return '{}_{}{}{}_{}{}{}_{}{}{}'.format(quad_term, l,m,n, l1,m1,n1, l2,m2,n2)

    def _classify_linear_components(self):

        solve_components  = []
        fixed_components  = {
            'linear'   : {},
            'tail'     : {'amplitudes': {}, 'exponents': {}},
            'quadratic': {},
        }

        for mode in self.inference_model.wf_model.Kerr_modes:
            l_ring, m_ring, n = mode
            mode_string = '{}{}{}'.format(l_ring, m_ring, n)
            ln_A_name   = 'ln_A_{}'.format(mode_string)
            phi_name    = 'phi_{}'.format(mode_string)
            status      = self._component_status(ln_A_name, phi_name)

            if status == 'free':
                solve_components.append({'kind': 'linear', 'mode': mode, 'ln_A_name': ln_A_name, 'phi_name': phi_name})
            else:
                fixed_components['linear'][mode] = self._fixed_complex_amplitude(ln_A_name, phi_name)

        if self.inference_model.wf_model.quadratic_modes is not None:
            for quad_term in self.inference_model.wf_model.quadratic_modes:
                for modes in self.inference_model.wf_model.quadratic_modes[quad_term]:
                    quad_string = self._quadratic_name(quad_term, modes)
                    ln_A_name   = 'ln_A_{}'.format(quad_string)
                    phi_name    = 'phi_{}'.format(quad_string)
                    status      = self._component_status(ln_A_name, phi_name)
                    key         = (quad_term, modes)

                    if status == 'free':
                        solve_components.append({'kind': 'quadratic', 'key': key, 'ln_A_name': ln_A_name, 'phi_name': phi_name})
                    else:
                        fixed_components['quadratic'][key] = self._fixed_complex_amplitude(ln_A_name, phi_name)

        if self.inference_model.wf_model.tail:
            for tail_mode in self.inference_model.wf_model.tail_modes:
                l_ring, m_ring = tail_mode
                if (l_ring, m_ring, 0) not in self.inference_model.wf_model.Kerr_modes:
                    raise ValueError(
                        "Linear inversion needs tail mode `{}{}0` in QNM-modes so the tail basis can be built.".format(l_ring, m_ring)
                    )

                tail_string    = '{}{}'.format(l_ring, m_ring)
                ln_A_name      = 'ln_A_tail_{}'.format(tail_string)
                phi_name       = 'phi_tail_{}'.format(tail_string)
                p_name         = 'p_tail_{}'.format(tail_string)

                if p_name in self.name_set:
                    raise ValueError(
                        "Linear inversion can solve tail amplitudes only at fixed `{}`. "
                        "Fix `{}` in [Priors] or use Minimization.".format(p_name, p_name)
                    )

                try:
                    fixed_components['tail']['exponents'][tail_mode] = self.fixed_params[p_name]
                except KeyError as exc:
                    raise ValueError(
                        "Linear inversion needs fixed `{}` for Kerr tails. "
                        "Add `fix-{} = value` to [Priors] or use Minimization.".format(p_name, p_name)
                    ) from exc

                status = self._component_status(ln_A_name, phi_name)
                if status == 'free':
                    solve_components.append(
                        {'kind': 'tail', 'tail_mode': tail_mode, 'ln_A_name': ln_A_name, 'phi_name': phi_name}
                    )
                else:
                    fixed_components['tail']['amplitudes'][tail_mode] = self._fixed_complex_amplitude(ln_A_name, phi_name)

        return solve_components, fixed_components

    def _waveform_from_components(self, components, include_const=False):

        return self.inference_model.wf_model.kerr_waveform_from_components(
            amplitudes           = components.get('linear', {}),
            tail_amplitudes      = components.get('tail', {}).get('amplitudes', {}),
            tail_exponents       = components.get('tail', {}).get('exponents', {}),
            quadratic_amplitudes = components.get('quadratic', {}),
            include_const        = include_const,
        )

    def _component_amplitudes(self, component, amplitude):

        amplitudes = {
            'linear'   : {},
            'tail'     : {'amplitudes': {}, 'exponents': self.fixed_components['tail']['exponents']},
            'quadratic': {},
        }

        if component['kind'] == 'linear':
            amplitudes['linear'][component['mode']] = amplitude
        elif component['kind'] == 'quadratic':
            amplitudes['quadratic'][component['key']] = amplitude
        elif component['kind'] == 'tail':
            amplitudes['tail']['amplitudes'][component['tail_mode']] = amplitude
        else:
            raise ValueError("Unknown linear-inversion component kind: {}".format(component['kind']))

        return amplitudes

    def _fixed_waveform(self):

        if (
            len(self.fixed_components['linear'])==0
            and len(self.fixed_components['tail']['amplitudes'])==0
            and len(self.fixed_components['quadratic'])==0
        ):
            return np.zeros(len(self.inference_model.data), dtype=np.complex128)

        return self._waveform_from_components(self.fixed_components, include_const=False)

    def _constant_waveform(self):

        waveform_model = self.inference_model.wf_model
        const_waveform = np.zeros(len(self.inference_model.data), dtype=np.complex128)

        if not(waveform_model.const_params==None):
            const_value = waveform_model.const_params[0]*np.cos(waveform_model.const_params[1])
            const_value = -const_value + 1j*waveform_model.const_params[0]*np.sin(waveform_model.const_params[1])
            const_waveform += const_value

        return const_waveform

    def _weighted_vector(self, waveform):

        err = 1e-16
        sigma_r = np.real(self.inference_model.error) + err
        sigma_i = np.imag(self.inference_model.error) + err

        return np.concatenate((np.real(waveform)/sigma_r, np.imag(waveform)/sigma_i))

    def _component_basis(self, component, amplitude):

        return self._waveform_from_components(self._component_amplitudes(component, amplitude), include_const=False)

    def _design_matrix(self):

        columns = []
        for component in self.solve_components:
            columns.append(self._weighted_vector(self._component_basis(component, 1.0 + 0.0j)))
            columns.append(self._weighted_vector(self._component_basis(component, 0.0 + 1.0j)))

        return np.column_stack(columns)

    def _regularized_inverse(self, matrix):

        return _regularized_symmetric_inverse(matrix, eigenvalue_tol=self.eigenvalue_tol)

    def _parameter_covariance(self, linear_solution, cartesian_covariance):

        transform = np.zeros((len(self.names), len(linear_solution)))
        name_indices = dict((name, i) for i, name in enumerate(self.names))

        for i, component in enumerate(self.solve_components):
            real_index = 2*i
            imag_index = real_index + 1

            amplitude_real = linear_solution[real_index]
            amplitude_imag = linear_solution[imag_index]
            amplitude_norm_squared = max(amplitude_real**2 + amplitude_imag**2, 1e-300)

            ln_A_index = name_indices[component['ln_A_name']]
            phi_index  = name_indices[component['phi_name']]

            transform[ln_A_index, real_index] =  amplitude_real/amplitude_norm_squared
            transform[ln_A_index, imag_index] =  amplitude_imag/amplitude_norm_squared
            transform[phi_index , real_index] = -amplitude_imag/amplitude_norm_squared
            transform[phi_index , imag_index] =  amplitude_real/amplitude_norm_squared

        covariance = np.dot(transform, np.dot(cartesian_covariance, transform.T))
        covariance = 0.5*(covariance + covariance.T)

        return covariance

    def solve_likelihood(self):

        baseline = self._constant_waveform() + self._fixed_waveform()
        data_vec = self._weighted_vector(self.inference_model.data - baseline)
        design   = self._design_matrix()

        fisher = np.dot(design.T, design)
        fisher = 0.5*(fisher + fisher.T)
        rhs    = np.dot(design.T, data_vec)

        fisher_inv, eigvals, eigvals_regularized = self._regularized_inverse(fisher)
        linear_solution = np.dot(fisher_inv, rhs)

        residual = data_vec - np.dot(design, linear_solution)
        cost     = 0.5*np.dot(residual, residual)

        print("* Linear inversion solved {} complex Kerr amplitudes.".format(len(self.solve_components)))
        print("* Linear inversion cost: {:.12e}".format(cost))
        print("* Fisher eigenvalue range before regularization: [{:.12e}, {:.12e}]".format(np.min(eigvals), np.max(eigvals)))
        print("* Fisher eigenvalue tolerance: {:.12e}".format(self.eigenvalue_tol))
        print("* Fisher condition after regularization: {:.12e}".format(np.max(eigvals_regularized)/np.min(eigvals_regularized)))

        self.covariance = self._parameter_covariance(linear_solution, fisher_inv)
        self.errors     = _errors_from_covariance(self.names, self.covariance)

        results = {}
        for i, component in enumerate(self.solve_components):
            complex_amplitude = linear_solution[2*i] + 1j*linear_solution[2*i+1]
            results[component['ln_A_name']] = np.log(max(np.abs(complex_amplitude), 1e-300))
            results[component['phi_name']]  = np.angle(complex_amplitude) % twopi

        return np.array([results[name] for name in self.names])

def run_inference(parameters, inference_model):

    if(parameters['Inference']['method'] == 'Minimization'):

        print('\nStarting minimization algorithm using `scipy.optimize.least_squares`.\n')
        
        minimization         = Minimization_Algorithm(inference_model, parameters)
        minimization_results = minimization.minimize_likelihood()
        
        point_estimate = dict(zip(inference_model.names, minimization_results))
        postprocess.save_point_estimates(point_estimate, parameters['I/O']['outdir'], errors=minimization.errors)
        results_object = postprocess.PointEstimateResults(point_estimate, errors=minimization.errors, covariance=minimization.covariance)

        point_estimate_posterior_samples = int(parameters['Inference'].get('point-estimate-posterior-samples', postprocess.point_estimate_posterior_samples))
        if(point_estimate_posterior_samples < 0):
            raise ValueError("Invalid point-estimate posterior option: `point-estimate-posterior-samples` must be non-negative.")
        if(point_estimate_posterior_samples > 0):
            postprocess.save_point_estimate_posterior(
                point_estimate,
                parameters['I/O']['outdir'],
                covariance=minimization.covariance,
                errors=minimization.errors,
                seed=parameters['Inference']['seed'],
                n_samples=point_estimate_posterior_samples,
            )
            results_object = postprocess.read_posterior_samples(parameters['I/O']['outdir'])
        else:
            postprocess.remove_point_estimate_posterior(parameters['I/O']['outdir'])
        minimization_railing_check(results_object, inference_model, parameters['I/O']['outdir'], tolerance=2.0)

    elif(is_linear_inversion_method(parameters['Inference']['method'])):

        print('\nStarting Kerr linear inversion using weighted normal equations.\n')
        
        linear_inversion         = KerrLinearInversion_Algorithm(inference_model, parameters)
        linear_inversion_results = linear_inversion.solve_likelihood()
        
        point_estimate = dict(zip(inference_model.names, linear_inversion_results))
        postprocess.save_point_estimates(point_estimate, parameters['I/O']['outdir'], errors=linear_inversion.errors)
        results_object = postprocess.PointEstimateResults(point_estimate, errors=linear_inversion.errors, covariance=linear_inversion.covariance)

        point_estimate_posterior_samples = int(parameters['Inference'].get('point-estimate-posterior-samples', postprocess.point_estimate_posterior_samples))
        if(point_estimate_posterior_samples < 0):
            raise ValueError("Invalid point-estimate posterior option: `point-estimate-posterior-samples` must be non-negative.")
        if(point_estimate_posterior_samples > 0):
            postprocess.save_point_estimate_posterior(
                point_estimate,
                parameters['I/O']['outdir'],
                covariance=linear_inversion.covariance,
                errors=linear_inversion.errors,
                seed=parameters['Inference']['seed'],
                n_samples=point_estimate_posterior_samples,
            )
            results_object = postprocess.read_posterior_samples(parameters['I/O']['outdir'])
        else:
            postprocess.remove_point_estimate_posterior(parameters['I/O']['outdir'])

    elif(parameters['Inference']['method'] == 'Nested-sampler'):
        
        if parameters['Inference']['sampler'] == 'cpnest':

            print('* Using CPNest version: `{}`.\n'.format(cpnest.__version__))
            print('* The sampling output appears in the `{}/Algorithm/cpnest.log` file.\n'.format( parameters['I/O']['outdir']))

            job = cpnest.CPNest(inference_model                                                  ,
                                verbose  = 3                                                     ,
                                nlive    = parameters['Inference']['nlive']                      ,
                                maxmcmc  = parameters['Inference']['maxmcmc']                    ,
                                seed     = parameters['Inference']['seed']                       , 
                                poolsize = 128                                                   ,
                                nthreads = 1                                                     ,
                                output   = os.path.join( parameters['I/O']['outdir'],'Algorithm'),
                                resume   = 1                                                     )
            job.run()

            results_object = job.get_posterior_samples(filename='posterior.dat')
            Evidence       = job.NS.logZ

        elif parameters['Inference']['sampler'] == 'raynest':

            """

                Summary of parallel options:

                - nnest: number of nested samplers running in parallel
                - nensemble: nnest*N_ev, where N_ev is the number of live points being substituted at each NS iteration. Requires: N_ev << nlive

                If you submit to a cluster:

                - Request n_cpu = nnest+nensemble
                - Can be directly submitted to multiple cores, and should take care by itself of the parallelization

            """

            import raynest, raynest.model

            print('* Using raynest version: `{}`.\n'.format(raynest.__version__))
            print('* The sampling output appears in the `{}/Algorithm/raynest.log` file.\n'.format( parameters['I/O']['outdir']))

            job = raynest.raynest(inference_model                                                   ,
                                  verbose   = 2                                                     ,
                                  nlive     = parameters['Inference']['nlive']                      ,
                                  maxmcmc   = parameters['Inference']['maxmcmc']                    ,
                                  seed      = parameters['Inference']['seed']                       , 
                                  nnest     = parameters['Inference']['nnest']                      ,
                                  nensemble = parameters['Inference']['nensemble']                  ,
                                  output    = os.path.join( parameters['I/O']['outdir'],'Algorithm'),
                                  resume    = 1                                                     )   
            job.run()
            results_object  = job.posterior_samples.ravel()
            posterior       = pd.DataFrame(results_object, columns = inference_model.names + ['logL', 'logPrior'])
            Evidence        = job.logZ 
            posterior.to_csv(os.path.join( parameters['I/O']['outdir'],'Algorithm/posterior.dat'), index = False)    

        store_evidence_to_file(parameters, Evidence)

        #==================================#
        # Posterior railing check section. #
        #==================================#

        railing_check(results_object, inference_model, parameters['I/O']['outdir'], parameters['Inference']['nlive'], parameters['Inference']['seed'], tolerance=2.0)

    else: raise ValueError('Method {} not recognised.'.format(parameters['Inference']['method']))

    return results_object
