import importlib, itertools as it, numpy as np, os, pandas as pd, traceback
from scipy.optimize  import least_squares as l_s

try:                import configparser
except ImportError: import ConfigParser as configparser

from cpnest.nest2pos import draw_posterior, compute_weights
import cpnest, cpnest.model
import pyRing.utils      as pyRing_utils
import bayRing.postprocess as postprocess
import bayRing.utils       as utils

twopi                  = 2.*np.pi
max_parameter_name_len = len('ln_A_tail_22')

# CPNest workers using the spawn start method must pickle the inference model.
# Register the factory-built classes as module globals so pickle can import them.
_DYNAMIC_INFERENCE_MODEL_PREFIX  = 'DynamicInferenceModel_'
_DYNAMIC_INFERENCE_MODEL_CLASSES = {}

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
    
    default_bounds_Kerr      = {'ln_A': [-20.0, 5.0]            ,
                                'phi' : [0.0, twopi]            }
    
    default_bounds_Kerr_tail = {'ln_A_tail': [-20.0, 5.0]       ,
                                'phi_tail' : [0.0, twopi]       ,
                                'p_tail'   : [-20.0,  20.0]     }
    
    default_bounds_TEOBPM    = {'phi_mrg': [0.0  , twopi]       ,
                                'c3A'    : [-10.0, 10.0 ]       ,
                                'c3p'    : [-10.0, 10.0 ]       ,
                                'c4p'    : [-10.0, 10.0 ]       ,
                                }
    if not(TEOB_template=='qc'):
        default_bounds_TEOBPM['c2A']          = [-10.0, 10.0]
        default_bounds_TEOBPM['c2p']          = [-10.0, 10.0]

    if(  wf_model=='Damped-sinusoids'): default_bounds = default_bounds_DS
    elif(wf_model=='Kerr'            ): default_bounds = default_bounds_Kerr
    elif(wf_model=='Kerr-tail'       ): default_bounds = default_bounds_Kerr_tail
    elif(wf_model=='KerrBinary'      ): default_bounds = {'phi': [0.0, twopi]}
    elif(wf_model=='TEOBPM'          ): default_bounds = default_bounds_TEOBPM

    return default_bounds

def railing_check(results_object, inference_model, outdir, nlive, seed, tolerance=2.0):

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
        header = ''
        for (i,param) in enumerate(inference_model.names):
            Prior_bins = np.linspace(inference_model.bounds[i][0], inference_model.bounds[i][-1], 100)
            low_rail, high_rail = pyRing_utils.railing_check(samples=results_object[param], prior_bins=Prior_bins, tolerance=tolerance)
            header +='{par}_low\t{par}_up\t'.format(par=param)
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
        np.savetxt(os.path.join(outdir, 'Algorithm/Parameters_prior_railing.txt'), np.column_stack(railing_parameters), fmt= "%d", header=header)
        
        if np.sum(railing_parameters) > 0:
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
                        Prior_bins = np.linspace(inference_model.bounds[i][0], inference_model.bounds[i][-1], 100)
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
                np.savetxt(os.path.join(outdir, f'Algorithm/Parameters_prior_railing_{chain_number}.txt'), np.column_stack(railing_parameters_chain), fmt= "%d", header=header_chain)
    except:
        print("\n* Warning: Prior railing file generation failed with error: {}.".format(traceback.print_exc()))

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
            self.TEOB_NR_fit   = self.wf_model.TEOB_NR_fit
            self.TEOB_template = self.wf_model.TEOB_template
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
                    if(not(self.TEOB_NR_fit) and not(name=='phi_mrg')): continue
                    fullname = '{}_{}{}'.format(name, self.wf_model.l_NR, self.wf_model.m_NR)
                    try:
                        self.fixed_params[fullname] = self.Config.getfloat("Priors",'fix-'+fullname)
                    except(configparser.NoOptionError):
                        single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds_TEOBPM)
                        self.names.append(fullname)
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
            model   = self.model(x_dict)
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

        return best_result.x

def run_inference(parameters, inference_model):

    if(parameters['Inference']['method'] == 'Minimization'):

        print('\nStarting minimization algorithm using `scipy.optimize.least_squares`.\n')
        
        minimization         = Minimization_Algorithm(inference_model, parameters)
        minimization_results = minimization.minimize_likelihood()
        
        results_object = dict(zip(inference_model.names, minimization_results))
        postprocess.save_results_minimization(results_object, parameters['I/O']['outdir'])

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
