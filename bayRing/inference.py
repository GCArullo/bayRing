import itertools as it, numpy as np, os, pandas as pd, traceback
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

def read_parameter_bounds(Config, configparser, basename, fullname, default_bounds):
    
    single_bounds = [0.0,0.0]
    
    try:                                                                                                     single_bounds[0] = Config.getfloat("Priors", fullname+'-min')
    except (KeyError, configparser.NoOptionError, configparser.NoSectionError, configparser.NoSectionError): single_bounds[0] = default_bounds[basename][0]
    try:                                                                                                     single_bounds[1] = Config.getfloat("Priors", fullname+'-max')
    except (KeyError, configparser.NoOptionError, configparser.NoSectionError, configparser.NoSectionError): single_bounds[1] = default_bounds[basename][1]

    print(('{} : [{}, {}]'.format(fullname.ljust(max_parameter_name_len), single_bounds[0], single_bounds[1])))

    return single_bounds

def read_parameter_start_minimization(Config, configparser, fullname, bounds, nseeds):
    
    
    try:                                                                                                     start_value = Config.getfloat("Priors", fullname+'-start')
    except (KeyError, configparser.NoOptionError, configparser.NoSectionError, configparser.NoSectionError): 
        start_values = np.random.uniform(bounds[0], bounds[1], nseeds)
        results      = []
        # Be careful, you should minimize across all parameters
        for start_value in start_values:
            l_s(self.fun, param_0, method = self.min_method)

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
        default_bounds_TEOBPM['A_peakdotdot'] = [-0.01, 0.0]

    if(  wf_model=='Damped-sinusoids'): default_bounds = default_bounds_DS
    elif(wf_model=='Kerr'            ): default_bounds = default_bounds_Kerr
    elif(wf_model=='Kerr-tail'       ): default_bounds = default_bounds_Kerr_tail
    elif(wf_model=='MMRDNP'          ): default_bounds = {'phi': [0.0, twopi]}
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
                        single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds)
                        self.names.append(fullname)
                        self.bounds.append(single_bounds)

                if(self.quadratic_modes is not None):

                    for quad_term in self.quadratic_modes:
                        for ((l,m,n),(l1,m1,n1),(l2,m2,n2)) in self.quadratic_modes[quad_term]:
                            for name in default_bounds.keys():

                                fullname      = '{}_{}_{}{}{}_{}{}{}_{}{}{}'.format(name, quad_term, l,m,n, l1,m1,n1, l2,m2,n2)
                                single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds)
                                self.names.append(fullname)
                                self.bounds.append(single_bounds)

                if(self.tail):
                    default_bounds_tail = read_default_bounds(self.wf_model.wf_model+'-tail')   
                    for (l_ring, m_ring) in self.tail_modes:
                        for name in default_bounds_tail.keys():

                            fullname      = '{}_{}{}'.format(name, l_ring, m_ring)
                            single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds_tail)
                            self.names.append(fullname)
                            self.bounds.append(single_bounds)
        
            elif(self.wf_model.wf_model=='Damped-sinusoids'):
            
                default_bounds = read_default_bounds(self.wf_model.wf_model)
                for i,name in it.product(list(range(self.N_ds_modes)),default_bounds.keys()):

                    fullname      = '{}_{}'.format(name, i)
                    single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds)
                    self.names.append(fullname)
                    self.bounds.append(single_bounds)
                    
            elif(self.wf_model.wf_model=='Kerr-Damped-sinusoids'):

                self.tail            = self.wf_model.tail
                self.quadratic_modes = self.wf_model.quadratic_modes

                default_bounds_Kerr = read_default_bounds('Kerr')   
                for (l_ring, m_ring, n) in self.Kerr_modes:
                    for name in default_bounds_Kerr.keys():

                        fullname      = '{}_{}{}{}'.format(name, l_ring, m_ring, n)
                        single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds_Kerr)
                        self.names.append(fullname)
                        self.bounds.append(single_bounds)

                if(self.quadratic_modes is not None):

                    for quad_term in self.quadratic_modes:
                        for ((l,m,n),(l1,m1,n1),(l2,m2,n2)) in self.quadratic_modes[quad_term]:
                            for name in default_bounds_Kerr.keys():

                                fullname      = '{}_{}_{}{}{}_{}{}{}_{}{}{}'.format(name, quad_term, l,m,n, l1,m1,n1, l2,m2,n2)
                                single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds_Kerr)
                                self.names.append(fullname)
                                self.bounds.append(single_bounds)

                if(self.tail):
                    default_bounds_tail = read_default_bounds('Kerr-tail')   
                    for (l_ring, m_ring) in self.tail_modes:
                        for name in default_bounds_tail.keys():

                            fullname      = '{}_{}{}'.format(name, l_ring, m_ring)
                            single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds_tail)
                            self.names.append(fullname)
                            self.bounds.append(single_bounds)

                default_bounds_DS = read_default_bounds('Damped-sinusoids')
                for i,name in it.product(list(range(self.N_ds_modes)),default_bounds_DS.keys()):

                    fullname      = '{}_{}'.format(name, i)
                    single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds_DS)
                    self.names.append(fullname)
                    self.bounds.append(single_bounds)

            elif(self.wf_model.wf_model=='MMRDNP'):

                default_bounds = read_default_bounds(self.wf_model.wf_model)   
                for name in default_bounds.keys():
                    single_bounds = read_parameter_bounds(Config, configparser, name, name, default_bounds)
                    self.names.append(name)
                    self.bounds.append(single_bounds)

            elif(self.wf_model.wf_model=='TEOBPM'):

                default_bounds_TEOBPM = read_default_bounds(self.wf_model.wf_model, TEOB_template=self.TEOB_template)   
                for name in default_bounds_TEOBPM.keys():
                    if(not(self.TEOB_NR_fit) and not(name=='phi_mrg')): continue
                    fullname = '{}_{}{}'.format(name, self.wf_model.l_NR, self.wf_model.m_NR)
                    single_bounds = read_parameter_bounds(Config, configparser, name, fullname, default_bounds_TEOBPM)
                    self.names.append(fullname)
                    self.bounds.append(single_bounds)

            self.residuals_tt = []
            self.grid_x       = []
            self.grid_y       = []
            
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
            
            fit_model = self.wf_model.waveform(x)
            
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
            
            if self.min_method == 'lm':
                
                fun_min = (np.real(self.data)-np.real(self.model(x_dict)))/np.real(self.error)+(np.imag(self.data)-np.imag(self.model(x_dict)))/np.imag(self.error)
                
            else:

                lh_r    = -0.5 * np.sum(((np.real(self.data)-np.real(self.model(x_dict)))/np.real(self.error))**2)
                lh_i    = -0.5 * np.sum(((np.imag(self.data)-np.imag(self.model(x_dict)))/np.imag(self.error))**2)
                self.residuals_tt.append(lh_r + lh_i)
                self.grid_x.append(x[0])
                self.grid_y.append(x[1])
                fun_min = lh_r + lh_i
            
            return fun_min
        
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

            if(self.wf_model.wf_model=='Damped-sinusoids'):
                # Order the frequencies per given polarisation (same as m1>m2 in LAL).
                for i in range(self.wf_model.N_ds_modes):
                    try:
                        if (x['f_{}'.format(i)] < x['f_{}'.format(i-1)]): return -np.inf
                    except(KeyError):
                        pass
            # In the case of Kerr tails, order the tails by exponent
            if(self.wf_model.wf_model=='Kerr' and self.wf_model.tail==1):
                for (l_ring, m_ring) in self.tail_modes:
                    # FIXME: temporarily valid only for two modes. Eventually do it for an arbitrary number of modes.
                    if (x['p_tail_{}{}'.format(l_ring, m_ring)] < x['p_tail_{}{}'.format(self.wf_model.l_NR, self.wf_model.m_NR)]): return -np.inf

            return 0.0
    
    return InferenceModel
            
        
class Minimization_Algorithm():
      
    def __init__(self, inference_model, parameters):

        self.inference_model = inference_model
        self.min_method      = inference_model.min_method
        self.bounds          = inference_model.access_bounds()
        self.names           = inference_model.access_names()

        self.iter_min        = parameters['Inference']['min-iter-min']
        self.iter_max        = parameters['Inference']['min-iter-max']

        # Convert bounds to a format compatible with `least_squares` arguments
        self.bounds_minim    = ([self.bounds[i][0] for i in range(len(self.bounds))], [self.bounds[i][1] for i in range(len(self.bounds))])

        self.start_values = []
        for i,name_x in enumerate(self.names):
            self.start_values.append(read_parameter_start_minimization(inference_model.Config, configparser, name_x, self.bounds[i]))

    def fun(self, x):
    
        function_to_minimize = self.inference_model.log_likelihood_ToMin(x)
    
        return function_to_minimize

    def minimize_likelihood(self):

        # Initialize the structures
        j_min    = self.iter_min
        j_max    = self.iter_max
        j        = 0
        x_min_tt = []
        res_tt   = []

        # Initial parameters and corresponding residuals
        x0_0, res_0 = l_s(self.fun, self.start_values, method = self.min_method)
        x_min       = x0_0
        x_min_tt.append(x_min)
        res_tt.append(  res_0)

        # Start the minimimization loop using the initial parameters
        min_fun_i = l_s(self.fun, x0_0, method = self.min_method)
        res_i     = min_fun_i.cost
        x0_i      = min_fun_i.x

        # Iterate until all of these conditions are met:
        # 1. The cost function is smaller than the *previous* step
        # 2. All the value are within the bounds
        # 3. The cost function is larger than the previous one but the number of iterations is smaller than j_min. Forces to do at least jmin iterations

        condition_1  = (np.abs(res_i) < np.abs(res_0))
        condition_2a = not(all([x_min[i] < self.bounds[i][1] for i in range(len(self.bounds))]))
        condition_2b = not(all([x_min[i] > self.bounds[i][0] for i in range(len(self.bounds))]))
        condition_2  = (condition_2a or condition_2b)
        condition_3  = (np.abs(res_i) >= np.abs(res_0) and j < j_min)

        while (condition_1 or condition_2 or condition_3):
            
            min_fun_tmp = l_s(self.fun, x0_i, method = self.min_method)
            res_0       = res_i
            res_i       = min_fun_tmp.cost
            
            condition_1  = (np.abs(res_i) < np.abs(res_0))
            condition_2  = all([x0_i[i] < self.bounds[i][1] for i in range(len(self.bounds))])
            condition_3  = all([x0_i[i] > self.bounds[i][0] for i in range(len(self.bounds))])

            if condition_1 and condition_2 and condition_3:
                
                x_min = x0_i
                x0_i  = min_fun_tmp.x
                x_min_tt.append(x_min)
                res_tt.append(  res_0)
                
            else:
                delta_A = read_jumps_from_user()
                # If you have not improved wrt to x0_i, try to change the initial guess
                x0_i  = []
                shrinkage_ratio = params['shrinkage_ratio']
                for i in range(0,len(self.Kerr_modes)):
                    epsilon_A = np.random.uniform(-1, 1)*2*delta_A[i]/shrinkage_ratio
                    x0_tmp    = min_fun_tmp.x[i] + epsilon_A
                    i_test    = 0
                    while x0_tmp > self.bounds[i][1] and x0_tmp < self.bounds[i][0]:
                        epsilon_A = np.random.uniform(-1, 1)*2*delta_A[i]/shrinkage_ratio
                        x0_tmp    = min_fun_tmp.x[i] + epsilon_A
                        i_test   += 1
                        if i_test > 100:
                            print('failure')
                            exit()
                    x0_i.append(x0_tmp)
                
                min_fun_tmp = l_s(self.fun, x0_i, method = self.min_method)
                x0_i        = min_fun_tmp.x
                res_i       = min_fun_tmp.cost
                j          += 1

            if(j > j_max): break
        
        return x_min_tt[np.argmin(res_tt)]

def run_inference(parameters, inference_model):

    if(parameters['Inference']['method'] == 'Minimization'):
        
        utils.minimisation_compatibility_check(parameters)

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