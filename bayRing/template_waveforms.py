import numpy as np

import cpnest.model

import pyRing.waveform as wf
import pyRing.utils    as pyr_utils
import bayRing.utils   as utils

class WaveformModel(cpnest.model.Model):
    
    def __init__(self, t_NR, tM_start, tM_peak, wf_model, N_ds_modes, Kerr_modes, metadata, qnm_cached, l_NR, m_NR, tail=0, tail_modes=None, quadratic_modes=None, const_params=None, KerrBinary_version = 'London2018', KerrBinary_amp_nc_version = 'bmrg-Jmrg', TEOB_NR_fit = 0, TEOB_template = 'qc'):

        self.t_NR                      = t_NR
        self.t_start                   = tM_start
        self.t_peak                    = tM_peak
        self.wf_model                  = wf_model
        self.Kerr_modes                = Kerr_modes
        self.metadata                  = metadata
        self.const_params              = const_params
        self.Mf, self.af               = self.metadata['Mf'], self.metadata['af']
        self.qnm_cached                = qnm_cached
        self.l_NR, self.m_NR           = l_NR, m_NR
        self.tail                      = tail
        self.quadratic_modes           = quadratic_modes
        self.N_ds_modes                = N_ds_modes
        self.tail_modes                = tail_modes
        self.KerrBinary_version        = KerrBinary_version
        self.KerrBinary_amp_nc_version = KerrBinary_amp_nc_version
        self.TEOB_NR_fit               = TEOB_NR_fit
        self.TEOB_template             = TEOB_template

        if not(const_params==None):
            self.const_r = [const_params[0]*np.cos(const_params[1])]
            self.const_i = [const_params[0]*np.sin(const_params[1])]
    
    def Kerr_waveform(self, params, fixed_params):

        amps, quad_amps, tail_parameters = {}, {}, {}
        
        # Read-in linear modes.
        for (l_ring, m_ring, n) in self.Kerr_modes:
            linear_string = '{}{}{}'.format(l_ring, m_ring, n)
            amp_value = utils.get_param_override(fixed_params,params,'ln_A_{}'.format(linear_string))
            phi_value = utils.get_param_override(fixed_params,params,'phi_{}'.format(linear_string))
            amps[(2, l_ring, m_ring, n)] = np.exp(amp_value) * np.exp(1j*(phi_value))
            
        # Read-in tail parameters.
        if(self.tail):
            for (l_ring, m_ring) in self.tail_modes:
                tail_string = '{}{}'.format(l_ring, m_ring)
                tail_parameters[(l_ring, m_ring)] = {}

                tail_amp_value = utils.get_param_override(fixed_params,params,'ln_A_tail_{}'.format(tail_string))
                tail_phi_value = utils.get_param_override(fixed_params,params, 'phi_tail_{}'.format(tail_string))
                tail_p_value   = utils.get_param_override(fixed_params,params,   'p_tail_{}'.format(tail_string))

                tail_parameters[(l_ring, m_ring)]['A']   = np.exp(tail_amp_value)
                tail_parameters[(l_ring, m_ring)]['phi'] =        tail_phi_value
                tail_parameters[(l_ring, m_ring)]['p']   =        tail_p_value

        # Read-in quadratic modes.
        if(self.quadratic_modes is not None):
            for quad_term in self.quadratic_modes:
                quad_amps[quad_term] = {}
                for ((l,m,n),(l1,m1,n1),(l2,m2,n2)) in self.quadratic_modes[quad_term]:
                    quad_string = '{}_{}{}{}_{}{}{}_{}{}{}'.format(quad_term, l,m,n, l1,m1,n1, l2,m2,n2)
                    quad_amp_value = utils.get_param_override(fixed_params,params,'ln_A_{}'.format(quad_string))
                    quad_phi_value = utils.get_param_override(fixed_params,params, 'phi_{}'.format(quad_string))
                    quad_amps[quad_term][((2,l,m,n),(2,l1,m1,n1),(2,l2,m2,n2))] = np.exp(quad_amp_value) * np.exp(1j*quad_phi_value)

        if('qf' in self.metadata): 
            TGR_parameters      = {}
            TGR_parameters['Q'] = self.metadata['qf']
            self.charge         = 1
        else:
            TGR_parameters      = None
            self.charge         = 0

        ringdown_model = wf.KerrBH(self.t_start                               ,
                                   self.Mf                                    ,
                                   self.af                                    ,
                                   amps                                       ,
                                   0.0                                        , # distance,    overrun by geom
                                   0.0                                        , # inclination, overrun by geom
                                   0.0                                        , # phi,         overrun by geom
                                    
                                   reference_amplitude = 0.0                  ,
                                   geom                = 1                    ,
                                   qnm_fit             = 0                    ,
                                   qnm_interpolants    = None                 , #self.qnm_interpolants,
                                    
                                   Spheroidal          = 0                    , # Spheroidal harmonics, overrun by geom
                                   amp_non_prec_sym    = 1                    ,
                                   tail_parameters     = tail_parameters      ,
                                   quadratic_modes     = quad_amps            ,
                                   quad_lin_prop       = 0                    ,
                                   qnm_cached          = self.qnm_cached      ,
                                   t_ref               = self.t_peak          ,

                                   charge              = self.charge          ,
                                   TGR_params          = TGR_parameters       ,
                                   )
        
        return ringdown_model
        
    def Damped_sinusoids_waveform(self, params, fixed_params):

        ringdown_model = np.zeros(len(self.t_NR), dtype=np.complex128)
        
        amp_value = utils.get_param_override(fixed_params,params,'ln_A_{}'.format(i))
        phi_value = utils.get_param_override(fixed_params,params, 'phi_{}'.format(i))
        f_value   = utils.get_param_override(fixed_params,params,   'f_{}'.format(i))
        tau_value = utils.get_param_override(fixed_params,params, 'tau_{}'.format(i))

        # In this case modes is an integer storing the number of free damped sinusoids
        for i in range(self.N_ds_modes):
            ringdown_model += wf.damped_sinusoid(np.exp(amp_value)   ,
                                                        f_value      ,
                                                        tau_value    ,
                                                        phi_value    ,
                                                        self.t_start ,
                                                        self.t_start ,
                                                        self.t_NR    )
            
        return ringdown_model

    def KerrBinary_waveform(self, params, fixed_params):

        TGR_parameters = {}
        KerrBinary_params  = {}

        if(self.KerrBinary_version=='noncircular'): noncircular_parameters = {'Emrg': self.metadata['Emrg'], 'Jmrg': self.metadata['Jmrg'], 'bmrg': self.metadata['bmrg']}
        else                                  : noncircular_parameters = {}

        KerrBinary_params['Mi'], KerrBinary_params['eta'], KerrBinary_params['chis'], KerrBinary_params['chia'] = pyr_utils.compute_KerrBinary_binary_quantities(self.metadata['m1'], self.metadata['m2'], self.metadata['chi1'], self.metadata['chi2'])  

        phi_value = utils.get_param_override(fixed_params,params,'phi')

        ringdown_model = wf.KerrBinary(self.t_start                                        ,
                                       self.t_peak                                         ,
                                       self.Mf                                             ,
                                       self.af                                             ,

                                       KerrBinary_params['Mi']                                 ,
                                       KerrBinary_params['eta']                                ,
                                       KerrBinary_params['chis']                               ,
                                       KerrBinary_params['chia']                               ,

                                       1.0                                                     , # distance     , dummy with geom=1
                                       0.0                                                     , # inclination  , dummy with geom=1
                                       phi_value                                               , 

                                       TGR_parameters                                          ,

                                       noncircular_params      = noncircular_parameters        ,
                                       noncircular_amp_version = self.KerrBinary_amp_nc_version,

                                       single_spherical_mode   = 1                             ,
                                       single_spherical_l      = self.l_NR                     ,
                                       single_spherical_m      = self.m_NR                     ,

                                       geom                    = 1                             ,
                                       qnm_fit                 = 0                             ,
                                       qnm_interpolants        = None                          ,
                                       qnm_cached              = self.qnm_cached               ,
                                       version                 = self.KerrBinary_version       )

        return ringdown_model

    def TEOBPM_waveform(self, params, fixed_params):

        TGR_parameters = {}
        
        modes          = [(self.l_NR,self.m_NR)]
        merger_phases  = {(self.l_NR,self.m_NR): params['phi_mrg_{}{}'.format(self.l_NR,self.m_NR)]}

        nu = (self.metadata['m1']*self.metadata['m2'])/(self.metadata['m1']+self.metadata['m2'])**2

        if(self.TEOB_NR_fit): 
            
            if(self.TEOB_template=='qc'):
                NR_fit_coeffs = {
                                (self.l_NR,self.m_NR): {
                                                        'c3A'           : params[            'c3A_{}{}'.format(self.l_NR,self.m_NR)]   ,
                                                        'c3p'           : params[            'c3p_{}{}'.format(self.l_NR,self.m_NR)]   ,
                                                        'c4p'           : params[            'c4p_{}{}'.format(self.l_NR,self.m_NR)]   ,
                                                        'omg_peak'      : self.metadata['omg_peak_{}{}'.format(self.l_NR,self.m_NR)]   ,
                                                        'A_peak_over_nu': self.metadata[  'A_peak_{}{}'.format(self.l_NR,self.m_NR)]/nu,
                                                        }
                                }
            elif(self.TEOB_template=='nc'):
                NR_fit_coeffs = {
                                (self.l_NR,self.m_NR): {
                                                        'c2A'           : params[                  'c2A_{}{}'.format(self.l_NR,self.m_NR)]   ,
                                                        'c3A'           : params[                  'c3A_{}{}'.format(self.l_NR,self.m_NR)]   ,
                                                        'c2p'           : params[                  'c2p_{}{}'.format(self.l_NR,self.m_NR)]   ,
                                                        'c3p'           : params[                  'c3p_{}{}'.format(self.l_NR,self.m_NR)]   ,
                                                        'c4p'           : params[                  'c4p_{}{}'.format(self.l_NR,self.m_NR)]   ,
                                                        'omg_peak'      : self.metadata[      'omg_peak_{}{}'.format(self.l_NR,self.m_NR)]   ,
                                                        'A_peak_over_nu': self.metadata[        'A_peak_{}{}'.format(self.l_NR,self.m_NR)]/nu,
                                                        'A_peakdotdot'  : params[         'A_peakdotdot_{}{}'.format(self.l_NR,self.m_NR)]/nu,
                                                        }
                                }
            else:
                raise ValueError("Unknown TEOB template selected: {}".format(self.TEOB_template))
            
            NR_fit_coeffs['Mf'] = self.Mf
            NR_fit_coeffs['af'] = self.af

        else                :
            NR_fit_coeffs = None

        if(  self.TEOB_template=='qc'): ecc_par = 0
        elif(self.TEOB_template=='nc'): ecc_par = 1

        TGR_parameters = {}
        ringdown_model = wf.TEOBPM(self.t_start,
                                   self.metadata['m1']          ,
                                   self.metadata['m2']          ,
                                   self.metadata['chi1']        ,
                                   self.metadata['chi2']        ,
                                   merger_phases                ,
                                   1.0                          , # distance     , dummy with geom=1
                                   0.0                          , # inclination  , dummy with geom=1
                                   0                            , # orbital phase, dummy with geom=1
                                   modes                        ,
                                   TGR_parameters               ,
                                   geom          = 1            ,
                                   ecc_par       = ecc_par      ,
                                   NR_fit_coeffs = NR_fit_coeffs)


        return ringdown_model

    def waveform(self, params, fixed_params):

        if (self.wf_model=='Kerr'):
            
            ringdown_model = self.Kerr_waveform(params, fixed_params)
            _, _, _, self.wf_r, self.wf_i = ringdown_model.waveform(self.t_NR)
    
        elif (self.wf_model=='Damped-sinusoids'):
            
            ringdown_model = self.Damped_sinusoids_waveform(params, fixed_params)
            self.wf_r, self.wf_i = np.real(ringdown_model), np.imag(ringdown_model)

        elif (self.wf_model=='Kerr-Damped-sinusoids'):

            ringdown_model_Kerr = self.Kerr_waveform(params, fixed_params) 
            ringdown_model_DS   = self.Damped_sinusoids_waveform(params, fixed_params)

            _, _, _, self.wf_r_Kerr, self.wf_i_Kerr = ringdown_model_Kerr.waveform(self.t_NR)
            self.wf_r_DS, self.wf_i_DS = np.real(ringdown_model_DS), np.imag(ringdown_model_DS)

            self.wf_r = self.wf_r_Kerr + self.wf_r_DS
            self.wf_i = self.wf_i_Kerr + self.wf_i_DS   

        elif (self.wf_model=='KerrBinary'):
            
            ringdown_model                = self.KerrBinary_waveform(params, fixed_params)
            _, _, _, self.wf_r, self.wf_i = ringdown_model.waveform(self.t_NR)
        
        elif (self.wf_model=='TEOBPM'):
            
            ringdown_model                = self.TEOBPM_waveform(params, fixed_params)
            _, _, _, self.wf_r, self.wf_i = ringdown_model.waveform(self.t_NR)

        else:
            raise ValueError("Unknown template selected: {}".format(self.wf_model))

        if not(self.const_params==None):
            self.wf_r = self.wf_r + self.const_r
            self.wf_i = self.wf_i + self.const_i

        # UNDERSTAND WHY!!!!
        if not(self.wf_model=='KerrBinary'): self.wf_r = -self.wf_r

        return self.wf_r + 1j * self.wf_i