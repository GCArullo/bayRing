import numpy as np

import cpnest.model

import pyRing.waveform as wf
import pyRing.utils    as pyr_utils
import bayRing.utils   as utils

class WaveformModel(cpnest.model.Model):
    
    def __init__(self, t_NR, tM_start, tM_peak, wf_model, N_ds_modes, Kerr_modes, metadata, fit_metadata, qnm_cached, l_NR, m_NR, N_ds_tails=0, tail=0, tail_modes=None, quadratic_modes=None, const_params=None, KerrBinary_version = 'London2018', KerrBinary_amp_nc_version = 'bmrg-Jmrg', TEOB_template = 'RatExp', TEOB_merger_data = 1, TEOB_global_fit = 0):

        self.t_NR                      = t_NR
        self.t_start                   = tM_start
        self.t_peak                    = tM_peak
        self.wf_model                  = wf_model
        self.Kerr_modes                = Kerr_modes
        self.metadata                  = metadata
        self.fit_metadata              = fit_metadata
        self.const_params              = const_params
        self.Mf, self.af               = self.metadata['Mf'], self.metadata['af']
        self.qnm_cached                = qnm_cached
        self.l_NR, self.m_NR           = l_NR, m_NR
        self.tail                      = tail
        self.quadratic_modes           = quadratic_modes
        self.N_ds_modes                = N_ds_modes
        self.N_ds_tails                = N_ds_tails
        self.tail_modes                = tail_modes
        self.KerrBinary_version        = KerrBinary_version
        self.KerrBinary_amp_nc_version = KerrBinary_amp_nc_version
        self.TEOB_template             = TEOB_template
        self.TEOB_merger_data          = TEOB_merger_data
        self.TEOB_global_fit           = TEOB_global_fit

        if not(const_params==None):
            self.const_r = [const_params[0]*np.cos(const_params[1])]
            self.const_i = [const_params[0]*np.sin(const_params[1])]
    
    def _Kerr_TGR_parameters(self):

        if('qf' in self.metadata): 
            TGR_parameters      = {}
            TGR_parameters['Q'] = self.metadata['qf']
            self.charge         = 1
        else:
            TGR_parameters      = None
            self.charge         = 0

        return TGR_parameters

    def _KerrBH_model(self, amps, tail_parameters=None, quadratic_modes=None):

        if tail_parameters is None: tail_parameters = {}
        if quadratic_modes is None: quadratic_modes = {}
        TGR_parameters = self._Kerr_TGR_parameters()

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
                                   quadratic_modes     = quadratic_modes      ,
                                   quad_lin_prop       = 0                    ,
                                   qnm_cached          = self.qnm_cached      ,
                                   t_ref               = self.t_peak          ,

                                   charge              = self.charge          ,
                                   TGR_params          = TGR_parameters       ,
                                   )
        
        return ringdown_model

    def _apply_waveform_conventions(self, wf_r, wf_i, include_const=True):

        wf_r = np.array(wf_r)
        wf_i = np.array(wf_i)

        if include_const and not(self.const_params==None):
            wf_r = wf_r + self.const_r
            wf_i = wf_i + self.const_i

        # UNDERSTAND WHY!!!!
        if not(self.wf_model=='KerrBinary'): wf_r = -wf_r

        return wf_r, wf_i

    def kerr_waveform_from_components(self, amplitudes=None, tail_amplitudes=None, tail_exponents=None, quadratic_amplitudes=None, include_const=True):

        if amplitudes is None: amplitudes = {}
        if tail_amplitudes is None: tail_amplitudes = {}
        if tail_exponents is None: tail_exponents = {}
        if quadratic_amplitudes is None: quadratic_amplitudes = {}

        amps = {}
        for (l_ring, m_ring, n) in self.Kerr_modes:
            try:
                amps[(2, l_ring, m_ring, n)] = amplitudes[(l_ring, m_ring, n)]
            except KeyError:
                pass

        for (l_ring, m_ring) in tail_amplitudes:
            if ((l_ring, m_ring, 0) in self.Kerr_modes) and ((2, l_ring, m_ring, 0) not in amps):
                amps[(2, l_ring, m_ring, 0)] = 0.0 + 0.0j

        tail_parameters = {}
        for (l_ring, m_ring), tail_amplitude in tail_amplitudes.items():
            tail_parameters[(l_ring, m_ring)] = {
                'A'  : np.abs(tail_amplitude),
                'phi': np.angle(tail_amplitude),
                'p'  : tail_exponents[(l_ring, m_ring)],
            }

        quad_amps = {}
        for quad_term, modes in quadratic_amplitudes:
            (l, m, n), (l1, m1, n1), (l2, m2, n2) = modes
            quad_amps.setdefault(quad_term, {})
            quad_amps[quad_term][((2,l,m,n),(2,l1,m1,n1),(2,l2,m2,n2))] = quadratic_amplitudes[(quad_term, modes)]

        ringdown_model = self._KerrBH_model(amps, tail_parameters=tail_parameters, quadratic_modes=quad_amps)
        _, _, _, wf_r, wf_i = ringdown_model.waveform(self.t_NR)
        wf_r, wf_i = self._apply_waveform_conventions(wf_r, wf_i, include_const=include_const)

        return wf_r + 1j * wf_i

    def kerr_waveform_from_complex_amplitudes(self, amplitudes, include_const=True):

        return self.kerr_waveform_from_components(amplitudes=amplitudes, include_const=include_const)

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

        self._Kerr_TGR_parameters()
        ringdown_model = self._KerrBH_model(amps, tail_parameters=tail_parameters, quadratic_modes=quad_amps)
        
        return ringdown_model
        
    def Damped_sinusoids_waveform(self, params, fixed_params):

        ringdown_model = np.zeros(len(self.t_NR), dtype=np.complex128)

        # Loop over each damped sinusoid mode
        for i in range(self.N_ds_modes):
            amp_value = utils.get_param_override(fixed_params, params, 'ln_A_{}'.format(i))
            phi_value = utils.get_param_override(fixed_params, params, 'phi_{}'.format(i))
            f_value   = utils.get_param_override(fixed_params, params, 'f_{}'.format(i))
            tau_value = utils.get_param_override(fixed_params, params, 'tau_{}'.format(i))

            ringdown_model += wf.damped_sinusoid(
                np.exp(amp_value),
                f_value,
                tau_value,
                phi_value,
                self.t_start,
                self.t_start,
                self.t_NR,
                real_waveform=1
            )

        for i in range(self.N_ds_tails):
            amp_tail_value = utils.get_param_override(fixed_params, params, 'ln_A_tail_{}'.format(i))
            phi_tail_value = utils.get_param_override(fixed_params, params, 'phi_tail_{}'.format(i))
            p_tail_value   = utils.get_param_override(fixed_params, params, 'p_tail_{}'.format(i))

            ringdown_model += wf.tail_factor(
                np.exp(amp_tail_value),
                phi_tail_value,
                p_tail_value,
                self.t_start,
                self.t_peak,
                self.t_NR
            )

        return ringdown_model


    def KerrBinary_waveform(self, params, fixed_params):

        TGR_parameters = {}
        KerrBinary_params  = {}

        if(self.KerrBinary_version=='noncircular'): noncircular_parameters = {'Emrg': self.metadata['Emrg'], 'Jmrg': self.metadata['Jmrg'], 'bmrg': self.metadata['bmrg']}
        else                                  : noncircular_parameters = {}

        KerrBinary_params['Mi'], KerrBinary_params['eta'], KerrBinary_params['chis'], KerrBinary_params['chia'] = pyr_utils.compute_KerrBinary_binary_quantities(self.metadata['m1'], self.metadata['m2'], self.metadata['chi1'], self.metadata['chi2'])  
        
        phi_value = utils.get_param_override(fixed_params,params,'phi')

        available_modes_with_given_lm = utils.filter_dict_by_key(pyr_utils.available_modes_dict_KerrBinary[self.KerrBinary_version], (self.l_NR,self.m_NR))

        ringdown_model = wf.KerrBinary(self.t_start                                            ,
                                       self.t_peak                                             ,
                                       self.Mf                                                 ,
                                       self.af                                                 ,

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

                                       modes                   = available_modes_with_given_lm ,

                                       geom                    = 1                             ,
                                       qnm_fit                 = 0                             ,
                                       qnm_interpolants        = None                          ,
                                       qnm_cached              = self.qnm_cached               ,
                                       version                 = self.KerrBinary_version       )

        return ringdown_model

    def TEOBPM_waveform(self, params, fixed_params):

        if self.TEOB_template=='HypTan':
            template_index = 0
        elif self.TEOB_template=='RatExp':
            template_index = 1
        else:
            raise ValueError("Unknown TEOB template: {}".format(self.TEOB_template))
        
        TGR_parameters = {}
        
        modes          = [(self.l_NR,self.m_NR)]
        merger_phases  = {(self.l_NR,self.m_NR): params['phi_mrg_{}{}'.format(self.l_NR,self.m_NR)]}

        nu = (self.metadata['m1']*self.metadata['m2'])/(self.metadata['m1']+self.metadata['m2'])**2

        if(self.TEOB_merger_data):
            NR_fit_coeffs = {
                            (self.l_NR,self.m_NR): {
                                                    'omg_peak'            : self.metadata['omg_peak_{}{}'.format(self.l_NR,self.m_NR)]       ,
                                                    'A_peak_over_nu'      : self.metadata['A_peak_{}{}'.format(self.l_NR,self.m_NR)]/nu      ,
                                                    }
                            }
            if(self.TEOB_template=='RatExp'):
                NR_fit_coeffs[(self.l_NR,self.m_NR)]['A_peakdotdot_over_nu'] = self.metadata['A_peak{}{}dotdot'.format(self.l_NR,self.m_NR)]/nu
        else:
            NR_fit_coeffs = {(self.l_NR,self.m_NR): {}}

        if not(self.TEOB_global_fit):
            NR_fit_coeffs[(self.l_NR,self.m_NR)]['c3A'] = params['c3A_{}{}'.format(self.l_NR,self.m_NR)]
            NR_fit_coeffs[(self.l_NR,self.m_NR)]['c3p'] = params['c3p_{}{}'.format(self.l_NR,self.m_NR)]
            NR_fit_coeffs[(self.l_NR,self.m_NR)]['c4p'] = params['c4p_{}{}'.format(self.l_NR,self.m_NR)]

            if(self.TEOB_template=='RatExp'):
                NR_fit_coeffs[(self.l_NR,self.m_NR)]['c2A'] = params['c2A_{}{}'.format(self.l_NR,self.m_NR)]
                NR_fit_coeffs[(self.l_NR,self.m_NR)]['c2p'] = params['c2p_{}{}'.format(self.l_NR,self.m_NR)]
        else:
            NR_fit_coeffs['ecc'] = self.metadata['ecc']
            NR_fit_coeffs['bmrg'] = self.metadata['bmrg']
            NR_fit_coeffs['Jmrg'] = self.metadata['Jmrg']
            NR_fit_coeffs['Emrg'] = self.metadata['Emrg']

            if self.fit_metadata is not None:
                fit_coeffs = {key: val for key, val in self.fit_metadata.items() if key.startswith(('c_2_', 'c_3_', 'c_4_'))}
                NR_fit_coeffs[(self.l_NR, self.m_NR)].update(fit_coeffs)

                NR_fit_coeffs[(self.l_NR, self.m_NR)]['fit_type'] = self.fit_metadata['fit_type']
                NR_fit_coeffs[(self.l_NR, self.m_NR)]['fit_order'] = self.fit_metadata['fit_order']

                for key in ['nu', 'ecc', 'bmrg', 'jmrg', 'emrg']:
                    norm_scale_key, norm_shift_key = 'norm_{}_scale'.format(key), 'norm_{}_shift'.format(key)
                    if norm_scale_key in self.fit_metadata:
                        NR_fit_coeffs[norm_scale_key] = self.fit_metadata[norm_scale_key]
                    if norm_shift_key in self.fit_metadata:
                        NR_fit_coeffs[norm_shift_key] = self.fit_metadata[norm_shift_key]

            else:
                if(self.TEOB_template=='RatExp'):
                    raise ValueError("TEOB global fit is enabled but no fit metadata provided.")

        NR_fit_coeffs['Mf'] = self.Mf
        NR_fit_coeffs['af'] = self.af

        TGR_parameters = {}
        ringdown_model = wf.TEOBPM(self.t_peak                  ,
                                   self.metadata['m1']          ,
                                   self.metadata['m2']          ,
                                   self.metadata['chi1']        ,
                                   self.metadata['chi2']        ,
                                   merger_phases                ,
                                   1.0                          , # distance     , dummy with geom=1
                                   0.0                          , # inclination  , dummy with geom=1
                                   0.0                          , # orbital phase, dummy with geom=1
                                   modes                        ,
                                   TGR_parameters               ,
                                   geom          = 1            ,
                                   template      = template_index ,
                                   merger_data   = self.TEOB_merger_data ,
                                   global_fit    = self.TEOB_global_fit ,
                                   NR_fit_coeffs = NR_fit_coeffs)
        return ringdown_model

    def waveform(self, params, fixed_params):

        if (self.wf_model=='Kerr'):
            
            ringdown_model = self.Kerr_waveform(params, fixed_params)
            _, _, _, self.wf_r, self.wf_i = ringdown_model.waveform(self.t_NR)
            self.wf_r, self.wf_i = self._apply_waveform_conventions(self.wf_r, self.wf_i)
    
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

        if not(self.wf_model=='Kerr'):
            self.wf_r, self.wf_i = self._apply_waveform_conventions(self.wf_r, self.wf_i)

        return self.wf_r + 1j * self.wf_i
