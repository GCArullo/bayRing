import numpy as np

import cpnest.model

import pyRing.waveform as wf
import pyRing.utils    as pyr_utils
import bayRing.utils   as utils

class WaveformModel(cpnest.model.Model):
    
    def __init__(self, t_NR, tM_start, tM_peak, wf_model, N_ds_modes, Kerr_modes, metadata, fit_metadata, qnm_cached, l_NR, m_NR, N_ds_tails=0, tail=0, tail_modes=None, quadratic_modes=None, const_params=None, KerrBinary_version = 'London2018', KerrBinary_amp_nc_version = 'bmrg-Jmrg', TEOB_template = 'RatExp', TEOB_merger_data = 1, TEOB_global_fit = 0, TEOB_NR_fit = 0, TEOB_qc_fit_type = None, sigmoid_flag=0, quadratic_fits=0, SEOB_modes_file='', NR_ID=None):

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
        self.TEOB_template             = {'qc': 'HypTan', 'nc': 'RatExp'}.get(TEOB_template, TEOB_template)
        self.TEOB_merger_data          = TEOB_merger_data
        self.TEOB_global_fit           = TEOB_global_fit
        self.TEOB_NR_fit               = TEOB_NR_fit
        self.TEOB_qc_fit_type          = TEOB_qc_fit_type
        self.sigmoid_flag              = sigmoid_flag
        self.quadratic_fits            = quadratic_fits
        self.SEOB_modes_file           = SEOB_modes_file
        self.NR_ID                     = NR_ID
        self._SEOB_mode                = None

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
                                   self.t_peak                                ,
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
        # (SEOBNRv5HM: the mode is already in the convention of the NR data.)
        if not(self.wf_model=='KerrBinary' or self.wf_model=='SEOBNRv5HM'): wf_r = -wf_r

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

        if(self.KerrBinary_version=='Carullo2024'): noncircular_parameters = {'Emrg': self.metadata['Emrg'], 'Jmrg': self.metadata['Jmrg'], 'bmrg': self.metadata['bmrg']}
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

        if self.TEOB_template not in ['HypTan', 'RatExp']:
            raise ValueError("Unknown TEOB template: {}".format(self.TEOB_template))

        TGR_parameters = {}

        modes          = [(self.l_NR,self.m_NR)]
        merger_phases  = {(self.l_NR,self.m_NR): params['phi_mrg_{}{}'.format(self.l_NR,self.m_NR)]}

        nu = (self.metadata['m1']*self.metadata['m2'])/(self.metadata['m1']+self.metadata['m2'])**2
        development_fit = bool(self.TEOB_NR_fit or self.sigmoid_flag or self.quadratic_fits)

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

        if not(self.TEOB_global_fit) or self.TEOB_NR_fit:
            lm = '{}{}'.format(self.l_NR,self.m_NR)
            for coefficient in (['c3A', 'c3p', 'c4p'] + (['c2A', 'c2p'] if self.TEOB_template=='RatExp' else [])):
                NR_fit_coeffs[(self.l_NR,self.m_NR)][coefficient] = utils.get_param_override(fixed_params,params,'{}_{}'.format(coefficient, lm))
            # Optional c4A (HypTan), replacing the peak-amplitude condition, with the pyRing sentinels (see pyRing TEOBPM):
            # its inputs, the asymptotic amplitude c1c4A = c1A + c4A and the origin tenv0 of the sigmoids, are fixed parameters.
            if(self._has_param(params, fixed_params, 'c4A_'+lm)):
                NR_fit_coeffs[(self.l_NR,self.m_NR)]['c4A'] = utils.get_param_override(fixed_params,params,'c4A_'+lm)
            for extra in ('c1c4A', 'tenv0'):
                if(('{}_{}'.format(extra, lm)) in fixed_params):
                    NR_fit_coeffs[(self.l_NR,self.m_NR)][extra] = fixed_params['{}_{}'.format(extra, lm)]
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
        template_index = 0 if self.TEOB_template == 'HypTan' else 1
        ecc_par = template_index

        if(development_fit):
            amps, dphis, t_sigmoids, width_sigmoids = {}, {}, {}, {}
            quad_amps, quad_dphis, quad_t_sigmoids, quad_width_sigmoids = {}, {}, {}, {}

            for (l_ring, m_ring, n) in self.Kerr_modes:
                linear_string = '{}{}{}'.format(l_ring, m_ring, n)
                amp_value = utils.get_param_override(fixed_params,params,'ln_A_{}'.format(linear_string))
                phi_value = utils.get_param_override(fixed_params,params,'phi_{}'.format(linear_string))
                dphi_value = utils.get_param_override(fixed_params,params,'dphi_{}'.format(linear_string))

                mode_key = (2, l_ring, m_ring, n)
                amps[mode_key] = np.exp(amp_value) * np.exp(1j*(phi_value))
                dphis[mode_key] = dphi_value

                if(self.sigmoid_flag):
                    t_sigmoids[mode_key] = utils.get_param_override(fixed_params,params,'t_sigmoid_{}'.format(linear_string))
                    width_sigmoids[mode_key] = utils.get_param_override(fixed_params,params,'width_sigmoid_{}'.format(linear_string))

            if(self.quadratic_fits and self.quadratic_modes is not None):
                for quad_term in self.quadratic_modes:
                    quad_amps[quad_term] = {}
                    quad_dphis[quad_term] = {}
                    quad_t_sigmoids[quad_term] = {}
                    quad_width_sigmoids[quad_term] = {}

                    for ((l,m,n),(l1,m1,n1),(l2,m2,n2)) in self.quadratic_modes[quad_term]:
                        quad_string = '{}_{}{}{}_{}{}{}_{}{}{}'.format(quad_term, l,m,n, l1,m1,n1, l2,m2,n2)
                        quad_key = ((2,l,m,n),(2,l1,m1,n1),(2,l2,m2,n2))

                        quad_amp_value = utils.get_param_override(fixed_params,params,'ln_A_{}'.format(quad_string))
                        quad_phi_value = utils.get_param_override(fixed_params,params,'phi_{}'.format(quad_string))
                        quad_phi_value = self._relative_quad_phase(fixed_params, params, quad_string, quad_phi_value)
                        quad_dphi_value = utils.get_param_override(fixed_params,params,'dphi_{}'.format(quad_string))

                        quad_amps[quad_term][quad_key] = np.exp(quad_amp_value) * np.exp(1j*(quad_phi_value))
                        quad_dphis[quad_term][quad_key] = quad_dphi_value

                        if(self.sigmoid_flag):
                            quad_t_sigmoids[quad_term][quad_key] = utils.get_param_override(fixed_params,params,'t_sigmoid_{}'.format(quad_string))
                            quad_width_sigmoids[quad_term][quad_key] = utils.get_param_override(fixed_params,params,'width_sigmoid_{}'.format(quad_string))

            # Optional complex constant added to the multipole (ln_A_constant_lm, phi_constant_lm)
            constant_amps = {}
            lm = '{}{}'.format(self.l_NR,self.m_NR)
            if(self._has_param(params, fixed_params, 'ln_A_constant_'+lm)):
                constant_amps[(self.l_NR,self.m_NR)] = np.exp(utils.get_param_override(fixed_params,params,'ln_A_constant_'+lm)) * np.exp(1j*utils.get_param_override(fixed_params,params,'phi_constant_'+lm))

            # Local NR-calibrated coefficients (TEOB-NR-fit=1) are used by pyRing only with global_fit=0.
            return wf.TEOBPM(self.t_peak                  ,
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
                             geom                = 1                                                ,
                             template            = self.TEOB_template                               ,
                             merger_data         = self.TEOB_merger_data                            ,
                             global_fit          = int(bool(self.TEOB_global_fit) and not(self.TEOB_NR_fit)),
                             NR_fit_coeffs       = NR_fit_coeffs                                    ,
                             amps                = amps                                             ,
                             dphis               = dphis                                            ,
                             t_sigmoids          = t_sigmoids                                       ,
                             width_sigmoids      = width_sigmoids                                   ,
                             quad_amps           = quad_amps                                        ,
                             quad_dphis          = quad_dphis                                       ,
                             quad_t_sigmoids     = quad_t_sigmoids                                  ,
                             quad_width_sigmoids = quad_width_sigmoids                              ,
                             sigmoid_flag        = self.sigmoid_flag                                ,
                             constant_amps       = constant_amps                                    )

        try:
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
                                       template      = self.TEOB_template ,
                                       merger_data   = self.TEOB_merger_data ,
                                       global_fit    = self.TEOB_global_fit ,
                                       NR_fit_coeffs = NR_fit_coeffs)
        except TypeError:
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
                                       ecc_par       = ecc_par      ,
                                       template      = template_index,
                                       NR_fit_coeffs = NR_fit_coeffs)
        return ringdown_model

    @staticmethod
    def _has_param(params, fixed_params, name):

        """True if `name` is a sampled (params: dict or cpnest LivePoint) or a fixed parameter."""

        return (name in fixed_params) or (name in (params.names if hasattr(params, 'names') else params))

    def _relative_quad_phase(self, fixed_params, params, quad_string, quad_phi_value):

        """
            If `fix-phirel_<quad>` is given, the phase of the quadratic term is phirel - phi_mrg_lm: only the overall phase phi_mrg is free,
            and the relative phase between the quadratic term and TEOBPM (invariant under a rotation of the binary) is fixed.
        """

        key = 'phirel_{}'.format(quad_string)
        if key in fixed_params:
            return fixed_params[key] - utils.get_param_override(fixed_params, params, 'phi_mrg_{}{}'.format(self.l_NR, self.m_NR))
        return quad_phi_value

    def _load_SEOBNRv5HM_mode(self):

        """
            SEOBNRv5HM (l_NR, m_NR) mode on the NR time grid, with the time origin at the peak of the SEOBNRv5HM (2,2) mode.
            The mode has no sampled intrinsic parameter (masses and spins are the ones of the NR metadata), hence it is computed only once.

            If `SEOB-modes-file` is given, the mode is read from that file: an npz with `t` (in units of M, measured from the peak of |h22|),
            `ID` (NR simulation IDs) and `h<l><m>` (complex modes, one row per ID). Otherwise it is generated with `pyseobnr`, which has to be
            installed in the same environment.
        """

        mode_string = '{}{}'.format(self.l_NR, self.m_NR)

        if(self.SEOB_modes_file):
            data = np.load(self.SEOB_modes_file, allow_pickle=True)
            IDs  = [str(x) for x in data['ID']]
            if not(str(self.NR_ID) in IDs)         : raise ValueError("SEOBNRv5HM template: simulation {} is not in {}".format(self.NR_ID, self.SEOB_modes_file))
            if not('h'+mode_string in data.files)  : raise ValueError("SEOBNRv5HM template: mode h{} is not in {}".format(mode_string, self.SEOB_modes_file))
            t_SEOB = np.asarray(data['t'], dtype=float)
            h_SEOB = np.asarray(data['h'+mode_string][IDs.index(str(self.NR_ID))], dtype=complex)
        else:
            try               : from pyseobnr.generate_waveform import generate_modes_opt
            except ImportError: raise ImportError("SEOBNRv5HM template: `pyseobnr` is not installed in this environment. Either install it, or pass precomputed modes through the `SEOB-modes-file` option.")
            m1, m2, chi1, chi2 = self.metadata['m1'], self.metadata['m2'], self.metadata['chi1'], self.metadata['chi2']
            # pyseobnr requires q>=1; exchanging the labels of the two bodies leaves the even-m modes unchanged.
            if(m1 < m2): m1, m2, chi1, chi2 = m2, m1, chi2, chi1
            t_SEOB, modes = generate_modes_opt(m1/m2, chi1, chi2, 0.02, approximant="SEOBNRv5HM")
            t_SEOB        = np.asarray(t_SEOB, dtype=float)
            h22, h_SEOB   = np.asarray(modes['2,2'], dtype=complex), np.asarray(modes['{},{}'.format(self.l_NR, self.m_NR)], dtype=complex)
            # Time origin at the peak of |h22|, located on a grid finer than the SEOB one.
            t_guess       = t_SEOB[np.argmax(np.abs(h22))]
            t_fine        = np.arange(t_guess - 2.0, t_guess + 2.0, 0.001)
            t_SEOB        = t_SEOB - t_fine[np.argmax(np.interp(t_fine, t_SEOB, np.abs(h22)))]

        # NR times measured from the peak of the NR (2,2) mode. Outside of the SEOB time span the template is zero.
        tau = np.asarray(self.t_NR, dtype=float) - self.t_peak

        return np.interp(tau, t_SEOB, h_SEOB.real, left=0.0, right=0.0) + 1j * np.interp(tau, t_SEOB, h_SEOB.imag, left=0.0, right=0.0)

    def SEOBNRv5HM_waveform(self, params, fixed_params):

        """
            SEOBNRv5HM template, treated as TEOBPM and KerrBinary: binary parameters from the NR metadata, time origin at the peak of the (2,2)
            mode (of the NR simulation for the data, of SEOBNRv5HM for the template; no time shift is fitted), and a free overall phase `phi`.
        """

        if(self._SEOB_mode is None): self._SEOB_mode = self._load_SEOBNRv5HM_mode()
        phi_value = utils.get_param_override(fixed_params,params,'phi')

        return self._SEOB_mode * np.exp(1j*phi_value)

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

        elif (self.wf_model=='SEOBNRv5HM'):

            ringdown_model       = self.SEOBNRv5HM_waveform(params, fixed_params)
            self.wf_r, self.wf_i = np.real(ringdown_model), np.imag(ringdown_model)
            
        else:
            raise ValueError("Unknown template selected: {}".format(self.wf_model))

        if not(self.wf_model=='Kerr'):
            self.wf_r, self.wf_i = self._apply_waveform_conventions(self.wf_r, self.wf_i)

        return self.wf_r + 1j * self.wf_i
