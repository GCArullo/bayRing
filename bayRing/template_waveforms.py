import pyRing.waveform as wf
import numpy as np
import cpnest.model

class WaveformModel(cpnest.model.Model):
    
    def __init__(self, t_NR, tM_start, wf_model, N_ds_modes, Kerr_modes, metadata, qnm_cached, l_NR, m_NR, tail=0, tail_modes=None, quadratic_modes=None, const_params=None):

        self.t_NR, self.t_start = t_NR, tM_start
        self.wf_model           = wf_model
        self.Kerr_modes         = Kerr_modes
        self.metadata           = metadata
        self.const_params       = const_params
        self.Mf, self.af        = self.metadata['Mf'], self.metadata['af']
        self.qnm_cached         = qnm_cached
        self.l_NR, self.m_NR    = l_NR, m_NR
        self.tail               = tail
        self.quadratic_modes    = quadratic_modes
        self.N_ds_modes         = N_ds_modes
        self.tail_modes         = tail_modes

        if not(const_params==None):
            self.const_r = [const_params[0]*np.cos(const_params[1])]
            self.const_i = [const_params[0]*np.sin(const_params[1])]
    
    def Kerr_waveform(self, params):

        amps, quad_amps, tail_parameters = {}, {}, {}
        
        # Read-in linear modes.
        for (l_ring, m_ring, n) in self.Kerr_modes:
            linear_string = '{}{}{}'.format(l_ring, m_ring, n)
            amps[(2, l_ring, m_ring, n)] = np.exp(params['ln_A_{}'.format(linear_string)]) * np.exp(1j*(params['phi_{}'.format(linear_string)]))
            
        # Read-in tail parameters.
        if(self.tail):
            for (l_ring, m_ring) in self.tail_modes:
                tail_string = '{}{}'.format(l_ring, m_ring)
                tail_parameters[(l_ring, m_ring)] = {}
                
                tail_parameters[(l_ring, m_ring)]['A']   = np.exp(params['ln_A_tail_{}'.format(tail_string)])
                tail_parameters[(l_ring, m_ring)]['phi'] =        params[ 'phi_tail_{}'.format(tail_string)]
                tail_parameters[(l_ring, m_ring)]['p']   =        params[   'p_tail_{}'.format(tail_string)]

        # Read-in quadratic modes.
        if(self.quadratic_modes is not None):
            for quad_term in self.quadratic_modes:
                quad_amps[quad_term] = {}
                for ((l,m,n),(l1,m1,n1),(l2,m2,n2)) in self.quadratic_modes[quad_term]:
                    quad_string = '{}_{}{}{}_{}{}{}_{}{}{}'.format(quad_term, l,m,n, l1,m1,n1, l2,m2,n2)
                    quad_amps[quad_term][((2,l,m,n),(2,l1,m1,n1),(2,l2,m2,n2))] = np.exp(params['ln_A_{}'.format(quad_string)]) * np.exp(1j*(params['phi_{}'.format(quad_string)]))

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
                                   interpolants        = None                 , #self.qnm_interpolants,
                                    
                                   Spheroidal          = 0                    , # Spheroidal harmonics, overrun by geom
                                   amp_non_prec_sym    = 1                    ,
                                   tail_parameters     = tail_parameters      ,
                                   quadratic_modes     = quad_amps            ,
                                   quad_lin_prop       = 0                    ,
                                   qnm_cached          = self.qnm_cached      ,

                                   charge              = self.charge          ,
                                   TGR_params          = TGR_parameters       ,
                                   )
        
        return ringdown_model
        
    def Damped_sinusoids_waveform(self, params):

        ringdown_model = np.zeros(len(self.t_NR), dtype=np.complex128)
        
        # In this case modes is an integer storing the number of free damped sinusoids
        for i in range(self.N_ds_modes):
            ringdown_model += wf.damped_sinusoid(np.exp(params[  'ln_A_{}'.format(i)]),
                                                        params[  'f_{}'.format(i)]    ,
                                                        params['tau_{}'.format(i)]    ,
                                                        params['phi_{}'.format(i)]    ,
                                                        self.t_start                  ,
                                                        self.t_NR                     )
            
        return ringdown_model

    def MMRDNP_waveform(self, params):

        TGR_parameters = {}
        MMRDNP_params  = {}

        MMRDNP_params['Mi']   = self.metadata['m1'] + self.metadata['m2']
        MMRDNP_params['eta']  = (self.metadata['m1']*self.metadata['m2'])/(MMRDNP_params['Mi'])**2
        MMRDNP_params['chis'] = (self.metadata['m1']*self.metadata['chi1'] + self.metadata['m2']*self.metadata['chi2'])/(MMRDNP_params['Mi'])
        MMRDNP_params['chia'] = (self.metadata['m1']*self.metadata['chi1'] - self.metadata['m2']*self.metadata['chi2'])/(MMRDNP_params['Mi'])

        ringdown_model = wf.MMRDNP(self.t_start                        ,
                                   self.Mf                             ,
                                   self.af                             ,

                                   MMRDNP_params['Mi']                 ,
                                   MMRDNP_params['eta']                ,
                                   MMRDNP_params['chis']               ,
                                   MMRDNP_params['chia']               ,

                                   1.0                                 , # distance, dummy with geom=1
                                   0.0                                 , # inclination, dummy with geom=1
                                   0.0                                 , # orbital phase, dummy with geom=1

                                   TGR_parameters                      ,

                                   single_l     = self.l_NR            ,
                                   single_m     = self.m_NR            ,
                                   single_mode  = 1                    ,

                                   geom         = 1                    ,
                                   qnm_fit      = 0                    ,
                                   interpolants = None                 ,
                                   qnm_cached   = self.qnm_cached      )

        return ringdown_model

    def TEOBPM_waveform(self, params):

        TGR_parameters = {}
        
        modes          = [(self.l_NR,self.m_NR)]
        merger_phases  = {mode : 0.0 for mode in modes}
        
        TGR_parameters = {}
        ringdown_model = wf.TEOBPM(self.t_start         ,
                                   self.metadata['m1']  ,
                                   self.metadata['m2']  ,
                                   self.metadata['chi1'],
                                   self.metadata['chi2'],
                                   merger_phases        ,
                                   1.0                  , # distance, dummy with geom=1
                                   0.0                  , # inclination, dummy with geom=1
                                   0.0                  , # orbital phase, dummy with geom=1
                                   modes                ,
                                   TGR_parameters       ,
                                   geom = 1             )


        return ringdown_model

    def waveform(self, params):

        if (self.wf_model=='Kerr'):
            
            ringdown_model = self.Kerr_waveform(params)
            _, _, _, self.wf_r, self.wf_i = ringdown_model.waveform(self.t_NR)
    
        elif (self.wf_model=='Damped-sinusoids'):
            
            ringdown_model = self.Damped_sinusoids_waveform(params)
            self.wf_r, self.wf_i = np.real(ringdown_model), np.imag(ringdown_model)

        elif (self.wf_model=='Kerr-Damped-sinusoids'):

            ringdown_model_Kerr = self.Kerr_waveform(params) 
            ringdown_model_DS   = self.Damped_sinusoids_waveform(params)

            _, _, _, self.wf_r_Kerr, self.wf_i_Kerr = ringdown_model_Kerr.waveform(self.t_NR)
            self.wf_r_DS, self.wf_i_DS = np.real(ringdown_model_DS), np.imag(ringdown_model_DS)

            self.wf_r = self.wf_r_Kerr + self.wf_r_DS
            self.wf_i = self.wf_i_Kerr + self.wf_i_DS   

        elif (self.wf_model=='MMRDNP'):
            
            ringdown_model                = self.MMRDNP_waveform(params)
            _, _, _, self.wf_r, self.wf_i = ringdown_model.waveform(self.t_NR)
        
        elif (self.wf_model=='TEOBPM'):
            
            ringdown_model                = self.TEOBPM_waveform(params)
            _, _, _, self.wf_r, self.wf_i = ringdown_model.waveform(self.t_NR)

        else:
            raise ValueError("Unknown template selected: {}".format(self.wf_model))

        if not(self.const_params==None):
            self.wf_r = self.wf_r + self.const_r
            self.wf_i = self.wf_i + self.const_i

        # UNDERSTAND WHY!!!!
        self.wf_r = -self.wf_r
                                    
        return self.wf_r + 1j * self.wf_i