# General python imports
import h5py, numpy as np, os
from scipy import interpolate

import sxs
try   : from cbhdb import simulation
except: pass

import bayRing.waveform_utils as waveform_utils
import bayRing.QNM_utils as QNM_utils
import pyRing.utils as pyRing_utils
import pyRing.waveform as wf

twopi = 2.*np.pi

def read_fake_NR(NR_catalog, fake_NR_modes):

    if(NR_catalog=='fake_NR'):

        fake_NR_modes_string   = fake_NR_modes.replace(',', '_')

        injection_modes_list     = []
        injection_modes_list_tmp = fake_NR_modes.split(',')
        for i in range(len(injection_modes_list_tmp)):
            l_fake_NR,m_fake_NR,n_fake_NR = int(injection_modes_list_tmp[i][0]),int(injection_modes_list_tmp[i][1]),int(injection_modes_list_tmp[i][2])
            injection_modes_list.append((l_fake_NR,m_fake_NR,n_fake_NR))

    else:
        fake_NR_modes_string = ''
        injection_modes_list = None

    return fake_NR_modes_string, injection_modes_list

def read_RWZ_simulation_parameters(sim_file):

    """
    
    Read the simulation parameters from the RWZ simulation file.

    Parameters
    ----------

    sim_file : str
        Path to the RWZ simulation file.

    Returns
    -------

    sim_params : dict
        Dictionary containing the simulation parameters.
    
    """

    sim_params = {}
    with open(sim_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == '#': continue
            else:
                line = line.split()
                try:    sim_params[line[0]] = float(line[1])
                except: sim_params[line[0]] = line[1]

    return sim_params

def read_Teukolsky_simulation_parameters(sim_file):

    """
    
    Read the simulation parameters from the Teukolsky simulation file.

    Parameters
    ----------

    sim_file : str
        Path to the Teukolsky simulation file.

    Returns
    -------

    sim_params : dict
        Dictionary containing the simulation parameters.
    
    """

    sim_params = {}
    with open(sim_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == '#': continue
            else:
                line = line.split()
                try:    sim_params[line[0]] = float(line[1])
                except: sim_params[line[0]] = line[1]

    return sim_params

def convert_resolution_level_Teukolsky(res_level):

    """

    Convert the resolution level of Teukolsky data to a string.

    Parameters
    ----------

    res_level : int or str

    Returns
    -------

    res_string : str
        String containing resolution level for Teukolsky simulations.
    
    """

    res_to_nx_nl = {
        1 : [182, 24],
        2 : [184, 24],
        3 : [186, 24],
        4 : [188, 24],
        5 : [190, 24],
        6 : [192, 24],
        7 : [194, 28],
        8 : [196, 28],
        9 : [198, 28],
        10: [200, 28]
    }

    if  (isinstance(res_level, int)): res_string = 'nx_'+str(res_to_nx_nl[res_level][0]) + '_nl_'+str(res_to_nx_nl[res_level][1])
    elif(isinstance(res_level, str)): res_string = res_level
    else              : raise ValueError(f"Allowed resolution levels for Teukolsky data are 1 (lowest) to 9 (highest), or specify res-nx and res-nl, while {res_level} was passed.")
    
    return res_string

######################
# Class for RIT sims #
######################

# Function taken from EOB_hyp repository. Credits to Rossella Gamba and Sebastiano Bernuzzi.

class Waveform_rit(object):
    def __init__(self, sims='', path='', ID='', ell=2, m=2):
        self.path   = path
        self.ID     = ID 
        self.smpath = sims
        self.ell    = ell
        self.m      = m
        self.metadata = {}

    def load_metadata(self):
        ID_str = str(self.ID)
        nm = self.smpath + '/RIT_eBBH_'+ID_str+'-n100-ecc_Metadata.txt'
        with open(nm, 'r') as f:
            lines = [l for l in f.readlines() if l.strip()] # rm empty
            for line in lines[1:]:
                if line[0]=="#": continue
                line = line.rstrip("\n")
                #line = line.split("#", 1)[0]
                key, val = line.split("= ")
                key = key.strip()
                self.metadata[key] = val
        return self.metadata 

    def load_hlm(self):
        ID_str = str(self.ID)
        nm = self.path + '/ExtrapStrain_RIT-eBBH-'+ID_str+'-n100.h5'
        f = h5py.File(nm, "r")
        u   =  f['NRTimes'][:]
        A   =  f[f'amp_l{self.ell}_m{self.m}']['Y'][:]
        A_u =  f[f'amp_l{self.ell}_m{self.m}']['X'][:]
        p   = -f[f'phase_l{self.ell}_m{self.m}']['Y'][:]
        p_u =  f[f'phase_l{self.ell}_m{self.m}']['X'][:]

        self.u  = u
        self.p  = self.interp_qnt(p_u, p, u)
        self.A  = self.interp_qnt(A_u, A, u)
        self.re = self.A*np.cos( self.p)
        self.im = self.A*np.sin(-self.p)

        return self.u, self.re, self.im, self.A, self.p
        
    def interp_qnt(self, x, y, x_new):
        f = interpolate.interp1d(x, y)
        yn= f(x_new)
        return yn

    def interpolate_hlm(self, u_new):
        re_i = self.interp_qnt(self.u, self.re, u_new)
        im_i = self.interp_qnt(self.u, self.im, u_new)
        A_i  = self.interp_qnt(self.u, self.A,  u_new)
        p_i  = self.interp_qnt(self.u, self.p,  u_new)
        return re_i, im_i, A_i, p_i

class Waveform_C2EFT(object):

    def __init__(self, path='', ell=2, m=2):

        self.path     = path
        self.ell      = ell
        self.m        = m
        self.metadata = {}

    def load_metadata(self):

        filename_metadata = os.path.join(self.path, 'metadata.txt')

        file_metadata = np.genfromtxt(filename_metadata, names=True)

        # Read initial quantities
        self.metadata['q']    = file_metadata['q']
        self.metadata['chi1'] = file_metadata['chi1']
        self.metadata['chi2'] = file_metadata['chi2']

        # Read final quantities
        self.metadata['Mf']   = file_metadata['Mf']
        self.metadata['af']   = file_metadata['af']

        # Read EFT coupling
        self.metadata['epsilon'] = file_metadata['epsilon']

        return self.metadata 

    def load_hlm(self):

        self.t,  self.re = np.loadtxt(os.path.join(self.path, 'strain_rh+22.dat'), unpack=True)
        self.t2, self.im = np.loadtxt(os.path.join(self.path, 'strain_Ih+22.dat'), unpack=True)

        return self.t, self.re, self.im
    
def read_NR_metadata(NR_sim, NR_catalog):

    """

    Read the metadata of the NR simulation.

    Parameters
    ----------

    NR_sim : NRsim object
        NRsim object containing the metadata of the NR simulation.

    NR_catalog : str
        Catalog of the NR simulation.

    Returns
    -------

    metadata : dict
        Dictionary containing the metadata of the NR simulation.

    """

    if(NR_catalog=='SXS'):

        M = 1.0
        metadata = {
                    'q'    : NR_sim.q,
                    'chi1' : NR_sim.chi1,
                    'chi2' : NR_sim.chi2,
                    'tilt1': NR_sim.tilt1,
                    'tilt2': NR_sim.tilt2,
                    'm1'   : pyRing_utils.m1_from_m_q(M, NR_sim.q),
                    'm2'   : pyRing_utils.m2_from_m_q(M, NR_sim.q),
                    'ecc'  : NR_sim.ecc,
                    'Mf'   : NR_sim.Mf,
                    'af'   : NR_sim.af,
                }
                
    elif(NR_catalog=='cbhdb'):

        M = 1.0
        metadata = {
                    'q'     : NR_sim.q,
                    'q1'    : NR_sim.q1,
                    'q2'    : NR_sim.q2,
                    'chi1'  : NR_sim.chi1,
                    'chi2'  : NR_sim.chi2,
                    'tilt1' : NR_sim.tilt1,
                    'tilt2' : NR_sim.tilt2,
                    'm1'    : pyRing_utils.m1_from_m_q(M, NR_sim.q),
                    'm2'    : pyRing_utils.m2_from_m_q(M, NR_sim.q),
                    'ecc'   : NR_sim.ecc,
                    'Mf'    : NR_sim.Mf,
                    'qf'    : NR_sim.qf,
                    'af'    : NR_sim.af,
                }

    elif(NR_catalog=='RIT'):

        M = 1.0
        metadata = {
                    'q'    : NR_sim.q,
                    'chi1' : NR_sim.chi1,
                    'chi2' : NR_sim.chi2,
                    'm1'   : pyRing_utils.m1_from_m_q(M, NR_sim.q),
                    'm2'   : pyRing_utils.m2_from_m_q(M, NR_sim.q),
                    'ecc'  : NR_sim.ecc,
                    'Mf'   : NR_sim.Mf,
                    'af'   : NR_sim.af,
                }

    elif(NR_catalog=='C2EFT'):

        M = 1.0
        metadata = {
                    'q'    : NR_sim.q,
                    'chi1' : NR_sim.chi1,
                    'chi2' : NR_sim.chi2,
                    'm1'   : pyRing_utils.m1_from_m_q(M, NR_sim.q),
                    'm2'   : pyRing_utils.m2_from_m_q(M, NR_sim.q),
                    'Mf'   : NR_sim.Mf,
                    'af'   : NR_sim.af,
                    'eps'  : NR_sim.eps,
                }

    elif(NR_catalog=='Teukolsky'):
        metadata = {
                    'Mf'   : NR_sim.Mf,
                    'af'   : NR_sim.af,
                }

    elif(NR_catalog=='charged_raw'):
        metadata = {
                    'q'     : NR_sim.q,
                    'Mf'    : NR_sim.Mf,
                    'qf'    : NR_sim.qf,
                    'af'    : NR_sim.af,
            }

    elif(NR_catalog=='RWZ'):
        metadata = {
                    'a_halo'    : NR_sim.a_halo,
                    'M_halo'    : NR_sim.M_halo,
                    'C'         : NR_sim.C,
                    'Mf'        : NR_sim.Mf,
                    'af'        : NR_sim.af,
	    }

    elif(NR_catalog=='fake_NR'):
        metadata = {
                    'q'     : NR_sim.q,
                    'Mf'    : NR_sim.Mf,
                    'af'    : NR_sim.af,
            }

    else: raise ValueError("Invalid option for NR catalog: {}".format(NR_catalog))

    return metadata

class NR_simulation():

    """

    Class for the NR simulation object.

    Parameters
    ----------

    NR_catalog : str
        Catalog of the NR simulation.

    NR_ID : str
        ID of the NR simulation.

    res_level : int
        Resolution level of the NR simulation.

    extrap_order : int
        Extrapolation order of the NR simulation.

    perturbation_order : int
        Perturbation order of the NR simulation (available for Teukolsky simulations only).

    NR_dir : str
        Directory storing local NR data.

    injection_modes_list : str
        Modes to be included in the strain obtained from the Kerr QNMs template.
        
    l : int
        l-mode of the NR simulation.

    m : int
        m-mode of the NR simulation.

    download : bool, optional
        If True, the NR simulation is downloaded from the NR catalog. Default: False.

    NR_error : str, optional
        Error of the NR simulation. Available options: 'align-with-mismatch-res-only', 'align-with-mismatch-res-and-extrap', 'align-with-mismatch-res-and-extrap-and-pert', 'constant-X' (with X=error value). Default: 'align-with-mismatch-res-only'.

    tM_start : float, optional
        Initial time of the fit. Default: 30.0.

    tM_end : float, optional
        Final time of the fit. Default: 150.0.

    t_delay_scd : float, optional
        Time delay between the NR simulation and the SCD simulation. Default: 0.0.

    t_min_mismatch : float, optional
        Initial time of the mismatch used to compute the error. Default: 2692.7480095302817 (for SXS:0305).

    t_max_mismatch : float, optional
        Final time of the mismatch used to compute the error. Default: 3792.7480095302817 (for SXS:0305).
        
    """

    def __init__(self                                           , 
                 NR_catalog                                     , 
                 NR_ID                                          , 
                 res_level                                      , 
                 extrap_order                                   , 
                 perturbation_order                             , 
                 NR_dir                                         , 
                 injection_modes_list                             , 
                 injection_times                                  , 
                 injection_noise                                      , 
                 injection_tail                                       , 
                 l                                              , 
                 m                                              , 
                 outdir                                         , 
                 download       = False                         , 
                 NR_error       = 'align-with-mismatch-res-only', 
                 tM_start       = 30.0                          , 
                 tM_end         = 150.0                         , 
                 t_delay_scd    = 0.0                           , 
                 t_min_mismatch = 2692.7480095302817            , 
                 t_max_mismatch = 3792.7480095302817            ):

        ####################
        # Input parameters #
        ####################

        self.NR_catalog       = NR_catalog
        self.NR_ID            = NR_ID
        self.res_level        = res_level
        self.extrap_order     = extrap_order

        self.l                = l
        self.m                = m
        self.pert_order       = perturbation_order

        self.NR_dir           = NR_dir
        self.outdir           = outdir

        self.fake_NR_modes    = injection_modes_list
        self.injection_noise  = injection_noise
        self.injection_tail   = injection_tail

        self.tM_start         = tM_start
        self.tM_end           = tM_end
        self.t_delay_scd      = t_delay_scd
        

        ######################
        # Read-in simulation #
        ######################
        
        if(self.NR_catalog=='fake_NR'):
            
            t_start, t_end, dt, self.q, self.Mf, self.af, self.A_dict, self.phi_dict, self.tail_dict = self.read_fake_NR_metadata()

            if(injection_times=='from-metadata'):

                self.t_start = t_start
                self.t_NR    = np.arange(self.t_start, t_end, dt)
                if(self.t_NR[0] < 0):
                    self.t_NR = self.t_NR - self.t_NR[0]

            elif(injection_times=='from-SXS-NR'):

                self.download      = download
                self.fake_error_NR = NR_error

                self.t_NR, self.NR_err_cmplx_SXS, self.t_start = self.extract_data_NR(t_min_mismatch, t_max_mismatch)

            else:

                raise ValueError("Unknown times option.")
                                            
            modes_input = []
            modes_input.append(','.join(['{}{}{}'.format(l_ring, m_ring, n) for l_ring, m_ring, n in self.fake_NR_modes]))

            metadata_tmp       = {}
            metadata_tmp['Mf'] = self.Mf
            metadata_tmp['af'] = self.af
           
            _, _, _, _, self.qnm_cached = QNM_utils.read_Kerr_modes(modes_input, None, None, self.l, self.m, metadata_tmp)

            amps = {}
        
            # Read-in linear modes.
            for (l_ring, m_ring, n) in self.fake_NR_modes:
                linear_string = '{}{}{}'.format(l_ring, m_ring, n)

                if 'A_{}'.format(linear_string) in self.A_dict.keys(): 
                    amps[(2, l_ring, m_ring, n)] = self.A_dict['A_{}'.format(linear_string)] * np.exp(1j*(self.phi_dict['phi_{}'.format(linear_string)]))
                else:
                    print("Mode not present in the metadata. Please update the metadata or change the input modes to be included in the template for the fake NR data.")
                    exit()

            ringdown_fun = wf.KerrBH(self.t_start                         ,
                                     self.Mf                              ,
                                     self.af                              ,
                                     amps                                 ,
                                     0.0                                  , # distance,    overrun by geom
                                     0.0                                  , # inclination, overrun by geom
                                     0.0                                  , # phi,         overrun by geom
                                    
                                     reference_amplitude = 0.0            ,
                                     geom                = 1              ,
                                     qnm_fit             = 0              ,
                                     interpolants        = None           ,
                                    
                                     Spheroidal          = 0              , # Spheroidal harmonics, overrun by geom
                                     amp_non_prec_sym    = 1              ,
                                     tail_parameters     = {}             ,
                                     quadratic_modes     = {}             ,
                                     quad_lin_prop       = 0              ,
                                     qnm_cached          = self.qnm_cached,

                                     charge              = 0              ,
                                     TGR_params          = None           ,
                                     )
            
            _, _, _, self.NR_r, self.NR_i = ringdown_fun.waveform(self.t_NR)

            self.NR_r = -self.NR_r

        elif(self.NR_catalog=='charged_raw'):

            # Load NR simulation
            path_NR_r     = self.NR_dir + f'/strains/{NR_ID}_times.dat'
            path_NR_i     = self.NR_dir + f'/strains/{NR_ID}_cross.dat'
            self.data_r   = np.genfromtxt(path_NR_r)
            self.data_i   = np.genfromtxt(path_NR_i)
            
            # Built NR waveform and time axis
            self.NR_r = np.array([self.data_r[i][1] for i in range(len(self.data_r))])
            self.NR_i = np.array([self.data_i[i][1] for i in range(len(self.data_i))])
            self.t_NR = np.array([self.data_r[i][0] for i in range(len(self.data_r))])

            # Define metadata in the class
            self.q, self.Mf, self.qf, self.af, self.ecc = self.read_charged_raw_metadata()
      
        elif(self.NR_catalog=='cbhdb'):
    
            # Load NR simulation
            for res_level_x in [2,1]:
                try:
                    path_waveform     = self.NR_dir + f'/{NR_ID}_lev-2.h5'
                    self.waveform_obj = simulation.Simulation.from_file(path_waveform)
                    self.res_level    = res_level_x
                    break
                except(ValueError):
                    pass
            print("\n* Setting the resolution level to the maximum available: {}\n".format(self.res_level))

            if('align-with-mismatch' in NR_error):
                try:
                    path_waveform2            = self.NR_dir + f'/{NR_ID}_lev-{self.res_level-1}.h5'
                    self.waveform_obj2        = simulation.Simulation.from_file(path_waveform2)
                    NR_h2                     = self.waveform_obj2.processed.rhlm_finite_radius[(self.l, self.m)]
                    t_res, NR_r_res, NR_i_res = self.waveform_obj2.processed.rhlm_finite_radius_times, NR_h2.real, NR_h2.imag
                except(ValueError):
                    print("Lower resolution not found!")
                    raise

            # Built NR waveform and time axis
            NR_h      = self.waveform_obj.processed.rhlm_finite_radius[(self.l, self.m)]
            self.NR_r = NR_h.real
            self.NR_i = NR_h.imag
            self.t_NR = self.waveform_obj.processed.rhlm_finite_radius_times

            # Define metadata in the class
            self.q, self.q1, self.q2, self.chi1, self.chi2, self.tilt1, self.tilt2, self.ecc, self.Mf, self.qf, self.af = self.read_cbhdb_metadata()

        elif(self.NR_catalog=='SXS'):
        
            self.download = download
            self.q, self.chi1, self.chi2, self.tilt1, self.tilt2, self.ecc, self.Mf, self.af = self.read_SXS_metadata()

            # Build NR waveform and time axis.

            if(self.res_level==-1):
                for res_level_x in [6,5,4,3,2,1]:
                    try: 
                        self.t_NR, self.NR_r, self.NR_i = self.read_hlm_from_SXS(self.extrap_order, res_level_x)
                        self.res_level = res_level_x
                        break
                    except(ValueError):
                        pass
                print("\n* Setting the resolution level to the maximum available: {}\n".format(self.res_level))
            else:
                self.t_NR, self.NR_r, self.NR_i = self.read_hlm_from_SXS(self.extrap_order, self.res_level)

            t_res,     NR_r_res,  NR_i_res  = self.read_hlm_from_SXS(self.extrap_order,   self.res_level-1)
            t_extr,    NR_r_extr, NR_i_extr = self.read_hlm_from_SXS(self.extrap_order+1, self.res_level)

        elif(self.NR_catalog=='RIT'):
        
            print('\n\n\nFIXME: figure out extrapolation order and resolution level for RIT\n\n\n')

            self.q, self.chi1, self.chi2, self.ecc, self.Mf, self.af = self.read_RIT_metadata()

            # Build NR waveform and time axis.
            self.t_NR, self.NR_r, self.NR_i = self.read_hlm_from_RIT()
            t_res,     NR_r_res,  NR_i_res  = None, None, None
            t_extr,    NR_r_extr, NR_i_extr = None, None, None

        elif(self.NR_catalog=='C2EFT'):
        
            print('\n\n\nFIXME: i) Should compare resolutions with same sigmas; ii) Pass as inputs extrapolation order and resolution level error for C2EFT.\n\n\n')

            res_1   = 64
            res_2   = 88
            sigma_1 = 0.0625
            sigma_2 = 0.1
            tau_1   = 0.005

            self.q, self.chi1, self.chi2, self.Mf, self.af, self.eps = self.read_C2EFT_metadata(resolution = res_1, sigma = sigma_1, tau = tau_1)
            self.ecc = 0.0

            # Build NR waveform and time axis. 
            self.t_NR, self.NR_r, self.NR_i = self.read_hlm_from_C2EFT(resolution = res_1, sigma = sigma_1, tau = tau_1)
            t_res,     NR_r_res,  NR_i_res  = self.read_hlm_from_C2EFT(resolution = res_2, sigma = sigma_2, tau = tau_1)
            t_extr,    NR_r_extr, NR_i_extr = self.read_hlm_from_C2EFT(resolution = res_1, sigma = sigma_2, tau = tau_1)

        elif(self.NR_catalog=='Teukolsky'):
        
            self.Mf, self.af                = self.read_Teukolsky_metadata()
            self.t_NR, self.NR_r, self.NR_i = self.read_hlm_from_Teukolsky(self.res_level)
            if not isinstance(self.res_level, str):
                try: t_res, NR_r_res,  NR_i_res = self.read_hlm_from_Teukolsky(self.res_level-1)
                except: print('\n* Teukolsky resolution level {} not available.\n'.format(self.res_level-1))
            else:
                if(NR_error=='resolution'): raise ValueError("Resolution error not yet available when using nx,nl as resolution indicators.")
            t_extr, NR_r_extr, NR_i_extr   = None, None, None
            self.ecc = 0.0
        
        elif(self.NR_catalog=='RWZ'):

            # Read the metadata
            self.a_halo, self.M_halo, self.C = self.read_RWZ_metadata()
            self.ecc, self.Mf, self.af = 0.0, 1.0, 0.0

            # Build NR waveform and time axis
            self.t_NR, self.NR_r, self.NR_i = self.read_hlm_from_RWZ()
            t_res,     NR_r_res,  NR_i_res  = None, None, None
            t_extr,    NR_r_extr, NR_i_extr = None, None, None

        # Auxiliary quantities for the reference NR simulation.
        self.NR_cpx                         = self.NR_r + 1j * self.NR_i
        self.NR_amp, self.NR_phi            = waveform_utils.amp_phase_from_re_im(self.NR_r, self.NR_i)

        ####################
        # Error estimation #
        ####################

        if(self.NR_catalog=='SXS'):

            if('constant' in NR_error):
                error_value                = float(NR_error.split('-')[-1])
                self.NR_err_cmplx          = self.generate_constant_error(error_value)

            else:

                # Align the waveforms minimising the mismatch over a [t_min, t_max] interval.
                if('align-with-mismatch' in NR_error):
                    
                    # Resolution error.
                    NR_r_res    , NR_i_res       = waveform_utils.align_waveforms_with_mismatch(self.t_NR, self.NR_amp, self.NR_phi,  t_res,  NR_r_res,  NR_i_res, t_min_mismatch, t_max_mismatch)
                    NR_r_err_res, NR_i_err_res   = np.abs(self.NR_r-NR_r_res), np.abs(self.NR_i-NR_i_res)

                    # Extrapolation error.  Align different extrapolation orders only if requested.
                    if(NR_error=='align-with-mismatch-all'): 
                        NR_r_extr, NR_i_extr     = waveform_utils.align_waveforms_with_mismatch(self.t_NR, self.NR_amp, self.NR_phi, t_extr, NR_r_extr, NR_i_extr, t_min_mismatch, t_max_mismatch)
                    NR_r_err_extr, NR_i_err_extr = np.abs(self.NR_r-NR_r_extr), np.abs(self.NR_i-NR_i_extr)

                # Align the waveforms at the peak.
                elif(NR_error=='align-at-peak'):

                    # Resolution error.
                    NR_r_res    , NR_i_res       = waveform_utils.align_waveforms_at_peak(self.t_NR, self.NR_amp, self.NR_phi, t_res, NR_r_res, NR_i_res)
                    NR_r_err_res, NR_i_err_res   = np.abs(self.NR_r-NR_r_res), np.abs(self.NR_i-NR_i_res)

                    # Extrapolation error. Do not align different extrapolation orders with this method.
                    NR_r_err_extr, NR_i_err_extr = np.abs(self.NR_r-NR_r_extr), np.abs(self.NR_i-NR_i_extr)

                else:
                    raise ValueError("Unknown NR error option.")
                
                # Global error
                self.NR_err_cmplx = np.sqrt(NR_r_err_extr**2 + NR_r_err_res**2) + 1j * np.sqrt(NR_i_err_extr**2 + NR_i_err_res**2)
            
        elif(self.NR_catalog=='RIT'):
            error_value       = float(NR_error.split('-')[-1])
            self.NR_err_cmplx = self.generate_constant_error(error_value)   

        elif(self.NR_catalog=='C2EFT'):

            if('constant' in NR_error):
                error_value                = float(NR_error.split('-')[-1])
                self.NR_err_cmplx          = self.generate_constant_error(error_value)

            else:

                # Align the waveforms minimising the mismatch over a [t_min, t_max] interval.
                if('align-with-mismatch' in NR_error):
                    
                    # Resolution error.
                    NR_r_res    , NR_i_res       = waveform_utils.align_waveforms_with_mismatch(self.t_NR, self.NR_amp, self.NR_phi,  t_res,  NR_r_res,  NR_i_res, t_min_mismatch, t_max_mismatch)
                    NR_r_err_res, NR_i_err_res   = np.abs(self.NR_r-NR_r_res), np.abs(self.NR_i-NR_i_res)

                    # Extrapolation error.
                    NR_r_extr    , NR_i_extr     = waveform_utils.align_waveforms_with_mismatch(self.t_NR, self.NR_amp, self.NR_phi,  t_extr,  NR_r_extr,  NR_i_extr, t_min_mismatch, t_max_mismatch)
                    NR_r_err_extr, NR_i_err_extr = np.abs(self.NR_r-NR_r_extr), np.abs(self.NR_i-NR_i_extr)
                    NR_r_err_extr, NR_i_err_extr = np.zeros(len(self.t_NR)), np.zeros(len(self.t_NR))

                # Align the waveforms at the peak.
                elif(NR_error=='align-at-peak'):

                    # Resolution error.
                    NR_r_res    , NR_i_res       = waveform_utils.align_waveforms_at_peak(self.t_NR, self.NR_amp, self.NR_phi, t_res, NR_r_res, NR_i_res)
                    NR_r_err_res, NR_i_err_res   = np.abs(self.NR_r-NR_r_res), np.abs(self.NR_i-NR_i_res)

                    # Extrapolation error.
                    NR_r_extr    , NR_i_extr     = waveform_utils.align_waveforms_at_peak(self.t_NR, self.NR_amp, self.NR_phi, t_extr, NR_r_extr, NR_i_extr)
                    NR_r_err_extr, NR_i_err_extr = np.abs(self.NR_r-NR_r_extr), np.abs(self.NR_i-NR_i_extr)

                # Global error
                self.NR_err_cmplx = np.sqrt(NR_r_err_extr**2 + NR_r_err_res**2) + 1j * np.sqrt(NR_i_err_extr**2 + NR_i_err_res**2)

        elif(self.NR_catalog=='RWZ'):

            # Waveforms at different resolution levels are already aligned.
            if(NR_error=='resolution'):
                raise ValueError("Resolution error not yet available for RWZ simulations.")
                if np.shape(self.NR_r) != np.shape(NR_r_res):
                    if np.shape(NR_r_res) < np.shape(self.NR_r):
                        NR_r_res = np.append(NR_r_res, np.zeros(len(self.NR_r) - len(NR_r_res))) 
                        NR_i_res = np.append(NR_r_res, np.zeros(len(self.NR_r) - len(NR_r_res)))
                    else:
                        NR_r_res = NR_r_res[:len(self.NR_r)]
                        NR_i_res = NR_i_res[:len(self.NR_r)]
                NR_r_err_res, NR_i_err_res = np.abs(self.NR_r-NR_r_res), np.abs(self.NR_i-NR_i_res)
                self.NR_err_cmplx          = NR_r_err_res + 1j * NR_i_err_res
            elif('constant' in NR_error):
                error_value                = float(NR_error.split('-')[-1])
                self.NR_err_cmplx          = self.generate_constant_error(error_value)
            else:
                raise ValueError("Unknown NR error option.")

        elif(self.NR_catalog=='Teukolsky'):

            # Waveforms at different resolution levels are already aligned.
            if(NR_error=='resolution'):
                if np.shape(self.NR_r) != np.shape(NR_r_res):
                    if np.shape(NR_r_res) < np.shape(self.NR_r):
                        NR_r_res = np.append(NR_r_res, np.zeros(len(self.NR_r) - len(NR_r_res))) 
                        NR_i_res = np.append(NR_r_res, np.zeros(len(self.NR_r) - len(NR_r_res)))
                    else:
                        NR_r_res = NR_r_res[:len(self.NR_r)]
                        NR_i_res = NR_i_res[:len(self.NR_r)]
                NR_r_err_res, NR_i_err_res = np.abs(self.NR_r-NR_r_res), np.abs(self.NR_i-NR_i_res)
                self.NR_err_cmplx          = NR_r_err_res + 1j * NR_i_err_res
            elif('constant' in NR_error):
                error_value                = float(NR_error.split('-')[-1])
                self.NR_err_cmplx          = self.generate_constant_error(error_value)
            else:
                raise ValueError("Unknown NR error option.")
                
        elif(self.NR_catalog=='cbhdb'):
            
            if('constant' in NR_error):
                error_value                = float(NR_error.split('-')[-1])
                self.NR_err_cmplx          = self.generate_constant_error(error_value)
        
            # Align the waveforms minimising the mismatch over a [t_min, t_max] interval.
            if('align-with-mismatch' in NR_error):
                
                # Resolution error.
                NR_r_res    , NR_i_res     = waveform_utils.align_waveforms_with_mismatch(self.t_NR, self.NR_amp, self.NR_phi,  t_res,  NR_r_res,  NR_i_res, t_min_mismatch, t_max_mismatch)
                NR_r_err_res, NR_i_err_res = np.abs(self.NR_r-NR_r_res), np.abs(self.NR_i-NR_i_res)
            
                self.NR_err_cmplx          = NR_r_err_res + 1j * NR_i_err_res


        elif(self.NR_catalog=='charged_raw'):
            if('constant' in NR_error):
                error_value                = float(NR_error.split('-')[-1])
                self.NR_err_cmplx          = self.generate_constant_error(error_value)
       
        elif(self.NR_catalog=='fake_NR'):
            
            if('gaussian' in NR_error):
                error_value                = float(NR_error.split('-')[-1])
                self.NR_err_cmplx          = np.array([(error_value + error_value*1j) for i in range(len(self.t_NR))])
                
                if not(self.injection_noise==None):
                    NR_inj_err_cmplx  = self.generate_gaussian_error(error_value, len(self.t_NR))
                    for i in range(len(self.NR_r)):
                        # self.NR_r[i] += np.real(NR_inj_err_cmplx[i])
                        # self.NR_i[i] += np.imag(NR_inj_err_cmplx[i])
                        self.NR_r[i] += error_value
                        self.NR_i[i] += error_value

            elif('constant' in NR_error):
                error_value                = float(NR_error.split('-')[-1])
                self.NR_err_cmplx          = self.generate_constant_error(error_value)
                
                if not(self.injection_noise==None):
                    for i in range(len(self.NR_r)):
                        # self.NR_r[i] += np.real(self.NR_err_cmplx[i])
                        # self.NR_i[i] += np.imag(self.NR_err_cmplx[i])
                        self.NR_r[i] += error_value
                        self.NR_i[i] += error_value

            elif(NR_error=='from-SXS-NR'):
                self.NR_err_cmplx          = self.NR_err_cmplx_SXS
            
                if not(self.injection_noise==None):
                    for i in range(len(self.NR_r)):
                        # self.NR_r[i] += np.random.normal(loc=0, scale=np.real(self.NR_err_cmplx.data[i]), size=1)[0]
                        # self.NR_i[i] += np.random.normal(loc=0, scale=np.imag(self.NR_err_cmplx.data[i]), size=1)[0]
                        self.NR_r[i] += np.real(self.NR_err_cmplx.data[i])
                        self.NR_i[i] += np.imag(self.NR_err_cmplx.data[i])

        # Start from zero.
        if(self.t_NR[0] < 0):
            self.t_NR = self.t_NR - self.t_NR[0]
        
        # Locate the merger time (which does not coincide with the peak in the eccentric case).
        self.t_peak = waveform_utils.find_peak_time(self.t_NR, self.NR_amp, self.ecc)

        # For convenience, for second order perturbations, allow the option to build the time axis from the secondary peak.
        if(self.NR_catalog=='Teukolsky' and self.pert_order=='lin'): 
            print("\n* The peak time has been set to the secondary peak time with a delay of: {}.".format(self.t_delay_scd))
            self.t_peak = self.t_peak + self.t_delay_scd

        self.NR_freq  = np.gradient(self.NR_phi, self.t_NR)/(twopi)
        
        # Restrict computations to [t_min, t_max]
        self.t_min, self.t_max = self.t_peak + tM_start, self.t_peak + tM_end
        idx_min, idx_max       = np.where((self.t_NR - self.t_min)>=0)[0][0], np.where((self.t_NR - self.t_max)<=0)[0][-1]
        self.t_NR_cut          = self.t_NR[idx_min:idx_max]

        self.NR_cpx_cut        = self.NR_cpx[idx_min:idx_max]
        self.NR_cpx_err_cut    = self.NR_err_cmplx[idx_min:idx_max]
        self.NR_r_cut          = self.NR_r[idx_min:idx_max]
        self.NR_i_cut          = self.NR_i[idx_min:idx_max]
        self.NR_amp_cut        = self.NR_amp[idx_min:idx_max]
        self.NR_phi_cut        = self.NR_phi[idx_min:idx_max]
        self.NR_freq_cut       = self.NR_freq[idx_min:idx_max]

        # Store the peaktime to facilitate post-processing
        print("\n* The peak time is t_peak = {}".format(self.t_peak))
        np.savetxt(os.path.join(self.outdir,'Peak_quantities/Peak_time.txt'), np.array([self.t_peak]), header = "t_peak [sim units]")
       
    def extract_data_NR(self, t_min_mismatch, t_max_mismatch):

        # Build NR time axis.

        if(self.res_level==-1):
            for res_level_x in [6,5,4,3,2,1]:
                try: 
                    t_NR, NR_r, NR_i = self.read_hml_from_SXS(self.extrap_order, res_level_x)
                    self.res_level = res_level_x
                    break
                except(ValueError):
                    pass
        else:
            t_NR, NR_r, NR_i = self.read_hml_from_SXS(self.extrap_order, self.res_level)

        NR_amp, NR_phi               = waveform_utils.amp_phase_from_re_im(NR_r, NR_i)

        # Build NR error array.

        if(self.fake_error_NR=='from-SXS-NR'):
            t_res,  NR_r_res,  NR_i_res  = self.read_hml_from_SXS(self.extrap_order,   self.res_level-1)
            t_extr, NR_r_extr, NR_i_extr = self.read_hml_from_SXS(self.extrap_order+1, self.res_level)

            NR_r_res    , NR_i_res       = waveform_utils.align_waveforms_with_mismatch(t_NR, NR_amp, NR_phi,  t_res,  NR_r_res,  NR_i_res, t_min_mismatch, t_max_mismatch)
            NR_r_err_res, NR_i_err_res   = np.abs(NR_r-NR_r_res), np.abs(NR_i-NR_i_res)

            NR_r_err_extr, NR_i_err_extr = np.abs(NR_r-NR_r_extr), np.abs(NR_i-NR_i_extr)

            NR_err_cmplx = np.sqrt(NR_r_err_extr**2 + NR_r_err_res**2) + 1j * np.sqrt(NR_i_err_extr**2 + NR_i_err_res**2)
        else:
            NR_err_cmplx = None
        
        if(t_NR[0] < 0):
            t_NR = t_NR - t_NR[0]
        t_peak   = t_NR[np.argmax(NR_amp)]

        return t_NR, NR_err_cmplx, t_peak
        
    def read_fake_NR_metadata(self):
        
        """
        
        Read the metadata to create the fake NR data using the QNMs template.

        Parameters
        ----------

        None.
        
        Returns
        -------

        t_start
            Initial time to generate the data.
        t_end
            Final time for which to generate the data
        dt
            Time step between each point.
        q
            Mass ratio.
        Mf
            Final mass of the remnant black hole.
        af
            Final dimensionless spin of the remnant black hole.
        A_dict
            Dictionary of the QNM modes amplitudes.
        phi_dict
            Dictionary of the QNM modes phases.
        """

        path_metadata = self.NR_dir + f'/metadata_{self.NR_ID}.txt'

        with open(path_metadata, 'r') as input_file:

            for line in input_file:
            
                if line.startswith("t_start"):
                    t_start = float(line.split(':')[1].strip().split()[0])
                
                elif line.startswith("t_end"):
                    t_end   = float(line.split(':')[1].strip().split()[0])

                elif line.startswith("dt"):
                    dt      = float(line.split(':')[1].strip().split()[0])

                elif line.startswith("q"):
                    q       = float(line.split(':')[1].strip().split()[0])

                elif line.startswith("Mf"):
                    Mf      = float(line.split(':')[1].strip().split()[0])

                elif line.startswith("af"):
                    af      = float(line.split(':')[1].strip().split()[0])
                    
                elif line.startswith("A_220"):
                    A_220   = float(line.split(':')[1].strip().split()[0])
                    
                elif line.startswith("phi_220"):
                    phi_220 = float(line.split(':')[1].strip().split()[0])
                
                elif line.startswith("A_220"):
                    A_220   = float(line.split(':')[1].strip().split()[0])
                    
                elif line.startswith("phi_220"):
                    phi_220 = float(line.split(':')[1].strip().split()[0])
                
                elif line.startswith("A_221"):
                    A_221   = float(line.split(':')[1].strip().split()[0])
                    
                elif line.startswith("phi_221"):
                    phi_221 = float(line.split(':')[1].strip().split()[0])

                elif line.startswith("A_320"):
                    A_320   = float(line.split(':')[1].strip().split()[0])
                    
                elif line.startswith("phi_320"):
                    phi_320 = float(line.split(':')[1].strip().split()[0])
              
                elif line.startswith("A_22_tail"):
                    A_22_tail = float(line.split(':')[1].strip().split()[0])
    
                elif line.startswith("p_22_tail"):
                    p_22_tail = float(line.split(':')[1].strip().split()[0])
                
                elif line.startswith("phi_22_tail"):
                    phi_22_tail = float(line.split(':')[1].strip().split()[0])
                 
                    break

        A_dict    = {'A_220' : A_220, 'A_221' : A_221, 'A_320' : A_320}
        phi_dict  = {'phi_220' : phi_220, 'phi_221' : phi_221, 'phi_320' : phi_320}
        tail_dict = {'A_22_tail' : A_22_tail, 'p_22_tail' : p_22_tail, 'phi_22_tail' : phi_22_tail}
       
        return t_start, t_end, dt, q, Mf, af, A_dict, phi_dict, tail_dict
        
    def read_cbhdb_metadata(self):
        
        """
        
        Read the metadata of the cbhdb waveform.

        Parameters
        ----------

        None.
        
        Returns
        -------

        q
            Mass ratio.
        q1
            Charge of the primary black hole.
        q2
            Charge of the secondary black hole.
        chi1
            Dimensionless spin of the primary black hole.
        chi2
            Dimensionless spin of the secondary black hole.
        tilt1
            Tilt of the primary black hole.
        tilt2
            Tilt of the secondary black hole.
        ecc
            Eccentricity of the binary.
        Mf
            Final mass of the remnant black hole.
        qf
            Final charge of the remnant black hole.
        chif
            Final dimensionless spin of the remnant black hole.

        """

        metadata         = self.waveform_obj.metadata
        
        tilt1, tilt2     = 0.0, 0.0
        M1 , M2          = metadata['initial_mass1'], metadata['initial_mass2']
        q1, q2, qf       = metadata['reference_charge1'], metadata['reference_charge2'], metadata['remnant_charge']
        q, Mf            = M2/M1 , metadata['remnant_mass']
        chi1, chi2, chif = metadata['reference_dimensionless_spin1'][2], metadata['reference_dimensionless_spin2'][2], metadata['remnant_dimensionless_spin'][2]
        ecc              = metadata['reference_eccentricity']

        return q, q1, q2, chi1, chi2, tilt1, tilt2, ecc, Mf, qf, chif
        
    def read_charged_raw_metadata(self):
        
        """
        
        Read the metadata of the charged raw waveform repo.

        Parameters
        ----------

        None.
        
        Returns
        -------

        q
            Mass ratio.
        Mf
            Final mass of the remnant black hole.
        qf
            Final charge of the remnant black hole.
        chif
            Final dimensionless spin of the remnant black hole.

        """

        path_metadata = self.NR_dir + f'/metadata/metadata_{self.NR_ID}.txt'

        with open(path_metadata, 'r') as input_file:

            for line in input_file:
            
                if line.startswith("Qf"):
                    qf = float(line.split(':')[1].strip().split()[0])
                    
                elif line.startswith("Mf"):
                    Mf = float(line.split(':')[1].strip().split()[0])
                    
                elif line.startswith("af"):
                    af = float(line.split(':')[1].strip().split()[0])
                
                elif line.startswith("q"):
                    q  = float(line.split(':')[1].strip().split()[0])
                    break

        ecc = 0.005 # From arXiv:2006.15764

        return q, Mf, qf, af, ecc

    def read_SXS_metadata(self):

        """

        Read the metadata of the SXS waveform.

        Parameters
        ----------

        None.

        Returns
        -------

        q
            Mass ratio.
        chi1
            Dimensionless spin of the primary black hole.
        chi2
            Dimensionless spin of the secondary black hole.
        tilt1
            Tilt of the primary black hole.
        tilt2
            Tilt of the secondary black hole.
        ecc
            Eccentricity of the binary.
        Mf
            Final mass of the remnant black hole.
        chif
            Final dimensionless spin of the remnant black hole.

        """
        
        metadata      = sxs.load("SXS:BBH:{}/Lev/metadata.json".format(self.NR_ID), download=self.download)
        
        tilt1, tilt2  = 0.0, 0.0

        q, Mf            = metadata['reference_mass_ratio'], metadata['remnant_mass']
        chi1, chi2, chif = metadata['reference_dimensionless_spin1'][2], metadata['reference_dimensionless_spin2'][2], metadata['remnant_dimensionless_spin'][2]
        ecc              = metadata['reference-eccentricity']

        return q, chi1, chi2, tilt1, tilt2, ecc, Mf, chif

    # FIXME: The two functions below have been written in a rush and should be adapted to the overall code style.
    def read_RIT_metadata(self):

        """

        Read the metadata of the RIT waveform.

        Parameters
        ----------

        None.

        Returns
        -------

        q
            Mass ratio.

        chi1
            Dimensionless spin of the primary black hole.

        chi2
            Dimensionless spin of the secondary black hole.

        tilt1
            Tilt of the primary black hole.

        tilt2
            Tilt of the secondary black hole.

        ecc
            Eccentricity of the binary. 

        Mf
            Final mass of the remnant black hole.

        chif
            Final dimensionless spin of the remnant black hole. 

        """

                
        h_NR = Waveform_rit(sims=os.path.join(self.NR_dir, 'Metadata'), path=os.path.join(self.NR_dir, 'Data'), ID=self.NR_ID)
        
        # Read intrinsic parameters
        data  = h_NR.load_metadata()
        m1    = float(data['initial-mass1'])
        m2    = float(data['initial-mass2'])
        chi1z = float(data['initial-bh-chi1z'])
        chi2z = float(data['initial-bh-chi2z'])
        q     = m1/m2
        nu    = q/(1+q)**2

        # Read initial conditions.
        # FIXME: these are the initial data before relaxation, so not precisely correct.
        r0   = float(data['initial-separation'])
        e0   = float(data['initial-ADM-energy'])
        j0   = float(data['initial-ADM-angular-momentum-z'])/nu
        ecc  = float(data['eccentricity'])

        Mf   = float(data['final-mass'])
        chif = float(data['final-chi'])

        return q, chi1z, chi2z, ecc, Mf, chif

    def read_hlm_from_RIT(self):

        """

        Read a given (l,m) mode of an RIT simulation.

        Parameters
        ----------

        None.

        Returns
        -------

        t_NR
            Time array of the (l,m) mode.

        wv_re
            Real part of the (l,m) mode.

        wv_im
            Imaginary part of the (l,m) mode.

        """
                
        h_NR = Waveform_rit(sims=os.path.join(self.NR_dir, 'Metadata'), path=os.path.join(self.NR_dir, 'Data'), ID=self.NR_ID, ell = self.l, m = self.m)                
        t_NR, wv_re, wv_im, _, _  = h_NR.load_hlm()

        t_NR = t_NR.astype(np.float64)
        
        return t_NR, wv_re, wv_im

    # FIXME: The two functions below have been written in a rush and should be adapted to the overall code style.
    def read_C2EFT_metadata(self, resolution, sigma, tau):

        """

        Read the metadata of the C2EFT waveform.

        Parameters
        ----------

        None.

        Returns
        -------

        q
            Mass ratio.

        chi1
            Dimensionless spin of the primary black hole.

        chi2
            Dimensionless spin of the secondary black hole.

        Mf
            Final mass of the remnant black hole.

        chif
            Final dimensionless spin of the remnant black hole. 

        eps 
            Coupling of the EFT.

        """

                
        h_NR = Waveform_C2EFT(path=os.path.join(self.NR_dir, self.NR_ID, 'Res_{resolution}_sigma{sigma}_tau_{tau}'.format(resolution=resolution, sigma=str(sigma).replace('.', 'p'), tau=str(tau).replace('.', 'p'))))
        
        # Read intrinsic parameters
        data  = h_NR.load_metadata()

        q     = float(data['q'])
        chi1z = float(data['chi1'])
        chi2z = float(data['chi2'])

        Mf   = float(data['Mf'])
        chif = float(data['af'])

        eps  = float(data['epsilon'])

        return q, chi1z, chi2z, Mf, chif, eps

    def read_hlm_from_C2EFT(self, resolution, sigma, tau):

        """

        Read a given (l,m) mode of an C2EFT simulation.

        Parameters
        ----------

        None.

        Returns
        -------

        t_NR
            Time array of the (l,m) mode.

        wv_re
            Real part of the (l,m) mode.

        wv_im
            Imaginary part of the (l,m) mode.

        """
                
        h_NR = Waveform_C2EFT(path=os.path.join(self.NR_dir, self.NR_ID, 'Res_{resolution}_sigma{sigma}_tau_{tau}'.format(resolution=resolution, sigma=str(sigma).replace('.', 'p'), tau=str(tau).replace('.', 'p'))), ell = self.l, m = self.m)                
        t_NR, wv_re, wv_im = h_NR.load_hlm()

        t_NR = t_NR.astype(np.float64)
        
        return t_NR, wv_re, wv_im

    def read_hlm_from_SXS(self, ExtOrd, LevRes):

        """

        Read a given (l,m) mode of an SXS simulation.

        Parameters
        ----------

        ExtOrd
            Extrapolation order of the waveform.

        LevRes
            Resolution level of the waveform.

        Returns
        -------

        t_NR
            Time array of the (l,m) mode.

        wv_re
            Real part of the (l,m) mode.

        wv_im
            Imaginary part of the (l,m) mode.

        """
        
        waveform = sxs.load("SXS:BBH:{}/Lev{}/rhOverM_Asymptotic_GeometricUnits_CoM.h5".format(self.NR_ID, LevRes), extrapolation_order=ExtOrd, download=self.download)
        
        time        = waveform.t
        mode_index  = waveform.index(self.l, self.m)
        waveform_lm = waveform[:, mode_index]
        
        return time, waveform_lm.real, waveform_lm.imag

    def read_Teukolsky_metadata(self):

        """

        Read the metadata of the Teukolsky waveform.

        Parameters
        ----------

        None.

        Returns
        -------

        Mf
            Mass of the black hole.

        af
            Dimensionless spin of the black hole.

        """ 
        
        res_level_string = convert_resolution_level_Teukolsky(self.res_level)
        sim_path         = os.path.join(self.NR_dir, '{}_{}'.format(res_level_string, self.NR_ID))

        # Simulation units are in M/2
        simulation_parameters = read_Teukolsky_simulation_parameters(os.path.join(sim_path, 'sim_params.txt'))
        Mf, af                = simulation_parameters['black_hole_mass']*2, simulation_parameters['black_hole_spin']*2

        return Mf, af

    def read_RWZ_metadata(self):

        """

        Read the metadata of the RWZ waveform.

        Parameters
        ----------

        None.

        Returns
        -------

        Mf
            Mass of the black hole.

        af
            Dimensionless spin of the black hole.

        """ 
        
        sim_path = os.path.join(self.NR_dir, '{}'.format(self.NR_ID))

        # Simulation units are in M/2
        simulation_parameters = read_RWZ_simulation_parameters(os.path.join(sim_path, 'sim_params.txt'))
        a_halo, M_halo, C     = simulation_parameters['a_halo'], simulation_parameters['M_halo'], simulation_parameters['C']

        return a_halo, M_halo, C

    def read_hlm_from_RWZ(self):

        """

        Read a given (l,m) mode of a RWZ simulation.

        Returns
        -------

        t_NR
            Time array of the (l,m) mode.

        wv_re
            Real part of the (l,m) mode.

        wv_im
            Imaginary part of the (l,m) mode.

        """
    
        print(self.NR_dir, self.NR_ID)

        sim_path  = os.path.join(self.NR_dir, '{}'.format(self.NR_ID), f'HplusHcrossLM{self.l}{self.m}.dat')
        sim_file  = np.genfromtxt(sim_path, names=True)
        
        time             = sim_file['t']
        waveform_real    = sim_file['hp']
        waveform_imag    = sim_file['hc']
        
        return time, waveform_real, waveform_imag

    def read_hlm_from_Teukolsky(self, res_level):

        """

        Read a given (l,m) mode of a Teukolsky simulation.

        Parameters
        ----------

        res_level
            Resolution level of the waveform.   

        Returns
        -------

        t_NR
            Time array of the (l,m) mode.

        wv_re
            Real part of the (l,m) mode.

        wv_im
            Imaginary part of the (l,m) mode.

        """
    
        res_level_string = convert_resolution_level_Teukolsky(res_level)
        sim_path         = os.path.join(self.NR_dir, '{}_{}'.format(res_level_string, self.NR_ID))
        time             = np.genfromtxt(os.path.join(sim_path, 'tvals.dat'))
        waveform_real    = np.genfromtxt(os.path.join(sim_path, '{pert}_h_{l}{m}_re.dat'.format(pert=self.pert_order, l=self.l, m=self.m)))
        waveform_imag    = np.genfromtxt(os.path.join(sim_path, '{pert}_h_{l}{m}_im.dat'.format(pert=self.pert_order, l=self.l, m=self.m)))
        
        # Need to excise the first time point, since the difference between two resolutions is zero, giving nans in the likelihood.
        return time[1:], waveform_real[1:], waveform_imag[1:]

    def generate_constant_error(self, error_value):

        """

        Generate a constant error for the NR waveform.

        Parameters
        ----------

        error_value
            Value of the error.

        Returns
        -------

        complex_error
            Complex error array.

        """
    
        complex_error = np.ones(len(self.NR_r)) * error_value * (1.+1.*1j)

        return complex_error
    
    def generate_gaussian_error(self, sigma, size):

        """

        Generate a constant error for the NR waveform.

        Parameters
        ----------

        sigma
            Standard deviation of the Gaussian distribution from which we extract the error values.
        size
            Lenght of the array of error.

        Returns
        -------

        complex_error
            Complex error array.

        """
        real_part = np.random.normal(loc=0, scale=sigma, size=size)
        imag_part = np.random.normal(loc=0, scale=sigma, size=size)

        complex_error = real_part + 1j * imag_part

        return complex_error