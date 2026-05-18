# General python imports
import h5py, json, numpy as np, os, pandas as pd, subprocess, tempfile
from scipy import interpolate

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "bayring_numba_cache"))
import sxs
try   : from cbhdb import simulation
except: pass

import bayRing.QNM_utils      as QNM_utils
import bayRing.injection      as injection_utils
import bayRing.template_waveforms as template_waveforms
import bayRing.utils          as utils
import bayRing.waveform_utils as waveform_utils
import pyRing.utils           as pyRing_utils

twopi = 2.*np.pi


def _prime_sxs_simulations_cache():

    try:
        from sxscatalog.simulations.simulations import Simulations
        if not hasattr(Simulations, "_simulations"):
            Simulations.load(download=False)
    except Exception:
        pass


def _parse_sxs_reference_eccentricity(ecc):
    if ecc is None:
        return 0.0
    if isinstance(ecc, str):
        ecc_text = ecc.strip()
        if ecc_text.lower() in {"", "nan", "none"}:
            return 0.0
        if ecc_text[0] in "<>":
            ecc_text = ecc_text[1:]
        return float(ecc_text)
    ecc = float(ecc)
    if np.isnan(ecc):
        return 0.0
    return ecc


def _symmetric_mass_ratio_from_q(q):

    try:
        q = float(q)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(q) or q <= 0.0:
        return None
    return q/(1.0 + q)**2


def _nearest_time_index(times, target_time):

    times = np.asarray(times, dtype=float)
    if len(times) == 0:
        raise ValueError("Cannot locate a merger sample on an empty time array.")
    return int(np.argmin(np.abs(times - float(target_time))))


def _gradient_on_grid(values, times):

    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if len(values) < 2:
        return np.zeros_like(values)
    edge_order = 2 if len(values) > 2 else 1
    return np.gradient(values, times, edge_order=edge_order)


def _compute_mode_merger_metadata(t_NR, NR_amp, NR_phi, ecc, mode, reference_peak_time=None, nu=None):

    mode_label = '{}{}'.format(int(mode[0]), int(mode[1]))
    t_NR       = np.asarray(t_NR, dtype=float)
    NR_amp     = np.asarray(NR_amp, dtype=float)
    NR_phi     = np.asarray(NR_phi, dtype=float)

    mode_peak_time = waveform_utils.find_peak_time(t_NR, NR_amp, ecc)
    peak_index     = _nearest_time_index(t_NR, mode_peak_time)
    omega          = _gradient_on_grid(NR_phi, t_NR)
    amp_dot        = _gradient_on_grid(NR_amp, t_NR)
    amp_dotdot     = _gradient_on_grid(amp_dot, t_NR)

    if reference_peak_time is None:
        reference_peak_time = mode_peak_time

    metadata = {
        'mode'                         : mode_label,
        't_peak_{}'.format(mode_label) : float(mode_peak_time),
        'A_peak_{}'.format(mode_label) : float(NR_amp[peak_index]),
        'omg_peak_{}'.format(mode_label): float(omega[peak_index]),
        'omega_peak_{}'.format(mode_label): float(omega[peak_index]),
        'A_peak{}dot'.format(mode_label): float(amp_dot[peak_index]),
        'A_peak{}dotdot'.format(mode_label): float(amp_dotdot[peak_index]),
        'DeltaT_{}'.format(mode_label) : float(mode_peak_time - reference_peak_time),
    }

    if nu is not None and np.isfinite(nu) and nu != 0.0:
        metadata['A_peak_over_nu_{}'.format(mode_label)] = metadata['A_peak_{}'.format(mode_label)]/nu
        metadata['A_peakdot_over_nu_{}'.format(mode_label)] = metadata['A_peak{}dot'.format(mode_label)]/nu
        metadata['A_peakdotdot_over_nu_{}'.format(mode_label)] = metadata['A_peak{}dotdot'.format(mode_label)]/nu

    return metadata


def _sxs_bitwise_axis(axis, ndim):

    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        raise np.exceptions.AxisError(axis, ndim=ndim)
    return axis


def _sxs_numpy_xor(x, reverse=False, preserve_dtype=False, axis=-1, **kwargs):

    x = np.asarray(x)
    itemsize = x.itemsize
    if itemsize not in [1, 2, 4, 8]:
        raise ValueError(f"Input array's byte size must be one of {{1, 2, 4, 8}}, not {itemsize}")

    dtype = np.dtype(f"u{itemsize}")
    u = x.view(dtype)
    axis = _sxs_bitwise_axis(axis, u.ndim)

    out = kwargs.pop("out", None)
    if kwargs:
        raise TypeError(f"Unexpected keyword argument(s): {', '.join(kwargs)}")

    if reverse:
        result = np.bitwise_xor.accumulate(u, axis=axis)
    else:
        moved = np.moveaxis(u, axis, 0)
        result_moved = np.empty_like(moved)
        result_moved[0] = moved[0]
        result_moved[1:] = np.bitwise_xor(moved[:-1], moved[1:])
        result = np.moveaxis(result_moved, 0, axis)

    if out is not None:
        out_view = np.asarray(out).view(dtype)
        np.copyto(out_view, result)
        result = out_view

    if preserve_dtype:
        return result.view(x.dtype)
    return result


def _sxs_numpy_diff(x, reverse=False, axis=-1, **kwargs):

    u = np.asarray(x)
    if issubclass(u.dtype.type, np.unsignedinteger):
        u = u.view(np.complex128)

    kwargs.pop("preserve_dtype", None)
    out = kwargs.pop("out", None)
    if kwargs:
        raise TypeError(f"Unexpected keyword argument(s): {', '.join(kwargs)}")

    axis = _sxs_bitwise_axis(axis, u.ndim)
    moved = np.moveaxis(u, axis, 0)
    result_moved = np.empty_like(moved)
    result_moved[0] = moved[0]
    if reverse:
        for i in range(1, moved.shape[0]):
            result_moved[i] = result_moved[i - 1] - moved[i]
    else:
        result_moved[1:] = moved[:-1] - moved[1:]

    result = np.moveaxis(result_moved, 0, axis)
    if out is not None:
        out_array = np.asarray(out)
        np.copyto(out_array, result)
        result = out_array

    return result


def _sxs_is_numba_bitwise_function(func, name):

    return (
        getattr(func, "__name__", None) == name
        and getattr(func, "__module__", None) == "sxs.utilities.bitwise"
    )


def _quaternionic_numpy_exp(q, qout):

    q = np.asarray(q)
    qout = np.asarray(qout)

    vnorm = np.linalg.norm(q[..., 1:4], axis=-1)
    exp_scalar = np.exp(q[..., 0])
    mask = vnorm > 10 * np.finfo(float).resolution

    qout[..., 0] = exp_scalar
    qout[..., 1:4] = 0.0
    qout[..., 0][mask] = exp_scalar[mask] * np.cos(vnorm[mask])

    scale = np.zeros_like(vnorm)
    scale[mask] = exp_scalar[mask] * np.sin(vnorm[mask]) / vnorm[mask]
    qout[..., 1:4] = scale[..., np.newaxis] * q[..., 1:4]


def _quaternionic_numpy_conj(q, qout):

    q = np.asarray(q)
    qout = np.asarray(qout)
    qout[..., 0] = q[..., 0]
    qout[..., 1:4] = -q[..., 1:4]


def _patch_sxs_numba_bitwise_decoder():

    if getattr(sxs, "_bayring_numpy_bitwise_decoder", False):
        return

    try:
        import sxs.utilities as sxs_utilities
        import sxs.utilities.bitwise as sxs_bitwise
        import sxs.waveforms.format_handlers.rotating_paired_diff_multishuffle_bzip2 as rpdmb
        import sxs.waveforms.format_handlers.rotating_paired_xor_multishuffle_bzip2 as rpxmb
        import quaternionic
    except Exception:
        return

    sxs_bitwise.diff = _sxs_numpy_diff
    sxs_bitwise.xor = _sxs_numpy_xor
    sxs_utilities.diff = _sxs_numpy_diff
    sxs_utilities.xor = _sxs_numpy_xor
    rpdmb.xor = _sxs_numpy_xor
    rpxmb.xor = _sxs_numpy_xor
    quaternionic.algebra_ufuncs.exp = _quaternionic_numpy_exp
    quaternionic.algebra_ufuncs.conj = _quaternionic_numpy_conj
    quaternionic.algebra_ufuncs.conjugate = _quaternionic_numpy_conj
    quaternionic.algebra_ufuncs.invert = _quaternionic_numpy_conj

    original_load = getattr(rpdmb, "_bayring_original_load", rpdmb.load)
    rpdmb._bayring_original_load = original_load

    def bayring_rpdmb_load(*args, **kwargs):

        diff = kwargs.get("diff")
        if diff is None or _sxs_is_numba_bitwise_function(diff, "diff"):
            kwargs["diff"] = _sxs_numpy_diff
        elif _sxs_is_numba_bitwise_function(diff, "xor"):
            kwargs["diff"] = _sxs_numpy_xor
        return original_load(*args, **kwargs)

    rpdmb.load = bayring_rpdmb_load
    sxs._bayring_numpy_bitwise_decoder = True


_patch_sxs_numba_bitwise_decoder()


def read_injection_modes(NR_catalog, injection_modes):

    if(injection_utils.is_injection_catalog(NR_catalog)):

        injection_modes_string   = injection_modes.replace(',', '_')

        injection_modes_list     = []
        injection_modes_list_tmp = injection_modes.split(',')
        for i in range(len(injection_modes_list_tmp)):
            l_injection, m_injection, n_injection = int(injection_modes_list_tmp[i][0]), int(injection_modes_list_tmp[i][1]), int(injection_modes_list_tmp[i][2])
            injection_modes_list.append((l_injection, m_injection, n_injection))

    else:
        injection_modes_string = ''
        injection_modes_list = None

    return injection_modes_string, injection_modes_list

def read_RWZ_env_simulation_parameters(sim_file):

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

    def __init__(self, NR_data_path='', csv_path='', fit_path='', ID='', ell=2, m=2, resolution_level=100):

        self.base             = NR_data_path
        self.metadata_path    = os.path.join(NR_data_path, 'Metadata')
        self.waveform_path    = os.path.join(NR_data_path, 'Data')
        self.psi4_path        = os.path.join(NR_data_path, 'Data/Psi4')
        self.csv_path         = csv_path 
        self.fit_path         = fit_path
        self.ID               = ID 
        self.ell              = ell
        self.m                = m
        self.resolution_level = resolution_level

        self.metadata       = {}

    def set_metadata_RIT(self, filename_to_set):

        full_name = os.path.join(self.metadata_path, filename_to_set)
        if not os.path.isfile(full_name):
            print(f"File not present locally. Attempting to download it from the online RIT catalog.")
            os.system(f'wget https://ccrgpages.rit.edu/~RITCatalog/Metadata/{filename_to_set} -P {self.metadata_path} --no-check-certificate')

        f = open(full_name, "r")

        return f

    def load_metadata(self):

        ID_str = str(self.ID)

        possible_name_formats_list = [
                                        os.path.join(self.base, f'RIT_eBBH_{ID_str}-n{self.resolution_level}-ecc_Metadata.txt'),
                                        os.path.join(self.base, f'RIT:eBBH:{ID_str}-n{self.resolution_level}-ecc_Metadata.txt'),
                                        os.path.join(self.base, f'RIT_BBH_{ID_str}-n{self.resolution_level}-id0_Metadata.txt'),
                                        os.path.join(self.base, f'RIT:BBH:{ID_str}-n{self.resolution_level}-id0_Metadata.txt'),
                                        os.path.join(self.base, f'RIT_BBH_{ID_str}-n{self.resolution_level}-id1_Metadata.txt'),
                                        os.path.join(self.base, f'RIT:BBH:{ID_str}-n{self.resolution_level}-id1_Metadata.txt'),
                                        os.path.join(self.base, f'RIT_BBH_{ID_str}-n{self.resolution_level}-id2_Metadata.txt'),
                                        os.path.join(self.base, f'RIT:BBH:{ID_str}-n{self.resolution_level}-id2_Metadata.txt'),
                                        os.path.join(self.base, f'RIT_BBH_{ID_str}-n{self.resolution_level}-id3_Metadata.txt'),
                                        os.path.join(self.base, f'RIT:BBH:{ID_str}-n{self.resolution_level}-id3_Metadata.txt')
                                     ]

        possible_name_formats_list = possible_name_formats_list
        for name_format in possible_name_formats_list:
            try:
                f  = self.set_metadata_RIT(name_format)
                break
            except:
                continue

        lines = [l for l in f.readlines() if l.strip()] # rm empty

        for line in lines[1:]:

            #line = line.split("#", 1)[0]
            if line[0]=="#": continue

            line               = line.rstrip("\n")
            key, val           = line.split("= ")
            key                = key.strip()
            self.metadata[key] = val

        try:
            additional_data = pd.read_csv(self.csv_path)     
            self.metadata[f'A_peak_{self.ell}{self.m}']      = additional_data.loc[additional_data['ID'] == int(self.ID), f'A_peak{self.ell}{self.m}'].values[0]
            self.metadata[f'omg_peak_{self.ell}{self.m}']    = additional_data.loc[additional_data['ID'] == int(self.ID), f'omega_peak{self.ell}{self.m}'].values[0]
            try:
                self.metadata['A_nr_error']                  = additional_data.loc[additional_data['ID'] == int(self.ID), f'A_nr_error'].values[0]
            except:
                print("No NR error found in the csv file. Setting it to 1e-3.")
                self.metadata['A_nr_error'] = 1e-3
            self.metadata[f'A_peak{self.ell}{self.m}dotdot'] = additional_data.loc[additional_data['ID'] == int(self.ID), f'A_peak{self.ell}{self.m}dotdot'].values[0]
            self.metadata['Emrg']                            = additional_data.loc[additional_data['ID'] == int(self.ID), f'Heff_til'].values[0]
            self.metadata['Jmrg']                            = additional_data.loc[additional_data['ID'] == int(self.ID), f'Jmrg_til'].values[0]
            self.metadata['bmrg']                            = additional_data.loc[additional_data['ID'] == int(self.ID), f'b_massless_EOB'].values[0]
        except:
            self.metadata[  f'A_peak_{self.ell}{self.m}']   = self.metadata[f'peak-ampl-l{self.ell}-m{self.m}']
            self.metadata[f'omg_peak_{self.ell}{self.m}']   = self.metadata[f'peak-omega-l{self.ell}-m{self.m}']
            self.metadata['A_nr_error']                     = 1e-3
            self.metadata['A_peak{self.ell}{self.m}dotdot'] = 0.0
            self.metadata[                       f'Emrg']   = 0.0
            self.metadata[                       f'Jmrg']   = 0.0
            self.metadata[                       f'bmrg']   = 0.0

        return self.metadata

    def set_h_data_RIT(self, filename_strain):

        full_name = os.path.join(self.waveform_path, filename_strain)
        if not os.path.isfile(full_name):
            print(f"File not present locally. Attempting to download it from the online RIT catalog.")
            os.system(f'wget https://ccrgpages.rit.edu/~RITCatalog/Data/{filename_strain} -P {self.waveform_path} --no-check-certificate')

        f = h5py.File(full_name, "r")

        return f

    def load_waveform_lm(self):

        ID_str = str(self.ID)

        possible_name_formats_list = [f'ExtrapStrain_RIT-eBBH-{ID_str}-n{self.resolution_level}.h5',
                                      f'ExtrapStrain_RIT-BBH-{ID_str}-n{self.resolution_level}.h5' ,]            
        for filename_strain in possible_name_formats_list:
            try:
                f = self.set_h_data_RIT(filename_strain)
                break
            except:
                continue

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
 
    def set_psi4_data_RIT(self, dir_name):

        asc_name    = os.path.join(dir_name, f'rPsi4_l{self.ell}_m{self.m}_rInf.asc')
        tar_gz_name = f'{dir_name}.tar.gz'
        tar_gz_path = os.path.join(self.psi4_path, tar_gz_name)

        if not os.path.isfile(tar_gz_path):
            print(f"File not present locally. Attempting to download it from the online RIT catalog.")
            os.system(f'wget https://ccrgpages.rit.edu/~RITCatalog/Data/{tar_gz_name} -P {self.psi4_path} --no-check-certificate')

        f = utils.read_psi4_RIT_format(tar_gz_path, asc_name)

        return f

    def load_psi4_lm(self):

        ID_str    = str(self.ID)

        possible_name_formats_list = [f'ExtrapPsi4_RIT-eBBH-{ID_str}-n{self.resolution_level}-ecc'                          ]    
        idx_names                  = [f'ExtrapPsi4_RIT-BBH-{ID_str}-n{self.resolution_level}-id{idx}' for idx in range(0, 4)]
        
        possible_name_formats_list = possible_name_formats_list + idx_names
        for dir_name in possible_name_formats_list:
            try:
                f = self.set_psi4_data_RIT(dir_name)
                break
            except:
                continue

        self.u  = f['time']
        self.A  = f['ampl']
        self.p  = f['phse']
        self.re = f['real']
        self.im = f['imag']

        return self.u, self.re, self.im, self.A, self.p

    def interp_qnt(self, x, y, x_new):

        f  = interpolate.interp1d(x, y)
        yn = f(x_new)

        return yn

    def interpolate_waveform_lm(self, u_new):

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

    def load_waveform_lm(self):

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
        Catalog of the NR simulation. Available options: ['SXS', 'cbhdb',
        'charged_raw', 'RIT', 'Teukolsky', 'injections']

    Returns
    -------

    metadata : dict
        Dictionary containing the metadata of the NR simulation.

    """
    if(NR_catalog=='SXS'):
        try:
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
                        'A_peak_22'     : NR_sim.A_peak_22,
                        'omg_peak_22'   : NR_sim.omg_peak_22,
                        'A_nr_error'    : NR_sim.A_nr_error,
                        'A_peak22dotdot': NR_sim.A_peak22dotdot,
                        'bmrg'          : NR_sim.bmrg,
                        'Emrg'          : NR_sim.Emrg,
                        'Jmrg'          : NR_sim.Jmrg,
                    }
            metadata.update(getattr(NR_sim, 'additional_metadata', {}))
        except:
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
                    'q'             : NR_sim.q,
                    'chi1'          : NR_sim.chi1,
                    'chi2'          : NR_sim.chi2,
                    'm1'            : pyRing_utils.m1_from_m_q(M, NR_sim.q),
                    'm2'            : pyRing_utils.m2_from_m_q(M, NR_sim.q),
                    'ecc'           : NR_sim.ecc,
                    'Mf'            : NR_sim.Mf,
                    'af'            : NR_sim.af,
                    'A_peak_22'     : NR_sim.A_peak_22,
                    'omg_peak_22'   : NR_sim.omg_peak_22,
                    'A_nr_error'    : NR_sim.A_nr_error,
                    'A_peak22dotdot': NR_sim.A_peak22dotdot,
                    'bmrg'          : NR_sim.bmrg,
                    'Emrg'          : NR_sim.Emrg,
                    'Jmrg'          : NR_sim.Jmrg,
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

    elif(NR_catalog=='RWZ-env'):
        metadata = {
                    'a_halo'    : NR_sim.a_halo,
                    'M_halo'    : NR_sim.M_halo,
                    'C'         : NR_sim.C,
                    'Mf'        : NR_sim.Mf,
                    'af'        : NR_sim.af,
	    }

    elif(injection_utils.is_injection_catalog(NR_catalog)):
        metadata = injection_utils.metadata_from_simulation(NR_sim)

    else: raise ValueError("Invalid option for NR catalog: {}".format(NR_catalog))

    metadata.update(getattr(NR_sim, 'additional_metadata', {}) or {})

    return metadata
    
class NR_simulation():

    """

    Class for the NR simulation object.

    Parameters
    ----------

    NR_catalog : str
        Catalog of the NR simulation. Available options: ['SXS', 'cbhdb', 'charged_raw', 'RIT', 'Teukolsky'].

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
        Error of the NR simulation. Available options are catalogue-dependent.
        For SXS: 'constant-X', 'align-with-mismatch-all',
        'align-with-mismatch-res-only', 'align-at-peak', and
        'late-time-const-error'. For Teukolsky: 'constant-X' and
        'resolution'. For RIT: 'constant-X' and 'late-time-const-error'.
        For injections: 'gaussian-X' and 'from-SXS-NR'. Default:
        'align-with-mismatch-all'.

    tM_start : float, optional
        Initial time of the fit. Default: 30.0.

    tM_end : float, optional
        Final time of the fit. Default: 150.0.

    t_delay_scd : float, optional
        Time delay between the NR simulation and the SCD simulation. Default: 0.0.

    t_min_mismatch : float, optional
        Lower mismatch-window input. When both mismatch-window inputs are in
        [0, 1], they use the legacy fractional pre-peak convention; otherwise
        values are offsets from the peak time. Default: 0.0.

    t_max_mismatch : float, optional
        Upper mismatch-window input. When both mismatch-window inputs are in
        [0, 1], they use the legacy fractional pre-peak convention; otherwise
        values are offsets from the peak time. Default: 30.0.
        
    """

    def __init__(self                                           , 
                 NR_catalog                                     , 
                 NR_ID                                          , 
                 res_level                                      , 
                 extrap_order                                   , 
                 perturbation_order                             , 
                 NR_dir                                         , 
                 additional_NR_properties                       ,
                 fits                                           , 
                 injection_modes_list                           , 
                 injection_times                                , 
                 injection_noise                                , 
                 injection_tail                                 , 
                 injection_parameters                           ,
                 l                                              , 
                 m                                              , 
                 outdir                                         , 
                 injection_model_parameters = None              ,
                 waveform_type  = 'strain'                      ,
                 download       = False                         , 
                 NR_error       = 'align-with-mismatch-all'     , 
                 tM_start       = 30.0                          , 
                 tM_end         = 150.0                         , 
                 t_delay_scd    = 0.0                           , 
                 t_peak_22      = 0.0                           ,
                 t_min_mismatch = 0.0                           ,
                 t_max_mismatch = 30.0                          ):

        ####################
        # Input parameters #
        ####################

        self.NR_catalog               = NR_catalog
        self.NR_ID                    = NR_ID
        self.res_level                = res_level
        self.extrap_order             = extrap_order
        self.waveform_type            = waveform_type

        self.l                        = l
        self.m                        = m
        self.pert_order               = perturbation_order

        self.NR_dir                   = NR_dir
        self.additional_NR_properties = utils.normalize_optional_path(additional_NR_properties)
        self.fits                     = utils.normalize_optional_path(fits)
        self.outdir                   = outdir

        self.injection_modes          = injection_modes_list
        self.injection_noise          = injection_noise
        self.injection_tail           = injection_tail
        self.injection_parameters     = injection_parameters
        self.injection_model_parameters = dict(injection_model_parameters or {})
        self.injection_model_parameters.setdefault('template', 'Kerr')
        self.injection_model_parameters.setdefault('N-DS-modes', 1)
        self.injection_model_parameters.setdefault('N-DS-tails', 0)
        self.injection_model_parameters.setdefault('QNM-modes', '220,221,320')
        self.injection_model_parameters.setdefault('QQNM-modes', '')
        self.injection_model_parameters.setdefault('Kerr-tail-modes', '22')
        self.injection_model_parameters.setdefault('KerrBinary-version', 'London2018')
        self.injection_model_parameters.setdefault('KerrBinary-final-state-nc-version', '')
        self.injection_model_parameters.setdefault('KerrBinary-amplitudes-nc-version', '')
        self.injection_model_parameters.setdefault('TEOB-template', 'HypTan')
        self.injection_model_parameters.setdefault('TEOB-calibration', 'qc')
        self.injection_model_parameters.setdefault('TEOB-global-fit', 1)
        self.injection_model_parameters.setdefault('TEOB-merger-data', 0)
        self.injection_model_parameters.setdefault('TEOB-mode-mixing', 0)
        self.injection_model_parameters.setdefault('TEOB-quadratic-44', 0)
        self.injection_model_parameters.setdefault('TEOB-quadratic-44-window-start', 10.0)
        self.injection_model_parameters.setdefault('TEOB-quadratic-44-window-width', 15.0)
        self.injection_model_parameters.setdefault('TEOB-quadratic-44-window-end', -1.0)
        self.injection_model_parameters.setdefault('TEOB-quadratic-44-window-steepness', 1.0)
        self.injection_model_parameters.setdefault('TEOB-quadratic-44-ratio-fit', 'khera-total')
        self.injection_model_parameters.setdefault('TEOB-tapered-overtone-44', 0)
        self.injection_model_parameters.setdefault('TEOB-tapered-overtone-44-window-start', 0.0)
        self.injection_model_parameters.setdefault('TEOB-tapered-overtone-44-window-width', 10.0)
        self.injection_model_parameters.setdefault('charge', 0)
        self.injection_truths         = None
        self.injection_metadata       = {}

        self.tM_start                 = tM_start
        self.tM_end                   = tM_end
        self.t_delay_scd              = t_delay_scd
        self.t_peak_22                = t_peak_22


        ######################
        # Read-in simulation #
        ######################
        
        if(injection_utils.is_injection_catalog(self.NR_catalog)):

            raw_injection_parameters = self.injection_parameters
            if raw_injection_parameters is None:
                self.read_injection_metadata()
                raw_injection_parameters = dict(self.injection_metadata_parameters)

            injection_times_config, self.injection_metadata, waveform_parameters = injection_utils.prepare_injection_parameters(
                raw_injection_parameters,
                self.injection_model_parameters,
            )
            for key, value in self.injection_metadata.items():
                setattr(self, key, value)

            if(injection_times=='from-metadata'):

                self.t_start = injection_times_config['t_start']
                self.t_NR    = np.arange(self.t_start, injection_times_config['t_end'], injection_times_config['dt'])
                if(self.t_NR[0] < 0):
                    self.t_NR = self.t_NR - self.t_NR[0]

            elif(injection_times=='from-SXS-NR'):

                self.download      = download
                self.injection_error_source = NR_error

                self.t_NR, self.NR_err_cmplx_SXS, self.t_start = self.extract_data_NR(t_min_mismatch, t_max_mismatch)

            else:

                raise ValueError("Unknown times option.")

            injection_peak = injection_times_config.get('t_peak', self.t_start)
            Kerr_modes, Kerr_quad_modes, qnm_cached = self._injection_Kerr_setup(self.injection_metadata)
            Kerr_tail_modes = QNM_utils.read_tail_modes(self.injection_model_parameters['Kerr-tail-modes'])
            fit_metadata = self._read_injection_fit_metadata()
            injection_template = self.injection_model_parameters['template']
            injection_tail = 0 if self.injection_tail is None else int(float(self.injection_tail))

            injection_model = template_waveforms.WaveformModel(
                self.t_NR,
                self.t_start,
                injection_peak,
                injection_template,
                self.injection_model_parameters['N-DS-modes'],
                Kerr_modes,
                self.injection_metadata,
                fit_metadata,
                qnm_cached,
                self.l,
                self.m,
                N_ds_tails                = self.injection_model_parameters['N-DS-tails'],
                tail                      = injection_tail,
                tail_modes                = Kerr_tail_modes,
                quadratic_modes           = Kerr_quad_modes,
                const_params              = None,
                KerrBinary_version        = self.injection_model_parameters['KerrBinary-version'],
                KerrBinary_amp_nc_version = self.injection_model_parameters['KerrBinary-amplitudes-nc-version'],
                TEOB_template             = self.injection_model_parameters['TEOB-template'],
                TEOB_calibration          = self.injection_model_parameters['TEOB-calibration'],
                TEOB_global_fit           = self.injection_model_parameters['TEOB-global-fit'],
                TEOB_merger_data          = self.injection_model_parameters['TEOB-merger-data'],
                TEOB_mode_mixing          = self.injection_model_parameters['TEOB-mode-mixing'],
                TEOB_quadratic_44         = self.injection_model_parameters['TEOB-quadratic-44'],
                TEOB_quadratic_44_window_start = self.injection_model_parameters['TEOB-quadratic-44-window-start'],
                TEOB_quadratic_44_window_width = self.injection_model_parameters['TEOB-quadratic-44-window-width'],
                TEOB_quadratic_44_window_end = self.injection_model_parameters['TEOB-quadratic-44-window-end'],
                TEOB_quadratic_44_window_steepness = self.injection_model_parameters['TEOB-quadratic-44-window-steepness'],
                TEOB_quadratic_44_ratio_fit = self.injection_model_parameters['TEOB-quadratic-44-ratio-fit'],
                TEOB_tapered_overtone_44  = self.injection_model_parameters['TEOB-tapered-overtone-44'],
                TEOB_tapered_overtone_44_window_start = self.injection_model_parameters['TEOB-tapered-overtone-44-window-start'],
                TEOB_tapered_overtone_44_window_width = self.injection_model_parameters['TEOB-tapered-overtone-44-window-width'],
            )

            try:
                injected_waveform = injection_model.waveform(waveform_parameters, {})
            except KeyError as exc:
                raise ValueError("Missing injection parameter `{}` for template `{}`.".format(exc.args[0], injection_template)) from exc

            self.NR_r = np.real(injected_waveform)
            self.NR_i = np.imag(injected_waveform)
            self.injection_truths = dict(waveform_parameters)

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
                    NR_h2                     = self.waveform_obj2.processed.rwaveform_lm_finite_radius[(self.l, self.m)]
                    t_res, NR_r_res, NR_i_res = self.waveform_obj2.processed.rwaveform_lm_finite_radius_times, NR_h2.real, NR_h2.imag
                except(ValueError):
                    print("Lower resolution not found!")
                    raise

            # Built NR waveform and time axis
            NR_h      = self.waveform_obj.processed.rwaveform_lm_finite_radius[(self.l, self.m)]
            self.NR_r = NR_h.real
            self.NR_i = NR_h.imag
            self.t_NR = self.waveform_obj.processed.rwaveform_lm_finite_radius_times

            # Define metadata in the class
            self.q, self.q1, self.q2, self.chi1, self.chi2, self.tilt1, self.tilt2, self.ecc, self.Mf, self.qf, self.af = self.read_cbhdb_metadata()

        elif(self.NR_catalog=='SXS'):
        
            self.download  = download
            self.q, self.chi1, self.chi2, self.tilt1, self.tilt2, self.ecc, self.Mf, self.af = self.read_SXS_metadata()
            self.additional_metadata = {}
            
            if self.additional_NR_properties:
                self.A_peak_22, self.omg_peak_22, self.A_nr_error, self.A_peak22dotdot, self.bmrg, self.Emrg, self.Jmrg = self.load_SXS_addn_metadata(csv_path=self.additional_NR_properties, ID_str=self.NR_ID)
            else:
                self.A_peak_22, self.omg_peak_22, self.A_nr_error, self.A_peak22dotdot = None, None, None, None
                self.bmrg, self.Emrg, self.Jmrg = None, None, None

            # Build NR waveform and time axis.
            if self.res_level == -1:
                # We add a maximum number of attempts to avoid the infinite loop

                # Max attempts corresponding to the 6 resolution levels
                max_attempts = 6

                # Counter for the number of attempts
                attempts = 0

                # Loop through each resolution level from 6 down to 1
                for res_level_x in [6, 5, 4, 3, 2, 1]:
                    try:

                        # Attempt to read the waveform for the current resolution level
                        self.t_NR, self.NR_r, self.NR_i = self.read_waveform_lm_from_SXS(self.extrap_order, res_level_x)
                        
                        # Set the resolution level if successful
                        self.res_level = res_level_x
                        print("\n* Resolution found at level: {}".format(self.res_level))

                        # Exit the loop if the level is valid and waveform is loaded
                        break
                    except Exception as e:

                        # If an error occurs (e.g., file not found or data issues), increment the attempt count
                        attempts += 1
                        print(f"\n*Error in attempt {attempts} with resolution level {res_level_x}: {e}")
                        
                        # If we reach the maximum number of attempts, break the loop and stop trying
                        if attempts >= max_attempts:
                            print("\n* Unable to find a valid resolution level. Stopping attempts.")
                            break
            else:
                # If a valid resolution level is already set, load the waveform with that resolution level
                self.t_NR, self.NR_r, self.NR_i = self.read_waveform_lm_from_SXS(self.extrap_order, self.res_level)

            t_res, NR_r_res, NR_i_res       = self.t_NR, self.NR_r, self.NR_i
            t_extr, NR_r_extr, NR_i_extr    = None, None, None
            sxs_comparison_error_options    = ['align-with-mismatch-res-only', 'align-with-mismatch-all', 'align-at-peak']
            if(NR_error in sxs_comparison_error_options):
                for lower_res_level in range(self.res_level - 1, 0, -1):
                    try:
                        t_res, NR_r_res, NR_i_res = self.read_waveform_lm_from_SXS(self.extrap_order, lower_res_level)
                        print('* Resolution error constructed with resolution level {}'.format(lower_res_level))
                        break
                    except ValueError:
                        pass
                else:
                    print('* No lower SXS resolution available; setting the resolution error to zero.')
                t_extr, NR_r_extr, NR_i_extr = self.read_waveform_lm_from_SXS(self.extrap_order+1, self.res_level)

        elif(self.NR_catalog=='RIT'):
        
            self.q, self.chi1, self.chi2, self.ecc, self.Mf, self.af, self.A_peak_22, self.omg_peak_22, self.A_nr_error, self.A_peak22dotdot, self.bmrg, self.Emrg, self.Jmrg = self.read_RIT_metadata()

            # Build NR waveform and time axis.
            self.t_NR, self.NR_r, self.NR_i = self.read_waveform_lm_from_RIT()
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
            self.t_NR, self.NR_r, self.NR_i = self.read_waveform_lm_from_C2EFT(resolution = res_1, sigma = sigma_1, tau = tau_1)
            t_res,     NR_r_res,  NR_i_res  = self.read_waveform_lm_from_C2EFT(resolution = res_2, sigma = sigma_2, tau = tau_1)
            t_extr,    NR_r_extr, NR_i_extr = self.read_waveform_lm_from_C2EFT(resolution = res_1, sigma = sigma_2, tau = tau_1)

        elif(self.NR_catalog=='Teukolsky'):
        
            self.Mf, self.af                = self.read_Teukolsky_metadata()
            self.t_NR, self.NR_r, self.NR_i = self.read_waveform_lm_from_Teukolsky(self.res_level)
            if not isinstance(self.res_level, str):
                try: t_res, NR_r_res,  NR_i_res = self.read_waveform_lm_from_Teukolsky(self.res_level-1)
                except: print('\n* Teukolsky resolution level {} not available.\n'.format(self.res_level-1))
            else:
                if(NR_error=='resolution'): raise ValueError("Resolution error not yet available when using nx,nl as resolution indicators.")
            t_extr, NR_r_extr, NR_i_extr   = None, None, None
            self.ecc = 0.0
        
        elif(self.NR_catalog=='RWZ-env'):

            # Read the metadata
            self.a_halo, self.M_halo, self.C = self.read_RWZ_env_metadata()
            self.ecc, self.Mf, self.af = 0.0, 1.0, 0.0

            # Build NR waveform and time axis
            if(self.res_level==-1):
                try:
                    self.res_level = self.highest_available_RWZ_resolution_level(self.extrap_order)
                    print("\n* Resolution found at level: {}".format(self.res_level))
                except FileNotFoundError:
                    if NR_error in ['resolution', 'align-at-peak']:
                        raise
                    self.res_level = None

            self.t_NR, self.NR_r, self.NR_i = self.read_waveform_lm_from_RWZ(self.res_level, self.extrap_order)
            if NR_error in ['resolution', 'align-at-peak']:
                res_level_error              = self.lower_available_RWZ_resolution_level(self.res_level, self.extrap_order)
                t_res,  NR_r_res,  NR_i_res  = self.read_waveform_lm_from_RWZ(res_level_error, self.extrap_order, allow_simple_fallback=False)
                if(NR_error=='align-at-peak'):
                    t_extr, NR_r_extr, NR_i_extr = self.read_waveform_lm_from_RWZ(self.res_level, self.extrap_order+1, allow_simple_fallback=False)
                else:
                    t_extr, NR_r_extr, NR_i_extr = None, None, None
            else:
                t_res,  NR_r_res,  NR_i_res  = None, None, None
                t_extr, NR_r_extr, NR_i_extr = None, None, None

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

            elif(NR_error == 'late-time-const-error'):
                error_value                = self.A_nr_error
                self.NR_err_cmplx          = self.generate_constant_error(error_value)

            else:

                # Align the waveforms minimizing the mismatch over a [t_min, t_max] interval.
                if(NR_error in ['align-with-mismatch-res-only', 'align-with-mismatch-all']):
                    
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
            
            if('constant' in NR_error):
                error_value                = float(NR_error.split('-')[-1])
                self.NR_err_cmplx          = self.generate_constant_error(error_value)

            elif(NR_error == 'late-time-const-error'):
                error_value                = self.A_nr_error
                self.NR_err_cmplx          = self.generate_constant_error(error_value)

        elif(self.NR_catalog=='C2EFT'):

            if('constant' in NR_error):
                error_value                = float(NR_error.split('-')[-1])
                self.NR_err_cmplx          = self.generate_constant_error(error_value)

            else:

                # Align the waveforms minimizing the mismatch over a [t_min, t_max] interval.
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

        elif(self.NR_catalog=='RWZ-env'):

            # Waveforms at different resolution levels are already aligned.
            if(NR_error=='resolution'):
                if len(self.NR_r) != len(NR_r_res):
                    if len(NR_r_res) < len(self.NR_r):
                        NR_r_res = np.append(NR_r_res, np.zeros(len(self.NR_r) - len(NR_r_res)))
                        NR_i_res = np.append(NR_i_res, np.zeros(len(self.NR_i) - len(NR_i_res)))
                    else:
                        NR_r_res = NR_r_res[:len(self.NR_r)]
                        NR_i_res = NR_i_res[:len(self.NR_r)]
                NR_r_err_res, NR_i_err_res = np.abs(self.NR_r-NR_r_res), np.abs(self.NR_i-NR_i_res)
                self.NR_err_cmplx          = NR_r_err_res + 1j * NR_i_err_res

            elif(NR_error=='align-at-peak'):
                # Resolution error.
                NR_r_res    , NR_i_res       = waveform_utils.align_waveforms_at_peak(self.t_NR, self.NR_amp, self.NR_phi, t_res, NR_r_res, NR_i_res)
                NR_r_err_res, NR_i_err_res   = np.abs(self.NR_r-NR_r_res), np.abs(self.NR_i-NR_i_res)

                # Extrapolation error.
                NR_r_extr    , NR_i_extr     = waveform_utils.align_waveforms_at_peak(self.t_NR, self.NR_amp, self.NR_phi, t_extr, NR_r_extr, NR_i_extr)
                NR_r_err_extr, NR_i_err_extr = np.abs(self.NR_r-NR_r_extr), np.abs(self.NR_i-NR_i_extr)
                # Global error
                self.NR_err_cmplx = np.sqrt(NR_r_err_extr**2 + NR_r_err_res**2) + 1j * np.sqrt(NR_i_err_extr**2 + NR_i_err_res**2)

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
        
            # Align the waveforms minimizing the mismatch over a [t_min, t_max] interval.
            if('align-with-mismatch' in NR_error):
                
                # Resolution error.
                NR_r_res    , NR_i_res     = waveform_utils.align_waveforms_with_mismatch(self.t_NR, self.NR_amp, self.NR_phi,  t_res,  NR_r_res,  NR_i_res, t_min_mismatch, t_max_mismatch)
                NR_r_err_res, NR_i_err_res = np.abs(self.NR_r-NR_r_res), np.abs(self.NR_i-NR_i_res)
            
                self.NR_err_cmplx          = NR_r_err_res + 1j * NR_i_err_res

        elif(self.NR_catalog=='charged_raw'):
            if('constant' in NR_error):
                error_value                = float(NR_error.split('-')[-1])
                self.NR_err_cmplx          = self.generate_constant_error(error_value)
       
        elif(injection_utils.is_injection_catalog(self.NR_catalog)):
            
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
        self.time_shift = 0.0
        if(self.t_NR[0] < 0 and self.waveform_type=='strain'):
            self.time_shift = float(self.t_NR[0])
            self.t_NR = self.t_NR - self.time_shift
        
        # Locate the merger time (which does not coincide with the peak in the eccentric case).
        self.mode_peak_time = waveform_utils.find_peak_time(self.t_NR, self.NR_amp, self.ecc)
        self.t_peak = self.mode_peak_time

        # For convenience, for second order perturbations, allow the option to build the time axis from the secondary peak.
        if(self.NR_catalog=='Teukolsky' and self.pert_order=='lin'): 
            print("\n* The peak time has been set to the secondary peak time with a delay of: {}.".format(self.t_delay_scd))
            self.t_peak = self.t_peak + self.t_delay_scd

        if not(self.t_peak_22==0.0):
            print("\n* The peak time has been set to the peak of the 22 mode: {}.".format(self.t_peak_22))
            self.t_peak = self.t_peak_22

        self.NR_freq  = np.gradient(self.NR_phi, self.t_NR)/(twopi)
        self._store_computed_merger_metadata()
        
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

    def _read_waveform_lm_for_mode(self, mode):

        original_l, original_m = self.l, self.m
        try:
            self.l, self.m = int(mode[0]), int(mode[1])
            if(self.NR_catalog=='SXS'):
                t_NR, NR_r, NR_i = self.read_waveform_lm_from_SXS(self.extrap_order, self.res_level)
            elif(self.NR_catalog=='RIT'):
                t_NR, NR_r, NR_i = self.read_waveform_lm_from_RIT()
            else:
                raise ValueError("Merger metadata for related modes is implemented for SXS and RIT waveforms.")
        finally:
            self.l, self.m = original_l, original_m

        t_NR = np.asarray(t_NR, dtype=float)
        if(getattr(self, 'time_shift', 0.0) != 0.0 and self.waveform_type=='strain'):
            t_NR = t_NR - self.time_shift
        return t_NR, np.asarray(NR_r, dtype=float), np.asarray(NR_i, dtype=float)

    def _related_merger_modes(self):

        mode = (int(self.l), int(self.m))
        related_modes = []
        if mode in template_waveforms.TEOB_MODE_MIXING_PARENTS:
            related_modes.append(template_waveforms.TEOB_MODE_MIXING_PARENTS[mode])
        return related_modes

    def _set_mode_merger_metadata(self, merger_metadata):

        if not hasattr(self, 'additional_metadata') or self.additional_metadata is None:
            self.additional_metadata = {}
        self.additional_metadata.update(merger_metadata)

        mode_label = merger_metadata['mode']
        for key in [
            't_peak_{}',
            'A_peak_{}',
            'omg_peak_{}',
            'omega_peak_{}',
            'A_peak{}dot',
            'A_peak{}dotdot',
            'DeltaT_{}',
        ]:
            metadata_key = key.format(mode_label)
            if metadata_key in merger_metadata:
                setattr(self, metadata_key, merger_metadata[metadata_key])

        if mode_label == '22':
            self.A_peak_22      = merger_metadata['A_peak_22']
            self.omg_peak_22    = merger_metadata['omg_peak_22']
            self.A_peak22dotdot = merger_metadata['A_peak22dotdot']

    def _write_computed_merger_metadata(self):

        if not hasattr(self, 'additional_metadata') or self.additional_metadata is None:
            return
        mode_entries = {
            key: value
            for key, value in self.additional_metadata.items()
            if (
                key.startswith(('t_peak_', 'A_peak_', 'omg_peak_', 'omega_peak_', 'DeltaT_'))
                or (key.startswith('A_peak') and (key.endswith('dot') or key.endswith('dotdot')))
                or key.startswith(('A_peak_over_nu_', 'A_peakdot_over_nu_', 'A_peakdotdot_over_nu_'))
            )
        }
        if not mode_entries:
            return
        serialisable_entries = {}
        for key, value in mode_entries.items():
            if isinstance(value, (np.floating, np.integer)):
                serialisable_entries[key] = float(value)
            else:
                serialisable_entries[key] = value
        outdir = os.path.join(self.outdir, 'Peak_quantities')
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, 'Merger_metadata.json'), 'w', encoding='utf-8') as handle:
            json.dump(serialisable_entries, handle, indent=2, sort_keys=True)

    def _store_computed_merger_metadata(self):

        nu = _symmetric_mass_ratio_from_q(getattr(self, 'q', None))
        selected_mode = (int(self.l), int(self.m))
        selected_metadata = _compute_mode_merger_metadata(
            self.t_NR,
            self.NR_amp,
            self.NR_phi,
            self.ecc,
            selected_mode,
            reference_peak_time=self.t_peak,
            nu=nu,
        )
        self._set_mode_merger_metadata(selected_metadata)

        for mode in self._related_merger_modes():
            if mode == selected_mode:
                continue
            try:
                t_NR, NR_r, NR_i = self._read_waveform_lm_for_mode(mode)
                NR_amp, NR_phi = waveform_utils.amp_phase_from_re_im(NR_r, NR_i)
                merger_metadata = _compute_mode_merger_metadata(
                    t_NR,
                    NR_amp,
                    NR_phi,
                    self.ecc,
                    mode,
                    reference_peak_time=self.t_peak,
                    nu=nu,
                )
                self._set_mode_merger_metadata(merger_metadata)
            except Exception as exc:
                print("* Could not compute TEOBPM merger metadata for related mode {}{}: {}".format(mode[0], mode[1], exc))

        self._write_computed_merger_metadata()

    def _injection_Kerr_setup(self, metadata):

        injection_template = self.injection_model_parameters['template']
        if injection_template == 'Damped-sinusoids':
            return [], None, {}

        cache_negative_m_qnms = (
            injection_template == 'KerrBinary'
            and self.injection_model_parameters['KerrBinary-version'] == 'Cheung2023'
            and metadata['af'] < 0.0
        )

        return QNM_utils.read_Kerr_modes(
            self.injection_model_parameters['QNM-modes'],
            self.injection_model_parameters['QQNM-modes'],
            self.injection_model_parameters['charge'],
            self.l,
            self.m,
            metadata,
            cache_negative_m_qnms=cache_negative_m_qnms,
        )

    def _read_injection_fit_metadata(self):

        if not(self.fits):
            return None

        fit_data = pd.read_csv(self.fits)

        return fit_data.iloc[0].to_dict()
       
    def extract_data_NR(self, t_min_mismatch, t_max_mismatch):

        # Build NR time axis.
        if(self.res_level==-1):
            for res_level_x in [6,5,4,3,2,1]:
                try: 
                    t_NR, NR_r, NR_i = self.read_waveform_lm_from_SXS(self.extrap_order, res_level_x)
                    self.res_level = res_level_x
                    break
                except(ValueError):
                    pass
        else:
            t_NR, NR_r, NR_i = self.read_waveform_lm_from_SXS(self.extrap_order, self.res_level)

        NR_amp, NR_phi               = waveform_utils.amp_phase_from_re_im(NR_r, NR_i)

        # Build NR error array.
        if(self.injection_error_source=='from-SXS-NR'):
            t_res,  NR_r_res,  NR_i_res  = self.read_waveform_lm_from_SXS(self.extrap_order,   self.res_level-1)
            t_extr, NR_r_extr, NR_i_extr = self.read_waveform_lm_from_SXS(self.extrap_order+1, self.res_level)

            NR_r_res    , NR_i_res       = waveform_utils.align_waveforms_with_mismatch(t_NR, NR_amp, NR_phi,  t_res,  NR_r_res,  NR_i_res, t_min_mismatch, t_max_mismatch)
            NR_r_err_res, NR_i_err_res   = np.abs(NR_r-NR_r_res), np.abs(NR_i-NR_i_res)

            NR_r_err_extr, NR_i_err_extr = np.abs(NR_r-NR_r_extr), np.abs(NR_i-NR_i_extr)

            NR_err_cmplx = np.sqrt(NR_r_err_extr**2 + NR_r_err_res**2) + 1j * np.sqrt(NR_i_err_extr**2 + NR_i_err_res**2)
        else:
            NR_err_cmplx = None

        # Construct positive t_NR and compute t_peak
        if(t_NR[0] < 0):
            t_NR = t_NR - t_NR[0]
        t_peak   = t_NR[np.argmax(NR_amp)]

        return t_NR, NR_err_cmplx, t_peak
        
    def read_injection_metadata(self):
        
        """
        
        Read metadata used to create injection data.

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
        """

        path_metadata = self.NR_dir + f'/metadata_{self.NR_ID}.txt'
        parsed_metadata = {}

        with open(path_metadata, 'r') as input_file:
            for line in input_file:
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                parsed_metadata[key.strip()] = float(value.strip().split()[0])

        missing_keys = [key for key in ['t_start', 't_end', 'dt'] if key not in parsed_metadata]
        if not('q' in parsed_metadata or ('m1' in parsed_metadata and 'm2' in parsed_metadata)):
            missing_keys.append('q or m1,m2')
        if len(missing_keys):
            raise ValueError("Missing mandatory injection metadata entries: {}".format(missing_keys))

        self.injection_metadata_parameters = dict(parsed_metadata)
       
        return (
            parsed_metadata['t_start'],
            parsed_metadata['t_end'],
            parsed_metadata['dt'],
            parsed_metadata.get('q'),
            parsed_metadata.get('Mf'),
            parsed_metadata.get('af'),
            {},
            {},
            {},
        )

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

        Read the metadata of the SXS waveform (with latest version released on 25th April 2025).

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
        
        _prime_sxs_simulations_cache()
        sim      = sxs.load("SXS:BBH:{}".format(self.NR_ID), download=self.download, auto_supersede=False, ignore_deprecation=True)
        metadata = sim.metadata
        
        tilt1, tilt2  = 0.0, 0.0

        q, Mf            = metadata['reference_mass_ratio'], metadata['remnant_mass']
        chi1, chi2, chif = metadata['reference_dimensionless_spin1'][2], metadata['reference_dimensionless_spin2'][2], metadata['remnant_dimensionless_spin'][2]
        ecc              = _parse_sxs_reference_eccentricity(metadata['reference-eccentricity'])

        return q, chi1, chi2, tilt1, tilt2, ecc, Mf, chif


    def load_SXS_addn_metadata(self, csv_path, ID_str):

        additional_data = pd.read_csv(csv_path) 
        row = additional_data.loc[additional_data['ID'] == int(ID_str)]
        if row.empty:
            raise ValueError(f"SXS:{ID_str} was not found in properties file `{csv_path}`.")
        row = row.iloc[0]
        metadata = {}
        for column, value in row.items():
            metadata[column] = value
            if column.startswith('A_peak') and column.endswith('dotdot'):
                mode = column[len('A_peak'):-len('dotdot')]
                if len(mode) == 2 and mode.isdigit():
                    metadata[f'A_peak{mode}dotdot'] = value
            elif column.startswith('A_peak'):
                mode = column[len('A_peak'):]
                if len(mode) == 2 and mode.isdigit():
                    metadata[f'A_peak_{mode}'] = value
            elif column.startswith('omega_peak'):
                mode = column[len('omega_peak'):]
                if len(mode) == 2 and mode.isdigit():
                    metadata[f'omega_peak_{mode}'] = value
                    metadata[f'omg_peak_{mode}'] = value
        self.additional_metadata = metadata
        A_peak_22 = metadata.get('A_peak_22')
        omg_peak_22 = metadata.get('omg_peak_22')
        A_nr_error = metadata.get('A_nr_error')
        A_peak22dotdot = metadata.get('A_peak22dotdot')
        bmrg = metadata.get('b_massless_EOB')
        Emrg = metadata.get('Heff_til')
        Jmrg = metadata.get('Jmrg_til')

        return A_peak_22, omg_peak_22, A_nr_error, A_peak22dotdot, bmrg, Emrg, Jmrg

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

                
        waveform_NR = Waveform_rit(NR_data_path=self.NR_dir, csv_path=self.additional_NR_properties, fit_path=self.fits, ID=self.NR_ID)
        
        # Read intrinsic parameters
        data        = waveform_NR.load_metadata()
        m1          = float(data['initial-mass1'])
        m2          = float(data['initial-mass2'])
        chi1z       = float(data['initial-bh-chi1z'])
        chi2z       = float(data['initial-bh-chi2z'])
        q           = m1/m2
        nu          = q/(1+q)**2

        # Read initial conditions.
        # FIXME: these are the initial data before relaxation, so not precisely correct.
        r0          = float(data['initial-separation'])
        e0          = float(data['initial-ADM-energy'])
        j0          = float(data['initial-ADM-angular-momentum-z'])/nu
        ecc         = float(data['eccentricity'])

        Mf          = float(data['final-mass'])
        chif        = float(data['final-chi'])

        # FIXME: Generalise to multiple modes with dictionaries.
        A_peak_22       = float(data['A_peak_22'])
        omg_peak_22     = float(data['omg_peak_22'])
        A_nr_error      = float(data['A_nr_error'])
        A_peak22dotdot  = float(data['A_peak22dotdot'])
        bmrg            = float(data['bmrg'])
        Emrg            = float(data['Emrg'])
        Jmrg            = float(data['Jmrg'])

        return q, chi1z, chi2z, ecc, Mf, chif, A_peak_22, omg_peak_22, A_nr_error, A_peak22dotdot, bmrg, Emrg, Jmrg

    def read_waveform_lm_from_RIT(self):

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
                
        waveform_NR = Waveform_rit(NR_data_path=self.NR_dir, ID=self.NR_ID, ell = self.l, m = self.m)                
        
        if  (self.waveform_type=='strain'): 
            t_NR, wv_re, wv_im, _, _  = waveform_NR.load_waveform_lm()
            t_NR = t_NR.astype(np.float64)
        elif(self.waveform_type=='psi4'  ): 

            # RIT Psi4 data are longer than the strain (likely because of the data conditioning involved in getting h), and have not been shifted to have the zero corresponding to the maximum of the amplitude.
            # Here, we compensate for the length difference, and in the code below we avoid shifting the time axis to have the zero set to the first element, to re-aligned it with the strain.
            t_NR_h,     _,     _, _, _  = waveform_NR.load_waveform_lm()
            t_NR  , wv_re, wv_im, _, _  = waveform_NR.load_psi4_lm()

            dt_missing  = t_NR_h[1] - t_NR_h[0]
            len_missing = dt_missing * (len(t_NR)-len(t_NR_h))

            t_NR = t_NR.astype(np.float64)
            t_NR = t_NR - len_missing
        
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

                
        waveform_NR = Waveform_C2EFT(path=os.path.join(self.NR_dir, self.NR_ID, 'Res_{resolution}_sigma{sigma}_tau_{tau}'.format(resolution=resolution, sigma=str(sigma).replace('.', 'p'), tau=str(tau).replace('.', 'p'))))
        
        # Read intrinsic parameters
        data  = waveform_NR.load_metadata()

        q     = float(data['q'])
        chi1z = float(data['chi1'])
        chi2z = float(data['chi2'])

        Mf   = float(data['Mf'])
        chif = float(data['af'])

        eps  = float(data['epsilon'])

        return q, chi1z, chi2z, Mf, chif, eps

    def read_waveform_lm_from_C2EFT(self, resolution, sigma, tau):

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
                
        waveform_NR = Waveform_C2EFT(path=os.path.join(self.NR_dir, self.NR_ID, 'Res_{resolution}_sigma{sigma}_tau_{tau}'.format(resolution=resolution, sigma=str(sigma).replace('.', 'p'), tau=str(tau).replace('.', 'p'))), ell = self.l, m = self.m)                
        t_NR, wv_re, wv_im = waveform_NR.load_waveform_lm()

        t_NR = t_NR.astype(np.float64)
        
        return t_NR, wv_re, wv_im

    def read_waveform_lm_from_SXS(self, ExtOrd, LevRes):

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
        
        _prime_sxs_simulations_cache()
        sim = sxs.load("SXS:BBH:{}/Lev{}".format(self.NR_ID, LevRes), download=self.download, extrapolation_order=ExtOrd, auto_supersede=False, ignore_deprecation=True)
        waveform = sim.h
        
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

    def read_RWZ_env_metadata(self):

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
        simulation_parameters = read_RWZ_env_simulation_parameters(os.path.join(sim_path, 'sim_params.txt'))
        a_halo, M_halo, C     = simulation_parameters['a_halo'], simulation_parameters['M_halo'], simulation_parameters['C']

        return a_halo, M_halo, C

    def available_RWZ_resolution_levels(self, extrap_order):

        sim_dir = os.path.join(self.NR_dir, '{}'.format(self.NR_ID))
        prefix  = f'HplusHcrossLM{self.l}{self.m}RL'
        suffix  = f'EP{extrap_order}.dat'
        levels  = []

        if not os.path.isdir(sim_dir):
            raise FileNotFoundError("RWZ simulation directory not found: {}".format(sim_dir))

        for filename in os.listdir(sim_dir):
            if filename.startswith(prefix) and filename.endswith(suffix):
                res_level = filename[len(prefix):-len(suffix)]
                try:
                    levels.append(int(res_level))
                except ValueError:
                    pass

        if not levels:
            raise FileNotFoundError("No RWZ waveform files found for mode ({}, {}) and extrapolation order {} in {}".format(self.l, self.m, extrap_order, sim_dir))

        return sorted(set(levels))

    def highest_available_RWZ_resolution_level(self, extrap_order):

        return self.available_RWZ_resolution_levels(extrap_order)[-1]

    def lower_available_RWZ_resolution_level(self, res_level, extrap_order):

        lower_levels = [level for level in self.available_RWZ_resolution_levels(extrap_order) if level < res_level]
        if not lower_levels:
            raise ValueError("Only a single RWZ resolution available below level {} for extrapolation order {}.".format(res_level, extrap_order))

        return lower_levels[-1]

    def read_waveform_lm_from_RWZ(self, res_level=None, extrap_order=None, allow_simple_fallback=True):

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

        sim_dir = os.path.join(self.NR_dir, '{}'.format(self.NR_ID))
        sim_paths = []
        if res_level == -1 and extrap_order is not None:
            try:
                res_level = self.highest_available_RWZ_resolution_level(extrap_order)
            except FileNotFoundError:
                if not allow_simple_fallback:
                    raise
                res_level = None
        if res_level is not None and extrap_order is not None:
            sim_paths.append(os.path.join(sim_dir, f'HplusHcrossLM{self.l}{self.m}RL{res_level}EP{extrap_order}.dat'))
        if allow_simple_fallback:
            sim_paths.append(os.path.join(sim_dir, f'HplusHcrossLM{self.l}{self.m}.dat'))

        for sim_path in sim_paths:
            if os.path.exists(sim_path):
                break
        else:
            raise FileNotFoundError("RWZ waveform file not found. Tried: {}".format(', '.join(sim_paths)))

        sim_file  = np.genfromtxt(sim_path, names=True)
        
        time             = sim_file['t']
        waveform_real    = sim_file['hp']
        waveform_imag    = sim_file['hc']
        
        return time, waveform_real, waveform_imag

    def read_hlm_from_RWZ(self, res_level, extrap_order):

        return self.read_waveform_lm_from_RWZ(res_level, extrap_order, allow_simple_fallback=False)

    def read_waveform_lm_from_Teukolsky(self, res_level):

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
