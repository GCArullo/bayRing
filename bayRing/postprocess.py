# Standard python packages
import corner, csv, hashlib, h5py, matplotlib.pyplot as plt, numpy as np, os, pickle, qnm, scipy.linalg as sl, seaborn as sns, shutil, numba
from scipy.interpolate           import interp1d
from scipy.optimize              import fmin
from itertools import product
import math

# GW-packages
from pycbc.psd                   import from_txt
from pycbc.types.timeseries      import TimeSeries
from pycbc.types.frequencyseries import FrequencySeries
from pycbc.filter                import sigma, match as compute_FD_match
import lal
from lal.antenna                 import AntennaResponse
import pyRing.utils              as pyRing_utils

# Package internal imports
import bayRing.utils             as utils
import bayRing.waveform_utils    as waveform_utils


# Costants
twopi = 2.*np.pi

# Color palette
colbBlue   = "#4477AA"
colbRed    = "#EE6677"
colbGreen  = "#228833"
colbYellow = "#CCBB44"
colbCyan   = "#66CCEE"
colbPurple = "#AA3377"
colbGray   = "#BBBBBB"

# Conversions
C_mt=(lal.MSUN_SI * lal.G_SI) / (lal.C_SI**3) #s, converts a mass expressed in solar masses into a time in seconds
C_md=(lal.MSUN_SI * lal.G_SI)/(1e6*lal.PC_SI*lal.C_SI**2) #adimensional, converts a mass expressed in solar masses to a distance in Megaparsec

strain_components = ('real', 'imag')
summary_percentiles = (5, 50, 95)
plot_percentiles = (50,)
point_estimate_methods = ('Minimization', 'Linear-inversion')
point_estimate_posterior_samples = 0
window_key_index = {
    'window_DX': 0,
    'window_SX': 1,
    'k': 2,
    'saturation_DX': 3,
    'saturation_SX': 4,
}
mismatch_smoothing_subfolders = {
    'below': 'Left_smoothing',
    'above': 'Right_smoothing',
    'below-and-above': 'Both_edges_smoothing',
}
mismatch_and_snr_diagnostics_filename = 'mismatch_and_snr_diagnostics.tsv'
mismatch_and_snr_parameters_filename = 'mismatch_and_snr_diagnostic_parameters.tsv'
mismatch_parameter_fieldnames = (
    'run_id',
    'diagnostic_type',
    'remnant_mass_solar_masses',
    'luminosity_distance_mpc',
    'start_time_M',
    'n_fft',
    'low_frequency_window_hz',
    'high_frequency_window_hz',
    'smoothing_steepness',
    'low_frequency_saturation',
    'high_frequency_saturation',
    'smoothing_direction',
)
mismatch_diagnostic_fieldnames = (
    'run_id',
    'diagnostic_type',
    'confidence_interval',
    'strain_data',
    'inclination',
    'azimuth',
    'psi',
    'mismatch',
    'optimal_snr',
    'optimal_snr_fd',
)

class PointEstimateResults(dict):

    def __init__(self, values, errors=None, covariance=None):

        super().__init__(values)
        self.errors     = {} if errors is None else dict(errors)
        self.covariance = covariance

def read_posterior_samples(outdir):

    posterior_path = _posterior_path(outdir)
    delimiter = None

    with open(posterior_path, 'r') as posterior_file:
        for line in posterior_file:
            line = line.strip()
            if(line!=''):
                delimiter = "," if "," in line else None
                break

    return np.genfromtxt(posterior_path, names=True, deletechars="", delimiter=delimiter)

def _point_estimate_path(outdir):

    return os.path.join(outdir, 'Algorithm', 'point_estimates.dat')

def _posterior_path(outdir):

    return os.path.join(outdir, 'Algorithm', 'posterior.dat')

def read_point_estimates(outdir):

    point_estimates_path = _point_estimate_path(outdir)
    values = {}
    errors = {}

    with open(point_estimates_path, 'r') as point_estimates_file:
        for line in point_estimates_file:
            line = line.strip()
            if(line == '' or line.startswith('#')):
                continue

            fields = line.split()
            if(len(fields) < 2):
                continue

            values[fields[0]] = float(fields[1])
            if(len(fields) > 2):
                try:
                    errors[fields[0]] = float(fields[2])
                except ValueError:
                    errors[fields[0]] = getattr(np, 'nan', float('nan'))

    if(len(values) == 0):
        raise ValueError("No point estimates found in {}.".format(point_estimates_path))

    return PointEstimateResults(values, errors=errors)

def _finite_point_estimate_errors(results, errors=None):

    if(errors is None):
        errors = getattr(results, 'errors', {})

    finite_errors = {}
    for name in results.keys():
        try:
            error = float(errors.get(name, 0.0))
        except (AttributeError, TypeError, ValueError):
            error = 0.0
        if(math.isfinite(error) and error > 0.0):
            finite_errors[name] = error

    return finite_errors

def point_estimate_parameter_samples(results, errors=None):

    mean_sample = dict(results)
    samples = [mean_sample]

    for name, error in _finite_point_estimate_errors(results, errors=errors).items():
        try:
            mean_value = float(mean_sample[name])
        except (TypeError, ValueError):
            continue

        upper_sample = dict(mean_sample)
        lower_sample = dict(mean_sample)
        upper_sample[name] = mean_value + error
        lower_sample[name] = mean_value - error
        samples.extend([lower_sample, upper_sample])

    return samples

def _structured_point_estimate_parameter_samples(results):

    names = results.dtype.names
    values = {}
    errors = {}

    for name in names:
        samples = np.atleast_1d(results[name])
        values[name] = float(np.mean(samples))
        if(len(samples) > 1):
            errors[name] = float(np.std(samples))

    return point_estimate_parameter_samples(values, errors=errors)

def waveform_parameter_samples(results, method=None):

    if isinstance(results, dict):
        if(method in point_estimate_methods):
            return point_estimate_parameter_samples(results)
        else:
            return [results]

    if(method in point_estimate_methods and getattr(results, 'dtype', None) is not None and results.dtype.names is not None):
        return _structured_point_estimate_parameter_samples(results)

    if(getattr(results, 'shape', None) == ()):
        return [results]

    return list(results)

def model_component_lists(results, inference_model, method=None):

    parameter_samples = waveform_parameter_samples(results, method)
    models_re_list    = []
    models_im_list    = []
    skipped_samples   = 0
    for p in parameter_samples:
        try:
            model = np.array(inference_model.model(p))
        except (FloatingPointError, OverflowError, TypeError, ValueError):
            skipped_samples += 1
            continue
        if not np.all(np.isfinite(model)):
            skipped_samples += 1
            continue
        models_re_list.append(np.real(model))
        models_im_list.append(np.imag(model))

    if not models_re_list:
        raise RuntimeError("No finite waveform samples could be constructed for postprocessing.")
    if skipped_samples:
        print("* Warning: skipped {} invalid point-estimate waveform sample(s) during postprocessing.".format(skipped_samples))

    return models_re_list, models_im_list

def _physical_strain_scale(M, dL):
    return (C_md * M) / dL

def _scaled_strain_components(NR_sim, M, dL):
    scale = _physical_strain_scale(M, dL)
    return {
        'real': NR_sim.NR_r_cut * scale,
        'imag': NR_sim.NR_i_cut * scale,
    }

def _percentile_waveforms(models_re_list, models_im_list, perc, M, dL):
    scale = _physical_strain_scale(M, dL)
    return (
        np.percentile(np.array(models_re_list), [perc], axis=0)[0] * scale,
        np.percentile(np.array(models_im_list), [perc], axis=0)[0] * scale,
    )

def _toeplitz_whitened_norm(acf, waveform):
    whitened = sl.solve_toeplitz(acf, waveform, check_finite=False)
    return whitened, np.sqrt(abs(np.dot(waveform, whitened)))

def _time_domain_mismatch(acf, reference, comparison):

    whitened_reference, reference_norm = _toeplitz_whitened_norm(acf, reference)
    _, comparison_norm = _toeplitz_whitened_norm(acf, comparison)

    if(reference_norm == 0.0 or comparison_norm == 0.0):
        raise ValueError("Cannot compute mismatch for a waveform with zero norm.")

    match = abs(np.dot(comparison, whitened_reference)) / (reference_norm * comparison_norm)
    match = np.minimum(1 - abs(1 - match), match)

    return 1 - match

def _format_diagnostic_value(value):
    if value is None:
        return ''
    np_generic = getattr(np, 'generic', ())
    if np_generic and isinstance(value, np_generic):
        value = value.item()
    if isinstance(value, (int, float)):
        return "{:.16g}".format(value)
    return str(value)

def _read_tsv_rows(path):
    if not(os.path.exists(path)):
        return []
    with open(path, 'r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames is None:
            return []
        return [dict(row) for row in reader]

def _write_tsv_rows(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fieldnames})

def _upsert_tsv_row(path, fieldnames, new_row, key_fields):
    rows = _read_tsv_rows(path)
    new_key = tuple(new_row.get(field, '') for field in key_fields)
    updated = False

    for row in rows:
        row_key = tuple(row.get(field, '') for field in key_fields)
        if row_key != new_key:
            continue
        for field in fieldnames:
            value = new_row.get(field, '')
            if value != '':
                row[field] = value
            elif field not in row:
                row[field] = ''
        updated = True
        break

    if not(updated):
        rows.append({field: new_row.get(field, '') for field in fieldnames})

    _write_tsv_rows(path, fieldnames, rows)

def _mismatch_diagnostics_path(outdir):
    return os.path.join(_mismatch_root_dir(outdir), mismatch_and_snr_diagnostics_filename)

def _mismatch_parameters_path(outdir):
    return os.path.join(_mismatch_root_dir(outdir), mismatch_and_snr_parameters_filename)

def _mismatch_run_id(diagnostic_type, M, dL, t_start_g, n_fft, window_size_DX,
                     window_size_SX, k, saturation_DX, saturation_SX, direction=None):
    parts = (
        diagnostic_type,
        M,
        dL,
        t_start_g,
        n_fft,
        window_size_DX,
        window_size_SX,
        k,
        saturation_DX,
        saturation_SX,
        direction,
    )
    digest_source = '\t'.join(_format_diagnostic_value(part) for part in parts)
    return 'run_' + hashlib.sha1(digest_source.encode('utf-8')).hexdigest()[:12]

def _mismatch_parameter_row(run_id, diagnostic_type, M, dL, t_start_g, n_fft,
                            window_size_DX, window_size_SX, k, saturation_DX,
                            saturation_SX, direction=None):
    return {
        'run_id': run_id,
        'diagnostic_type': diagnostic_type,
        'remnant_mass_solar_masses': _format_diagnostic_value(M),
        'luminosity_distance_mpc': _format_diagnostic_value(dL),
        'start_time_M': _format_diagnostic_value(t_start_g),
        'n_fft': _format_diagnostic_value(n_fft),
        'low_frequency_window_hz': _format_diagnostic_value(window_size_DX),
        'high_frequency_window_hz': _format_diagnostic_value(window_size_SX),
        'smoothing_steepness': _format_diagnostic_value(k),
        'low_frequency_saturation': _format_diagnostic_value(saturation_DX),
        'high_frequency_saturation': _format_diagnostic_value(saturation_SX),
        'smoothing_direction': _format_diagnostic_value(direction),
    }

def _record_mismatch_diagnostic(
    outdir,
    diagnostic_type,
    M,
    dL,
    t_start_g,
    n_fft,
    window_size_DX,
    window_size_SX,
    k,
    saturation_DX,
    saturation_SX,
    direction=None,
    confidence_interval=None,
    strain_data=None,
    inclination=None,
    azimuth=None,
    psi=None,
    mismatch=None,
    optimal_snr=None,
    optimal_snr_fd=None,
):
    run_id = _mismatch_run_id(
        diagnostic_type, M, dL, t_start_g, n_fft, window_size_DX,
        window_size_SX, k, saturation_DX, saturation_SX, direction
    )
    _upsert_tsv_row(
        _mismatch_parameters_path(outdir),
        mismatch_parameter_fieldnames,
        _mismatch_parameter_row(
            run_id, diagnostic_type, M, dL, t_start_g, n_fft,
            window_size_DX, window_size_SX, k, saturation_DX,
            saturation_SX, direction
        ),
        ('run_id',),
    )
    _upsert_tsv_row(
        _mismatch_diagnostics_path(outdir),
        mismatch_diagnostic_fieldnames,
        {
            'run_id': run_id,
            'diagnostic_type': diagnostic_type,
            'confidence_interval': _format_diagnostic_value(confidence_interval),
            'strain_data': _format_diagnostic_value(strain_data),
            'inclination': _format_diagnostic_value(inclination),
            'azimuth': _format_diagnostic_value(azimuth),
            'psi': _format_diagnostic_value(psi),
            'mismatch': _format_diagnostic_value(mismatch),
            'optimal_snr': _format_diagnostic_value(optimal_snr),
            'optimal_snr_fd': _format_diagnostic_value(optimal_snr_fd),
        },
        ('run_id', 'diagnostic_type', 'confidence_interval', 'strain_data', 'inclination', 'azimuth', 'psi'),
    )
    return _mismatch_diagnostics_path(outdir)

def _windowed_result_path(outdir, prefix, M, dL, t_start_g, n_fft, window_size_DX,
                          window_size_SX, k, saturation_DX, saturation_SX,
                          include_saturation=True, fd=False):
    filename = (
        f"{prefix}_M_{M}_dL_{dL}_t_s_{round(t_start_g,1)}M"
        f"_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz"
        f"_k_{round(k,2)}"
    )
    if include_saturation:
        filename += f"_satDX_{round(saturation_DX,1)}_satSD_{round(saturation_SX,1)}"
    filename += f"_NFFT_{n_fft}"
    if fd:
        filename += "_FD"
    return os.path.join(outdir, 'Algorithm/Mismatch', f"{filename}.txt")

def _initialise_result_files(*path_headers):
    for path, header in path_headers:
        with open(path, 'w') as outfile:
            outfile.write(header)

def _append_result(path, *values):
    with open(path, 'a') as outfile:
        outfile.write('\t'.join(map(str, values)) + '\n')

def _mode_series_with_negative_m_symmetry(mode_series, include_negative_m):

    explicit_modes = set(mode_series.keys())
    expanded = dict(mode_series)
    if not(include_negative_m):
        return expanded

    for (l_value, m_value), series in mode_series.items():
        if(m_value == 0):
            continue
        counterpart = (l_value, -m_value)
        if(counterpart not in explicit_modes and counterpart not in expanded):
            expanded[counterpart] = ((-1)**l_value) * np.conjugate(series)

    return expanded

def _project_modes_to_polarizations(mode_series, inclination, azimuth, include_negative_m=True):

    projected_strain = 0.0j
    for (l_value, m_value), series in _mode_series_with_negative_m_symmetry(mode_series, include_negative_m).items():
        y_lm = lal.SpinWeightedSphericalHarmonic(inclination, azimuth, -2, l_value, m_value)
        projected_strain = projected_strain + series * y_lm

    return np.real(projected_strain), -np.imag(projected_strain)

def _project_modes_to_detector(mode_series, inclination, azimuth, F_plus, F_cross, include_negative_m=True):

    h_plus, h_cross = _project_modes_to_polarizations(mode_series, inclination, azimuth, include_negative_m)

    return F_plus * h_plus + F_cross * h_cross

def _mode_percentile_waveform(model_samples, percentile):

    samples = np.asarray(model_samples)

    return (
        np.percentile(np.real(samples), percentile, axis=0)
        + 1j*np.percentile(np.imag(samples), percentile, axis=0)
    )

def _load_hm_mode_products(run_parameters):

    product_path = os.path.join(run_parameters['I/O']['outdir'], 'NR_sim.pkl')
    with open(product_path, 'rb') as product_file:
        NR_sim, model_samples, _ = pickle.load(product_file)

    return {
        'mode': (run_parameters['NR-data']['l-NR'], run_parameters['NR-data']['m']),
        'outdir': run_parameters['I/O']['outdir'],
        'time': np.asarray(NR_sim.t_NR_cut),
        'nr': np.asarray(NR_sim.NR_r_cut) + 1j*np.asarray(NR_sim.NR_i_cut),
        'model_samples': np.asarray(model_samples),
    }

def _common_mode_time(mode_products):

    t_start = max(product['time'][0] for product in mode_products)
    t_end   = min(product['time'][-1] for product in mode_products)
    if(t_end <= t_start):
        raise ValueError("The selected NR modes do not have an overlapping fit interval.")

    n_points = min(len(product['time']) for product in mode_products)

    return np.linspace(t_start, t_end, n_points)

def _interpolate_mode_series(mode_products, common_time, percentile=None):

    interpolated = {}
    for product in mode_products:
        if(percentile is None):
            series = product['nr']
        else:
            series = _mode_percentile_waveform(product['model_samples'], percentile)
        interpolated[product['mode']] = interp1d(product['time'], series, bounds_error=False, fill_value=0.0)(common_time)

    return interpolated

def _hm_sum_output_dir(base_outdir, t_start, n_start_times):

    outdir = os.path.join(base_outdir, 'HM_sum')
    if(n_start_times > 1):
        label = "{:.12g}".format(float(t_start)).replace('-', 'm').replace('+', '').replace('.', 'p')
        outdir = os.path.join(outdir, "t_start_{}M".format(label))

    return outdir

def _mismatch_root_dir(outdir):
    path = os.path.join(outdir, 'Algorithm/Mismatch')
    os.makedirs(path, exist_ok=True)
    return path

def _mismatch_subfolder(direction):
    try:
        return mismatch_smoothing_subfolders[direction]
    except KeyError:
        allowed = "', '".join(mismatch_smoothing_subfolders)
        raise ValueError("Invalid mismatch smoothing direction '{}'. Choose between '{}'.".format(direction, allowed))

def _mismatch_plot_dir(outdir, direction):
    save_path = os.path.join(_mismatch_root_dir(outdir), _mismatch_subfolder(direction))
    os.makedirs(save_path, exist_ok=True)
    return save_path

def _window_values(data, index):
    return sorted(set(key[index] for key in data.keys()))

def _groups_from_keys(data, indices):
    return sorted(set(tuple(key[index] for index in indices) for key in data.keys()))

def _groups_from_grid(data, indices):
    return product(*(_window_values(data, index) for index in indices))

def _key_matches(key, indices, values):
    return all(key[index] == value for index, value in zip(indices, values))

def read_results_object_from_previous_inference(parameters):

    if(parameters['Inference']['method'] in point_estimate_methods):

        n_samples = int(parameters['Inference'].get('point-estimate-posterior-samples', point_estimate_posterior_samples))
        posterior_path = _posterior_path(parameters['I/O']['outdir'])
        point_estimates_path = _point_estimate_path(parameters['I/O']['outdir'])

        if(n_samples > 0 and os.path.exists(posterior_path)):
            results_object = read_posterior_samples(parameters['I/O']['outdir'])
        elif(os.path.exists(point_estimates_path)):
            results_object = read_point_estimates(parameters['I/O']['outdir'])
        else:
            results_object = read_posterior_samples(parameters['I/O']['outdir'])

    elif(parameters['Inference']['method'] == 'Nested-sampler'):

        if(parameters['Inference']['sampler'] == 'cpnest'):
            results_object = read_posterior_samples(parameters['I/O']['outdir'])
        elif(parameters['Inference']['sampler'] == 'raynest'):
            filename        = os.path.join( parameters['I/O']['outdir'],'Algorithm/raynest.h5')
            h5_file         = h5py.File(filename,'r')
            results_object  = h5_file['combined'].get('posterior_samples')

    else: raise ValueError('Method {} not recognised.'.format(parameters['Inference']['method']))

    return results_object

def print_point_estimate(results_object, names, method):

    """

    Print the point estimates of the results of a minimization or a nested sampling algorithm.

    Parameters
    ----------

    results_object : object
        Object containing the results of the minimization or nested sampling algorithm.

    names : list
        List of the names of the parameters.

    method : str
        Method used to obtain the results from which the point estimates will be drawn. Can be either 'Minimization' or 'Nested-sampler'.

    Returns
    -------

    Nothing, but prints the point estimates.

    """

    if(isinstance(results_object, dict)):
        longest_name_length = utils.find_longest_name_length(results_object.keys())
        for key in results_object.keys():
            print('{} : {:.12f}'.format(key.ljust(longest_name_length), results_object[key]))
    else:
        longest_name_length = utils.find_longest_name_length(names)
        for key in names:
            median      = np.median(results_object[key])
            lower_bound = median-np.percentile(results_object[key], 5)
            upper_bound = np.percentile(results_object[key], 95)-median
            print('{} : {:.12f} + {:.12f} - {:.12f}'.format(key.ljust(longest_name_length), median, upper_bound, lower_bound))

    return

def save_point_estimates(results, outdir, errors=None):

    """

    Save the point estimates and one-sigma errors for point-estimate methods.

    Post-processing reads this file directly when no Gaussian point-estimate
    posterior is requested.

    """

    if(errors is None):
        errors = getattr(results, 'errors', {})

    point_estimates_path = _point_estimate_path(outdir)
    missing_error = getattr(np, 'nan', float('nan'))

    with open(point_estimates_path, 'w') as outfile:
        outfile.write('# parameter\tvalue\tsigma\n')
        for name in results.keys():
            outfile.write('{}\t{}\t{}\n'.format(name, results[name], errors.get(name, missing_error)))

    return point_estimates_path

def _point_estimate_covariance(names, covariance=None, errors=None):

    if(errors is None):
        errors = {}

    if(covariance is None):
        variances = []
        for name in names:
            error = errors.get(name, 0.0)
            try:
                error = float(error)
            except (TypeError, ValueError):
                error = 0.0
            if not(np.isfinite(error)):
                error = 0.0
            variances.append(error**2)
        return np.diag(variances)

    covariance = np.asarray(covariance, dtype=float)
    if(covariance.shape != (len(names), len(names)) or not(np.all(np.isfinite(covariance)))):
        return _point_estimate_covariance(names, covariance=None, errors=errors)

    covariance = 0.5*(covariance + covariance.T)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = np.maximum(eigvals, 0.0)

    return np.dot(eigvecs*eigvals, eigvecs.T)

def save_point_estimate_posterior(results, outdir, covariance=None, errors=None, seed=None, n_samples=point_estimate_posterior_samples):

    """

    Save a Gaussian posterior approximation for point-estimate methods.

    The samples are drawn from the multivariate normal distribution centered on
    the point estimate, using the supplied parameter covariance. If the full
    covariance is unavailable, the diagonal covariance implied by the stored
    one-sigma errors is used.

    """

    n_samples = int(n_samples)
    if(n_samples < 0):
        raise ValueError("Cannot save a point-estimate posterior with a negative number of samples.")
    if(n_samples == 0):
        return None

    names = list(results.keys())
    if(len(names)==0):
        raise ValueError("Cannot save a point-estimate posterior with no parameters.")

    mean       = np.array([float(results[name]) for name in names], dtype=float)
    if(errors is None):
        errors = getattr(results, 'errors', {})
    covariance = _point_estimate_covariance(names, covariance=covariance, errors=errors)

    rng     = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mean, covariance, size=n_samples)

    posterior_path = _posterior_path(outdir)
    np.savetxt(posterior_path, samples, header='\t'.join(names), delimiter='\t')

    return posterior_path

def remove_point_estimate_posterior(outdir):

    posterior_path = _posterior_path(outdir)
    if(os.path.exists(posterior_path)):
        os.remove(posterior_path)
        return posterior_path

    return None

def store_and_print_amp_phi(amp_name, phi_name, t0, omega, tau, results_object, longest_name_length, outdir):

    """

    Store and print the amplitude and phase of the inferred mode when defined at t0.

    Parameters
    ----------

    amp_name : str
        Name of the amplitude parameter.

    phi_name : str
        Name of the phase parameter.

    t0 : float
        Time at which the amplitude and phase are defined.

    omega : float
        Frequency of the mode.

    tau : float
        Damping time of the mode.

    results_object : dict
        Dictionary containing the results of the inference algorithm.

    longest_name_length : int
        Length of the longest parameter name.

    outdir : str
        Output directory.

    Returns
    -------

    Nothing, but stores and prints the amplitude and phase of the inferred mode.

    """

    exp_tau_factor   = np.exp(t0/tau)
    sum_omega_factor = t0 * omega

    amp_median =  np.exp(np.median(    results_object[amp_name]    )) *   exp_tau_factor
    amp_lower  =  np.exp(np.percentile(results_object[amp_name],  5)) *   exp_tau_factor
    amp_upper  =  np.exp(np.percentile(results_object[amp_name], 95)) *   exp_tau_factor
    phi_median =        (np.median(    results_object[phi_name]    )  - sum_omega_factor)%(2*np.pi)
    phi_lower  =        (np.percentile(results_object[phi_name],  5)  - sum_omega_factor)%(2*np.pi)
    phi_upper  =        (np.percentile(results_object[phi_name], 95)  - sum_omega_factor)%(2*np.pi)

    amp_lower_err  = amp_median - amp_lower
    amp_upper_err  = amp_upper  - amp_median
    phi_lower_err  = phi_median - phi_lower
    phi_upper_err  = phi_upper  - phi_median

    print('{} : {:.12f} + {:.12f} - {:.12f}'.format(amp_name.split('ln_')[-1].ljust(longest_name_length), amp_median, amp_upper_err, amp_lower_err))
    print('{} : {:.12f} + {:.12f} - {:.12f}'.format(phi_name.ljust(longest_name_length), phi_median, phi_upper_err, phi_lower_err))

    outFile_amp = open(os.path.join(outdir,'Peak_quantities/amps_tpeak.txt'), 'a')
    outFile_amp.write('{}\t{}\t{}\t{}\n'.format(amp_name.ljust(longest_name_length), amp_median, amp_lower, amp_upper))
    outFile_amp.close()
    outFile_phi = open(os.path.join(outdir,'Peak_quantities/phis_tpeak.txt'), 'a')
    outFile_phi.write('{}\t{}\t{}\t{}\n'.format(phi_name.ljust(longest_name_length), phi_median, phi_lower, phi_upper))
    outFile_phi.close()

    return

def post_process_amplitudes(t0, results_object, NR_metadata, qnm_cached, modes, quad_modes, outdir):

    """

    Post-process the amplitudes and phases of the inferred modes.

    Parameters
    ----------

    t0 : float
        Time at which the amplitude and phase are defined.

    results_object : dict
        Dictionary containing the results of the inference algorithm.

    NR_metadata : dict
        Dictionary containing the metadata of the NR simulation.

    qnm_interpolants : dict
        Dictionary containing the interpolants of the QNM frequencies and damping times.

    modes : list
        List of the modes to be inferred.

    quad_modes : list
        List of the quadrupole modes to be inferred.

    outdir : str
        Output directory.

    Returns
    -------

    Nothing, but stores and prints the amplitude and phase of the inferred mode.

    """

    print('\n* Amplitudes and phases at t_peak:\n')

    outFile_amp = open(os.path.join(outdir,'Peak_quantities/amps_tpeak.txt'), 'w')
    outFile_phi = open(os.path.join(outdir,'Peak_quantities/phis_tpeak.txt'), 'w')
    outFile_amp.write('#name\tmedian\tlower\tupper\n')
    outFile_phi.write('#name\tmedian\tlower\tupper\n')
    outFile_amp.close()
    outFile_phi.close()

    Mf = NR_metadata['Mf']
    af = NR_metadata['af']

    if 'qf' in NR_metadata.keys(): qf = NR_metadata['qf']
    else                         : qf = None

    if (quad_modes is not None): longest_name_length = len('phi_diff_x-yz_x-yz_x-yz')
    else                       : longest_name_length = len('phi_x-yz')

    for (l_x, m_x, n_x) in modes:

        amp_name = 'ln_A_{}{}{}'.format(l_x, m_x, n_x)
        phi_name =  'phi_{}{}{}'.format(l_x, m_x, n_x)

        omega, tau = qnm_cached[(2,l_x,m_x,n_x)]['f'] * twopi, qnm_cached[(2,l_x,m_x,n_x)]['tau']

        store_and_print_amp_phi(amp_name, phi_name, t0, omega, tau, results_object, longest_name_length, outdir)

    if(quad_modes is not None):
        for quad_term in quad_modes.keys():
            for ((l,m,n),(l1,m1,n1),(l2,m2,n2)) in quad_modes[quad_term]:

                quad_string = '{}_{}{}{}_{}{}{}_{}{}{}'.format(quad_term, l,m,n, l1,m1,n1, l2,m2,n2)
                amp_name = 'ln_A_{}'.format(quad_string)
                phi_name = 'phi_{}'.format(quad_string)

                omega1, tau1 = qnm_cached[(2,l1, m1, n1)]['f'] * twopi, qnm_cached[(2,l1, m1, n1)]['tau']
                omega2, tau2 = qnm_cached[(2,l2, m2, n2)]['f'] * twopi, qnm_cached[(2,l2, m2, n2)]['tau']

                tau   = (tau1 * tau2)/(tau1 + tau2)
                if  (quad_term=='sum' ): omega = omega1 + omega2
                elif(quad_term=='diff'): omega = omega1 - omega2

                store_and_print_amp_phi(amp_name, phi_name, t0, omega, tau, results_object, longest_name_length, outdir)

    return

def l2norm_residual_vs_nr(results_object, inference_model, NR_sim, outdir, method=None):

    """

    Compare the residual of the fit with the NR error.

    Find the peak time of the amplitude.

    Parameters
    ----------

    results_object : dict
        Dictionary containing the results of the inference algorithm.

    inference_model : Nested sampler object
        Nested sampler object.

    NR_sim : NR_sim
        NR simulation object.

    outdir : str
        output directory

    Returns
    ---------

    Nothing, but prints and stores in a file the L2 norm of residuals and NR_error.

    """

    NR_err_r, NR_err_i = np.real(NR_sim.NR_cpx_err_cut), np.imag(NR_sim.NR_cpx_err_cut)
    NR_r, NR_i         = np.real(NR_sim.NR_cpx_cut)     , np.imag(NR_sim.NR_cpx_cut)
    t_cut = NR_sim.t_NR_cut

    models_re_list, models_im_list = model_component_lists(results_object, inference_model, method)

    wf_r = np.percentile(np.array(models_re_list),[50], axis=0)[0]
    wf_i = np.percentile(np.array(models_im_list),[50], axis=0)[0]

    l2_NR       = np.trapz(np.sqrt(      NR_err_r** 2 +       NR_err_i** 2), t_cut)
    l2_residual = np.trapz(np.sqrt((NR_r - wf_r) ** 2 + (NR_i - wf_i) ** 2), t_cut)

    print(f'* L2 norm of residual is: {l2_residual}')
    print(f'* L2 norm of NR error is: {l2_NR}\n')

    outFile_L2_errors = open(os.path.join(outdir,'Algorithm/L2_errors.txt'), 'w')
    outFile_L2_errors.write('# L2 norm of residual is \n')
    outFile_L2_errors.write(f'{l2_residual} \n')
    outFile_L2_errors.write('# L2 norm of NR error is \n')
    outFile_L2_errors.write(f'{l2_NR} \n')

    return

def init_plotting():

    """

    Function to set the default plotting parameters.

    Parameters
    ----------
    None

    Returns
    -------
    Nothing, but sets the default plotting parameters.

    """

    plt.rcParams['figure.max_open_warning'] = 0

    plt.rcParams['mathtext.fontset']  = 'stix'
    plt.rcParams['font.family']       = 'STIXGeneral'

    plt.rcParams['font.size']         = 14
    plt.rcParams['axes.linewidth']    = 1
    plt.rcParams['axes.labelsize']    = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize']    = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize']   = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize']   = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize']   = plt.rcParams['font.size']
    plt.rcParams['xtick.major.size']  = 3
    plt.rcParams['xtick.minor.size']  = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size']  = 3
    plt.rcParams['ytick.minor.size']  = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1

    plt.rcParams['legend.frameon']             = False
    plt.rcParams['legend.loc']                 = 'center left'
    plt.rcParams['contour.negative_linestyle'] = 'solid'

    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')

    return

def compare_with_GR_QNMs(results_object, qnm_cached, NR_sim, outdir):

    l,m              = NR_sim.l, NR_sim.m
    f_samples        = results_object['f_0']
    f_rd_fundamental = qnm_cached[(2,l,m,0)]['f']

    plt.figure()
    sns.histplot(f_samples       , color="darkred", fill=True , alpha=0.9, label='EFT fund mode')
    plt.axvline(f_rd_fundamental, color='black', linestyle='--', lw=2.2,  label='GR fund mode')
    plt.xlabel(r'$f_{fund}$')
    plt.ylabel(r'$p(f_{fund})$')
    plt.legend(loc='best')
    plt.savefig(os.path.join(outdir,'Plots/Results/f_fundamental.pdf'), bbox_inches='tight')

    return

def compute_pycbc_optimal_SNR(asd_file, h, n, f_min, f_max, delta_f):

    # Ensure PSD matches the waveform's `delta_f`
    #delta_f = 2*f_max/n
    n= 2*f_max/delta_f
    psd     = from_txt(
                        filename        = asd_file,
                        length          = n,
                        delta_f         = delta_f,
                        low_freq_cutoff = f_min,
                        is_asd_file     = True
                    )

    h_tilde = h.to_frequencyseries(delta_f=delta_f)
    fd_snr  = sigma(h_tilde, psd=psd, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)

    return fd_snr

@numba.njit
def fast_interpolation(x, xp, fp):
    """Numba-accelerated linear interpolation."""
    return np.interp(x, xp, fp)

def interpolate_waveform(t_start_g, t_end_g, M, wf_lNR, acf):

    """
    Interpolates the waveform to match the length of the autocovariance function (ACF).

    Parameters
    ----------
    - t_start_g (float) : Start time in geometrical units.
    - t_end_g   (float) : End time in geometrical units.
    - M (float)         : Mass of the system.
    - wf_lNR (array)    : The original NR waveform data.
    - acf (array)       : The autocovariance function (defines new length).

    Returns
    -------
    - wf_int (array) Interpolated waveform with the same length as `acf`.
    """

    # Compute start and end time in physical units
    t_start = t_start_g * C_mt * M
    t_end   = t_end_g * C_mt * M

    # Generate time arrays
    t_array = np.linspace(t_start, t_end, len(wf_lNR))  # Original waveform time
    t_int   = np.linspace(t_start, t_end, len(acf))       # Target interpolation time

    # Use Numba-optimized interpolation
    wf_int = fast_interpolation(t_int, t_array, wf_lNR)

    return wf_int

def convert_asd_to_pycbc_psd(asd_file, delta_f):

    """
    Load an ASD file, compute the PSD, and convert it to a PyCBC FrequencySeries.

    Parameters
    ----------

    asd_file (str): Path to the ASD file (two columns: frequency, ASD value)

    Returns
    -------

    pycbc.types.FrequencySeries: The computed PSD as a FrequencySeries object.
    """

    # Load ASD data from file
    data       = np.loadtxt(asd_file)
    asd_values = data[:, 1]   # Second column: ASD values

    # Compute PSD by squaring ASD values
    psd_values = asd_values ** 2

    print(f"Loaded ASD file: {asd_file}, PSD length: {len(psd_values)}")

    # Convert to PyCBC FrequencySeries
    psd = FrequencySeries(psd_values, delta_f=delta_f)

    return psd

def clear_directory(directory_path):

    """
    Clears all files inside a directory without deleting the directory itself.

    Parameters
    ----------
        directory_path (str): Path to the directory to be cleared.

    Returns
    -------
        None
    """

    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path): os.unlink(file_path)     # Delete files and symlinks
                elif os.path.isdir(file_path)                            : shutil.rmtree(file_path) # Recursively delete folders
            except Exception as e: print(f"Failed to delete {file_path}: {e}")

    else:
        os.makedirs(directory_path, exist_ok=True)

def truncate_and_interpolate_acf(t_ACF, ACF_smoothed, M, t_start_g, t_end_g, t_NR_s, print_truncation_info):

    """
    Truncate and interpolate the Autocorrelation Function (ACF) based on time constraints.

    Parameters
    ----------
        ACF_smoothed (np.ndarray): the original smoothed ACF.
        t_ACF (np.ndarray): the time axis associated to ACF_smoothed.
        t_start (float): Start time for analysis [geometric units].
        t_end (float): End time for analysis [geometric units].
        t_NR_s (np.ndarray): NR time array in seconds, starting at 0, and ending at t_end-t_start.
        N_sim (int): The number of points for the interpolated ACF.

    Returns
    -------
        np.ndarray: The new array corresponding to the interpolated ACF on the NR time array.
    """

    # First, we take only the first half of the ACF, which is the one associated to positive frequencies
    half_index        = len(ACF_smoothed) // 2
    t_ACF_half        = t_ACF[:half_index]
    ACF_smoothed_half = ACF_smoothed[:half_index]

    # Compute the truncation point (t_rd = t_end - t_start)
    t_rd  = (t_end_g - t_start_g) * C_mt * M
    index = np.argmin(np.abs(t_ACF_half - t_rd))

    # Truncate the ACF to ringdown analysis (See https://arxiv.org/abs/2107.05609 for discussion on truncation)
    ACF_truncated   = ACF_smoothed_half[:index+1]
    t_ACF_truncated = t_ACF_half[:index+1]

    # Perform linear interpolation
    interpolator = interp1d(t_ACF_truncated, ACF_truncated, kind='linear', fill_value="extrapolate")

    # Then, resample the truncated ACF with the NR array (expressed in seconds)
    ACF_trunc    = interpolator(t_NR_s)

    if print_truncation_info:

        print("\nTruncation info:")
        print(f"\nt_start[g] = {t_start_g}")
        print(f"\nt_end[g] = {t_end_g}")
        print("\nACF time array expr. in [s] (full): ", t_ACF)
        print("\nACF time array expr. in [s] (first half, associated to positive frequencies): ", t_ACF_half)
        print("\nTruncated ACF time array expr. in [s] : ", t_ACF_truncated)
        print("\nTruncated waveform time array expr. in geometrical units : ", t_NR_s/(M*C_mt))

    return ACF_trunc

def mismatch_sanity_checks(NR_sim, results, inference_model, outdir, method, acf, M, dL, t_start_g, t_end_g, window_size_DX, window_size_SX, k):

    """
    Performs sanity checks for mismatch computation.

    Parameters
    ----------

    NR_sim : NR_sim
        NR simulation object.

    results : dict
        Dictionary containing the results object.

    inference_model : inference_model
        Nested sampling model object.

    outdir : string
        Output directory.

    method : string
        Method used to fit the waveform.

    acf : array
        Autocovariance function of the noise (expressed in seconds).

    M : float
        Mass of the remnant (expressed in solar masses).

    dL: float
        Luminosity distance of the source with respect to the observer (expressed in Megaparsec).

    Returns
    -------

    Nothing, only creates sanity plots.
    """

    # outdir
    sanity_checks_dir = os.path.join(outdir, 'Algorithm/Mismatch', 'Sanity_Checks')

    # create folder
    os.makedirs(sanity_checks_dir, exist_ok=True)

    #start and end times of the analysis [s]
    t_start = t_start_g * C_mt * M
    t_end   = t_end_g * C_mt * M
    t_trunc = np.linspace(t_start, t_end, len(NR_sim.t_NR_cut))

    # Calculate scaled NR waveform components
    NR_r = NR_sim.NR_r_cut * (C_md * M) / dL
    NR_i = NR_sim.NR_i_cut * (C_md * M) / dL

    # Initialize lists to store waveform components
    models_re_list, models_im_list = model_component_lists(results, inference_model, method)

    wf_r_quantiles = {}
    wf_i_quantiles = {}

    for perc in [5, 50, 95]:
        wf_r_quantiles[perc] = np.percentile(np.array(models_re_list), [perc], axis=0)[0] * (C_md * M) / dL
        wf_i_quantiles[perc] = np.percentile(np.array(models_im_list), [perc], axis=0)[0] * (C_md * M) / dL

    # Compute whitened NR components
    whiten_NR_r = sl.solve_toeplitz(acf, NR_r, check_finite=False)
    whiten_NR_i = sl.solve_toeplitz(acf, NR_i, check_finite=False)

    # Compute whitened waveform quantiles
    wf_r_whitened = {perc: sl.solve_toeplitz(acf, wf_r_quantiles[perc], check_finite=False) for perc in [5, 50, 95]}
    wf_i_whitened = {perc: sl.solve_toeplitz(acf, wf_i_quantiles[perc], check_finite=False) for perc in [5, 50, 95]}

    # Create Toeplitz matrix from acf and compute its inverse
    acf_toeplitz     = sl.toeplitz(acf)
    acf_toeplitz_inv = np.linalg.inv(acf_toeplitz)

    # Apply whitening using the Toeplitz inverse matrix
    toeplitz_whitened_NR_r = np.dot(acf_toeplitz_inv, NR_r)
    toeplitz_whitened_NR_i = np.dot(acf_toeplitz_inv, NR_i)
    wf_r_toeplitz_whitened = {perc: np.dot(acf_toeplitz_inv, wf_r_quantiles[perc]) for perc in [5, 50, 95]}
    wf_i_toeplitz_whitened = {perc: np.dot(acf_toeplitz_inv, wf_i_quantiles[perc]) for perc in [5, 50, 95]}

    # Generate plot for real components (No Whitening)
    plt.figure(figsize=(10, 6))
    plt.plot(t_trunc, NR_r, label='NR_r', color='blue', linewidth=1.5)
    plt.plot(t_trunc, wf_r_quantiles[50], label='50% CI', linestyle='-', color=colbRed)
    plt.title('Real Component Comparison (No Whitening)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    filename = f"Real_Component_No_Whitening_M_{M}_dL_{dL}_M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}.pdf"
    plt.savefig(os.path.join(sanity_checks_dir, filename))
    plt.close()

    # Generate plot for imaginary components (No Whitening)
    plt.figure(figsize=(10, 6))
    plt.plot(t_trunc, NR_i, label='NR_i', color=colbBlue, linewidth=1.5)
    plt.plot(t_trunc, wf_i_quantiles[50], label='50% CI', linestyle='-', color=colbRed)
    plt.title('Imaginary Component Comparison (No Whitening)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    #plt.xlim(1.1285,1.13)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    filename = f"Imaginary_Component_No_Whitening_M_{M}_dL_{dL}_M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}.pdf"
    plt.savefig(os.path.join(sanity_checks_dir, filename))
    plt.close()

    # Generate plot for real components (Whitening with solve_toeplitz)
    plt.figure(figsize=(10, 6))
    plt.plot(t_trunc, whiten_NR_r, label='NR_r (whitened)', color=colbBlue, linewidth=1.5)
    plt.plot(t_trunc, wf_r_whitened[50], label='50% CI (whitened)', linestyle='-', color=colbRed)
    plt.title('Real Component Comparison (Whitened with solve_toeplitz)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude (Whitened)')
    #plt.xlim(1.1285,1.13)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    filename = f"Real_Component_Whitened_M_{M}_dL_{dL}_M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}.pdf"
    plt.savefig(os.path.join(sanity_checks_dir, filename))
    plt.close()

    # Generate plot for imaginary components (Whitening with solve_toeplitz)
    plt.figure(figsize=(10, 6))
    plt.plot(t_trunc, whiten_NR_i, label='NR_i (whitened)', color=colbBlue, linewidth=1.5)
    plt.plot(t_trunc, wf_i_whitened[50], label='50% CI (whitened)', linestyle='-', color=colbRed)
    plt.title('Imaginary Component Comparison (Whitened with solve_toeplitz)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude (Whitened)')
    #plt.xlim(1.1285,1.13)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    filename = f"Imaginary_Component_Whitened_M_{M}_dL_{dL}_M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}.pdf"
    plt.savefig(os.path.join(sanity_checks_dir, filename))
    plt.close()

    # Generate plot for real components (Toeplitz Whitening)
    plt.figure(figsize=(10, 6))
    plt.plot(t_trunc, toeplitz_whitened_NR_r, label='NR_r (Toeplitz whitened)', color=colbBlue, linewidth=1.5)
    plt.plot(t_trunc, wf_r_toeplitz_whitened[50], label='50% CI (Toeplitz whitened)', linestyle='-', color=colbRed)
    plt.title('Real Component Comparison (Toeplitz Whitening)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude (Toeplitz Whitened)')
    #plt.xlim(1.1285,1.13)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    filename = f"Real_Component_Toeplitz_Whitening_M_{M}_dL_{dL}_M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}.pdf"
    plt.savefig(os.path.join(sanity_checks_dir, filename))
    plt.close()

    # Generate plot for imaginary components (Toeplitz Whitening)
    plt.figure(figsize=(10, 6))
    plt.plot(t_trunc, toeplitz_whitened_NR_i, label='NR_i (Toeplitz whitened)', color=colbBlue, linewidth=1.5)
    plt.plot(t_trunc, wf_i_toeplitz_whitened[50], label='50% CI (Toeplitz whitened)', linestyle='-', color=colbRed)
    plt.title('Imaginary Component Comparison (Toeplitz Whitening)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude (Toeplitz Whitened)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    filename = f"Imaginary_Component_Toeplitz_Whitening_M_{M}_dL_{dL}_M_wDX_{round(window_size_DX,1)}Hz_wSX_{round(window_size_SX,1)}Hz_k_{round(k,2)}.pdf"
    plt.savefig(os.path.join(sanity_checks_dir, filename))
    plt.close()

    return

def compute_mismatch_check_TD_FD(NR_sim, results, inference_model, outdir, method, acf, N_FFT, M, dL, t_start_g, t_end_g, f_min, f_max, asd_file, window_size, k, compare_TD_FD, sanity_check_mm):

    """
    OLD VERSION. Compute the mismatch of the model with respect to NR simulations.
    """

    # File paths for saving results
    mismatch_filename = f"Mismatch_M_{M}_dL_{dL}_t_s_{round(t_start_g,1)}M_w_{round(window_size,1)}_k_{round(k,2)}_NFFT_{N_FFT}.txt"
    mismatch_filename_fd = f"Mismatch_M_{M}_dL_{dL}_t_s_{round(t_start_g,1)}M_w_{round(window_size,1)}_k_{round(k,2)}_NFFT_{N_FFT}_FD.txt"
    outFile_path = os.path.join(outdir, 'Algorithm/Mismatch', mismatch_filename)
    outFile_path_fd = os.path.join(outdir, 'Algorithm/Mismatch', mismatch_filename_fd)

    with open(outFile_path, 'w') as outFile_mismatch, open(outFile_path_fd, 'w') as outFile_mismatch_fd:
        outFile_mismatch.write('#CI\tStrain_data\tMismatch\n')
        outFile_mismatch_fd.write('#CI\tStrain_data\tFD_Mismatch\n')

    # Extract NR waveform components
    NR_r = NR_sim.NR_r_cut * (C_md * M) / dL
    NR_i = NR_sim.NR_i_cut * (C_md * M) / dL
    NR_dict = {'real': NR_r, 'imag': NR_i}
    models_re_list, models_im_list = model_component_lists(results, inference_model, method)

    for NR_quant, NR_data in NR_dict.items():
        try:
            NR_int = interpolate_waveform(t_start_g, t_end_g, M, wf_lNR=NR_data, acf=acf)
            whiten_whiten_h_NR = sl.solve_toeplitz(acf, NR_int, check_finite=False)
            h_NR_h_NR_sqrt = np.sqrt(abs(np.dot(NR_int, whiten_whiten_h_NR)))

        except Exception as e:
            print(f"Error in NR scalar product for {NR_quant}: {e}")
            continue

        # Load waveform template
        models_re_list, models_im_list = model_component_lists(results, inference_model, method)

        for perc in [5, 50, 95]:
            try:
                wf_r = np.percentile(np.array(models_re_list), [perc], axis=0)[0]
                wf_i = np.percentile(np.array(models_im_list), [perc], axis=0)[0]

                wf_r *= (C_md * M) / dL
                wf_i *= (C_md * M) / dL
                wf_quant = {'real': wf_r, 'imag': wf_i}

                wf_int = interpolate_waveform(t_start_g, t_end_g, M, wf_lNR=wf_quant[NR_quant], acf=acf)
                whiten_whiten_h_wf = sl.solve_toeplitz(acf, wf_int, check_finite=False)
                h_wf_h_wf_sqrt = np.sqrt(abs(np.dot(wf_int, whiten_whiten_h_wf)))
                h_wf_h_NR = np.dot(wf_int, whiten_whiten_h_NR)

                TD_match = h_wf_h_NR / (h_NR_h_NR_sqrt * h_wf_h_wf_sqrt)
                TD_mismatch = 1 - TD_match

                with open(outFile_path, 'a') as outFile_mismatch:
                    outFile_mismatch.write(f'{perc}\t{NR_quant}\t{TD_mismatch}\n')

                if compare_TD_FD:
                    psd   = convert_asd_to_pycbc_psd(asd_file, f_min, f_max, delta_f=2*f_max/len(acf))
                    h_TS  = TimeSeries(wf_int, delta_t=1/(2*f_max))
                    nr_TS = TimeSeries(NR_int, delta_t=1/(2*f_max))

                    FD_match_m  = float(compute_FD_match(h_TS, nr_TS, psd=psd, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)[0])
                    FD_mismatch = 1 - FD_match_m

                    with open(outFile_path_fd, 'a') as outFile_mismatch_fd: outFile_mismatch_fd.write(f'{perc}\t{NR_quant}\t{FD_mismatch}\n')

            except Exception as e:
                print(f"Error processing mismatch for {perc}% CI and {NR_quant}: {e}")
                continue

    return

def compute_mismatch_hplus_hcross(NR_sim, results, inference_model, outdir, method, acf, N_FFT, M, dL, t_start_g, f_min, f_max, asd_file, window_size_DX, window_size_SX, k, saturation_DX, saturation_SX, mismatch_print_flag, compare_TD_FD, direction=None):

    """
    Compute the mismatch of the model with respect to NR simulations.
    """

    print(f"\n* Computing mismatch for plus and cross polarizations assuming: M={M}, D_L={dL}.")

    for NR_quant, NR_data in _scaled_strain_components(NR_sim, M, dL).items():
        try:
            # Compute <NR|NR>
            whiten_whiten_h_NR, h_NR_h_NR_sqrt = _toeplitz_whitened_norm(acf, NR_data)

            if mismatch_print_flag: print(f"<NR|NR>**0.5={h_NR_h_NR_sqrt:.3f}")

        except Exception as e:
            print(f"Error in NR scalar product for {NR_quant}: {e}")
            continue

        # Load waveform template
        models_re_list, models_im_list = model_component_lists(results, inference_model, method)

        for perc in summary_percentiles:
            try:
                # Extract waveform (geometric units)
                wf_r, wf_i = _percentile_waveforms(models_re_list, models_im_list, perc, M, dL)
                wf_quant = {'real': wf_r, 'imag': wf_i}

                # Compute scalar products with h_wf
                _, h_wf_h_wf_sqrt = _toeplitz_whitened_norm(acf, wf_quant[NR_quant])
                h_wf_h_NR         = np.dot(wf_quant[NR_quant], whiten_whiten_h_NR)

                # Match computation
                TD_match    = abs(h_wf_h_NR) / (h_NR_h_NR_sqrt * h_wf_h_wf_sqrt)

                # Avoid numerical overflow correction
                TD_match    = np.minimum(1 - abs(1 - TD_match), TD_match)

                # Mismatch computation
                TD_mismatch = 1 - TD_match

                if mismatch_print_flag:
                    print(f"<h|h>**0.5={h_wf_h_wf_sqrt:.3f}")
                    print(f"<h|NR>={h_wf_h_NR:.3f}")

                if(perc==50): print(f"* Time-domain mismatch (h {NR_quant}): {TD_mismatch}")

                _record_mismatch_diagnostic(
                    outdir, "strain_components", M, dL, t_start_g, N_FFT,
                    window_size_DX, window_size_SX, k, saturation_DX, saturation_SX,
                    direction=direction, confidence_interval=perc,
                    strain_data=NR_quant, mismatch=TD_mismatch
                )

            except Exception as e:
                print(f"Error processing mismatch for {perc}% CI and {NR_quant}: {e}")
                continue

    return

def _nr_waveform_payload(label, time, real, imag, **metadata):

    payload = {
        'label': label,
        'time': np.asarray(time),
        'real': np.asarray(real),
        'imag': np.asarray(imag),
    }
    payload.update(metadata)

    return payload

def _try_read_sxs_waveform(NR_sim, extrap_order, res_level):

    try:
        time, real, imag = NR_sim.read_waveform_lm_from_SXS(extrap_order, res_level)
    except Exception:
        return None

    return _nr_waveform_payload(
        'Lev{}_N{}'.format(res_level, extrap_order),
        time,
        real,
        imag,
        res_level=res_level,
        extrap_order=extrap_order,
    )

def _try_read_rwz_waveform(NR_sim, extrap_order, res_level):

    try:
        time, real, imag = NR_sim.read_waveform_lm_from_RWZ(
            res_level, extrap_order, allow_simple_fallback=False
        )
    except Exception:
        return None

    return _nr_waveform_payload(
        'RL{}_EP{}'.format(res_level, extrap_order),
        time,
        real,
        imag,
        res_level=res_level,
        extrap_order=extrap_order,
    )

def _int_value_or_none(value):

    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _sxs_nr_comparison_pairs(NR_sim):

    pairs = []
    extrap_order = _int_value_or_none(getattr(NR_sim, 'extrap_order', None))
    if(extrap_order is None):
        return pairs

    resolution_waveforms = []
    for res_level in [6, 5, 4, 3, 2, 1]:
        waveform = _try_read_sxs_waveform(NR_sim, extrap_order, res_level)
        if(waveform is None):
            continue
        resolution_waveforms.append(waveform)
        if(len(resolution_waveforms) == 2):
            break

    extrapolation_res_level = _int_value_or_none(getattr(NR_sim, 'res_level', None))
    if(len(resolution_waveforms) == 2):
        high, low = resolution_waveforms
        pairs.append((
            'nr_resolution_Lev{}_vs_Lev{}'.format(high['res_level'], low['res_level']),
            'NR resolution',
            high,
            low,
        ))
        extrapolation_res_level = high['res_level']

    if(extrapolation_res_level is not None):
        base = _try_read_sxs_waveform(NR_sim, extrap_order, extrapolation_res_level)
        next_extrap = _try_read_sxs_waveform(NR_sim, extrap_order + 1, extrapolation_res_level)
        if(base is not None and next_extrap is not None):
            pairs.append((
                'nr_extrapolation_N{}_vs_N{}'.format(extrap_order, extrap_order + 1),
                'NR extrapolation',
                base,
                next_extrap,
            ))

    return pairs

def _rwz_nr_comparison_pairs(NR_sim):

    pairs = []
    extrap_order = _int_value_or_none(getattr(NR_sim, 'extrap_order', None))
    if(extrap_order is None):
        return pairs

    try:
        resolution_levels = NR_sim.available_RWZ_resolution_levels(extrap_order)
    except Exception:
        resolution_levels = []

    extrapolation_res_level = _int_value_or_none(getattr(NR_sim, 'res_level', None))
    if(len(resolution_levels) >= 2):
        high_res, low_res = resolution_levels[-1], resolution_levels[-2]
        high = _try_read_rwz_waveform(NR_sim, extrap_order, high_res)
        low = _try_read_rwz_waveform(NR_sim, extrap_order, low_res)
        if(high is not None and low is not None):
            pairs.append((
                'nr_resolution_RL{}_vs_RL{}'.format(high_res, low_res),
                'NR resolution',
                high,
                low,
            ))
            extrapolation_res_level = high_res

    if(extrapolation_res_level is not None):
        base = _try_read_rwz_waveform(NR_sim, extrap_order, extrapolation_res_level)
        next_extrap = _try_read_rwz_waveform(NR_sim, extrap_order + 1, extrapolation_res_level)
        if(base is not None and next_extrap is not None):
            pairs.append((
                'nr_extrapolation_EP{}_vs_EP{}'.format(extrap_order, extrap_order + 1),
                'NR extrapolation',
                base,
                next_extrap,
            ))

    return pairs

def _teukolsky_nr_comparison_pairs(NR_sim):

    res_level = _int_value_or_none(getattr(NR_sim, 'res_level', None))
    if(res_level is None):
        return []

    high = None
    low = None
    try:
        time, real, imag = NR_sim.read_waveform_lm_from_Teukolsky(res_level)
        high = _nr_waveform_payload(
            'Lev{}'.format(res_level), time, real, imag, res_level=res_level
        )
        time, real, imag = NR_sim.read_waveform_lm_from_Teukolsky(res_level - 1)
        low = _nr_waveform_payload(
            'Lev{}'.format(res_level - 1), time, real, imag, res_level=res_level - 1
        )
    except Exception:
        return []

    return [(
        'nr_resolution_Lev{}_vs_Lev{}'.format(res_level, res_level - 1),
        'NR resolution',
        high,
        low,
    )]

def _nr_comparison_pairs(NR_sim):

    cache_key = '_bayring_nr_comparison_pairs'
    cached_pairs = getattr(NR_sim, cache_key, None)
    if(cached_pairs is not None):
        return cached_pairs

    catalog = getattr(NR_sim, 'NR_catalog', None)
    if(catalog == 'SXS'):
        pairs = _sxs_nr_comparison_pairs(NR_sim)
    elif(catalog == 'RWZ-env'):
        pairs = _rwz_nr_comparison_pairs(NR_sim)
    elif(catalog == 'Teukolsky'):
        pairs = _teukolsky_nr_comparison_pairs(NR_sim)
    else:
        pairs = []

    setattr(NR_sim, cache_key, pairs)

    return pairs

def _normalise_nr_time_to_analysis_grid(time, analysis_time):

    time = np.asarray(time)
    if(len(time) == 0 or len(analysis_time) == 0):
        return time
    if(analysis_time[0] >= 0.0 and time[0] < 0.0):
        return time - time[0]

    return time

def _interpolate_nr_waveform_component(waveform, component, analysis_time):

    time = _normalise_nr_time_to_analysis_grid(waveform['time'], analysis_time)
    if(len(time) == 0):
        raise ValueError("comparison waveform has an empty time array")
    if(time[0] > analysis_time[0] or time[-1] < analysis_time[-1]):
        raise ValueError("comparison waveform does not cover the analysis interval")

    return interp1d(time, waveform[component], bounds_error=False, fill_value=0.0)(analysis_time)

def _align_nr_comparison_waveform(reference_waveform, comparison_waveform, analysis_time):

    analysis_time = np.asarray(analysis_time)
    if(len(analysis_time) < 2):
        raise ValueError("comparison waveform alignment requires at least two analysis samples")

    reference_time = _normalise_nr_time_to_analysis_grid(reference_waveform['time'], analysis_time)
    comparison_time = _normalise_nr_time_to_analysis_grid(comparison_waveform['time'], analysis_time)
    if(len(reference_time) == 0 or len(comparison_time) == 0):
        raise ValueError("comparison waveform alignment requires non-empty time arrays")

    reference_amp, reference_phi = waveform_utils.amp_phase_from_re_im(
        reference_waveform['real'], reference_waveform['imag']
    )
    comparison_amp, comparison_phi = waveform_utils.amp_phase_from_re_im(
        comparison_waveform['real'], comparison_waveform['imag']
    )

    reference_amp_interp = interp1d(reference_time, reference_amp, fill_value=0.0, bounds_error=False)
    reference_phi_interp = interp1d(reference_time, reference_phi, fill_value=0.0, bounds_error=False)
    comparison_amp_interp = interp1d(comparison_time, comparison_amp, fill_value=0.0, bounds_error=False)
    comparison_phi_interp = interp1d(comparison_time, comparison_phi, fill_value=0.0, bounds_error=False)

    t_min_mismatch = analysis_time[0]
    t_max_mismatch = analysis_time[-1]
    mask = np.logical_and(reference_time >= t_min_mismatch, reference_time <= t_max_mismatch)
    alignment_time = reference_time[mask]
    if(len(alignment_time) < 2):
        raise ValueError("comparison waveform alignment window has fewer than two samples")

    def alignment_mismatch(deltaT_deltaPhi):
        deltaT, deltaPhi = deltaT_deltaPhi[0], deltaT_deltaPhi[1]
        ref_amp = reference_amp_interp(alignment_time)
        comp_amp = comparison_amp_interp(alignment_time - deltaT)
        ref_phi = reference_phi_interp(alignment_time)
        comp_phi = comparison_phi_interp(alignment_time - deltaT)

        norm_ref = np.sum(np.abs(ref_amp)**2)
        norm_comp = np.sum(np.abs(comp_amp)**2)
        if(norm_ref == 0.0 or norm_comp == 0.0):
            return np.inf

        numerator = np.real(
            np.sum(ref_amp * comp_amp * np.exp(-1j * (ref_phi - comp_phi - deltaPhi)))
        )

        return 1.0 - numerator / np.sqrt(norm_ref * norm_comp)

    rough_deltaPhi = reference_phi_interp(t_min_mismatch) - comparison_phi_interp(t_min_mismatch)
    deltaT, deltaPhi = fmin(
        alignment_mismatch, np.array([0.0, rough_deltaPhi]),
        ftol=1e-15, disp=False
    )
    aligned_complex = comparison_amp_interp(reference_time - deltaT) * np.exp(
        1j * (comparison_phi_interp(reference_time - deltaT) + deltaPhi)
    )
    aligned_real, aligned_imag = np.real(aligned_complex), -np.imag(aligned_complex)

    aligned_waveform = dict(comparison_waveform)
    aligned_waveform['time'] = reference_time
    aligned_waveform['real'] = aligned_real
    aligned_waveform['imag'] = aligned_imag

    return aligned_waveform

def compute_nr_comparison_mismatches(NR_sim, outdir, acf, N_FFT, M, dL, t_start_g,
                                     window_size_DX, window_size_SX, k,
                                     saturation_DX, saturation_SX, direction=None):

    pairs = _nr_comparison_pairs(NR_sim)
    if(len(pairs) == 0):
        return

    analysis_time = np.asarray(NR_sim.t_NR_cut)
    scale = _physical_strain_scale(M, dL)

    for diagnostic_type, print_label, reference_waveform, comparison_waveform in pairs:
        pair_label = '{} vs {}'.format(reference_waveform['label'], comparison_waveform['label'])
        try:
            aligned_comparison_waveform = _align_nr_comparison_waveform(
                reference_waveform, comparison_waveform, analysis_time
            )
        except Exception as exc:
            print("* Skipping {} mismatch ({}): alignment failed: {}".format(print_label, pair_label, exc))
            continue
        for component in strain_components:
            try:
                reference = _interpolate_nr_waveform_component(reference_waveform, component, analysis_time) * scale
                comparison = _interpolate_nr_waveform_component(aligned_comparison_waveform, component, analysis_time) * scale
                mismatch = _time_domain_mismatch(acf, reference, comparison)
            except Exception as exc:
                print("* Skipping {} mismatch ({}, h {}): {}".format(print_label, pair_label, component, exc))
                continue

            print("* {} mismatch ({}, h {}): {}".format(print_label, pair_label, component, mismatch))
            _record_mismatch_diagnostic(
                outdir, diagnostic_type, M, dL, t_start_g, N_FFT,
                window_size_DX, window_size_SX, k, saturation_DX, saturation_SX,
                direction=direction, strain_data=component, mismatch=mismatch
            )

    return

def compute_mismatch_htot(NR_sim, results, inference_model, outdir, method, acf, N_FFT, M, dL, ra, dec, psi, t_start_g, window_size_DX, window_size_SX, k, saturation_DX, saturation_SX, direction=None):

    """
    Compute the mismatch of the model with respect to NR simulations.

    """
    print(f"* Computing mismatch for the strain assuming: M={M}, D_L={dL}, ra={ra}, dec={dec}, psi={psi}")

    # Extract NR waveform components (physical units)
    NR_dict = _scaled_strain_components(NR_sim, M, dL)

    # Compute polarizations
    resp            = AntennaResponse('H1', ra=ra, dec=dec, psi=psi, tensor=True, times=1126259462.43)
    F_plus, F_cross = resp.plus, resp.cross
    NR_data         = F_plus * NR_dict['real'] + F_cross * NR_dict['imag']

    # Compute <NR|NR>
    whiten_whiten_h_NR, h_NR_h_NR_sqrt = _toeplitz_whitened_norm(acf, NR_data)

    # Load waveform template
    models_re_list, models_im_list = model_component_lists(results, inference_model, method)

    for perc in summary_percentiles:
        # Extract waveform (geometric units)
        wf_r, wf_i = _percentile_waveforms(models_re_list, models_im_list, perc, M, dL)
        wf         = F_plus * wf_r + F_cross * wf_i

        # Compute scalar products with h_wf
        _, h_wf_h_wf_sqrt = _toeplitz_whitened_norm(acf, wf)
        h_wf_h_NR         = np.dot(wf, whiten_whiten_h_NR)

        # Match/mismatch computations
        TD_match    = h_wf_h_NR / (h_NR_h_NR_sqrt * h_wf_h_wf_sqrt)
        TD_mismatch = 1 - TD_match

        _record_mismatch_diagnostic(
            outdir, "detector_strain", M, dL, t_start_g, N_FFT,
            window_size_DX, window_size_SX, k, saturation_DX, saturation_SX,
            direction=direction, confidence_interval=perc,
            strain_data="detector", psi=psi, mismatch=TD_mismatch
        )

    return

def compute_optimal_SNR(NR_sim, results, inference_model, outdir, method, acf, N_FFT, M, dL, t_start_g, t_end_g, f_min, f_max, asd_file, window_size_DX, window_size_SX, k, saturation_DX, saturation_SX, compare_TD_FD, direction=None):
    """
    Compute the optimal SNR of the model waveform.
    """
    print(f"\n* Optimal SNR computation for plus and cross polarizations assuming: M={M}, D_L={dL}.")

    models_re_list, models_im_list = model_component_lists(results, inference_model, method)

    for NR_quant in _scaled_strain_components(NR_sim, M, dL):
        for perc in summary_percentiles:
            try:
                wf_r, wf_i = _percentile_waveforms(models_re_list, models_im_list, perc, M, dL)
                wf_int = interpolate_waveform(t_start_g, t_end_g, M, wf_lNR=wf_r if NR_quant == "real" else wf_i, acf=acf)

                optimal_SNR_TD = np.sqrt(abs(np.dot(wf_int, sl.solve_toeplitz(acf, wf_int, check_finite=False))))

                _record_mismatch_diagnostic(
                    outdir, "strain_components", M, dL, t_start_g, N_FFT,
                    window_size_DX, window_size_SX, k, saturation_DX, saturation_SX,
                    direction=direction, confidence_interval=perc,
                    strain_data=NR_quant, optimal_snr=optimal_SNR_TD
                )

                if(perc==50): print(f"* Optimal TD SNR (h {NR_quant}): {optimal_SNR_TD}")

            except Exception as e:
                print(f"Error processing optimal SNR for {perc}% CI and {NR_quant}: {e}")
                continue

    print('\n')

    return

def compute_optimal_SNR_compare_TD_FD(NR_sim, results, inference_model, outdir, method, acf, acf_tot, N_FFT, M, dL, t_start_g, t_end_g, f_min, f_max, delta_f, asd_file, window_size_DX, window_size_SX, k, saturation_DX, saturation_SX, direction=None):
    """
    Compute the optimal SNR of the model waveform.

    Parameters:
        downsampling_factor (int): The factor by which the waveform will be downsampled. Default is 10.
    """
    print("\nProcessing optimal SNR computation (with TD/FD check) for plus and cross polarizations.\n")

    models_re_list, models_im_list = model_component_lists(results, inference_model, method)

    # Loop through the real and imaginary NR data
    for NR_quant in _scaled_strain_components(NR_sim, M, dL):
        for perc in summary_percentiles:
            try:
                # Extract the percentiles of the real and imaginary parts of the model waveform
                wf_r, wf_i = _percentile_waveforms(models_re_list, models_im_list, perc, M, dL)

                # Interpolate the waveform based on the start and end times
                wf_int = interpolate_waveform(t_start_g, t_end_g, M, wf_lNR=wf_r if NR_quant == "real" else wf_i, acf=acf_tot)

                # Pad the downsampled waveform to match the length of acf_tot if necessary
                if len(wf_int) < len(acf_tot):
                    pad_width = len(acf_tot) - len(wf_int)
                    wf_int = np.pad(wf_int, (0, pad_width))

                # Time series
                T = 1/delta_f
                h_TS = TimeSeries(wf_int, delta_t=1/(2*f_max))

                h_TS.start_time = t_start_g * C_mt * M

                # Compute the optimal SNR in the frequency domain (FD)
                optimal_SNR_FD = compute_pycbc_optimal_SNR(asd_file, h_TS, len(acf_tot), f_min, f_max, delta_f)

                # Print the results for the optimal SNR in FD (and TD, but untill a certain point, or computations can be heavy)
                optimal_SNR_TD = None
                if T<0.5:
                    optimal_SNR_TD = np.sqrt(abs(np.dot(wf_int, sl.solve_toeplitz(acf_tot, wf_int, check_finite=False))))
                    print(f"Optimal TD SNR for perc {perc}, {NR_quant} part: {optimal_SNR_TD}")
                print(f"Optimal FD SNR for perc {perc}, {NR_quant} part: {optimal_SNR_FD}")
                _record_mismatch_diagnostic(
                    outdir, "strain_components", M, dL, t_start_g, N_FFT,
                    window_size_DX, window_size_SX, k, saturation_DX, saturation_SX,
                    direction=direction, confidence_interval=perc,
                    strain_data=NR_quant, optimal_snr=optimal_SNR_TD,
                    optimal_snr_fd=optimal_SNR_FD
                )

            except Exception as e:
                print(f"Error processing optimal SNR for {perc}% CI and {NR_quant}: {e}")
                continue

def compute_higher_mode_sum_mismatch(mode_products, base_parameters, t_start, n_start_times, acf, N_FFT,
                                     window_size_DX, window_size_SX, k, saturation_DX, saturation_SX,
                                     direction=None):

    base_outdir = base_parameters['I/O']['outdir']
    M, dL, ra, dec, psi = waveform_utils.extract_GW_parameters(base_parameters)
    azimuth = base_parameters['Mismatch-GW-parameters']['azimuth']
    inclinations = base_parameters['Mismatch-GW-parameters']['inclination-list']
    polarisations = base_parameters['Mismatch-GW-parameters'].get('polarisation-list', [psi])
    include_negative_m = bool(base_parameters['Mismatch-GW-parameters']['hm-include-negative-m'])

    outdir = _hm_sum_output_dir(base_outdir, t_start, n_start_times)
    diagnostic_path = None

    common_time = _common_mode_time(mode_products)
    scale = _physical_strain_scale(M, dL)
    nr_modes = {
        mode: series * scale
        for mode, series in _interpolate_mode_series(mode_products, common_time).items()
    }
    model_modes_by_percentile = {
        percentile: {
            mode: series * scale
            for mode, series in _interpolate_mode_series(mode_products, common_time, percentile).items()
        }
        for percentile in summary_percentiles
    }

    detector_responses = []
    for polarisation in polarisations:
        resp = AntennaResponse('H1', ra=ra, dec=dec, psi=polarisation, tensor=True, times=1126259462.43)
        detector_responses.append((polarisation, resp.plus, resp.cross))

    for inclination in inclinations:
        nr_by_polarisation = []
        for polarisation, F_plus, F_cross in detector_responses:
            NR_data = _project_modes_to_detector(
                nr_modes, inclination, azimuth, F_plus, F_cross, include_negative_m
            )
            whiten_whiten_h_NR, h_NR_h_NR_sqrt = _toeplitz_whitened_norm(acf, NR_data)
            if(h_NR_h_NR_sqrt == 0.0):
                continue
            nr_by_polarisation.append((polarisation, F_plus, F_cross, whiten_whiten_h_NR, h_NR_h_NR_sqrt))

        if(len(nr_by_polarisation) == 0):
            print("* Skipping HM-summed mismatch at inclination {} because the NR norm is zero for every polarisation.".format(inclination))
            continue

        for percentile in summary_percentiles:
            best_result = None
            for polarisation, F_plus, F_cross, whiten_whiten_h_NR, h_NR_h_NR_sqrt in nr_by_polarisation:
                wf = _project_modes_to_detector(
                    model_modes_by_percentile[percentile], inclination, azimuth, F_plus, F_cross, include_negative_m
                )
                _, h_wf_h_wf_sqrt = _toeplitz_whitened_norm(acf, wf)
                if(h_wf_h_wf_sqrt == 0.0):
                    continue
                h_wf_h_NR = np.dot(wf, whiten_whiten_h_NR)
                TD_match = abs(h_wf_h_NR) / (h_NR_h_NR_sqrt * h_wf_h_wf_sqrt)
                TD_match = np.minimum(1 - abs(1 - TD_match), TD_match)
                TD_mismatch = 1 - TD_match
                if(best_result is None or TD_mismatch < best_result[1]):
                    best_result = (polarisation, TD_mismatch)

            if(best_result is None):
                print("* Skipping HM-summed mismatch for percentile {} at inclination {} because the model norm is zero for every polarisation.".format(percentile, inclination))
                continue

            polarisation, TD_mismatch = best_result
            diagnostic_path = _record_mismatch_diagnostic(
                outdir, "higher_mode_sum", M, dL, t_start, N_FFT,
                window_size_DX, window_size_SX, k, saturation_DX, saturation_SX,
                direction=direction, confidence_interval=percentile,
                inclination=inclination, azimuth=azimuth, psi=polarisation,
                mismatch=TD_mismatch
            )

    if diagnostic_path is not None:
        print("* HM-summed mismatch written to `{}`.".format(diagnostic_path))
    else:
        print("* No HM-summed mismatch values were written for t-start = {} M.".format(t_start))

    return

def run_higher_mode_mismatch_scan(run_parameters_list, base_parameters):

    groups = {}
    for run_parameters in run_parameters_list:
        if not(run_parameters['I/O'].get('mode-output', False)):
            continue
        key = run_parameters['Inference']['t-start']
        groups.setdefault(key, []).append(run_parameters)

    if not(groups):
        return

    print('\n* Computing higher-mode summed mismatch diagnostics.\n')
    M, _, _, _, _ = waveform_utils.extract_GW_parameters(base_parameters)

    for t_start, group_parameters in sorted(groups.items()):
        mode_products = []
        for run_parameters in group_parameters:
            product_path = os.path.join(run_parameters['I/O']['outdir'], 'NR_sim.pkl')
            if not(os.path.exists(product_path)):
                print("* Skipping HM-summed mismatch for t-start = {} M because `{}` is missing.".format(t_start, product_path))
                mode_products = []
                break
            mode_products.append(_load_hm_mode_products(run_parameters))

        if(len(mode_products) < 2):
            continue

        common_time = _common_mode_time(mode_products)
        duration_g = common_time[-1] - common_time[0]
        t_NR_s = (common_time - common_time[0]) * M * C_mt
        t_start_s, t_end_s = 0.0, duration_g * C_mt * M
        NR_length = len(common_time)

        try:
            apply_window, _, _, C1_flag, mismatch_print_flag, mismatch_section_plot_flag = \
                waveform_utils.extract_flags(base_parameters['Flags'])

            (f_min, f_max, dt, _, N_points, n_FFT_points, asd_path,
             n_iterations_C1, window_sizes_DX, window_sizes_SX,
             steepness_values, saturation_DX_values, saturation_SX_values,
             direction) = waveform_utils.extract_and_compute_psd_parameters(
                base_parameters['Mismatch-PSD-settings'], mismatch_print_flag
            )

            n_fft_values = [N_points] if n_FFT_points == 1 else list(
                map(int, np.logspace(np.log10(NR_length), np.log10(2 * N_points), n_FFT_points))
            )

            grid = product(
                n_fft_values,
                window_sizes_DX,
                window_sizes_SX,
                steepness_values,
                saturation_DX_values,
                saturation_SX_values,
            )

            for N_fft, window_size_DX, window_size_SX, k, saturation_DX, saturation_SX in grid:
                if (t_end_s - t_start_s) > 1 / (f_min + window_size_DX) and direction != 'above':
                    print("Please provide (t_end-t_start) < 1/(f_min+window_DX) for HM-summed mismatch.")
                    print("Forbidden frequency:", f_min + window_size_DX)
                    continue

                window_args = (window_size_DX, window_size_SX, k, saturation_DX, saturation_SX)
                if apply_window == 1:
                    PSD_smoothed, ACF_smoothed = waveform_utils.acf_from_asd_with_smoothing(
                        asd_path, f_min, f_max, N_fft, *window_args,
                        direction, C1_flag, n_iterations_C1
                    )
                else:
                    PSD_smoothed, ACF_smoothed = waveform_utils.acf_from_asd_no_window_at_edges(
                        asd_path, f_min, f_max, N_fft
                    )

                t_ACF = np.linspace(0, N_fft * dt, len(ACF_smoothed))
                ACF_truncated_NR = truncate_and_interpolate_acf(
                    t_ACF, ACF_smoothed, M, 0.0, duration_g, t_NR_s, mismatch_print_flag
                )
                compute_higher_mode_sum_mismatch(
                    mode_products, base_parameters, t_start, len(groups), ACF_truncated_NR, N_fft, *window_args,
                    direction=direction
                )

        except Exception as e:
            print("* HM-summed mismatch failed for t-start = {} M: {}".format(t_start, e))

    return

def plot_NR_vs_model(NR_sim, template, metadata, results, inference_model, outdir, method, tail_flag, extract_damping_time_flag):

    """

    Plot the NR waveform against the model waveform.

    Parameters
    ----------

    NR_sim : NR_sim
        NR simulation object.

    template : template
        Template object.

    metadata : dict
        Dictionary containing the metadata.

    results : dict
        Dictionary containing the results object.

    inference_model : inference_model
        Nested sampling model object.

    outdir : string
        Output directory.

    method : string
        Method used to fit the waveform.

    Returns
    -------

    Nothing, but plots the simulation/model comparison and saves the figure.

    """

    init_plotting()

    #take NR elements
    NR_r, NR_i, NR_r_err, NR_i_err, NR_amp, NR_f, t_NR, t_peak                                                = NR_sim.NR_r, NR_sim.NR_i, np.real(NR_sim.NR_err_cmplx), np.imag(NR_sim.NR_err_cmplx), NR_sim.NR_amp, NR_sim.NR_freq, NR_sim.t_NR, NR_sim.t_peak
    t_cut, tM_start, tM_end, NR_r_cut, NR_i_cut, NR_r_err_cut, NR_i_err_cut, NR_amp_cut, NR_phi_cut, NR_f_cut = NR_sim.t_NR_cut, NR_sim.tM_start, NR_sim.tM_end, NR_sim.NR_r_cut, NR_sim.NR_i_cut, np.real(NR_sim.NR_cpx_err_cut), np.imag(NR_sim.NR_cpx_err_cut), NR_sim.NR_amp_cut, NR_sim.NR_phi_cut, NR_sim.NR_freq_cut

    wf_data_type = NR_sim.waveform_type

    l,m = NR_sim.l, NR_sim.m

    f_rd_fundamental    = template.qnm_cached[(2,l,m,0)]['f']
    tau_rd_fundamental  = template.qnm_cached[(2,l,m,0)]['tau']

    plot_overtones_flag = 0
    f_rd_overtones      = {}
    for n in [1,3,7,9]:
        omega_n, _, _     = qnm.modes_cache(s=-2,l=l,m=m,n=n)(a=np.abs(metadata['af']))
        f_rd_overtones[n] = (np.real(omega_n) / metadata['Mf']) * (1./twopi)

    try:
        m1, m2, chi1, chi2 = metadata['m1'], metadata['m2'], metadata['chi1'], metadata['chi2'],
        f_peak             = utils.F_mrg_Nagar(m1, m2, chi1, chi2, geom=1)
    except:
        f_peak             = None

    # get the amplitude at the time close to the peak
    amp_peak = NR_amp[np.argmin(np.abs(t_NR - t_peak))]

    lw_small        = 0.5
    lw_medium       = 1.2
    lw_std          = 1.8
    lw_large        = 2.2

    color_NR        = 'k'
    color_model     = '#cc0033'
    color_t_start   = 'mediumseagreen' #'#990066', '#cc0033', '#ff0000'
    color_t_peak    = 'royalblue'
    color_f_overt   = 'darkorange'

    alpha_std       = 1.0
    alpha_med       = 0.8

    ls_t            = '--'
    ls_f            = '--'

    if(tail_flag) :
        fontsize_legend = 20
        fontsize_labels = 25
        color_f_ring    = 'royalblue'
    else:
        fontsize_legend = 18
        fontsize_labels = 23
        color_f_ring    = 'forestgreen'

    if(not(tail_flag) and not(wf_data_type=='psi4') and (NR_sim.NR_catalog=='SXS' or NR_sim.NR_catalog=='RIT')): tM_end = 80
    if(wf_data_type=='psi4'):
        tM_end = 120
        label_data = '\psi_{4,%s%s}'%(l,m)
    else:
        label_data = 'h_{%s%s}'%(l,m)

    ########################
    # Waveforms comparison #
    ########################

    if(tail_flag):
        f   = plt.figure(figsize=(8,12))
        ax2 = plt.subplot(2,1,1)
        ax4 = plt.subplot(2,1,2)

        rescale = 1.4
    else:
        f   = plt.figure(figsize=(12,8))
        ax1 = plt.subplot(2,2,1)
        ax2 = plt.subplot(2,2,2)
        ax3 = plt.subplot(2,2,3)
        ax4 = plt.subplot(2,2,4)

        ax1.set_xlim([-10, tM_end])
        ax3.set_xlim(ax1.get_xlim())

        rescale = 1.0

    ax2.set_xlim(-10, tM_end)
    ax4.set_xlim(ax2.get_xlim())

    ################
    # Plot NR data #
    ################

    if not(tail_flag):
        ax1.plot(t_NR - t_peak, NR_r,                                                      c=color_NR,      lw=lw_std,    alpha=alpha_std, ls='-' )
        ax1.axvline(tM_start,                                                              c=color_t_start, lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax1.axvline(0.0, label=r'$t_{\rm peak}$',                                          c=color_t_peak,  lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax1.set_ylabel(r'$\mathrm{Re[%s]}$'%(label_data), fontsize=fontsize_labels)

        ax3.plot(t_NR - t_peak, NR_i,                                                      c=color_NR,      lw=lw_std,    alpha=alpha_std, ls='-' )
        ax3.axvline(tM_start, label=r'$t_{\rm start} = t_{\rm peak} \, + %d \mathrm{M}$'%tM_start, c=color_t_start, lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax3.axvline(0.0,                                                                   c=color_t_peak,  lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax3.set_ylabel(r'$\mathrm{Im[%s]}$'%(label_data), fontsize=fontsize_labels)
        ax3.set_xlabel(r'$t - t_{peak} \, [\mathrm{M}]$', fontsize=fontsize_labels)

    if not(tail_flag):
        if(extract_damping_time_flag):
            ax2.semilogy(t_NR - t_peak, NR_amp*np.e**((t_NR - t_peak)/tau_rd_fundamental), label=r'$\mathrm{NR}$', c=color_NR,      lw=lw_std,    alpha=alpha_std, ls='-' )
        else:
            ax2.semilogy(t_NR - t_peak, NR_amp                                           , label=r'$\mathrm{NR}$', c=color_NR,      lw=lw_std,    alpha=alpha_std, ls='-' )
    else             :
        ax2.semilogy(    t_NR - t_peak, NR_amp                                           , label=r'$\mathrm{NR}$', c=color_NR,      lw=lw_std,    alpha=alpha_std, ls='-' )
    ax2.axvline(tM_start,                                                                                          c=color_t_start, lw=lw_std,    alpha=alpha_std, ls=ls_t)
    if(not(tail_flag)): ax2.axvline(0.0,                                                                           c=color_t_peak,  lw=lw_std,    alpha=alpha_std, ls=ls_t)

    if(not(tail_flag) and (NR_sim.NR_catalog=='SXS' or NR_sim.NR_catalog=='RIT')):
        if(extract_damping_time_flag):
            ax2.set_ylim([1e-1*amp_peak, 10*amp_peak])
        else:
            ax2.set_ylim([1e-3*amp_peak, 2*amp_peak ])
    elif(  tail_flag  and (NR_sim.NR_catalog=='SXS' or NR_sim.NR_catalog=='RIT')):
        ax2.set_ylim(    [2*1e-4, 2*np.max(NR_amp)])

    ax2.set_xlabel(r'$\mathrm{t - t_{peak} \, [M}]$', fontsize=fontsize_labels)

    ax4.plot(t_NR - t_peak, NR_f,                                                          c=color_NR,      lw=lw_std,     alpha=alpha_std, ls='-' )
    ax4.axhline(f_rd_fundamental, label=r'$\mathit{f_{%d%d0}}$'%(l,m),                     c=color_f_ring,  lw=lw_std,     alpha=alpha_std, ls=ls_f)
    if(plot_overtones_flag):
        for n in [1,3,9]:
            if(n==1): leg = r'$\mathit{f_{%d%dn}}$'%(l,m)
            else    : leg = None
            ax4.axhline(f_rd_overtones[n], label=leg,         c=color_f_overt, lw=lw_std*0.4, alpha=alpha_std, ls=ls_f)

    if(tail_flag):
        ax4.axhline(0.0,      label=r'$\mathit{f_{\rm tail}}$',                            c=color_model,   lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax4.axvline(tM_start, label=r'$\mathrm{t_{start} = t_{peak} \, + %d M}$'%tM_start, c=color_t_start, lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax4.axvline(0.0,                                                                   c=color_t_peak,  lw=lw_std,    alpha=alpha_std, ls=ls_t)
    else         :
        ax4.axvline(0.0,                                                                   c=color_t_peak,  lw=lw_std,    alpha=alpha_std, ls=ls_t)
    ax4.set_xlabel(r'$t - t_{peak} \, [\mathrm{M}]$'    , fontsize=fontsize_labels)

    # Find the index of zero
    t_peak_idx = np.argmin(np.abs(t_NR - t_peak))

    if not(tail_flag):
        try   : ax4.set_ylim([-1.5*NR_f[t_peak_idx], 3.5*NR_f[t_peak_idx]])
        except: pass
    else:
        ax4.set_ylim([-0.08, 0.28])

    ################################
    # Plot waveform reconstruction #
    ################################

    if not(inference_model==None):

        models_re_list, models_im_list = model_component_lists(results, inference_model, method)

        for perc in [50, 5, 95]:

            wf_r = np.percentile(np.array(models_re_list),[perc], axis=0)[0]
            wf_i = np.percentile(np.array(models_im_list),[perc], axis=0)[0]

            wf_amp, wf_phi = waveform_utils.amp_phase_from_re_im(wf_r, wf_i)
            wf_f           = np.gradient(wf_phi, t_cut)/(twopi)

            if(perc==50):
                if not(tail_flag):
                    ax1.plot(t_cut - t_peak, wf_r,                                               c=color_model, lw=lw_large*rescale, alpha=alpha_std, ls='-')
                    ax3.plot(t_cut - t_peak, wf_i,                                               c=color_model, lw=lw_large*rescale, alpha=alpha_std, ls='-')
                    if(extract_damping_time_flag):
                        ax2.semilogy(t_cut - t_peak, wf_amp*np.e**((t_cut - t_peak)/tau_rd_fundamental), label=r'$\mathrm{%s}$'%(template.wf_model), c=color_model, lw=lw_large*rescale, alpha=alpha_std, ls='-' )
                    else:
                        ax2.semilogy(t_cut - t_peak, wf_amp                                            , label=r'$\mathrm{%s}$'%(template.wf_model), c=color_model, lw=lw_large*rescale, alpha=alpha_std, ls='-' )
                else:
                    ax2.semilogy(    t_cut - t_peak, wf_amp                                            , label=r'$\mathrm{%s}$'%(template.wf_model), c=color_model, lw=lw_large*rescale, alpha=alpha_std, ls='-' )
                ax4.plot(            t_cut - t_peak, wf_f,                                                                                           c=color_model, lw=lw_large*rescale, alpha=alpha_std, ls='-' )
            else:
                if not(tail_flag):
                    ax1.plot(        t_cut - t_peak, wf_r                                                                                          , c=color_model, lw=lw_std,           alpha=alpha_med, ls='--')
                    ax3.plot(        t_cut - t_peak, wf_i                                                                                          , c=color_model, lw=lw_std,           alpha=alpha_med, ls='--')
                    if(extract_damping_time_flag):
                        ax2.semilogy(t_cut - t_peak, wf_amp*np.e**((t_cut - t_peak)/tau_rd_fundamental)                                            , c=color_model, lw=lw_std,           alpha=alpha_med, ls='--')
                    else:
                        ax2.semilogy(t_cut - t_peak, wf_amp                                                                                        , c=color_model, lw=lw_std,           alpha=alpha_med, ls='--')
                else:
                    ax2.semilogy(    t_cut - t_peak, wf_amp                                                                                        , c=color_model, lw=lw_std,           alpha=alpha_med, ls='--')
                ax4.plot(            t_cut - t_peak, wf_f                                                                                          , c=color_model, lw=lw_std,           alpha=alpha_med, ls='--')


        if(tail_flag):
            # Plot QNM waveform reconstruction
            qnm_samples = [_sample_with_suppressed_tail(sample, template) for sample in waveform_parameter_samples(results, method)]
            models_re_list, models_im_list = _model_component_lists_from_samples(qnm_samples, inference_model)

            for perc in [50, 5, 95]:

                wf_r = np.percentile(np.array(models_re_list),[perc], axis=0)[0]
                wf_i = np.percentile(np.array(models_im_list),[perc], axis=0)[0]

                wf_amp, wf_phi = waveform_utils.amp_phase_from_re_im(wf_r, wf_i)
                wf_f           = np.gradient(wf_phi, t_cut)/(twopi)

                if(perc==50):
                    ax2.semilogy(t_cut - t_peak, wf_amp, label=r'$\mathrm{%s \,\, QNMs}$'%(template.wf_model), c='royalblue', lw=lw_large*1.4, alpha=alpha_std, ls='-' )
                    ax4.plot(    t_cut - t_peak, wf_f,                                                         c='royalblue', lw=lw_large*1.4, alpha=alpha_std, ls='-' )
                else:
                    ax2.semilogy(t_cut - t_peak, wf_amp,                                                       c='royalblue', lw=lw_std,       alpha=alpha_med, ls='--' )
                    ax4.plot(    t_cut - t_peak, wf_f,                                                         c='royalblue', lw=lw_std,       alpha=alpha_med, ls='--' )

    if not(tail_flag):
        ax1.set_ylabel(r'$\mathit{Re(%s)}$'%(label_data)                           , fontsize=fontsize_labels*rescale)
        ax3.set_ylabel(r'$\mathit{Im(%s)}$'%(label_data)                           , fontsize=fontsize_labels*rescale)
    ax2.set_ylabel(    r'$\mathit{A_{%d%d}(t)} \cdot e^{t/\tau_{%d%d0}}$'%(l,m,l,m), fontsize=fontsize_labels*rescale)
    ax4.set_ylabel(    r'$\mathit{f_{%d%d}\,(t)}$'%(l,m)                           , fontsize=fontsize_labels*rescale)

    plt.rcParams['legend.frameon'] = True

    ax2.legend(    loc='best', fontsize=fontsize_legend, shadow=True)
    ax4.legend(    loc='best', fontsize=fontsize_legend, shadow=True)

    if not(tail_flag):
        ax1.legend(loc='best', fontsize=fontsize_legend, shadow=True)
        ax3.legend(loc='best', fontsize=fontsize_legend, shadow=True)
        ax1.set_xlim(ax3.get_xlim())
        ax1.set_xticklabels([])
        plt.suptitle('{}-{}'.format(NR_sim.NR_catalog, NR_sim.NR_ID), size=28)

    ax2.set_xlim(ax4.get_xlim())
    ax2.set_xticklabels([])
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.subplots_adjust(hspace=0, wspace=0.27)
    if(tail_flag): leg_name_tail = '_tail'
    else         : leg_name_tail = ''
    plt.savefig(os.path.join(outdir, f'Plots/Comparisons/Waveform_reconstruction{leg_name_tail}.pdf'), bbox_inches='tight')

    if (tail_flag): plt.rcParams['legend.frameon'] = False

    if (inference_model==None): return

    ############################
    # Residuals reconstruction #
    ############################

    f   = plt.figure(figsize=(12,8))
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)
    ax3 = plt.subplot(2,2,3)
    ax4 = plt.subplot(2,2,4)

    ax1.set_xlim([tM_start, tM_end])
    ax2.set_xlim(ax1.get_xlim())
    ax3.set_xlim(ax1.get_xlim())
    ax4.set_xlim(ax1.get_xlim())

    ax1.errorbar(t_cut - t_peak, np.zeros(len(NR_r_cut)), yerr=np.array(NR_r_err_cut), label=r'$\mathrm{NR error}$', c=color_NR, lw=lw_small, alpha=alpha_std, ls='-', capsize=0.15)
    ax3.errorbar(t_cut - t_peak, np.zeros(len(NR_i_cut)), yerr=np.array(NR_i_err_cut),                               c=color_NR, lw=lw_small, alpha=alpha_std, ls='-', capsize=0.15)

    for perc in [50, 5, 95]:

        wf_r = np.percentile(np.array(models_re_list),[perc], axis=0)[0]
        wf_i = np.percentile(np.array(models_im_list),[perc], axis=0)[0]
        wf_amp, wf_phi = waveform_utils.amp_phase_from_re_im(wf_r, wf_i)
        wf_f           = np.gradient(wf_phi, t_cut)/(twopi)

        if(perc==50):
            ax1.plot(t_cut - t_peak, wf_r   - NR_r_cut  ,                                                  c=color_model, lw=lw_large, alpha=alpha_std, ls='-' )
            ax2.plot(t_cut - t_peak, wf_amp - NR_amp_cut,                                                  c=color_model, lw=lw_large, alpha=alpha_std, ls='-' )
            ax3.plot(t_cut - t_peak, wf_i   - NR_i_cut  , label=r'$\mathrm{%s - NR}$'%(template.wf_model), c=color_model, lw=lw_large, alpha=alpha_std, ls='-' )
            ax4.plot(t_cut - t_peak, wf_f   - NR_f_cut  ,                                                  c=color_model, lw=lw_large, alpha=alpha_std, ls='-' )
        else:
            ax1.plot(t_cut - t_peak, wf_r   - NR_r_cut  ,                                                  c=color_model, lw=lw_std, alpha=alpha_med, ls='--')
            ax2.plot(t_cut - t_peak, wf_amp - NR_amp_cut,                                                  c=color_model, lw=lw_std, alpha=alpha_med, ls='--')
            ax3.plot(t_cut - t_peak, wf_i   - NR_i_cut  ,                                                  c=color_model, lw=lw_std, alpha=alpha_med, ls='--')
            ax4.plot(t_cut - t_peak, wf_f   - NR_f_cut  ,                                                  c=color_model, lw=lw_std, alpha=alpha_med, ls='--')
    ax1.legend(loc='best', fontsize=fontsize_legend)
    ax3.legend(loc='best', fontsize=fontsize_legend)

    ax1.set_ylabel(r'$\mathit{Re(%s)}$'%(label_data), fontsize=fontsize_labels)
    ax2.set_ylabel(r'$\mathit{A(t)}$'               , fontsize=fontsize_labels)
    ax3.set_ylabel(r'$\mathit{Im(%s)}$'%(label_data), fontsize=fontsize_labels)
    ax4.set_ylabel(r'$\mathit{f\,(t)}$'             , fontsize=fontsize_labels)

    ax3.set_xlabel(r'$t - t_{peak} \, [\mathrm{M}]$', fontsize=fontsize_labels)
    ax4.set_xlabel(r'$t - t_{peak} \, [\mathrm{M}]$', fontsize=fontsize_labels)

    ax3.set_xlim(ax1.get_xlim())
    ax4.set_xlim(ax2.get_xlim())
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    plt.suptitle('{}-{} residuals'.format(NR_sim.NR_catalog, NR_sim.NR_ID), size=28)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.subplots_adjust(hspace=0, wspace=0.3)

    plt.savefig(os.path.join(outdir, 'Plots/Comparisons/Residuals_reconstruction.pdf'), bbox_inches='tight')

    # Decay rate
    if(tail_flag):

        plt.figure(figsize=(6,6))

        log_t_NR         = np.log(t_NR  - t_peak)
        log_t_cut        = np.log(t_cut - t_peak)

        log_A_NR         = np.log(NR_amp)
        dlog_A_NR_dlog_t = utils.diff1(log_t_NR, log_A_NR)

        models_re_list = [np.real(np.array(inference_model.model(p))) for p in results]
        models_im_list = [np.imag(np.array(inference_model.model(p))) for p in results]

        for perc in [50, 5, 95]:
            wf_r = np.percentile(np.array(models_re_list),[perc], axis=0)[0]
            wf_i = np.percentile(np.array(models_im_list),[perc], axis=0)[0]

            wf_amp, _ = waveform_utils.amp_phase_from_re_im(wf_r, wf_i)

            log_A_wf         = np.log(wf_amp)
            dlog_A_wf_dlog_t = utils.diff1(log_t_cut, log_A_wf)

            plt.plot(t_cut - t_peak, dlog_A_wf_dlog_t, c=color_model, lw=lw_std, alpha=alpha_med, ls='-')

        plt.axhline(0.0, c='k', ls='--', lw=0.7)
        plt.axhline(-1.0, c='mediumseagreen', ls='--',  label='Okuzumi+',  lw=1.2)
        plt.axhline(-1.3, c='crimson',        ls='--', label='Albanesi+', lw=1.7)
        plt.plot(    t_NR  - t_peak, dlog_A_NR_dlog_t, c=color_NR,    lw=lw_std, alpha=alpha_med, ls='-')
        plt.xlim([75, 100])
        plt.ylim([-3.4, 1.5])
        plt.xlabel(r'$t - t_{peak} \, [\mathrm{M}]$',                   fontsize=fontsize_labels)
        plt.ylabel(r'$\mathrm{p}$', fontsize=fontsize_labels)
        plt.legend(loc='best', fontsize=fontsize_labels*0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'Plots/Comparisons/Decay_rate.pdf'), bbox_inches='tight')

    return

def _init_fancy_plotting():

    plt.rcParams['figure.max_open_warning'] = 0
    plt.rcParams['mathtext.fontset']        = 'stix'
    plt.rcParams['font.family']             = 'STIXGeneral'
    plt.rcParams['font.size']               = 17
    plt.rcParams['axes.linewidth']          = 1
    plt.rcParams['axes.labelsize']          = 20
    plt.rcParams['axes.titlesize']          = 1.3*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize']         = 15
    plt.rcParams['xtick.labelsize']         = 15
    plt.rcParams['ytick.labelsize']         = 15
    plt.rcParams['xtick.major.size']        = 3
    plt.rcParams['xtick.minor.size']        = 3
    plt.rcParams['xtick.major.width']       = 1
    plt.rcParams['xtick.minor.width']       = 1
    plt.rcParams['ytick.major.size']        = 3
    plt.rcParams['ytick.minor.size']        = 3
    plt.rcParams['ytick.major.width']       = 1
    plt.rcParams['ytick.minor.width']       = 1
    plt.rcParams['legend.frameon']          = False
    plt.rcParams['contour.negative_linestyle'] = 'solid'

def _style_fancy_axes(axes):

    for ax in np.ravel(axes):
        ax.grid(True, which='major', color='0.85', lw=0.6, alpha=0.75)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out')

def _as_1d_float_array(values):

    return np.asarray(values, dtype=float).reshape(-1)

def _fancy_data_label(NR_sim):

    l, m = NR_sim.l, NR_sim.m
    if(NR_sim.waveform_type=='psi4'):
        return r'\psi_{4,%d%d}'%(l,m)
    return r'h_{%d%d}'%(l,m)

def _fancy_tM_end(NR_sim, tM_end, tail_flag):

    if(not(tail_flag) and not(NR_sim.waveform_type=='psi4') and (NR_sim.NR_catalog=='SXS' or NR_sim.NR_catalog=='RIT')):
        return 80
    if(NR_sim.waveform_type=='psi4'):
        return 120
    return tM_end

def _model_waveform_quantiles(models_re_list, models_im_list, t_cut):

    t_cut    = _as_1d_float_array(t_cut)
    model_re = np.asarray([_as_1d_float_array(model_re) for model_re in models_re_list], dtype=float)
    model_im = np.asarray([_as_1d_float_array(model_im) for model_im in models_im_list], dtype=float)

    model_amp, model_phi = waveform_utils.amp_phase_from_re_im(model_re, model_im)
    model_f              = np.gradient(model_phi, t_cut, axis=-1)/(twopi)

    quantiles = {}
    for perc in [5, 50, 95]:
        quantiles[perc] = {
            'real': np.percentile(model_re , perc, axis=0),
            'imag': np.percentile(model_im , perc, axis=0),
            'amp' : np.percentile(model_amp, perc, axis=0),
            'freq': np.percentile(model_f  , perc, axis=0),
        }

    return quantiles

def _plot_series_with_band(ax, x, median, lower=None, upper=None, color='firebrick',
                           label=None, lw=1.8, alpha=0.18, linestyle='-', semilogy=False):

    x      = _as_1d_float_array(x)
    median = _as_1d_float_array(median)

    if(semilogy):
        ax.semilogy(x, median, c=color, lw=lw, label=label, ls=linestyle)
    else:
        ax.plot(x, median, c=color, lw=lw, label=label, ls=linestyle)

    if(lower is None or upper is None):
        return

    lower = _as_1d_float_array(lower)
    upper = _as_1d_float_array(upper)
    band_low  = np.minimum(lower, upper)
    band_high = np.maximum(lower, upper)

    if(semilogy):
        mask = (band_low > 0.0) & (band_high > 0.0) & np.isfinite(band_low) & np.isfinite(band_high)
        if(np.any(mask)):
            ax.fill_between(x[mask], band_low[mask], band_high[mask], color=color, alpha=alpha, lw=0)
    else:
        ax.fill_between(x, band_low, band_high, color=color, alpha=alpha, lw=0)

def _copy_parameter_sample(sample):

    if(isinstance(sample, dict)):
        return sample.copy()
    if(hasattr(sample, 'to_dict')):
        return sample.to_dict()
    if(hasattr(sample, 'dtype') and sample.dtype.names is not None):
        return {name: sample[name] for name in sample.dtype.names}
    try:
        return sample.copy()
    except AttributeError:
        return sample

def _sample_with_suppressed_tail(sample, template):

    sample_qnm = _copy_parameter_sample(sample)
    tail_modes = getattr(template, 'tail_modes', None) or [(2,2), (3,2)]

    for mode in tail_modes:
        try:
            l_ring, m_ring = mode[:2]
        except TypeError:
            continue
        try:
            sample_qnm['ln_A_tail_{}{}'.format(l_ring, m_ring)] = np.log(1e-32)
        except (TypeError, ValueError, IndexError):
            pass

    return sample_qnm

def _model_component_lists_from_samples(samples, inference_model):

    models_re_list = [np.real(np.array(inference_model.model(p))) for p in samples]
    models_im_list = [np.imag(np.array(inference_model.model(p))) for p in samples]

    return models_re_list, models_im_list

def _scaled_amplitude(amp, t_rel, tau_rd_fundamental, tail_flag, extract_damping_time_flag):

    if(not(tail_flag) and extract_damping_time_flag):
        return amp*np.e**(t_rel/tau_rd_fundamental)
    return amp

def plot_fancy_residual(NR_sim, template, metadata, results, inference_model, outdir, method, tail_flag=False):

    """

    Plot model residuals against the NR error with the same content as the
    standard residual plot, but using filled uncertainty bands.

    """

    if(inference_model is None):
        return

    _init_fancy_plotting()

    t_peak = float(NR_sim.t_peak)
    t_cut, tM_start, tM_end = _as_1d_float_array(NR_sim.t_NR_cut), float(NR_sim.tM_start), float(NR_sim.tM_end)
    NR_r_cut, NR_i_cut      = _as_1d_float_array(NR_sim.NR_r_cut), _as_1d_float_array(NR_sim.NR_i_cut)
    NR_r_err_cut            = _as_1d_float_array(np.real(NR_sim.NR_cpx_err_cut))
    NR_i_err_cut            = _as_1d_float_array(np.imag(NR_sim.NR_cpx_err_cut))
    NR_amp_cut, NR_f_cut    = _as_1d_float_array(NR_sim.NR_amp_cut), _as_1d_float_array(NR_sim.NR_freq_cut)
    t_NR, NR_amp            = _as_1d_float_array(NR_sim.t_NR), _as_1d_float_array(NR_sim.NR_amp)
    l, m                    = NR_sim.l, NR_sim.m

    tM_end     = _fancy_tM_end(NR_sim, tM_end, tail_flag)
    label_data = _fancy_data_label(NR_sim)
    x_cut      = t_cut - t_peak

    models_re_list, models_im_list = model_component_lists(results, inference_model, method)
    quantiles = _model_waveform_quantiles(models_re_list, models_im_list, t_cut)
    has_band  = len(models_re_list) > 1

    color_model = '#cc0033'
    color_error = '0.25'
    lw_std      = 1.8

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(11.5, 7.0), sharex='col')
    ax1, ax2 = axs[0]
    ax3, ax4 = axs[1]
    _style_fancy_axes(axs)

    for ax in np.ravel(axs):
        ax.axhline(0.0, c='0.2', lw=0.8, alpha=0.65, ls=':')
        ax.set_xlim([tM_start, tM_end])

    ax1.fill_between(x_cut, -NR_r_err_cut, NR_r_err_cut,
                     color=color_error, alpha=0.16, lw=0, label=r'$\mathrm{NR\ error}$')
    ax3.fill_between(x_cut, -NR_i_err_cut, NR_i_err_cut,
                     color=color_error, alpha=0.16, lw=0)

    _plot_series_with_band(
        ax1, x_cut,
        quantiles[50]['real'] - NR_r_cut,
        quantiles[5]['real']  - NR_r_cut if has_band else None,
        quantiles[95]['real'] - NR_r_cut if has_band else None,
        color=color_model, label=r'$\mathrm{%s - NR}$'%(template.wf_model), lw=lw_std
    )
    _plot_series_with_band(
        ax2, x_cut,
        quantiles[50]['amp'] - NR_amp_cut,
        quantiles[5]['amp']  - NR_amp_cut if has_band else None,
        quantiles[95]['amp'] - NR_amp_cut if has_band else None,
        color=color_model, lw=lw_std
    )
    _plot_series_with_band(
        ax3, x_cut,
        quantiles[50]['imag'] - NR_i_cut,
        quantiles[5]['imag']  - NR_i_cut if has_band else None,
        quantiles[95]['imag'] - NR_i_cut if has_band else None,
        color=color_model, lw=lw_std
    )
    _plot_series_with_band(
        ax4, x_cut,
        quantiles[50]['freq'] - NR_f_cut,
        quantiles[5]['freq']  - NR_f_cut if has_band else None,
        quantiles[95]['freq'] - NR_f_cut if has_band else None,
        color=color_model, lw=lw_std
    )

    ax1.set_ylabel(r'$\Delta \mathrm{Re[%s]}$'%(label_data))
    ax2.set_ylabel(r'$\Delta A_{%d%d}(t)$'%(l,m))
    ax3.set_ylabel(r'$\Delta \mathrm{Im[%s]}$'%(label_data))
    ax4.set_ylabel(r'$\Delta f_{%d%d}(t)$'%(l,m))
    ax3.set_xlabel(r'$t - t_{peak} \, [\mathrm{M}]$')
    ax4.set_xlabel(r'$t - t_{peak} \, [\mathrm{M}]$')

    for ax in [ax1, ax3]:
        handles, labels = ax.get_legend_handles_labels()
        if(len(handles)>0):
            ax.legend(loc='best')
    fig.suptitle('{}-{} residuals'.format(NR_sim.NR_catalog, NR_sim.NR_ID), size=26)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.subplots_adjust(hspace=0.05, wspace=0.37)

    leg_name_tail = '_tail' if tail_flag else ''
    fig.savefig(os.path.join(outdir, f'Plots/Comparisons/Residuals_reconstruction{leg_name_tail}.pdf'), bbox_inches='tight')
    plt.close(fig)

    if(tail_flag):
        positive_NR  = (t_NR  - t_peak) > 0
        positive_cut = (t_cut - t_peak) > 0

        if(np.any(positive_NR) and np.any(positive_cut)):
            fig_decay, ax_decay = plt.subplots(figsize=(6.0, 5.2))
            _style_fancy_axes([ax_decay])

            log_t_NR         = np.log(t_NR[positive_NR] - t_peak)
            log_t_cut        = np.log(t_cut[positive_cut] - t_peak)
            log_A_NR         = np.log(np.clip(NR_amp[positive_NR], np.finfo(float).tiny, None))
            dlog_A_NR_dlog_t = utils.diff1(log_t_NR, log_A_NR)

            def decay_rate(amp):
                log_A_wf = np.log(np.clip(amp[positive_cut], np.finfo(float).tiny, None))
                return utils.diff1(log_t_cut, log_A_wf)

            p_mid = decay_rate(quantiles[50]['amp'])
            ax_decay.plot(t_cut[positive_cut] - t_peak, p_mid, c=color_model, lw=lw_std, label=r'$\mathrm{%s}$'%(template.wf_model))

            if(has_band):
                p_low  = decay_rate(quantiles[5]['amp'])
                p_high = decay_rate(quantiles[95]['amp'])
                ax_decay.fill_between(t_cut[positive_cut] - t_peak, np.minimum(p_low, p_high), np.maximum(p_low, p_high),
                                      color=color_model, alpha=0.18, lw=0)

            ax_decay.axhline(0.0 , c='k',              ls='--', lw=0.7)
            ax_decay.axhline(-1.0, c='mediumseagreen', ls='--', label='Okuzumi+',  lw=1.2)
            ax_decay.axhline(-1.3, c='crimson',        ls='--', label='Albanesi+', lw=1.7)
            ax_decay.plot(t_NR[positive_NR] - t_peak, dlog_A_NR_dlog_t, c='k', lw=1.5, alpha=0.8, label=r'$\mathrm{NR}$')
            ax_decay.set_xlim([75, 100])
            ax_decay.set_ylim([-3.4, 1.5])
            ax_decay.set_xlabel(r'$t - t_{peak} \, [\mathrm{M}]$')
            ax_decay.set_ylabel(r'$\mathrm{p}$')
            ax_decay.legend(loc='best')
            fig_decay.tight_layout()
            fig_decay.savefig(os.path.join(outdir, 'Plots/Comparisons/Decay_rate.pdf'), bbox_inches='tight')
            plt.close(fig_decay)

    return

def plot_fancy_reconstruction(NR_sim, template, metadata, results, inference_model, outdir, method,
                              tail_flag=False, extract_damping_time_flag=False):

    """

    Plot the NR waveform and its reconstruction with the same content as the
    standard reconstruction plot, but using filled uncertainty bands.

    """

    _init_fancy_plotting()

    NR_r, NR_i = _as_1d_float_array(NR_sim.NR_r), _as_1d_float_array(NR_sim.NR_i)
    NR_amp     = _as_1d_float_array(NR_sim.NR_amp)
    NR_f       = _as_1d_float_array(NR_sim.NR_freq)
    t_NR       = _as_1d_float_array(NR_sim.t_NR)
    t_peak     = float(NR_sim.t_peak)
    t_cut, tM_start, tM_end = _as_1d_float_array(NR_sim.t_NR_cut), float(NR_sim.tM_start), float(NR_sim.tM_end)
    NR_r_cut, NR_i_cut      = _as_1d_float_array(NR_sim.NR_r_cut), _as_1d_float_array(NR_sim.NR_i_cut)
    NR_r_err_cut            = _as_1d_float_array(np.real(NR_sim.NR_cpx_err_cut))
    NR_i_err_cut            = _as_1d_float_array(np.imag(NR_sim.NR_cpx_err_cut))

    l, m       = NR_sim.l, NR_sim.m
    label_data = _fancy_data_label(NR_sim)
    tM_end     = _fancy_tM_end(NR_sim, tM_end, tail_flag)

    f_rd_fundamental   = template.qnm_cached[(2,l,m,0)]['f']
    tau_rd_fundamental = template.qnm_cached[(2,l,m,0)]['tau']

    plot_overtones_flag = 0
    f_rd_overtones      = {}
    try:
        for n in [1,3,7,9]:
            omega_n, _, _     = qnm.modes_cache(s=-2,l=l,m=m,n=n)(a=np.abs(metadata['af']))
            f_rd_overtones[n] = (np.real(omega_n) / metadata['Mf']) * (1./twopi)
    except Exception:
        f_rd_overtones = {}

    amp_peak = NR_amp[np.argmin(np.abs(t_NR - t_peak))]
    x_NR     = t_NR  - t_peak
    x_cut    = t_cut - t_peak

    color_NR      = 'k'
    color_model   = '#cc0033'
    color_qnm     = 'royalblue'
    color_t_start = 'mediumseagreen'
    color_t_peak  = 'royalblue'
    color_f_overt = 'darkorange'
    color_f_ring  = 'royalblue' if tail_flag else 'forestgreen'

    lw_NR    = 1.8
    lw_model = 2.0
    ls_t     = '--'
    ls_f     = '--'

    if(tail_flag):
        fig, (ax2, ax4) = plt.subplots(nrows=2, ncols=1, figsize=(7.2, 8.0), sharex=True)
        axes = [ax2, ax4]
    else:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10.8, 7.2))
        ax1, ax2 = axs[0]
        ax3, ax4 = axs[1]
        axes = np.ravel(axs)

    _style_fancy_axes(axes)

    if not(tail_flag):
        ax1.plot(x_NR, NR_r, c=color_NR, lw=lw_NR, label=r'$\mathrm{NR}$')
        ax3.plot(x_NR, NR_i, c=color_NR, lw=lw_NR)
        ax1.fill_between(x_cut, NR_r_cut - NR_r_err_cut, NR_r_cut + NR_r_err_cut,
                         color=color_NR, alpha=0.10, lw=0)
        ax3.fill_between(x_cut, NR_i_cut - NR_i_err_cut, NR_i_cut + NR_i_err_cut,
                         color=color_NR, alpha=0.10, lw=0)

        ax1.axvline(tM_start, c=color_t_start, lw=1.5, alpha=1.0, ls=ls_t)
        ax1.axvline(0.0, label=r'$t_{\rm peak}$', c=color_t_peak, lw=1.5, alpha=1.0, ls=ls_t)
        ax3.axvline(tM_start, label=r'$t_{\rm start} = t_{\rm peak} \, + %d \mathrm{M}$'%tM_start,
                    c=color_t_start, lw=1.5, alpha=1.0, ls=ls_t)
        ax3.axvline(0.0, c=color_t_peak, lw=1.5, alpha=1.0, ls=ls_t)

        ax1.set_xlim([-10, tM_end])
        ax3.set_xlim(ax1.get_xlim())
        ax1.set_ylabel(r'$\mathrm{Re[%s]}$'%(label_data))
        ax3.set_ylabel(r'$\mathrm{Im[%s]}$'%(label_data))
        ax3.set_xlabel(r'$t - t_{peak} \, [\mathrm{M}]$')

    NR_amp_plot = _scaled_amplitude(NR_amp, x_NR, tau_rd_fundamental, tail_flag, extract_damping_time_flag)
    ax2.semilogy(x_NR, NR_amp_plot, label=r'$\mathrm{NR}$', c=color_NR, lw=lw_NR)
    ax2.axvline(tM_start, c=color_t_start, lw=1.5, alpha=1.0, ls=ls_t)
    if not(tail_flag):
        ax2.axvline(0.0, c=color_t_peak, lw=1.5, alpha=1.0, ls=ls_t)

    if(not(tail_flag) and (NR_sim.NR_catalog=='SXS' or NR_sim.NR_catalog=='RIT')):
        if(extract_damping_time_flag):
            ax2.set_ylim([1e-1*amp_peak, 10*amp_peak])
        else:
            ax2.set_ylim([1e-3*amp_peak, 2*amp_peak])
    elif(tail_flag and (NR_sim.NR_catalog=='SXS' or NR_sim.NR_catalog=='RIT')):
        ax2.set_ylim([2*1e-4, 2*np.max(NR_amp)])

    ax4.plot(x_NR, NR_f, c=color_NR, lw=lw_NR, label=r'$\mathrm{NR}$')
    ax4.axhline(f_rd_fundamental, label=r'$\mathit{f_{%d%d0}}$'%(l,m), c=color_f_ring, lw=1.5, ls=ls_f)
    if(plot_overtones_flag):
        for n in [1,3,9]:
            if(n in f_rd_overtones):
                leg = r'$\mathit{f_{%d%dn}}$'%(l,m) if n==1 else None
                ax4.axhline(f_rd_overtones[n], label=leg, c=color_f_overt, lw=0.8, ls=ls_f)

    if(tail_flag):
        ax4.axhline(0.0, label=r'$\mathit{f_{\rm tail}}$', c=color_model, lw=1.5, ls=ls_t)
        ax4.axvline(tM_start, label=r'$\mathrm{t_{start} = t_{peak} \, + %d M}$'%tM_start,
                    c=color_t_start, lw=1.5, alpha=1.0, ls=ls_t)
        ax4.axvline(0.0, c=color_t_peak, lw=1.5, alpha=1.0, ls=ls_t)
    else:
        ax4.axvline(0.0, c=color_t_peak, lw=1.5, alpha=1.0, ls=ls_t)

    ax2.set_xlim([-10, tM_end])
    ax4.set_xlim(ax2.get_xlim())
    ax4.set_xlabel(r'$t - t_{peak} \, [\mathrm{M}]$')

    t_peak_idx = np.argmin(np.abs(t_NR - t_peak))
    if not(tail_flag):
        try:
            ax4.set_ylim([-1.5*NR_f[t_peak_idx], 3.5*NR_f[t_peak_idx]])
        except Exception:
            pass
    else:
        ax4.set_ylim([-0.08, 0.28])

    if(inference_model is not None):
        models_re_list, models_im_list = model_component_lists(results, inference_model, method)
        quantiles = _model_waveform_quantiles(models_re_list, models_im_list, t_cut)
        has_band  = len(models_re_list) > 1

        model_label = r'$\mathrm{%s}$'%(template.wf_model)
        amp_50 = _scaled_amplitude(quantiles[50]['amp'], x_cut, tau_rd_fundamental, tail_flag, extract_damping_time_flag)
        amp_5  = _scaled_amplitude(quantiles[5]['amp'] , x_cut, tau_rd_fundamental, tail_flag, extract_damping_time_flag) if has_band else None
        amp_95 = _scaled_amplitude(quantiles[95]['amp'], x_cut, tau_rd_fundamental, tail_flag, extract_damping_time_flag) if has_band else None

        if not(tail_flag):
            _plot_series_with_band(ax1, x_cut, quantiles[50]['real'],
                                   quantiles[5]['real'] if has_band else None,
                                   quantiles[95]['real'] if has_band else None,
                                   color=color_model, label=model_label, lw=lw_model)
            _plot_series_with_band(ax3, x_cut, quantiles[50]['imag'],
                                   quantiles[5]['imag'] if has_band else None,
                                   quantiles[95]['imag'] if has_band else None,
                                   color=color_model, lw=lw_model)

        _plot_series_with_band(ax2, x_cut, amp_50, amp_5, amp_95,
                               color=color_model, label=model_label, lw=lw_model, semilogy=True)
        _plot_series_with_band(ax4, x_cut, quantiles[50]['freq'],
                               quantiles[5]['freq'] if has_band else None,
                               quantiles[95]['freq'] if has_band else None,
                               color=color_model, lw=lw_model)

        if(tail_flag):
            qnm_samples = [_sample_with_suppressed_tail(sample, template) for sample in waveform_parameter_samples(results, method)]
            models_re_qnm, models_im_qnm = _model_component_lists_from_samples(qnm_samples, inference_model)
            qnm_quantiles = _model_waveform_quantiles(models_re_qnm, models_im_qnm, t_cut)
            qnm_has_band  = len(models_re_qnm) > 1

            _plot_series_with_band(ax2, x_cut, qnm_quantiles[50]['amp'],
                                   qnm_quantiles[5]['amp'] if qnm_has_band else None,
                                   qnm_quantiles[95]['amp'] if qnm_has_band else None,
                                   color=color_qnm, label=r'$\mathrm{%s \,\, QNMs}$'%(template.wf_model),
                                   lw=lw_model, semilogy=True)
            _plot_series_with_band(ax4, x_cut, qnm_quantiles[50]['freq'],
                                   qnm_quantiles[5]['freq'] if qnm_has_band else None,
                                   qnm_quantiles[95]['freq'] if qnm_has_band else None,
                                   color=color_qnm, lw=lw_model)

    if(extract_damping_time_flag and not(tail_flag)):
        ax2.set_ylabel(r'$\mathit{A_{%d%d}(t)} \cdot e^{t/\tau_{%d%d0}}$'%(l,m,l,m))
    else:
        ax2.set_ylabel(r'$\mathit{A_{%d%d}(t)}$'%(l,m))
    ax4.set_ylabel(r'$\mathit{f_{%d%d}\,(t)}$'%(l,m))

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if(len(handles)>0):
            ax.legend(loc='best')

    if not(tail_flag):
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        fig.suptitle('{}-{}'.format(NR_sim.NR_catalog, NR_sim.NR_ID), size=26)

    fig.tight_layout(rect=[0,0,1,0.94] if not(tail_flag) else [0,0,1,1])
    fig.subplots_adjust(hspace=0.05, wspace=0.27)

    leg_name_tail = '_tail' if tail_flag else ''
    fig.savefig(os.path.join(outdir, f'Plots/Comparisons/Waveform_reconstruction{leg_name_tail}.pdf'), bbox_inches='tight')
    plt.close(fig)

    return

def global_corner(x, names, output, truths=None):

    """

    Create a corner plot of all parameters.

    Parameters
    ----------

    x       : dictionary
        Dictionary of parameters.
    names   : list
        List of parameter names.
    output  : string
        Output directory.

    Returns
    -------

    Nothing, but saves a corner plot to the output directory.

    """
    
    samples = []
    for xy in names: samples.append(np.array(x[xy]))
    samples = np.transpose(samples)
    if samples.ndim < 2 or samples.shape[0] < 2:
        print('* Skipping corner plot: at least two samples are required.')
        return
    mask    = [i for i in range(samples.shape[-1]) if not all(samples[:,i]==samples[0,i]) ]
    if len(mask) == 0:
        print('* Skipping corner plot: all samples are constant.')
        return
    labels  = list(np.array(names)[mask])

    if not(truths is None):
        truths = [truths[i] for i in mask]

    fig = plt.figure(figsize=(10,10))
    C   = corner.corner(samples[:,mask],
                        quantiles     = [0.05, 0.5, 0.95],
                        labels        = labels,
                        color         = 'darkred',
                        show_titles   = True,
                        title_kwargs  = {"fontsize": 12},
                        use_math_text = True,
                        truths = truths
                        )
    plt.savefig(os.path.join(output, 'Plots', 'Results', 'corner.pdf'), bbox_inches='tight')

    return

def read_injection_truths(names, NR_sim):

    if not(hasattr(NR_sim, 'injection_truths')) or NR_sim.injection_truths is None:
        return None

    truths = []
    for name in names:
        truths.append(NR_sim.injection_truths.get(name, None))

    return truths

def plot_multiple_psd(psd_data, f_min, f_max, outdir, direction, window):
    """
    Plot multiple smoothed PSD curves in function of frequency.

    Parameters:
        psd_data (dict): A dictionary where keys are labels (str) and values are PSD arrays (np.ndarray).
        f_min (float): Minimum frequency.
        f_max (float): Maximum frequency.
        outdir (str): Output directory for saving the plot.
        direction (str): 'below' or 'above' to distinguish between smoothing directions.
        window (float): The smoothing window size.

    Returns:
        None
    """
    try:
        save_path = _mismatch_plot_dir(outdir, direction)

        # Set x-axis range based on direction
        if direction == "below":
            x_min, x_max = f_min/2, f_min + window
        elif direction == "above":
            x_min, x_max = f_max - window, f_max
        elif direction == "below-and-above":
            x_min, x_max = f_min, f_max

        # Create the plot
        plt.figure(figsize=(12, 8))

        for label, PSD_smoothed in psd_data.items():
            freq = np.linspace(0, f_max, len(PSD_smoothed))
            plt.plot(freq, PSD_smoothed, label=label, linestyle="dotted", linewidth=1.5)

        # Add labels, title, and grid
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD [Hz^-1]")
        plt.title(f"Smoothed PSD for Various Parameters ({direction.capitalize()})")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(x_min, x_max)
        #plt.legend()
        plt.grid(True)

        # Save the plot
        filename = "Multiple_Smoothed_PSD.pdf"
        path = os.path.join(save_path, filename)
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"Failed to generate smoothed PSD plot ({direction}): {e}")

def plot_psd_and_acf(psd_data, acf_data, asd_filepath, f_min, f_max, outdir, direction):
    """
    Plot multiple smoothed PSD and ACF curves in a single figure with two subplots.

    Parameters:
        psd_data (dict): A dictionary where keys are labels (str) and values are PSD arrays (np.ndarray).
        acf_data (dict): A dictionary where keys are labels (str) and values are ACF arrays (np.ndarray).
        f_min (float): Minimum frequency.
        f_max (float): Maximum frequency.
        t_start_g (float): Start time for ACF plot.
        t_end_g (float): End time for ACF plot.
        outdir (str): Output directory for saving the plot.
        direction (str): 'below', 'above', or 'below-and-above' to distinguish between smoothing directions.
        window (float): The smoothing window size.

    Returns:
        None
    """
    try:

        # Load ASD file and convert it to PSD
        freq_file, asd_file = np.loadtxt(asd_filepath, unpack=True)
        psd_file            = asd_file**2

        save_path = _mismatch_plot_dir(outdir, direction)

        # ------------------ Plot PSD ------------------ #
        fig_psd, ax_psd = plt.subplots(figsize=(10, 6))
        ax_psd.plot(freq_file, psd_file, label="No window application", linewidth=1.2, linestyle='--', color="black")

        for i, (label, PSD_smoothed) in enumerate(psd_data.items()):
            freq = np.linspace(0, f_max, len(PSD_smoothed))
            alpha = max(0.3, 1 - (i * 0.15))  # Decrease opacity for different curves
            ax_psd.plot(freq, PSD_smoothed, label=label, linewidth=2, color=colbBlue, alpha=alpha)

        ax_psd.set_xlabel("Frequency [Hz]", fontsize=22)
        ax_psd.set_ylabel("PSD [Hz^-1]",fontsize=22)
        #ax_psd.set_title(f"Smoothed PSD ({direction.capitalize()})")
        ax_psd.set_xscale("log")
        ax_psd.set_yscale("log")
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax_psd.set_xlim(f_min * 0.9, f_max * 1.1)
        ax_psd.grid(True)
        ax_psd.legend(loc="upper right")

        # Save PSD figure
        filename_psd = "PSD_Smoothed.pdf"
        path_psd = os.path.join(save_path, filename_psd)
        plt.tight_layout()
        plt.savefig(path_psd)
        plt.close(fig_psd)

        # ------------------ Plot ACF ------------------ #
        fig_acf, ax_acf = plt.subplots(figsize=(10, 6))

        # Duration time
        dt = 1 / (2 * f_max)

        for i, (label, ACF_smoothed) in enumerate(acf_data.items()):
            n_fft = len(ACF_smoothed)
            duration = n_fft * dt
            t_array = np.linspace(0, duration, n_fft)
            alpha = max(0.3, 1 - (i * 0.15))  # Decrease opacity for different curves
            ax_acf.plot(t_array, ACF_smoothed, label=label, linewidth=2, color=colbBlue, alpha=alpha)

        ax_acf.set_xlabel("Time [s]")
        ax_acf.set_ylabel("ACF")
        ax_acf.set_title(f"Smoothed ACF ({direction.capitalize()})")
        ax_acf.grid(True)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax_acf.legend(loc="upper center")

        # Save ACF figure
        filename_acf = "ACF_Smoothed.pdf"
        path_acf = os.path.join(save_path, filename_acf)
        plt.tight_layout()
        plt.savefig(path_acf)
        plt.close(fig_acf)

    except Exception as e:
        print(f"Failed to generate smoothed PSD and ACF plots ({direction}): {e}")

def plot_psd_near_fmin_fmax(psd_data, f_min, f_max, window_size_DX, window_size_SX, outdir, direction):
    """
    Plot PSD curves near f_min and f_max in a single figure with two side-by-side subplots.

    Parameters:
        psd_data (dict): Dictionary where keys are labels (str) and values are PSD arrays (np.ndarray).
        f_min (float): Minimum frequency.
        f_max (float): Maximum frequency.
        window_size_DX (float): The smoothing window size on the left of f_min.
        window_size_SX (float): The smoothing window size on the right of f_max.
        outdir (str): Output directory for saving the plot.
        direction (str): 'below', 'above', or 'below-and-above' to distinguish between smoothing directions.

    Returns:
        None
    """
    try:

        save_path = _mismatch_plot_dir(outdir, direction)

        # Set x-axis limits for zoomed regions
        x_min1, x_max1 = f_min * 0.9, (f_min + window_size_DX)  # Zoom near f_min
        x_min2, x_max2 = (f_max - window_size_SX), f_max  # Zoom near f_max

        # Create figure with two side-by-side subplots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # ------------------ Plot PSD near f_min and f_max ------------------
        for i, (label, PSD_smoothed) in enumerate(psd_data.items()):
            freq = np.linspace(0, f_max, len(PSD_smoothed))  # Generate frequency axis
            alpha = max(0.3, 1 - (i * 0.15))  # Decrease opacity for different curves

            # Identify indices for zoomed regions
            idx_min1, idx_max1 = np.argmin(np.abs(freq - x_min1)), np.argmin(np.abs(freq - x_max1))
            idx_min2, idx_max2 = np.argmin(np.abs(freq - x_min2)), np.argmin(np.abs(freq - x_max2))

            # y-axis limits dynamically
            y_min1, y_max1 = PSD_smoothed[idx_min1], PSD_smoothed[idx_max1]
            y_min2, y_max2 = PSD_smoothed[idx_min2], PSD_smoothed[idx_max2]

            # Re-order y-lims for valid ranges
            y_min1, y_max1 = min(y_min1, y_max1), max(y_min1, y_max1)
            y_min2, y_max2 = min(y_min2, y_max2), max(y_min2, y_max2)

            # Prevent identical y-limits for the second subplot (f_max)
            if y_min2 == y_max2:
                y_max2 += 1e-5  # Add a small offset to create a valid range

            # Plot for both subplots with alpha variation
            axs[0].plot(freq, PSD_smoothed, label=label, linewidth=1.5, color=colbBlue, alpha=alpha)
            axs[1].plot(freq, PSD_smoothed, label=label, linewidth=1.5, color=colbRed, alpha=alpha)

        # Adjust subplot 1 (near f_min)
        axs[0].set_xlabel("Frequency [Hz]")
        axs[0].set_ylabel("PSD [Hz^-1]")
        axs[0].set_title(f"Smoothed PSD near f_min ({direction.capitalize()})")
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        axs[0].set_xlim(x_min1 * 0.99, x_max1 * 1.01)
        axs[0].set_ylim(y_min1 * 0.5, y_max1 * 2)
        axs[0].grid(True)

        # Adjust subplot 2 (near f_max)
        axs[1].set_xlabel("Frequency [Hz]")
        axs[1].set_ylabel("PSD [Hz^-1]")
        axs[1].set_title(f"Smoothed PSD near f_max ({direction.capitalize()})")
        axs[1].set_yscale("log")

        # Prevent identical x-limits for the second subplot (f_max)
        if x_min2 == x_max2:
            x_max2 += 1e-5  # Add a small offset to create a valid range

        # Set x-limits and y-limits for the second subplot
        axs[1].set_xlim(x_min2, x_max2)
        axs[1].set_ylim(y_min2, y_max2)
        axs[1].grid(True)

        # Adjust layout and save the plot
        plt.tight_layout()
        filename = "PSD_Near_fmin_fmax.pdf"
        path = os.path.join(save_path, filename)
        plt.savefig(path)
        plt.close(fig)

    except Exception as e:
        print(f"Failed to generate PSD plots near f_min and f_max ({direction}): {e}")

def plot_acf_interpolated(t_array, t_trunc, ACF_smoothed, truncated_acf, outdir, window_size_DX, window_size_SX, k, saturation_DX, saturation_SX, direction):

    save_path = _mismatch_plot_dir(outdir, direction)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot acf interpolated
    plt.plot(t_array,ACF_smoothed,label="Original ACF", color=colbBlue)
    plt.plot(t_trunc,truncated_acf,label="Truncated ACF",linestyle="dotted", color=colbRed)
    plt.legend()
    plt.xlabel("t [s]")
    plt.xlim(t_trunc[0],t_trunc[-1])
    plt.ylim(min(ACF_smoothed)*0.99999, max(ACF_smoothed)*1.00001)

    # Save the plot
    filename = f"Truncated_ACF_wDX={round(window_size_DX,1)}Hz_wSX={round(window_size_SX,1)}Hz_k={round(k,3)}_sat_DX={round(saturation_DX,0)}_sat_SX={round(saturation_SX,0)}.pdf"
    path = os.path.join(save_path, filename)
    plt.savefig(path)
    plt.close()

def _finish_parameter_scan_plot(save_path, filename, xlabel, ylabel, xscale=None, yscale=None, legend=False):
    plt.xlabel(xlabel, fontsize=26)
    plt.ylabel(ylabel, fontsize=26)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    if legend:
        plt.legend()
    if xscale:
        plt.xscale(xscale)
    if yscale:
        plt.yscale(yscale)
    plt.grid(True)
    plt.savefig(os.path.join(save_path, filename))
    plt.close()

def _nfft_label(n_fft, rounded):
    return round(n_fft, 0) if rounded else n_fft

def _component_scan_filename(prefix, M, dL, component, suffix, direction, n_fft, round_nfft=False):
    return (
        f"{prefix}_M={M}M0_dL={dL}Mpc_{component}_{suffix}"
        f"_direction={direction}_NFFT_{_nfft_label(n_fft, round_nfft)}.pdf"
    )

def _condition_scan_filename(M, dL, suffix, direction, n_fft, round_nfft=False):
    return (
        f"Condition_Number_M={M}M0_dL={dL}Mpc_{suffix}"
        f"_direction={direction}_NFFT_{_nfft_label(n_fft, round_nfft)}.pdf"
    )

def _plot_component_metric_scan(data, outdir, direction, M, dL, N_fft, prefix, ylabel,
                                x_index, xlabel, fixed_indices, suffix_builder,
                                use_grid_groups=False, xscale=None, legend=False,
                                round_nfft=False):
    save_path = _mismatch_plot_dir(outdir, direction)
    group_fn  = _groups_from_grid if use_grid_groups else _groups_from_keys

    for n_fft_value in N_fft:
        for fixed_values in group_fn(data, fixed_indices):
            suffix = suffix_builder(*fixed_values)
            for component in strain_components:
                plt.figure(figsize=(12, 8))
                for perc in plot_percentiles:
                    x_vals, metric_vals = [], []
                    for key, values in data.items():
                        if _key_matches(key, fixed_indices, fixed_values):
                            x_vals.append(key[x_index])
                            metric_vals.append(values[component][perc])
                    plt.plot(x_vals, metric_vals, label=f"{perc}% CI", marker='o')

                filename = _component_scan_filename(
                    prefix, M, dL, component, suffix, direction, n_fft_value, round_nfft
                )
                _finish_parameter_scan_plot(
                    save_path, filename, xlabel, ylabel, xscale=xscale, legend=legend
                )

def _plot_condition_scan(data, outdir, direction, M, dL, N_fft, x_index, xlabel,
                         fixed_indices, suffix_builder, xscale=None, yscale=None,
                         round_nfft=True):
    save_path = _mismatch_plot_dir(outdir, direction)

    for n_fft_value in N_fft:
        for fixed_values in _groups_from_grid(data, fixed_indices):
            x_vals, cond_vals = [], []
            for key, cond_number in data.items():
                if _key_matches(key, fixed_indices, fixed_values):
                    x_vals.append(key[x_index])
                    cond_vals.append(cond_number)

            plt.figure(figsize=(12, 8))
            plt.plot(x_vals, cond_vals, marker='o')
            filename = _condition_scan_filename(
                M, dL, suffix_builder(*fixed_values), direction, n_fft_value, round_nfft
            )
            _finish_parameter_scan_plot(
                save_path, filename, xlabel, "Condition Number", xscale=xscale, yscale=yscale
            )

def _plot_condition_scan_all(data, outdir, direction, M, dL, N_fft, x_index, xlabel):
    save_path = _mismatch_plot_dir(outdir, direction)

    for n_fft_value in N_fft:
        x_vals, cond_vals = [], []
        for key, cond_number in data.items():
            x_vals.append(key[x_index])
            cond_vals.append(cond_number)

        plt.figure(figsize=(12, 8))
        plt.plot(x_vals, cond_vals, marker='o')
        filename = f"Condition_Number_M={M}M0_dL={dL}Mpc_direction={direction}_NFFT_{n_fft_value}.pdf"
        _finish_parameter_scan_plot(
            save_path, filename, xlabel, "Condition Number", xscale="log", yscale="log"
        )

def plot_mismatch_by_window_DX(mismatch_data, outdir, direction, M, dL, N_fft):
    """
    Plot mismatch for real and imaginary components against window_size_DX for fixed other parameters.
    """
    _plot_component_metric_scan(
        mismatch_data, outdir, direction, M, dL, N_fft,
        "Mismatch", "Mismatch", window_key_index['window_DX'], r"$w_l$ [Hz]",
        (window_key_index['k'], window_key_index['saturation_DX'], window_key_index['saturation_SX'], window_key_index['window_SX']),
        lambda k, sDX, sSX, wsx: f"k={round(k,0)}_satDX={sDX:.2e}_satSX={sSX:.2e}_wsx={wsx}",
    )

def plot_mismatch_by_window_SX(mismatch_data, outdir, direction, M, dL, N_fft):
    """
    Plot mismatch for real and imaginary components against window_size_SX for fixed other parameters.
    """
    _plot_component_metric_scan(
        mismatch_data, outdir, direction, M, dL, N_fft,
        "Mismatch", "Mismatch", window_key_index['window_SX'], r"$w_h$ [Hz]",
        (window_key_index['k'], window_key_index['saturation_DX'], window_key_index['saturation_SX'], window_key_index['window_DX']),
        lambda k, sDX, sSX, wdx: f"k={round(k,0)}_satDX={sDX:.2e}_satSX={sSX:.2e}_wdx={wdx}",
    )

def plot_optimal_SNR_by_window_DX(optimal_SNR_data, outdir, direction, M, dL, N_fft):
    """
    Plot optimal SNR for real and imaginary components against window_size_DX for fixed k and saturations.
    """
    _plot_component_metric_scan(
        optimal_SNR_data, outdir, direction, M, dL, N_fft,
        "Optimal_SNR", "Optimal SNR", window_key_index['window_DX'], r"$w_l$ [Hz]",
        (window_key_index['k'], window_key_index['saturation_DX'], window_key_index['saturation_SX'], window_key_index['window_SX']),
        lambda k, sDX, sSX, wsx: f"k={round(k,0)}_satDX={sDX:.2e}_satSX={sSX:.2e}_wsx={wsx}",
    )

def plot_optimal_SNR_by_window_SX(optimal_SNR_data, outdir, direction, M, dL, N_fft):
    """
    Plot optimal SNR for real and imaginary components against window_size_SX for fixed k and saturations.
    """
    _plot_component_metric_scan(
        optimal_SNR_data, outdir, direction, M, dL, N_fft,
        "Optimal_SNR", "Optimal SNR", window_key_index['window_SX'], r"$w_h$ [Hz]",
        (window_key_index['k'], window_key_index['saturation_DX'], window_key_index['saturation_SX'], window_key_index['window_DX']),
        lambda k, sDX, sSX, wdx: f"k={round(k,0)}_satDX={sDX:.2e}_satSX={sSX:.2e}_wdx={wdx}",
    )

def plot_condition_number_by_window_DX(condition_numbers, outdir, direction, M, dL, N_fft):
    """
    Plot condition number against window_size_DX.
    """
    _plot_condition_scan_all(
        condition_numbers, outdir, direction, M, dL, N_fft,
        window_key_index['window_DX'], r"$w_l$ [Hz]"
    )

def plot_condition_number_by_window_SX(condition_numbers, outdir, direction, M, dL, N_fft):
    """
    Plot condition number against window_size_SX.
    """
    _plot_condition_scan_all(
        condition_numbers, outdir, direction, M, dL, N_fft,
        window_key_index['window_SX'], r"$w_h$ [Hz]"
    )

def plot_mismatch_by_k(mismatch_data, outdir, direction, M, dL, N_fft):
    """
    Plot mismatch for real and imaginary components by varying k, keeping window sizes and saturations fixed.
    """
    _plot_component_metric_scan(
        mismatch_data, outdir, direction, M, dL, N_fft,
        "Mismatch", "Mismatch", window_key_index['k'], "k",
        (window_key_index['window_DX'], window_key_index['window_SX'], window_key_index['saturation_DX'], window_key_index['saturation_SX']),
        lambda wDX, wSX, sDX, sSX: f"wDX={round(wDX,1)}_wSX={round(wSX,1)}_satDX={sDX:.2e}_satSX={sSX:.2e}",
        use_grid_groups=True, xscale="log", legend=True, round_nfft=True,
    )

def plot_optimal_SNR_by_k(optimal_SNR_data, outdir, direction, M, dL, N_fft):
    """
    Plot optimal SNR for real and imaginary components by varying k, keeping window sizes and saturations fixed.
    """
    _plot_component_metric_scan(
        optimal_SNR_data, outdir, direction, M, dL, N_fft,
        "Optimal_SNR", "Optimal SNR", window_key_index['k'], "k",
        (window_key_index['window_DX'], window_key_index['window_SX'], window_key_index['saturation_DX'], window_key_index['saturation_SX']),
        lambda wDX, wSX, sDX, sSX: f"wDX={round(wDX,1)}_wSX={round(wSX,1)}_satDX={sDX:.2e}_satSX={sSX:.2e}",
        use_grid_groups=True, xscale="log", legend=True, round_nfft=True,
    )

def plot_condition_number_by_k(condition_numbers, outdir, direction, M, dL, N_fft):
    """
    Plot the condition number of the ACF Toeplitz matrix by varying k, keeping window sizes, saturation_DX, and saturation_SX fixed.
    """
    _plot_condition_scan(
        condition_numbers, outdir, direction, M, dL, N_fft,
        window_key_index['k'], "k",
        (window_key_index['window_DX'], window_key_index['window_SX'], window_key_index['saturation_DX'], window_key_index['saturation_SX']),
        lambda wDX, wSX, sDX, sSX: f"wDX={round(wDX,1)}_wSX={round(wSX,1)}_satDX={sDX:.2e}_satSX={sSX:.2e}",
        xscale="log", round_nfft=True,
    )

def plot_mismatch_by_saturation_DX(mismatch_data, outdir, direction, M, dL, N_fft):
    """
    Plot mismatch for real and imaginary components by varying saturation_DX, keeping window sizes, k, and saturation_SX fixed.
    """
    _plot_component_metric_scan(
        mismatch_data, outdir, direction, M, dL, N_fft,
        "Mismatch", "Mismatch", window_key_index['saturation_DX'], r"$\mathcal{I}_l$",
        (window_key_index['window_DX'], window_key_index['window_SX'], window_key_index['k'], window_key_index['saturation_SX']),
        lambda wDX, wSX, k, sSX: f"wDX={round(wDX,1)}_wSX={round(wSX,1)}_k={round(k,0)}_satSX={sSX:.2e}",
        use_grid_groups=True, xscale="log", legend=True, round_nfft=True,
    )

def plot_optimal_SNR_by_saturation_DX(optimal_SNR_data, outdir, direction, M, dL, N_fft):
    """
    Plot optimal SNR for real and imaginary components by varying saturation_DX, keeping window sizes, k, and saturation_SX fixed.
    """
    _plot_component_metric_scan(
        optimal_SNR_data, outdir, direction, M, dL, N_fft,
        "Optimal_SNR", "Optimal SNR", window_key_index['saturation_DX'], r"$\mathcal{I}_l$",
        (window_key_index['window_DX'], window_key_index['window_SX'], window_key_index['k'], window_key_index['saturation_SX']),
        lambda wDX, wSX, k, sSX: f"wDX={round(wDX,1)}_wSX={round(wSX,1)}_k={round(k,0)}_satSX={sSX:.2e}",
        use_grid_groups=True, xscale="log", legend=True, round_nfft=True,
    )

def plot_condition_number_by_saturation_DX(condition_numbers, outdir, direction, M, dL, N_fft):
    """
    Plot the condition number of the ACF Toeplitz matrix by varying saturation_DX, keeping window sizes, k, and saturation_SX fixed.
    """
    _plot_condition_scan(
        condition_numbers, outdir, direction, M, dL, N_fft,
        window_key_index['saturation_DX'], r"$\mathcal{I}_l$",
        (window_key_index['window_DX'], window_key_index['window_SX'], window_key_index['k'], window_key_index['saturation_SX']),
        lambda wDX, wSX, k, sSX: f"wDX={round(wDX,1)}_wSX={round(wSX,1)}_k={round(k,0)}_satSX={sSX:.2e}",
        xscale="log", yscale="log", round_nfft=True,
    )

def plot_mismatch_by_saturation_SX(mismatch_data, outdir, direction, M, dL, N_fft):
    """
    Plot mismatch for real and imaginary components by varying saturation_SX, keeping window sizes, k, and saturation_DX fixed.
    """
    _plot_component_metric_scan(
        mismatch_data, outdir, direction, M, dL, N_fft,
        "Mismatch", "Mismatch", window_key_index['saturation_SX'], r"$\mathcal{I}_h$",
        (window_key_index['window_DX'], window_key_index['window_SX'], window_key_index['k'], window_key_index['saturation_DX']),
        lambda wDX, wSX, k, sDX: f"wDX={round(wDX,1)}_wSX={round(wSX,1)}_k={round(k,0)}_satDX={sDX:.2e}",
        use_grid_groups=True, xscale="log", legend=True, round_nfft=True,
    )

def plot_optimal_SNR_by_saturation_SX(optimal_SNR_data, outdir, direction, M, dL, N_fft):
    """
    Plot optimal SNR for real and imaginary components by varying saturation_SX, keeping window sizes, k, and saturation_DX fixed.
    """
    _plot_component_metric_scan(
        optimal_SNR_data, outdir, direction, M, dL, N_fft,
        "Optimal_SNR", "Optimal SNR", window_key_index['saturation_SX'], r"$\mathcal{I}_h$",
        (window_key_index['window_DX'], window_key_index['window_SX'], window_key_index['k'], window_key_index['saturation_DX']),
        lambda wDX, wSX, k, sDX: f"wDX={round(wDX,1)}_wSX={round(wSX,1)}_k={round(k,0)}_satDX={sDX:.2e}",
        use_grid_groups=True, xscale="log", legend=True, round_nfft=True,
    )

def plot_condition_number_by_saturation_SX(condition_numbers, outdir, direction, M, dL, N_fft):
    """
    Plot the condition number of the ACF Toeplitz matrix by varying saturation_SX, keeping window sizes, k, and saturation_DX fixed.
    """
    _plot_condition_scan(
        condition_numbers, outdir, direction, M, dL, N_fft,
        window_key_index['saturation_SX'], r"$\mathcal{I}_h$",
        (window_key_index['window_DX'], window_key_index['window_SX'], window_key_index['k'], window_key_index['saturation_DX']),
        lambda wDX, wSX, k, sDX: f"wDX={round(wDX,1)}_wSX={round(wSX,1)}_k={round(k,0)}_satDX={sDX:.2e}",
        xscale="log", yscale="log", round_nfft=True,
    )

def plot_condition_numbers(outdir, condition_numbers, thresholds=(1e3, 1e6)):

    """
    Plot the condition numbers of the ACF Toeplitz matrix as a function of window size for different k values,
    including shaded regions to indicate conditioning quality.

    Parameters:
        condition_numbers (dict): Dictionary with keys as (window_size, k) and values as condition numbers.
        outdir (str): Directory to save the plot.
        thresholds (tuple): Thresholds for marking the zones (well-conditioned, moderately, poorly).
                            Default: (1e3, 1e6).
    """

    # Ensure the input is a dictionary
    if not isinstance(condition_numbers, dict):
        raise ValueError("condition_numbers must be a dictionary.")

    # Extract unique values of k and sorted window sizes
    ks = sorted(set(k for _, k in condition_numbers.keys()))
    window_sizes = sorted(set(ws for ws, _ in condition_numbers.keys()))

    # Define the regions
    low_threshold, high_threshold = thresholds

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Add shaded regions
    plt.axhspan(0, low_threshold, color='green', alpha=0.1, label="Well-Conditioned")
    plt.axhspan(low_threshold, high_threshold, color='yellow', alpha=0.1, label="Moderately Conditioned")
    plt.axhspan(high_threshold, 10 * high_threshold, color='red', alpha=0.1, label="Poorly Conditioned")

    for k in ks:
        # Extract condition numbers for each window_size at the current k
        cond_numbers = [condition_numbers[(ws, k)] for ws in window_sizes]
        plt.plot(window_sizes, cond_numbers, label=f"k = {k}", linestyle="dotted")

    # Configure plot
    plt.xlabel("Window Size")
    plt.ylabel("Condition Number")
    plt.title("Condition Number of ACF Toeplitz Matrix vs Window Size")
    plt.yscale("log")
    plt.legend(title="Steepness (k)", loc="upper left")
    plt.grid(True)

    # Save plot to file
    os.makedirs(outdir, exist_ok=True)
    plot_file_path = os.path.join(outdir, "Algorithm/Condition_Numbers_Plot.pdf")
    plt.savefig(plot_file_path)
    #print(f"Condition number plot saved to {plot_file_path}")

def plot_mismatch_optimal_SNR_condition_number_window_parameters(mismatch_data, optimal_SNR_data, condition_numbers, outdir, direction, M, dL, N_FFT):
    """
    Plots mismatch, optimal SNR, and condition number data if the corresponding x-axis variable has dim > 1.
    """

    plot_specs = (
        (mismatch_data, (
            (plot_mismatch_by_window_DX, "window_DX"),
            (plot_mismatch_by_window_SX, "window_SX"),
            (plot_mismatch_by_k, "k"),
            (plot_mismatch_by_saturation_DX, "saturation_DX"),
            (plot_mismatch_by_saturation_SX, "saturation_SX"),
        )),
        (optimal_SNR_data, (
            (plot_optimal_SNR_by_window_DX, "window_DX"),
            (plot_optimal_SNR_by_window_SX, "window_SX"),
            (plot_optimal_SNR_by_k, "k"),
            (plot_optimal_SNR_by_saturation_DX, "saturation_DX"),
            (plot_optimal_SNR_by_saturation_SX, "saturation_SX"),
        )),
        (condition_numbers, (
            (plot_condition_number_by_window_DX, "window_DX"),
            (plot_condition_number_by_window_SX, "window_SX"),
            (plot_condition_number_by_k, "k"),
            (plot_condition_number_by_saturation_DX, "saturation_DX"),
            (plot_condition_number_by_saturation_SX, "saturation_SX"),
        )),
    )

    for data_dict, functions in plot_specs:
        for plot_function, x_key in functions:
            index = window_key_index[x_key]
            if len({key[index] for key in data_dict.keys()}) > 1:
                plot_function(data_dict, outdir, direction, M, dL, N_FFT)

def plot_mismatch_vs_NFFT(N_FFT_list, N_points, M, dL, t_start_g_true, window_DX_list, window_SX_list, k_list, saturation_DX_list, saturation_SX_list,  outdir, direction):

    """
    Loop through all combinations of windowing parameters and plot mismatch (real & imaginary at 50% CI) vs N_FFT.
    """

    save_path = _mismatch_plot_dir(outdir, direction)
    diagnostic_rows = _read_tsv_rows(_mismatch_diagnostics_path(outdir))
    mismatches_by_run = {}
    for row in diagnostic_rows:
        if row.get('diagnostic_type') != 'strain_components':
            continue
        if row.get('confidence_interval') != '50':
            continue
        mismatch = row.get('mismatch')
        if mismatch in (None, ''):
            continue
        mismatches_by_run[(row.get('run_id'), row.get('strain_data'))] = float(mismatch)

    for (window_DX, window_SX, k, satDX, satSX) in product(
        window_DX_list, window_SX_list, k_list, saturation_DX_list, saturation_SX_list
    ):
        real_mismatches = []
        imag_mismatches = []
        nffts_found = []

        for N_fft in N_FFT_list:
            run_id = _mismatch_run_id(
                "strain_components", M, dL, t_start_g_true, N_fft,
                window_DX, window_SX, k, satDX, satSX, direction
            )
            component_mismatches = {
                'real': mismatches_by_run.get((run_id, 'real')),
                'imag': mismatches_by_run.get((run_id, 'imag')),
            }
            if all(value is not None for value in component_mismatches.values()):
                nffts_found.append(N_fft)
                real_mismatches.append(component_mismatches['real'])
                imag_mismatches.append(component_mismatches['imag'])
                continue

            legacy_path = _windowed_result_path(
                outdir, "Mismatch", M, dL, t_start_g_true, N_fft,
                window_DX, window_SX, k, satDX, satSX
            )
            if os.path.exists(legacy_path):
                with open(legacy_path, "r") as f:
                    component_mismatches = {'real': None, 'imag': None}
                    lines = f.readlines()[1:]
                    for line in lines:
                        perc, component, mismatch = line.strip().split()
                        if perc == "50" and component in component_mismatches:
                            component_mismatches[component] = float(mismatch)
                    if all(value is not None for value in component_mismatches.values()):
                        nffts_found.append(N_fft)
                        real_mismatches.append(component_mismatches['real'])
                        imag_mismatches.append(component_mismatches['imag'])
            else:
                print(f"No mismatch data found for run_id={run_id}")

        if not nffts_found:
            print(f"Skipping: No data for combo wDX={window_DX}, wSX={window_SX}, k={k}, satDX={satDX}, satSX={satSX}")
            continue

        # Sort by NFFT
        nffts_found, real_mismatches, imag_mismatches = zip(
            *sorted(zip(nffts_found, real_mismatches, imag_mismatches))
        )

        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(nffts_found, real_mismatches, marker='o', label='real', linewidth=2, color=colbGreen)
        plt.plot(nffts_found, imag_mismatches, marker='s', label='imaginary', linewidth=2, color=colbBlue)
        plt.axvline(x=N_points, color="black", linestyle='--', linewidth=1)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$N_{\rm FFT}$", fontsize=13)
        plt.ylabel("Mismatch (50% CI)", fontsize=13)
        plt.title(
            rf"$w_{{\rm DX}}$={window_DX:.1f}, $w_{{\rm SX}}$={window_SX:.1f}, "
            rf"$k$={k:.2f}, satDX={satDX:.1f}, satSX={satSX:.1f}",
            fontsize=12
        )
        plt.grid(True, which="both", ls=':')
        plt.legend()
        plt.tight_layout()

        # Save
        fname = (
            f"Mismatch_vs_NFFT_M_{M}_dL_{dL}_t_s_{round(t_start_g_true,1)}M_"
            f"wDX_{round(window_DX,1)}_wSX_{round(window_SX,1)}_k_{round(k,2)}"
            f"_satDX_{round(satDX,1)}_satSX_{round(satSX,1)}.pdf"
        )
        full_path = os.path.join(save_path, fname)
        plt.savefig(full_path, bbox_inches="tight")
        print(f"\nSaved: {full_path}\n")
        plt.close()

def run_mismatch_and_SNR_computation(NR_sim, results_object, inference_model, parameters, wf_utils):
    """
    Run the mismatch computation section for a given NR simulation and inference model.

    Parameters
    ----------
    NR_sim : object
        Numerical relativity simulation data.
    results_object : object
        Results object from the inference.
    inference_model : object
        Inference model instance.
    parameters : dict
        Parameter dictionary with all analysis settings.
    wf_utils : module
        Utility module for waveform operations.

    Returns
    -------
    psd_data, acf_data, mismatch_data, optimal_SNR_data, condition_numbers_data : dict
        Collected data products for later postprocessing and plotting.
    """

    psd_data, acf_data, mismatch_data, optimal_SNR_data, condition_numbers_data = {}, {}, {}, {}, {}

    try:
        #---------------------------------------------#
        # Extract GW and PSD parameters
        #---------------------------------------------#
        outdir = parameters['I/O']['outdir']
        method = parameters['Inference']['method']
        M, dL, ra, dec, psi = wf_utils.extract_GW_parameters(parameters)
        t_start_g_true = parameters['Inference']['t-start']
        t_start_g, t_end_g, t_NR_s, NR_length = wf_utils.extract_NR_params(NR_sim, M)
        t_start, t_end = t_start_g * C_mt * M, t_end_g * C_mt * M

        apply_window, compare_TD_FD, _, C1_flag, mismatch_print_flag, mismatch_section_plot_flag = \
            wf_utils.extract_flags(parameters['Flags'])

        (f_min, f_max, dt, delta_f, N_points, n_FFT_points, asd_path,
         n_iterations_C1, window_sizes_DX, window_sizes_SX,
         steepness_values, saturation_DX_values, saturation_SX_values,
         direction) = wf_utils.extract_and_compute_psd_parameters(
            parameters['Mismatch-PSD-settings'], mismatch_print_flag
        )

        n_fft_values = [N_points] if n_FFT_points == 1 else list(
            map(int, np.logspace(np.log10(NR_length), np.log10(2 * N_points), n_FFT_points))
        )

        #---------------------------------------------#
        # Main iteration loop
        #---------------------------------------------#
        grid = product(
            n_fft_values,
            window_sizes_DX,
            window_sizes_SX,
            steepness_values,
            saturation_DX_values,
            saturation_SX_values,
        )

        for N_fft, window_size_DX, window_size_SX, k, saturation_DX, saturation_SX in grid:
            if (t_end - t_start) > 1 / (f_min + window_size_DX) and direction != 'above':
                print("Please provide (t_end-t_start) < 1/(f_min+window_size_DX).")
                print("Forbidden frequency:", f_min + window_size_DX)
                return None

            window_args = (window_size_DX, window_size_SX, k, saturation_DX, saturation_SX)
            label = (
                f"wDX={window_size_DX:.1f}Hz, wSX={window_size_SX:.1f}Hz, "
                f"k={k:.1f}, satDX={saturation_DX:.1f}, satSX={saturation_SX:.1f}, N_FFT={N_fft}"
            )

            try:
                if apply_window == 1:
                    print(f"* Applying window with parameters: w_DX={window_size_DX:.1f}Hz, w_SX={window_size_SX:.1f}Hz, "
                          f"k={k:.1f}, satDX={saturation_DX:.1f}, satSX={saturation_SX:.1f}, N_FFT={N_fft}")
                    PSD_smoothed, ACF_smoothed = wf_utils.acf_from_asd_with_smoothing(
                        asd_path, f_min, f_max, N_fft, *window_args,
                        direction, C1_flag, n_iterations_C1
                    )
                else:
                    print("* Computing ACF from PSD without smoothing")
                    PSD_smoothed, ACF_smoothed = wf_utils.acf_from_asd_no_window_at_edges(
                        asd_path, f_min, f_max, N_fft
                    )

                psd_data[label] = PSD_smoothed
                acf_data[label] = ACF_smoothed

                t_ACF = np.linspace(0, N_fft * dt, len(ACF_smoothed))
                ACF_truncated_NR = truncate_and_interpolate_acf(
                    t_ACF, ACF_smoothed, M, t_start_g, t_end_g, t_NR_s, mismatch_print_flag
                )

                common_args = (
                    NR_sim, results_object, inference_model, outdir, method,
                    ACF_truncated_NR, N_fft, M, dL
                )

                compute_mismatch_hplus_hcross(
                    *common_args, t_start_g_true, f_min, f_max, asd_path,
                    *window_args, mismatch_print_flag, compare_TD_FD, direction=direction
                )

                compute_nr_comparison_mismatches(
                    NR_sim, outdir, ACF_truncated_NR, N_fft, M, dL,
                    t_start_g_true, *window_args, direction=direction
                )

                compute_optimal_SNR(
                    *common_args, t_start_g_true, t_end_g, f_min, f_max, asd_path,
                    *window_args, compare_TD_FD, direction=direction
                )

                condition_numbers_data[window_args] = wf_utils.compute_condition_number(ACF_truncated_NR)

                if mismatch_section_plot_flag == 1:
                    plot_acf_interpolated(
                        t_ACF, t_NR_s, ACF_smoothed, ACF_truncated_NR,
                        outdir, *window_args, direction
                    )

            except Exception as e:
                print(f"* Mismatch computation failed for wDX={window_size_DX}, wSX={window_size_SX}, k={k}: {e}")

        #---------------------------------------------#
        # Postprocessing plots
        #---------------------------------------------#
        if mismatch_section_plot_flag == 1:
            plot_psd_and_acf(psd_data, acf_data, asd_path, f_min, f_max, outdir, direction)

    except Exception as e:
        print(f"\n* Mismatch computation failed. Check parameters and input data.\nError: {e}")
