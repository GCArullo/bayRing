import math


CATALOG_NAME = 'injections'
LEGACY_CATALOG_NAMES = {'fake_NR'}
CATALOG_NAMES = {CATALOG_NAME} | LEGACY_CATALOG_NAMES
TIME_KEYS = {'t_start', 't_end', 'dt', 't_peak'}
REQUIRED_KEYS = ['t_start', 't_end', 'dt', 'q', 'Mf', 'af']
METADATA_KEYS = {
    'q',
    'm1',
    'm2',
    'chi1',
    'chi2',
    'tilt1',
    'tilt2',
    'ecc',
    'Mf',
    'af',
    'qf',
    'bmrg',
    'Emrg',
    'Jmrg',
    'A_nr_error',
}
METADATA_PREFIXES = ('A_peak', 'omg_peak')


def _is_metadata_key(key):
    return key in METADATA_KEYS or key.startswith(METADATA_PREFIXES)


def is_injection_catalog(catalog):
    return catalog in CATALOG_NAMES


def _set_parameter(parameters, key, value):
    if key in parameters and not math.isclose(parameters[key], value):
        raise ValueError("Conflicting injection values supplied for `{}`.".format(key))
    parameters[key] = value


def _derived_mass_metadata(metadata):
    if 'm1' in metadata and 'm2' in metadata:
        return

    q = metadata['q']
    metadata.setdefault('m1', q/(1.0 + q))
    metadata.setdefault('m2', 1.0/(1.0 + q))


def split_injection_parameters(injection_parameters):
    """
    Split an injection dictionary into times, metadata and waveform parameters.

    Waveform parameters use the same names as the inference model.  Legacy Kerr
    amplitude keys such as ``A_220`` are converted to ``ln_A_220``.
    """

    if injection_parameters is None:
        raise ValueError("No injection parameters were supplied.")

    missing_keys = [name for name in REQUIRED_KEYS if name not in injection_parameters]
    if len(missing_keys):
        raise ValueError("Missing mandatory injection parameters: {}".format(missing_keys))

    times = {key: float(injection_parameters[key]) for key in TIME_KEYS if key in injection_parameters}
    metadata = {}
    waveform_parameters = {}

    for key, raw_value in injection_parameters.items():
        if key in TIME_KEYS:
            continue

        value = float(raw_value)
        if _is_metadata_key(key):
            metadata[key] = value
        elif key.startswith('A_') and key.endswith('_tail'):
            tail_mode = key[len('A_'):-len('_tail')]
            _set_parameter(waveform_parameters, 'ln_A_tail_{}'.format(tail_mode), math.log(value))
        elif key.startswith('phi_') and key.endswith('_tail'):
            tail_mode = key[len('phi_'):-len('_tail')]
            _set_parameter(waveform_parameters, 'phi_tail_{}'.format(tail_mode), value)
        elif key.startswith('p_') and key.endswith('_tail'):
            tail_mode = key[len('p_'):-len('_tail')]
            _set_parameter(waveform_parameters, 'p_tail_{}'.format(tail_mode), value)
        elif key.startswith('A_'):
            mode = key[len('A_'):]
            _set_parameter(waveform_parameters, 'ln_A_{}'.format(mode), math.log(value))
        else:
            _set_parameter(waveform_parameters, key, value)

    _derived_mass_metadata(metadata)
    metadata.setdefault('chi1', 0.0)
    metadata.setdefault('chi2', 0.0)
    metadata.setdefault('tilt1', 0.0)
    metadata.setdefault('tilt2', 0.0)
    metadata.setdefault('ecc', 0.0)
    metadata.setdefault('bmrg', 0.0)
    metadata.setdefault('Emrg', 0.0)
    metadata.setdefault('Jmrg', 0.0)

    return times, metadata, waveform_parameters


def metadata_from_simulation(NR_sim):
    metadata = dict(getattr(NR_sim, 'injection_metadata', {}))
    for key in ['q', 'Mf', 'af']:
        metadata.setdefault(key, getattr(NR_sim, key))
    return metadata
