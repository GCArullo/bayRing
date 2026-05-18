import math

import pyRing.utils as pyr_utils


CATALOG_NAME = 'injections'
CATALOG_NAMES = {CATALOG_NAME}
TIME_KEYS = {'t_start', 't_end', 'dt', 't_peak'}
REQUIRED_TIME_KEYS = ['t_start', 't_end', 'dt']
REQUIRED_REMNANT_KEYS = ['Mf', 'af']
NR_INFORMED_TEMPLATES = {'KerrBinary', 'TEOBPM'}
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
LEGACY_PARAMETER_KEYS = {'Kerr-parameters'}


def _is_metadata_key(key):
    return key in METADATA_KEYS or key.startswith(METADATA_PREFIXES)


def is_injection_catalog(catalog):
    return catalog in CATALOG_NAMES


def _set_parameter(parameters, key, value):
    if key in parameters and not math.isclose(parameters[key], value):
        raise ValueError("Conflicting injection values supplied for `{}`.".format(key))
    parameters[key] = value


def _reject_legacy_parameter_keys(injection_parameters):
    legacy_keys = []
    for key in injection_parameters:
        if key in LEGACY_PARAMETER_KEYS:
            legacy_keys.append(key)
        elif key.startswith('A_') and not (key.startswith('A_peak') or key == 'A_nr_error'):
            legacy_keys.append(key)
        elif key.startswith('phi_') and key.endswith('_tail'):
            legacy_keys.append(key)
        elif key.startswith('p_') and key.endswith('_tail'):
            legacy_keys.append(key)

    if legacy_keys:
        raise ValueError(
            "Unsupported legacy injection parameter key(s): {}. "
            "Use the current waveform parameter names, e.g. `ln_A_220`, "
            "`ln_A_tail_22`, `phi_tail_22` and `p_tail_22`.".format(sorted(legacy_keys))
        )


def _set_binary_mass_metadata(metadata):
    has_component_masses = 'm1' in metadata and 'm2' in metadata
    has_q = 'q' in metadata

    if not (has_component_masses or has_q):
        raise ValueError("Missing mandatory injection binary mass parameter: provide `q` or both `m1` and `m2`.")

    if has_component_masses:
        if metadata['m2'] == 0.0:
            raise ValueError("Injection parameter `m2` must be non-zero.")

        q_from_masses = metadata['m1']/metadata['m2']
        if has_q and not math.isclose(metadata['q'], q_from_masses):
            raise ValueError("Conflicting injection values supplied for `q`, `m1` and `m2`.")
        metadata.setdefault('q', q_from_masses)
        return

    q = metadata['q']
    metadata.setdefault('m1', q/(1.0 + q))
    metadata.setdefault('m2', 1.0/(1.0 + q))


def _split_current_injection_parameters(injection_parameters):
    times = {key: float(injection_parameters[key]) for key in TIME_KEYS if key in injection_parameters}
    metadata = {}
    waveform_parameters = {}

    for key, raw_value in injection_parameters.items():
        if key in TIME_KEYS:
            continue

        value = float(raw_value)
        if _is_metadata_key(key):
            metadata[key] = value
        else:
            _set_parameter(waveform_parameters, key, value)

    _set_binary_mass_metadata(metadata)
    metadata.setdefault('chi1', 0.0)
    metadata.setdefault('chi2', 0.0)
    metadata.setdefault('tilt1', 0.0)
    metadata.setdefault('tilt2', 0.0)
    metadata.setdefault('ecc', 0.0)
    metadata.setdefault('bmrg', 0.0)
    metadata.setdefault('Emrg', 0.0)
    metadata.setdefault('Jmrg', 0.0)

    return times, metadata, waveform_parameters


def _missing_keys(keys, parameters):
    return [name for name in keys if name not in parameters]


def _model_template(model_parameters):
    return dict(model_parameters or {}).get('template', 'Kerr')


def _noncircular_parameters(metadata, final_state_nc_version):
    missing_keys = _missing_keys(final_state_nc_version.split('-'), metadata)
    if missing_keys:
        raise ValueError(
            "Missing noncircular injection metadata required by "
            "`KerrBinary-final-state-nc-version`: {}.".format(missing_keys)
        )
    return {key: metadata[key] for key in final_state_nc_version.split('-')}


def _compute_nr_informed_remnant(metadata, model_parameters):
    template = _model_template(model_parameters)
    supplied_remnant_keys = [key for key in REQUIRED_REMNANT_KEYS if key in metadata]
    if supplied_remnant_keys:
        raise ValueError(
            "NR-informed injection template `{}` derives `Mf` and `af` from binary "
            "parameters. Remove independent remnant parameter(s): {}.".format(template, supplied_remnant_keys)
        )

    remnant_model = pyr_utils.RemnantModel()
    if (
        template == 'KerrBinary'
        and model_parameters.get('KerrBinary-version', 'London2018') == 'Carullo2024'
    ):
        final_state_nc_version = model_parameters.get('KerrBinary-final-state-nc-version', '')
        if not final_state_nc_version:
            raise ValueError(
                "KerrBinary Carullo2024 injections require "
                "`KerrBinary-final-state-nc-version` so remnant parameters can be "
                "computed from the noncircular NR fit."
            )
        nc_parameters = _noncircular_parameters(metadata, final_state_nc_version)
        Mf, af = remnant_model.compute_remnant_parameters_from_inspiral_aligned_spins_noncircular_parameters(
            metadata['m1'],
            metadata['m2'],
            metadata['chi1'],
            metadata['chi2'],
            nc_parameters,
            final_state_nc_version,
        )
    else:
        Mf, af = remnant_model.compute_remnant_parameters_from_inspiral_aligned_spins_quasicircular_parameters(
            metadata['m1'],
            metadata['m2'],
            metadata['chi1'],
            metadata['chi2'],
        )

    metadata['Mf'] = float(Mf)
    metadata['af'] = float(af)


def _validate_remnant_parameters(metadata, model_parameters):
    template = _model_template(model_parameters)
    if template in NR_INFORMED_TEMPLATES:
        _compute_nr_informed_remnant(metadata, model_parameters)
        return

    missing_keys = _missing_keys(REQUIRED_REMNANT_KEYS, metadata)
    if missing_keys:
        raise ValueError("Missing mandatory injection remnant parameters: {}".format(missing_keys))


def prepare_injection_parameters(injection_parameters, model_parameters=None):
    """
    Build injection times, metadata and waveform parameters for a template.

    The input dictionary must use the current parameter names.  NR-informed
    templates derive ``Mf`` and ``af`` from the binary parameters through
    pyRing's remnant fits.
    """

    if injection_parameters is None:
        raise ValueError("No injection parameters were supplied.")

    _reject_legacy_parameter_keys(injection_parameters)

    missing_keys = _missing_keys(REQUIRED_TIME_KEYS, injection_parameters)
    if len(missing_keys):
        raise ValueError("Missing mandatory injection parameters: {}".format(missing_keys))

    model_parameters = dict(model_parameters or {})
    times, metadata, waveform_parameters = _split_current_injection_parameters(injection_parameters)
    _validate_remnant_parameters(metadata, model_parameters)

    return times, metadata, waveform_parameters


def split_injection_parameters(injection_parameters, model_parameters=None):
    return prepare_injection_parameters(injection_parameters, model_parameters)


def metadata_from_simulation(NR_sim):
    metadata = dict(getattr(NR_sim, 'injection_metadata', {}))
    for key in ['q', 'Mf', 'af']:
        metadata.setdefault(key, getattr(NR_sim, key))
    return metadata
