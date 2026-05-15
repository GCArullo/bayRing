Module Guide
------------

This page is a high-level map of the source tree. It avoids importing heavy
runtime dependencies during documentation builds.

Package Entry Points
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 52

   * - Module
     - Responsibility
   * - ``bayRing.bayRing``
     - Command-line workflow: read config, load NR data, build model, run
       inference and post-process.
   * - ``bayRing.initialise``
     - Defaults, config parsing, output-directory creation and help text.
   * - ``bayRing.NR_waveforms``
     - Catalogue readers, NR metadata extraction, error estimation and
       waveform cropping.
   * - ``bayRing.QNM_utils``
     - QNM mode parsing, quadratic-mode parsing and QNM frequency cache setup.
   * - ``bayRing.template_waveforms``
     - Waveform-model wrapper used by the inference model.
   * - ``bayRing.inference``
     - Dynamic sampler model, priors, nested sampling, minimization and linear
       inversion.
   * - ``bayRing.postprocess``
     - Point estimates, posterior reads, waveform plots, mismatch/SNR
       diagnostics and corner plots.
   * - ``bayRing.waveform_utils``
     - Waveform alignment, PSD/ACF utilities, physical scaling and helper
       calculations.
   * - ``bayRing.utils``
     - Shared utility functions for fixed parameters, paths and local data
       prefixes.

Main Runtime Objects
~~~~~~~~~~~~~~~~~~~~

``NR_waveforms.NR_simulation``
   Loads a selected catalogue simulation, builds ``NR_cpx_cut`` and
   ``NR_cpx_err_cut``, stores metadata and tracks peak-time conventions.

``template_waveforms.WaveformModel``
   Holds model configuration and evaluates the selected template on the NR time
   grid. It delegates Kerr, KerrBinary and TEOBPM physics to pyRing waveform
   classes.

``inference.Dynamic_InferenceModel``
   Builds a sampler-compatible class from either ``cpnest.model.Model`` or
   ``raynest.model.Model``. It creates the parameter names, bounds, fixed
   parameter dictionary, likelihood and prior.

``inference.Minimization_Algorithm``
   Runs bounded least squares with one or more starting points and estimates
   local errors from the inverse weighted Fisher matrix.

``inference.KerrLinearInversion_Algorithm``
   Solves the weighted normal equations for free Kerr complex amplitudes.

Testing Pointers
~~~~~~~~~~~~~~~~

The test suite currently covers utility functions, waveform utilities,
source-structure expectations, template-waveform behaviour and inference
helpers:

.. code-block:: bash

   pytest tests

The linear-inversion tests use fake waveform objects so the numerical method
can be checked without loading full NR data.
