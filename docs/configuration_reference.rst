Configuration Reference
-----------------------

The default value controls the cast applied to each option.

``bayRing`` configuration files are ``ini`` files parsed by
``configparser``. The parser starts from defaults defined in
``bayRing.initialise.read_config`` and overrides only the keys present in the
file.

Minimal Shape
~~~~~~~~~~~~~

A minimal run can be as short as:

.. code-block:: ini

   [I/O]
   outdir = my_run
   screen-output = 1

   [NR-data]
   error = constant-0.0001

   [Model]
   template = Kerr
   QNM-modes = 220

   [Inference]
   nlive = 32
   maxmcmc = 32
   t-start = 30.0
   t-end = 80.0

Missing sections are allowed when every key in that section can use its
default. The ``[Priors]`` section is optional unless you need custom bounds,
fixed values or minimization start values.

[I/O]
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 18 44

   * - Key
     - Default
     - Meaning
   * - ``run-type``
     - ``full``
     - ``full``, ``post-processing`` or ``plot-NR-only``.
   * - ``screen-output``
     - ``0``
     - ``0`` redirects stdout/stderr to files; ``1`` leaves output on the
       terminal.
   * - ``show-plots``
     - ``0``
     - Show Matplotlib windows after plotting.
   * - ``extract-damping-time-flag``
     - ``1``
     - Extract damping-time information from amplitude decay in comparison
       plots where supported.
   * - ``outdir``
     - ``./``
     - Output directory for samples, logs, plots and diagnostics.

[NR-data]
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 40

   * - Key
     - Default
     - Meaning
   * - ``download``
     - ``1``
     - Ask the catalogue reader to download data where implemented.
   * - ``dir``
     - empty string
     - Local NR data directory.
   * - ``catalog``
     - ``SXS``
     - Catalogue backend.
   * - ``ID``
     - ``0305``
     - Simulation identifier.
   * - ``extrap-order``
     - ``2``
     - SXS extrapolation order.
   * - ``res-level``
     - ``-1``
     - Resolution level. For SXS, ``-1`` selects the highest available level.
   * - ``res-nx``
     - ``0``
     - Teukolsky radial collocation count. Overrides ``res-level`` when paired
       with nonzero ``res-nl``.
   * - ``res-nl``
     - ``0``
     - Teukolsky angular collocation count.
   * - ``pert-order``
     - ``lin``
     - Teukolsky perturbation order.
   * - ``l-NR``
     - ``2``
     - Spherical NR multipole ``l`` to fit.
   * - ``m``
     - ``2``
     - Spherical NR multipole ``m`` to fit.
   * - ``error``
     - ``align-with-mismatch-res-only``
     - NR error prescription.
   * - ``error-t-min``
     - ``3e-1``
     - Lower fractional mismatch-alignment window input.
   * - ``error-t-max``
     - ``4e-3``
     - Upper fractional mismatch-alignment window input.
   * - ``add-const``
     - ``0.0,0.0``
     - Complex constant offset encoded as amplitude,phase.
   * - ``properties-file``
     - empty string
     - Optional CSV with additional simulation properties.
   * - ``fits-file``
     - empty string
     - Optional CSV with model-fit coefficients.
   * - ``t-peak-22``
     - ``0.0``
     - Explicit ``(2,2)`` peak time. Nonzero values override peak detection.
   * - ``waveform-type``
     - ``strain``
     - Waveform quantity to read where supported.

[Model]
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 24 32

   * - Key
     - Default
     - Meaning
   * - ``template``
     - ``Kerr``
     - Waveform family.
   * - ``N-DS-modes``
     - ``1``
     - Number of damped-sinusoid modes.
   * - ``QNM-modes``
     - ``220,221,320``
     - Linear Kerr QNM list.
   * - ``QQNM-modes``
     - empty string
     - Quadratic QNM coupling list.
   * - ``Kerr-tail``
     - ``0``
     - Add Kerr tail terms.
   * - ``Kerr-tail-modes``
     - ``22``
     - Tail mode list written as ``lm`` strings.
   * - ``KerrBinary-version``
     - ``London2018``
     - ``London2018``, ``Cheung2023`` or ``noncircular``.
   * - ``KerrBinary-amplitudes-nc-version``
     - empty string
     - Noncircular amplitude-correction variables.
   * - ``TEOB-template``
     - ``HypTan``
     - TEOB post-merger template branch, either ``HypTan`` or ``RatExp``.
   * - ``TEOB-global-fit``
     - ``1``
     - Use calibrated TEOB global fits. If ``0``, sample local TEOB
       calibration coefficients.
   * - ``TEOB-merger-data``
     - ``0``
     - Use NR merger peak quantities from ``properties-file`` instead of the
       quasi-circular peak fits.

Some templates override ``QNM-modes`` internally. For example ``KerrBinary``
and ``TEOBPM`` set their calibrated mode lists based on the selected template
version.

[Inference]
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 20 36

   * - Key
     - Default
     - Meaning
   * - ``method``
     - ``Nested-sampler``
     - ``Nested-sampler``, ``Minimization`` or ``Linear-inversion``.
       ``Linear-inversion`` is available only for ``Kerr`` models whose free
       parameters are paired amplitude/phase variables.
   * - ``likelihood``
     - ``gaussian``
     - ``gaussian`` or ``laplace``.
   * - ``sampler``
     - ``cpnest``
     - Nested sampler: ``cpnest`` or ``raynest``.
   * - ``nlive``
     - ``256``
     - Number of live points.
   * - ``maxmcmc``
     - ``256``
     - Maximum MCMC steps for nested sampling.
   * - ``seed``
     - ``1234``
     - Random seed.
   * - ``nnest``
     - ``1``
     - Number of raynest nested samplers.
   * - ``nensemble``
     - ``1``
     - Number of raynest ensemble processes.
   * - ``t-start``
     - ``20.0``
     - Fit start time in ``M`` relative to the selected peak.
   * - ``t-end``
     - ``140.0``
     - Fit end time in ``M`` relative to the selected peak.
   * - ``dt-scd``
     - ``0.0``
     - Delay used for Teukolsky second-order peak conventions.
   * - ``min-method``
     - ``trf``
     - SciPy least-squares method: ``trf`` or ``dogbox``.
   * - ``min-iter-max``
     - ``1000``
     - Maximum least-squares function evaluations per seed.
   * - ``n-random-seeds``
     - ``16``
     - Number of minimization starts.
   * - ``linear-inversion-eigenvalue-tol``
     - ``1e-10``
     - Fisher eigenvalue floor for linear inversion.

When ``method`` is ``Minimization`` or ``Linear-inversion``, ``nlive`` and
``maxmcmc`` are set to ``None`` after parsing because no nested sampler is run.

[Priors]
~~~~~~~~

The ``[Priors]`` section has no fixed list of keys. It uses parameter names
created by the selected waveform model.

Override bounds:

.. code-block:: ini

   [Priors]
   ln_A_220-min = -12.0
   ln_A_220-max = 0.0

Fix values:

.. code-block:: ini

   [Priors]
   fix-ln_A_220 = -5.0
   fix-phi_220 = 1.0

Set minimization start values:

.. code-block:: ini

   [Priors]
   ln_A_220-start = -6.0
   phi_220-start = 3.14

Parameter names are documented in :doc:`waveform_models` and printed at
startup.

[Mismatch-PSD-settings]
~~~~~~~~~~~~~~~~~~~~~~~

These settings control post-inference mismatch and optimal-SNR diagnostics.
They do not define the inference likelihood.

.. list-table::
   :header-rows: 1
   :widths: 30 20 40

   * - Key
     - Default
     - Meaning
   * - ``asd-path``
     - empty string
     - ASD file path. The code can use its default public ASD path when empty.
   * - ``obs-time``
     - ``0.0``
     - Observation time in seconds. If empty/zero, inferred from the PSD grid.
   * - ``direction``
     - ``below-and-above``
     - Smooth below, above or both PSD edges.
   * - ``window_DX``, ``window_DX_max``
     - ``0.8``, ``10.0``
     - Low-frequency smoothing window range.
   * - ``window_SX``, ``window_SX_max``
     - ``0.8``, ``10.0``
     - High-frequency smoothing window range.
   * - ``n_window_DX``, ``n_window_SX``
     - ``1``, ``1``
     - Number of grid values for the two window ranges.
   * - ``steepness``, ``steepness_max``
     - ``7.0``, ``200.0``
     - Smoothing steepness range.
   * - ``n_steepness``
     - ``1``
     - Number of steepness grid values.
   * - ``saturation_DX``, ``saturation_DX_max``
     - ``1.0``, ``5.0``
     - Low-frequency saturation range.
   * - ``saturation_SX``, ``saturation_SX_max``
     - ``1.0``, ``5.0``
     - High-frequency saturation range.
   * - ``n_saturation_DX``, ``n_saturation_SX``
     - ``1``, ``1``
     - Number of saturation grid values.
   * - ``n_FFT_points``
     - ``1``
     - Number of FFT lengths to scan.
   * - ``n_iterations_C1``
     - ``1``
     - Number of C1 smoothing iterations.

[Mismatch-GW-parameters]
~~~~~~~~~~~~~~~~~~~~~~~~

These settings set the physical strain scale and detector response quantities
used in mismatch/SNR diagnostics.

.. list-table::
   :header-rows: 1
   :widths: 20 18 52

   * - Key
     - Default
     - Meaning
   * - ``M``
     - ``60``
     - Remnant mass in solar masses.
   * - ``dL``
     - ``410``
     - Luminosity distance in Mpc.
   * - ``ra``
     - ``1.375``
     - Right ascension in radians.
   * - ``dec``
     - ``-0.2108``
     - Declination in radians.
   * - ``psi``
     - ``2.659``
     - Polarization angle in radians.

[Flags]
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 16 44

   * - Key
     - Default
     - Meaning
   * - ``apply_window``
     - ``1``
     - Apply PSD edge smoothing.
   * - ``C1_flag``
     - ``1``
     - Apply C1 fixing after smoothing.
   * - ``clear_directory``
     - ``1``
     - Clear mismatch smoothing-output subdirectories before running.
   * - ``compare_TD_FD``
     - ``0``
     - Also compute frequency-domain checks where implemented.
   * - ``mismatch_print_flag``
     - ``0``
     - Print extra mismatch diagnostic information.
   * - ``mismatch_section_plot_flag``
     - ``0``
     - Save PSD/ACF/window sanity plots.
