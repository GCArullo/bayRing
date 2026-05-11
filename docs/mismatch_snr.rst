Mismatch & SNR Diagnostics
--------------------------

After inference, ``bayRing`` computes waveform mismatch and optimal-SNR
diagnostics using a detector-scaled strain, an ASD/PSD file and an
autocorrelation function. These diagnostics live in the post-processing path.
They are separate from the diagonal NR-error likelihood used during inference.

Physical Scaling
~~~~~~~~~~~~~~~~

The NR waveform is dimensionless. For mismatch and SNR diagnostics, it is
scaled to a source with remnant mass ``M`` and luminosity distance ``dL``:

.. container:: key-equation

   .. math::

      h_{\mathrm{phys}}(t) =
      \frac{G M_\odot M}{c^2\,d_L\,\mathrm{Mpc}}\,
      h_{\mathrm{NR}}(t).

The parameters are set in ``[Mismatch-GW-parameters]``:

.. code-block:: ini

   [Mismatch-GW-parameters]
   M = 60
   dL = 410
   ra = 1.375
   dec = -0.2108
   psi = 2.659

``ra``, ``dec`` and ``psi`` are used with LAL antenna-response utilities when
building plus/cross diagnostics.

PSD And ACF Inputs
~~~~~~~~~~~~~~~~~~

The diagnostic section starts from an ASD file:

.. code-block:: ini

   [Mismatch-PSD-settings]
   asd-path = /path/to/asd.txt

If ``asd-path`` is empty, the helper routines can use their default public ASD
path. The PSD/ASD is interpolated, optionally smoothed at the frequency edges,
and transformed into an autocorrelation function used for time-domain inner
products.

Smoothing Controls
~~~~~~~~~~~~~~~~~~

PSD edge smoothing is controlled by:

.. code-block:: ini

   [Flags]
   apply_window = 1
   C1_flag = 1

   [Mismatch-PSD-settings]
   direction = below-and-above
   window_DX = 0.8
   window_SX = 0.8
   steepness = 7.0
   saturation_DX = 1.0
   saturation_SX = 1.0

``direction`` selects which PSD edge is smoothed:

.. list-table::
   :header-rows: 1
   :widths: 24 56

   * - Value
     - Meaning
   * - ``below``
     - Smooth the low-frequency edge.
   * - ``above``
     - Smooth the high-frequency edge.
   * - ``below-and-above``
     - Smooth both edges.

The ``*_max`` and ``n_*`` settings define a grid over smoothing parameters.
For example:

.. code-block:: ini

   window_DX = 0.8
   window_DX_max = 10.0
   n_window_DX = 4

asks the post-processing loop to evaluate several low-frequency smoothing
window values between the minimum and maximum.

Time-Domain Inner Product
~~~~~~~~~~~~~~~~~~~~~~~~~

The diagnostic inner product uses the truncated ACF on the NR time interval.
For a residual or waveform vector ``x``, the whitened norm is computed through
a Toeplitz solve:

.. math::

   \|x\|_C^2 = x^T C^{-1} x,

where ``C`` is the Toeplitz covariance matrix implied by the ACF. The code uses
``scipy.linalg.solve_toeplitz`` for the time-domain norm.

The mismatch between NR and model waveforms has the usual normalized form:

.. container:: key-equation

   .. math::

      \mathcal{M} =
      1 - \frac{(h_{\mathrm{NR}},h_{\mathrm{model}})}
      {\sqrt{(h_{\mathrm{NR}},h_{\mathrm{NR}})
      (h_{\mathrm{model}},h_{\mathrm{model}})}}.

The optimal SNR diagnostic is:

.. math::

   \rho_{\mathrm{opt}} =
   \sqrt{(h,h)}.

Percentile Waveforms
~~~~~~~~~~~~~~~~~~~~

For posterior samples, the plotting utilities construct model-waveform
percentiles. The default mismatch/SNR files use the configured summary
percentiles and plot percentiles from ``bayRing.postprocess``:

.. code-block:: python

   summary_percentiles = (5, 50, 95)
   plot_percentiles = (50,)

For point-estimate methods, ``bayRing`` writes a Gaussian approximation to
``Algorithm/posterior.dat`` so the same percentile-based machinery can be used.

Output Files
~~~~~~~~~~~~

Mismatch and SNR text files are written under:

.. code-block:: text

   outdir/Algorithm/Mismatch/

File names encode mass, distance, start time, window settings and FFT length,
for example:

.. code-block:: text

   Mismatch_M_60_dL_410_t_s_30.0M_wDX_0.8Hz_wSX_0.8Hz_k_7.0_satDX_1.0_satSD_1.0_NFFT_4096.txt
   Optimal_SNR_M_60_dL_410_t_s_30.0M_wDX_0.8Hz_wSX_0.8Hz_k_7.0_satDX_1.0_satSD_1.0_NFFT_4096.txt

Each file stores confidence interval or percentile labels, the strain
component and the corresponding mismatch or SNR value.

Sanity Plots
~~~~~~~~~~~~

Enable diagnostic plots with:

.. code-block:: ini

   [Flags]
   mismatch_section_plot_flag = 1

Depending on the smoothing direction, plots are placed under one of:

.. code-block:: text

   Algorithm/Mismatch/Left_smoothing/
   Algorithm/Mismatch/Right_smoothing/
   Algorithm/Mismatch/Both_edges_smoothing/

These plots are useful when tuning window widths, steepness, saturation and
FFT length. Inspect them whenever changing PSD settings; a numerically smooth
ACF is not guaranteed by a syntactically valid configuration.

Frequency-Domain Checks
~~~~~~~~~~~~~~~~~~~~~~~

Set:

.. code-block:: ini

   [Flags]
   compare_TD_FD = 1

to request frequency-domain comparison outputs where implemented. This is a
diagnostic check on the PSD/ACF and time/frequency-domain treatments. It is
not a different inference likelihood.

Common Failure Modes
~~~~~~~~~~~~~~~~~~~~

Too long a time interval for the smoothing edge
   The code checks ``t_end - t_start < 1/(f_min + window_DX)`` when smoothing
   below the band. If violated, reduce the time interval or change the PSD
   smoothing settings.

Ill-conditioned ACF
   Inspect the PSD edge smoothing and FFT length. Enable
   ``mismatch_section_plot_flag`` and compare condition-number plots.
