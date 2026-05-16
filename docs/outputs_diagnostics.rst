Outputs & Diagnostics
---------------------

Every run writes a structured output directory. The exact contents depend on
``run-type``, inference method, sampler and enabled diagnostics.

Directory Layout
~~~~~~~~~~~~~~~~

At startup, ``bayRing`` creates:

.. code-block:: text

   outdir/
     Algorithm/
       Mismatch/
     Peak_quantities/
     Plots/
       Chains/
       Comparisons/
       Results/

For nested-sampler runs, ``Plots/Chains`` is used for sampler-chain plots.
``Plots/Comparisons`` stores waveform and residual reconstruction figures.
``Plots/Results`` stores posterior summaries such as corner plots.

Run Metadata
~~~~~~~~~~~~

The output directory stores run provenance through pyRing's ``store_git_info``
helper. For ``run-type = full``, the input config file is copied into the
output directory when possible.

If ``screen-output = 0``, the terminal output is redirected to:

.. code-block:: text

   stdout_bayRing.txt
   stderr_bayRing.txt

These files are often the first place to check on a remote job.

Algorithm Outputs
~~~~~~~~~~~~~~~~~

Nested-sampler runs write sampler products under ``Algorithm``. Important
files include:

.. list-table::
   :header-rows: 1
   :widths: 30 50

   * - File
     - Meaning
   * - ``Algorithm/posterior.dat``
     - Posterior samples used by downstream plotting and summaries.
   * - ``Algorithm/Evidence.txt``
     - Log evidence stored as ``logZ``.
   * - ``Algorithm/cpnest.log``
     - CPNest log when ``sampler = cpnest``.
   * - ``Algorithm/raynest.log`` or ``Algorithm/raynest.h5``
     - Raynest products when ``sampler = raynest``.
   * - ``Algorithm/Parameters_prior_railing.txt``
     - Flags for posterior samples railing against prior bounds.

Point-estimate methods also write:

.. list-table::
   :header-rows: 1
   :widths: 30 50

   * - File
     - Meaning
   * - ``Algorithm/point_estimates.dat``
     - Best-fit values and one-sigma error estimates.
   * - ``Algorithm/posterior.dat``
     - Optional synthetic Gaussian posterior approximation centred on the point
       estimate, written only when
       ``point-estimate-posterior-samples > 0``.

The synthetic point-estimate posterior is not a replacement for nested
sampling. It is a convenience product for diagnostic plots.

Peak Quantities
~~~~~~~~~~~~~~~

``Peak_quantities`` stores parameter estimates translated to peak-time
conventions where supported by post-processing.

Comparison Plots
~~~~~~~~~~~~~~~~

Waveform and residual comparisons are saved under:

.. code-block:: text

   Plots/Comparisons/

Common files include:

.. list-table::
   :header-rows: 1
   :widths: 36 44

   * - File
     - Meaning
   * - ``Waveform_reconstruction.pdf``
     - NR waveform and model reconstruction.
   * - ``Residuals_reconstruction.pdf``
     - NR-model residuals.
   * - ``Decay_rate.pdf``
     - Decay-rate diagnostic where available.
   * - ``Waveform_reconstruction_no_tail_format.pdf``
     - Additional no-tail-format figure for tail runs where produced.

Posterior Plots
~~~~~~~~~~~~~~~

For nested-sampler runs, ``bayRing`` attempts a global corner plot:

.. code-block:: text

   Plots/Results/corner.pdf

CPNest posterior PDFs and chain plots are moved from ``Algorithm`` into
``Plots/Results`` and ``Plots/Chains`` when the sampler produces them.

Mismatch & SNR Outputs
~~~~~~~~~~~~~~~~~~~~~~

Mismatch and optimal-SNR products are written under:

.. code-block:: text

   Algorithm/Mismatch/

See :doc:`mismatch_snr` for the diagnostics table layout and PSD/ACF controls.

Post-Processing Existing Runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To regenerate diagnostics without rerunning inference:

.. code-block:: ini

   [I/O]
   run-type = post-processing
   outdir = existing_run_directory

   [Inference]
   method = Nested-sampler
   sampler = cpnest

The method and sampler still matter because they tell post-processing how to
read the saved results. For point-estimate methods, post-processing reads
``Algorithm/point_estimates.dat`` by default and uses ``Algorithm/posterior.dat``
only when ``point-estimate-posterior-samples > 0``.

NR-Only Plotting
~~~~~~~~~~~~~~~~

To load NR data and plot the waveform without running inference:

.. code-block:: ini

   [I/O]
   run-type = plot-NR-only

This path creates the model and NR simulation object, makes reconstruction
plots with no posterior object, prints a message and exits.
