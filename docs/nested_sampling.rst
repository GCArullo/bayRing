Nested Sampling
---------------

``Nested-sampler`` is the posterior and evidence calculation path. It is the
default method for production model comparison, uncertainty estimates and
prior-volume dependent statements.

Use :doc:`inference_methods` for the shared likelihood, prior and constraint
definitions used by all inference methods.

When To Use It
~~~~~~~~~~~~~~

Use nested sampling when the analysis requires:

* **Posterior samples:** credible intervals, correlations and multimodal
  structure.
* **Evidence:** model comparison through the nested-sampler log-evidence.
* **Prior-volume effects:** checks where posterior support depends on prior
  widths or non-rectangular constraints.

Nested sampling is slower than the point-estimate methods. The quick examples
therefore use small settings that are suitable for smoke tests, not final
science runs.

Minimal Configuration
~~~~~~~~~~~~~~~~~~~~~

Nested sampling is selected in ``[Inference]``:

.. code-block:: ini

   [Inference]
   method = Nested-sampler
   likelihood = gaussian
   sampler = cpnest
   nlive = 256
   maxmcmc = 256
   seed = 1234

The likelihood is evaluated on the same selected NR time samples described in
:doc:`inference_methods`.

Sampler Backends
~~~~~~~~~~~~~~~~

``sampler`` can be:

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Sampler
     - Behaviour
   * - ``cpnest``
     - Default sampler. Output is written under ``outdir/Algorithm`` and the
       posterior is read from ``posterior.dat``.
   * - ``raynest``
     - Alternative nested sampler. Uses ``nnest`` and ``nensemble`` for
       parallel nested-sampler and ensemble-process control.

Settings
~~~~~~~~

The most important nested-sampler controls are:

* **nlive:** number of live points. Increase this for production runs,
  broad priors or visibly structured posteriors.
* **maxmcmc:** maximum internal MCMC length for ``cpnest`` proposals.
* **seed:** random seed for reproducible diagnostic runs.
* **nnest:** number of nested-sampler processes used by ``raynest``.
* **nensemble:** number of ensemble processes used by ``raynest``.

The right ``nlive`` and ``maxmcmc`` values depend on model dimension, prior
volume, start time, error prescription and waveform degeneracies. A short test
can confirm the pipeline, but it does not establish sampler convergence.

Outputs
~~~~~~~

Nested-sampler outputs include:

* **Posterior samples:** written under the sampler output directory and used by
  post-processing.
* **Evidence:** saved by the sampler backend.
* **Prior-railing diagnostics:** generated when posterior support accumulates
  close to prior boundaries.
* **Waveform reconstructions:** produced during post-processing when waveform
  plotting is enabled.
* **Mismatch/SNR diagnostics:** produced when the corresponding options are
  enabled.

If posterior railing is detected, inspect the printed warning and
``Algorithm/Parameters_prior_railing*.txt`` before interpreting the result.

Validation Checklist
~~~~~~~~~~~~~~~~~~~~

Before using a nested-sampler result scientifically:

* **Prior boundaries:** check prior-railing diagnostics and corner plots.
* **Sampler settings:** rerun with larger ``nlive`` or ``maxmcmc`` when the
  posterior is broad, multimodal or near a boundary.
* **Fit window:** vary ``t-start`` or ``t-end`` to test ringdown stability.
* **Error model:** compare at least one alternative NR uncertainty
  prescription when available.
* **Residuals:** inspect waveform residuals, not only posterior summaries.
* **Evidence:** compare evidences only between runs with compatible data,
  likelihoods, priors and sampler settings.
