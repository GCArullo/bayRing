Minimization
------------

``Minimization`` is a point-estimate method based on bounded nonlinear
least-squares. It is useful for fast checks of model behaviour, start-time
choices and prior setup before committing to a nested-sampler run.

Use :doc:`inference_methods` for the shared likelihood, prior and constraint
definitions used by all inference methods.

Minimal Configuration
~~~~~~~~~~~~~~~~~~~~~

The minimization path is selected in ``[Inference]``:

.. code-block:: ini

   [Inference]
   method = Minimization
   min-method = trf
   min-iter-max = 1000
   n-random-seeds = 16

Allowed ``min-method`` values are ``trf`` and ``dogbox``, passed to
``scipy.optimize.least_squares``.

Residual Vector
~~~~~~~~~~~~~~~

The residual vector contains:

.. code-block:: text

   real weighted data-model residuals
   imaginary weighted data-model residuals
   optional penalty residuals for non-rectangular priors

Real and imaginary components are weighted by the NR uncertainty vector
described in :doc:`inference_methods`. Non-rectangular prior constraints are
added as large penalty residuals, so the optimizer sees them as part of the
least-squares objective.

Starting Points
~~~~~~~~~~~~~~~

Starting points are read from ``[Priors]`` when present:

.. code-block:: ini

   [Priors]
   ln_A_220-start = -4.0
   phi_220-start = 3.0

If no start value is supplied, ``bayRing`` draws random starts uniformly within
the parameter bounds. When ``n-random-seeds > 1``, the first starting point is
the midpoint of the bounds and the remaining points are random. The best
least-squares result by cost is kept.

Solver Controls
~~~~~~~~~~~~~~~

Important minimization controls are:

* **min-method:** SciPy least-squares backend, either ``trf`` or
  ``dogbox``.
* **min-iter-max:** maximum number of solver iterations.
* **n-random-seeds:** number of starting points tried before keeping the
  best result.
* **\*-start prior entries:** explicit starting values for selected free
  parameters.

Increasing ``n-random-seeds`` is useful when the objective is multimodal or
when phase parameters create several locally plausible solutions.

Error Estimate
~~~~~~~~~~~~~~

After minimization, the code estimates parameter errors from the inverse local
Fisher matrix:

.. math::

   C \simeq (J^T J)^{-1},

where ``J`` is the weighted residual Jacobian returned by SciPy. Eigenvalues
are regularized through the same symmetric-inverse helper used by linear
inversion.

This covariance is a local approximation around the best fit. It does not
replace a posterior calculation when the likelihood is non-Gaussian, the
solution is close to a bound, or several modes in parameter space are relevant.

Outputs
~~~~~~~

Minimization runs write point-estimate products compatible with the usual
post-processing path:

* **Point estimates:** best-fit parameter values and local uncertainties.
* **Synthetic posterior samples:** Gaussian approximations used by plotting
  utilities.
* **Waveform reconstructions:** fitted waveform and residual plots when
  plotting is enabled.
* **Mismatch/SNR diagnostics:** available through the same diagnostic switches
  used by other methods.

Validation Checklist
~~~~~~~~~~~~~~~~~~~~

After a minimization run:

* **Cost comparison:** check whether different seeds converge to the same
  solution.
* **Boundaries:** inspect whether best-fit parameters sit on prior bounds.
* **Residuals:** verify that residuals are structureless over the fitted
  interval.
* **Method comparison:** compare with ``Nested-sampler`` for any result used in
  a final interpretation.
* **Window stability:** vary the fit window before trusting trends in fitted
  parameters.
