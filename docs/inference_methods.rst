Inference Methods
-----------------

``bayRing`` supports three inference paths:

.. code-block:: ini

   [Inference]
   method = Nested-sampler

Allowed values are ``Nested-sampler``, ``Minimization`` and
``Linear-inversion``.

.. toctree::
   :maxdepth: 1

   nested_sampling
   minimization
   linear_inversion

Likelihood
~~~~~~~~~~

The inference model compares a complex NR waveform ``d`` with a complex model
``h(theta)``. The residual is:

.. math::

   r(\theta) = d - h(\theta).

For the default Gaussian likelihood, real and imaginary residuals are weighted
by the corresponding real and imaginary parts of the NR error vector:

.. container:: key-equation

   .. math::

      \log L(\theta) =
      -\frac{1}{2}\sum_i
      \left[
      \left(\frac{\mathrm{Re}[r_i(\theta)]}{\mathrm{Re}[\sigma_i] + \epsilon}\right)^2
      +
      \left(\frac{\mathrm{Im}[r_i(\theta)]}{\mathrm{Im}[\sigma_i] + \epsilon}\right)^2
      \right],

where ``epsilon = 1e-16`` avoids division by zero. This is a diagonal
time-domain likelihood on the selected NR samples. PSD/ACF settings are used
later by mismatch and optimal-SNR diagnostics, not by this inference
likelihood.

Priors And Bounds
~~~~~~~~~~~~~~~~~

Default bounds are model dependent and are defined in
``bayRing.inference.read_default_bounds``. Override a bound in ``[Priors]``:

.. code-block:: ini

   [Priors]
   ln_A_220-min = -12.0
   ln_A_220-max = 0.0
   phi_220-min = 0.0
   phi_220-max = 6.283185307179586

Fix a parameter with:

.. code-block:: ini

   [Priors]
   fix-ln_A_220 = -5.0
   fix-phi_220 = 1.0

The fixed-value syntax removes that parameter from the free parameter list.
The waveform model still sees the value through the fixed-parameter override.

Non-Rectangular Prior Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some constraints are enforced in addition to simple rectangular bounds:

.. list-table::
   :header-rows: 1
   :widths: 28 52

   * - Model feature
     - Constraint
   * - ``Damped-sinusoids``
     - Frequencies are ordered by mode index.
   * - Kerr tails
     - Tail exponents are ordered relative to the fitted NR multipole tail.

Nested sampling applies these constraints in ``log_prior`` by returning
``-inf`` when the constraint is violated. Minimization adds large penalty
residuals so the least-squares objective respects the same shape.

Choosing A Method
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 52

   * - Goal
     - Recommended method
   * - Quick check of a nonlinear model
     - ``Minimization``
   * - Fast Kerr amplitude extraction
     - ``Linear-inversion`` if the template is ``Kerr`` and all free
       parameters are paired amplitudes/phases.
   * - Posterior intervals and evidence
     - ``Nested-sampler``
   * - Production model comparison
     - ``Nested-sampler`` with raised ``nlive``/``maxmcmc`` and stability
       checks.
   * - Debugging priors and start times
     - ``Minimization`` followed by a nested-sampler run if the solution is
       physically meaningful.

Point-estimate methods are excellent diagnostics, but they do not replace a
posterior calculation when uncertainties, prior-volume effects or evidences
matter.
