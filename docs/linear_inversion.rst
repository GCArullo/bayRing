Linear Inversion
----------------

``Linear-inversion`` is a point-estimate method for Kerr models whose remaining
free parameters are complex amplitudes. It is designed for fast amplitude
extraction and for cross-checking nested-sampler or minimization results.

The repository also contains a method note:
:download:`Linear_inversion.pdf <linear_inversion_method.pdf>`.

When To Use It
~~~~~~~~~~~~~~

Use linear inversion when all of the following hold:

* **Template:** the waveform template is ``Kerr``.
* **Free parameters:** the only unknowns are amplitude/phase pairs.

Do not use it when any free parameter is nonlinear:

* damped-sinusoid frequencies;
* damped-sinusoid damping times;
* TEOB calibration coefficients;
* ``KerrBinary`` phase parameters;
* tail exponents.

Example
~~~~~~~

Run:

.. code-block:: bash

   bayRing --config-file \
     config_files/config_SXS_0305_Kerr_220_linear_inversion.ini

The example selects:

.. code-block:: ini

   [Model]
   template  = Kerr
   QNM-modes = 220,221,222,223

   [Inference]
   method = Linear-inversion
   t-start = 0.0
   t-end = 80.0
   linear-inversion-eigenvalue-tol = 1e-10

With no fixed amplitude/phase pairs in ``[Priors]``, the solver estimates all
listed complex amplitudes.

Linearisation Output
~~~~~~~~~~~~~~~~~~~~

The default tolerance is:

.. code-block:: ini

   [Inference]
   linear-inversion-eigenvalue-tol = 1e-10

The solver prints:

* **Solved amplitudes:** the number of complex amplitudes solved.
* **Cost:** the weighted least-squares cost.
* **Raw Fisher spectrum:** the eigenvalue range before regularization.
* **Tolerance:** the eigenvalue floor used in the solve.
* **Conditioning:** the condition number after regularization.

Interpretation:

* a large condition number means the selected basis is poorly conditioned;
* a result dominated by the eigenvalue floor means the data and error vector do
  not constrain the selected amplitudes cleanly.

Validation Checklist
~~~~~~~~~~~~~~~~~~~~

After a linear-inversion run:

* **Point estimates:** inspect ``Algorithm/point_estimates.dat``.
* **Waveform plots:** inspect waveform and residual reconstruction plots.
* **Method comparison:** compare against ``method = Minimization`` for the same
  fixed/free parameter set.
* **Posterior check:** run nested sampling if posterior structure or evidence
  matters.
* **Eigenvalue tolerance:** vary ``linear-inversion-eigenvalue-tol`` only as a
  numerical diagnostic, not as a fitting knob.
