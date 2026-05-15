Tutorials & Examples
--------------------

Runnable examples live in ``config_files``. Run any example with:

.. code-block:: bash

   bayRing --config-file config_files/name_of_config.ini

All configuration options are discoverable from the command-line interface:

.. code-block:: bash

   bayRing --help

Core Usage Options
~~~~~~~~~~~~~~~~~~

``catalog``
   Selects the NR data backend. The common tracked examples use ``SXS``,
   ``RIT``, ``RWZ-env``, ``Teukolsky``, ``C2EFT`` and ``cbhdb``.

``l-NR`` and ``m``
   Select the spherical multipole of the NR waveform that is fitted. These do
   not have to match every QNM ``l`` in the model because spheroidal QNMs with
   the same ``m`` can contribute to a spherical NR multipole.

``template``
   Selects the waveform family used in the fit. Common options include
   ``Kerr``, ``Damped-sinusoids``, ``Kerr-Damped-sinusoids``, ``KerrBinary``
   and ``TEOBPM``.

``QNM-modes``
   Selects the linear Kerr QNMs used by the ``Kerr`` template. A mode is
   written as three digits ``lmn``; negative ``m`` is written with a minus sign,
   for example ``2-20``.

``method``
   Selects ``Nested-sampler``, ``Minimization`` or ``Linear-inversion``.
   Nested sampling returns posterior samples and evidence; the other methods
   produce point estimates and synthetic posterior samples for downstream
   plotting. ``Linear-inversion`` is available only for ``Kerr`` models whose
   free parameters are paired amplitude/phase variables; nonlinear parameters
   must be fixed or absent.

``t-start`` and ``t-end``
   Select the fit interval in units of the total mass ``M``. The interval is
   usually measured relative to the peak of the selected waveform amplitude.

``error``
   Selects how the uncertainty assigned to the NR waveform is computed.
   Options depend on the selected catalogue. The simplest quick examples use a
   constant complex error such as ``constant-0.0001``.

Quick Versus Production Configs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Files containing ``quick`` are intentionally light. They usually set a
constant NR error, narrower fit windows, or low live-point counts.

Files without ``quick`` are still examples, but they are closer to the settings
used for a substantive run. Always inspect the model, priors, NR error and
time interval before using an example as a production config.

Point-Estimate Examples
~~~~~~~~~~~~~~~~~~~~~~~

Two examples run without nested sampling:

.. list-table::
   :header-rows: 1
   :widths: 35 55

   * - File
     - Description
   * - ``config_SXS_0305_Kerr_220_minimization.ini``
     - Uses ``method = Minimization`` and solves the bounded least-squares
       problem with multiple starting seeds.
   * - ``config_SXS_0305_Kerr_220_linear_inversion.ini``
     - Uses ``method = Linear-inversion`` and solves directly for free complex
       Kerr amplitudes. Nonlinear parameters must be fixed or absent.

Both methods save point estimates and synthetic posterior samples so the usual
post-processing functions can run.

Tracked Config Files
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 36 54

   * - File
     - Description
   * - ``config_SXS_0305_Kerr_220_quick.ini``
     - Fast SXS ``0305`` Kerr ``(2,2,0)`` run with constant NR error.
   * - ``config_SXS_0305_Kerr_220.ini``
     - Same physical setup as the quick Kerr example, with less aggressive
       example settings.
   * - ``config_SXS_0305_DS_1_quick.ini``
     - SXS ``0305`` agnostic one-mode damped-sinusoid fit.
   * - ``config_SXS_0305_Kerr_220_minimization.ini``
     - Point-estimate least-squares Kerr ``(2,2,0)`` fit.
   * - ``config_SXS_0305_Kerr_220_linear_inversion.ini``
     - Linear-inversion Kerr example over ``(2,2,n)`` overtones.
   * - ``config_SXS_0305_KerrBinary_London_22_quick.ini``
     - ``KerrBinary`` with the London 2018 calibration on the ``(2,2)``
       multipole.
   * - ``config_SXS_0305_KerrBinary_London_HMs_quick.ini``
     - ``KerrBinary`` London example for a higher multipole.
   * - ``config_SXS_0305_KerrBinary_Cheung_33_quick.ini``
     - ``KerrBinary`` Cheung 2023 calibration on the ``(3,3)`` multipole.
   * - ``config_SXS_0305_KerrBinary_Cheung_22_30M_PSD_window.ini``
     - ``KerrBinary`` example with mismatch PSD/window controls enabled.
   * - ``config_SXS_0305_TEOBPM_22_quick.ini``
     - TEOB post-merger model on the ``(2,2)`` multipole.
   * - ``config_SXS_0305_TEOBPM_HMs_quick.ini``
     - TEOB post-merger model on a higher multipole.
   * - ``config_RIT_1300_Kerr_220_quick.ini``
     - RIT catalogue Kerr ``(2,2,0)`` example.
   * - ``config_RIT_1339_KerrBinary_noncircular_220_quick.ini``
     - RIT ``KerrBinary`` Carullo2024 example using additional metadata.
   * - ``config_RWZ-env_001_DS_2_quick.ini``
     - Not public NR data. RWZ-environment damped-sinusoid example with
       ``m = 0``.
   * - ``config_Teukolsky_a_0.7_Kerr_220_lin_quick.ini``
     - Not public NR data. Teukolsky linear-perturbation Kerr ``(2,2,0)``
       quick run.
   * - ``config_Teukolsky_a_0.7_Kerr_220_lin.ini``
     - Not public NR data. Teukolsky Kerr example with several linear modes,
       including negative ``m``.
   * - ``config_Teukolsky_a_0.7_Kerr_220_quad.ini``
     - Not public NR data. Teukolsky quadratic-mode example for an
       ``l = 4, m = 4`` NR multipole.
   * - ``config_C2EFT_001_Kerr_220_quick.ini``
     - Not public NR data. C2EFT Kerr quick example.
   * - ``config_C2EFT_001_Kerr_220.ini``
     - Not public NR data. C2EFT Kerr example with broader settings.
   * - ``config_cbhdb_000001_Kerr_220_quick.ini``
     - Not public NR data. Charged black-hole database example.
