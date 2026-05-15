Install & Run
-------------

The source installation is the recommended workflow for most ``bayRing``
analyses because catalogue paths, fits and waveform-model experiments often
need local edits.

Developer Installation
~~~~~~~~~~~~~~~~~~~~~~

Clone the `bayRing repository <https://github.com/GCArullo/bayRing>`_ and
expose its location through ``BAYRING_PREFIX``:

.. code-block:: bash

   git clone https://github.com/GCArullo/bayRing.git
   cd bayRing
   export BAYRING_PREFIX=$PWD

For a permanent setup, add the export line to the shell startup file used on
your machine, for example ``~/.bashrc`` or ``~/.zshrc``.

Install Python dependencies and the package:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .

The editable install is useful for development because changes to files under
``bayRing/`` are picked up without reinstalling the package.

Pip Installation
~~~~~~~~~~~~~~~~

Tagged releases can be installed from PyPI:

.. code-block:: bash

   pip install bayRing

The PyPI release can be behind the source repository. Use it only when the
tracked release contains the model and catalogue features required by the
analysis.

pyRing Dependency
~~~~~~~~~~~~~~~~~

``bayRing`` uses ``pyRing`` waveform classes and utilities. The README
recommends installing the ``generalise_NR_informed_models`` branch of
``pyRing`` for source-based development; see the `pyRing installation
documentation
<https://lscsoft.docs.ligo.org/pyring/#install-and-run>`_.

First Run
~~~~~~~~~

From the repository root, run the simplest tracked example:

.. code-block:: bash

   bayRing --config-file config_files/config_SXS_0305_Kerr_220_quick.ini

This quick example fits the SXS ``0305`` waveform with a single Kerr
``(2,2,0)`` QNM, constant NR error, and deliberately small sampler settings:

.. code-block:: ini

   [Model]
   template = Kerr
   QNM-modes = 220

   [Inference]
   nlive   = 32
   maxmcmc = 32
   t-start = 30.0
   t-end   = 80.0

Use this run to verify installation and output structure. It is not a
production sampler configuration.

Running A Config File
~~~~~~~~~~~~~~~~~~~~~

Every run uses the same syntax:

.. code-block:: bash

   bayRing --config-file path/to/config.ini

Missing options use defaults.

Run Types
~~~~~~~~~

``run-type`` controls how much work is performed:

.. list-table::
   :header-rows: 1
   :widths: 22 58

   * - Value
     - Behaviour
   * - ``full``
     - Load NR data, run inference, save results and make diagnostics.
   * - ``post-processing``
     - Re-read previous results from ``outdir`` and remake downstream
       diagnostics.
   * - ``plot-NR-only``
     - Load the NR waveform and model object, make NR-only plots, then exit.
