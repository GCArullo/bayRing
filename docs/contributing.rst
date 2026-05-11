Contributing
------------

Issues, feature requests and bug reports should be opened on the project
repository. Code changes should follow the usual branch or fork workflow:

1. Create a branch for the change.
2. Make the smallest coherent code and documentation edits.
3. Add or update a runnable config or test when the behaviour changes.
4. Run the relevant tests.
5. Submit a merge request or pull request with the scientific and software
   scope stated explicitly.

Development Setup
~~~~~~~~~~~~~~~~~

Use an editable source install:

.. code-block:: bash

   git clone https://github.com/GCArullo/bayRing.git
   cd bayRing
   export BAYRING_PREFIX=$PWD
   pip install -r requirements.txt
   pip install -e .

Run tests with:

.. code-block:: bash

   pytest tests

Build docs with:

.. code-block:: bash

   cd docs
   make html

Adding A New Waveform
~~~~~~~~~~~~~~~~~~~~~

The source-module workflow is:

1. Add the model to the accepted template choices in
   ``bayRing.initialise.read_config``.
2. Add compatibility checks in ``bayRing.initialise.read_config`` where the
   model has catalogue, multipole or metadata requirements.
3. Define default prior bounds in ``bayRing.inference.read_default_bounds``.
4. Add parameter-name construction and fixed-parameter handling in
   ``bayRing.inference.Dynamic_InferenceModel``.
5. Add waveform construction in ``bayRing.template_waveforms.WaveformModel``.
6. Add the selected model branch to ``WaveformModel.waveform``.
7. Add at least one runnable config under ``config_files``.
8. Add focused tests for parser behaviour and waveform-parameter construction
   when practical.
9. Document the model in :doc:`waveform_models` and update
   :doc:`configuration_reference`.

Adding A New NR Catalogue
~~~~~~~~~~~~~~~~~~~~~~~~~

A catalogue integration should be explicit about data layout, metadata and
error estimates. The implementation path is:

1. Add the catalogue name to parser documentation and validation in
   ``bayRing.initialise.read_config``.
2. Add catalogue-specific loading to ``bayRing.NR_waveforms.NR_simulation``.
3. Add a metadata dictionary branch in
   ``bayRing.NR_waveforms.read_NR_metadata``.
4. Add helper readers for local files, metadata tables or downloaded files in
   ``bayRing.NR_waveforms``.
5. Add an NR error prescription that is scientifically meaningful for that
   catalogue.
6. Add a runnable config under ``config_files``.
7. Document local data path requirements in :doc:`nr_data`.
