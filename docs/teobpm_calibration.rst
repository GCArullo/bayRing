TEOBPM SXS calibration workflow
===============================

``bayRing-teobpm-calibrate`` prepares and analyses TEOBPM calibration campaigns
against non-eccentric, non-precessing SXS simulations. It is an orchestration
layer above normal ``bayRing --config-file`` runs: it writes manifests and one
standard bayRing config per selected simulation/mode, then the expensive local
fits can be launched either from the generated shell script or through the
campaign runner.

The workflow supports two families:

* ``nonspinning``: filters to simulations with negligible aligned and transverse
  spins;
* ``aligned-spin``: requires a completed nonspinning global-fit JSON and fits
  only spin corrections on top of that base layer.

Typical campaign preparation
----------------------------

.. code-block:: bash

   bayRing-teobpm-calibrate prepare \
     --family nonspinning \
     --output-dir runs/teobpm_nonspinning \
     --modes 22,21,33,32,44,43 \
     --template RatExp \
     --t-start 0

The command writes:

* ``campaign_config.json`` with the parsed campaign settings;
* ``manifests/dataset_manifest.json`` with the train/validation assignment;
* ``local_fit_index.csv`` with one indexed job per selected simulation/mode;
* ``local_fit_configs/*.ini`` with one normal bayRing config per fit;
* ``run_local_fits.sh`` with the corresponding ``bayRing`` commands.

Calibration configs follow pyRing's current TEOBPM time convention. The
``(2,2)`` peak is the reference, and mode ``lm`` starts at
``t_peak_22 + DeltaT_lm(q, chi1, chi2)``. The generated ``t-start`` is therefore
``0`` for ``22`` and pyRing's ``DeltaT_lm`` for higher modes; ``--t-start`` adds
an extra offset on top of that convention.

This per-mode start is only used while constructing local fit coefficients.
Once coefficients exist, waveform mismatch evaluation for local-fit plots and
global-fit checks must compare from the ``(2,2)`` peak with ``t-start = 0`` and
``tref = peak22``. Higher modes can therefore be empty over the initial
``0 <= t < DeltaT_lm`` part of the comparison window.

Local fits and collection
-------------------------

Run the indexed ``22`` local fits first with:

.. code-block:: bash

   bayRing-teobpm-calibrate run-local-fits \
     --campaign-dir runs/teobpm_nonspinning \
     --split training \
     --modes 22 \
     --workers 4

Then fill the higher-mode metadata and run every higher mode whose dependencies
are available. This can include ``32`` because its mixed-mode parent is ``22``;
hold back ``43`` until ``33`` has completed:

.. code-block:: bash

   bayRing-teobpm-calibrate fill-hm-inputs \
     --campaign-dir runs/teobpm_nonspinning

   bayRing-teobpm-calibrate run-local-fits \
     --campaign-dir runs/teobpm_nonspinning \
     --split training \
     --modes 21,33,32,31,44,42,41,55 \
     --workers 4

After ``33`` finishes, fill the mixed ``43`` parent coefficients and run ``43``
as its own parallel simulation batch:

.. code-block:: bash

   bayRing-teobpm-calibrate fill-hm-inputs \
     --campaign-dir runs/teobpm_nonspinning

   bayRing-teobpm-calibrate run-local-fits \
     --campaign-dir runs/teobpm_nonspinning \
     --split training \
     --modes 43 \
     --workers 4

The runner skips jobs that already contain
``Algorithm/point_estimates.dat`` unless ``--force`` is passed. It writes
``local_fit_run_summary.csv`` and per-job stdout/stderr logs under the local-fit
output directories. The ``--split`` option can be used to run the training jobs
first and the validation jobs only after the global fit has been constructed.

After the jobs complete, collect the bayRing point estimates and construction
mismatch files into the long-form table consumed by the global-fit step:

.. code-block:: bash

   bayRing-teobpm-calibrate collect-local-fits \
     --campaign-dir runs/teobpm_nonspinning \
     --split training

The collection step writes ``local_fit_summary.csv``, ``mismatch_summary.csv``,
``local_fit_collection_failures.csv`` and
``local_fit_collection_summary.json``. It also constructs relative mode phases
``delta_phi_lm`` from the local ``phi_mrg_lm`` estimates using the ``22`` mode
as reference. Representative construction mismatches are stored as
``construction_mismatch`` in ``local_fit_summary.csv`` so they are not mistaken
for 22-peak evaluation mismatches. Re-run collection with ``--split all`` after
validation local fits are available if the same table should drive both
global-fit construction and validation diagnostics.

Local-fit plots
---------------

Plot the collected local coefficients with:

.. code-block:: bash

   bayRing-teobpm-calibrate plot-local-fits \
     --local-fit-table runs/teobpm_nonspinning/local_fit_summary.csv \
     --output-dir runs/teobpm_nonspinning/local_fit_plots

To include waveform mismatch comparisons in the local-fit plot directory, pass
one or more explicit evaluation mismatch tables. Those tables must be generated
after the local fit coefficients exist and with the comparison starting at the
``22`` peak:

.. code-block:: bash

   bayRing-teobpm-calibrate plot-local-fits \
     --local-fit-table runs/teobpm_nonspinning/local_fit_summary.csv \
     --mismatch-table runs/teobpm_nonspinning/local_fit_evaluation_mismatches.csv \
     --label "Local fit evaluation" \
     --output-dir runs/teobpm_nonspinning/local_fit_plots

Global fits
-----------

Construct the versioned pyRing-ingestable file from the collected table with:

.. code-block:: bash

   bayRing-teobpm-calibrate global-fit \
     --family nonspinning \
     --local-fit-table runs/teobpm_nonspinning/local_fit_summary.csv \
     --output-file runs/teobpm_nonspinning/teobpm_global_fit.json

Aligned-spin fits must point to the nonspinning fit:

.. code-block:: bash

   bayRing-teobpm-calibrate global-fit \
     --family aligned-spin \
     --base-nonspinning-file runs/teobpm_nonspinning/teobpm_global_fit.json \
     --local-fit-table runs/teobpm_aligned/local_fit_summary.csv \
     --output-file runs/teobpm_aligned/teobpm_global_fit.json

Validation and reporting
------------------------

Validation tables use the same long-form coefficient columns. The validation
command writes prediction residuals and parameter-space/histogram diagnostics.
Pass explicit evaluation mismatch tables to add waveform mismatch views over
the calibration coordinates: spinning rows are shown over ``nu``--``chi_eff``
and nonspinning rows are shown against ``nu``. These evaluation mismatch tables
must be produced with the comparison starting at the ``22`` peak:

.. code-block:: bash

   bayRing-teobpm-calibrate validate \
     --global-fit-file runs/teobpm_aligned/teobpm_global_fit.json \
     --validation-table runs/teobpm_aligned/validation_summary.csv \
     --mismatch-table runs/teobpm_aligned/global_evaluation_mismatches.csv \
     --label "New global fit" \
     --output-dir runs/teobpm_aligned/validation

Fixed-coefficient mismatch campaigns can be compared with the same plotting
convention. The command accepts one or more CSV tables containing ``nu``,
``chi_eff`` or equivalent spin columns, and either a representative ``mismatch``
column or wide comparison columns such as ``new_global_mismatch`` and
``existing_teobpm_mismatch``. If the table declares ``t_start``/``t-start`` or
``tref``/``reference_time`` metadata, those values are checked against
``t_start = 0`` and ``tref = peak22``:

.. code-block:: bash

   bayRing-teobpm-calibrate plot-mismatch-comparison \
     --mismatch-table runs/teobpm_aligned/new_global_mismatches.csv \
     --label "New global fit" \
     --mismatch-table runs/teobpm_aligned/current_teobpm_mismatches.csv \
     --label "Existing TEOBPM global fit" \
     --output-dir runs/teobpm_aligned/mismatch_comparison

The command writes point-level and summary CSV files plus
``teobpm_mismatch_comparison_spinning.png`` and/or
``teobpm_mismatch_comparison_nonspinning.png`` depending on the rows supplied.

The report command renders a LaTeX summary and compiles it when ``pdflatex`` is
available. When campaign summaries and validation figures exist, they are
included automatically:

.. code-block:: bash

   bayRing-teobpm-calibrate report \
     --campaign-dir runs/teobpm_aligned \
     --output-tex runs/teobpm_aligned/report/teobpm_calibration_report.tex

The generated global-fit file uses schema
``bayRing.teobpm.global-fit.v1``. pyRing consumes that file through its
``teob-calibration-file`` option on the matching TEOBPM integration branch.

Appendix A reproduction
-----------------------

The quasi-circular checks in Appendix A of arXiv:2604.15431 can be reproduced
from the public ``nc_ringdown`` metadata with the dedicated Appendix A commands.
They prepare the exact equal-mass/aligned-spin and nonspinning SXS IDs used in
the paper, run both ``HypTan`` and ``RatExp`` local fits over
``t_peak <= t <= t_peak + 80M``, collect the point estimates, and construct the
same one-dimensional polynomial global fits:

.. code-block:: bash

   bayRing-teobpm-calibrate appendix-a-prepare \
     --output-dir runs/appendix_a \
     --nc-ringdown-dir /path/to/nc_ringdown \
     --method Minimization \
     --n-random-seeds 16 \
     --t-start 0 \
     --t-end 80

   bayRing-teobpm-calibrate run-local-fits \
     --campaign-dir runs/appendix_a \
     --workers 4 \
     --timeout 1800

   bayRing-teobpm-calibrate appendix-a-collect \
     --campaign-dir runs/appendix_a

   bayRing-teobpm-calibrate appendix-a-global-fit \
     --campaign-dir runs/appendix_a

   bayRing-teobpm-calibrate appendix-a-plot \
     --campaign-dir runs/appendix_a

The nonspinning subset can also be compared directly against current reference
fits. ``HypTan`` coefficients are compared against the current TEOB/HypTan
nonspinning formulas embedded in the workflow; ``RatExp`` coefficients are
compared against the noneccentric Rao/Carullo ``nc_ringdown`` fit
``src/data/fits/nc_fits_sxs_non-spinning/order_fits_nu_1.csv``. The Appendix A
workflow writes ``TEOB-calibration = qc`` and rejects fit variables such as
``ecc``, ``bmrg``, ``Emrg`` and ``Jmrg``:

.. code-block:: bash

   bayRing-teobpm-calibrate appendix-a-compare-nonspinning \
     --campaign-dir runs/appendix_a \
     --nc-ringdown-dir /path/to/nc_ringdown \
     --rao-reference current-nc

This command writes coefficient comparison panels, point-level residual tables,
grid/point curves, and a compact residual-summary CSV under
``appendix_a_plots/``. To compare against the one-dimensional SXS ``nu`` fits
from the Appendix A reproduction notebooks instead, pass
``--rao-reference appendix-nu --rao-catalog sxs --rao-order 1``.

The Appendix A fit file uses schema ``bayRing.teobpm.appendix-a.v1`` and is
written to ``appendix_a_global_fits.json``. For the equal-mass/aligned-spin
sequence the independent coordinate is ``S_hat``; for the nonspinning sequence
it is ``nu``. The implemented polynomial degrees match the reproduction
notebooks: nonspinning targets are linear in ``nu``; equal-mass ``RatExp``
targets are cubic in ``S_hat``; equal-mass ``HypTan`` targets are cubic except
``c3p``, which is quartic.

The plotting command writes one coefficient/residual figure per paper panel,
plus mismatch-vs-parameter diagnostics. The ``HypTan`` panels also overlay the
published TEOB ``Past Fits`` curves used in the Appendix A comparison.
