[![pypi](https://badge.fury.io/py/bayRing.svg)](https://pypi.org/project/bayRing/1.0.0/) |[![version](https://img.shields.io/pypi/pyversions/bayRing.svg)](https://pypi.org/project/bayRing/)| |[![license](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)| |[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8284026.svg)](https://doi.org/10.5281/zenodo.8284026)| 

bayRing
=======

# Description

Inference package targeting ringdown modeling of numerical relativity waveforms, using Bayesian sampling algorithm for parameters extraction.  

Relies on [pyRing](https://git.ligo.org/lscsoft/pyring) for waveform interfacing and on [cpnest](https://github.com/johnveitch/cpnest/tree/master) or [raynest](https://github.com/wdpozzo/raynest) for sampling.

# Installation

The package can be installed through: 

  ```
  pip install -r requirements.txt
  pip install .
  ```

An alternative to the latter instruction is `python setup.py install`.

# Usage

* The general execution syntax is: `bayRing --config-file config.ini`

* `bayRing --help` allows to explore all the available options and default values.

# Examples

Available in the `config_files` directory, see the corresponding README file.

To run the simplest example: 

  ```
  cd config_files
  bayRing --config-file config_SXS_0305_Kerr_220_quick.ini
  ```

# Contributing

If you have a request for an additional feature, spot any mistake or find any problem with using the code, please open an issue.

To develop the code, we follow a standard {create a branch/fork-apply your edits-submit a merge request} workflow.

For additional questions, feedback and suggestions feel free to reach by email to `gregorio.carullo@ligo.org`. Thanks for taking time to contribute to the code, your help is greatly appreciated!

## How to add a new waveform

1. Add the model to the list of available templates [here](https://github.com/GCArullo/bayRing/blob/8053d9232bbace0fb8ec114ce084fb4c65bcb5e5/bayRing/initialise.py#L273).
2. Add any model-specific structures or compatibility checks [here](https://github.com/GCArullo/bayRing/blob/8053d9232bbace0fb8ec114ce084fb4c65bcb5e5/bayRing/initialise.py#192).
3. Declare prior default bounds for model calibration parameters [here](https://github.com/GCArullo/bayRing/blob/8053d9232bbace0fb8ec114ce084fb4c65bcb5e5/bayRing/inference.py#L73).
4. Add the model-specific parser structure [here](https://github.com/GCArullo/bayRing/blob/8053d9232bbace0fb8ec114ce084fb4c65bcb5e5/bayRing/inference.py#L313).
5. Construct the waveform template [here](https://github.com/GCArullo/bayRing/blob/8053d9232bbace0fb8ec114ce084fb4c65bcb5e5/bayRing/template_waveforms.py#L87).
6. Add the call to the waveform template [here](https://github.com/GCArullo/bayRing/blob/8053d9232bbace0fb8ec114ce084fb4c65bcb5e5/bayRing/template_waveforms.py#L169).
7. Add an example configuration file, similar to e.g. [this](https://github.com/GCArullo/bayRing/blob/8053d9232bbace0fb8ec114ce084fb4c65bcb5e5/config_files/config_SXS_0305_Kerr_220_quick.ini).

## How to add a new NR catalog

Instructions coming soon...



.. |pypi| image:: https://badge.fury.io/py/bayRing.svg
   :target: https://pypi.org/project/bayRing/1.0.0/
.. |version| image:: https://img.shields.io/pypi/pyversions/bayRing.svg
   :target: https://pypi.org/project/bayRing/
.. |license| image:: https://img.shields.io/badge/License-MIT-red.svg
   :target: https://opensource.org/licenses/MIT
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8284026.svg
   :target: https://doi.org/10.5281/zenodo.8284026
