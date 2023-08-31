[![pypi](https://badge.fury.io/py/bayRing.svg)](https://pypi.org/project/bayRing/1.0.0/) [![version](https://img.shields.io/pypi/pyversions/bayRing.svg)](https://pypi.org/project/bayRing/) [![license](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8284026.svg)](https://doi.org/10.5281/zenodo.8284026)

bayRing
=======

# Description

Inference package targeting ringdown modeling of numerical relativity waveforms, using Bayesian sampling algorithm for parameters extraction.  

Relies on [pyRing](https://git.ligo.org/lscsoft/pyring) for waveform interfacing, on [qnm](https://github.com/duetosymmetry/qnm/) for QNM frequencies computations and on [cpnest](https://github.com/johnveitch/cpnest/tree/master)/[raynest](https://github.com/wdpozzo/raynest) for sampling.

# Installation

Tagged versions can be installed through: `pip install bayRing`.

The source code can be installed through: 

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

# Citing

When referencing ``bayRing`` in your publications, please cite the software Zenodo release:
   
  ```
      @software{carullo_gregorio_2023_8284026,
      author       = {Carullo, Gregorio and De Amicis, Marina and Redondo-Yuste, Jaime},
      title        = {bayRing},
      month        = aug,
      year         = 2023,
      publisher    = {Zenodo},
      version      = {1.0.0},
      doi          = {10.5281/zenodo.8284026},
      url          = {https://doi.org/10.5281/zenodo.8284026},
      howpublished = "\href{https://github.com/GCArullo/bayRing}{github.com/GCArullo/bayRing}",
      }
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

1. Add the catalog to the list of available ones [here](https://github.com/GCArullo/bayRing/blob/bfff5de8e156497c6fba548cf83d951166cb1612/bayRing/initialise.py#L231).
2. Add any catalog-specific structures or compatibility checks [here](https://github.com/GCArullo/bayRing/blob/bfff5de8e156497c6fba548cf83d951166cb1612/bayRing/initialise.py#L187).
3. Add a parser structure for simulations parameters [here](https://github.com/GCArullo/bayRing/blob/bfff5de8e156497c6fba548cf83d951166cb1612/bayRing/NR_waveforms.py#L34).
4. Add a metadata entry [here](https://github.com/GCArullo/bayRing/blob/bfff5de8e156497c6fba548cf83d951166cb1612/bayRing/NR_waveforms.py#L331).
5. Add a reading structure for the metadata and the waveform [here](https://github.com/GCArullo/bayRing/blob/bfff5de8e156497c6fba548cf83d951166cb1612/bayRing/NR_waveforms.py#L1387).
6. Add a call to waveform and metadata inside the NR simulation class [here](https://github.com/GCArullo/bayRing/blob/bfff5de8e156497c6fba548cf83d951166cb1612/bayRing/NR_waveforms.py#L643).
7. Add a method to estimate the NR error (resolution, extrapolation etc.) [here](https://github.com/GCArullo/bayRing/blob/bfff5de8e156497c6fba548cf83d951166cb1612/bayRing/NR_waveforms.py#L736).
8. Add an example configuration file [here](https://github.com/GCArullo/bayRing/blob/bfff5de8e156497c6fba548cf83d951166cb1612/config_files/config_RWZ_001_DS_2_quick.ini).

An example of a new catalog implementation can be found in [this](https://github.com/GCArullo/bayRing/pull/8) merge request.
