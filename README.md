[![pypi](https://badge.fury.io/py/bayRing.svg)](https://pypi.org/project/bayRing/1.0.0/)
[![python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://pypi.org/project/bayRing/)
[![license](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8284026.svg)](https://doi.org/10.5281/zenodo.8284026)

bayRing
=======

# Description

Inference package targeting ringdown modeling of numerical relativity waveforms, using a Bayesian method based on stochastic sampling for parameters extraction.  

Requires Python 3.10 or newer.

Relies on [pyRing](https://git.ligo.org/lscsoft/pyring) for waveform interfacing. For source-based development, install the [`generalise_NR_informed_models`](https://git.ligo.org/lscsoft/pyring/-/tree/generalise_NR_informed_models) branch of pyRing. It also relies on [qnm](https://github.com/duetosymmetry/qnm/) for QNM frequencies computations and on [cpnest](https://github.com/johnveitch/cpnest/tree/master)/[raynest](https://github.com/wdpozzo/raynest) for sampling.

# Documentation

The documentation can be found at [https://gcarullo.github.io/bayRing/](https://gcarullo.github.io/bayRing/).
