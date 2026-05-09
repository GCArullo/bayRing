[![pypi](https://badge.fury.io/py/bayRing.svg)](https://pypi.org/project/bayRing/1.0.0/)
[![version](https://img.shields.io/pypi/pyversions/bayRing.svg)](https://pypi.org/project/bayRing/)
[![license](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8284026.svg)](https://doi.org/10.5281/zenodo.8284026)

bayRing
=======

# Description

Inference package targeting ringdown modeling of numerical relativity waveforms, using a Bayesian method based on stochastic sampling for parameters extraction.  

Relies on [pyRing](https://git.ligo.org/lscsoft/pyring) for waveform interfacing, on [qnm](https://github.com/duetosymmetry/qnm/) for QNM frequencies computations and on [cpnest](https://github.com/johnveitch/cpnest/tree/master)/[raynest](https://github.com/wdpozzo/raynest) for sampling.

# Documentation

The documentation can be found at [https://gcarullo.github.io/bayRing/](https://gcarullo.github.io/bayRing/).

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
