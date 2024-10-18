# Example configuration files description

This directory collects a set of example configuration files to showcase the basic usage options of the package. All options are explorable through `bayRing --help`.

The classes of models employed correspond to the ones described in e.g. [here](https://agenda.infn.it/event/39201/contributions/241342/attachments/125382/184845/TEONGrav.pdf) around slide 70 ("Agnostic" corresponds to "Damped-sinusoids").

The `quick` files shows how to obtain quick and dirty results (decreasing the values of sampler settings, setting a constant NR error, imposing tight prior bounds, decreasing the duration of the data time series). The simplest of such examples can be run with:

`bayRing --config-file config_SXS_0305_Kerr_220_quick.ini`

The other files instead use default priors (very broad), a large chunk of the time axis, the full NR error and conservative sampler settings useful when exploring a high-dimensional parameter space.

Usage of NR catalogs beyond the SXS is considered to be "developer" level, i.e. it requires the user to: i) download the corresponding NR simulations and placing them in the directory structure expected from the code (see the `NR_waveforms.py` module); ii) use the NR error option best suited to the simulation under consideration. Some of the simulations classes implemented might not yet be publicly available.
