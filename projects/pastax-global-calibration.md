## People involved
- Vadim Bertrand
- Amine Ouhechou

## Project description
[`pastax`](https://pastax.readthedocs.io/en/latest/) allows to calibrate stochastic trajectory simulators using gradient-based optimizers through JAX automatic differentiation capability.
The aim of this project is to build a Proof-Of-Concept for calibrating a pastax simulator (Smagorinsky diffusion model) from a large (i.e. global over several years) amount of observed drifters trajectories and colocated surface geostrophic currents.

## Background information
`pastax` code can be found here: [https://github.com/vadmbertr/pastax](https://github.com/vadmbertr/pastax).
We use the **CRPS** as the metric for quantifying the "distance" between an ensemble of simulated trajectories and an observed trajectory ([Thorey et al. 2018](https://doi.org/10.1002/qj.2940)).
We use the **Liu Index** as the metric for quantifying the "distance" between pair of trajectories ([Liu and Weisberg 2011](https://doi.org/10.1029/2010JC006837)).

## Planned work
The work is divided in two main stages:
1. Building the pipeline allowing to **efficiently** iterates through batches of samples (pair of trajectory and colocated surface currents fields).
	- select the spatio-temporal window around a trajectory,
	- creates JAX Pytrees for the trajectory and the fields (using classes implemented in `pastax`).
	- *note*: useful links:
		- [https://earthmover.io/blog/cloud-native-dataloader/](https://earthmover.io/blog/cloud-native-dataloader/),
		- [https://xbatcher.readthedocs.io/en/latest/demo.html](https://xbatcher.readthedocs.io/en/latest/demo.html),
		- [https://github.com/patrick-kidger/equinox/issues/137](https://github.com/patrick-kidger/equinox/issues/137),
		- [https://docs.jax.dev/en/latest/distributed_data_loading.html](https://docs.jax.dev/en/latest/distributed_data_loading.html).
2. For each batch updates the model according to the gradient of the objective function.
	- simulate an ensemble of possible solutions for each trajectory,
	- evaluate the loss and the gradient,
	- update the model.
	- *note*: might be necessary/useful to use some sort of gradient accumulation ([Orbax](https://orbax.readthedocs.io/en/latest/index.html), [Optax](https://optax.readthedocs.io/en/latest/_collections/examples/gradient_accumulation.html)?).

## Material
Two notebooks:
- One for showcasing the calibration pipeline
- One providing computational timing plots:
	- loading the data,
 	- calibrating the simulator.

## Success metrics
Calibrating the simulator globaly for 20 years of data should take less than 48 hours (by extrapolating the numbers observed on a smaller time period).
