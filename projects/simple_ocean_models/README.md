Here is a JAX-based repository that gathers simple ocean physical models compatible with efficient JAX-based inversion tools. 

The models are built using two main libraries:
  - Equinox (https://github.com/patrick-kidger/equinox) for PyTree manipulation routines
  - Diffrax (https://docs.kidger.site/diffrax/) for numerical differential equation solvers

The implementation of the models greatly facilitates the dynamic control of model parameters to fit some observations of the model state. The models are compatible with both forward and reverse modes. The numerical solvers can easily be changed. Finally, the models sharing the same API, they can be used the same way regardless the equations solved.

The repository is composed of two models:
  - Multi-Layer Quasi-Geostrophic model, built upon the Somax JAX-based library (https://github.com/jejjohnson/somax)
  - 1D Unsteady Ekman model

One example of twin experiments using the optax library is also provided. 
