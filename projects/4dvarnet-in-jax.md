# 4DVarNet Implementation in JAX


## People involved:
Emilio González Zamora, Anne Durif, Shashank Kumar Roy


## Project description
The objective of our project is implementing 4DVarNet, a neural network model for data assimilation, in the JAX framework. The standard implementation of 4DVarNet is in PyTorch. For a specific case called the ‘base.yaml’, an experiment configuration, we want to implement the corresponding codes leveraging efficient packages in the JAX ecosystem.
The outcome of our project will allow the possibility of hybrid modeling in 4DVarnet using JAX.


## Background information:
The repository in PyTorch: https://github.com/CIA-Oceanix/4dvarnet-starter 
The first paper by Fablet et.al., where the 4DVar-Net was first introduced: https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2021ms002572


## Planned work:
We plan to replace the original .py files listed in the deliverable, and located inside the /src folder in the 4dvarnet-starter repository. We will focus on the ‘base’ experiment and create the JAX equivalent codes for the experiment.


## Deliverable:
A GitHub repo containing the JAX-based 4DVarNet from which we can use hydra and run training/inference using the command line.

Transformed files: xp/base_jax.yaml, src/_init_jax_.py, src/data_jax.py, src/models_jax.py, src/train_jax.py, src/utils_jax.py, src/versioning_cb_jax.py


## Success metrics:
Measure the 4DVarNet training efficiency (time and memory allocation), accuracy metrics (RMSE, lambda (x,t)) in PyTorch and JAX, on the example dataset (mentioned in the GitHub).
