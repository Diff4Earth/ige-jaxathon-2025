## People involved : 
- Hugo F.
- :(

## Project description 
The final goal is to analyze the performance of sparse (unstructured) solvers such as GMRES in JAX (in jax.scipy.sparse). 
The matrices in particular arise from partial differential equations with the Ultraspherical discretization (https://arxiv.org/abs/1202.1347).
In 1D, the matrices are banded with varying bandwidth, but in higher dimensions, it becomes less efficient to use banded LU compared to iterative methods. 
One particular difficulty is the preconditionning of the solver, with *NONE* being available in JAX. We aim to start with a simple Jacobi (diagonal) preconditioner and see if the convergence is satisfactory or not.
I know (from experiments) that the ILU(0) preconditionner is really efficient on these sparse matrices. If Jacobi is not leading to fast convergence, we need (at least) sparse triangular solves that are not provided by JAX. It might be interesting to look at binding cupy solvers into the JAX pipeline (w/ adjoint).

## Background information : 
- [A fast and well-conditioned spectral method](https://arxiv.org/abs/1202.1347)
- [Dedalus: A flexible framework for numerical simulations with spectral methods](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.023068)
- [JAX sparse doc](https://docs.jax.dev/en/latest/jax.experimental.sparse.html)
- [cupy ilu doc](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.spilu.html#cupyx.scipy.sparse.linalg.spilu)

## Planned work : 
1. Generate the ultraspherical spectral matrices for linear PDEs (e.g., a Laplacian, Biharmonic), with already implemented Python code (CPU).
2. Try solving with GMRES without preconditioner.
3. Search for the best preconditioner (while maintaining jax.grad availability).
4. Analyze the performance for the different strategies.

## Success metrics : 
- Mostly timings (time to convergence).

## Deliverable :
- A notebook

