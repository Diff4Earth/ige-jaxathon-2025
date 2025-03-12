## People involved

* Julien
* Romain

## Project description 

Our goal is, starting from a surface in the ocean, make it more neutral.


## Background information

* Jackett and McDougall 1997, https://doi.org/10.1175/1520-0485(1997)027%3C0237:ANDVFT%3E2.0.CO;2
* Klocker, A., McDougall, T. J., and Jackett, D. R.: A new method for forming approximately neutral surfaces, Ocean Sci., 5, 155–172, https://doi.org/10.5194/os-5-155-2009, 2009
* Stanley et al. 2021 https://doi.org/10.1029/2020MS002436 
  Algorithmic Improvements to Finding Approximately Neutral Surfaces
* https://github.com/geoffstanley/neutralocean
* https://neutralocean.readthedocs.io/en/latest/API.html#neutralocean.surface.omega.omega_surf

## Planned work


1. Notebook that can load 3D T/S, xgcm.Grid
   * --> *Done*
1. Create a function that take 3D fields and computes the neutral direction on fixed grid
   * T/W/U/V/F point? What is its native position?
   * Schema / discretization
   * xgcm
   * Done only once
   * use `neutralocean.ntp.ntp_epsilon_errors`
   * write the equation to get the slope (or equivalently the $\Delta z$) from $\epsilon$
   * --> *Done*
2. notebook getting a surface and assess its neutrality pointwise
   * angle interpolated in vertical: using `jax.numpy.interp`
   * plot 2D map of difference of angle
   * --> *Done*
3. Create a function that takes 1 surface + the 3D neutral slopes (and T-S field? or N2?) and compute its neutrality as a scalar.
   * this is a JAX function
   * differentiable => to allow optimization
   * what are the math? What formula?
   * What are the input, do we need to cheat to achieve differentiability?
   * local weighting by N2? How to deal with wetting? ...
   * --> *WIP*
4. Optimization
   * define cost function (scalar neutrality + maybe gamma0 + other regularizations)
   * iterative process to minimize the cost function
   * use notebook 2 to validate

## What we deliver

1 notebook + 1 python module containing the functions
(compute neutral direction for each grid point, assess neutrality pointwise, scalar neutrality, optimization process)

## Success metrics


* The new surface after optimization should be more neutral than the 1st guess
* It should be fast (enough): comparable speed with neutralocean for computing 1 surface
