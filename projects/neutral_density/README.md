# Neutral density 


The aim of this project is to explore how gradient based optimization could be used for computing neutral surfaces for ocean data analysis. 

## People involved 
Romain Caneill and Julien Le Sommer

## Specific objectives 

Neutral surfaces are surface which are locally approximately tangent to the neutral tangent plane (Jackett and McDougall 1997). We have worked on the formulation of the problem of their computation as a variational problem where one tries to minimize a cost function by adjusting the depth of a surface at each grid point on a grid so that on average the surface is as tangent to the neutral tangent plane as possible. 


## Key outcome 

The results are illustrated in the notebook : [jaxathon_neutral_density_project_achievement.ipynb](./jaxathon_neutral_density_project_achievement.ipynb)

Starting from ocean temperature and salinity data from a global atlas, we have managed to define a local measure of the neutrality of a surface. 

At the discrete level a surface is defined by the depth of the cell centers on a 2D grid. 

Our loss is the RMSE of local difference between the slope pf the surface and target slopes derived from the temperature and salinity field. We have then tested how to minimize this cost function with optax. 

The computation appears to be efficient and amenable but we have encoutered issues with the regularity of the loss. 




## Perspectives  

Future work will involve regularizing the loss with a local term penalizing deviations from a given cast. A more precise approach to the dis retization of the loss could alos involve computing the diviations directly on the surface itself without interpolation. This would requires recoding in Jax some thermodynamic function. 


## Resources 

* Jackett and McDougall 1997 :  https://doi.org/10.1175/1520-0485(1997)027%3C0237:ANDVFT%3E2.0.CO;2
* Klocker et al. 2009 :  https://doi.org/10.5194/os-5-155-2009, 2009
* Stanley et al. 2021  := https://doi.org/10.1029/2020MS002436
* neutralocean : https://github.com/geoffstanley/neutralocean

