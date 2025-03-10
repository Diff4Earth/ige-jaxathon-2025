## People involved : 

* Julien Le Sommer
* Romain Caneill

## Project description 

*Provide a 5-lines description of what you intend to achieve by the end of the hackathon*

Our goal is, starting from a surface in the ocean, make it more neutral.


## Background information : 
*Provide any information (GitHub repository, reference to scientific paper) useful to describe the starting point of your project*  

* 

## Planned work : 
*Please describe here what would be the main activities of the group during the hackathon*. 

0. Notebook that can load 3D T/S, xgcm.Grid
1. Create a function that take 3D fields and computes the neutral direction on fixed grid
  * T/W/U/V/F point? What is its native position?
  * Schema / discretization
  * xgcm
  * Done only once
2. notebook getting a surface and assess its neutrality pointwise
  * plot 2D map of difference of angle
  * angle interpolated in vertical
3. Create a function that takes 1 surface + the 3D neutral slopes (and T-S field? or N2?) and compute its neutrality as a scalar.
  * this is a JAX function
  * differentiable => to allow optimization
  * what are the math? What formula?
  * What are the input, do we need to cheat to achieve differentiability?
  * local weighting by N2? How to deal with wetting? ...
4. Optimization
  * define cost function (scalar neutrality + maybe gamma0 + other regularizations)
  * iterative process to minimize the cost function
  * use notebook 2 to validate

## Success metrics : 
*Please provide a criteria on the basis of which you will assess whether you have achieved your objectives for the hackathon*


