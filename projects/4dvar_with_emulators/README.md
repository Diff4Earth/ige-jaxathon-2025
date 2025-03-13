# ShallowWaterJAX

This project is inscribed in the JAXhathon 2025 event: https://github.com/Diff4Earth/ige-jaxathon-2025


## People involved : 
1. Minh Nguyen (thanh.nguyen@hereon.de)
2. Anthony Frion (anthony.frion@hereon.de)
3. Ali Bekar (ali.bekar@hereon.de)

## Project description 

We consider a simulator with time-varying system state $x_t \in \mathbb{R}^K$ and constant parameters $\theta \in \mathbb{R}^P$. The simulators is defined by numerical integration of tendencies $f$ that are functions of current state $x_t$, resulting a deterministic update $\mathcal{M}$ 

**Data assimilation**: We seek for such an initial state for the simulator/emulator that predictions would match to available observation.
## Background information : 
*Provide any information (GitHub repository, reference to scientific paper) useful to describe the starting point of your project*  

## Planned work : 
We will provide a notebook including

1. Create UNET2D on Jax Flax (Ali)
   - Implementing UNET2D using flax.nnx having the same structure with our current pytorch code.
2. Load trained model parameters (Anthony)
   - Loading the checkpoint of trained model in pytorch.
   - Loading all trained model's parameters to UNET2D (Jax)
3. Verify forecast state and Jacobian (Ali)
   - Implementing the ForwardEuler time integration scheme
   - Constraining the closed boundary conditions for ssh, u, v (Shallow Water problem with closed boundary)
4. Coding 4DVar using Jax (Minh + Ali + Anthony)
   - Implementing the mask
   - Implementing the function to retrieve observation and applying the mask to obs
   - Implementing the loss function
   - Implementing the 4DVar algorithm

## Success metrics : 
*Please provide a criteria on the basis of which you will assess whether you have achieved your objectives for the hackathon*
![An example of observational data](https://github.com/Diff4Earth/ige-jaxathon-2025/blob/main/projects/4dvar_with_emulators/observations.png)
<figcaption>An example of observational data</figcaption>


![X0 (Initial guess)](https://github.com/Diff4Earth/ige-jaxathon-2025/blob/main/projects/4dvar_with_emulators/initial_guess.png)
<figcaption>X0 (Initial guess)</figcaption>

![Found X0 using data assimilation 4DVar](https://github.com/Diff4Earth/ige-jaxathon-2025/blob/main/projects/4dvar_with_emulators/found_x0.png)
<figcaption>Found X0 using data assimilation 4DVar</figcaption>

![](https://github.com/Diff4Earth/ige-jaxathon-2025/blob/main/projects/4dvar_with_emulators/forecasting.png)
![](https://github.com/Diff4Earth/ige-jaxathon-2025/blob/main/projects/4dvar_with_emulators/forecasting2.png)
![](https://github.com/Diff4Earth/ige-jaxathon-2025/blob/main/projects/4dvar_with_emulators/forecasting3.png)
<figcaption>Forecasting starting from the found X0</figcaption>

