## People involved : 
1. Minh Nguyen 
2. Anthony Frion 
3. Ali Bekar

## Project description 

We consider a simulator with time-varying system state $x_t \in \mathbb{R}^K$ and constant parameters $\theta \in \mathbb{R}^P$. The simulators is defined by numerical integration of tendencies $f$ that are functions of current state $x_t$, resulting a deterministic update $\mathcal{M}$ \\

$\textbf{Data assimilation.}$ We seek for such an initial state for the simulator/emulator that predictions would match to available observation.
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
4. Write 4DVar coding using Jax (Minh + Ali + Anthony

## Success metrics : 
*Please provide a criteria on the basis of which you will assess whether you have achieved your objectives for the hackathon*
