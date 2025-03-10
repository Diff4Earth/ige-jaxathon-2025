## People involved : 
Minh Nguyen
Anthony Frion
Ali Bekar

## Project description 
We consider a simulator with time-varying system state $x_t \in \mathbb{R}^K$ and constant parameters $\theta \in \mathbb{R}^P$. The simulators is defined by numerical integration of tendencies $f$ that are functions of current state $x_t$, resulting a deterministic update $\mathcal{M}$ \\

$\textbf{Data assimilation.}$ We seek for such an initial state for the simulator/emulator that predictions would match to available observation.

![Uploading image.png…]()



## Background information : 
*Provide any information (GitHub repository, reference to scientific paper) useful to describe the starting point of your project*  

## Planned work : 
*Please describe here what would be the main activities of the group during the hackathon*.
1. Create UNET2D on Jax Flax (Ali)
2. Verify forecast state and Jacobian (Ali)
3. Load trained model parameters (Anthony)
3. Write 4DVar coding using Jax (Minh + Ali + Anthony)

## Success metrics : 
*Please provide a criteria on the basis of which you will assess whether you have achieved your objectives for the hackathon*
