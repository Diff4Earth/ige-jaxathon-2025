# 4DVarNet Implementation in JAX


## People involved
Emilio González Zamora, Anne Durif, Shashank Kumar Roy


## Project description
Our project aims to implement 4DVarNet, a neural network model for data assimilation, in the JAX framework. The standard implementation of 4DVarNet is in PyTorch. We want to implement the corresponding codes leveraging efficient packages in the JAX ecosystem.

We implement 4DVarNet for a small toy problem where we generate a dataset using Lorenz-63 simulations. The assimilation window has 5 time points, and each time point has $\[x,y,z\]^T$. The corresponding observations are $\[x+ \mathcal{N}(0,\sigma),y+ \mathcal{N}(0,\sigma)\]$. 
 


## Background information
Data assimilation

Problem:  Over an observation window indexed by $i$, for observations sequence $Y^i=\{y^i_0,y^i_1,...y^i_n\}$ on ${\Omega^i_1,\Omega^i_2,..\Omega^i_n} \in \Omega$, find the state sequence $X^i=\{x^i_0,x^i_1,...x^i_n\}$. 

The optimal solution $X_{i}^{*}$ minimizes a cost function  

 $$  X^{i*}= \argmin_{X^i} \mathbf{U}_\Phi \left(X^i,Y^i,\Omega^i\right)
$$, where U denotes the 4DVar cost, given by, 
   $$ \mathbf{U}_\Phi \left(X^i,Y^i\right)= \|H(X^i)-Y^i\|+ \textcolor{red}{\|X^i- \Phi(X^i)\|}$$

4DVarNet solves the above weak-constraint problem to obtain a trajectory over the assimilation window by learning both $\Phi$ and 

We also have a **Neural Solver** ,
     $\Gamma\left(U_\Phi, X^{init}, Y\right)$: A network that learns to efficiently solve the 4DVar cost.   
   which is implemented in the following manner:
   $${$X^{k+1}=X^k-P(g_k), \ g_k=\mathcal{S}_\theta(\nabla_X \mathbf{U}_\Phi \left(X^i,Y^i\right)), $}$$ 
where 
    $\mathcal{S}_\theta: \text{model architecture} $: ConvLSTM model.


The repository with the full PyTorch implementation: https://github.com/CIA-Oceanix/4dvarnet-starter 
The first paper by Fablet et.al. (2021), where the 4DVarNet framework was first introduced: https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2021ms002572
A repository containing a starting point for  the implementation of 4DVarNet  in JAX:
https://github.com/jejjohnson/jejeqx/tree/48489d643b3722638d8e4e20d3e32393b1a580a5


## Planned work

With the 4dvarnet-starter as our base repository, we implement the different parts of the 4DVar cost and the network models located inside the /src folder. We will focus on the ‘base’ experiment and create the JAX equivalent codes. For the training logic we keep the PyTorch dataloaders, which are convenient for any general model in JAX or PyTorch.

What we did:
We went through the organization of the Python scripts of the 4dvarnet-starter repository. We used various LLMs (ChatGPT, Mistral) to translate the scripts from PyTorch to JAX. However, a direct translation is too complex with these tools (details and variations from one library to the other).
Instead, we identified the most important modules of the 4DVarNet architecture. We started to implement the main ones, from scratch and with the help of LLMs, on a simple use case: the Lorenz63 system. We built a Jupyter Notebook with their implementation, details on some encountered issues, and the work in progress.

## Deliverable
A notebook containing the beginning of the 4dvarnet implementation for Lorenz-63 model. We have a ConvLSTM model for learning the fast solver, and the prior cost model which is the Bilinear autoencoder. Our goal is that our work can be reused and achieved for interested researchers.


## Success metrics
A Jupyter Notebook with the implementation (and explanation) in jax and flax of the main modules of the 4DVarNet algorithm.
