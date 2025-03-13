import equinox as eqx
import jax.numpy as jnp
import diffrax
from diffrax import diffeqsolve, ODETerm, SaveAt, Euler
from types import SimpleNamespace
import sys
import functools

import jax
jax.config.update("jax_enable_x64", True)


# Model definition
class UNSTKmodel(eqx.Module):
    """
    Unsteady Ekman model (dissipation at layers interfaces, no Rayleigh damping)    
    
    Description of attributes:
        - U0 : initial zonal current
        - V0 : initial meridional current
        - Nl : number of layer (not used for now)
        - TAx : wind forcing X
        - TAy : wind forcing Y
        - K : control parameters
        - fc : Coriolis
        - solver : diffrax solver to time step
        - AD_mode : whether to use forward or backward auto diff

    Note: All class attributes are parameters, 
            the decision to differentiate wrt one or the other is made using diffrax filters.
    """
    # variables
    U0 : float
    V0 : float
    # parameters
    Nl : int
    TAx : jnp.array
    TAy : jnp.array
    K : jnp.array
    fc : jnp.array
    
    solver : diffrax.AbstractItoSolver

    AD_mode : str

    def __call__(self, t0, t1, dt, save_traj = []):
        """
        Inputs:
            - t0 : initial time
            - t1 : end time
            - dt : time step
            - save_traj : list of time instant to save solution at
            
        Outputs:
            - a solution object that contains the solution of the ODE
                see https://docs.kidger.site/diffrax/api/solution/
                
        Note: 
            save_traj: if empty, save only first and last instant.
                otherwise, save every specified instants.
        """
        term = self.ODE()
        if len(save_traj)==0:
            saveat = SaveAt(t0=True,t1=True)
        else:
            saveat = SaveAt(ts=save_traj)
        
        # Auto-diff mode
        if self.AD_mode=='F':
            adjoint = ForwardMode() #ForwardMode()
        else:
            adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=None) # None # diffrax.RecursiveCheckpointAdjoint(checkpoints=None)
        
        
        print(adjoint)
        # runtime parameters
        y0 = self.U0,self.V0
        # control
        K = jnp.exp( jnp.asarray(self.K) )
        # forcing
        fc = self.fc
        TAx = self.TAx # not used, we make idealized forcing below
        TAy = self.TAy
        
        TAx_t = lambda t: jnp.sin(2*jnp.pi*t/86400*8) # <- this could be defined outside of the model class
        
        args = fc, K, TAx_t, TAy 
        return diffeqsolve(term, 
                           self.solver, 
                           t0=t0, 
                           t1=t1, 
                           y0=y0, 
                           args=args, 
                           dt0=dt, 
                           saveat=saveat,
                           adjoint=adjoint) # here this is needed to be able to forward AD
    
    def ODE(self):
        """
        Continuous equation:
        
            dU/dt = +ifU + dTau/dz, with in the ocean Tau = Kz*dU/dz
            
        Integrated equation over depth H:

            dU/dt = +ifU + K0*TA - K1*U
            
            U : Complex current (zonal + i*meridional)
            TA : COmplex wind stress
            
            K0 = 1/H
            K1 = Kz/H
        
        Boundary conditions:
            surface : Tau = Tau0 = wind stress
            bottom : U = 0., 
        
        
        
        Note: diffrax doesnt support complex number so I split into 2 coupled ODEs
        """    
        def vector_field(t, C, args):
            U,V = C
            fc, K, TAx, TAy = args
            d_U = fc*V + K[0]*TAx(t) - K[1]*U
            d_V = -fc*U + K[0]*TAy - K[1]*V
            d_y = d_U,d_V
            return d_y
            
        return ODETerm(vector_field)



# this needs to be added to diffeqsolve to be able to forward AD
# see: https://github.com/patrick-kidger/optimistix/pull/51#issuecomment-2105948574
class ForwardMode(diffrax.AbstractAdjoint):
    def loop(
        self,
        *,
        solver,
        throw,
        passed_solver_state,
        passed_controller_state,
        **kwargs,
    ):
        del throw, passed_solver_state, passed_controller_state
        inner_while_loop = functools.partial(diffrax._adjoint._inner_loop, kind="lax")
        outer_while_loop = functools.partial(diffrax._adjoint._outer_loop, kind="lax")
        # Support forward-mode autodiff.
        # TODO: remove this hack once we can JVP through custom_vjps.
        if isinstance(solver, diffrax.AbstractRungeKutta) and solver.scan_kind is None:
            solver = eqx.tree_at(
                lambda s: s.scan_kind, solver, "lax", is_leaf=lambda x: x is None
            )
        final_state = self._loop(
            solver=solver,
            inner_while_loop=inner_while_loop,
            outer_while_loop=outer_while_loop,
            **kwargs,
        )
        return final_state
