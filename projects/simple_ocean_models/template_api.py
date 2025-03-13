"""
This is an example of the API that model should follow
"""
import equinox as eqx
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt
import functools
import diffrax

# Model definition
class model(eqx.Module):
    """
    
    """
    ini_state : jnp.array
    parameter1 : float
    forcing1 : jnp.array
    boundary_condition : jnp.array
    grid : any
    solver : any
    AD_mode : str
        
        
    def __call__(self, t0, t1, dt, save_traj = []):
        """
        save_traj : a list. if empty, save only first and last instant.
                otherwise, save every specified instants.
        """
        term = self.ODE()
        if len(save_traj)==0:
            saveat = SaveAt(t0=True,t1=True)
        else:
            saveat = SaveAt(ts=save_traj)

        # Auto-diff mode
        if self.AD_mode=='F':
            adjoint = ForwardMode()
        else:
            adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=100)
        y0 = self.ini_state
        
        return diffeqsolve(term, self.solver, t0=t0, t1=t1, y0=y0, dt0=dt, saveat=saveat)
        
    
    
    def ODE(self):
        """
        simple ODE:
        
            df/dt = -f(x,t) -> lambda t,x : -f
            should be coded like :
            
            vector_field = lambda t,f : -f
        
        coupled ODE:

            see https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations

            def vector_field(t, y, args):
                prey, predator = y
                α, β, γ, δ = args
                d_prey = α * prey - β * prey * predator
                d_predator = -γ * predator + δ * prey * predator
                d_y = d_prey, d_predator
                return d_y
        """
        vector_field = lambda t,x : x # here its identity but you want to add your model

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

