"""
Forward Quasi-Geostrophic Multi-Layer (QGML) model based on the 
`somax<https://github.com/jejjohnson/somax>`_ package.

The QGML model is essentially a re-write of the
`mgq_doublegyre_og.py<https://github.com/jejjohnson/somax/blob/main/scripts/mqg_doublegyre_og.py>`_ script 
by J. Emmanuel Johnson adapted to follow the simple model API template. Description of the model can be found 
in Thiry et al. (2024) [1]_.

Note that currently, only double wind-driven gyre case is supported.
Parts of the script are not flexible, e.g. capacitance matrices are 
set to None by default, which restricts the domain type of the grid to 'rectangular'. 
In order to fully take advantage of the flexibility of the somax QGML model,
the module needs to be expanded.

References
----------
.. [1] Louis Thiry, Long Li, Guillaume Roullet, and Etienne Mémin. 
    MQGeometry-1.0: a multi-layer quasi-geostrophic solver
    on non-rectangular geometries. (2024). url : 
    `doi.org/10.5194/gmd-17-1749-2024 <https://doi.org/10.5194/gmd-17-1749-2024>`_.
"""

import equinox as eqx
import jax.numpy as jnp
import diffrax
from diffrax import diffeqsolve, ODETerm, SaveAt, Euler
import einops

from jaxtyping import Array, Float
import jax.numpy as jnp
import jax

from somax.domain import Domain
from somax.masks import MaskGrid
from somax._src.models.qg.domain import LayerDomain
from somax._src.models.qg.params import QGParams
from somax.interp import center_avg_2D
from somax._src.models.qg.elliptical import DSTSolution, calculate_helmholtz_dst, compute_homogeneous_solution
from somax._src.models.qg.forcing import calculate_bottom_drag, calculate_wind_forcing

from somax._src.models.qg.operators import calculate_psi_from_pv, equation_of_motion


def get_dst_sol(psi0, grid, params):
    """
    Get the DSTSolution based on initial streamfunction and grid.

    Parameters
    ----------
    psi0 : Array
    Initial streamfunction JAX Array.

    grid : QGMLgrid
    PyTree with grid specs.

    params : QGParams
    PyTree with somax QG model parameters.

    Returms
    -------
    dst_sol : DSTSolution
    PyTree defining the input to the spectral solver with discrete sine 
    transform (DST) needed to solve the somax QG equation of motion.
    """

    H_mat = calculate_helmholtz_dst(grid.xy_domain, grid.layer_domain, params)
    lambda_sq = params.f0**2 * einops.rearrange(grid.layer_domain.lambda_sq, "Nz -> Nz 1 1")
    
    homsol = compute_homogeneous_solution(
        psi0, 
        lambda_sq=lambda_sq,
        H_mat=H_mat
    )
    
    # calculate homogeneous solution
    homsol_i = jax.vmap(center_avg_2D)(homsol) * grid.masks.center.values
    homsol_mean = einops.reduce(homsol_i, "Nz Nx Ny -> Nz 1 1", reduction="mean")
    
    cap_matrices = None
    
    dst_sol = DSTSolution(
        homsol=homsol, 
        homsol_mean=homsol_mean, 
        H_mat=H_mat,
        capacitance_matrix=cap_matrices
    )

    return dst_sol


class QGMLgrid(eqx.Module):
    """
    Initialize the QGML model grid settings

    Parameters
    ----------
    resolution : int
    Number of grid cells in x and y direction of the rectangular domain.

    Lx : float
    Length of the rectangular domain in x direction.

    Ly : float
    Length of the rectangular domain in y direction.

    Attributes
    ----------
    xy_domain : Domain

    layer_domain : LayerDomain

    masks : MaskGrid
    """
    xy_domain : Domain
    layer_domain : LayerDomain
    masks : MaskGrid

    def __init__(self,
        resolution=512,
        Lx=5_120.0e3,
        Ly=5_120.0e3):

        # initialize the xy_domain
        Nx, Ny = resolution, resolution
        dx, dy = Lx / Nx, Ly / Ny
        self.xy_domain = Domain(
            xmin=(0.0,0.0), 
            xmax=(Lx,Ly),
            Lx=(Lx,Ly),
            Nx=(Nx, Ny), 
            dx=(dx, dy)
        )

        # initialize masks    
        mask = jnp.ones((Nx,Ny))
        mask = mask.at[0].set(0.0)
        mask = mask.at[-1].set(0.0)
        mask = mask.at[:,0].set(0.0)
        mask = mask.at[:,-1].set(0.0)
        self.masks = MaskGrid.init_mask(mask, location="node")

        heights = [400.0, 1_100.0, 2_600.0]
        reduced_gravities = [0.025, 0.0125]
    
        # initialize layer domain
        with jax.default_device(jax.devices("cpu")[0]):
            self.layer_domain = LayerDomain(heights, reduced_gravities, correction=False)
    
    
def forcing_fn(
    psi: Float[Array, "Nz Nx Ny"],
    dq: Float[Array, "Nz Nx-1 Ny-1"],
    domain: Domain,
    layer_domain: LayerDomain,
    params: QGParams,
    masks: MaskGrid,
) -> Float[Array, "Nz Nx Ny"]:
    """
    Set upper and lower layer of the QGML model based on forcing.

    Parameters
    ----------
    psi : Array
    Streamfunction JAX array.

    dq : Array
    dPV JAX array.

    domain : Domain
    Model domain retrevied from QGMLgrid.

    layer_domain : LayerDomain
    Model layer domain retrevied from QGMLgrid.

    params : QGParams
    Somax QG model parameters.

    masks : MaskGrid
    Model masks retrevied from QGMLgrid.

    Returns
    ----------
    dq : Array
    dPv JAX array with imposed wind and bottom friction forcing.
    """

    wind_forcing = calculate_wind_forcing(
        domain=domain,
        params=params,
        H_0=layer_domain.heights[0],
        tau0=0.08/1_000.0,
    )
    
    # add wind forcing
    dq = dq.at[0].add(wind_forcing)
    
    # calculate bottom drag
    bottom_drag = calculate_bottom_drag(
        psi=psi, domain=domain,
        H_z=layer_domain.heights[-1],
        delta_ek=params.delta_ek,
        f0=params.f0,
        masks_psi=masks.node
    )
    
    # add bottom drag
    dq = dq.at[-1].add(bottom_drag)
    
    return dq


class QGMLmodel(eqx.Module):
    """
    The QGML model adapted to the simple model API.

    Attributes
    ----------
    psi0 : Array
    Initial streamfunction JAX Array.

    q0 : Array
    Initial PV JAX Array. 

    grid : GQMLgrid
    Grid defining the model domain.

    dst_sol : DSTSolution

    f0 : float
    Coriolis [s^-1]

    beta: float 
    coriolis gradient [m^-1 s^-1]

    tau0: float 
    wind stress magnitude m/s^2
    
    y0: float 
    
    a_2: float
    laplacian diffusion coef [m^2/s]

    a_4: float
    biharmonic diffusion coef [m^4/s]

    bcco: float 
    boundary condition coef. (non-dim.)

    delta_ek: float 
    eckman height [m]

    num_pts : int

    method : str

    max_steps : int
    Maximum amount of steps for diffrax. Need to be imposed for long
    runtime.

    solver : diffrax.AbstractItoSolver
    ODE solver

    """
    # variables
    psi0 : Array
    q0 : Array

    # grid
    grid : QGMLgrid
    dst_sol : DSTSolution

    # somax qg model parameters
    f0 : float
    beta : float
    tau0 : float
    y0 : float
    a_2 : float
    a_4 : float
    bcco : float
    delta_ek : float
    num_pts : int
    method : str

    # solver
    max_steps : int
    solver : diffrax.AbstractItoSolver

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

        # runtime parameters
        y0 = self.psi0, self.q0

        # grid
        grid = self.grid
        dst_sol = self.dst_sol

        params = QGParams(
            f0 = self.f0,
            beta = self.beta,
            tau0 = self.tau0,
            y0 = self.y0,
            a_2 = self.a_2,
            a_4 = self.a_4,
            bcco = self.bcco,
            delta_ek = self.delta_ek,
            num_pts = self.num_pts,
            method = self.method
        )

        args = grid, params, dst_sol
        
        return diffeqsolve(term, self.solver, t0=t0, t1=t1, y0=y0, args=args, dt0=dt, saveat=saveat, max_steps=self.max_steps)


    def ODE(self):

        def vector_field(t, y, args):

            psi, q = y
            grid, params, dst_sol = args

            dq = equation_of_motion(
                q=q, psi=psi, params=params,
                domain=grid.xy_domain, layer_domain=grid.layer_domain,
                forcing_fn=forcing_fn,
                masks=grid.masks
            )

            dpsi = calculate_psi_from_pv(
                q=dq,
                params=params,
                domain=grid.xy_domain,
                layer_domain=grid.layer_domain,
                mask_node=grid.masks.node,
                dst_sol=dst_sol,
                remove_beta=False
            )
    
            return dpsi, dq

        return ODETerm(vector_field)


