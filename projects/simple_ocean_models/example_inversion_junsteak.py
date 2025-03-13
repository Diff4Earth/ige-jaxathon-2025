import matplotlib.pyplot as plt
from template_api import model
import sys 
import xarray as xr
import numpy as np

import jax
import jax.tree_util as jtu
import optax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from diffrax import Euler
import diffrax
import equinox as eqx

#sys.path.append('./core/')
#from core.state_junsteak import USTKdata, USTKparameters, USTKstate
from junsteak import UNSTKmodel

# model setup
Nl=1            # number of layers
# initial conditions
U0 = 0.0        # (m/s), zonal velocity
V0 = 0.0        # (m/s), meridional velocity
# forcing
TAx = 0.0 # m2/s2, constant forcing for now
TAy = 0.02
fc = 1e-4

Ktarget     = jnp.asarray([-10.,-9.])
Kinitial    = jnp.asarray([-9.,-8.5]) # [-9.,-8.5]

# Time integration
solver=Euler()  # solver to use
dt=60           # time step
t0=0            # initial time
t1=84600        # final time

# optimizer things
AD_mode = 'forward'
itmax = 100
lr = 1e-1
opt = optax.adam(lr)
#opt = optax.lbfgs() # not working just like this, need more investigation ...

# adding a realistic forcing
# file_forcing = '/home/jacqhugo/Datlas_2025/Reconstruct_inertial_current/JAXathon/croco_1h_inst_surf_2006-02-01-2006-02-28_cons_-50.0_-35.0.nc'
# ds_f = xr.open_dataset(file_forcing)
# dt_forcing= 3600 # s
# time_forcing = jnp.arange(0,len(ds_f.time)*dt_forcing,dt_forcing, dtype=float)
# ds_f['time'] = time_forcing
# TAx = ds_f.oceTAUX.sel(time=slice(t0,t1))
# TAy = ds_f.oceTAUY.sel(time=slice(t0,t1))
# time_forcing = jnp.arange(t0,t1,dt_forcing)
# TAx_t = diffrax.LinearInterpolation(time_forcing, TAx)
# TAy_t = diffrax.LinearInterpolation(time_forcing, TAy)

# using realistic observation : OSSE
# WIP

###################################
# EXPERIMENT SETUP
###################################
# Initialise model = class instanciation and initial state and forcing
true_model = UNSTKmodel(U0=U0, V0=V0, Nl=1, TAx=TAx, TAy=TAy, K=Ktarget, fc=fc, solver=solver, AD_mode=AD_mode)  

# truth
ltime = jnp.arange(0,84600,60)
sol = true_model(t0=t0,t1=t1,dt=dt,save_traj=ltime)
truth = sol.ys

# Making observations from truth
time_obs = jnp.arange(t0,t1,60*20+600)
sol = true_model(t0=t0,t1=t1,dt=dt,save_traj=time_obs)
obs = sol.ys

# New model to use in minimisation
#K=jnp.asarray([-9.,-8.5]) # [-10,-9] # here we need 1) a jax array 2) that consist of floats
mymodel = UNSTKmodel(U0=U0, V0=V0, Nl=1, TAx=TAx, TAy=TAy, K=Kinitial, fc=fc, solver=solver, AD_mode=AD_mode)  
first_guess = mymodel(t0=t0,t1=t1,dt=dt,save_traj=ltime).ys

# filter model attribute
# this is where the magic happens:
#   we filter the class attributes that we dont want to be differentiated
#   (by default all non array/float are filtered out)
filter_spec = jtu.tree_map(lambda arr: False, mymodel) # keep nothing
filter_spec = eqx.tree_at( lambda tree: tree.K, filter_spec, replace=True) # keep only K

# loss function
def loss_fn(sol,obs): # <- this is problem specific, but also model specific
    return jnp.mean( (sol[0]-obs[0])**2 + (sol[1]-obs[1])**2 )


# this should be general enough to not need any modification in another model
def loss_grad( dynamic_model, static_model, tobs, t0, t1, dt, obs):
    mymodel = eqx.combine(dynamic_model, static_model)
    sol = mymodel(t0, t1, dt, save_traj=tobs).ys
    return loss_fn(sol,obs)

def loss_for_jacfwd(sol,obs):
    y = loss_fn(sol,obs)
    return y, y # <- trick to have a similar behavior than value_and_grad (but here returns grad, value)

def loss_jacfwd( dynamic_model, static_model, tobs, t0, t1, dt, obs):
    mymodel = eqx.combine(dynamic_model, static_model)
    sol = mymodel(t0, t1, dt, save_traj=tobs).ys
    return loss_for_jacfwd(sol,obs)


if AD_mode=='forward':
    dloss_fn = eqx.filter_jacfwd(loss_jacfwd, has_aux=True)
elif AD_mode=='forward':
    dloss_fn = eqx.filter_value_and_grad(loss_grad)

# this function is common to all model, nothing is specific to the model    
# |
# v
# one step of the minimizer
@eqx.filter_jit
def step_minimize( model, opt_state, filter_spec, t0, t1, dt, tobs, obs):
    dynamic_model, static_model = eqx.partition(model, filter_spec)
    #value, grad = loss_grad(dynamic_model, static_model, tobs, t0, t1, dt, obs)
    grad, value  = dloss_fn(dynamic_model, static_model, tobs, t0, t1, dt, obs)
    updates, opt_state = opt.update(grad, opt_state)
    model = eqx.apply_updates(model, updates)
    return value, model, opt_state

###################################
# MINIMISATION
###################################
# initialise optimizer
opt_state = opt.init(mymodel)
# optimize loop
for it in range(itmax):
    value, mymodel, opt_state = step_minimize(mymodel, opt_state, filter_spec, t0, t1, dt, time_obs, obs)
    print("it, J, K :",it, value, mymodel.K) # value, mymodel
print('Final K is:',mymodel.K)
newsolution = mymodel(t0, t1, dt, save_traj=ltime).ys

###################################
# PLOTTING
###################################
fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=100)
ax.plot(ltime, first_guess[0], c='b',label='first guess')
ax.plot(ltime,truth[0],c='k',label='truth')
ax.plot(ltime,newsolution[0],c='r',label='reconstruct')
ax.scatter(time_obs,obs[0],marker='x',c='r',label='obs')
ax.set_ylabel('U')
fig.savefig('minimisation_example.png')
plt.legend()

# fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=100)
# ax.plot(sol.ys[0],sol.ys[1])
# ax.set_xlabel('U')
# ax.set_ylabel('V')

# fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=100)
# time = np.arange(t0,t1,dt)
# ax.plot(np.sin(2*np.pi*time/86400*8))
# ax.set_ylabel('forcage TAx')

plt.show()



# a faire
# - clean up code
# - utiliser jacfwd at loss step
# - notebook d'un example de forward et un probleme inversion simple
# - def myoptimizer ? myoptimizer(opt, itmax, tolerance)
# - add a realistic forcing in the model definition (but keep the option to have idealized ones)
# - set AD_mode to 'F' if not defined in model instanciation