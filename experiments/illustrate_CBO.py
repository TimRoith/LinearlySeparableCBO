#%%
from cbx.dynamics import CBO
from cbx.objectives import Rastrigin
from cbx.scheduler import multiply
from cbx.noise import noise
import numpy as np
from cbx.dynamics.pdyn import CBXDynamic

import matplotlib.pyplot as plt

from precond_additive_utils import *

#%% 2d visualization
d = 2
f = Rastrigin(b=2)
A = np.random.randn(d,d)
Q, R = np.linalg.qr(A)
rot = 'random'
if rot == 'none':
    Q = np.eye(d)
x = np.random.uniform(-1,1, (1,10,d))
f_rot = Rot_fun(Rastrigin(b=2), Q)
ckwargs = {
    'dt':0.05, 'sigma':5.1, 'lamda':1., 'max_it':1000,
    'track_args': {'names':['x', 'consensus', 'energy']},
    'alpha':1e2, 'verbosity':1
}
sched   = multiply(factor=1.05, name='alpha', maximum=1e5)
dyn_sep = CBO(f_rot,x=x, noise = 'anisotropic', **ckwargs)
dyn_sep.optimize()
# %%
from cbx.plotting import PlotDynamicHistory
fig, ax = plt.subplots(1,1)
pl = PlotDynamicHistory(
    dyn_sep, objective_args={'x_min':-3, 'x_max':3, 'cmap':'magma'},
    ax=ax)
pl.plot_objective()
c = np.array(dyn_sep.history['consensus'])  # shape (n_steps, 1, 1, d)
x = np.array(dyn_sep.history['x'])          # shape (n_steps, N, 1, d)
x = x.reshape(-1, 2)
ts = np.arange(0, c.shape[0], 1)
plt.scatter(c[ts,0,0,0], c[ts,0,0,1], color='red', label='consensus trajectory')
plt.scatter(x[ts,0], x[ts,1], color='white', alpha=0.5)

glob_min = Q@np.array([2,2])
plt.scatter(glob_min[0], glob_min[1], color='green', label='global minimum', marker='X', s=100)
plt.savefig(f'results/CBO_2d_{rot}.png')

# %% higher d
d = 16
A = np.random.randn(d,d)
Q, R = np.linalg.qr(A)
x = np.random.uniform(-3,3, (1,40,d))
f     = Rastrigin(b=2)
f_rot = Rot_fun(f, Q)



ckwargs = {
    'dt':0.05, 'sigma':5.1, 'lamda':1., 'max_it':10000,
    'track_args': {'names':['x', 'consensus', 'energy']},
    'alpha':1e2, 'verbosity':1
}

sched = multiply(factor=1.05, name='alpha', maximum=1e5)
dyn     = CBO(f,x=x,noise = 'anisotropic', **ckwargs)
dyn_rot = CBO(f_rot,x=x,noise = 'anisotropic', **ckwargs)
dyn.optimize()
dyn_rot.optimize()
# %% recover Q only from trajectories
from blockSeparableFunctions.hessianSamplingUtils import *
from blockSeparableFunctions.experimentUtils import *

w_est_cleaned, cluster_onbs = block_recovery_pipeline(f_rot, 16, {})
check_unit_weight_estimates(Q, w_est_cleaned, just_counts=True)

# %%
f_re_rot = Rot_fun(f_rot, w_est_cleaned.T)
dyn_re_rot = CBO(f_re_rot,x=x, noise = 'anisotropic', **ckwargs)
dyn_re_rot.optimize()

#%% CBO with rotated noise
rot_noise_sub = Rot_noise(w_est_cleaned)
dyn_rot_noise = CBO(f_rot,x=x,noise = rot_noise_sub, **ckwargs)
dyn_rot_noise.optimize()
# %% compare loss trajectories
import scienceplots
plt.style.use('science')
fig, ax = plt.subplots(1,1, figsize=(6,3))
for dy in [dyn, dyn_rot, dyn_re_rot, dyn_rot_noise]:
    e = np.array(dy.history['energy']).min(axis=-1)
    plt.loglog(e)
plt.legend(['CBO', 
            'CBO on $f(Q\cdot)$', 
            r'CBO on $f(Q_{\text{est}}^\top Q \cdot)$',
            'CBO with rotated noise'])
plt.savefig('results/CBO_loss_traj.pdf')

# %%
