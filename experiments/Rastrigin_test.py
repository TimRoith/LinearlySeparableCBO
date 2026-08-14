#%%
import sys

sys.path.append('./..')
from cbx.dynamics import CBO, CBS
from cbx.noise import covariance_noise
from cbx.scheduler import multiply
from cbx.utils.termination import max_eval_term
import numpy as np
import matplotlib.pyplot as plt
from experimentUtils import check_unit_weight_estimates, sample_quasiorthogonal_subspaces
from CBO_utils import PreconditionedFunction, RotFunCBO, RotNoiseCBO, SplitDimCBO, Rot_fun, estimate_Q
from problems import RastriginProblem, SphereProblem, RastriginLikeProblem, MicroringProblem, DifferentPowersProblem
# %% Setup of test problem: Rastrigin with a random rotation in d dimensions
d   = 100
P   = RastriginProblem(d=d)
#P   = MicroringProblem(d=d)
#P   = RastriginLikeProblem(d=d)
#P   = SphereProblem(d=d)
#f   = P.generate_rotated()['objective']
#P   = DifferentPowersProblem(d=d, offset=1)
WW   = np.random.randn(d//2, d)
W    = np.zeros((d, d))
W[:d//2, :] = WW
f   = P.generate_transformed(W=W)['objective']
#f   = P.generate_transformed(quasi_orth=True, perturb = 5.)['objective']

N   = 40
x0  = P.sample(1, N)
# scheduler
sched = multiply(factor=1.005, name='alpha', maximum=1e5)

#%% kwargs
CBO_kwargs = {
    'dt':0.03, 'sigma':7.1, 'max_it':10000,
    'noise': 'anisotropic',
    'track_args': {'names':['best_energy']},
    'alpha':80., 'verbosity':1, 'f_dim': '3D',

}
max_f_eval = int(6e6)
Term = max_eval_term(max_eval = max_f_eval)

Qest_kwargs = {'n_hess':100, 'sval_cutoff':0.001, 
               'eps_hessian':1e-3, 'r':20.,
               'restarts' : 800, 'n_iter':200}
n_f_hess_evals       = (2 * d**2 + 2*d) * Qest_kwargs['n_hess']
max_f_eval_w_hess = max(0, max_f_eval - n_f_hess_evals)
print('Number of hessian function evaluations: ', n_f_hess_evals)
Term_w_hess = max_eval_term(max_eval = max_f_eval_w_hess)
print('Number of possible CBO steps after hess estimation: ', max_f_eval_w_hess)
#%%
Q = estimate_Q(f, d = d, **Qest_kwargs)
print("Number of true weights | wrong estimates | missed weights")
check_unit_weight_estimates(f.Q, Q, just_counts=True, prec = 1e-8)

#%% CBO on standard function
dyn = CBO(f.f, x=x0.copy(), **CBO_kwargs, term_criteria = [Term],
          #post_process = P.post_process
          )
dyn.optimize(sched = sched)
bp = dyn.best_particle.squeeze()
print('Finsihed with energy: ', f.f(dyn.best_particle))
print('Distance to minimizer:', P.distance_to_minimum(bp, mixed=False))

#%% CBO on rotated function
dyn_rot = CBO(f, x=x0.copy(), **CBO_kwargs, term_criteria = [Term])
dyn_rot.optimize(sched = sched)
print('Finsihed with energy: ', f(dyn_rot.best_particle))
print('Distance to minimizer:', P.distance_to_minimum(dyn_rot.best_particle.squeeze()))

# %% CBO with back rotation
dyn_RF = RotFunCBO(f, x=x0.copy(), Q = Q, 
                   **CBO_kwargs, term_criteria = [Term_w_hess],
                   )
dyn_RF.optimize(sched = sched)
print('Finished CBO with re-rotated function')

bp = dyn_RF.best_particle.squeeze() @ np.linalg.pinv(Q).T
print('Finsihed with energy: ', f(bp))
print('Distance to minimizer:', P.distance_to_minimum(bp))

# %% CBO with rotated noise
Q_inv = np.linalg.pinv(Q)
def pp(dyn): 
      dyn.x = np.clip(dyn.x @ Q.T, -1e8,1e8) @ Q_inv.T

dyn_RN = RotNoiseCBO(f, x=x0.copy() @ Q_inv.T, 
                     Q = Q, post_process = pp,
                     **CBO_kwargs| {'sigma':7.1}, term_criteria = [Term_w_hess])
dyn_RN.optimize(sched = sched)
print('Finished CBO with rotated noise')
print('Finsihed with energy: ', f(dyn_RN.best_particle))
print('Distance to minimizer:', P.distance_to_minimum(dyn_RN.best_particle.squeeze()))

# %% Split CBO along dimensions
Term_split = max_eval_term(max_eval = max_f_eval_w_hess // d)
CBO_split_kwargs = {
    'dt':0.01, 'sigma':1.1, 'max_it':1000,
    'noise': 'anisotropic',
    'track_args': {'names':['best_energy']},
    'alpha':1., 'verbosity':0, 'f_dim': '3D',
    'term_criteria' : [Term_split]
}
dyn_SD = SplitDimCBO(d = d, CBO_kwargs=CBO_split_kwargs, sched = sched)
x = dyn_SD(f, Q = Q, parallel=True, x0 = x0.copy())
print('Finsihed with energy: ', f(x))
print('Distance to minimizer:', P.distance_to_minimum(x))

# %%
