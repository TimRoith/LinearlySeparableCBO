#%%
import sys
sys.path.append('./..')
import numpy as np
from cbx.objectives import Rastrigin
from cbx.utils.termination import max_eval_term
from cbx.scheduler import multiply
from cbx.dynamics import CBO, CBS

from problems import random_invertible
from CBO_utils import estimate_Q, RotFunCBO
from experimentUtils import check_unit_weight_estimates

#%%
d = 100
R = Rastrigin(b = 2.)
W = random_invertible(d, quasi_orth=True, perturb = 3)

def f(x):
    return R(x @ W.T) + 0.00001*R(x)

Qest_kwargs = {'n_hess':10, 'sval_cutoff':0.001, 'eps_hessian':1e-4, 
             'restarts' : 800, 'n_iter':200}
CBO_kwargs = {
    'dt':0.03, 'sigma':7.1, 'max_it':10000,
    'noise': 'anisotropic',
    'track_args': {'names':['best_energy']},
    'alpha':80., 'verbosity':1, 'f_dim': '3D',

}
max_f_eval = int(1e6)
Term = max_eval_term(max_eval = max_f_eval)
sched = multiply(factor=1.005, name='alpha', maximum=1e5)
n_f_hess_evals       = (2 * d**2 + 2*d) * Qest_kwargs['n_hess']
max_f_eval_w_hess = max(0, max_f_eval - n_f_hess_evals)

Term_w_hess = max_eval_term(max_eval = max_f_eval_w_hess)
print(max_f_eval_w_hess, n_f_hess_evals)
#%%
Q = estimate_Q(f, d = d, **Qest_kwargs)
print("Number of true weights | wrong estimates | missed weights")
check_unit_weight_estimates(W, Q, just_counts=True, prec = 1e-8)
#%% CBO on standard function
x0 = np.random.uniform(-5, 5, size=(1, 40,d))
dyn = CBO(f, x=x0.copy(), **CBO_kwargs, term_criteria = [Term],
          #post_process = P.post_process
          )
dyn.optimize(sched = sched)
bp = dyn.best_particle.squeeze()
print('Finsihed with energy: ', f.f(dyn.best_particle))
print('Distance to minimizer:', P.distance_to_minimum(bp, mixed=False))

#%% CBO on rotated function
dyn_RF = RotFunCBO(f, x=x0.copy(), Q = Q, 
                   **CBO_kwargs, term_criteria = [Term_w_hess],
                   )
dyn_RF.optimize(sched = sched)
# %%
