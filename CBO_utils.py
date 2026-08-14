from cbx.dynamics import CBO
from joblib import Parallel, delayed
import numpy as np

from hessianSamplingUtils import filter_weight_estimates, hessian_fd_batch, run_pga_multistart, sample_uniform_ball

# Rotate the objective functions
class Rot_fun:
    def __init__(self, f, Q):
        self.f = f
        self.Q = Q
        self.Q_inv = np.linalg.pinv(Q)

    def __call__(self, x):
        x_rot = x @ self.Q_inv.T
        return self.f(x_rot)

# rotates the noise in the dynamic
class Rot_noise:
    def __init__(self, Q):
        self.Q = Q
        self.Q_inv = np.linalg.pinv(Q)

    def __call__(self, dyn):
        z = dyn.sampler(size=dyn.drift.shape)
        return (dyn.dt**0.5  * (dyn.drift @ self.Q.T) * z) @ self.Q_inv.T

# Preconditioned function
class PreconditionedFunction:
    def __init__(self, f, W):
        self._W = W 
        self._f = f

    @property
    def input_dim(self):
        return self._W.shape[1]
    
    def __call__(self, x):
        return self._f(x @ self._W.T)
    
def complete_basis(vecs, d):
    vecs = np.atleast_2d(np.asarray(vecs, dtype=float))
    k = vecs.shape[0]
    if k >= d:
        return vecs[:d]

    Q_given, _ = np.linalg.qr(vecs.T)
    filler = np.random.randn(d, d - k)
    filler -= Q_given @ (Q_given.T @ filler)
    filler_onb, _ = np.linalg.qr(filler)
    return np.vstack([vecs, filler_onb.T])

# estimate matrix Q
def estimate_Q(f, d, n_hess = 1000, sval_cutoff = 0.001, eps_hessian = 1e-4,
               restarts = 200, n_iter = 100, r = 2):
    xs_hessians = sample_uniform_ball(n_hess, d, r)
    hess = hessian_fd_batch(f, xs_hessians, eps=eps_hessian)

    # Subspace approx
    U, S, Vh = np.linalg.svd(hess.reshape([n_hess, d*d]), full_matrices=False)
    inds = np.where(S > sval_cutoff)[0]
    W_vec = Vh[inds, :]

    # Weight recovery from subspace via multistart PGA (pass the low-rank
    # factor directly, see run_batch_pga — avoids ever forming the d²×d² P)
    w_pga = run_pga_multistart(restarts, W_vec, n_iter = n_iter)

    w_est = filter_weight_estimates(w_pga)
    return complete_basis(w_est, d)

class RotFunCBO(CBO):
    def __init__(self, f, Q = None, Qest_kwargs = {}, **kwargs):
        self.Q = Q if Q is not None else estimate_Q(f, self.d[0], **Qest_kwargs)
        super().__init__(Rot_fun(f, self.Q), **kwargs)

class RotNoiseCBO(CBO):
    def __init__(self, f, Q = None, Qest_kwargs = {}, **kwargs):
        self.Q = Q if Q is not None else estimate_Q(f, self.d[0], **Qest_kwargs)
        Q_inv = np.linalg.pinv(self.Q)

        def default_post_process(dyn, max_thresh=1e8):
            dyn.x = np.clip(dyn.x @ self.Q.T, -max_thresh, max_thresh) @ Q_inv.T

        kwargs.setdefault('post_process', default_post_process)
        super().__init__(f, **kwargs | {'noise': Rot_noise(self.Q)})

def CBO_factory(d, CBO_kwargs = {}, sched = None):
    def CBO_1d(f, **extra_kwargs):
        dyn = CBO(f, d=d, **{**CBO_kwargs, **extra_kwargs})
        result = dyn.optimize(sched = sched)
        return result
    return CBO_1d

class SplitDimCBO:
    def __init__(self, d = None, CBO_kwargs = {}, Qest_kwargs = {}, sched = None):
        if d is None:
            if 'd' in CBO_kwargs:
                d = CBO_kwargs['d']
            elif 'x' in CBO_kwargs:
                d = CBO_kwargs['x'].shape[-1]
            else:
                raise ValueError(
                    "d must be given explicitly, or inferable from CBO_kwargs['d'] or CBO_kwargs['x']."
                )
        self.d = d
        self.Qest_kwargs = Qest_kwargs

        sub_kwargs = {k: v for k, v in CBO_kwargs.items() if k not in ('d', 'x')}
        self.factory = CBO_factory(d=1, CBO_kwargs=sub_kwargs, sched=sched)

    
    def __call__(self, f, x0 = None, Q = None, parallel = True):
        Q = Q if Q is not None else estimate_Q(f, d = self.d, **self.Qest_kwargs)
        Q_inv = np.linalg.pinv(Q)
        f_pre = [PreconditionedFunction(f, Q_inv[:, i:i+1]) for i in range(Q.shape[0])]

        if x0 is not None:
            x0_pre = [np.asarray(x0) @ Q[i:i+1, :].T for i in range(Q.shape[0])]
        else:
            x0_pre = [None] * len(f_pre)

        def run(f_i, x0_i):
            extra_kwargs = {} if x0_i is None else {'x': x0_i}
            return self.factory(f_i, **extra_kwargs)

        if parallel:
            x_pre_opt_cbo = np.array(Parallel(n_jobs=-1)(
                delayed(run)(f_i, x0_i) for f_i, x0_i in zip(f_pre, x0_pre)
            ))
        else:
            x_pre_opt_cbo = np.array(
                [run(f_i, x0_i) for f_i, x0_i in zip(f_pre, x0_pre)]
                ).squeeze()
        return x_pre_opt_cbo.squeeze() @ Q_inv.T