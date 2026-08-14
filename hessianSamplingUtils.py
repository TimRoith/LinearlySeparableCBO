import numpy as np
from typing import Callable, Union
from joblib import Parallel, delayed


###############################
# General sampling utilities
###############################
def sample_uniform_sphere(n: int, d: int, r: float = 1) -> np.ndarray:
    """
    Sample uniformly from the surface of a d-dimensional sphere of radius r.

    Parameters
    ----------
    n : int   – number of samples
    d : int   – dimension of the ambient space
    r : float – radius of the sphere (default 1)

    Returns
    -------
    np.ndarray of shape (n, d)
    """
    X = np.random.randn(n, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return r * X


def sample_uniform_ball(n: int, d: int, r: float = 1, center: Union[np.ndarray, float] = 0) -> np.ndarray:
    """
    Sample uniformly from the interior of a d-dimensional ball of radius r.

    Uses the Muller / Barthe–Naor method:
      1. Draw a direction uniformly from the sphere.
      2. Draw a radial coordinate with the correct d-dimensional marginal,
         i.e. R = r * U^(1/d) where U ~ Uniform(0, 1).

    Parameters
    ----------
    n      : int                    – number of samples
    d      : int                    – dimension of the ambient space
    r      : float                  – radius of the ball (default 1)
    center : array-like or float    – centre of the ball, shape (d,) or scalar (default 0)

    Returns
    -------
    np.ndarray of shape (n, d)
    """
    directions = sample_uniform_sphere(n, d, r=1)
    radii = r * np.random.uniform(0, 1, size=(n, 1)) ** (1.0 / d)
    return radii * directions + np.asarray(center)


def sample_uniform_shell(
    n: int, d: int, r_min: float, r_max: float
) -> np.ndarray:
    """
    Sample uniformly from the d-dimensional spherical shell
    { x : r_min <= ||x|| <= r_max }.

    The volume element in d dimensions gives the radial CDF:

        P(R <= t) = (t^d - r_min^d) / (r_max^d - r_min^d)

    Inverting:  R = (r_min^d + U * (r_max^d - r_min^d))^(1/d),  U ~ Uniform(0,1)

    Parameters
    ----------
    n     : int   – number of samples
    d     : int   – dimension of the ambient space
    r_min : float – inner radius of the shell
    r_max : float – outer radius of the shell

    Returns
    -------
    np.ndarray of shape (n, d)
    """
    if r_min < 0:
        raise ValueError("r_min must be non-negative.")
    if r_max <= r_min:
        raise ValueError("r_max must be strictly greater than r_min.")

    directions = sample_uniform_sphere(n, d, r=1)
    u = np.random.uniform(0, 1, size=(n, 1))
    r_min_d = r_min ** d
    r_max_d = r_max ** d
    radii = (r_min_d + u * (r_max_d - r_min_d)) ** (1.0 / d)
    return radii * directions

################################
# Utilities for the Hessians
################################

def hessian_fd(f, x, eps=1e-5):
    """
    Compute Hessian of f at x using central finite differences.
    x: 1D array of shape (d,)
    Returns H: array of shape (d, d)
    """
    d = x.shape[0]
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            ei = np.zeros(d); ei[i] = 1.0
            ej = np.zeros(d); ej[j] = 1.0
            H[i, j] = (
                f(x + eps*ei + eps*ej)
                - f(x + eps*ei - eps*ej)
                - f(x - eps*ei + eps*ej)
                + f(x - eps*ei - eps*ej)
            ) / (4 * eps**2)
            H[j, i] = H[i, j]  # symmetry
    return H

def hessian_fd_batch(f_1d, X, eps=1e-5, n_jobs=-1, pairs_per_job=10):
    """
    Compute Hessian of f at each row of X using central finite differences.
    X: 2D array of shape (n, d)
    Returns H: array of shape (n, d, d)

    The d*(d+1)/2 (i, j) index pairs are independent finite-difference probes
    of the (black-box) f_1d, so they're distributed across threads via
    joblib's threading backend: numpy releases the GIL during the underlying
    BLAS/ufunc calls that dominate each f_1d evaluation, so this gets real
    parallelism without a process pool's pickling/IPC overhead. Batching
    every (i, j) pair into one giant vectorized call instead (rather than
    splitting work across cores) was tried and measured slower, not faster,
    for typical black-box objectives -- see git history.

    pairs_per_job : int -- pairs handled per thread dispatch (empirically,
        ~8-10 balances dispatch overhead against parallelism; default 10).
    n_jobs : int -- passed to joblib.Parallel; -1 uses all available cores.
    """
    n, d = X.shape
    H = np.zeros((n, d, d))

    ii, jj = np.triu_indices(d)
    n_pairs = ii.shape[0]

    def _chunk(I, J):
        p = len(I)
        out = np.empty((n, p))
        for k in range(p):
            i, j = I[k], J[k]
            ei = np.zeros(d); ei[i] = 1.0
            ej = np.zeros(d); ej[j] = 1.0
            out[:, k] = (
                f_1d(X + eps*ei + eps*ej)
                - f_1d(X + eps*ei - eps*ej)
                - f_1d(X - eps*ei + eps*ej)
                + f_1d(X - eps*ei - eps*ej)
            ) / (4 * eps**2)
        return out

    chunks = [(ii[s:s + pairs_per_job], jj[s:s + pairs_per_job]) for s in range(0, n_pairs, pairs_per_job)]
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_chunk)(I, J) for I, J in chunks
    )

    for (I, J), out in zip(chunks, results):
        H[:, I, J] = out
        H[:, J, I] = out

    return H


#######################################
# Projected gradient ascent related
#######################################
def run_batch_pga(
    v_inits: np.ndarray,
    W_vec: np.ndarray,
    step_size: float = 2,
    n_iter: int = 1000,
    return_val_traj: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Batched projected gradient ascent with optional objective trajectory tracking.

    Parameters
    ----------
    v_inits          : array-like, shape (n, d) – initial vectors
    W_vec            : array-like, shape (r, d²) – low-rank factor of the projection
                        matrix P = W_vec.T @ W_vec (r = number of retained singular
                        vectors). Passing the factor instead of the full (d², d²)
                        matrix avoids ever forming/multiplying by it: that product
                        costs O(d⁴) per iteration, while the factored form costs
                        O(r·d²), which is far cheaper whenever r << d².
    step_size        : float – gradient step size (default 2)
    n_iter           : int   – number of iterations (default 1000)
    return_val_traj  : bool  – if True, also return the objective value trajectory (default False)

    Returns
    -------
    V : np.ndarray, shape (n, d) – converged vectors, one per starting point
    val_traj : np.ndarray, shape (n, n_iter+1) – objective trajectories (only if return_val_traj=True)
        val_traj[i, t] is the objective value for the i-th run at iteration t.
    """
    V = np.array(v_inits, dtype=float)
    W_vec = np.array(W_vec, dtype=float)
    n, d = V.shape

    if return_val_traj:
        val_traj = np.empty((n, n_iter + 1))

    def outer_flat(V):
        return np.einsum("ni,nj->nij", V, V, optimize=True).reshape(n, d * d)  # (n, d²)

    def objective(V):
        proj = outer_flat(V) @ W_vec.T                          # (n, r)
        return np.einsum("nr,nr->n", proj, proj, optimize=True)  # (n,)

    if return_val_traj:
        val_traj[:, 0] = objective(V)

    for t in range(n_iter):
        outer  = outer_flat(V)
        Pouter = (outer @ W_vec.T) @ W_vec   # (n, d²), i.e. P @ outer via the low-rank factor
        grad   = np.einsum("nij,nj->ni", Pouter.reshape(n, d, d), V, optimize=True)
        V      = V + 2 * step_size * grad
        V     /= np.linalg.norm(V, axis=1, keepdims=True)

        if return_val_traj:
            val_traj[:, t + 1] = objective(V)

    if return_val_traj:
        return V, val_traj
    return V

def run_pga_multistart(
    n_starts: int,
    W_vec: np.ndarray,
    step_size: float = 2,
    n_iter: int = 1000,
    return_val_traj: bool = False,
) -> np.ndarray:
    """
    Multi-start projected gradient ascent from random initialisations on the unit sphere.

    Parameters
    ----------
    n_starts  : int   – number of random starting points
    W_vec     : np.ndarray, shape (r, d²) – low-rank factor of the projection matrix,
                see `run_batch_pga`.
    step_size : float – gradient step size (default 2)
    n_iter    : int   – number of iterations (default 1000)
    return_val_traj  : bool  – if True, also return the objective value trajectory (default False)

    Returns
    -------
    If return_val_traj is False:
        np.ndarray, shape (n_starts, d) – final iterates
    If return_val_traj is True:
        tuple[np.ndarray, np.ndarray] – (final iterates, objective trajectories)
        final iterates: np.ndarray, shape (n_starts, d)
        objective trajectories: np.ndarray, shape (n_starts, n_iter+1)
    """
    d = int(round(W_vec.shape[1] ** 0.5))
    v_inits = sample_uniform_sphere(n_starts, d, r=1)
    return run_batch_pga(v_inits, W_vec, step_size=step_size, n_iter=n_iter, return_val_traj=return_val_traj)

def filter_weight_estimates_inds(
    weight_estimates: np.ndarray,
    cutoff: float = 0.01,
) -> list[int]:
    """
    Deduplicate rows of weight_estimates up to sign, returning indices.

    A candidate row is kept only if its absolute inner product with every
    already-accepted row is strictly less than (1 - cutoff).

    Parameters
    ----------
    weight_estimates : array-like, shape (n, d)
    cutoff           : float – tolerance; rows with |<v, w>| >= 1 - cutoff
                               are considered duplicates (default 0.01)

    Returns
    -------
    list[int] – indices of the deduplicated rows in weight_estimates
    """
    candidates = np.array(weight_estimates, dtype=float)

    accepted_idx = [0]
    accepted = [candidates[0]]

    for i, v in enumerate(candidates[1:], start=1):
        accepted_arr = np.array(accepted)
        abs_dots = np.abs(accepted_arr @ v)
        if np.all(abs_dots < 1.0 - cutoff):
            accepted_idx.append(i)
            accepted.append(v)

    return accepted_idx

def filter_weight_estimates(
    weight_estimates: np.ndarray,
    cutoff: float = 0.01,
) -> np.ndarray:
    """
    Deduplicate rows of weight_estimates up to sign.

    Parameters
    ----------
    weight_estimates : array-like, shape (n, d)
    cutoff           : float – tolerance; rows with |<v, w>| >= 1 - cutoff
                               are considered duplicates (default 0.01)

    Returns
    -------
    np.ndarray, shape (m, d) – deduplicated rows, m <= n
    """
    candidates = np.array(weight_estimates, dtype=float)
    indices = filter_weight_estimates_inds(candidates, cutoff)
    return candidates[indices]
