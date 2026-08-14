import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union

from sklearn.gaussian_process.kernels import RBF



# This function is used to check the weight estimates.
# In particular, we need to know which estimates
# * are close to a true weight (up to sign)
# * are not close to any true weight (wrong estimates)
# * which true weights are not close to any estimate (missed weights)

def check_unit_weight_estimates(
    weights: np.ndarray,
    weight_estimates: np.ndarray,
    prec: float = 0.001,
    just_counts: bool = False,
) -> tuple[list[int], list[int], Union[list[int], tuple[int, int, int]]]:
    """
    Compare ground-truth weights against estimates, up to sign.
    Returns the indices of matched/unmatched rows, or their counts.

    Parameters
    ----------
    weights          : array-like, shape (m, d) – ground-truth weights
    weight_estimates : array-like, shape (n, d) – estimated weights
    prec             : float – tolerance for matching (default 0.001)
    just_counts      : bool – if True, return lengths instead of index lists (default False)

    Returns
    -------
    idx_true    : list[int] or int – indices (or count) of rows in weights found in weight_estimates
    idx_false   : list[int] or int – indices (or count) of rows in weight_estimates not matching any weight
    idx_missing : list[int] or int – indices (or count) of rows in weights not found in weight_estimates
    """
    W = np.array(weights, dtype=float)
    E = np.array(weight_estimates, dtype=float)
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)

    abs_dots = np.abs(W @ E.T)                               # (m, n) -- cosine similarity, both sides unit-norm
    matched_weights   = np.any(abs_dots >= 1.0 - prec, axis=1)  # (m,)
    matched_estimates = np.any(abs_dots >= 1.0 - prec, axis=0)  # (n,)

    idx_true    = np.where(matched_weights)[0].tolist()
    idx_false   = np.where(~matched_estimates)[0].tolist()
    idx_missing = np.where(~matched_weights)[0].tolist()

    if just_counts:
        return len(idx_true), len(idx_false), len(idx_missing)
    return idx_true, idx_false, idx_missing


# For exploration purposes, we sometimes need to check whether an estimate is close to a set of candidate vectors,
# e.g., the set of all true weights

def search_for_vecs(
    w_target: np.ndarray,
    w_est: np.ndarray,
    metric: str = "scalarProduct",
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each row vector in w_target, find the closest row vector in w_est
    using the specified similarity metric.

    Parameters
    ----------
    w_target : np.ndarray of shape (n, d)
    w_est    : np.ndarray of shape (m, d)
    metric   : str, either "scalarProduct" (default) or "norm"
        - "scalarProduct": similarity = |w_target[i] · w_est[j]|, sign-invariant.
          Best match has largest value (closest to 1).
        - "norm": similarity = ||w_target[i] - w_est[j]||_2.
          Best match has smallest value (closest to 0).

    Returns
    -------
    best_scores : np.ndarray of shape (n,)
        The best metric value for each row of w_target.
    indices     : np.ndarray of shape (n,)
        Index in w_est of the closest vector for each row of w_target.
    """
    if metric == "scalarProduct":
        scores = np.abs(w_target @ w_est.T)       # (n, m), higher is better
        indices = np.argmax(scores, axis=1)
    elif metric == "norm":
        diffs = (
            np.sum(w_target ** 2, axis=1, keepdims=True)
            + np.sum(w_est ** 2, axis=1)
            - 2 * (w_target @ w_est.T)
        )
        scores = np.sqrt(np.clip(diffs, 0, None)) # (n, m), lower is better
        indices = np.argmin(scores, axis=1)
    else:
        raise ValueError(f"metric must be 'scalarProduct' or 'norm', got '{metric}'.")

    best_scores = scores[np.arange(len(w_target)), indices]  # (n,)
    return best_scores, indices

# Plotting utilities
def plot_gram_matrix(W: np.ndarray, show_all_indices: bool = True, title: str = "Gram Matrix $WW^T$") -> None:
    """
    Plot the Gram matrix W @ W.T as a heatmap.

    Parameters
    ----------
    W : np.ndarray of shape (m, d)
    show_all_indices : bool, default True
        Whether to show all indices on the axes.
    title : str, default "Gram Matrix $WW^T$"
        Title for the plot.

    Returns
    -------
    None
    """
    G = W @ W.T  # (m, m)
    m = G.shape[0]

    fig, ax = plt.subplots()
    im = ax.imshow(G, cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    if show_all_indices:
        ax.set_xticks(range(m))
        ax.set_yticks(range(m))
        ax.set_xticklabels(range(m))
        ax.set_yticklabels(range(m))
    plt.tight_layout()
    plt.show()

# For evaluation purposes, we have to be able to generate random ONS

def sample_ons(d, n):
    """
    Sample a random orthonormal system of n vectors in R^d.

    Parameters
    ----------
    d : int
        Dimension of the ambient space.
    n : int
        Number of orthonormal vectors to sample (must satisfy n <= d).

    Returns
    -------
    Q : np.ndarray of shape (n, d)
        Matrix whose columns are orthonormal vectors.
    """
    assert n <= d, "Cannot have more orthonormal vectors than the dimension"
    A = np.random.randn(d, n)
    Q, _ = np.linalg.qr(A)
    return Q.T  # Return as (n, d) for consistency with other functions



# In order to generate linearly separable functions that are very varied and highly non-convex, we work with pre RKHS functions

class PreRKHSfunction:
    """A function in a Reproducing Kernel Hilbert Space (RKHS), represented as a
    finite weighted sum of kernel evaluations:

        f(x) = sum_i alpha_i * k(x_i, x)

    where x_i are base points, alpha_i are coefficients, and k is a kernel function.

    Parameters
    ----------
    kernel : sklearn-compatible kernel
        A kernel object implementing the sklearn interface (e.g. from
        ``sklearn.gaussian_process.kernels``).
    base_points : array-like of shape (n, d)
        The n base points in R^d.
    coeffs : array-like of shape (n,)
        Coefficients alpha_i for the weighted sum.
    rkhs_norm : float, optional
        If provided, the coefficients are rescaled so that the RKHS norm of
        the function equals this value.
    """

    def __init__(self, kernel, base_points, coeffs, rkhs_norm=None):
        self._base_points = np.atleast_2d(base_points)
        self._coeffs = np.array(coeffs).reshape(-1)
        self._kernel = kernel
        self._kernel_matrix = kernel(base_points, base_points)
        self._rkhs_norm = np.sqrt(self._coeffs @ self._kernel_matrix @ self._coeffs)
        if rkhs_norm is not None:
            self._coeffs *= rkhs_norm / self._rkhs_norm
            self._rkhs_norm = np.sqrt(self._coeffs @ self._kernel_matrix @ self._coeffs)

    def __call__(self, xs):
        """Evaluate the RKHS function on a batch of input points.

        Parameters
        ----------
        xs : np.ndarray of shape (..., d)
            Input points. The last dimension must match the input dimension d
            of the kernel. Arbitrary batch dimensions are supported.

        Returns
        -------
        np.ndarray of shape (...,)
            The function evaluated at each input point, with the same batch
            shape as ``xs`` but with the last dimension removed.
        """
        shape = xs.shape
        return np.reshape(
            self._coeffs @ self._kernel(self._base_points, xs.reshape(-1, shape[-1])),
            shape[:-1])

    def __sub__(self, other):
        """Subtract two PreRKHSfunctions defined on the same kernel.

        Returns a new PreRKHSfunction whose base points are the union of both
        operands' base points and whose coefficient vector is

            [alpha, -beta]

        where alpha are the coefficients of ``self`` and beta are the
        coefficients of ``other``.  The resulting function satisfies

            (self - other)(x) = self(x) - other(x)

        pointwise, because

            sum_i alpha_i k(x_i, x) - sum_j beta_j k(y_j, x)
            = sum_i alpha_i k(x_i, x) + sum_j (-beta_j) k(y_j, x).

        Parameters
        ----------
        other : PreRKHSfunction
            Must use the same kernel (checked via sklearn's equality operator).

        Returns
        -------
        PreRKHSfunction
            The difference self - other.
        """
        if not isinstance(other, PreRKHSfunction):
            return NotImplemented
        if self._kernel != other._kernel:
            raise ValueError(
                "__sub__: both operands must share the same kernel, "
                f"got {self._kernel} and {other._kernel}."
            )

        combined_base_points = np.vstack([self._base_points, other._base_points])
        combined_coeffs      = np.concatenate([self._coeffs, -other._coeffs])

        return PreRKHSfunction(self._kernel, combined_base_points, combined_coeffs)

    def __add__(self, other):
        """Add two PreRKHSfunctions defined on the same kernel.

        Returns a new PreRKHSfunction whose base points are the union of both
        operands' base points and whose coefficient vector is

            [alpha, beta]

        where alpha are the coefficients of ``self`` and beta are the
        coefficients of ``other``.  The resulting function satisfies

            (self + other)(x) = self(x) + other(x)

        pointwise, because

            sum_i alpha_i k(x_i, x) + sum_j beta_j k(y_j, x).

        Parameters
        ----------
        other : PreRKHSfunction
            Must use the same kernel (checked via sklearn's equality operator).

        Returns
        -------
        PreRKHSfunction
            The sum self + other.
        """
        if not isinstance(other, PreRKHSfunction):
            return NotImplemented
        if self._kernel != other._kernel:
            raise ValueError(
                "__add__: both operands must share the same kernel, "
                f"got {self._kernel} and {other._kernel}."
            )

        combined_base_points = np.vstack([self._base_points, other._base_points])
        combined_coeffs      = np.concatenate([self._coeffs, other._coeffs])

        return PreRKHSfunction(self._kernel, combined_base_points, combined_coeffs)

    @property
    def rkhs_norm(self):
        """float : The RKHS norm of this function, defined as sqrt(alpha^T K alpha),
        where K is the kernel matrix evaluated at the base points."""
        return self._rkhs_norm


def sample_pre_rkhs_function(kernel, x_min=-1, x_max=1, n_base_points=10, rkhs_norm=None, d=1):
    base_points = np.random.uniform(low=x_min, high=x_max, size=(n_base_points, d))
    coeffs = np.random.randn(n_base_points)
    return PreRKHSfunction(kernel, base_points, coeffs, rkhs_norm)


class LinearProjectionAdditivePreRKHSFunction:
    """
    Given functions f0, ..., fk with input dimensions dims[0], ..., dims[k]
    and a matrix W of shape (d1+...+dk, d), computes

        f(x) = f0(W[:dims[0], :] @ x) + ... + fk(W[dims[0]+...+dims[k-1]:, :] @ x)

    Supports arbitrary batch dimensions.

    Parameters
    ----------
    functions : list
        List of callable functions (e.g. PreRKHSfunction instances).
    W : np.ndarray of shape (sum(dims), d)
        Projection matrix.
    dims : list[int], optional
        Input dimension of each function. Defaults to [1, ..., 1] if not set.
    """
    def __init__(self, functions: list, W: np.ndarray, dims: Union[list[int], None] = None):
        if dims is None:
            dims = [1] * len(functions)

        if len(functions) != len(dims):
            raise ValueError(
                f"Number of functions ({len(functions)}) must match "
                f"number of dims ({len(dims)})."
            )
        if W.shape[0] != sum(dims):
            raise ValueError(
                f"W must have {sum(dims)} rows (sum of dims), got {W.shape[0]}."
            )
        self.functions = functions
        self.dims = dims
        self.W = W
        self.splits = np.cumsum([0] + dims)

    def __call__(self, xs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        xs : np.ndarray of shape (..., d)

        Returns
        -------
        np.ndarray of shape (...,)
        """
        return sum(
            f(xs @ self.W[self.splits[i]:self.splits[i+1], :].T)
            for i, f in enumerate(self.functions)
        )

# Convenience functions for the recovery pipeline and experiments
def sample_linearly_block_separable_func(block_dims, W):
    """
    Sample a linearly block separable function with given block dimensions.

    TODO Add options to choose a different function generator for the blocks

    Args:
        d: Total dimension of the input space.
        block_dims: List of integers specifying the dimensions of each block.
                    The sum of block_dims should be equal to d.
        W: Linear transformation matrix.

    Returns:
        A LinearlyBlockSeparableFunction instance.
    """
    assert sum(block_dims) == W.shape[0], "Sum of block dimensions must match the number of rows in W."

    # Sample functions
    kernel = RBF(length_scale=0.5)

    block_funcs = []

    for block_dim in block_dims:
        if block_dim == 1:
            block_func = sample_pre_rkhs_function(kernel, x_min=-10, x_max=10, n_base_points=50, rkhs_norm=1.0, d=1)
        else:
            block_func = sample_pre_rkhs_function(kernel, x_min=-10, x_max=10, n_base_points=50**2, rkhs_norm=1.0, d=block_dim)
        block_funcs.append(block_func)


    # Create the linearly block separable function
    linearly_block_separable_func = LinearProjectionAdditivePreRKHSFunction(block_funcs, W, dims=block_dims)

    return linearly_block_separable_func

# The next function is used to sample generic (generically non-orthogonal) scalar
# weight vectors for the linearly separable case.

def sample_direct_subspaces(d: int, block_dims: list[int]) -> list[np.ndarray]:
    """
    Sample a list of direct (pairwise trivially-intersecting) subspaces of R^d
    that are generically non-orthogonal.

    Strategy
    --------
    1. Sample a random invertible matrix A of shape (d, d).
    2. Partition the standard basis of R^d into blocks of sizes block_dims,
       giving subspaces spanned by columns of A in each block.
    3. The subspaces are direct because A is invertible: if a vector lies in
       two distinct blocks' images under A, it must be zero.
    4. The subspaces are generically non-orthogonal because A is a generic
       (non-orthogonal) matrix.

    Parameters
    ----------
    d          : int       – ambient dimension
    block_dims : list[int] – list of subspace dimensions; must sum to <= d

    Returns
    -------
    list of np.ndarray, item i has shape (block_dims[i], d)
        Each array's rows form a basis for the i-th subspace.

    Raises
    ------
    ValueError if sum(block_dims) > d
    """
    if sum(block_dims) > d:
        raise ValueError(
            f"sum(block_dims)={sum(block_dims)} exceeds ambient dimension d={d}."
        )

    # Random invertible matrix: columns are the "twisted" basis vectors
    A = np.random.randn(d, d)   # square, invertible with probability 1

    bases = []
    start = 0
    for dim in block_dims:
        # Columns of A in this block span a dim-dimensional subspace of R^d
        # Return as (dim, d) so rows are basis vectors
        basis = A[:, start : start + dim].T   # (dim, d)
        bases.append(basis)
        start += dim

    return bases

def sample_quasiorthogonal_matrix(d, perturb=0.0):
    Q, _ = np.linalg.qr(np.random.randn(d, d))
    if perturb:
        Q = Q + perturb * np.random.randn(d, d)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return Q

def sample_quasiorthogonal_subspaces(
    d: int,
    block_dims: list[int],
    perturb_max: Union[float, None] = None,
    min_norm: Union[float, None] = None,
    max_corr: Union[float, None] = None,
    max_tries: int = 10000,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Sample mutually quasi-orthogonal subspaces of R^d, returning both a
    non-orthonormal basis and an ONB for each subspace.

    Strategy
    --------
    1. Sample a global ONB of R^d and partition it into blocks of sizes
       block_dims — giving exactly orthogonal subspaces.
    2. If perturb_max is set, perturb each ONB block by adding a small random
       component outside its span and re-orthonormalising. The perturbation
       magnitude is tuned so that the maximum cosine principal angle between
       any two subspaces does not exceed perturb_max.
    3. For each (possibly perturbed) ONB block, apply a random invertible
       transformation within the subspace to produce a non-orthonormal basis,
       subject to optional min_norm and max_corr constraints.

    Principal angles
    ----------------
    The largest principal angle between subspaces U and V is
        sigma_max(Q_U @ Q_V.T)
    where Q_U, Q_V are their ONBs (rows). We enforce
        sigma_max <= perturb_max
    for all pairs (i, j) with i != j.

    Parameters
    ----------
    d           : int        – ambient dimension
    block_dims  : list[int]  – subspace dimensions; must sum to <= d
    perturb_max : float|None – maximum cosine of principal angle between any
                               pair of subspaces (0 = orthogonal, 1 = collinear)
    min_norm    : float|None – minimum row norm in returned block_mats
    max_corr    : float|None – maximum pairwise absolute correlation between
                               rows within each block_mats entry
    max_tries   : int        – max re-sampling attempts per block (default 10000)

    Returns
    -------
    block_mats : list of np.ndarray, item i has shape (block_dims[i], d)
        Rows are a non-orthonormal basis for subspace i.
    block_onbs : list of np.ndarray, item i has shape (block_dims[i], d)
        Rows form an ONB for subspace i.
    """
    if sum(block_dims) > d:
        raise ValueError(
            f"sum(block_dims)={sum(block_dims)} exceeds ambient dimension d={d}."
        )

    # ------------------------------------------------------------------ #
    # Step 1: start from a global ONB partitioned into orthogonal blocks  #
    # ------------------------------------------------------------------ #
    Q, _ = np.linalg.qr(np.random.randn(d, d))   # (d, d) orthogonal
    starts = np.cumsum([0] + block_dims)

    block_onbs = [Q[:, starts[j]:starts[j+1]].T   # (dim_j, d)
                  for j in range(len(block_dims))]

    # ------------------------------------------------------------------ #
    # Step 2: optional perturbation                                       #
    # ------------------------------------------------------------------ #
    def max_cosine_principal_angle(onbs: list[np.ndarray]) -> float:
        """Max singular value of Q_i @ Q_j.T over all i != j."""
        best = 0.0
        for i in range(len(onbs)):
            for j in range(i + 1, len(onbs)):
                sv = np.linalg.svd(onbs[i] @ onbs[j].T, compute_uv=False)
                best = max(best, sv[0])
        return best

    def perturb_onb(onb: np.ndarray, noise_scale: float) -> np.ndarray:
        """Add isotropic noise to onb rows and re-orthonormalise."""
        noisy = onb + noise_scale * np.random.randn(*onb.shape)
        Q_new, _ = np.linalg.qr(noisy.T, mode="reduced")
        return Q_new.T   # (dim, d)

    if perturb_max is not None:
        # Binary-search for noise_scale that achieves the desired perturb_max
        lo, hi = 0.0, 10.0
        for _ in range(50):
            mid = (lo + hi) / 2
            candidate = [perturb_onb(onb, mid) for onb in block_onbs]
            if max_cosine_principal_angle(candidate) < perturb_max:
                lo = mid
                block_onbs = candidate   # keep the last valid candidate
            else:
                hi = mid

        # Final re-perturb at lo to ensure constraint is satisfied
        for _ in range(max_tries):
            candidate = [perturb_onb(onb, lo) for onb in block_onbs]
            if max_cosine_principal_angle(candidate) <= perturb_max:
                block_onbs = candidate
                break
        else:
            raise RuntimeError(
                f"Could not achieve perturb_max={perturb_max} within {max_tries} tries."
            )

    # ------------------------------------------------------------------ #
    # Step 3: non-orthonormal bases via random invertible transform       #
    # ------------------------------------------------------------------ #
    def satisfies_constraints(W: np.ndarray) -> bool:
        if min_norm is not None:
            if np.any(np.linalg.norm(W, axis=1) < min_norm):
                return False
        if max_corr is not None:
            norms = np.linalg.norm(W, axis=1, keepdims=True)
            W_normed = W / norms
            corr_mat = np.abs(W_normed @ W_normed.T)
            np.fill_diagonal(corr_mat, 0.0)
            if np.any(corr_mat > max_corr):
                return False
        return True

    block_mats = []
    for onb in block_onbs:
        dim = onb.shape[0]

        if dim == 1:
            W = np.random.randn(1, 1) @ onb
            if min_norm is not None:
                W *= max(1.0, min_norm / np.linalg.norm(W))
            block_mats.append(W)
            continue

        for _ in range(max_tries):
            A = np.random.randn(dim, dim)
            W = A @ onb
            if satisfies_constraints(W):
                block_mats.append(W)
                break
        else:
            raise RuntimeError(
                f"Could not satisfy min_norm/max_corr constraints for block "
                f"of dimension {dim} after {max_tries} attempts."
            )

    return block_mats, block_onbs
