import numpy as np
from numpy.typing import ArrayLike
import cbx
from cbx.utils.objective_handling import cbx_objective
from typing import Union
from experimentUtils import sample_quasiorthogonal_matrix

def random_rotation(d):
    """Generate a random rotation matrix in R^d."""
    A = np.random.randn(d,d)
    Q, R = np.linalg.qr(A)
    return Q

def random_invertible(d, quasi_orth = False, perturb=0.0):
    """Generate random invertible matrix"""
    if quasi_orth:
        return sample_quasiorthogonal_matrix(d, perturb=perturb)

    rank = 0
    while rank < d:
        A = np.random.normal(0,1,size=(d,d))
        rank = np.linalg.matrix_rank(A)
    return A

class Rot_fun:
    def __init__(self, f, Q):
        self.f = f
        self.Q = Q

    def __call__(self, x):
        x_rot = x @ self.Q.T
        return self.f(x_rot)

class Problem:
    """Base class. Subclasses must set d, domain, f_min and implement generate()."""
    d:      int
    domain: tuple              # (low, high) scalars
    f_min:  float              # known global minimum value
    name:   str    = 'default' # Name of the method
    minimum: Union[ArrayLike, None]

    def generate(self):
        """Return {'objective': f} with a numpy-compatible f. Used by the cbx pipeline."""
        raise NotImplementedError

    def sample(self, n_nodes, n_particles):
        """Uniform sample over domain, returns (n_nodes, n_particles, d)."""
        low, high = self.domain
        return low + (high - low) * np.random.rand(n_nodes, n_particles, self.d)

    def distance_to_minimum(self, x, mixed=True, ord=np.inf):
        """Success-criterion distance from x to a global minimizer.

        mixed=True  (default): x is in ambient/transformed coordinates,
                     compared against self.minimum (set by generate_transformed
                     / generate_rotated; falls back to the untransformed
                     self.minimum if neither was ever called).
        mixed=False: x is in the native/untransformed frame, compared against
                     self.minimum_orig (the pre-transform minimizer).

        Default implementation assumes a unique minimizer; override for
        problems (e.g. periodic ones) where many x achieve the true optimum.
        """
        target = self.minimum if mixed else getattr(self, 'minimum_orig', self.minimum)
        return float(np.linalg.norm(np.asarray(x).reshape(-1) - np.asarray(target), ord=ord))

    def post_process(self, dyn, sfac = 10.):
        """Clip particles back into domain after each CBO step (numpy/cbx pipeline)."""
        low, high = self.domain
        diff = high - low
        low, high = low - sfac * diff, high + sfac * diff
        np.nan_to_num(dyn.x, copy=False, nan=high)
        np.clip(dyn.x, low, high, out=dyn.x)
    
    def generate_transformed(self, W = None, **kwargs):
        "Return a linearly transformed version of the problem"
        if W is None:
            self.W = random_invertible(self.d, **kwargs)
        else:
            self.W = W
        self.Winv = np.linalg.pinv(self.W)
        f = self.generate()['objective']
        if self.minimum is not None:
            self.minimum_orig = self.minimum.copy()
            if self.Winv.shape[1] == self.d:
                self.minimum = self.Winv @ self.minimum
        return {'objective': Rot_fun(f, self.W)}



class RastriginProblem(Problem):
    def __init__(self, d=10, b=None):
        self.d, self.b = d, b
        self.domain = (-5.12, 5.12)
        self.f_min  = 0.0

        if b is None:
            self.b = np.random.uniform(-2.0, 2.0, size=d)
        self.minimum = self.b.copy()

    def generate(self):
        return {'objective': cbx.objectives.Rastrigin(b=self.b)}


class SphereProblem(Problem):
    """Separable, unimodal, well-conditioned: f(x) = ||x||^2."""
    def __init__(self, d=10):
        self.d = d
        self.domain = (-5.12, 5.12)
        self.f_min  = 0.0
        self.minimum = np.zeros(d)

    def generate(self):
        return {'objective': cbx.objectives.Quadratic(alpha=1.0)}


class DifferentPowers(cbx_objective):
    """BBOB f14: f(x) = sum_i |x_i|^(2 + 4*i/(d-1))."""
    def __init__(self, d, offset = 0):
        super().__init__()
        self.exponents = 2 + 4 * np.arange(offset, offset + d) / max(d - 1, 1)

    def apply(self, x):
        return np.sum(np.abs(x) ** self.exponents, axis=-1)


class DifferentPowersProblem(Problem):
    """BBOB f14, separable, unimodal, ill-conditioned via heterogeneous
    exponents rather than heterogeneous quadratic coefficients -- unlike
    Ellipsoid this has position-varying curvature (except at x_0, whose
    exponent is exactly 2), which estimate_Q's Hessian-based recovery
    actually needs."""
    def __init__(self, d=10, offset=0):
        self.d = d
        self.offset = offset
        self.domain = (-5.12, 5.12)
        self.f_min  = 0.0
        self.minimum = np.zeros(d)

    def generate(self):
        return {'objective': DifferentPowers(self.d, offset=self.offset)}


class MultiWellFunction(cbx_objective):
    """Sum of independent per-coordinate multi-well potentials, each with
    several locally stable minima plus a linear bias breaking the degeneracy
    to a unique global optimum: V_i(y) = (y^2-b^2)^2 - h*cos(k*y) + bias_i*y.

    A synthetic, Rastrigin-like separable multimodal test function (bistable
    quartic envelope with a cosine ripple layered on top), not tied to any
    particular physical system."""
    def __init__(self, bias, b=2.0, h=0.3, k=3.0):
        super().__init__()
        self.bias = np.asarray(bias)
        self.b, self.h, self.k = b, h, k

    def apply(self, x):
        return np.sum((x**2 - self.b**2)**2 - self.h * np.cos(self.k * x) + self.bias * x, axis=-1)


class RastriginLikeProblem(Problem):
    """Separable multimodal test function (MultiWellFunction): a bistable
    quartic well with a cosine ripple per coordinate, in the same spirit as
    Rastrigin but with a different well shape and a per-coordinate linear
    bias that breaks the degeneracy to a unique global optimum."""
    def __init__(self, d=10, b=2.0, h=0.3, k=3.0, bias_scale=1.0):
        self.d = d
        self.b, self.h, self.k = b, h, k
        self.domain = (-5.12, 5.12)
        self.f_min = None  # depends on the random bias; set in generate()
        self.bias = np.random.uniform(-bias_scale, bias_scale, size=d)
        self.minimum = np.array([
            _grid_argmin_1d(lambda y: (y**2 - b**2)**2 - h * np.cos(k * y) + bi * y, *self.domain)
            for bi in self.bias
        ])

    def generate(self):
        obj = MultiWellFunction(self.bias, b=self.b, h=self.h, k=self.k)
        self.f_min = float(obj(self.minimum))
        return {'objective': obj}


def _grid_argmin_1d(f, lo, hi, n=20001):
    ys = np.linspace(lo, hi, n)
    return ys[np.argmin(f(ys))] 
 
def banded_crosstalk(d, decay=0.4, bandwidth=3, coupling=0.3, seed=None):
    """Positive, banded, diagonally-dominant thermal-crosstalk matrix.
 
    W_ij = decay**|i-j| for |i-j|<=bandwidth (else 0), then each row's
    off-diagonal part is scaled so its sum = `coupling` (<1), and the
    diagonal set to 1. Strict diagonal dominance => invertible, and
    condition number grows smoothly with `coupling`.
 
    coupling -> 0 : W ~ I (easy).   coupling -> 1 : near-singular (hard).
    """
    assert 0.0 <= coupling < 1.0
    rng = np.random.default_rng(seed)
    idx = np.arange(d)
    dist = np.abs(idx[:, None] - idx[None, :])
    W = (decay ** dist) * (dist <= bandwidth) * (dist > 0)   # off-diagonals
    W *= rng.uniform(0.7, 1.3, W.shape)                      # mild asymmetry
    row = W.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    W = coupling * W / row                                   # off-diag rows sum to `coupling`
    np.fill_diagonal(W, 1.0)
    return W
 
 
import numpy as np


class MicroringWell(cbx_objective):
    """Per-ring all-pass through-port transmission summed over rings.

    Single-ring transmission is the standard all-pass intensity response
    (Bogaerts et al., Laser & Photonics Reviews 6(1):47-73, 2012):

        T(phi) = (a^2 - 2 a r cos phi + r^2) / (1 - 2 a r cos phi + (a r)^2),

    with r the self-coupling coefficient and a the round-trip amplitude
    transmission (loss). T is 2*pi-periodic with minima at phi = 2*pi*N,
    reaching 0 at critical coupling a = r. The array objective is the sum
    of per-ring transmissions; W (applied outside) mixes heater drives into
    ring phases. Rings may be identical (scalar a,r) or non-identical
    (array a,r) -- the method does not require identical g.
    """
    def __init__(self, phi0, a=0.95, r=0.85):
        super().__init__()
        self.phi0 = np.asarray(phi0)
        self.a = np.asarray(a)
        self.r = np.asarray(r)

    def apply(self, x):
        phi = x + self.phi0
        a, r = self.a, self.r
        num = a**2 - 2*a*r*np.cos(phi) + r**2
        den = 1.0 - 2*a*r*np.cos(phi) + (a*r)**2
        return (num/den).sum(axis=-1)      # sum over rings; use .mean for mean
 
 
class MicroringProblem(Problem):
    """Identical microring array with unknown thermal crosstalk supplied by
    generate_transformed (recommend a banded-positive W, not a rotation).

    Single-ring transmission is the all-pass through-port intensity response
    (Bogaerts et al., Laser & Photonics Reviews 6(1):47-73, 2012),
    parametrized by self-coupling r and round-trip amplitude transmission a;
    it is 2*pi-periodic with minima at phi = 2*pi*N and reaches 0 at critical
    coupling a = r. The array objective sums per-ring transmissions.

    `self.minimum` is the per-ring resonance nearest 0 in UNMIXED coords; the
    base class maps it through Winv. The objective is invariant under adding
    a full FSR (2*pi in phase) to any ring, so several equally-good global
    minimizers exist -- score with `distance_to_minimum` (phase residual) or
    the optimality gap f(x) - f_min, both order-agnostic.
    """
    name = 'microring'

    def __init__(self, d=16, a=0.95, r=0.85, phi_scale=np.pi, seed=None):
        self.d, self.a, self.r = d, a, r
        self.domain = (0.0, 8.0)                 # heater range (box)
        rng = np.random.default_rng(seed)
        self.phi0 = rng.uniform(-phi_scale, phi_scale, d)

        def _T(y, p):                            # single-ring transmission
            phi = y + p
            num = a**2 - 2*a*r*np.cos(phi) + r**2
            den = 1.0 - 2*a*r*np.cos(phi) + (a*r)**2
            return num/den

        # nearest resonance (T minimal at phi = 2*pi*N): unmixed minimizer
        self.minimum = np.array([
            _grid_argmin_1d(lambda y, p=p: _T(y, p), self.domain[0], self.domain[1])
            for p in self.phi0
        ])
        self.f_min = None

    def generate(self):
        obj = MicroringWell(self.phi0, self.a, self.r)
        self.f_min = float(obj(self.minimum))    # = d * Tmin, invariant under exact linear W
        return {'objective': obj}

    def distance_to_minimum(self, x, mixed=True, ord=np.inf):
        x = np.asarray(x).reshape(-1)
        y = x @ self.W.T if (mixed and hasattr(self, 'W')) else x
        z = (y + self.phi0 + np.pi) % (2 * np.pi) - np.pi
        return float(np.linalg.norm(z, ord=ord))