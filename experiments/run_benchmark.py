"""
Benchmark CBO / RotFunCBO / RotNoiseCBO / SplitDimCBO across benchmark problems,
under a fixed total function-evaluation budget, over many seeds.

Usage:
    python -u run_benchmark.py <config.yaml>

One config = one whole sweep (all methods x problems x seeds). Each
(method, problem, seed) run is cached to its own JSON file under
results/<config_stem>_runs/, so a killed/resumed job skips finished work.
Aggregate into a success-rate table with aggregate_table.py.
"""
import sys, os, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cma
from cbx.dynamics import CBO
from cbx.utils.termination import max_eval_term
from cbx.scheduler import multiply

sys.path.append(str(Path(__file__).parent.parent))
from config_utils import load_config
from problems import RastriginProblem, SphereProblem, DifferentPowersProblem, RastriginLikeProblem, MicroringProblem
from CBO_utils import RotFunCBO, RotNoiseCBO, SplitDimCBO, estimate_Q
from experimentUtils import check_unit_weight_estimates

RESULTS_DIR = Path(__file__).parent.parent / 'results'

PROBLEM_BUILDERS = {
    'rastrigin': RastriginProblem,
    'sphere': SphereProblem,
    'different_powers': DifferentPowersProblem,
    'rastrigin_like': RastriginLikeProblem,
    'microring': MicroringProblem,
}

# problem_cfg entries may carry these to override the global cbo_kwargs /
# qest_kwargs / split_cbo_kwargs for that problem only (e.g. microring needs
# a different n_hess/sval_cutoff and dt/sigma than the other problems) --
# excluded from the kwargs forwarded to the Problem constructor. 'label' is
# likewise reserved: it's how two entries sharing the same `name` (e.g.
# different_powers with offset=0 vs offset=1) get distinct result files /
# table columns instead of colliding.
RESERVED_PROBLEM_KEYS = {'name', 'label', 'cbo_kwargs', 'qest_kwargs', 'split_cbo_kwargs', 'success_thresh'}


def problem_label(problem_cfg):
    """Identifier used for result caching/aggregation -- defaults to `name`,
    but a problem_cfg entry can set `label` to disambiguate multiple entries
    that share the same `name` with different constructor kwargs."""
    return problem_cfg.get('label', problem_cfg['name'])


def hess_eval_cost(d, n_hess):
    """Number of black-box evaluations estimate_Q's Hessian sampling consumes."""
    return (2 * d**2 + 2 * d) * n_hess


def run_cmaes(f, P, d, budget, seed):
    """BIPOP-CMA-ES on the black-box (transformed) objective -- the standard
    strong, structure-agnostic baseline: no separability assumption, so
    unlike rotfun/rotnoise/splitdim it spends its whole budget optimizing,
    none on Q-estimation.

    Configuration follows pycma/Hansen's own published defaults for exactly
    this class of problem (BBOB-style multimodal, as these benchmark
    functions are), not tuned per-problem here:
      - restarts=9, bipop=True: bi-population restarts, the configuration
        that won the original BBOB-2009 comparison and is pycma's own
        documented example for a multimodal (Rastrigin) objective.
      - sigma0 = 0.3 * domain width: pycma's standard rule-of-thumb absent
        problem-specific knowledge of the optimum's location.
      - a fresh random start point per restart (x0 as a domain-sampling
        callable, matching pycma's own multimodal example) rather than one
        fixed x0, so restarts actually explore different basins.
      - bounds=[-1e8, 1e8]: not a real constraint (matches the implicit
        safety clip cbx's own CBO applies by default when no post_process
        is given), just guards against numerical blow-up so the comparison
        stays apples-to-apples with the other (likewise unconstrained)
        methods.
    """
    low, high = P.domain
    sigma0 = 0.3 * (high - low)

    def x0_sampler():
        return low + (high - low) * np.random.rand(d)

    options = {
        'maxfevals': budget,
        'bounds': [-1e8, 1e8],
        'seed': int(seed) + 1,   # cma treats seed=0 as "use current time"
        'verbose': -9,
        'verb_log': 0,
    }
    res = cma.fmin(f, x0_sampler, sigma0, options, restarts=9, bipop=True)
    x_best = np.asarray(res[0]).reshape(-1)
    n_evals = int(res[3])
    return x_best, n_evals


def run_path(config_stem, method, problem_name, seed):
    d = RESULTS_DIR / f'{config_stem}_runs'
    d.mkdir(parents=True, exist_ok=True)
    return d / f'{method}_{problem_name}_s{seed}.json'


def build_landscape(problem_cfg, d, transform_cfg, N, seed):
    """Seeds deterministically from (problem_cfg, seed) only, so every method
    run against the same seed sees the identical problem, transform, and x0."""
    np.random.seed(seed)
    name = problem_cfg['name']
    kwargs = {k: v for k, v in problem_cfg.items() if k not in RESERVED_PROBLEM_KEYS}
    P = PROBLEM_BUILDERS[name](d=d, **kwargs)
    f = P.generate_transformed(**transform_cfg)['objective']
    x0 = P.sample(1, N)
    return P, f, x0


def run_one(method, problem_cfg, cfg, seed):
    d = cfg['d']
    budget = cfg['max_query_budget']
    N = cfg['N']
    thresh = problem_cfg.get('success_thresh', cfg['success_thresh'])
    ord_ = cfg.get('success_ord', 'inf')
    ord_ = np.inf if ord_ == 'inf' else ord_

    P, f, x0 = build_landscape(problem_cfg, d, cfg['transform'], N, seed)

    cbo_kwargs = dict(cfg['cbo_kwargs']) | dict(problem_cfg.get('cbo_kwargs', {}))
    qest_kwargs = dict(cfg['qest_kwargs']) | dict(problem_cfg.get('qest_kwargs', {}))
    sched_cfg = cfg.get('scheduler')
    q_recovery = None

    def make_sched():
        return multiply(**sched_cfg) if sched_cfg else None

    def q_recovery_check(Q):
        q_true, q_wrong, q_missed = check_unit_weight_estimates(f.Q, Q, just_counts=True, prec=1e-8 / (2**0.5))
        return {'true': q_true, 'wrong': q_wrong, 'missed': q_missed}

    if method == 'cbo_standard':
        term = max_eval_term(max_eval=budget)
        dyn = CBO(f.f, x=x0.copy(), **cbo_kwargs, term_criteria=[term])
        dyn.optimize(sched=make_sched())
        x_best = dyn.best_particle.squeeze()
        mixed = False
        n_evals = int(dyn.num_f_eval[0])

    elif method == 'cbo_transformed':
        term = max_eval_term(max_eval=budget)
        dyn = CBO(f, x=x0.copy(), **cbo_kwargs, term_criteria=[term])
        dyn.optimize(sched=make_sched())
        x_best = dyn.best_particle.squeeze()
        mixed = True
        n_evals = int(dyn.num_f_eval[0])

    elif method == 'cmaes':
        x_best, n_evals = run_cmaes(f, P, d, budget, seed)
        mixed = True

    elif method in ('rotfun', 'rotnoise'):
        hess_cost = hess_eval_cost(d, qest_kwargs['n_hess'])
        remaining = max(1, budget - hess_cost)
        Q = estimate_Q(f, d, **qest_kwargs)
        q_recovery = q_recovery_check(Q)
        term = max_eval_term(max_eval=remaining)
        cls = RotFunCBO if method == 'rotfun' else RotNoiseCBO
        dyn = cls(f, x=x0.copy(), Q=Q, **cbo_kwargs, term_criteria=[term])
        dyn.optimize(sched=make_sched())
        x_best = dyn.best_particle.squeeze()
        if method == 'rotfun':
            x_best = x_best @ np.linalg.pinv(Q).T
        mixed = True
        n_evals = hess_cost + int(dyn.num_f_eval[0])

    elif method == 'splitdim':
        hess_cost = hess_eval_cost(d, qest_kwargs['n_hess'])
        remaining = max(1, budget - hess_cost)
        Q = estimate_Q(f, d, **qest_kwargs)
        q_recovery = q_recovery_check(Q)
        split_kwargs = dict(cfg['split_cbo_kwargs']) | dict(problem_cfg.get('split_cbo_kwargs', {}))
        term_split = max_eval_term(max_eval=max(1, remaining // d))
        split_kwargs['term_criteria'] = [term_split]
        split = SplitDimCBO(d=d, CBO_kwargs=split_kwargs, sched=make_sched())
        # parallel=False: the outer ProcessPoolExecutor already parallelizes
        # across (method, problem, seed) jobs -- nesting SplitDimCBO's own
        # joblib pool inside that would oversubscribe the machine's cores.
        x_best = split(f, Q=Q, x0=x0.copy(), parallel=False)
        mixed = True
        n_evals = hess_cost + remaining  # upper bound: each sub-run is capped there

    else:
        raise ValueError(f"unknown method '{method}'")

    x_best = np.asarray(x_best).reshape(-1)
    f_best = float(f.f(x_best)) if method == 'cbo_standard' else float(f(x_best))
    f_min = float(P.f_min)
    dist = P.distance_to_minimum(x_best, mixed=mixed, ord=ord_)
    gap = f_best - f_min
    return {
        'method': method, 'problem': problem_label(problem_cfg), 'seed': seed,
        'x_best': x_best.tolist(), 'f_best': f_best, 'f_min': f_min,
        'dist': dist, 'gap': gap, 'success': bool(gap < thresh), 'thresh': thresh,
        'n_evals': int(n_evals), 'q_recovery': q_recovery,
    }


def run_one_cached(method, problem_cfg, cfg, seed, config_stem):
    path = run_path(config_stem, method, problem_label(problem_cfg), seed)
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    result = run_one(method, problem_cfg, cfg, seed)
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w') as fh:
        json.dump(result, fh)
    tmp.replace(path)
    return result


def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_d100.yaml'
    cfg = load_config(config_name)
    config_stem = Path(config_name).stem

    methods = cfg['methods']
    problems = cfg['problems']
    n_seeds = cfg['n_seeds']

    jobs = [(m, p, s) for m in methods for p in problems for s in range(n_seeds)]
    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    print(f"Running {len(jobs)} (method, problem, seed) jobs with {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(run_one_cached, m, p, cfg, s, config_stem): (m, problem_label(p), s)
            for m, p, s in jobs
        }
        for fut in as_completed(futures):
            m, pname, s = futures[fut]
            try:
                res = fut.result()
                print(f"  done: {m:16s} {pname:12s} seed={s:3d}  "
                      f"success={res['success']!s:5s}  dist={res['dist']:.4g}  "
                      f"f_best={res['f_best']:.4g} (f_min={res['f_min']:.4g})  evals={res['n_evals']}")
            except Exception as e:
                print(f"  FAILED: {m} {pname} seed={s}: {e!r}")


if __name__ == '__main__':
    main()
