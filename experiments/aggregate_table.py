"""
Scan the per-run JSON cache for a benchmark config and print a success-rate
table: rows = methods, columns = problems. Also writes a .tex file with a
booktabs table of the same data, for inclusion in a paper.

Usage:
    python aggregate_table.py <config.yaml>
"""
import sys, json
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from config_utils import load_config

METHOD_LABELS = {
    'cbo_standard': r'CBO on $g$',
    'cbo_transformed': r'CBO on $f=g \circ W$',
    'rotfun': r'CBO on $\tilde{g} = f \circ \tilde{W}^\dagger$',
    'rotnoise': r'CBO with transformed noise',
    'splitdim': r'Componentwise CBO on $\tilde{g}$',
    'cmaes': r'BIPOP-CMA-ES on $f=g \circ W$',
}

PROBLEM_LABELS = {
    'rastrigin': 'Rastrigin',
    'rastrigin_like': 'Rastrigin-Like',
    'sphere': 'Squared Norm',
    'different_powers': 'Different Powers',
    'different_powers_offset0': 'Different Powers ($S=0$)',
    'different_powers_offset10': 'Different Powers ($S=10$)',
    'microring': 'Microring Array',
}


def problem_label(problem_cfg):
    """Must match run_benchmark.py's problem_label -- the identifier used for
    result caching/aggregation, defaulting to `name` unless `label` is set to
    disambiguate multiple entries sharing the same `name`."""
    return problem_cfg.get('label', problem_cfg['name'])


def tex_escape(s):
    return s.replace('_', r'\_')


def build_latex_table(methods, problems, successes, q_recoveries):
    col_spec = 'l' + 'c' * len(problems)
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\begin{tabular}{' + col_spec + '}',
        r'\toprule',
        'Method & ' + ' & '.join(PROBLEM_LABELS.get(p, tex_escape(p)) for p in problems) + r' \\',
        r'\midrule',
    ]
    for m in methods:
        cells = []
        for p in problems:
            runs = successes[m][p]
            if not runs:
                cells.append('n/a')
            else:
                cell = f'{100 * np.mean(runs):.0f}\\%'
                q_runs = q_recoveries[m][p]
                if q_runs:
                    fracs = [q['true'] / (q['true'] + q['missed']) for q in q_runs]
                    cell += f' ({100 * np.mean(fracs):.0f}\\%)'
                cells.append(cell)
        lines.append(tex_escape(METHOD_LABELS.get(m, m)) + ' & ' + ' & '.join(cells) + r' \\')
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\caption{Success rates.}',
        r'\label{tab:success-rates}',
        r'\end{table}',
    ]
    return '\n'.join(lines) + '\n'


def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_d100.yaml'
    cfg = load_config(config_name)
    config_stem = Path(config_name).stem
    results_dir = Path(__file__).parent.parent / 'results'
    runs_dir = results_dir / f'{config_stem}_runs'

    methods = cfg['methods']
    problems = [problem_label(p) for p in cfg['problems']]
    n_seeds = cfg['n_seeds']

    successes = {m: {p: [] for p in problems} for m in methods}
    q_recoveries = {m: {p: [] for p in problems} for m in methods}
    missing = []
    for m in methods:
        for p in problems:
            for s in range(n_seeds):
                path = runs_dir / f'{m}_{p}_s{s}.json'
                if not path.exists():
                    missing.append(path.name)
                    continue
                with open(path) as fh:
                    res = json.load(fh)
                successes[m][p].append(res['success'])
                q_recovery = res.get('q_recovery')
                if q_recovery is not None:
                    q_recoveries[m][p].append(q_recovery)

    label_width = max(len(METHOD_LABELS.get(m, m)) for m in methods) + 2
    col_width = max(12, max(len(p) for p in problems) + 2)

    header = ' ' * label_width + ''.join(f'{p:>{col_width}}' for p in problems)
    print(header)
    print('-' * len(header))
    for m in methods:
        row = f'{METHOD_LABELS.get(m, m):<{label_width}}'
        for p in problems:
            runs = successes[m][p]
            if not runs:
                cell = 'n/a'
            else:
                cell = f'{100 * np.mean(runs):.0f}% ({len(runs)}/{n_seeds})'
            row += f'{cell:>{col_width}}'
        print(row)

    if missing:
        print(f'\n{len(missing)} run(s) not yet completed, e.g.: {missing[:5]}')

    tex = build_latex_table(methods, problems, successes, q_recoveries)
    tex_path = results_dir / f'{config_stem}.tex'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(tex_path, 'w') as fh:
        fh.write(tex)
    print(f'\nLaTeX table written to {tex_path}')


if __name__ == '__main__':
    main()
