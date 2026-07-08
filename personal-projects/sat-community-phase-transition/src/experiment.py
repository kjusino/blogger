"""Sweep clause/variable ratio (alpha) x community mixing parameter (mu),
solving many random instances at each grid point and recording search
effort and realized modularity. Writes one row per solved instance to CSV.
"""

import csv
import random
import time

from .cnf import community_3sat
from .community import planted_modularity
from .solver import solve

DEFAULT_N_VARS = 80
DEFAULT_N_COMMUNITIES = 4
DEFAULT_DECISION_CAP = 2_000_000

CSV_FIELDS = [
    "n_vars", "n_communities", "alpha", "mu", "trial", "n_clauses",
    "satisfiable", "hit_cap", "decisions", "backtracks", "modularity_q",
    "runtime_sec",
]


def run_sweep(alphas, mus, trials, n_vars=DEFAULT_N_VARS,
              n_communities=DEFAULT_N_COMMUNITIES,
              decision_cap=DEFAULT_DECISION_CAP, seed=0, progress=None):
    """Run the full (alpha, mu, trial) grid and return a list of dict rows.

    `progress`, if given, is called as progress(done, total, alpha, mu)
    after each grid point (all trials for that alpha/mu combination).
    """
    rng = random.Random(seed)
    rows = []
    total_points = len(alphas) * len(mus)
    done_points = 0

    for mu in mus:
        for alpha in alphas:
            n_clauses = round(alpha * n_vars)
            for trial in range(trials):
                cnf = community_3sat(
                    n_vars=n_vars, n_clauses=n_clauses,
                    n_communities=n_communities, mu=mu, rng=rng,
                )
                t0 = time.time()
                result = solve(cnf, decision_cap=decision_cap)
                runtime = time.time() - t0
                q = planted_modularity(cnf)

                rows.append({
                    "n_vars": n_vars,
                    "n_communities": n_communities,
                    "alpha": alpha,
                    "mu": mu,
                    "trial": trial,
                    "n_clauses": n_clauses,
                    "satisfiable": int(result.satisfiable),
                    "hit_cap": int(result.hit_cap),
                    "decisions": result.decisions,
                    "backtracks": result.backtracks,
                    "modularity_q": q,
                    "runtime_sec": runtime,
                })

            done_points += 1
            if progress is not None:
                progress(done_points, total_points, alpha, mu)

    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
