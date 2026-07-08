"""Classical (pivot-and-eliminate) BKW attack on LPN.

Recovers the secret one b-bit block at a time. For a target block, every
other block is eliminated in turn: samples are bucketed by their bits in
that block, the first sample seen in a bucket becomes the "pivot", and every
later sample in the same bucket is XORed with the pivot and emitted (the
pivot itself is not re-emitted). This zeroes the eliminated block in every
output sample while composing the noise bits, which is what squares the
noise bias at each level.

After all non-target blocks are eliminated, the surviving samples depend
only on the target block's bits. The target block is then recovered by
exhaustive correlation: try every 2^b candidate value and keep the one
whose predicted labels agree best with the (noise-amplified) samples.
"""

import time
from collections import defaultdict

from .oracle import LPNOracle, dot_mod2, popcount
from . import theory


def eliminate_block(samples, block_mask):
    """One pivot-and-eliminate reduction level for a single block mask."""
    pivots = {}
    out = []
    for a, b in samples:
        key = a & block_mask
        pivot = pivots.get(key)
        if pivot is None:
            pivots[key] = (a, b)
        else:
            pa, pb = pivot
            out.append((a ^ pa, b ^ pb))
    return out


def reduce_all_but_target(samples, n, b, target_idx):
    """Eliminate every block except `target_idx`, in ascending block order."""
    num_blocks = n // b
    for j in range(num_blocks):
        if j == target_idx:
            continue
        mask = ((1 << b) - 1) << (j * b)
        samples = eliminate_block(samples, mask)
        if not samples:
            break
    return samples


def solve_block(samples, target_idx, b):
    """Exhaustive correlation solve for the b bits of the target block.

    Returns (best_guess, best_score, margin) where margin is best_score
    minus the second-best score (a crude confidence signal).
    """
    shift = target_idx * b
    local_a = [(a >> shift) & ((1 << b) - 1) for a, _ in samples]
    labels = [lab for _, lab in samples]

    scores = [0] * (1 << b)
    for a_loc, lab in zip(local_a, labels):
        for g in range(1 << b):
            bit = (-1) ** (lab ^ (popcount(a_loc & g) & 1))
            scores[g] += bit

    best_g = max(range(1 << b), key=lambda g: scores[g])
    ordered = sorted(scores, reverse=True)
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0)
    return best_g, scores[best_g], margin


def recover_secret(n, b, oracle: LPNOracle, confidence_const: float = 20.0,
                    margin: float = 1.3, rng=None):
    """Run all `a` recovery passes and assemble the full n-bit guess."""
    if n % b != 0:
        raise ValueError("n must be a multiple of b")
    a_levels = n // b
    secret_hat = 0
    total_raw_queries = 0
    per_block = []

    for target in range(a_levels):
        n0 = int(theory.queries_per_pass(n, b, oracle.tau, confidence_const, margin))
        samples = oracle.sample(n0)
        total_raw_queries += n0
        reduced = reduce_all_but_target(samples, n, b, target)
        guess, score, score_margin = solve_block(reduced, target, b)
        secret_hat |= guess << (target * b)
        per_block.append({
            "target_block": target,
            "n0": n0,
            "surviving_samples": len(reduced),
            "score": score,
            "score_margin": score_margin,
        })

    return secret_hat, {
        "n": n,
        "b": b,
        "a_levels": a_levels,
        "total_raw_queries": total_raw_queries,
        "per_block": per_block,
    }


def attack(n, b, tau, secret=None, rng=None, confidence_const: float = 20.0,
           margin: float = 1.3):
    """End-to-end attack: build an oracle, recover the secret, grade it."""
    oracle = LPNOracle(n, tau, secret=secret, rng=rng)
    t0 = time.perf_counter()
    secret_hat, stats = recover_secret(n, b, oracle, confidence_const, margin, rng)
    elapsed = time.perf_counter() - t0

    hamming = popcount(secret_hat ^ oracle.secret)
    stats.update({
        "tau": tau,
        "secret": oracle.secret,
        "secret_hat": secret_hat,
        "success": hamming == 0,
        "hamming_distance": hamming,
        "wall_time_sec": elapsed,
    })
    return stats
