"""LPN sample oracle.

Secrets and query vectors are represented as Python ints used as length-n
bitmasks. This keeps XOR / dot-product-mod-2 operations exact and fast for
the small n (<= ~32) this study needs, with no floating point involved.
"""

import random


def popcount(x: int) -> int:
    return x.bit_count()


def dot_mod2(a: int, b: int) -> int:
    """<a, b> mod 2 for two n-bit vectors packed as ints."""
    return popcount(a & b) & 1


class LPNOracle:
    """Samples (a_i, b_i) with b_i = <a_i, s> XOR e_i, e_i ~ Bernoulli(tau)."""

    def __init__(self, n: int, tau: float, secret: int = None, rng: random.Random = None):
        if not (0 <= tau < 0.5):
            raise ValueError("tau must be in [0, 0.5)")
        self.n = n
        self.tau = tau
        self.rng = rng or random.Random()
        if secret is None:
            secret = self.rng.getrandbits(n)
        if secret.bit_length() > n:
            raise ValueError("secret does not fit in n bits")
        self.secret = secret
        self.num_queries = 0

    def sample(self, count: int):
        """Draw `count` fresh (a, b) samples. Counts against num_queries."""
        out = []
        n, s, tau, rng = self.n, self.secret, self.tau, self.rng
        for _ in range(count):
            a = rng.getrandbits(n)
            noiseless = dot_mod2(a, s)
            e = 1 if rng.random() < tau else 0
            out.append((a, noiseless ^ e))
        self.num_queries += count
        return out
