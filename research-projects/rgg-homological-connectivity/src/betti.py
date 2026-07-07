"""Betti number computation for a 2-skeleton Vietoris-Rips complex, over GF(2).

Betti_0 = number of connected components (standard fact, and also equals
n - rank(d1) since over GF(2) the signed and unsigned incidence matrices
coincide -- computed here via union-find, which is exact and much cheaper
than a matrix rank).

Betti_1 = dim(ker d1) - rank(d2)
        = (|E| - rank(d1)) - rank(d2)
        = (|E| - (n - components)) - rank(d2)
"""

from typing import Tuple

from .gf2_linalg import gf2_rank
from .graph import num_components
from .simplicial_complex import SimplicialComplex, boundary2_rows


def betti_numbers(complex_: SimplicialComplex) -> Tuple[int, int]:
    """Return (betti_0, betti_1) for the given 2-skeleton Rips complex."""
    n = complex_.n_vertices
    c = num_components(n, complex_.edges)
    rank_d1 = n - c
    rows = boundary2_rows(complex_)
    rank_d2 = gf2_rank(rows, complex_.num_edges) if complex_.num_edges else 0
    dim_ker_d1 = complex_.num_edges - rank_d1
    betti_1 = dim_ker_d1 - rank_d2
    return c, betti_1
