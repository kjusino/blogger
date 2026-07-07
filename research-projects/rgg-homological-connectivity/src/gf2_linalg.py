"""Rank computation over GF(2), the field simplicial-homology boundary maps
in this project are defined over.

Each row of the matrix is packed into a single Python arbitrary-precision
integer (bit c set <=> entry in column c is 1). Row XOR is then a single
big-integer XOR, which CPython implements in C over machine words -- far
faster than a Python-level loop over individual bits.
"""

from typing import List, Sequence


def row_from_indices(indices: Sequence[int]) -> int:
    """Pack a sparse 0/1 row (given by the column indices that are 1) into an int."""
    row = 0
    for idx in indices:
        row |= 1 << idx
    return row


def gf2_rank(rows: List[int], num_cols: int) -> int:
    """Rank, over GF(2), of the matrix whose rows are the given bitmask integers.

    Implemented as an incremental "linear basis" (a.k.a. XOR basis): each row
    is reduced against a dict of basis vectors keyed by leading-bit position,
    using `int.bit_length()` (a single, near-O(1) operation on CPython's
    bignum representation) to jump straight to the next bit to eliminate,
    rather than scanning bit-by-bit or re-sweeping every other row whenever a
    pivot is found (the naive Gauss-Jordan approach this replaced). Each row
    is visited once; num_cols is unused but kept for API compatibility with
    the matrix-shape-based interface callers expect.
    """
    del num_cols  # bit_length()-based reduction does not need an explicit bound
    basis: dict = {}
    for row in rows:
        v = row
        while v:
            lead = v.bit_length() - 1
            pivot = basis.get(lead)
            if pivot is None:
                basis[lead] = v
                break
            v ^= pivot
    return len(basis)
