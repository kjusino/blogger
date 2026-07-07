from src.gf2_linalg import gf2_rank, row_from_indices


def test_row_from_indices_packs_bits():
    row = row_from_indices([0, 2, 3])
    assert row == 0b1101


def test_rank_of_identity_is_full():
    rows = [row_from_indices([i]) for i in range(5)]
    assert gf2_rank(rows, 5) == 5


def test_rank_of_all_zero_rows_is_zero():
    rows = [0, 0, 0]
    assert gf2_rank(rows, 4) == 0


def test_rank_detects_linear_dependence_mod_2():
    # row2 = row0 XOR row1 (dependent over GF(2))
    row0 = row_from_indices([0, 1])
    row1 = row_from_indices([1, 2])
    row2 = row0 ^ row1
    assert gf2_rank([row0, row1, row2], 3) == 2


def test_rank_of_empty_matrix_is_zero():
    assert gf2_rank([], 5) == 0


def test_rank_all_ones_row_is_one():
    rows = [row_from_indices([0, 1, 2, 3])] * 3
    assert gf2_rank(rows, 4) == 1


def test_rank_matches_brute_force_gf2_gaussian_elimination():
    # Cross-check against a slow, independent GF(2) rank implementation.
    import itertools
    import random

    rng = random.Random(0)
    for _ in range(20):
        n_rows, n_cols = rng.randint(1, 6), rng.randint(1, 6)
        matrix = [[rng.randint(0, 1) for _ in range(n_cols)] for _ in range(n_rows)]
        rows = [row_from_indices([c for c in range(n_cols) if matrix[r][c]]) for r in range(n_rows)]
        assert gf2_rank(rows, n_cols) == _brute_force_gf2_rank(matrix, n_rows, n_cols)


def _brute_force_gf2_rank(matrix, n_rows, n_cols):
    mat = [row[:] for row in matrix]
    rank = 0
    for col in range(n_cols):
        pivot = None
        for r in range(rank, n_rows):
            if mat[r][col]:
                pivot = r
                break
        if pivot is None:
            continue
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        for r in range(n_rows):
            if r != rank and mat[r][col]:
                mat[r] = [(a ^ b) for a, b in zip(mat[r], mat[rank])]
        rank += 1
        if rank == n_rows:
            break
    return rank
