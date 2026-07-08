import random

from lpn_bkw.bkw import eliminate_block, reduce_all_but_target, solve_block
from lpn_bkw.oracle import LPNOracle, dot_mod2


def test_eliminate_block_zeroes_target_block_and_composes_labels():
    # 8-bit space, block 0 = bits [0,4), block 1 = bits [4,8).
    block0_mask = 0b00001111
    samples = [
        (0b00111010, 1),
        (0b01011010, 0),  # shares block-0 bits (...1010) with the first
        (0b11000001, 1),
        (0b00100001, 1),  # shares block-0 bits (...0001) with the third
    ]
    out = eliminate_block(samples, block0_mask)
    assert len(out) == 2
    for a, b in out:
        assert a & block0_mask == 0
    # First pair: pivot (0b00111010,1), combined -> a=0b01100000, b=1
    assert (0b01100000, 1) in out
    # Second pair: pivot (0b11000001,1), combined -> a=0b11100000, b=0
    assert (0b11100000, 0) in out


def test_eliminate_block_singleton_bucket_is_dropped():
    samples = [(0b0011, 1)]
    out = eliminate_block(samples, block_mask=0b1111)
    assert out == []


def test_reduce_all_but_target_leaves_only_target_bits_set():
    n, b = 12, 4
    rng = random.Random(42)
    oracle = LPNOracle(n, tau=0.0, rng=rng)
    samples = oracle.sample(400)
    for target in range(n // b):
        reduced = reduce_all_but_target(samples, n, b, target)
        assert reduced, "expected surviving samples for a noiseless instance"
        other_bits_mask = ((1 << n) - 1) ^ (((1 << b) - 1) << (target * b))
        for a, _ in reduced:
            assert a & other_bits_mask == 0


def test_solve_block_recovers_exact_block_when_noiseless():
    n, b = 12, 4
    rng = random.Random(99)
    secret = 0b101100111010
    oracle = LPNOracle(n, tau=0.0, secret=secret, rng=rng)
    samples = oracle.sample(400)
    for target in range(n // b):
        reduced = reduce_all_but_target(samples, n, b, target)
        expected_block = (secret >> (target * b)) & ((1 << b) - 1)
        guess, score, margin = solve_block(reduced, target, b)
        assert guess == expected_block
        # Noiseless: every surviving sample agrees with the true guess.
        assert score == len(reduced)
        assert margin > 0
