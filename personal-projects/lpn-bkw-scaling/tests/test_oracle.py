import random

from lpn_bkw.oracle import LPNOracle, dot_mod2, popcount


def test_popcount():
    assert popcount(0) == 0
    assert popcount(0b1011) == 3
    assert popcount((1 << 10) - 1) == 10


def test_dot_mod2_basic():
    assert dot_mod2(0b101, 0b101) == 0  # 1+1 = 2 -> mod 2 = 0
    assert dot_mod2(0b101, 0b100) == 1
    assert dot_mod2(0b000, 0b111) == 0


def test_oracle_rejects_bad_tau():
    for bad in (-0.1, 0.5, 0.9):
        try:
            LPNOracle(8, bad)
            assert False, f"tau={bad} should have been rejected"
        except ValueError:
            pass


def test_oracle_secret_fits_in_n_bits():
    try:
        LPNOracle(4, 0.1, secret=1 << 10)
        assert False
    except ValueError:
        pass


def test_oracle_noiseless_labels_match_secret():
    rng = random.Random(1234)
    oracle = LPNOracle(16, tau=0.0, secret=0b1010110000111100, rng=rng)
    samples = oracle.sample(500)
    for a, b in samples:
        assert b == dot_mod2(a, oracle.secret)
    assert oracle.num_queries == 500


def test_oracle_noise_rate_matches_tau_statistically():
    rng = random.Random(7)
    tau = 0.25
    oracle = LPNOracle(20, tau=tau, rng=rng)
    samples = oracle.sample(20000)
    flips = sum(1 for a, b in samples if b != dot_mod2(a, oracle.secret))
    empirical_tau = flips / len(samples)
    assert abs(empirical_tau - tau) < 0.02
