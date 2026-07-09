"""Monte Carlo estimation of the logical error rate for a given code
distance, physical error rate, and decoder."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .decoders import mwpm, union_find
from .noise import sample_errors
from .stabilizer import ToricLattice

DECODERS = {
    "mwpm": mwpm.decode,
    "union_find": union_find.decode,
}


@dataclass
class RunResult:
    L: int
    p: float
    decoder: str
    shots: int
    logical_errors: int
    total_decode_seconds: float

    @property
    def logical_error_rate(self) -> float:
        return self.logical_errors / self.shots

    @property
    def mean_decode_seconds(self) -> float:
        return self.total_decode_seconds / self.shots


def run_trials(
    lattice: ToricLattice,
    p: float,
    decoder: str,
    shots: int,
    rng: np.random.Generator,
) -> RunResult:
    decode_fn = DECODERS[decoder]
    logical_errors = 0
    total_time = 0.0
    for _ in range(shots):
        h_err, v_err = sample_errors(lattice.L, p, rng)
        syndrome = lattice.syndrome(h_err, v_err)

        t0 = time.perf_counter()
        h_corr, v_corr = decode_fn(lattice, syndrome)
        total_time += time.perf_counter() - t0

        # The correction must reproduce the observed syndrome exactly --
        # otherwise the decoder itself is buggy, not merely "unlucky".
        residual_syndrome = lattice.syndrome(h_err ^ h_corr, v_err ^ v_corr)
        assert not residual_syndrome.any(), "decoder returned a correction with the wrong syndrome"

        if not lattice.is_trivial(h_err ^ h_corr, v_err ^ v_corr):
            logical_errors += 1

    return RunResult(
        L=lattice.L,
        p=p,
        decoder=decoder,
        shots=shots,
        logical_errors=logical_errors,
        total_decode_seconds=total_time,
    )
