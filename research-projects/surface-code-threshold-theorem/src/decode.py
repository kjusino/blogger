"""Monte Carlo sampling + minimum-weight perfect matching decoding for a
stim circuit, with Wilson-score confidence intervals on the logical
error rate."""

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pymatching
import stim


@dataclass(frozen=True)
class DecodeResult:
    distance: int
    p: float
    shots: int
    logical_errors: int

    @property
    def logical_error_rate(self) -> float:
        return self.logical_errors / self.shots

    @property
    def confidence_interval(self) -> tuple[float, float]:
        return wilson_interval(self.logical_errors, self.shots)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    More reliable than the normal approximation when the observed rate
    is near 0 (common far below threshold, where a naive symmetric CI
    would extend below zero).
    """
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half_width = (z * sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def sample_and_decode(
    circuit: stim.Circuit, shots: int, seed: int, distance: int, p: float
) -> DecodeResult:
    """Sample `shots` runs of `circuit`, decode each with MWPM, and count
    how many end in a logical error (predicted observable != actual).

    `distance` and `p` are not recoverable from a compiled stim circuit,
    so the caller passes them through purely to label the result.
    """
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    sampler = circuit.compile_detector_sampler(seed=seed)
    detectors, observables = sampler.sample(shots=shots, separate_observables=True)

    predictions = matcher.decode_batch(detectors)
    errors = int(np.any(predictions != observables, axis=1).sum())

    return DecodeResult(distance=distance, p=p, shots=shots, logical_errors=errors)
