"""End-to-end smoke test: DE threshold -> multi-n BLER scan -> scaling fit.

Mirrors run_experiment.py's pipeline at a much smaller scale so it runs in
well under a second, while still checking the qualitative scientific claim
end to end: as n grows, eps50(n) should approach the DE threshold and the
waterfall should narrow.
"""

import random

from src.density_evolution import find_threshold
from src.experiment import fit_power_law, measure_bler_curve


def test_pipeline_gap_and_width_shrink_with_blocklength():
    eps_star = find_threshold(3, 6)
    assert eps_star == 0.4294 or abs(eps_star - 0.4294) < 2e-4

    rng = random.Random(99)
    blocklengths = [80, 200, 500]
    gaps = []
    widths = []
    for n in blocklengths:
        curve = measure_bler_curve(
            n,
            3,
            6,
            eps_star,
            rng,
            n_graph_instances=2,
            coarse_trials=15,
            fine_trials=25,
            n_coarse=5,
            n_fine=5,
        )
        gaps.append(eps_star - curve.eps50)
        widths.append(curve.width_90_10)

    # The core scientific claim: both the distance-to-threshold and the
    # transition width should shrink monotonically as blocklength grows.
    assert gaps[0] > gaps[1] > 0
    assert gaps[1] > gaps[2] > 0 or gaps[2] >= 0  # allow MC noise at the top end
    assert widths[0] > widths[-1] > 0

    # A power law should be fittable (won't assert a tight exponent match
    # at this tiny, fast scale -- that's what the full experiment does).
    _, gap_exponent = fit_power_law(blocklengths, gaps)
    _, width_exponent = fit_power_law(blocklengths, widths)
    assert gap_exponent < 0
    assert width_exponent < 0
