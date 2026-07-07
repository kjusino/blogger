"""Closed-form asymptotic predictions this project tests against simulation.

Penrose (1997, "The longest edge of the random minimal spanning tree";
1999, "A strong law for the longest edge of the minimal spanning tree";
see also Penrose's 2003 book *Random Geometric Graphs*, Thm 13.2 and 13.17)
shows that for n points i.i.d. uniform on a unit-area torus, the random
geometric graph with connection radius r_n is connected asymptotically
almost surely iff

    n * pi * r_n^2 - log(n) -> +infinity,

and disconnected a.a.s. iff that quantity -> -infinity. The critical
window is centered on the radius where the two terms balance:

    r_c(n) = sqrt(log(n) / (pi * n)).

The falsifiable target used throughout this project is the dimensionless
ratio

    ratio(n) = r_empirical(n) / r_c(n),

where r_empirical(n) is an independently *measured* quantity (the mean
longest-MST-edge over simulated point sets), not a rearrangement of the
formula above. Penrose's theorem predicts ratio(n) -> 1 as n -> infinity;
this is a genuine empirical test, not a tautology.
"""

import math


def penrose_threshold_radius(n: int) -> float:
    """r_c(n) = sqrt(log(n) / (pi * n)) -- Penrose's connectivity threshold."""
    if n <= 1:
        raise ValueError("Penrose's threshold requires n > 1")
    return math.sqrt(math.log(n) / (math.pi * n))


def threshold_ratio(r_empirical: float, n: int) -> float:
    """r_empirical / r_c(n); theory predicts this ratio -> 1 as n grows."""
    return r_empirical / penrose_threshold_radius(n)
