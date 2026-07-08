"""Finite groups used as grokking tasks, plus their irreducible representations.

Each group exposes:
  - `elements`: list of element ids (0..order-1)
  - `identity`, `order`
  - `mul(a, b)` / `inv(a)`: group operation and inverse, on element ids
  - `table()`: the full |G| x |G| Cayley (multiplication) table
  - `irreps()`: list of (label, dim, rho) where `rho(g)` returns the
    representation matrix (numpy array, possibly complex) of element `g`

The irreps are the ingredient the spectral-alignment analysis (grok/spectral.py)
needs: they define the group's natural harmonic basis (Fourier modes for an
abelian group, matrix-coefficient functions for a non-abelian one), against
which we measure whether a trained embedding table has organized itself along
group-theoretic lines.
"""
import numpy as np


class CyclicGroup:
    """Z/pZ under addition. Abelian; irreps are the classical DFT modes."""

    name = "cyclic"

    def __init__(self, p):
        self.p = p
        self.order = p
        self.elements = list(range(p))
        self.identity = 0

    def mul(self, a, b):
        return (a + b) % self.p

    def inv(self, a):
        return (-a) % self.p

    def table(self):
        p = self.p
        return np.array([[self.mul(a, b) for b in range(p)] for a in range(p)])

    def irreps(self):
        p = self.p
        irreps = []
        for k in range(p // 2 + 1):
            def rho(g, k=k):
                theta = 2 * np.pi * k * g / p
                return np.array([[np.exp(1j * theta)]])
            irreps.append((f"freq_{k}", 1, rho))
        return irreps


class DihedralGroup:
    """Symmetries of a regular n-gon (rotations + reflections), n odd.

    Elements are pairs (i, f) with i in Z_n (rotation power) and f in {0,1}
    (reflection flag), flattened to a single id `i * 2 + f`. Restricted to odd
    n so every group has exactly 2 linear irreps + (n-1)/2 two-dimensional
    irreps (the even-n case has 2 extra linear irreps and is not needed here).
    """

    name = "dihedral"

    def __init__(self, n):
        if n % 2 == 0:
            raise ValueError("DihedralGroup here only supports odd n")
        self.n = n
        self.order = 2 * n
        self.elements = list(range(2 * n))
        self.identity = self._pack(0, 0)

    def _pack(self, i, f):
        return (i % self.n) * 2 + (f % 2)

    def _unpack(self, g):
        return g // 2, g % 2

    def mul(self, a, b):
        i1, f1 = self._unpack(a)
        i2, f2 = self._unpack(b)
        sign = 1 if f1 == 0 else -1
        new_i = (i1 + sign * i2) % self.n
        new_f = (f1 + f2) % 2
        return self._pack(new_i, new_f)

    def inv(self, a):
        i, f = self._unpack(a)
        if f == 0:
            return self._pack((-i) % self.n, 0)
        return a  # every reflection is its own inverse

    def table(self):
        g = self.order
        return np.array([[self.mul(a, b) for b in range(g)] for a in range(g)])

    def irreps(self):
        n = self.n
        irreps = []

        def rho_trivial(g):
            return np.array([[1.0]])
        irreps.append(("trivial", 1, rho_trivial))

        def rho_sign(g, self=self):
            _, f = self._unpack(g)
            return np.array([[(-1.0) ** f]])
        irreps.append(("sign", 1, rho_sign))

        for k in range(1, (n - 1) // 2 + 1):
            def rho_k(g, k=k, self=self):
                i, f = self._unpack(g)
                theta = 2 * np.pi * k * i / n
                c, s = np.cos(theta), np.sin(theta)
                rot = np.array([[c, -s], [s, c]])
                if f == 0:
                    return rot
                reflect = np.array([[1.0, 0.0], [0.0, -1.0]])
                return rot @ reflect
            irreps.append((f"rho_{k}", 2, rho_k))
        return irreps


class QuaternionGroup:
    """Q8 = {1,-1,i,-i,j,-j,k,-k}. Smallest non-abelian group with a
    quaternionic-type (not realizable over R at its complex dimension)
    irrep, used here as a stress test for the spectral-alignment method.
    """

    name = "quaternion"
    _units = ["1", "i", "j", "k"]
    _labels = ["1", "-1", "i", "-i", "j", "-j", "k", "-k"]

    # base[u1][u2] = (sign, unit) for unit*unit multiplication, ignoring the
    # signed prefix of each operand (folded in separately in `mul`).
    _base = {
        ("1", "1"): (1, "1"), ("1", "i"): (1, "i"), ("1", "j"): (1, "j"), ("1", "k"): (1, "k"),
        ("i", "1"): (1, "i"), ("i", "i"): (-1, "1"), ("i", "j"): (1, "k"), ("i", "k"): (-1, "j"),
        ("j", "1"): (1, "j"), ("j", "i"): (-1, "k"), ("j", "j"): (-1, "1"), ("j", "k"): (1, "i"),
        ("k", "1"): (1, "k"), ("k", "i"): (1, "j"), ("k", "j"): (-1, "i"), ("k", "k"): (-1, "1"),
    }

    def __init__(self):
        self.order = 8
        self.elements = list(range(8))
        self.identity = 0  # "1"

    def _decompose(self, g):
        label = self._labels[g]
        if label.startswith("-"):
            return -1, label[1:]
        return 1, label

    def _compose(self, sign, unit):
        idx = self._units.index(unit)
        label = unit if sign > 0 else f"-{unit}"
        return self._labels.index(label)

    def mul(self, a, b):
        s1, u1 = self._decompose(a)
        s2, u2 = self._decompose(b)
        base_sign, unit = self._base[(u1, u2)]
        return self._compose(s1 * s2 * base_sign, unit)

    def inv(self, a):
        # every element has order <=4; -1 is self-inverse and central,
        # 1 is its own inverse, and for x in {i,j,k,-i,-j,-k}, x*x = -1 or 1
        # so inv(x) = x if x^2==1 else -x when x^2==-1.
        if self.mul(a, a) == self.identity:
            return a
        return self._compose(-1, self._decompose(a)[1]) if self._decompose(a)[0] == 1 else \
            self._labels.index(self._decompose(a)[1])

    def table(self):
        return np.array([[self.mul(a, b) for b in range(8)] for a in range(8)])

    def irreps(self):
        irreps = []

        def make_linear(sign_i, sign_j):
            sign_k = sign_i * sign_j

            def rho(g, self=self):
                # {-1,1} is the commutator subgroup of Q8, so every linear
                # character is constant on {x,-x} pairs: it depends only on
                # the unit (1/i/j/k), never on the +/- prefix.
                _, u = self._decompose(g)
                signs = {"1": 1, "i": sign_i, "j": sign_j, "k": sign_k}
                return np.array([[signs[u]]], dtype=complex)
            return rho

        irreps.append(("trivial", 1, make_linear(1, 1)))
        irreps.append(("chi_i", 1, make_linear(1, -1)))
        irreps.append(("chi_j", 1, make_linear(-1, 1)))
        irreps.append(("chi_k", 1, make_linear(-1, -1)))

        base = {
            "1": np.eye(2, dtype=complex),
            "i": np.array([[1j, 0], [0, -1j]]),
            "j": np.array([[0, 1], [-1, 0]]),
            "k": np.array([[0, 1j], [1j, 0]]),
        }

        def rho_2d(g, self=self):
            s, u = self._decompose(g)
            return s * base[u]
        irreps.append(("rho_2d", 2, rho_2d))
        return irreps


def group_task_pairs(group):
    """All (a, b, a*b) triples — the full supervised dataset for a group task."""
    g = group.order
    a_idx, b_idx = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    a_idx = a_idx.reshape(-1)
    b_idx = b_idx.reshape(-1)
    labels = np.array([group.mul(int(a), int(b)) for a, b in zip(a_idx, b_idx)])
    return a_idx, b_idx, labels
