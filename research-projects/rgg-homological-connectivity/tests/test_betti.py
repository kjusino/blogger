from src.betti import betti_numbers
from src.simplicial_complex import SimplicialComplex


def test_single_isolated_vertex():
    complex_ = SimplicialComplex(n_vertices=1, edges=[], triangles=[])
    b0, b1 = betti_numbers(complex_)
    assert (b0, b1) == (1, 0)


def test_n_isolated_vertices_no_edges():
    complex_ = SimplicialComplex(n_vertices=5, edges=[], triangles=[])
    b0, b1 = betti_numbers(complex_)
    assert (b0, b1) == (5, 0)


def test_hollow_triangle_is_one_component_one_cycle():
    # 3 vertices, 3 edges, the triangle NOT filled in -> a hollow loop.
    complex_ = SimplicialComplex(n_vertices=3, edges=[(0, 1), (0, 2), (1, 2)], triangles=[])
    b0, b1 = betti_numbers(complex_)
    assert (b0, b1) == (1, 1)


def test_filled_triangle_is_contractible():
    # Same graph, but the 2-simplex is filled in -> contractible disk, no cycle.
    complex_ = SimplicialComplex(n_vertices=3, edges=[(0, 1), (0, 2), (1, 2)], triangles=[(0, 1, 2)])
    b0, b1 = betti_numbers(complex_)
    assert (b0, b1) == (1, 0)


def test_four_cycle_has_one_independent_loop():
    # C4: 0-1-2-3-0, no diagonals, no triangles -> one independent 1-cycle.
    complex_ = SimplicialComplex(n_vertices=4, edges=[(0, 1), (1, 2), (2, 3), (0, 3)], triangles=[])
    b0, b1 = betti_numbers(complex_)
    assert (b0, b1) == (1, 1)


def test_two_disjoint_hollow_triangles():
    complex_ = SimplicialComplex(
        n_vertices=6,
        edges=[(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)],
        triangles=[],
    )
    b0, b1 = betti_numbers(complex_)
    assert (b0, b1) == (2, 2)


def test_k4_fully_filled_is_topologically_a_sphere():
    # K4 with all 4 triangular faces filled is exactly the boundary of a
    # tetrahedron, i.e. homeomorphic to S^2: b0=1, b1=0 (H_1 of a sphere is
    # trivial), b2=1 (not computed here, but Euler char check: V-E+F =
    # 4-6+4 = 2 = b0-b1+b2 = 1-0+1, consistent).
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    complex_ = SimplicialComplex(n_vertices=4, edges=edges, triangles=triangles)
    b0, b1 = betti_numbers(complex_)
    assert b0 == 1
    assert b1 == 0


def test_k4_graph_without_any_filled_triangles_has_three_independent_cycles():
    # Euler characteristic check: V - E = 4 - 6 = -2 = b0 - b1 => b1 = 1 - (-2) = 3.
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    complex_ = SimplicialComplex(n_vertices=4, edges=edges, triangles=[])
    b0, b1 = betti_numbers(complex_)
    assert (b0, b1) == (1, 3)
