import numpy as np

from src.simplicial_complex import SimplicialComplex, boundary2_rows, build_complex


def test_enumerate_triangles_on_k4_finds_all_four():
    # K4: every pair of {0,1,2,3} is an edge -> C(4,3) = 4 triangles.
    from src.simplicial_complex import _enumerate_triangles

    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    triangles = _enumerate_triangles(4, edges)
    assert sorted(triangles) == [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def test_enumerate_triangles_on_a_cycle_finds_none():
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    from src.simplicial_complex import _enumerate_triangles

    assert _enumerate_triangles(4, edges) == []


def test_enumerate_triangles_no_duplicates_or_missing_on_larger_graph():
    # Two triangles sharing an edge: {0,1,2} and {1,2,3}.
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    from src.simplicial_complex import _enumerate_triangles

    triangles = _enumerate_triangles(4, edges)
    assert sorted(triangles) == [(0, 1, 2), (1, 2, 3)]


def test_build_complex_on_three_close_points_forms_one_triangle():
    points = np.array([[0.1, 0.1], [0.12, 0.1], [0.11, 0.12]])
    complex_ = build_complex(points, r=0.05)
    assert complex_.num_edges == 3
    assert complex_.num_triangles == 1


def test_build_complex_no_edges_when_radius_too_small():
    points = np.array([[0.1, 0.1], [0.9, 0.9]])
    complex_ = build_complex(points, r=0.01)
    assert complex_.num_edges == 0
    assert complex_.num_triangles == 0


def test_boundary2_rows_shape_matches_edge_count():
    complex_ = SimplicialComplex(
        n_vertices=3, edges=[(0, 1), (0, 2), (1, 2)], triangles=[(0, 1, 2)]
    )
    rows = boundary2_rows(complex_)
    assert len(rows) == 1
    # The single triangle's boundary touches all 3 edges: bits 0,1,2 set -> 0b111
    assert rows[0] == 0b111
