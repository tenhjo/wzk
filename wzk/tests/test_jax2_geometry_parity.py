from itertools import combinations

import numpy as np

from wzk import geometry
from wzk import jax2


def test_rotation_between_vectors():
    rng = np.random.default_rng(0)
    a = rng.normal(size=3)
    b = rng.normal(size=3)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    r0 = geometry.rotation_between_vectors(a, b)
    r1 = jax2.rotation_between_vectors(a, b)
    assert np.allclose(np.asarray(r0), np.asarray(r1), atol=1e-6, rtol=1e-6)


def test_get_orthonormal():
    a = np.array([0.3, -0.7, 1.2])
    b0 = geometry.get_orthonormal(a)
    b1 = jax2.get_orthonormal(a)
    assert np.allclose(np.asarray(b0), np.asarray(b1), atol=1e-8, rtol=1e-8)


def test_line_line_and_pairs():
    rng = np.random.default_rng(1)
    line_a = rng.normal(size=(2, 3))
    line_b = rng.normal(size=(2, 3))

    xa0, xb0 = geometry.line_line(line_a, line_b)
    xa1, xb1 = jax2.line_line(line_a, line_b)
    assert np.allclose(np.asarray(xa0), np.asarray(xa1), atol=1e-6, rtol=1e-6)
    assert np.allclose(np.asarray(xb0), np.asarray(xb1), atol=1e-6, rtol=1e-6)

    m, n = 7, 6
    lines = rng.normal(size=(m, n, 2, 3))
    pairs = np.array(list(combinations(np.arange(n), 2)))

    xa0, xb0 = geometry.line_line_pairs(lines=lines, pairs=pairs)
    xa1, xb1 = jax2.line_line_pairs(lines=lines, pairs=pairs)
    assert np.allclose(np.asarray(xa0), np.asarray(xa1), atol=1e-6, rtol=1e-6)
    assert np.allclose(np.asarray(xb0), np.asarray(xb1), atol=1e-6, rtol=1e-6)


def test_capsule_capsule():
    rng = np.random.default_rng(2)
    for _ in range(200):
        capsule_a, capsule_b = rng.normal(size=(2, 2, 3))
        radius_a, radius_b = rng.uniform(0.0, 0.4, size=2)

        _, _, d0 = geometry.capsule_capsule(line_a=capsule_a, radius_a=radius_a,
                                            line_b=capsule_b, radius_b=radius_b)
        _, _, d1 = jax2.capsule_capsule(line_a=capsule_a, radius_a=radius_a,
                                        line_b=capsule_b, radius_b=radius_b)
        assert np.allclose(np.asarray(d0), np.asarray(d1), atol=1e-6, rtol=1e-6)


def test_angle_and_distance_helpers():
    rng = np.random.default_rng(3)
    p = rng.normal(size=3)
    x0 = rng.normal(size=3)
    x1 = rng.normal(size=3)

    d0 = geometry.distance_point_line(p=p, x0=x0, x1=x1, clip=True)
    d1 = jax2.distance_point_line(p=p, x0=x0, x1=x1, clip=True)
    assert np.allclose(np.asarray(d0), np.asarray(d1), atol=1e-6, rtol=1e-6)

    a = rng.normal(size=(5, 3))
    b = rng.normal(size=(5, 3))
    ang0 = geometry.angle_between_vectors(a, b)
    ang1 = jax2.angle_between_vectors(a, b)
    assert np.allclose(np.asarray(ang0), np.asarray(ang1), atol=1e-6, rtol=1e-6)
