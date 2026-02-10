import numpy as np

from wzk.alg import tsp


def test_solve_tsp_small_default():
    x0 = np.empty((0, 2), dtype=float)
    x1 = np.array([[0.0, 0.0]], dtype=float)
    x2 = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)

    assert np.array_equal(tsp.solve_tsp(x=x0, time_limit=1, verbose=0), np.arange(0))
    assert np.array_equal(tsp.solve_tsp(x=x1, time_limit=1, verbose=0), np.arange(1))
    assert np.array_equal(tsp.solve_tsp(x=x2, time_limit=1, verbose=0), np.arange(2))


def test_order_q_with_tsp_small_identity():
    q = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    q_ord, route = tsp.order_q_with_tsp(q=q, time_limit_sec=1)

    assert q_ord.dtype == np.float32
    assert np.array_equal(route, np.array([0, 1], dtype=int))
    assert np.allclose(q_ord, q.astype(np.float32))


def test_order_q_with_tsp_no_anchor_matches_direct_tsp():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(20, 3))

    q_ord, route = tsp.order_q_with_tsp(q=q, anchor_q=None, time_limit_sec=1)
    route_direct = np.asarray(tsp.solve_tsp(x=q.astype(np.float32), time_limit=1, verbose=0), dtype=int)

    assert route.shape == (q.shape[0],)
    assert np.array_equal(np.sort(route), np.arange(q.shape[0], dtype=int))
    assert np.array_equal(route, route_direct)
    assert np.allclose(q_ord, q.astype(np.float32)[route])


def test_order_q_with_tsp_with_anchor_matches_manual():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(25, 2))
    anchor_q = np.array([10.0, -3.0])

    q_ord, route = tsp.order_q_with_tsp(q=q, anchor_q=anchor_q, time_limit_sec=1)

    route_all = np.asarray(
        tsp.solve_tsp(
            x=np.concatenate([anchor_q.reshape(1, -1).astype(np.float32), q.astype(np.float32)], axis=0),
            time_limit=1,
            verbose=0,
        ),
        dtype=int,
    )
    route_expected = route_all[route_all != 0] - 1

    assert route.shape == (q.shape[0],)
    assert np.array_equal(np.sort(route), np.arange(q.shape[0], dtype=int))
    assert np.array_equal(route, route_expected)
    assert np.allclose(q_ord, q.astype(np.float32)[route])


def test_order_q_with_tsp_alias():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(10, 2))
    anchor_q = np.array([0.1, 0.2], dtype=float)

    q_ord_a, route_a = tsp.order_q_with_tsp(q=q, anchor_q=anchor_q, time_limit_sec=1)
    q_ord_b, route_b = tsp._order_q_with_tsp(q=q, anchor_q=anchor_q, time_limit_sec=1)

    assert np.array_equal(route_a, route_b)
    assert np.allclose(q_ord_a, q_ord_b)
