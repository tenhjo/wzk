# Travelling salesman problem  https://developers.google.com/optimization/routing/tsp
import numpy as np
from scipy.spatial import distance_matrix
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from wzk.logger import setup_logger

logger = setup_logger(__name__)


def get_route(manager, routing, assignment):
    index = routing.Start(0)
    route = []

    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = assignment.Value(routing.NextVar(index))

    return np.array(route, dtype=int)


def _solve_tsp(x, dist_mat=None, time_limit=10):
    """
    Get the index list for the optimal route for all points, starting at the first
    :param x:
    :param dist_mat: optional
    :param time_limit: seconds
    :return:
    """
    x = np.asarray(x)
    assert x.ndim == 2, f"x must be 2D (n, d), got {x.shape}"

    n = len(x)
    if n <= 2:
        return np.arange(n, dtype=int)

    if dist_mat is None:
        dist_mat = distance_matrix(x, x)

    if not (dist_mat.dtype == np.int64 or dist_mat.dtype == np.int32 or dist_mat.dtype == np.int16):
        min_dist = dist_mat[dist_mat != 0].min()
        if min_dist < 1:
            dist_mat /= min_dist
    dist_mat = np.round(dist_mat).astype(np.int32)

    # Create the routing index manager. num_cities, num_vehicles, depot
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_mat[from_node, to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define the cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    # Additional search parameters
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = time_limit

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters=search_parameters)
    route = get_route(manager=manager, routing=routing, assignment=assignment)

    cost = dist_mat[route, np.roll(route, - 1)].sum()
    logger.info("TSP cost for %s points after %ss: %s", x.shape, time_limit, cost)

    return route


def _extend_distmat(dist_mat, x, x_new):
    n = len(x)
    dist_mat = np.asarray(dist_mat)
    assert dist_mat.shape == (n, n), f"dist_mat must have shape {(n, n)}, got {dist_mat.shape}"

    d_home = distance_matrix(x_new, x)[0]
    dist_mat2 = np.empty((n + 1, n + 1), dtype=dist_mat.dtype)
    dist_mat2[0, 0] = 0
    dist_mat2[0, 1:] = d_home
    dist_mat2[1:, 0] = d_home
    dist_mat2[1:, 1:] = dist_mat
    return dist_mat2


def solve_tsp(x: np.ndarray, dist_mat=None, x_home=None, time_limit=10):
    """
    Solve TSP on x and return a route over x indices.
    If x_home is provided, anchor the route by prepending x_home to the optimization problem
    and remove that anchor index from the returned route.
    """
    x = np.asarray(x)
    assert x.ndim == 2, f"x must be 2D (n, d), got {x.shape}"
    n = len(x)
    if n <= 2:
        return np.arange(n, dtype=int)

    if x_home is None:
        return _solve_tsp(x=x, time_limit=time_limit, dist_mat=dist_mat)

    x_home = np.asarray(x_home).reshape((1, -1))
    assert x_home.shape[1] == x.shape[1]

    x2 = np.concatenate([x_home, x], axis=0)
    if dist_mat is not None:
        dist_mat = _extend_distmat(x=x, x_new=x_home, dist_mat=dist_mat)

    route = _solve_tsp(x=x2, dist_mat=dist_mat, time_limit=time_limit)

    route = route[route != 0] - 1
    return route


def order_q_with_tsp(*,
                     q: np.ndarray,
                     anchor_q: np.ndarray | None = None,
                     time_limit_sec: int = 3) -> tuple[np.ndarray, np.ndarray]:
    q_np = np.asarray(q, dtype=np.float32)
    assert q_np.ndim == 2, f"q must be 2D (n, d), got {q_np.shape}"

    n_q = q_np.shape[0]
    if n_q <= 2:
        route = np.arange(n_q, dtype=int)
        return q_np, route

    route_np = np.asarray(
        solve_tsp(
            x=q_np,
            x_home=anchor_q,
            time_limit=time_limit_sec,
        ),
        dtype=int,
    )
    return q_np[route_np], route_np


def _order_q_with_tsp(*,
                      q: np.ndarray,
                      anchor_q: np.ndarray | None = None,
                      time_limit_sec: int = 3) -> tuple[np.ndarray, np.ndarray]:
    return order_q_with_tsp(q=q, anchor_q=anchor_q, time_limit_sec=time_limit_sec)
