from __future__ import annotations

from collections.abc import Callable

import numpy as np

# from pyOpt.pyNLPQLP.pyNLPQLP import NLPQLP
from pyOpt import Optimization
from pyOpt.pyCOBYLA.pyCOBYLA import COBYLA
from pyOpt.pySLSQP.pySLSQP import SLSQP
from scipy import optimize

from wzk.logger import setup_logger
from wzk.opt import random

logger = setup_logger(__name__)

# from scipy.optimize import least_squares, slsqp


# bayes_opt and pyswarms seem so infinitely slow compared to SLSQP, not sure when they are better suited

default_options = {"maxiter": 1000,
                   "disp": True,  # True
                   "ftol": 1e-12,
                   "sens_step": 1e-7,
                   "sens_type": "fd",  # "fd" or "cs"
                   "pll_type": None}  # "POA" or None  | I measured no speed difference)


class SolverPar:

    __slots__ = ["name", "options"]

    def __init__(self, name: str = "PyOpt-SLSQP") -> None:

        self.name = name
        self.options = default_options.copy()


def print_result(res: object, log_level: int) -> None:
    if log_level > 1:
        logger.debug("------ Result ------")
        logger.debug(res)


def fw_objective_pyopt(fun: Callable[[np.ndarray], float]) -> Callable:

    def fun2(_x: np.ndarray) -> tuple[float, list, int]:
        f = fun(_x)
        g = []
        fail = 0
        return f, g, fail

    return fun2


def create_opt_problem(fun: Callable[[np.ndarray], float],
                       x0: np.ndarray,
                       lower: float = -1., upper: float = +1.) -> Optimization:
    opt_prob = Optimization("", fw_objective_pyopt(fun))
    for i, x0_i in enumerate(x0):
        opt_prob.addVar(f"x{i+1}", "c", lower=lower, upper=upper, value=x0_i)
    opt_prob.addObj("f")
    return opt_prob


def minimize_cobyla(fun: Callable[[np.ndarray], float],
                    x0: np.ndarray, options: dict,
                    log_level: int = 0) -> np.ndarray:
    cobyla = COBYLA(pll_type=options["pll_type"])
    cobyla.setOption("IPRINT", 3)
    cobyla.setOption("MAXFUN", 10000)

    opt_prob = create_opt_problem(fun=fun, x0=x0)

    cobyla(opt_prob)

    res = opt_prob.solution(0)
    print_result(res=res, log_level=log_level)

    vs = res.getVarSet()
    x = np.array([vs[key].value for key in range(len(x0))])
    return x


def minimize_slsqp(fun: Callable[[np.ndarray], float],
                   x0: np.ndarray, options: dict,
                   log_level: int = 0) -> np.ndarray:
    slsqp = SLSQP(pll_type=options["pll_type"])
    slsqp.setOption("IPRINT", 0 if log_level > 5 else -1)
    slsqp.setOption("MAXIT", options["maxiter"])
    try:
        slsqp.setOption("ACC", options["ftol"])
    except KeyError:
        pass

    opt_prob = create_opt_problem(fun=fun, x0=x0)

    slsqp(opt_prob,
          sens_type=options["sens_type"],
          sens_step=options["sens_step"])

    res = opt_prob.solution(0)
    print_result(res=res, log_level=log_level)

    vs = res.getVarSet()
    x = np.array([vs[key].value for key in range(len(x0))])
    return x


# --- Bayes Opt ---
def ih_get_keys(n: int) -> list[str]:
    return [f"x{i:0>3}" for i in range(n)]


def fw_objective_bayes(fun: Callable[[np.ndarray], float]) -> Callable:

    def fun2(**kwargs: float) -> float:
        x = np.array([kwargs[k] for k in ih_get_keys(len(kwargs))])
        return -fun(x)

    return fun2


def minimize_bayes_opt(fun: Callable[[np.ndarray], float],
                       x0: np.ndarray, options: dict,
                       log_level: int = 0) -> np.ndarray:
    from bayes_opt import BayesianOptimization
    low = -0.1
    high = +0.1
    n = len(x0)
    bounds = {k: (low, high) for i, k in enumerate(ih_get_keys(n))}

    optimizer = BayesianOptimization(
        f=fw_objective_bayes(fun),
        pbounds=bounds,
    )

    optimizer.maximize(
        init_points=2000,
        n_iter=20,
    )

    x = optimizer.max["params"]
    x = np.array([x[k] for k in ih_get_keys(n)])
    return x


# --- Swarms ---
def fw_objective_swarms(fun: Callable[[np.ndarray], float]) -> Callable:

    def fun2(x: np.ndarray) -> np.ndarray:
        return np.array([fun(xx) for xx in x])

    return fun2


def minimize_swarms(fun: Callable[[np.ndarray], float],
                    x0: np.ndarray, options: dict,
                    log_level: int = 0) -> np.ndarray:
    import pyswarms

    n = len(x0)
    low = -0.1
    high = +0.1
    bounds = ( np.full(n, low), np.full(n, high))
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    optimizer = pyswarms.single.GlobalBestPSO(n_particles=500, dimensions=n, options=options, bounds=bounds)
    o, x = optimizer.optimize(fw_objective_swarms(fun), iters=1000)

    return x


def minimize(fun: Callable[[np.ndarray], float],
             x0: np.ndarray,
             solver: str = "PyOpt-SLSQP",
             options: dict | None = None,
             log_level: int = 0) -> np.ndarray:
    if options is None:
        options = default_options.copy()

    if "PyOpt" in solver:
        if solver == "PyOpt-SLSQP":
            x = minimize_slsqp(fun=fun, x0=x0, options=options, log_level=log_level - 1)
        elif solver == "PyOpt-COBYLA":
            x = minimize_cobyla(fun=fun, x0=x0, options=options, log_level=log_level - 1)
        else:
            raise ValueError(f"Unknown optimizer: {solver}")

    elif "SciPy" in solver:
        if solver == "SciPy-LS":
            x = optimize.least_squares(fun=fun, x0=x0, method="lm").x

        else:
            solver = solver.split("-")[1]
            solver = solver.lower()
            x = optimize.minimize(fun=fun, x0=x0, method=solver, tol=1e-13,
                                  options={"disp": True, "maxiter": 1000}).x

    elif "wzk" in solver:
        if solver == "wzk-randomball":
            n_outer = 200
            n_inner = 50
            x, o = random.random_ball_search(fun=fun, x0=x0, n_outer=n_outer, n_inner=n_inner)
        else:
            raise ValueError(f"Unknown optimizer: {solver}")

    elif "Bayes" in solver:
        x = minimize_bayes_opt(fun=fun, x0=x0, options=options)

    elif "Swarms" in solver:
        x = minimize_swarms(fun=fun, x0=x0, options=options)

    else:
        raise ValueError(f"Unknown optimizer: {solver}")

    return x
