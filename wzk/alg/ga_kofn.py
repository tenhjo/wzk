from __future__ import annotations

from collections.abc import Callable

import numpy as np

from wzk.logger import setup_logger
from wzk.math2.math2 import random_subset

logger = setup_logger(__name__)


def rank(x: np.ndarray) -> np.ndarray:
    temp = x.argsort()
    res = np.empty_like(temp)
    res[temp] = np.arange(len(x))
    return res


def rank_2d(x: np.ndarray) -> np.ndarray:
    temp = x.argsort(axis=1)
    res = np.empty_like(temp)
    res[np.arange(x.shape[0])[:, np.newaxis], temp] = np.arange(x.shape[1])[np.newaxis, :]
    return res


def kofn(n: int, k: int, fitness_fun: Callable[[np.ndarray], np.ndarray],
         pop_size: int = 200, pop0: np.ndarray | None = None, n_gen: int = 500,
         n_keep_best: int | None = None, tourney_size: int | None = None,
         mut_prob: float = 0.01, mut_frac: float | None = None,
         mutate_best: int = 0,
         log_level: int = 0) -> tuple[np.ndarray, np.ndarray]:

    tourney_size = int(max(np.ceil(pop_size / 10), 2)) if tourney_size is None else tourney_size
    n_keep_best = int(np.floor(pop_size / 10)) if n_keep_best is None else n_keep_best
    mut_prob = mut_prob if mut_frac is None else 1 - (1 - mut_frac) ^ (1 / k)

    hall_of_fame = np.empty((n_gen + 1, k), dtype=int)
    hall_of_all = np.empty((n_gen + 1, pop_size, k), dtype=int)
    fitness_best = np.empty(n_gen + 1)
    fitness_avg = np.empty(n_gen + 1)

    if pop0 is None:
        pop = np.array([np.random.choice(n, size=k, replace=False) for _ in range(pop_size)])
    else:
        pop = pop0

    fitness = fitness_fun(pop)

    sort_idx = np.argsort(fitness)
    fitness = fitness[sort_idx]
    fitness_old = fitness.copy()
    pop = pop[sort_idx]

    hall_of_fame[0] = pop[0].copy()
    hall_of_all[0] = pop.copy()
    fitness_best[0] = fitness[0].copy()
    fitness_avg[0] = np.mean(fitness).copy()

    for g in range(n_gen):
        if log_level > 2:
            logger.debug("Generation %d/%d | best: %s | avg: %s", g, n_gen, fitness_best[g], fitness_avg[g])
        parents1, parents2 = parents_tournament(fitness=fitness, tourney_size=tourney_size)

        offspring = create_offspring(pop=pop, parents1=parents1, parents2=parents2)

        offspring = mutate(pop=offspring, n=n, mut_prob=mut_prob)

        if mutate_best > 0:
            best_mutants = mutate(pop=np.repeat(pop[:n_keep_best], mutate_best, axis=0), n=n, mut_prob=mut_prob)
            offspring = np.concatenate([offspring, best_mutants])

        offspring = np.sort(offspring, axis=1)
        offspring = np.unique(offspring, axis=0)
        if offspring.shape[0] < pop_size - n_keep_best:
            offspring = np.concatenate([offspring,
                                        random_subset(n=n, k=k, m=pop_size - n_keep_best - offspring.shape[0])],
                                       axis=0)

        fitness = fitness_fun(offspring)

        pop, fitness = keep_best(pop=offspring, pop_old=pop, fitness=fitness, fitness_old=fitness_old,
                                 n_keep_best=n_keep_best)

        fitness_old = fitness.copy()
        hall_of_fame[g+1] = pop[0].copy()
        hall_of_all[g+1] = pop.copy()
        fitness_best[g+1] = fitness[0].copy()
        fitness_avg[g+1] = np.mean(fitness).copy()

    if log_level > 0:
        from wzk import new_fig
        fig, ax = new_fig()
        ax.plot(fitness_avg, c="r", label="average")
        ax.plot(fitness_best, c="b", label="best")
        ax.set_xlabel("Generations")
        ax.set_ylabel("Fitness")
        ax.legend()

    return hall_of_fame[np.argmin(fitness_best)], hall_of_all


def parents_tournament(fitness: np.ndarray, tourney_size: int) -> tuple[np.ndarray, np.ndarray]:
    def tournament_rank(t: np.ndarray, f: np.ndarray) -> np.ndarray:
        ts = t.shape[-1]
        return (ts - rank_2d(x=f[t])) / (ts * (ts + 1) / 2)

    pop_size = len(fitness)

    tourneys1 = random_subset(n=pop_size, k=tourney_size, m=pop_size)
    tourneys2 = random_subset(n=pop_size, k=tourney_size, m=pop_size)

    rank1 = tournament_rank(t=tourneys1, f=fitness)
    rank2 = tournament_rank(t=tourneys2, f=fitness)
    parents1 = np.array([np.random.choice(a=tourneys1[p], p=rank1[p]) for p in range(pop_size)])
    parents2 = np.array([np.random.choice(a=tourneys2[p], p=rank2[p]) for p in range(pop_size)])
    return parents1, parents2


def create_offspring(pop: np.ndarray, parents1: np.ndarray, parents2: np.ndarray) -> np.ndarray:
    pop_size, k = pop.shape
    parents12 = np.concatenate([pop[parents1], pop[parents2]], axis=-1)
    offspring = np.array([np.random.choice(np.unique(parents12), k, replace=False) for _ in range(pop_size)])
    return offspring


def mutate(pop: np.ndarray, n: int, mut_prob: float) -> np.ndarray:
    pop_size, k = pop.shape

    chosen = np.random.random(size=(pop_size, k)) < mut_prob
    chosen = np.array(chosen)

    for i in range(pop_size):
        sum_chosen = np.sum(chosen[i])
        mut = np.random.choice(n, sum_chosen+k, replace=False)
        mut = np.setdiff1d(mut, pop[i])
        pop[i, chosen[i]] = np.random.choice(mut, sum_chosen, replace=False)

    return pop


def keep_best(*, pop: np.ndarray, pop_old: np.ndarray,
              fitness: np.ndarray, fitness_old: np.ndarray,
              n_keep_best: int = 0) -> tuple[np.ndarray, np.ndarray]:

    pop_size, k = pop_old.shape
    if n_keep_best != 0:
        sort_idx = np.argsort(fitness)

        pop = np.concatenate([pop_old[:n_keep_best], pop[sort_idx][:pop_size-n_keep_best]])
        fitness = np.concatenate([fitness_old[:n_keep_best], fitness[sort_idx][:pop_size-n_keep_best]])

    sort_idx = np.argsort(fitness)
    fitness = fitness[sort_idx]
    pop = pop[sort_idx]

    return pop, fitness


def test_kofn() -> None:
    def dummy_fitness(i: np.ndarray) -> np.ndarray:
        return np.sum(i, axis=-1)

    best, last_gen = kofn(n=10000, k=10, pop_size=100, fitness_fun=dummy_fitness, n_gen=1000, log_level=3,
                          mut_prob=0.1, mutate_best=2, n_keep_best=50)
    logger.debug(np.sort(best))
