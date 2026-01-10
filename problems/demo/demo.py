import numpy as np
from pyfmto.problem import SingleTaskProblem, MultiTaskProblem
from typing import Union

from .. import benchmarks

class StpA(SingleTaskProblem):

    def __init__(self, dim: int, **kwargs):
        super().__init__(dim=dim, obj=1, lb=-1, ub=1, **kwargs)

    def _eval_single(self, x: np.ndarray):
        return np.cos(sum(x ** 2))


class StpB(SingleTaskProblem):

    def __init__(self, dim: int, **kwargs):
        super().__init__(dim=dim, obj=1, lb=-2, ub=2, **kwargs)

    def _eval_single(self, x: np.ndarray):
        return np.sin(sum(x))


class Demo(MultiTaskProblem):
    """
    dim: 2
    """
    def __init__(self, dim: int = 2, **kwargs):
        super().__init__(dim, **kwargs)

    def _init_tasks(self, dim, **kwargs) -> Union[list[SingleTaskProblem], tuple[SingleTaskProblem, ...]]:
        # Task IDs are assigned automatically from 1 to len(tasks).
        # If a step (e.g., StpB) is unavailable when dim > 10, and you don't
        # want to renumber subsequent tasks, set the IDs manually as shown:
        #
        #     if dim > 10:
        #         tasks = [StpA(dim), benchmarks.Rastrigin(dim), benchmarks.Ackley(dim)]
        #         ids = [1, 3, 4]
        #         for i, task in zip(ids, tasks):
        #             task.id = i
        #     else:
        #         return StpA(dim), StpB(dim), benchmarks.Rastrigin(dim), benchmarks.Ackley(dim)

        return StpA(dim), StpB(dim), benchmarks.Rastrigin(dim), benchmarks.Ackley(dim)

