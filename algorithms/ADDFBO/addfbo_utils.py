import numpy as np
from enum import Enum, auto
from pyDOE import lhs
from pyfmto.framework import ClientPackage as CPkg
from smt.surrogate_models import KRG, KPLS, GPX
from typing import Union

T_Bound = Union[int, float, list, tuple, np.ndarray]


__all__ = [
    "Actions",
    "SmtModel",
    "ClientPackage",
    "DifferentialEvolution",
]


def init_samples(x_lb, x_ub, dim: int, size: int):
    return lhs(dim, size) * (x_ub - x_lb) + x_lb


class ClientPackage(CPkg):
    def __init__(
            self,
            cid: int,
            action: Enum,
            version: int = None,
            dim: int = None,
            d_share_y: np.ndarray = None,
            d_share_y_std: np.ndarray = None,
    ):
        super().__init__(cid=cid, action=action)
        self.version = version
        self.d_share_y = d_share_y
        self.d_share_y_std = d_share_y_std
        self.dim = dim


class Actions(Enum):
    PULL_AUX = auto()
    PUSH_UPDATE = auto()
    PULL_UPDATE = auto()


class SmtModel:
    def __init__(self, model_name='GPX', dim=None):

        if model_name == 'KRG':
            self.model = KRG(theta0=[1e-2], print_global=False)
        elif model_name == 'KPLS':
            self.model = KPLS(eval_n_comp=True, print_global=False)
        elif model_name == 'GPX':
            if dim is None:
                self.model = GPX(theta0=[1e-2], print_global=False)
            else:
                self.model = GPX(theta0=[1e-2], print_global=False, kpls_dim=dim)
        else:
            raise ValueError('Model name must be KRG or KPLS or GPX')
        self.is_fitted = False
        self.data_size = 0

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.model.set_training_values(x, y)
        self.model.train()
        self.data_size = x.shape[0]
        self.is_fitted = True

    def predict(self, x, return_std=False):
        mean = self.model.predict_values(x)
        if return_std:
            mse = self.model.predict_variances(x)
            std = np.sqrt(mse)
            return mean, std
        return mean


class DifferentialEvolution:
    """
    Reference:
    Li, Jian-Yu, et al. “A Meta-Knowledge Transfer-Based
    Differential Evolution for Multitask Optimization.”
    (B) IEEE TEVC, Aug. 2022, pp. 719–34,
    https://doi.org/10.1109/tevc.2021.3131236.
    """
    def __init__(self,x_lb: T_Bound, x_ub: T_Bound, dim:int, max_gen=20, F=0.4, CR=0.9, num_elite=5, init_pop_size=200):
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.dim = dim
        if isinstance(x_lb, (int, float)):
            self.x_bound = (self.x_lb, self.x_ub)
        elif isinstance(x_lb, (list, tuple, np.ndarray)):
            self.x_bound = (self.x_lb[0], self.x_ub[0])
        else:
            raise ValueError(f"Type of x_lb is {type(x_lb)}, but it should be int or float or iterable")
        self.max_gen = max_gen
        self.F = F
        self.CR = CR
        self.num_elite = num_elite
        self.init_pop_size = init_pop_size
        self.obj_func = None

    def minimize(self, obj_func, archive=None):
        assert callable(obj_func)
        self.obj_func = obj_func

        parents = init_samples(self.x_lb, self.x_ub, dim=self.dim, size=self.init_pop_size)
        if archive is not None:
            parents = np.vstack((parents, archive))

        parents, predictions = self._selection(parents)
        for _ in range(self.max_gen):
            v = self._mutation(parents, self.num_elite, self.F)
            u = self._crossover(v, parents, self.CR)
            candidates = np.vstack((v, u))
            parents, predictions = self._selection(candidates)

        return parents[0], predictions[0]

    def _mutation(self, parents, num_elite, F):
        """
            该实现是对参考文献所提的两种变异方式的折中（既有精英解的指导，又有随机解的多样）
            当 elite_rate=1 时，该算法退化为 DE/rand/1
            当 elite_rate=0 时，该算法退化为 DE/best/1
        """
        row, col = parents.shape

        c1 = np.random.randint(0, num_elite, row)
        c2 = np.random.randint(0, row, row)
        c3 = np.random.randint(0, row, row)

        r1 = parents[c1]
        r2 = parents[c2]
        r3 = parents[c3]

        res = r1 + F * (r2 - r3)
        res = np.clip(res, self.x_lb, self.x_ub)
        return res

    def _crossover(self, V, X, CR):
        shape = V.shape
        rand_mat = np.random.uniform(size=shape)
        res = np.zeros(shape)

        ind_leq_cr = rand_mat <= CR
        ind_greater_cr = rand_mat > CR
        ind_j_rand = self._create_j_rand_mat(shape)

        res[ind_leq_cr] = V[ind_leq_cr]
        res[ind_greater_cr] = X[ind_greater_cr]
        res[ind_j_rand] = V[ind_j_rand]

        return res

    def _selection(self, candidates):
        y = self.obj_func(candidates)
        y = y.flatten()
        sorted_index = np.argsort(y)

        sorted_candidates = candidates[sorted_index]
        sorted_y = y[sorted_index]

        return sorted_candidates[:self.init_pop_size], sorted_y[:self.init_pop_size]

    @staticmethod
    def _create_j_rand_mat(shape):
        j_rand_mat = np.zeros(shape=shape, dtype=bool)
        col_indices = np.random.randint(0, shape[1], shape[0])
        row_indices = np.arange(shape[0])
        j_rand_mat[row_indices, col_indices] = True
        return j_rand_mat
