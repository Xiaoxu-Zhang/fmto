import math
import pprint
import traceback

import numpy as np
from typing import Optional, Any

from pydacefit.corr import corr_gauss
from pydacefit.dace import DACE
from pydacefit.regr import regr_constant

from pyfmto.framework import Client, record_runtime
from pyfmto.utilities import logger

from .iaffbo_utils import Actions, ClientPackage, TorchMLPClassifier


class IaffboClient(Client):
    """
    ucb_flag: LCB     # one of: EI, UCB, LCB
    candidates: 100   # number of random candidate points for AF
    pop_size: 100     # population size for winner-loser search
    wmax: 100         # iterations of winner-loser update (align to MATLAB)
    phi: 0.1          # velocity scaling for loser update
    cl_num: 6         # number of clusters on server
    transfer: true    # whether to use aggregated classifier for search when available
    sample_pairs: 0   # 0 or <=0 means full pairs; else sample up to this number per source
    hidden_scale: 1.5 # scale for first hidden layer size (1.5*input_dim)
    train_frac: 0.75  # fraction of pairs used for training (rest for validation)
    seed: 123         # seed for client RNG
    """

    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        self.ucb_flag: str = kwargs.get('ucb_flag', 'LCB').upper()
        self.candidates: int = int(kwargs.get('candidates', 100))
        self.pop_size: int = int(kwargs.get('pop_size', 100))
        self.wmax: int = int(kwargs.get('wmax', 100))
        self.phi: float = float(kwargs.get('phi', 0.1))
        self.cl_num: int = int(kwargs.get('cl_num', 6))
        self.transfer: bool = bool(kwargs.get('transfer', True))
        self.sample_pairs: int = int(kwargs.get('sample_pairs', 0))
        self.hidden_scale: float = float(kwargs.get('hidden_scale', 1.5))
        self.train_frac: float = float(kwargs.get('train_frac', 0.75))
        self.seed: Optional[int] = kwargs.get('seed', None)
        self.rng = np.random.default_rng(self.seed)
        self.problem.auto_update_solutions = True

        # surrogate
        theta0 = 5.0 * np.ones(self.dim)
        self.gp = DACE(regr=regr_constant, corr=corr_gauss,
                       theta=theta0, thetaL=1e-5 * np.ones(self.dim), thetaU=100 * np.ones(self.dim))

        # classifier settings
        self.label_mode: str = kwargs.get('label_mode', 'three')  # 'three' or 'matlab_two'
        self.nn_init: str = kwargs.get('nn_init', 'nguyen_widrow')  # 'nguyen_widrow' or 'xavier'
        self.scg_sigma0: float = float(kwargs.get('scg_sigma0', 1e-4))
        self.scg_lambda_init: float = float(kwargs.get('scg_lambda_init', 1e-6))
        self._clf: Optional[TorchMLPClassifier] = None

        # sync
        self.version = 1
        self.agg_params: dict[str, Any] = {}
        self.clf_for_search: Optional[TorchMLPClassifier] = None

    def optimize(self):
        if not self.solutions.initialized:
            self._sync_init()
        else:
            self._step()

    def _sync_init(self):
        pkg = ClientPackage(
            cid=self.id,
            action=Actions.PUSH_INIT,
            data={'cl_num': self.cl_num}
        )
        self.request_server(pkg)

    @record_runtime('Step')
    def _step(self):
        self._train_classifier()
        self._push_clf_params()
        self._pull_agg_params()
        self._build_classifier()
        x_next = self._winner_loser_search()
        self.problem.evaluate(x_next)
        self.version += 1

    def _build_classifier(self):
        if self.transfer and self.rng.random() < 0.5:
            self.clf_for_search = self._build_clf_from_params(self.agg_params)
            logger.debug(f"{self.name} Built clf_for_search with agg params")
        else:
            self.clf_for_search = self._clf
            logger.debug(f"{self.name} Set clf_for_search by local clf")

    def fit_gp(self):
        X = self.solutions.x
        Y = self.solutions.y
        # remove deduplicate rows
        try:
            _, idx = np.unique(X, axis=0, return_index=True)
            idx = np.sort(idx)  # keep first-seen order (stable)
            X_fit = X[idx]
            Y_fit = Y[idx]
        except Exception:
            X_fit, Y_fit = X, Y
        self.gp.fit(X_fit, self.problem.normalize_y(Y_fit))
        logger.debug(f"{self.name} GP fit: {X_fit.shape[0]} points")
        return X_fit, Y_fit

    def init_candidates(self, y_norm: np.ndarray):
        cand = self.problem.random_uniform_x(self.candidates)
        mean, var = self.gp.predict(cand, return_mse=True)
        return cand, self._af(mean.reshape(-1, 1), var.reshape(-1, 1), y_norm)

    def _af(self, mean: np.ndarray, var: np.ndarray, Y_hist_norm: np.ndarray):
        eps = 1e-12
        std = np.sqrt(np.abs(var))
        if self.ucb_flag == 'UCB':
            return (mean + 2.0 * std).reshape(-1, 1)
        elif self.ucb_flag == 'EI':
            fmin = np.min(Y_hist_norm)
            setv = (fmin - mean)
            z = setv / (std + eps)
            from scipy.stats import norm
            E = setv * norm.cdf(z) + std * norm.pdf(z)
            return -E.reshape(-1, 1)
        else:
            return (mean - 2.0 * std).reshape(-1, 1)

    def _build_pairwise(self):
        X, Y = self.fit_gp()
        y_norm = self.problem.normalize_y(Y)
        cand, acq = self.init_candidates(y_norm)

        ya = Y.flatten()
        Xa = X
        Xb = cand
        yb = acq.flatten()
        XA, yA = self._pairs_from(Xa, ya, max_pairs=self.sample_pairs)
        XB, yB = self._pairs_from(Xb, yb, max_pairs=self.sample_pairs)
        if XA.size == 0 and XB.size == 0:
            return np.empty((0, 2 * self.dim)), np.empty((0,))
        XX = np.vstack([z for z in [XA, XB] if z.size > 0])
        yy = np.concatenate([z for z in [yA, yB] if z.size > 0], axis=0)
        XX = self._normalize_pairs(XX)
        shuf = self.rng.permutation(len(yy))
        x_res, y_res = XX[shuf], yy[shuf]
        if self.label_mode == 'matlab_two':
            mask = y_res != 0
            res = x_res[mask], y_res[mask]
        else:
            res = x_res, y_res
        logger.debug(f"{self.name} Pairwise built: {res[0].shape[0]} pairs")
        return res

    def _pairs_from(self, X: np.ndarray, v: np.ndarray, max_pairs: int):
        n = X.shape[0]
        if n < 2:
            return np.empty((0, 2 * self.dim)), np.empty((0,))
        total = n * (n - 1) // 2
        if max_pairs is None or max_pairs <= 0 or max_pairs >= total:
            # full pairs i<j
            idx_i = []
            idx_j = []
            for i in range(n - 1):
                m = n - i - 1
                idx_i.extend([i] * m)
                idx_j.extend(list(range(i + 1, n)))
            idx_i = np.array(idx_i)
            idx_j = np.array(idx_j)
        else:
            m = max_pairs
            idx_i = self.rng.integers(0, n, size=m)
            idx_j = self.rng.integers(0, n, size=m)
            mask = idx_i != idx_j
            idx_i, idx_j = idx_i[mask], idx_j[mask]
            if idx_i.size == 0:
                return np.empty((0, 2 * self.dim)), np.empty((0,))
        Xi = X[idx_i]
        Xj = X[idx_j]
        vi = v[idx_i].reshape(-1)
        vj = v[idx_j].reshape(-1)
        diff = vi - vj
        # MATLAB: Output(i)-Output(j)>0 -> +1 (second better); <0 -> -1 (first better)
        tol = 1e-12
        y = np.where(diff < -tol, -1, np.where(diff > tol, 1, 0))
        XX = np.hstack([Xi, Xj])
        return XX, y

    def _normalize_pairs(self, XX: np.ndarray):
        lb = self.lb if isinstance(self.lb, np.ndarray) else np.ones(self.dim) * self.lb
        ub = self.ub if isinstance(self.ub, np.ndarray) else np.ones(self.dim) * self.ub
        den = (ub - lb)
        den[den == 0.0] = 1.0
        Xi = XX[:, :self.dim]
        Xj = XX[:, self.dim:]
        Xi = (Xi - lb) / den
        Xj = (Xj - lb) / den
        return np.hstack([Xi, Xj])

    def _train_classifier(self):
        XX, yy = self._build_pairwise()
        if XX.size == 0:
            self._clf = None
            return
        d = XX.shape[1]  # input dim equals size(TrainIn,2) in MATLAB
        h1 = int(max(2, math.ceil(self.hidden_scale * d)))
        h2 = int(max(2, d))
        h3 = int(max(2, d // 2))
        num_classes = 2 if self.label_mode == 'matlab_two' else 3
        clf = TorchMLPClassifier(input_dim=d, hidden=(h1, h2, h3), num_classes=num_classes, init_mode=self.nn_init)
        if self.agg_params is not None and self.agg_params.get('weights'):
            try:
                clf.set_params({'weights': self.agg_params['weights'],
                                'biases': self.agg_params['biases']})
            except Exception:
                logger.error(f"{self.name} set params failed: \n{traceback.print_exc()}")
        # stratified split 3/4 for train per class (ceil as in MATLAB)
        labels = yy.reshape(-1)
        uniq = np.unique(labels)
        train_idx = []
        val_idx = []
        for c in uniq:
            idx_c = np.where(labels == c)[0]
            if idx_c.size == 0:
                continue
            k_c = int(max(1, math.ceil(self.train_frac * idx_c.size)))
            sh = self.rng.permutation(idx_c.size)
            idx_c_sh = idx_c[sh]
            train_idx.append(idx_c_sh[:k_c])
            val_idx.append(idx_c_sh[k_c:])
        if train_idx:
            tr = np.concatenate(train_idx)
        else:
            m = XX.shape[0]
            k = int(max(1, math.ceil(self.train_frac * m)))
            idx = self.rng.permutation(m)
            tr = idx[:k]
        clf.fit(XX[tr], yy[tr], max_epochs=300, lr=1e-2, method='scg', scg_sigma0=self.scg_sigma0, scg_lambda_init=self.scg_lambda_init)
        # p_err on validation
        if val_idx:
            va = np.concatenate(val_idx) if len(val_idx) > 1 else val_idx[0]
            if np.size(va) > 0:
                pred = clf.predict(XX[va])
                # if matlab_two, ensure labels are binary {-1,+1}
                y_true = yy[va]
                if self.label_mode == 'matlab_two':
                    y_true = np.where(y_true < 0, -1, 1)
                p_err = float(np.mean(pred != y_true))
                self.record_round_info('p_err', f"{p_err:.3f}")
        logger.debug(f"{self.name} Classifier built with agg_params")
        self._clf = clf

    @staticmethod
    def str_dict(data: Optional[dict]):
        return pprint.pformat(data, indent=2, width=40, compact=False)

    def _push_clf_params(self):
        params = self._clf.get_params() if self._clf is not None else None
        if params is not None:
            from .iaffbo_utils import flatten_params_matlab_order
            data = {
                'vector': flatten_params_matlab_order(params['weights'], params['biases']),
                'size': 2 if self.label_mode == 'matlab_two' else 3
            }
            data.update(params)
        else:
            data = {}
        pkg = ClientPackage(cid=self.id, action=Actions.PUSH_UPDATE, version=self.version, data=data)
        self.request_server(package=pkg, msg='Push local classifier')
        logger.debug(f"Client {self.id} push classifier: {repr(list(data.keys()))}")

    def _pull_init_params(self) -> Optional[dict]:
        # Non-blocking try: if already has an aggregated model (from previous round), return it
        pkg = ClientPackage(cid=self.id, action=Actions.PULL_UPDATE, version=self.version, data=None)
        return self.request_server(package=pkg, repeat=1, interval=0.0, msg='Try pull agg init')

    def _pull_agg_params(self) -> Optional[dict]:
        # Strong sync: block until new aggregated version is available
        pkg = ClientPackage(cid=self.id, action=Actions.PULL_UPDATE, version=self.version, data=None)
        self.agg_params = self.request_server(package=pkg, repeat=100, interval=1., msg='Pull aggregated classifier')

    def check_pkg(self, pkg) -> bool:
        return pkg is not None

    def _build_clf_from_params(self, params: dict):
        if params is None:
            logger.warning(f"{self.name} Classifier params is None")
            return self._clf
        weights = params.get('weights', [])
        biases = params.get('biases', [])
        if not weights or not biases:
            logger.warning(f"{self.name} Classifier params value is empty")
            return self._clf
        # MATLAB orientation: weight[0] (out1,in) ⇒ input_dim = in
        d_in = weights[0].shape[1]
        # hidden sizes are subsequent layer outputs
        hidden_outs = [w.shape[0] for w in weights[:-1]]
        if len(hidden_outs) == 1:
            h_tuple = (hidden_outs[0], max(2, hidden_outs[0]//1), max(2, hidden_outs[0]//2))
        elif len(hidden_outs) == 2:
            h_tuple = (hidden_outs[0], hidden_outs[1], max(2, hidden_outs[1]//2))
        else:
            h_tuple = (hidden_outs[0], hidden_outs[1], hidden_outs[2])
        num_classes = weights[-1].shape[0]
        clf = TorchMLPClassifier(input_dim=d_in, hidden=h_tuple, num_classes=num_classes, init_mode=self.nn_init)
        clf.set_params({'weights': weights, 'biases': biases})
        return clf

    @record_runtime('Search')
    def _winner_loser_search(self) -> np.ndarray:
        lb = self.lb if isinstance(self.lb, np.ndarray) else np.ones(self.dim) * self.lb
        ub = self.ub if isinstance(self.ub, np.ndarray) else np.ones(self.dim) * self.ub
        popsize = max(4, self.pop_size)
        if self.clf_for_search is None:
            logger.warning(f"{self.name} clf for search is None, random uniform a new x")
            return self.problem.random_uniform_x(1).reshape(-1)
        Decs = self.problem.random_uniform_x(popsize)
        # ensure even population size
        if Decs.shape[0] % 2 == 1:
            Decs = np.vstack([Decs, self.problem.random_uniform_x(1)])
        LoserVel = np.zeros_like(Decs[:Decs.shape[0]//2])
        for _ in range(self.wmax):
            rank = self.rng.permutation(popsize)
            loser = rank[:popsize // 2]
            winner = rank[popsize // 2:]
            XX_pairs = np.hstack([Decs[loser], Decs[winner]])
            XX_pairs_n = self._normalize_pairs(XX_pairs)
            try:
                pred = self.clf_for_search.predict(XX_pairs_n)
            except Exception:
                pred = np.ones(loser.shape[0], dtype=int)
            repl = np.where(pred == -1)[0]
            tmp = loser[repl].copy()
            loser[repl] = winner[repl]
            winner[repl] = tmp
            LoserDec = Decs[loser]
            WinnerDec = Decs[winner]
            if LoserVel.shape[0] != LoserDec.shape[0]:
                LoserVel = np.zeros_like(LoserDec)
            R1 = self.rng.random((LoserDec.shape[0], 1))
            R2 = self.rng.random((LoserDec.shape[0], 1))
            R3 = self.rng.random((LoserDec.shape[0], 1))
            LoserVel = R1 * LoserVel + self.phi * R2 * (WinnerDec - LoserDec) + R3 * (np.mean(Decs, axis=0) - LoserDec)
            LoserDec = LoserDec + LoserVel
            LoserDec = np.minimum(np.maximum(LoserDec, lb), ub)
            Decs = np.vstack([WinnerDec, LoserDec])
        ridx = int(self.rng.integers(0, Decs.shape[0]))
        return Decs[ridx]
