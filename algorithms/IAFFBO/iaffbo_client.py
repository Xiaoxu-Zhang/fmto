import math
import numpy as np
from typing import Optional

# PyTorch-based classifier (patternnet-like)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None

from pydacefit.corr import corr_gauss
from pydacefit.dace import DACE
from pydacefit.regr import regr_constant

from pyfmto.framework import Client, ClientPackage, ServerPackage, record_runtime
from pyfmto.utilities import logger

from .iaffbo_utils import Actions


# NOTE: RNG 将在客户端实例化为 self.rng 以便可复现实验


class TorchMLPClassifier:
    def __init__(self, input_dim: int, hidden: tuple[int, int, int], num_classes: int = 3, init_mode: str = 'nguyen_widrow'):
        if torch is None:
            raise ImportError("PyTorch is required for IAFFBO classifier. Please install torch.")
        h1, h2, h3 = hidden
        self.model = nn.Sequential(
            nn.Linear(input_dim, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
            nn.Linear(h2, h3), nn.Tanh(),
            nn.Linear(h3, num_classes)
        )
        # Prefer GPU → MPS → CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        try:
            from pyfmto.utilities import logger as _logger
            _logger.info(f"IAFFBO Torch device: {self.device}")
        except Exception:
            pass
        # Initialization
        if init_mode == 'nguyen_widrow':
            self._init_nguyen_widrow()
        else:
            # Xavier init for tanh
            gain = nn.init.calculate_gain('tanh')
            with torch.no_grad():
                for layer in self.model:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=gain)
                        nn.init.zeros_(layer.bias)

    def _init_nguyen_widrow(self):
        # Nguyen–Widrow initialization for tanh
        with torch.no_grad():
            prev_size = None
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    in_f, out_f = layer.in_features, layer.out_features
                    dev = layer.weight.device
                    # random in [-0.5, 0.5]
                    w = torch.rand(out_f, in_f, device=dev) - 0.5
                    # normalize rows
                    w = w / (w.norm(dim=1, keepdim=True) + 1e-12)
                    beta = 0.7 * (out_f) ** (1.0 / in_f)
                    w = w * beta
                    b = (torch.rand(out_f, device=dev) - 0.5) * 2 * beta
                    layer.weight[:] = w
                    layer.bias[:] = b
        
    def _flatten(self) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
        params = []
        shapes = []
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                w = layer.weight
                b = layer.bias
                params.append(w.view(-1))
                params.append(b.view(-1))
                shapes.append(tuple(w.shape))
                shapes.append(tuple(b.shape))
        if params:
            return torch.cat(params), shapes
        return torch.tensor([], device=self.device), shapes

    def _assign(self, vec: torch.Tensor, shapes: list[tuple[int, ...]]):
        idx = 0
        with torch.no_grad():
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    w_shape = shapes[idx]; idx += 1
                    b_shape = shapes[idx]; idx += 1
                    w_num = int(np.prod(w_shape))
                    b_num = int(np.prod(b_shape))
                    w = vec[:w_num].view(w_shape)
                    b = vec[w_num:w_num + b_num].view(b_shape)
                    vec = vec[w_num + b_num:]
                    layer.weight[:] = w
                    layer.bias[:] = b

    def _loss_and_grad(self, X_t: torch.Tensor, y_t: torch.Tensor, crit: nn.Module) -> tuple[float, torch.Tensor]:
        # forward + backward to get loss and flat grad
        self.model.zero_grad(set_to_none=True)
        logits = self.model(X_t)
        loss = crit(logits, y_t)
        loss.backward()
        # flatten grad in same order
        grads = []
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                grads.append(layer.weight.grad.view(-1))
                grads.append(layer.bias.grad.view(-1))
        g = torch.cat(grads) if grads else torch.tensor([], device=self.device)
        return float(loss.detach().cpu().item()), g.detach()

    def fit(self, X: np.ndarray, y: np.ndarray, max_epochs: int = 300, lr: float = 1e-3, method: str = 'lbfgs', scg_sigma0: float = 1e-4, scg_lambda_init: float = 1e-6):
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        # map labels {-1,0,+1} -> {0,1,2}
        y_np = y.reshape(-1)
        y_cls = np.where(y_np < 0, 0, np.where(y_np > 0, 2, 1)).astype(np.int64)
        y_t = torch.from_numpy(y_cls).to(self.device)
        crit = nn.CrossEntropyLoss().to(self.device)
        self.model.train()
        if method == 'scg':
            # SCG training
            w, shapes = self._flatten()
            if w.numel() == 0:
                return
            w = w.detach().clone().to(self.device)
            # initial f,g
            f, g = self._loss_and_grad(X_t, y_t, crit)
            g = g.to(self.device)
            p = -g
            success = True
            sigma0 = scg_sigma0
            lamb = scg_lambda_init
            lamb_bar = 1.0e20
            for _ in range(max_epochs):
                if success:
                    mu = (p * g).sum().item()
                    if mu >= 0:
                        p = -g
                        mu = (p * g).sum().item()
                    kappa = (p * p).sum().item()
                    sigma = sigma0 / (math.sqrt(kappa) + 1e-12)
                    # directional second derivative approx
                    w_try = w + sigma * p
                    self._assign(w_try, shapes)
                    _, g1 = self._loss_and_grad(X_t, y_t, crit)
                    s = (g1 - g) / sigma
                    delta = (p * s).sum().item()
                    if delta <= 0:
                        s = s + ( (-delta) / (kappa + 1e-12) + lamb) * p
                        delta = (p * s).sum().item()
                # compute step
                alpha = - mu / (delta + lamb + 1e-12)
                w_new = w + alpha * p
                # evaluate new point
                self._assign(w_new, shapes)
                f_new, g_new = self._loss_and_grad(X_t, y_t, crit)
                g_new = g_new.to(self.device)
                Delta = 2.0 * (f_new - f) / (alpha * mu + 1e-12)
                if Delta >= 0:
                    # accept
                    w = w_new
                    f_old = f
                    f = f_new
                    g_old = g
                    g = g_new
                    success = True
                else:
                    # reject
                    success = False
                # adjust lambda
                if Delta < 0.25:
                    lamb = min(lamb_bar, lamb + (delta * (1.0 - Delta)) / (kappa + 1e-12))
                elif Delta > 0.75:
                    lamb = max(1.0e-15, lamb * 0.5)
                # update direction
                if success:
                    gg = (g * g).sum().item()
                    gg_old = (g_old * g_old).sum().item() if 'g_old' in locals() else gg
                    mu = (p * g).sum().item()
                    beta_pr = max(0.0, (gg - (g * g_old).sum().item()) / (mu + 1e-12)) if 'g_old' in locals() else 0.0
                    p = -g + beta_pr * p
                    # convergence checks
                    if gg < 1e-8 or abs(f_old - f) < 1e-12:
                        break
            # assign final weights
            self._assign(w, shapes)
        elif method == 'lbfgs':
            optz = optim.LBFGS(self.model.parameters(), lr=lr, max_iter=20, history_size=100, line_search_fn='strong_wolfe')
            def closure():
                optz.zero_grad()
                logits = self.model(X_t)
                loss = crit(logits, y_t)
                loss.backward()
                return loss
            for _ in range(max_epochs):
                optz.step(closure)
        else:
            optz = optim.Adam(self.model.parameters(), lr=lr)
            for _ in range(max_epochs):
                optz.zero_grad()
                logits = self.model(X_t)
                loss = crit(logits, y_t)
                loss.backward()
                optz.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
            logits = self.model(X_t)
            preds = logits.argmax(dim=1).cpu().numpy()
        # map {0,1,2} -> {-1,0,+1}
        out = np.empty_like(preds)
        out[preds == 0] = -1
        out[preds == 1] = 0
        out[preds == 2] = 1
        return out

    def get_params(self) -> dict:
        weights = []
        biases = []
        with torch.no_grad():
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    # MATLAB orientation: (out, in)
                    w = layer.weight.detach().cpu().numpy()  # (out, in)
                    b = layer.bias.detach().cpu().numpy().reshape(-1, 1)  # (out,1)
                    weights.append(w)
                    biases.append(b)
        return {'weights': weights, 'biases': biases}

    def set_params(self, params: dict):
        wi = 0
        with torch.no_grad():
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    w = params['weights'][wi]
                    b = params['biases'][wi]
                    # assign in MATLAB orientation directly
                    layer.weight[:] = torch.from_numpy(w).to(layer.weight.device)
                    layer.bias[:] = torch.from_numpy(b.reshape(-1)).to(layer.bias.device)
                    wi += 1


class IaffboClient(Client):
    """
    ucb_flag: LCB     # one of: EI, UCB, LCB
    candidates: 100   # number of random candidate points for AF
    pop_size: 100     # population size for winner-loser search
    wmax: 100         # iterations of winner-loser update (align to MATLAB)
    phi: 0.1          # velocity scaling for loser update
    cl_num: 6         # number of clusters on server
    transfer: true    # whether to use aggregated classifier for search when available
    sample_pairs: 0     # 0 or <=0 means full pairs; else sample up to this number per source
    hidden_scale: 1.5 # scale for first hidden layer size (1.5*input_dim)
    train_frac: 0.75  # fraction of pairs used for training (rest for validation)
    """

    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        kwargs = self.update_kwargs(kwargs)
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
        # per-client RNG for reproducibility
        self.rng = np.random.default_rng(self.seed)

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
        self.prev_ver = 0

    # ---------- main loop ----------
    def optimize(self):
        if not self.solutions.initialized:
            self._sync_init()
        else:
            self._step()

    def _sync_init(self):
        pkg = ClientPackage(cid=self.id, action=Actions.PUSH_INIT, data={'dim': self.dim, 'bound': (self.lb, self.ub), 'cl_num': self.cl_num})
        self.request_server(pkg)
        pkg = ClientPackage(cid=self.id, action=Actions.PULL_INIT)
        self.request_server(pkg)

    @record_runtime('Step')
    def _step(self):
        # 1) fit GP on current data (normalize y to [0,1])
        X = self.solutions.x
        Y = self.solutions.y
        if X.size == 0:
            return
        # deduplicate rows
        try:
            _, idx = np.unique(X, axis=0, return_index=True)
            idx = np.sort(idx)  # keep first-seen order (stable)
            X_fit = X[idx]
            Y_fit = Y[idx]
        except Exception:
            X_fit, Y_fit = X, Y
        y_norm, _, _ = self._normalize_y(Y_fit)
        self.gp.fit(X_fit, y_norm)

        # 2) candidates and AF values
        cand = self._random_in_bounds(self.candidates)
        mean, var = self.gp.predict(cand, return_mse=True)
        acq = self._af(mean.reshape(-1, 1), var.reshape(-1, 1), y_norm)

        # 3) build pairwise dataset (history + candidate AF)
        XX, yy = self._build_pairs(X_fit, y_norm.reshape(-1, 1), cand, acq)
        # if matlab_two, drop 0-class pairs
        if self.label_mode == 'matlab_two':
            mask = yy != 0
            XX, yy = XX[mask], yy[mask]

        # 3.5) try non-blocking pull aggregated params for initialization (previous round)
        pre_agg = self._try_pull_agg_for_init()

        # 4) train classifier (PyTorch) with init params（若无聚合则使用本地随机初始化）
        self._train_classifier(XX, yy, init_params=pre_agg)

        # 5) pull aggregated params（本轮强同步）
        agg_params = self._pull_aggregated_params()

        # 6) pick classifier for search (50% aggregated vs local if available)
        clf_for_search = self._clf
        if self.transfer and agg_params is not None and self.rng.random() < 0.5:
            try:
                clf_for_search = self._build_clf_from_params(agg_params)
            except Exception as e:
                logger.debug(f"Client {self.id} build clf from agg params failed: {e}")
        # Strong sync: ensure all clients have aggregated before proceed (barrier already via _pull_aggregated_params)

        # 7) evolutionary search guided by classifier
        x_next = self._winner_loser_search(clf_for_search)
        y_next = self.problem.evaluate(x_next)
        self.solutions.append(x_next.reshape(1, -1), y_next.reshape(1, -1))
        self.record_round_info('FE', str(self.solutions.size))
        self.record_round_info('Best', f"{self.solutions.y_min:.2f}")

    # ---------- components ----------
    def _random_in_bounds(self, n):
        lb = self.lb if isinstance(self.lb, np.ndarray) else np.ones(self.dim) * self.lb
        ub = self.ub if isinstance(self.ub, np.ndarray) else np.ones(self.dim) * self.ub
        return self.rng.random((n, self.dim)) * (ub - lb) + lb

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

    def _build_pairs(self, X, Y, cand, acq):
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
        return XX[shuf], yy[shuf]

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

    def _train_classifier(self, XX: np.ndarray, yy: np.ndarray, init_params: Optional[dict] = None):
        if XX.size == 0:
            self._clf = None
            return
        d = XX.shape[1]  # input dim equals size(TrainIn,2) in MATLAB
        h1 = int(max(2, math.ceil(self.hidden_scale * d)))
        h2 = int(max(2, d))
        h3 = int(max(2, d // 2))
        num_classes = 2 if self.label_mode == 'matlab_two' else 3
        clf = TorchMLPClassifier(input_dim=d, hidden=(h1, h2, h3), num_classes=num_classes, init_mode=self.nn_init)
        if init_params is not None and init_params.get('weights'):
            try:
                clf.set_params({'weights': [np.array(w) for w in init_params['weights']],
                                'biases': [np.array(b) for b in init_params['biases']]})
            except Exception:
                pass
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
        self._clf = clf
        # training finished; force push to let server count this round's src before pull
        self._push_local_classifier()

    @staticmethod
    def _normalize_y(Y: np.ndarray):
        y = Y.flatten()
        ymin = float(np.min(y))
        ymax = float(np.max(y))
        den = (ymax - ymin)
        if den <= 0:
            yn = np.zeros_like(y)
        else:
            yn = (y - ymin) / den
        return yn.reshape(-1, 1), ymin, ymax

    def _push_local_classifier(self):
        params = self._clf.get_params() if self._clf is not None else {'weights': [], 'biases': []}
        if params['weights']:
            from .iaffbo_utils import flatten_params_matlab_order
            vec = flatten_params_matlab_order(params['weights'], params['biases'])
            data = {'vector': vec, 'weights': params['weights'], 'biases': params['biases'], 'size': (2 if self.label_mode=='matlab_two' else 3)}
        else:
            data = {'vector': np.array([]), 'weights': [], 'biases': [], 'size': 0}
        pkg = ClientPackage(cid=self.id, action=Actions.PUSH_UPDATE, data=data)
        self.request_server(package=pkg, msg='Push local classifier')

    def _try_pull_agg_for_init(self) -> Optional[dict]:
        # Non-blocking try: if already has an aggregated model (from previous round), return it
        from .iaffbo_utils import AggData
        pkg = ClientPackage(cid=self.id, action=Actions.PULL_UPDATE, data=None)
        res: ServerPackage = self.request_server(package=pkg, repeat=1, interval=0.0, msg='Try pull agg init')
        if res is not None and isinstance(res.data, AggData):
            # 不更新 prev_ver，仅用于本轮初始化
            return res.data.agg_res
        return None

    def _pull_aggregated_params(self) -> Optional[dict]:
        # Strong sync: block until new aggregated version is available
        from .iaffbo_utils import AggData
        pkg = ClientPackage(cid=self.id, action=Actions.PULL_UPDATE, data=None)
        while True:
            res: ServerPackage = self.request_server(package=pkg, repeat=100, interval=0.5, msg='Pull aggregated classifier')
            if res is None:
                continue
            if isinstance(res.data, AggData):
                if res.data.version > self.prev_ver:
                    self.prev_ver = res.data.version
                    return res.data.agg_res
            # no aggregated params yet; wait and retry
            self.record_round_info('Sync', 'waiting agg')

    def _build_clf_from_params(self, params: dict):
        weights = [np.array(w) for w in params.get('weights', [])]
        biases = [np.array(b) for b in params.get('biases', [])]
        if not weights or not biases:
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
    def _winner_loser_search(self, clf) -> np.ndarray:
        lb = self.lb if isinstance(self.lb, np.ndarray) else np.ones(self.dim) * self.lb
        ub = self.ub if isinstance(self.ub, np.ndarray) else np.ones(self.dim) * self.ub
        popsize = max(4, self.pop_size)
        if clf is None:
            return self._random_in_bounds(1).reshape(-1)
        Decs = self._random_in_bounds(popsize)
        # ensure even population size
        if Decs.shape[0] % 2 == 1:
            Decs = np.vstack([Decs, self._random_in_bounds(1)])
        LoserVel = np.zeros_like(Decs[:Decs.shape[0]//2])
        for _ in range(self.wmax):
            rank = self.rng.permutation(popsize)
            loser = rank[:popsize // 2]
            winner = rank[popsize // 2:]
            XX_pairs = np.hstack([Decs[loser], Decs[winner]])
            XX_pairs_n = self._normalize_pairs(XX_pairs)
            try:
                pred = clf.predict(XX_pairs_n)
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
