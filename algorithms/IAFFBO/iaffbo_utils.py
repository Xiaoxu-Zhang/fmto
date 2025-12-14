import math
import numpy as np
import torch

from enum import Enum, auto
from pyfmto.framework import ClientPackage as Pkg
from torch import nn as nn, optim as optim


class Actions(Enum):
    PUSH_INIT = auto()
    PULL_UPDATE = auto()
    PUSH_UPDATE = auto()


class AggData:
    def __init__(self, version: int, src_num: int, agg_res: dict):
        self.version = version
        self.src_num = src_num
        self.agg_res = agg_res


class ServerPackage:
    def __init__(
            self,
            desc: str,
            data: dict = None,
    ):
        self.desc = desc
        self.data = data


class ClientPackage(Pkg):
    def __init__(
            self,
            cid: int,
            action: Enum,
            version: int = None,
            data: dict = None,
    ):
        super().__init__(cid, action)
        self.version = version
        self.data = data


def flatten_params(weights: list[np.ndarray], biases: list[np.ndarray]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for w in weights:
        parts.append(w.reshape(-1))
    for b in biases:
        parts.append(b.reshape(-1))
    if len(parts) == 0:
        return np.array([], dtype=float)
    return np.concatenate(parts, axis=0)


def flatten_params_matlab_order(weights: list[np.ndarray], biases: list[np.ndarray]) -> np.ndarray:
    """
    Flatten in MATLAB-like order: IW first (W0), then successive LWs (W1..), then biases (b0..).
    Expect each weight matrix shape = (out, in) as in MATLAB IW/LW.
    Bias vectors shape = (out, 1) preferred (or (out,)).
    """
    if not weights:
        return np.array([], dtype=float)

    all_params = weights + biases
    return np.concatenate([p.reshape(-1) for p in all_params], axis=0)


def average_params(params_list: list[dict]) -> dict:
    """
    Element-wise average for MLP-like parameters.

    params: {
        'weights': [w0, w1, ...],
        'biases': [b0, b1, ...]
    }
    """
    if not params_list:
        return {}
    n_layers = len(params_list[0]['weights'])
    if n_layers == 0:
        return {}
    avg_w = []
    avg_b = []
    for li in range(n_layers):
        mats = [p['weights'][li] for p in params_list]
        avg_w.append(np.mean(np.stack(mats, axis=0), axis=0))
    for li in range(len(params_list[0]['biases'])):
        vecs = [p['biases'][li] for p in params_list]
        avg_b.append(np.mean(np.stack(vecs, axis=0), axis=0))
    return {'weights': avg_w, 'biases': avg_b}


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
