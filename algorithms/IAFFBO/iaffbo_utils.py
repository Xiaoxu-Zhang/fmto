import numpy as np
from enum import Enum, auto
from pyfmto.framework import ClientPackage as Pkg


class Actions(Enum):
    PUSH_INIT = auto()
    PULL_INIT = auto()
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
    parts: list[np.ndarray] = []
    # IW
    parts.append(weights[0].reshape(-1))
    # LWs
    for w in weights[1:]:
        parts.append(w.reshape(-1))
    # biases in layer order
    for b in biases:
        parts.append(b.reshape(-1))
    return np.concatenate(parts, axis=0)


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

