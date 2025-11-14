import numpy as np
from collections import defaultdict
from typing import Callable

from sklearn.cluster import KMeans

from pyfmto.framework import Server, ClientPackage, ServerPackage, DataArchive
from pyfmto.utilities import logger

from .iaffbo_utils import Actions, AggData, flatten_params_matlab_order, average_params


class IaffboServer(Server):
    """
    cl_num: 6
    """
    def __init__(self, **kwargs):
        super().__init__()
        kwargs = self.update_kwargs(kwargs)
        self.cl_num: int = int(kwargs.get('cl_num', 6))
        self.client_bounds = []
        self.clients_data: dict[int, DataArchive] = defaultdict(DataArchive)

    def handle_request(self, client_data: ClientPackage) -> ServerPackage:
        action_map: dict[Actions, Callable[[ClientPackage], ServerPackage]] = {
            Actions.PUSH_INIT: self._push_init,
            Actions.PULL_INIT: self._pull_init,
            Actions.PUSH_UPDATE: self._push_update,
            Actions.PULL_UPDATE: self._pull_update,
        }
        action = action_map.get(client_data.action)
        if action:
            return action(client_data)
        logger.error(f"Unknown action: {client_data.action}")
        return ServerPackage('status', {'status': 'error'})

    def _push_init(self, client_data: ClientPackage) -> ServerPackage:
        bound = client_data.data.get('bound') if isinstance(client_data.data, dict) else None
        if bound is not None:
            self.client_bounds.append(bound)
        cl_num = client_data.data.get('cl_num') if isinstance(client_data.data, dict) else None
        if cl_num is not None:
            self.cl_num = int(cl_num)
        return ServerPackage('status', {'status': 'ok'})

    def _pull_init(self, client_data: ClientPackage) -> ServerPackage:
        return ServerPackage('Init', {'status': 'ok'})

    def _push_update(self, client_data: ClientPackage) -> ServerPackage:
        cid = client_data.cid
        data = client_data.data
        self.clients_data[cid].add_src(data)
        return ServerPackage('SaveStatus', {'status': 'ok'})

    def _pull_update(self, client_data: ClientPackage) -> ServerPackage:
        lts = self.clients_data[client_data.cid].get_latest_res()
        return ServerPackage('LatestUpdate', data=lts)

    def aggregate(self, client_id=None):
        if self.num_clients == 0:
            return
        ids = self.sorted_ids
        vers = np.asarray([self.clients_data[cid].num_src for cid in ids])
        res_vers = np.asarray([self.clients_data[cid].num_res for cid in ids])
        target_ver = int(np.min(res_vers))
        ready = np.all(vers > target_ver)
        if not ready:
            return
        vectors = []
        params_by_cid: dict[int, dict] = {}
        for cid in ids:
            src = self.clients_data[cid].src_data[target_ver]
            vec = src.get('vector')
            if vec is None or (hasattr(vec, 'size') and vec.size == 0):
                weights = [np.array(w) for w in src.get('weights', [])]
                biases = [np.array(b) for b in src.get('biases', [])]
                vec = flatten_params_matlab_order(weights, biases)
            vec = np.array(vec).reshape(-1)
            if vec.size == 0:
                vec = np.zeros(4, dtype=float)
            vectors.append(vec)
            params_by_cid[cid] = {
                'weights': [np.array(w) for w in src.get('weights', [])],
                'biases': [np.array(b) for b in src.get('biases', [])],
            }
        max_len = max(v.shape[0] for v in vectors)
        V = np.stack([np.pad(v, (0, max_len - v.shape[0])) for v in vectors], axis=0)
        k = min(self.cl_num, V.shape[0])
        try:
            km = KMeans(n_clusters=k, n_init=1, random_state=None)
            labels = km.fit_predict(V)
        except Exception:
            labels = np.zeros(V.shape[0], dtype=int)
            k = 1
        for cluster_id in range(k):
            members = [i for i, lab in enumerate(labels) if lab == cluster_id]
            if not members:
                continue
            cids = [ids[i] for i in members]
            plist = [params_by_cid[cid] for cid in cids]
            agg = average_params(plist) if plist and plist[0]['weights'] else {}
            for cid in cids:
                ver = self.clients_data[cid].num_res + 1
                self.clients_data[cid].add_res(AggData(version=ver, src_num=self.num_clients, agg_res=agg))
                logger.debug(f"Aggregated IAFFBO params for client {cid} ver {ver}")
