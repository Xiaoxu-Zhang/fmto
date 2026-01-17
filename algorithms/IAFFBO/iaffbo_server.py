import numpy as np
from typing import Any
from sklearn.cluster import KMeans
from pyfmto.framework import Server, SyncDataManager
from pyfmto.utilities.loggers import logger
from .iaffbo_utils import Actions, average_params, ClientPackage


class IaffboServer(Server):
    """
    cl_num: 6
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.cl_num: int = int(kwargs.get('cl_num', 6))
        self.clients_data = SyncDataManager()
        self.agg_interval = 1.

    def handle_request(self, client_data: ClientPackage) -> Any:
        if client_data.action == Actions.PUSH_INIT:
            return self._push_init(client_data)
        elif client_data.action == Actions.PUSH_UPDATE:
            return self._push_update(client_data)
        elif client_data.action == Actions.PULL_UPDATE:
            return self._pull_update(client_data)
        else:
            raise ValueError(f"Unknown action: {client_data.action}")

    def _push_init(self, client_data: ClientPackage):
        cl_num = client_data.data.get('cl_num') if isinstance(client_data.data, dict) else None
        if cl_num is not None:
            self.cl_num = int(cl_num)
        return 'success'

    def _push_update(self, client_data: ClientPackage) -> Any:
        self.clients_data.update_src(client_data.cid, client_data.version, client_data.data)
        return 'success'

    def _pull_update(self, client_data: ClientPackage) -> Any:
        return self.clients_data.get_res(client_data.cid, client_data.version)

    @property
    def should_agg(self) -> bool:
        a = self.num_clients > 0
        b = self.clients_data.num_clients == self.num_clients
        lts_res_ver = min([self.clients_data.lts_res_ver(i) for i in self.sorted_ids])
        lts_src_ver = min([self.clients_data.lts_src_ver(i) for i in self.sorted_ids])
        c = lts_res_ver < lts_src_ver
        return a and b and c

    def aggregate(self):
        logger.info(
            f"Aggregating context\n"
            f"Should agg: {self.should_agg}\n"
            f"{self.clients_data.data_info}"
        )
        if not self.should_agg:
            return
        target_ver = self.clients_data.available_src_ver
        vectors = []
        params_by_cid: dict[int, dict] = {}
        for cid in self.sorted_ids:
            src = self.clients_data.get_src(cid, target_ver)
            logger.debug(f"Client {cid} src data {repr(list(src.keys())) if src is not None else None}")
            vec = src.get('vector', np.ndarray([]))
            vec = np.array(vec).reshape(-1)
            if vec.size == 0:
                vec = np.zeros(4, dtype=float)
            vectors.append(vec)
            params_by_cid[cid] = {
                'weights': src.get('weights', []),
                'biases': src.get('biases', []),
            }
        if not vectors:
            logger.warning(f"No vectors to aggregate, target_ver {target_ver}")
            return
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
            cids = [self.sorted_ids[i] for i in members]
            plist = [params_by_cid[cid] for cid in cids]
            agg = average_params(plist) if plist and plist[0]['weights'] else {}
            for cid in cids:
                self.clients_data.update_res(cid, target_ver, agg)
                logger.debug(f"Aggregated IAFFBO params for client {cid} ver {target_ver}")
