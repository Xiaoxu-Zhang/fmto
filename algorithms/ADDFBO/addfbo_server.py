import numpy as np
from pyDOE import lhs
from tabulate import tabulate

from pyfmto.framework import Server, SyncDataManager
from pyfmto.utilities import logger
from .addfbo_utils import Actions, ClientPackage


class AddfboServer(Server):
    """
    log_agg: True
    d_aux_size: 1000
    retain_ratio: 0.3
    global_gp_size: 100
    row_size: 50  # row_size <= d_aux_size
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.d_aux_size = kwargs['d_aux_size']
        self.retain_ratio = kwargs['retain_ratio']
        self.global_gp_size = kwargs['global_gp_size']
        self._log_agg = kwargs['log_agg']
        self._row_size = kwargs['row_size']
        self.data_initialized = False
        self.dim = None
        self._is_shared = False
        self.d_share_x: dict[int, np.ndarray] = {}
        self.d_share_y_avg = None
        self.clients_data = SyncDataManager()
        self.agg_interval = 0.3

    def handle_request(self, pkg: ClientPackage):
        if pkg.action == Actions.PULL_AUX:
            return self._handle_pull_aux(pkg)
        elif pkg.action == Actions.PUSH_UPDATE:
            return self._handle_push(pkg)
        elif pkg.action == Actions.PULL_UPDATE:
            return self._handle_pull_klg(pkg)
        else:
            raise ValueError(f"Unknown client action: {pkg.action}")

    def aggregate(self):
        target_ver = self.clients_data.available_src_ver
        if target_ver > self.lts_res_ver and self.num_clients == self.clients_data.num_clients:
            agg_res = self.aggregation(version=target_ver)
            self.clients_data.update_res(
                cid=0,
                version=target_ver,
                data=agg_res
            )

    @property
    def lts_res_ver(self):
        return self.clients_data.lts_res_ver(0)

    @property
    def src_vers(self):
        src_versions = []
        for cid in self.sorted_ids:
            src_versions.append(self.clients_data.lts_src_ver(cid))
        return np.array(src_versions)

    def _handle_pull_aux(self, pkg: ClientPackage):
        self.dim = pkg.dim
        ver = pkg.version
        if ver in self.d_share_x:
            return self.d_share_x[ver]
        else:
            x_samples = lhs(self.dim, samples=self.d_aux_size)
            self.d_share_x[ver] = x_samples
            return x_samples

    def _handle_push(self, pkg: ClientPackage):
        cid = pkg.cid
        self.clients_data.update_src(cid, pkg.version, pkg)
        logger.debug(f"Received [client {pkg.cid} update V{pkg.version}]")
        return 'success'

    def _handle_pull_klg(self, pkg: ClientPackage):
        return self.clients_data.get_res(0, pkg.version)

    def aggregation(self, version):
        logger.debug("Using filtered auxiliary data averaging")
        selection_num = 10
        y_perd_mat = []
        y_mask_mat = []
        for cid in self.sorted_ids:
            update: ClientPackage = self.clients_data.get_src(cid, version)
            if update is None:
                logger.error(f"Client {cid} source V{version} not found, "
                             f"latest source is V{self.clients_data.lts_src_ver(cid)}")
                raise RuntimeError
            y_mean = update.d_share_y
            y_std = update.d_share_y_std

            std_asc_idx = np.argsort(y_std)
            selected_idx = std_asc_idx[:selection_num]
            bool_index = np.zeros_like(std_asc_idx, dtype=bool)
            bool_index[selected_idx] = True

            y_mask_mat.append(bool_index)
            y_perd_mat.append(y_mean)
        y_perd_mat = np.array(y_perd_mat)
        y_mask_mat = np.array(y_mask_mat)

        y_avg_arr, y_mask_arr = self._cal_filtered_avg(y_perd_mat, y_mask_mat)
        results = {'d_aux_y_avg': y_avg_arr,
                   'col_mask': y_mask_arr}
        return results

    def _cal_filtered_avg(self, y_pred_mat, y_mask_mat):
        assert y_pred_mat.shape == y_mask_mat.shape
        col_y_num = np.sum(y_mask_mat, axis=0)
        y_mask_arr = col_y_num > 0
        y_pred_mat[~y_mask_mat] = 0
        sum_y_pred = np.sum(y_pred_mat, axis=0)
        col_y_num[col_y_num == 0] = 1
        avg_y_pred = sum_y_pred / col_y_num
        avg_y_pred = np.clip(avg_y_pred, 0, 1)
        if self._log_agg:
            self.log_agg_table(sum_y_pred, col_y_num, avg_y_pred)
        return avg_y_pred, y_mask_arr

    def log_agg_table(self, sum_y_pred, col_y_num, avg_y_pred):
        headers = [f"{i}" for i in range(self._row_size)]
        indices = ['sum_y_pred', 'col_y_num', 'avg_y_pred']
        tab_data = [
            np.round(sum_y_pred[:self._row_size], 2),
            np.round(col_y_num[:self._row_size], 2),
            np.round(avg_y_pred[:self._row_size], 2)]
        tab = tabulate(tab_data, headers=headers, showindex=indices)
        logger.debug(f"\n{'=' * 30} Aggregation Table {'=' * 30}\n{tab}")
