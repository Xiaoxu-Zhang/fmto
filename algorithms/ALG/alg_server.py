from typing import Any

from pyfmto.framework import Server, SyncDataManager

from algorithms.ALG.alg_utils import Actions, ClientPackage


class AlgServer(Server):
    """
    beta: 0.5
    """
    def __init__(self, **kwargs):
        super().__init__()
        kwargs = self.update_kwargs(kwargs)
        self.beta = kwargs['beta']
        self.clients_data = SyncDataManager()

    def handle_request(self, pkg: ClientPackage) -> Any:
        if pkg.action == Actions.PUSH:
            # save client pushed data into data manager
            self.clients_data.update_src(cid=pkg.cid, version=pkg.version, data=pkg)
            return 'save success'  # return any non-None value
        elif pkg.action == Actions.PULL:
            # return the client's agg result of the specified version in package
            return self.clients_data.get_res(cid=pkg.cid, version=pkg.version)
        else:
            raise ValueError(f"Unknown action {pkg.action}")

    def aggregate(self):
        # get the latest same version that all clients have uploaded
        ver_ready = self.clients_data.available_src_ver

        # aggregate data for each client
        # the self.sorted_ids is the sorted client ids, which is started from 1
        if not self.should_agg:
            return
        for cid in self.sorted_ids:
            if self.clients_data.get_res(cid=cid, version=ver_ready) is not None:
                # if the client's agg result existed, skip the agg process
                continue
            # We store the ClientPackage instance directly in the clients_data
            # So we can get the same type of data
            src: ClientPackage = self.clients_data.get_src(cid=cid, version=ver_ready)
            self.clients_data.update_res(
                cid=cid,
                version=ver_ready,
                data=f"src ver is {src.version}; data is {src.data}"
            )

    @property
    def should_agg(self) -> bool:
        ver_ready = self.clients_data.available_src_ver != -1
        clients_ready = self.clients_data.num_clients == self.num_clients
        return ver_ready and clients_ready
