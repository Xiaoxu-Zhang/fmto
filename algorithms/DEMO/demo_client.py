import time
from random import random

from pyfmto.framework import Client, record_runtime
from pyfmto.utilities import logger

from .demo_utils import ClientPackage, Actions


class DemoClient(Client):
    """
    alpha: 0.2
    """
    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        self.alpha = kwargs['alpha']
        self.version = 1  # version for client-server sync or inter-client process sync

        # update solutions automatically, so you just need to call
        # self.problem.evaluate(x), the solution will be updated
        self.problem.auto_update_solutions = True

        # otherwise, if you have other processing before update solutions,
        # you need to update manually after the processing.
        #
        #     x = ...
        #     y = self.problem.evaluate(x)
        #     self.problem.solution.append(x, y)

    @record_runtime("Total")
    def optimize(self):
        self.push()
        self.pull()
        x = self.find_next()
        self.problem.evaluate(x)
        self.version += 1

    @record_runtime("Find")
    def find_next(self):
        time.sleep(random())  # simulate the time cost
        return self.problem.random_uniform_x(1)

    @record_runtime()
    def push(self):
        # Here, the push() is to send the knowledge data
        # to the server. So, data should be stored in the pkg.
        # The push operation will be response timely, so the
        # 'repeat' usually keep its default.
        pkg = ClientPackage(cid=self.id, action=Actions.PUSH, version=self.version, data=f"{self.name}'s alpha is {self.alpha}")
        self.request_server(pkg)

    @record_runtime()
    def pull(self):
        # Here, the pull() is to get the latest data from the server
        pkg = ClientPackage(cid=self.id, action=Actions.PULL, version=self.version)

        # if response from the server is None or not meet the requirement
        # defined in self.check_pkg(), the request will be auto retried
        # max to repeat times after interval seconds until reach the max
        # repeat.
        resp = self.request_server(pkg, repeat=100, interval=0.1)
        logger.info(f"Pull response: {resp}")

    def check_pkg(self, pkg) -> bool:
        # Define your checking logic here
        # If return True, the package will be accepted, which
        # will be returned by the self.request_server()
        # If return False, the package will be dropped, which
        # will cause auto retry, the max retry times can be set by
        # self.request_server(..., repeat=10), default is 10
        return pkg is not None
