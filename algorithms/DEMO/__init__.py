from pyfmto.framework import AlgorithmData

from .demo_client import DemoClient
from .demo_server import DemoServer


class DEMO(AlgorithmData):
    client = DemoClient
    server = DemoServer
