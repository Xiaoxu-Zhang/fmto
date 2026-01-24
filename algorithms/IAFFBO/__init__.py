from pyfmto.framework import AlgorithmData

from .iaffbo_client import IaffboClient
from .iaffbo_server import IaffboServer


class IAFFBO(AlgorithmData):
    client = IaffboClient
    server = IaffboServer
