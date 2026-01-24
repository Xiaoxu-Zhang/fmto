from pyfmto.framework import AlgorithmData

from .addfbo_client import AddfboClient
from .addfbo_server import AddfboServer


class ADDFBO(AlgorithmData):
    client = AddfboClient
    server = AddfboServer
