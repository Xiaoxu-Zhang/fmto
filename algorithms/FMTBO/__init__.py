from pyfmto.framework import AlgorithmData

from .fmtbo_client import FmtboClient
from .fmtbo_server import FmtboServer


class FMTBO(AlgorithmData):
    client = FmtboClient
    server = FmtboServer
