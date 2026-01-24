from .fdemd_client import FdemdClient
from .fdemd_server import FdemdServer
from pyfmto.framework import AlgorithmData


class FDEMD(AlgorithmData):
    client = FdemdClient
    server = FdemdServer
