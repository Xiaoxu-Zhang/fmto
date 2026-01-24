from pyfmto.framework import AlgorithmData

from .bo_client import BoClient
from .bo_server import BoServer


class BO(AlgorithmData):
    client = BoClient
    server = BoServer
