from enum import Enum, auto
from pyfmto.framework import ClientPackage as Pkg


class Actions(Enum):
    PUSH = auto()
    PULL = auto()


class ClientPackage(Pkg):
    def __init__(self, cid: int, action: Enum, version: int, data: str = None):
        super().__init__(cid, action)
        self.version = version
        self.data = data
        # action can be any immutable type, but Enum is recommended
        # for example, action can be a string:
        #
        #    class ClientPackage(Pkg):
        #        def __init__(self, cid: int, action: str, data: Any):
        #            super().__init__(cid, action)
        #            self.data = data
        #
        # In this case, action not depend on any other class, but
        # you may need to remember what the exactly name of actions
        #
        # You can extend this class to add more fields
        # for example:
        #
        #     class ClientPackage(Pkg):
        #         def __init__(self, cid: int, action: Enum, data: dict, msg: str):
        #             super().__init__(cid, action)
        #             self.msg = msg
        #             self.data = data
        #
