import time
from socket import socket
from typing import Tuple
from codebase.real_world.base.base_client import BaseClient


class RealTime(BaseClient):
    def __init__(self, host: str, port: int, trans: Tuple, sock: socket) -> None:
        super().__init__(host, port, trans)

        self.set_socket(sock)

    def _send(self, data: str):
        data = data + "\n"
        self.send(data)
        self.receive()

    def realTime_startDirectServoJoints(self):
        theCommand = "startDirectServoJoints"
        self._send(theCommand)
        time.sleep(0.3)

    def realTime_stopDirectServoJoints(self):
        theCommand = "stopDirectServoJoints"
        self._send(theCommand)
        time.sleep(0.3)

    def realTime_startDirectServoCartesian(self):
        theCommand = "stDcEEf_"
        self._send(theCommand)
        time.sleep(0.3)

    def realTime_stopDirectServoCartesian(self):
        theCommand = "stopDirectServoJoints"
        self._send(theCommand)
        time.sleep(0.3)

    def realTime_startImpedanceJoints(
        self, weightOfTool, cOMx, cOMy, cOMz, cStiffness, rStiffness, nStiffness
    ):
        theCommand = "startSmartImpedanceJoints"
        theCommand = theCommand + "_" + str(weightOfTool)
        theCommand = theCommand + "_" + str(cOMx)
        theCommand = theCommand + "_" + str(cOMy)
        theCommand = theCommand + "_" + str(cOMz)
        theCommand = theCommand + "_" + str(cStiffness)
        theCommand = theCommand + "_" + str(rStiffness)
        theCommand = theCommand + "_" + str(nStiffness) + "_"
        self._send(theCommand)
        time.sleep(0.3)

    def realTime_stopImpedanceJoints(self):
        theCommand = "stopImpedanceJoints"
        self._send(theCommand)
        time.sleep(0.3)
