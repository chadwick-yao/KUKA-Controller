import time
from socket import socket


class RealTime:
    def __init__(self, sock: socket):
        self.sock = sock

    def send(self, data: str):
        data = data + "\n"
        self.sock.send(data.encode("utf-8"))
        self.sock.recv(1024).decode("utf-8")

    def realTime_startDirectServoJoints(self):
        theCommand = "startDirectServoJoints"
        self.send(theCommand)
        time.sleep(0.3)

    def realTime_stopDirectServoJoints(self):
        theCommand = "stopDirectServoJoints"
        self.send(theCommand)
        time.sleep(0.3)

    def realTime_startDirectServoCartesian(self):
        theCommand = "stDcEEf_"
        self.send(theCommand)
        time.sleep(0.3)

    def realTime_stopDirectServoCartesian(self):
        theCommand = "stopDirectServoJoints"
        self.send(theCommand)
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
        self.send(theCommand)
        time.sleep(0.3)

    def realTime_stopImpedanceJoints(self):
        theCommand = "stopImpedanceJoints"
        self.send(theCommand)
        time.sleep(0.3)
