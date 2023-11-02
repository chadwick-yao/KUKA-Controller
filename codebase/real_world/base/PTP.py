import math
import sys
import time
from typing import Tuple
from codebase.real_world.base.getters import Getters
from codebase.real_world.base.senders import Senders
from socket import socket
from codebase.real_world.base.base_client import BaseClient


class PTP(BaseClient):
    def __init__(self, host: str, port: int, trans: Tuple, sock: socket) -> None:
        super().__init__(host, port, trans)

        self.set_socket(sock)
        self.sender = Senders(host, port, trans, sock)
        self.getter = Getters(host, port, trans, sock)

    def _send(self, data: str):
        data = data + "\n"
        self.send(data)
        message = self.receive()
        # print(message)
        # sys.stdout.flush()
        time.sleep(0.05)

    def awaitConfirmation(self):
        message = self.sock.recv(1024).decode("utf-8")
        # print(message)
        sys.stdout.flush()

    ## Arc motions
    def movePTPArc_AC(self, theta, c, k, vel):
        # print(theta)
        # print(c)
        # print(k)
        # print(vel)
        assert len(c) == 3, "Center of circle should be an array of 3 elements"
        assert len(k) == 3, "Orientation vector should be an array of 3 elements"
        assert len(theta) == 1, "Angle of an arc should be a scalar"
        assert len(vel) == 1, "Relative velocity should be a scalar"

        theta_ = theta[0]
        pos = self.getter.get_EEF_pos()
        r_2 = (
            math.pow(c[0] - pos[0], 2)
            + math.pow(c[1] - pos[1], 2)
            + math.pow(c[2] - pos[2], 2)
        )
        r = math.pow(r_2, 0.5)

        assert r != 0, "Radius can not be zero"
        assert theta_ != 0, "Angle can not be zero"

        # calculate unit vector
        r_2 = math.pow(k[0], 2) + math.pow(k[1], 2) + math.pow(k[2], 2)
        normK = math.pow(r_2, 0.5)
        assert normK != 0, "Norm of direction vector k shall not be zero"

        k[0] = k[0] / normK
        k[1] = k[1] / normK
        k[2] = k[2] / normK
        # print('k normalized successfully')
        s = [c[0] - pos[0], c[1] - pos[1], c[2] - pos[2]]
        s[0] = -s[0] / r
        s[1] = -s[1] / r
        s[2] = -s[2] / r
        n = [
            (k[1] * s[2] - k[2] * s[1]),
            (k[2] * s[0] - k[0] * s[2]),
            (k[0] * s[1] - k[1] * s[0]),
        ]
        # print('Performing arc motion based on c1 and c2')
        angle = theta_ / 2
        c1 = self.rotTheThing(angle, r, s, n, c)
        angle = theta_
        c2 = self.rotTheThing(angle, r, s, n, c)
        for i in range(3, 6):
            c1.append(pos[i])
            c2.append(pos[i])
        self.movePTPCirc1OrintationInter(c1, c2, vel)

    def rotTheThing(self, theta, r, s, n, c):
        c1 = [0, 0, 0]
        cos_ = math.cos(theta)
        sin_ = math.sin(theta)
        c1[0] = r * cos_ * s[0] + r * sin_ * n[0] + c[0]
        c1[1] = r * cos_ * s[1] + r * sin_ * n[1] + c[1]
        c1[2] = r * cos_ * s[2] + r * sin_ * n[2] + c[2]
        print(c1)
        return c1

    def checkErrorInRelVel(self, relVel):
        if not isinstance(relVel, (int, float)):
            print("Relative velocity should be a scalar")
            return True
        if not 0 < relVel < 1:
            print("Relative velocity should be in the range (0, 1)")
            return True
        return False

    def movePTPArcXY_AC(self, theta, c, vel):
        assert (
            len(theta) == 1
        ), "Error in function [movePTPArcXY_AC]: Rotation angle should be a scalar"
        assert (
            len(c) == 2
        ), "Error in function [movePTPArcXY_AC]: Center of rotation should be an array of two elements [x, y]"
        assert (
            len(vel) == 1
        ), "Error in function [movePTPArcXY_AC]: Velocity should be a scalar"
        k = [0, 0, 1]
        pos = self.getter.get_EEF_pos()
        c1 = [c[0], c[1], pos[2]]
        self.movePTPArc_AC(theta, c1, k, vel)

    def movePTPArcXZ_AC(self, theta, c, vel):
        assert (
            len(theta) == 1
        ), "Error in function [movePTPArcXZ_AC]: Rotation angle should be a scalar"
        assert (
            len(c) == 2
        ), "Error in function [movePTPArcXZ_AC]: Center of rotation should be an array of two elements [x, z]"
        assert (
            len(vel) == 1
        ), "Error in function [movePTPArcXZ_AC]: Velocity should be a scalar"

        k = [0, 1, 0]
        pos = self.getter.get_EEF_pos()
        c1 = [c[0], pos[1], c[1]]
        self.movePTPArc_AC(theta, c1, k, vel)

    def movePTPArcYZ_AC(self, theta, c, vel):
        assert (
            len(theta) == 1
        ), "Error in function [movePTPArcYZ_AC]: Rotation angle should be a scalar"
        assert (
            len(c) == 2
        ), "Error in function [movePTPArcYZ_AC]: Center of rotation should be an array of two elements [x, z]"
        assert (
            len(vel) == 1
        ), "Error in function [movePTPArcYZ_AC]: Velocity should be a scalar"

        k = [1, 0, 0]
        pos = self.getter.get_EEF_pos()
        c1 = [pos[0], c[1], c[2]]
        self.movePTPArc_AC(theta, c1, k, vel)

    def movePTPCirc1OrintationInter(self, f1, f2, relVel):
        assert (
            len(f1) == 6
        ), "Error in function [movePTPCirc1OrintationInter]: The first frame should be an array of 6 elements [x, y, z, alpha, beta, gamma]"
        assert (
            len(f2) == 6
        ), "Error in function [movePTPCirc1OrintationInter]: The second frame should be an array of 6 elements [x, y, z, alpha, beta, gamma]"
        assert (
            len(relVel) == 1
        ), "Error in function [movePTPCirc1OrintationInter]: Relative velocity should be a scalar"

        buff = "jRelVel_"
        buff = buff + str(relVel[0])
        buff = buff + "_"
        self._send(buff)
        self.sender.sendCirc1FramePos(f1)
        self.sender.sendCirc2FramePos(f2)
        theCommand = "doPTPinCSCircle1_"
        self._send(theCommand)
        self.awaitConfirmation()  # bug fixed on 1st October 2019, awaiting end of blocking motion

    def movePTPLineEEF(self, pos, vel):
        assert (
            len(vel) == 1
        ), "Error in function [movePTPLineEEF]: Velocity shall be a scalar"
        assert (
            len(pos) == 6
        ), "Error in function [movePTPLineEEF]: Position should be an array of 6 elements"

        buff = "jRelVel_" + str(vel[0]) + "_"
        command = buff
        self._send(command)
        self.sender.sendEEfPositions(pos)
        theCommand = "doPTPinCS"
        self._send(theCommand)
        self.awaitConfirmation()  # bug fixed on 1st October 2019, awaiting end of blocking motion

    def movePTPLineEefRelEef(self, pos, vel):
        assert (
            len(vel) == 1
        ), "Error in function [movePTPLineEefRelEef]: Velocity should be a scalar"
        assert (
            len(pos) == 3
        ), "Error in function [movePTPLineEefRelEef]: Position should be an array of 3 elements [x, y, z]"

        buff = "jRelVel_" + str(vel[0]) + "_"
        command = buff
        self._send(command)

        newPos = [0, 0, 0, 0, 0, 0]
        newPos[0] = pos[0]
        newPos[1] = pos[1]
        newPos[2] = pos[2]

        self.sender.sendEEfPositions(newPos)

        theCommand = "doPTPinCSRelEEF"
        self._send(theCommand)
        self.awaitConfirmation()  # bug fixed on 1st October 2019, awaiting end of blocking motion

    def movePTPLineEefRelBase(self, pos, vel):
        assert len(pos) == 3, "Position should be an array of three elements [x, y, z]"
        assert len(vel) == 1, "Velocity should be a scalar"

        buff = "jRelVel_" + str(vel[0]) + "_"
        command = buff
        self._send(command)

        newPos = [0, 0, 0, 0, 0, 0]
        newPos[0] = pos[0]
        newPos[1] = pos[1]
        newPos[2] = pos[2]

        self.sender.sendEEfPositions(newPos)

        theCommand = "doPTPinCSRelBase"
        self._send(theCommand)
        self.awaitConfirmation()  # bug fixed on 1st October 2019, awaiting end of blocking motion

    # joint space
    def movePTPJointSpace(self, jpos, relVel):
        assert (
            len(jpos) == 7
        ), "Error in function [movePTPJointSpace]: Joints positions shall be an array of 7 elements"
        assert (
            len(relVel) == 1
        ), "Error in function [movePTPJointSpace]: Relative velocity should be a scalar"

        buff = "jRelVel_" + str(relVel[0]) + "_"
        command = buff
        self._send(command)
        self.sender.sendJointsPositions(jpos)
        theCommand = "doPTPinJS"
        self._send(theCommand)
        self.awaitConfirmation()

    def movePTPHomeJointSpace(self, relVel):
        assert (
            len(relVel) == 1
        ), "Error in function [movePTPHomeJointSpace]: Relative velocity should be a scalar"

        buff = "jRelVel_" + str(relVel[0]) + "_"
        command = buff
        self._send(command)
        jpos = [0, 0, 0, 0, 0, 0, 0]
        self.sender.sendJointsPositions(jpos)
        theCommand = "doPTPinJS"
        self._send(theCommand)
        self.awaitConfirmation()

    def movePTPTransportPositionJointSpace(self, relVel):
        assert (
            len(relVel) == 1
        ), "Error in function [movePTPTransportPositionJointSpace]: Relative velocity should be a scalar"
        jpos = [0, 0, 0, 0, 0, 0, 0]
        jpos[3] = 25 * math.pi / 180
        jpos[5] = 90 * math.pi / 180
        self.movePTPJointSpace(jpos, relVel)
