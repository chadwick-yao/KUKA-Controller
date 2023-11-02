import numpy as np
from typing import Tuple, Union
import logging
import socket
import time

FORMAT = "[%(asctime)s][%(levelname)s]: %(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

from codebase.real_world.base.PTP import PTP
from codebase.real_world.base.getters import Getters
from codebase.real_world.base.senders import Senders
from codebase.real_world.base.setters import Setters
from codebase.real_world.base.RealTime import RealTime
from codebase.real_world.base.base_client import BaseClient


class iiwaPy3(BaseClient):
    def __init__(
        self,
        host: str = "172.31.1.147",
        port: int = 30001,
        trans: Tuple = (0, 0, 0, 0, 0, 0),
    ) -> None:
        super().__init__(host, port, trans)

        self.connect()
        self.setter = Setters(host, port, trans, self.sock)
        self.getter = Getters(self.sock)
        self.sender = Senders(self.sock)
        self.rtl = RealTime(self.sock)
        self.ptp = PTP(self.sock)
        self.TCPtrans = trans

    def connect(self):
        try:
            self.set_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            self.sock.settimeout(15.0)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to {self.host}:{self.port}")
        except socket.error as e:
            logger.error(f"Connection failed: {e}")

        # Update the transform of the TCP if one is specified
        if all(num == 0 for num in self.trans):
            logger.info("No TCP transform in Flange Frame is defined.")
            logger.info(
                f"The following (default) TCP transform is utilized: {self.trans}"
            )
            return

        logger.info("Trying to mount the following TCP transform:")
        string_tuple = (
            "x (mm)",
            "y (mm)",
            "z (mm)",
            "alfa (rad)",
            "beta (rad)",
            "gamma (rad)",
        )

        for i in range(6):
            print(string_tuple[i] + ": " + str(self.trans[i]))

        da_message = "TFtrans_" + "_".join(map(str, self.trans)) + "\n"
        self.send(da_message)
        return_ack_nack = self.receive()

        if "done" in return_ack_nack:
            logger.info("Specified TCP transform mounted successfully")
        else:
            raise RuntimeError("Could not mount the specified TCP")

    def reset_initial_state(self):
        init_jpos = [0, np.pi * 20 / 180, 0, -np.pi * 80 / 180, 0, np.pi * 80 / 180, 0]
        init_vel = [0.05]

        self.movePTPJointSpace(jpos=init_jpos, relVel=init_vel)

    # PTP motion
    """
    Joint space motion
    """

    def movePTPJointSpace(self, jpos, relVel):
        self.ptp.movePTPJointSpace(jpos, relVel)

    def movePTPHomeJointSpace(self, relVel):
        self.ptp.movePTPHomeJointSpace(relVel)

    def movePTPTransportPositionJointSpace(self, relVel):
        self.ptp.movePTPTransportPositionJointSpace(relVel)

    """
    Cartesian linear  motion
    """

    def movePTPLineEEF(self, pos, vel):
        self.ptp.movePTPLineEEF(pos, vel)

    def movePTPLineEefRelBase(self, pos, vel):
        self.ptp.movePTPLineEefRelBase(pos, vel)

    def movePTPLineEefRelEef(self, pos, vel):
        self.ptp.movePTPLineEefRelEef(pos, vel)

    """
    Circular motion
    """

    def movePTPCirc1OrintationInter(self, f1, f2, vel):
        self.ptp.movePTPCirc1OrintationInter(f1, f2, vel)

    def movePTPArcYZ_AC(self, theta, c, vel):
        self.ptp.movePTPArcYZ_AC(theta, c, vel)

    def movePTPArcXZ_AC(self, theta, c, vel):
        self.ptp.movePTPArcXZ_AC(theta, c, vel)

    def movePTPArcXY_AC(self, theta, c, vel):
        self.ptp.movePTPArcXY_AC(theta, c, vel)

    def movePTPArc_AC(self, theta, c, k, vel):
        self.ptp.movePTPArc_AC(theta, c, k, vel)

    # realtime motion control
    def realTime_stopImpedanceJoints(self):
        self.rtl.realTime_stopImpedanceJoints()

    def realTime_startDirectServoCartesian(self):
        self.rtl.realTime_startDirectServoCartesian()

    def realTime_stopDirectServoCartesian(self):
        self.rtl.realTime_stopDirectServoCartesian()

    def realTime_stopDirectServoJoints(self):
        self.rtl.realTime_stopDirectServoJoints()

    def realTime_startDirectServoJoints(self):
        self.rtl.realTime_startDirectServoJoints()

    def realTime_startImpedanceJoints(
        self, weightOfTool, cOMx, cOMy, cOMz, cStiness, rStifness, nStifness
    ):
        self.rtl.realTime_startImpedanceJoints(
            weightOfTool, cOMx, cOMy, cOMz, cStiness, rStifness, nStifness
        )

    # Joint space servo command
    def sendJointsPositionsGetMTorque(self, x):
        return self.sender.sendJointsPositionsGetMTorque(x)

    def sendJointsPositionsGetExTorque(self, x):
        return self.sender.sendJointsPositionsGetExTorque(x)

    def sendJointsPositionsGetActualEEFpos(self, x):
        return self.sender.sendJointsPositionsGetActualEEFpos(x)

    def sendJointsPositionsGetEEF_Force_rel_EEF(self, x):
        return self.sender.sendJointsPositionsGetEEF_Force_rel_EEF(x)

    def sendJointsPositionsGetActualJpos(self, x):
        return self.sender.sendJointsPositionsGetActualJpos(x)

    # Crtesian space servo command
    def sendEEfPosition(self, x):
        self.sender.sendEEfPosition(x)

    def sendEEfPositionGetExTorque(self, x):
        return self.sender.sendEEfPositionExTorque(x)

    def sendEEfPositionGetActualEEFpos(self, x):
        return self.sender.sendEEfPositionGetActualEEFpos(x)

    def sendEEfPositionGetActualJpos(self, x):
        return self.sender.sendEEfPositionGetActualJpos(x)

    def sendEEfPositionGetEEF_Force_rel_EEF(self, x):
        return self.sender.sendEEfPositionGetEEF_Force_rel_EEF(x)

    def sendEEfPositionGetMTorque(self, x):
        return self.sender.sendEEfPositionMTorque(x)

    # getters
    def getEEFPos(self):
        return self.getter.get_EEF_pos()

    def getEEF_Force(self):
        return self.getter.get_EEF_force()

    def getEEFCartizianPosition(self):
        return self.getter.get_EEF_CartizianPos()

    def getEEF_Moment(self):
        return self.getter.get_EEF_moment()

    def getJointsPos(self):
        return self.getter.get_JointPos()

    def getJointsExternalTorques(self):
        return self.getter.get_Joints_ExternalTorques()

    def getJointsMeasuredTorques(self):
        return self.getter.get_Joints_MeasuredTorques()

    def getMeasuredTorqueAtJoint(self, x):
        return self.getter.get_MeasuredTorques_at_Joint(x)

    def getEEFCartizianOrientation(self):
        return self.getter.get_EEF_CartizianOrientation()

    # get pin states
    def getPin3State(self):
        return self.getter.get_pinState(3)

    def getPin10State(self):
        return self.getter.get_pinState(10)

    def getPin13State(self):
        return self.getter.get_pinState(13)

    def getPin16State(self):
        return self.getter.get_pinState(16)

    # setters
    def set_OnOff(self, cmd):
        self.setter.set_OnOff(cmd)
