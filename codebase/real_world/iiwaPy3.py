from codebase.real_world.utils.base_client import BaseClient
from codebase.real_world.utils.getters import Getters
from codebase.real_world.utils.senders import Senders
from codebase.real_world.utils.setters import Setters
from codebase.real_world.utils.RealTime import RealTime
from codebase.real_world.utils.PTP import PTP
from typing import Tuple, Union
import common.spacemouse as pyspacemouse
from common.spacemouse import DeviceSpec
import threading


class iiwaPy3(BaseClient):
    def __init__(
        self,
        SpaceMouseConf: DeviceSpec,
        PosSensibility: float,
        RotSensitivity: float,
        host: str = "127.0.0.1",
        port: int = 30001,
        trans: Tuple = (0, 0, 0, 0, 0, 0),
    ) -> None:
        super().__init__(host, port, trans)

        self.connect()
        self.setter = Setters(self.sock)
        self.getter = Getters(self.sock)
        self.sender = Senders(self.sock)
        self.rtl = RealTime(self.sock)
        self.ptp = PTP(self.sock)
        self.TCPtrans = trans

        HID = pyspacemouse.open(
            callback=SpaceMouseConf.callback,
            dof_callback=SpaceMouseConf.dof_callback,
            dof_callback_arr=SpaceMouseConf.dof_callback_arr,
            button_callback=SpaceMouseConf.button_callback,
            button_callback_arr=SpaceMouseConf.button_callback_arr,
            set_nonblocking_loop=SpaceMouseConf.set_nonblocking_loop,
            device=SpaceMouseConf.device,
            path=SpaceMouseConf.path,
            DeviceNumber=SpaceMouseConf.DeviceNumber,
        )

        self.HIDevice = HID
        self.pos_sensitivity = PosSensibility
        self.rot_sensitivity = RotSensitivity
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        raise NotImplementedError

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
    def sendEEfPosition(self, x):
        self.sender.sendEEfPosition(x)

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
