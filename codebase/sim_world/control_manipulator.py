import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)

import time
import datetime
import math
import numpy as np
import logging
import threading
import common.spacemouse as pyspacemouse
from common.spacemouse import DeviceConfig
from common.spacemouse import *
from typing import Optional, Callable, List, Tuple, Union
from codebase.sim_world.base.control_robot import BaseRobot
from utils.data_utils import *
from collections import namedtuple

import api.sim as sim

FORMAT = "[%(asctime)s][%(levelname)s]: %(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


## define your DofCallback funtions & DofCallbackarr
def show_control_state(state: namedtuple):
    """Dof event callback funtion"""
    if state:
        print(
            "\t".join(
                [
                    "%4s %+.4f" % (k, getattr(state, k))
                    for k in ["x", "y", "z", "roll", "pitch", "yaw"]
                ]
            )
        )


## define your ButtonCallback funtion & ButtonCallbackarr
def show_button_status(state, buttons):
    """Button event callback funtion"""
    print((("[" + " ".join(["%2d, " % buttons[k] for k in range(len(buttons))])) + "]"))


class ManipulatorRobot(BaseRobot):
    """Wrapper for controlling Manipulator in CoppeliaSim with SpaceMouse"""

    def __init__(
        self,
        SpaceMouseConf: DeviceConfig,
        Address: str,
        Port: int,
        RobotName: str,
        TargetName: str,
        ObjName: Optional[List],
        DataDir: str,
        DefaultCam: Union[List, str, None] = None,
        OtherCam: Union[List, str, None] = None,
        PosSensitivity: float = 1.0,
        RotSensitivity: float = 1.0,
    ) -> None:
        super().__init__(
            RobotName=RobotName,
            TargetName=TargetName,
            Address=Address,
            Port=Port,
            DataDir=DataDir,
            DefaultCam=DefaultCam,
            OtherCam=OtherCam,
        )

        ## Connect SpaceMouse Device

        # show device lists
        logger.info(f"Mounted device: {list_devices()}")

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

        ## scene
        self.robot_dicts = {}

        assert self.clientID != -1, "Failed to connect to simulation server."
        logger.info(
            f"Connecting to {self.robot_name} , through address {self.address} and port {self.port}."
        )

        self.obj_handle = {obj_name: None for obj_name in ObjName}
        self.frame_info_list = list()

        self._enable = False
        self.single_click_and_hold = False
        self.gripper_changing = False
        self.CloseOrOpen = "close"
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.pos_sensitivity = PosSensitivity
        self.rot_sensitivity = RotSensitivity

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        ## launch a listener thread to listen to SpaceMouse
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def setup_all(self):
        self._setup_robot()
        # self._setup_cameras()

    def _setup_robot(self):
        """setup any object you want here"""
        super()._setup_robot()
        self.obj_handle = {
            obj_name: sim.simxGetObjectHandle(
                self.clientID, obj_name, sim.simx_opmode_blocking
            )[1]
            for obj_name in self.obj_handle.keys()
        }

    def run(self):
        super().run()
        """ Listener method that keeps pulling new message. """
        while True:
            if self._enable:
                ## Read (pos, orient) from SpaceMouse
                _, dof_changed, button_changed = self.HIDevice.read()

                # button function
                if button_changed:
                    if self.HIDevice.control_gripper[0] == 0:  # release left button
                        self.single_click_and_hold = False
                    elif self.HIDevice.control_gripper[0] == 1:  # press left button
                        if not self.single_click_and_hold:  # 0 -> 1
                            self.gripper_changing = True
                        self.single_click_and_hold = True
                    if self.HIDevice.control_gripper[1] == 1:  # press right button
                        self._reset_state = 1
                        self._enable = False

    def start_control(self):
        self._reset_internal_state()
        self._reset_state = 0
        self._enable = True

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        # Reset 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        # Reset grasp
        self.single_click_and_hold = False
        self.last_gripper_state = False

        # (array([ 0.125     , -0.275114  ,  0.39874786]), array([3.1415925 , 0.08726646, 3.1415925 ]))
        target_pose = (
            np.array([0.125, -0.275114, 0.39874786]),
            np.array([3.1415925, 0, 3.1415925]),
        )
        # block (array([ 0.12800001, -0.27599999,  0.22499999]), array([-3.51055849e-17,  5.06398772e-18,  1.22060484e-20]))
        sim_ret, block_handle = sim.simxGetObjectHandle(
            self.clientID, "block", sim.simx_opmode_blocking
        )
        block_pose = (np.array([0.128, -0.276, 0.225]), np.array([0, 0, 0]))
        self._set_pose(self.targetHanle, target_pose)
        self._set_pose(block_handle, target_pose)
        time.sleep(0.01)  # wait

    def input2action(self):
        state: dict = self.get_controller_state

        dpos, rotation, raw_rotation, grasp, reset = [
            state[key] for key in state.keys()
        ]

        # if we are resetting, directly return None value
        if reset:
            return None, None

        # some pre=processing FIXME
        dpos = dpos * np.array([-1, -1, 1])
        raw_rotation[0], raw_rotation[1] = raw_rotation[1], raw_rotation[0]
        action = (dpos, raw_rotation)
        orig_pose = self._get_pose(self.targetHanle, use_quat=False)
        target_pose = (action[0] + orig_pose[0], action[1] + orig_pose[1])
        self._set_pose(self.targetHanle, target_pose)

        # gripper position setting

        if self.gripper_changing:
            self.gripper_changing = False
            if self.CloseOrOpen == "close":
                grasp = -1
                self.CloseOrOpen = "open"
            elif self.CloseOrOpen == "open":
                grasp = 1
                self.CloseOrOpen = "close"
            res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(
                self.clientID,
                "ROBOTIQ_85",
                sim.sim_scripttype_childscript,
                "ROBOTIQ_CloseOpen",
                [grasp],
                [],
                [],
                b"",
                sim.simx_opmode_blocking,
            )

        # time.sleep(0.01) # wait
        if self._reset_state:
            self._reset_state = 0
            self._enable = True
            self._reset_internal_state()

    @property
    def get_controller_state(self):
        """
        Grab the current state of the SpaceMouse

            Returns:
                dict: a dictionary contraining dpos, nor, unmodified orn, grasp, and reset
        """

        dpos = self.HIDevice.control_pose[:3] * 0.01 * self.pos_sensitivity
        roll, pitch, yaw = self.HIDevice.control_pose[3:] * 0.05 * self.rot_sensitivity

        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.HIDevice.control_gripper,
            reset=self._reset_state,
        )


if __name__ == "__main__":
    SpaceMouseConf = DeviceConfig(
        # dof_callback = show_control_state
    )
    robot = ManipulatorRobot(
        SpaceMouseConf,
        Address="127.0.0.1",
        Port=19999,
        RobotName="LBR_iiwa_7_R800",
        TargetName="targetSphere",
        DataDir="data",
        ObjName=["ROBOTIQ_85"],
    )

    # cnt = 0
    # start_time = time.time() * 1000

    robot.setup_all()
    robot.start_control()
    while True:
        # cnt += 1
        # if cnt % 1000 == 0:
        #     cnt = 0
        #     end_time = time.time() * 1000
        #     print(f"Execute one time input2action need {(end_time - start_time - 0.02*1000) / 1000} ms averagely.")
        #     start_time = end_time
        robot.input2action()
