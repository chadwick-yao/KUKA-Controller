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
from common.spacemouse import *
from typing import Optional, Callable, List, Tuple, Union
from codebase.sim_world.base.control_robot import BaseRobot
from common.data_utils import *
from collections import namedtuple

import api.sim as sim

FORMAT = '[%(asctime)s][%(levelname)s]: %(message)s'
logging.basicConfig(
    level = logging.INFO,
    format = FORMAT,
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

## define your DofCallback funtions & DofCallbackarr
def show_control_state(state: namedtuple):
    """ Dof event callback funtion """
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
    """ Button event callback funtion """
    print(
        (
            (
                "["
                + " ".join(["%2d, " % buttons[k] for k in range(len(buttons))])
            )
            + "]"
        )
    )

class DeviceConfig:
    def __init__(self,
                 callback: Callable[[object], None] = None,
                 dof_callback: Callable[[object], None] = None,
                 dof_callback_arr: List[DofCallback] = None,
                 button_callback: Callable[[object, list], None] = None,
                 button_callback_arr: List[ButtonCallback] = None,
                 set_nonblocking_loop: bool = True,
                 device: str = "SpaceMouse Wireless",
                 path: str = None,
                 DeviceNumber: int = 0,
                 ) -> None:
        check_config(callback, dof_callback, dof_callback_arr, button_callback, button_callback_arr)
        self.callback = callback
        self.dof_callback = dof_callback
        self.dof_callback_arr = dof_callback_arr
        self.button_callback = button_callback
        self.button_callback_arr = button_callback_arr
        self.set_nonblocking_loop = set_nonblocking_loop
        self.device = device
        self.path = path
        self.DeviceNumber = DeviceNumber

class iiwaRobot(BaseRobot):
    """ Wrapper for controlling iiwa in CoppeliaSim with SpaceMouse """
    
    def __init__(self,
                 SpaceMouseConf: DeviceConfig,
                 Address: str,
                 Port: int,
                 RobotName: str,
                 TargetName: str,
                 DataDir: str,
                 DefaultCam: Union[List, str, None] = None,
                 OtherCam: Union[List, str, None] = None,
                 PosSensitivity: float = 1.0,
                 RotSensitivity: float = 1.0,
                 ) -> None:
        
        super().__init__(
            RobotName = RobotName, 
            TargetName = TargetName, 
            Address = Address, 
            Port = Port,
            DataDir = DataDir,
            DefaultCam = DefaultCam
        )

        ## Connect SpaceMouse Device

        # show device lists
        logger.info(f'Mounted device: {list_devices()}')

        HID = pyspacemouse.open(
            callback = SpaceMouseConf.callback,
            dof_callback = SpaceMouseConf.dof_callback,
            dof_callback_arr = SpaceMouseConf.dof_callback_arr,
            button_callback = SpaceMouseConf.button_callback,
            button_callback_arr = SpaceMouseConf.button_callback_arr,
            set_nonblocking_loop = SpaceMouseConf.set_nonblocking_loop,
            device = SpaceMouseConf.device,
            path = SpaceMouseConf.path,
            DeviceNumber = SpaceMouseConf.DeviceNumber
        )
        self.HIDevice = HID

        ## scene 
        self.robot_dicts = {}
        self.cam_names = OtherCam

        assert self.clientID != -1, "Failed to connect to simulation server."
        logger.info(f"Connecting to {self.robot_name} , through address {self.address} and port {self.port}.")
        
        self.obj_handle = None
        self.frame_info_list = list()

        self._enable = False
        self.single_click_and_hold = False
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
        self.setup_robot()
        self.setup_cameras()

    def setup_robot(self):
        super().setup_robot()
        """ set up robot, if available """
        
        if self.robot_name is not None:
            sim_ret, self.robotHandle = sim.simxGetObjectHandle(self.clientID, self.robot_name, sim.simx_opmode_blocking)
            sim_ret, self.targetHanle = sim.simxGetObjectHandle(self.clientID, self.target_name, sim.simx_opmode_blocking)
        else:
            # set robot handle to target object if no robot used
            sim_ret, self.targetHanle = sim.simxGetObjectHandle(self.clientID, self.target_name, sim.simx_opmode_blocking)
            self.robotHandle = self.targetHanle

    def setup_cameras(self):
        super().setup_cameras()

        @staticmethod
        def _get_K(resolution):
            width, height = resolution
            view_angle = (54.70 / 180) * math.pi
            fx = (width / 2.) / math.tan(view_angle / 2)
            fy = fx
            cx = width / 2.
            cy = height / 2.
            # cam_intrinsics = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
            return cam_intrinsics
        assert len(self.cam_names) != 0, "No cameras to add, exiting..."

        for cam_name in self.cam_names:
            sim_ret, cam_handle = sim.simxGetObjectHandle(self.clientID, cam_name, sim.simx_opmode_blocking)
            _, resolution, _ = sim.simxGetVisionSensorImage(self.clientID, cam_handle, 0, sim.simx_opmode_streaming) # Recommended simx_opmode_streaming (the first call) and simx_opmode_buffer (the following calls)

            cam_intrinsic = _get_K(resolution)

            # Get camera pose and intrinsics in simulation
            sim_ret, cam_position = sim.simxGetObjectPosition(self.clientID, cam_handle, -1, sim.simx_opmode_blocking) # absolute position
            sim_ret, cam_quat = sim.simxGetObjectQuaternion(self.clientID, cam_handle, -1, sim.simx_opmode_blocking)

            cam_pose = get_pose_mat((cam_position, cam_quat))
            cam_depth_scale = 1
            cam_info_dict = {
                'name': cam_name,
                'handle': cam_handle,
                'pose': cam_pose.tolist(),
                'intrinsics': cam_intrinsic.tolist(),
                'depth_scale': cam_depth_scale,
                'im_shape': [resolution[1], resolution[0]]
            }

            self.camera_dicts[cam_name] = cam_info_dict

    def run(self):
        super().run()
        """ Listener method that keeps pulling new message. """
        while True:
            if self._enable:
            ## Read (pos, orient) from SpaceMouse
                _, dof_changed, button_changed = self.HIDevice.read()

                # button function
                if button_changed:
                    if self.control_gripper[0] == 0:    # release left button
                        self.single_click_and_hold = False
                    elif self.control_gripper[0] == 1:  # press left button
                        self.single_click_and_hold = True
                    elif self.control_gripper[1] == 1:  # press right button
                        self._reset_state = 1
                        self._enabled = False
                        self._reset_internal_state()
    
    def close(self):
        super().close()
        """ kill the connection """
        # make sure that the last command sent out had time to arrive
        sim.simxGetPingTime(self.clientID)
        # close the connection to CoppeliaSim:
        sim.simxFinish(self.clientID)

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

    def get_controller_state(self):
        """
        Grab the current state of the SpaceMouse

            Returns:
                dict: a dictionary contraining dpos, nor, unmodified orn, grasp, and reset
        """

        dpos = self.control_pose[:3] * 0.005 * self.pos_sensitivity
        roll, pitch, yaw = self.control_pose[3:] * 0.005 * self.rot_sensitivity

        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            reset=self._reset_state,
        )

    def input2action(self):
        raise NotImplementedError

    @property
    def control_pose(self):
        """ return current pose of SpaceMouse """
        return np.array(
            [
                getattr(self.HIDevice.tuple_state, k) 
                for k in ["x", "y", "z", "roll", "pitch", "yaw"]
            ]
        )
    
    @property
    def control_gripper(self):
        """
        return current gripper commonds
            1st button to control gripper
            2nd button to control whether restart
        """
        return np.array(getattr(self.HIDevice.tuple_state, "buttons"))


if __name__=="__main__":
    SpaceMouseConf = DeviceConfig(
        dof_callback = show_control_state
    )
    robot = iiwaRobot(
        SpaceMouseConf,
        Address = "127.0.0.1",
        Port = 19999,
        RobotName = "KUKA_iiwa7",
        TargetName = "target",
        DataDir = "data"
    )