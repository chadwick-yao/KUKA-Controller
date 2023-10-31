import api.sim as sim
from abc import ABCMeta, abstractmethod
from typing import Optional, List, Union
import numpy as np
import time
import datetime
import copy
import logging
import pathlib
import math
import json
from utils.data_utils import *

logger = logging.getLogger(__name__)


class BaseRobot(metaclass=ABCMeta):
    """ base robot class """

    def __init__(self,
                 RobotName: str,
                 TargetName: str,
                 DataDir: str,
                 DefaultCam: Union[List, str, None] = None,
                 OtherCam: Union[List, str, None] = None,
                 Address: str = "127.0.0.1",
                 Port: int = 19999,
                 ) -> None:
        self.robot_name = RobotName
        self.target_name = TargetName
        self.address = Address
        self.port = Port
        self.default_cam = DefaultCam
        self.cam_names = OtherCam
        self.meta_data = None
        self.marker_poses = None
        self.camera_dicts = {}

        sim.simxFinish(-1)  # in case, close all existed connections first
        self.clientID = sim.simxStart(
            connectionAddress = self.address, 
            connectionPort = self.port,
            waitUntilConnected = True,
            doNotReconnectOnceDisconnected = True,
            timeOutInMs = 5000,
            commThreadCycleInMs = 5,
        )

        # set path for data saving
        self.data_dir = pathlib.Path(DataDir)

        if not self.data_dir.is_dir():
            self.data_dir.mkdir()
            logger.info(f"Create data directory {self.data_dir}")
        
        self.img_path = self.data_dir / "rgb"
        self.depth_path = self.data_dir / "depth"
        self.pose_path = self.data_dir / "pose.json"

    @abstractmethod
    def _setup_robot(self):
        """ set up robot, if available """
        
        if self.robot_name is not None:
            sim_ret, self.robotHandle = sim.simxGetObjectHandle(self.clientID, self.robot_name, sim.simx_opmode_blocking)
            sim_ret, self.targetHanle = sim.simxGetObjectHandle(self.clientID, self.target_name, sim.simx_opmode_blocking)
        else:
            # set robot handle to target object if no robot used
            sim_ret, self.targetHanle = sim.simxGetObjectHandle(self.clientID, self.target_name, sim.simx_opmode_blocking)
            self.robotHandle = self.targetHanle

    def _setup_cameras(self):
        """ set up sim cameras, if available """

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

    def _close(self):
        """ kill the connection """
        # make sure that the last command sent out had time to arrive
        sim.simxGetPingTime(self.clientID)
        # close the connection to CoppeliaSim:
        sim.simxFinish(self.clientID)

    def _get_pose(self, obj_handle, use_quat=True):
        """ obtain object pose with position and rotation """

        assert obj_handle is not None, "object handler is not set."

        sim_ret, position = sim.simxGetObjectPosition(self.clientID, obj_handle, -1, sim.simx_opmode_blocking)

        if use_quat:
            sim_ret, rotation = sim.simxGetObjectQuaternion(self.clientID, obj_handle, -1, sim.simx_opmode_blocking)
        else:
            sim_ret, rotation = sim.simxGetObjectOrientation(self.clientID, obj_handle, -1, sim.simx_opmode_blocking)

        pose = (np.array(position), np.array(rotation))

        return pose
    
    def _set_pose(self, obj_handle, target_pose):
        """ set object into target pose """

        assert obj_handle is not None and target_pose is not None, "Object handler or target pose is not set."
        if len(target_pose) != 2 and not isinstance(target_pose, tuple):
            raise NotImplementedError("Only original VREP format is allowed for robot control at present.")

        target_pos, target_rot = target_pose
        sim.simxSetObjectPosition(self.clientID, obj_handle, -1, target_pos, sim.simx_opmode_blocking)

        if len(target_rot) == 4:    # Quaternion
            sim.simxSetObjectQuaternion(self.clientID, obj_handle, -1, target_rot, sim.simx_opmode_blocking)
        elif len(target_rot) == 3:  # Orientation
            sim.simxSetObjectOrientation(self.clientID, obj_handle, -1, target_rot, sim.simx_opmode_blocking)
        else:
            raise NotImplementedError("Unsupported rotation type.")
        

    def _get_meta(self):
        if self.meta_data is not None:
            return
        
        cams_info = copy.deepcopy(self.camera_dicts)
        for cam in cams_info.keys():
            cams_info[cam].pop('handle')

        timestamp = datetime.datetime.now().timestamp()

        meta_data = {
            'cam_default': self.default_cam,
            'cam_info': cams_info,
            'robot_info': self.robotHandle,
            'time': timestamp
        }

        self.meta_data = meta_data

    def _check_pose(self, target_obj_handle, mode="INFO"):
        sim_ret, orientation = sim.simxGetObjectOrientation(self.clientID, target_obj_handle, -1, sim.simx_opmode_blocking)
        sim_ret, position = sim.simxGetObjectPosition(self.clientID, target_obj_handle, -1, sim.simx_opmode_blocking)
        sim_ret, quaternion = sim.simxGetObjectQuaternion(self.clientID, target_obj_handle, -1, sim.simx_opmode_blocking)

        if mode == "INFO":
            logger.info(f"Trans: {position}, Orient: {orientation}, Quat: {quaternion}")
        elif mode == "JSON":
            try:
                with open(str(self.pose_path), "r") as file:
                    self.marker_poses = json.load(file)
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                self.marker_poses = {}

            marker_id = len(list(self.marker_poses.keys()))
            self.marker_poses[marker_id] = {
                "Pos": position,
                "Orient": orientation,
                "Quat": quaternion
            }

            with open(str(self.pose_path), "w") as file:
                json.dump(self.marker_poses, file, indent=4)
        else:
            raise NotImplementedError(f"Method not implemente for mode: {mode}.")

    def _get_camera_data(self, cam_info, need_depth=False):
        """ 
        obtain images from sim cameras 
            Return:
                (rgb image, depth image, camera name)
        """
        assert isinstance(cam_info, dict), "Camera Info must saved in dict type."

        cam_handle = cam_info["handle"]
        sim_ret, resolution, raw_image = sim.simxGetVisionSensorImage(self.clientID, cam_handle, 0, sim.simx_opmode_streaming)

        color_img = np.asarray(raw_image, dtype=np.uint8)
        color_img = color_img.resize([resolution[0], resolution[1], 3])

        color_img = np.fliplr(color_img)
        # color_img = np.flipud(color_img)

        if need_depth:
            sim_ret, resolution, depth_buffer = sim.simxGetVisionSensorDepthBuffer(self.clientID, cam_handle, sim.simx_opmode_buffer)

            depth_img = np.asarray(depth_buffer, dtype=np.uint8)
            depth_img = depth_img.resize([resolution[0], resolution[1]])
            depth_img = np.fliplr(depth_img)
            # depth_img = np.flipud(depth_img)

            zNear = 0.01
            zFar = 10
            depth_img = depth_img * (zFar - zNear) + zNear

            depth_scale = cam_info['depth_scale']
            depth_img = depth_img * depth_scale
        else:
            depth_img = np.array([])

        return color_img, depth_img, cam_info['name']

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def input2action(self):
        pass