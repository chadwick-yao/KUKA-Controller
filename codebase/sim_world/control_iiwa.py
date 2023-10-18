import time
import logging
import common.spacemouse as pyspacemouse
from common.spacemouse import *
from typing import Optional, Callable, List, Tuple
from codebase.sim_world.base.control_robot import BaseRobot

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

## define your ButtonCallback funtion & ButtonCallbackarr

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
                 ) -> None:
        super().__init__()

        ## Connect SpaceMouse Device

        # show device lists
        logger.info(f'Currently. connecting to {list_devices()}')

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
        sim.simxFinish(-1)  # in case, close all existed connections first
        self.robot_dicts = {}
        self.camera_dicts = {}
        

if __name__=="__main__":
    pass


# success = pyspacemouse.open()
# if success:
#     while True:
#         state = pyspacemouse.read()
#         print(state)