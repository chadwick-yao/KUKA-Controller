import sys
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

import time
import copy
import math
import logging
import numpy as np
from termcolor import colored, cprint

import utils.transform_utils as T

from common.spacemouseX import SpaceMouse
from codebase.real_world.robotiq85 import Robotiq85

from codebase.real_world.iiwaPy3 import iiwaPy3
from codebase.real_world.interpolators.linear_interpolator import LinearInterpolator

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    interpolator = LinearInterpolator(
        ndim=3,
        controller_freq=100,
        policy_freq=20,
        ramp_ratio=0.5,
    )
    ori_interpolator = copy.deepcopy(interpolator)
    ori_interpolator.set_states(ori="euler")
    use_interpolator = True

    pos_sensitivity = 0.5
    rot_sensitivity = 0.005

    last_button = [False, False]

    GRIPPER = Robotiq85()
    GRIPPER.activate()
    time.sleep(3)
    GRIPPER.reset()
    logger.info(f"Gripper is prepared.")

    REMOTER = iiwaPy3(
        host="172.31.1.147",
        port=30001,
        trans=(0, 0, 0, 0, 0, 0),
    )
    REMOTER.reset_initial_state()
    logger.info(f"Have reset initial state successfully!")

    init_eef_pos = REMOTER.getEEFPos()
    current_eef_pos = copy.deepcopy(init_eef_pos)

    try:
        REMOTER.realTime_startDirectServoCartesian()
        cprint("Start real time Servo Cartesian mode.", "magenta")

        with SpaceMouse(max_value=300) as sm:
            for _ in range(2000):
                action = sm.get_motion_state_transformed()
                current_button = [sm.is_button_pressed(0), sm.is_button_pressed(1)]
                action = action[[0, 1, 2, 5, 4, 3]]
                action[5] = -action[5]
                time.sleep(1 / 100)
                next_eef_pos = np.zeros_like(current_eef_pos)

                next_eef_pos[:3] = current_eef_pos[:3] + action[:3] * pos_sensitivity
                next_eef_pos[3:] = current_eef_pos[3:] + action[3:] * rot_sensitivity

                # ! TODO: here got a problem about INTERPOLATOR
                use_interpolator = False
                if use_interpolator:
                    interpolator.set_start(current_eef_pos[:3])
                    ori_interpolator.set_start(current_eef_pos[3:])

                    interpolator.set_goal(next_eef_pos[:3])
                    ori_interpolator.set_start(next_eef_pos[3:])

                    while interpolator.step < interpolator.total_steps:
                        _ = REMOTER.sendEEfPositionGetActualJpos(
                            np.concatenate(
                                [
                                    interpolator.get_interpolated_goal(),
                                    ori_interpolator.get_interpolated_goal(),
                                ]
                            )
                        )
                        # logger.info(
                        #     f"Successfully moved to {np.around(next_eef_pos, decimals=2)}"
                        # )
                        time.sleep(1 / 100)

                    current_eef_pos = copy.deepcopy(next_eef_pos)
                else:
                    _ = REMOTER.sendEEfPositionGetActualJpos(next_eef_pos)
                    # logger.info(
                    #     f"Successfully moved to {np.around(next_eef_pos, decimals=2)}"
                    # )

                    if current_button[0] and not last_button[0]:
                        if GRIPPER.is_closed():
                            GRIPPER.open()
                            logger.info("Button 1 has been pressed.\nGripper opened!")
                        elif GRIPPER.is_opened():
                            GRIPPER.close()
                            logger.info("Button 1 has been pressed.\nGripper closed!")
                    # ! TODO: fix reset problem
                    if current_button[1] and not last_button[1]:
                        # stop Servo Cartesian
                        REMOTER.realTime_stopDirectServoCartesian()

                        _ = REMOTER.reset_initial_state()

                        time.sleep(5)
                        # start Servo Cartesian again
                        REMOTER.realTime_startDirectServoCartesian()

                        time.sleep(5)
                        logger.info("Button 2 has been pressed.\nRobot env reset!")

                    time.sleep(1 / 100)

                    last_button = current_button
                    current_eef_pos = copy.deepcopy(next_eef_pos)
    except:
        raise RuntimeError
    finally:
        REMOTER.realTime_stopDirectServoCartesian()
        logger.info(f"Stoped Servo Cartesian mode successfully!")
        REMOTER.reset_initial_state()
        logger.info(f"Reset back to initial state!")
        REMOTER.close()
