import time
import copy
import math
import numpy as np
from codebase.real_world.iiwaPy3 import iiwaPy3
from codebase.real_world.spacemouse import SpaceMouse
from codebase.real_world.interpolators.linear_interpolator import LinearInterpolator
import utils.transform_utils as T
import logging

logger = logging.getLogger(__name__)


from datetime import datetime

start_time = datetime.now()


def getSecs():
    dt = datetime.now() - start_time
    secs = (dt.days * 24 * 60 * 60 + dt.seconds) + dt.microseconds / 1000000.0
    return secs


if __name__ == "__main__":
    interpolator = LinearInterpolator(
        ndim=3,
        controller_freq=100,
        policy_freq=20,
        ramp_ratio=0.5,
    )
    ori_interpolator = copy.deepcopy(interpolator)
    ori_interpolator.set_states(ori="euler")

    REMOTER = iiwaPy3(
        host="172.31.1.147",
        port=30001,
        trans=(0, 0, 0, 0, 0, 0),
    )

    REMOTER.reset_initial_state()
    logger.info(f"Have reset initial state successfully!")

    pos_sensitivity = 0.5
    rot_sensitivity = 0.005

    current_eef_pos = REMOTER.getEEFPos()
    REMOTER.realTime_startDirectServoCartesian()
    logger.info(f"Start real time Servo Cartesian mode.")

    try:
        with SpaceMouse(max_value=300) as sm:
            for _ in range(500):
                action = sm.get_motion_state_transformed()
                action = action[[0, 1, 2, 5, 4, 3]]
                time.sleep(1 / 100)
                next_eef_pos = np.zeros_like(current_eef_pos)

                next_eef_pos[:3] = current_eef_pos[:3] + action[:3] * pos_sensitivity
                next_eef_pos[3:] = current_eef_pos[3:] + action[3:] * rot_sensitivity

                # ! TODO: here got a problem
                ori_interpolator, interpolator = None, None
                if interpolator is not None and ori_interpolator is not None:
                    interpolator.set_start(current_eef_pos[:3])
                    ori_interpolator.set_start(current_eef_pos[3:])

                    interpolator.set_goal(next_eef_pos[:3])
                    ori_interpolator.set_start(next_eef_pos[3:])

                    print(f"Current: {current_eef_pos}")
                    print(f"Next: {next_eef_pos}")
                    while interpolator.step < interpolator.total_steps:
                        jointPositions = REMOTER.sendEEfPositionGetActualJpos(
                            np.concatenate(
                                [
                                    interpolator.get_interpolated_goal(),
                                    ori_interpolator.get_interpolated_goal(),
                                ]
                            )
                        )
                        logger.info(
                            f"Successfully moved to {np.around(next_eef_pos, decimals=2)}"
                        )
                        time.sleep(1 / 100)

                    current_eef_pos = copy.deepcopy(next_eef_pos)
                else:
                    jointPositions = REMOTER.sendEEfPositionGetActualJpos(next_eef_pos)
                    logger.info(
                        f"Successfully moved to {np.around(next_eef_pos, decimals=2)}"
                    )
                    time.sleep(1 / 100)

                    current_eef_pos = copy.deepcopy(next_eef_pos)
    except:
        REMOTER.realTime_stopDirectServoCartesian()
        logger.info(f"Stoped Servo Cartesian mode successfully!")
        REMOTER.reset_initial_state()
        logger.info(f"Reset back to initial state!")
        REMOTER.close()

    REMOTER.realTime_stopDirectServoCartesian()
    logger.info(f"Stoped Servo Cartesian mode successfully!")
    REMOTER.reset_initial_state()
    logger.info(f"Reset back to initial state!")
    REMOTER.close()
