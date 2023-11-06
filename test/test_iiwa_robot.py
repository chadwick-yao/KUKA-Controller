import sys
import time
import pathlib
import numpy as np

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

import time
from multiprocessing.managers import SharedMemoryManager
from codebase.real_world.iiwaPy3 import IIWAPositionalController

if __name__ == "__main__":
    with SharedMemoryManager() as shm_manager:
        with IIWAPositionalController(
            shm_manager=shm_manager, host="172.31.1.147", port=30001, receive_keys=None
        ) as robot:
            if robot.ready_servo.is_set() and robot.is_ready:
                init_pose = np.array(
                    [
                        530.7596789513458,
                        -0.013584256207491383,
                        494.38409670862484,
                        3.141558899639146,
                        8.640816915947057e-05,
                        3.141534157166136,
                    ]
                )

                for _ in range(10):
                    tmp = robot.get_state()
                    for key, value in tmp.items():
                        print(f"{key}: {value}")
                    time.sleep(0.5)
