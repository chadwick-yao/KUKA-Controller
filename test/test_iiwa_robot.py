import sys
import pathlib

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
            for _ in range(10):
                time.sleep(3)
