import time
import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
print(ROOT_DIR)
sys.path.append(ROOT_DIR)

from multiprocessing.managers import SharedMemoryManager
from codebase.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from codebase.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from codebase.real_world.robotiq85 import Robotiq85

if __name__ == "__main__":
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    with Robotiq85(
        shm_manager=shm_manager, frequency=100, receive_keys=None
    ) as gripper:
        time.sleep(1)
        for _ in range(3):
            time.sleep(1)
            gripper.execute(pose=[1])
            time.sleep(1)
            gripper.execute(pose=[0])
