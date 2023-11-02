import os
import cv2
import time
import enum
import json

import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp

from typing import Optional, Callable, Dict
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager

from codebase.shared_memory.shared_ndarray import SharedNDArray
from codebase.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from codebase.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty


class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4


class SingleRealsense(mp.Process):
    MAX_PATH_LENGTH = 4096  # linux path has a limit of 4096 bytes

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        serial_number,
        resolution=(1280, 720),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        record_fps=None,
        enable_color=True,
        enable_depth=False,
        enable_infrared=False,
        get_max_k=30,
        advanced_mode_config=None,
        transform: Optional[Callable[[Dict], Dict]] = None,
        vis_transform: Optional[Callable[[Dict], Dict]] = None,
        recording_transform: Optional[Callable[[Dict], Dict]] = None,
        video_recorder: Optional[VideoRecorder] = None,
        verbose=False,
    ):
        super().__init__()
