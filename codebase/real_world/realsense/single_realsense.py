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
from codebase.real_world.realsense.video_recoder import VideoRecorder


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

        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()
        if enable_color:
            examples["color"] = np.empty(shape=shape + (3,), dtype=np.uint8)
        if enable_depth:
            examples["depth"] = np.empty(shape=shape, dtype=np.uint16)
        if enable_infrared:
            examples["infrared"] = np.empty(shape=shape, dtype=np.uint8)
        examples["camera_capture_timestamp"] = 0.0
        examples["camera_receive_timestamp"] = 0.0
        examples["timestamp"] = 0.0
        examples["step_idx"] = 0

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples
            if vis_transform is None
            else vis_transform(dict(examples)),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps,
        )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps,
        )

        # create command queue
        examples = {
            "cmd": Command.SET_COLOR_OPTION.value,
            "option_enum": rs.option.exposure.value,
            "option_value": 0.0,
            "video_path": np.array("a" * self.MAX_PATH_LENGTH),
            "recording_start_time": 0.0,
            "put_start_time": 0.0,
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=examples, buffer_size=128
        )

        # create shared array for intrinsics
        intrinsics_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager, shape=(7,), dtype=np.float64
        )
        intrinsics_array.get()[:] = 0

        # create video recorder
        if video_recorder is None:
            # realsense uses bgr24 pixel format
            # default thread_type to FRAEM
            # i.e. each frame uses one core
            # instead of all cores working on all frames.
            # this prevents CPU over-subpscription and
            # improves performance significantly
            video_recorder = VideoRecorder.create_h264(
                fps=record_fps,
                codec="h264",
                input_pix_fmt="bgr24",
                crf=18,
                thread_type="FRAME",
                thread_count=1,
            )

        # copied variables
        self.serial_number = serial_number
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_infrared = enable_infrared
        self.advanced_mode_config = advanced_mode_config
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue
        self.intrinsics_array = intrinsics_array
