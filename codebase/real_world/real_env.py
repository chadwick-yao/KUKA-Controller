import time
import math
import shutil
import pathlib
import numpy as np
from termcolor import colored, cprint

from typing import Dict, Optional, Union, List, Tuple

from multiprocessing.managers import SharedMemoryManager
from codebase.real_world.realsense.video_recoder import VideoRecorder
from codebase.real_world.robotiq85 import Robotiq85
from codebase.real_world.iiwaPy3 import IIWAPositionalController
from codebase.real_world.realsense.multi_realsense import (
    MultiRealsense,
    SingleRealsense,
)
from common.timestamp_accumulator import (
    TimestampActionAccumulator,
    TimestampObsAccumulator,
    align_timestamps,
)
from codebase.real_world.realsense.multi_camera_visualizer import MultiCameraVisualizer
from common.replay_buffer import ReplayBuffer
from utils.cv2_utils import get_image_transform, optimal_row_cols

DEFAULT_OBS_KEY_MAP = {
    # robot
    "EEFpos": "robot_eef_pos",
    "EEFrot": "robot_eef_rot",
    "Jpos": "robot_joint",
    # gripper
    "OpenOrClose": "gripper_pose",
    "camera_0": "agent_view",
    "camera_1": "view_in_hand",
    # timestamps
    "step_idx": "step_idx",
    "timestamps": "timestamps",
}


class RealEnv:
    def __init__(
        self,
        output_dir: str,
        robot_ip: str = "172.31.1.147",
        robot_port: int = 30001,
        # env params
        frequency: int = 10,
        n_obs_steps: int = 2,
        # obs
        obs_image_resolution: Tuple = (640, 480),
        max_obs_buffer_size: int = 30,
        camera_serial_numbers: Optional[List] = None,
        obs_key_map: Dict = DEFAULT_OBS_KEY_MAP,
        obs_float32: bool = False,
        # action
        max_pos_speed: float = 32,
        max_rot_speed: float = 0.5,
        # video capture params
        video_capture_fps: int = 30,
        video_capture_resolution: Tuple = (1280, 720),
        # saving params
        record_raw_video: bool = True,
        thread_per_video: int = 2,
        video_crf: int = 21,
        # vis params
        enable_multi_cam_vis: bool = True,
        multi_cam_vis_resolution: Tuple = (1280, 720),
        # shared memory
        shm_manager: Optional[SharedMemoryManager] = None,
        **kwargs,
    ) -> None:
        assert frequency <= video_capture_fps
        output_dir: pathlib.Path = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath("videos")
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath("replay_buffer.zarr").absolute())
        replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode="a")

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()

        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution,
            # obs output rgb
            bgr_to_rgb=True,
        )
        color_transform = color_tf
        if obs_float32:
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255

        def transform(data):
            data["color"] = color_transform(data["color"])
            return data

        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(camera_serial_numbers),
            in_wh_ratio=obs_image_resolution[0] / obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution,
        )
        vis_color_transform = get_image_transform(
            input_res=video_capture_resolution, output_res=(rw, rh), bgr_to_rgb=False
        )

        def vis_transform(data):
            data["color"] = vis_color_transform(data["color"])
            return data

        recording_transform = None
        recording_fps = video_capture_fps
        recording_pix_fmt = "bgr24"
        if not record_raw_video:
            recording_transform = transform
            recording_fps = frequency
            recording_pix_fmt = "rgb24"

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps,
            codec="h264",
            input_pix_fmt=recording_pix_fmt,
            crf=video_crf,
            thread_type="FRAME",
            thread_count=thread_per_video,
        )

        realsense = MultiRealsense(
            serial_numbers=camera_serial_numbers,
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            record_fps=recording_fps,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            get_max_k=max_obs_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            recording_transform=recording_transform,
            video_recorder=video_recorder,
            verbose=False,
        )

        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense, row=row, col=col, rgb_to_bgr=False
            )

        robot = IIWAPositionalController(
            shm_manager=shm_manager,
            host=robot_ip,
            port=robot_port,
            receive_keys=None,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
        )
        gripper = Robotiq85(
            shm_manager=shm_manager,
            frequency=100,
            receive_keys=None,
        )

        self.realsense = realsense
        self.robot = robot
        self.gripper = gripper
        self.multi_cam_vis = multi_cam_vis
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.obs_key_map = obs_key_map
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_realsense_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.delta_action_accumulator = None
        self.stage_accumulator = None

        self.start_time = None

    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready and self.gripper.is_ready

    def start(self, wait=True):
        self.realsense.start(wait=False)
        self.robot.start(wait=False)
        self.gripper.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.gripper.stop(wait=False)
        self.robot.stop(wait=False)
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.realsense.start_wait()
        self.robot.start_wait()
        self.gripper.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()

    def stop_wait(self):
        self.gripper.stop_wait()
        self.robot.stop_wait()
        self.realsense.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> Dict:
        assert self.is_ready

        # get data
        # 30 Hz, camera_receive_timestamp
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency))
        self.last_realsense_data = self.realsense.get(k=k, out=self.last_realsense_data)

        # 125 hz, robot_receive_timestamp
        last_robot_data = self.robot.get_all_state()
        last_gripper_data = self.gripper.get_all_state()
        # both have more than n_obs_steps data

        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = np.max(
            [x["timestamp"][-1] for x in self.last_realsense_data.values()]
        )
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)

        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value["timestamp"]
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            camera_obs[f"camera_{camera_idx}"] = value["color"][this_idxs]

        # robot obs
        robot_timestamps = last_robot_data["robot_receive_timestamp"]
        this_timestamps = robot_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        robot_obs_raw = dict()
        for k, v in last_robot_data.items():
            if k in self.obs_key_map:
                robot_obs_raw[self.obs_key_map[k]] = v

        # align robot obs
        robot_obs = dict()
        for k, v in robot_obs_raw.items():
            robot_obs[k] = v[this_idxs]

        # gripper obs
        gripper_timestamps = last_gripper_data["gripper_receive_timestamp"]
        this_timestamps = gripper_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)
        gripper_obs_raw = dict()
        for k, v in last_gripper_data.items():
            if k in self.obs_key_map:
                gripper_obs_raw[self.obs_key_map[k]] = v

        # align gripper obs
        for k, v in gripper_obs_raw.items():
            robot_obs[k] = v[this_idxs]

        robot_obs_raw.update(gripper_obs_raw)
        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(robot_obs_raw, robot_timestamps)

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data["timestamp"] = obs_align_timestamps
        return obs_data

    def exec_actions(
        self,
        actions: np.ndarray,
        timestamps: np.ndarray,
        delta_actions: np.ndarray,
        stages: Optional[np.ndarray] = None,
    ):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if not isinstance(delta_actions, np.ndarray):
            delta_actions = np.array(delta_actions)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_delta_actions = delta_actions[is_new]
        new_stages = stages[is_new]

        for i in range(len(new_actions)):
            self.robot.servoL(
                pose=new_actions[i][:-1], duration=1 / self.frequency
            )  # duration = SpaceMouseDeadzone / RealEnvFrequency
            self.gripper.execute(
                pose=[new_actions[i][-1]], duration=1 / self.frequency
            )

        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(new_actions, new_timestamps)
        if self.delta_action_accumulator is not None:
            self.delta_action_accumulator.put(new_delta_actions, new_timestamps)
        if self.stage_accumulator is not None:
            self.stage_accumulator.put(new_stages, new_timestamps)

    def get_robot_state(self):
        return self.robot.get_state()

    def get_gripper_state(self):
        return self.gripper.get_state()

    # recording API
    def start_episode(self, start_time=None):
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(str(this_video_dir.joinpath(f"{i}.mp4").absolute()))

        # start recording on realsense
        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time, dt=1 / self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time, dt=1 / self.frequency
        )
        self.delta_action_accumulator = TimestampActionAccumulator(
            start_time=start_time, dt=1 / self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time, dt=1 / self.frequency
        )
        print(f"Episode {episode_id} started!")

    def end_episode(self):
        assert self.is_ready

        # stop video recorder

        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps

            actions = self.action_accumulator.actions
            delta_actions = self.delta_action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            stages = self.stage_accumulator.actions
            # TODO: check here, obs_timestamps
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            if n_steps > 0:
                episode = dict()
                episode["timestamp"] = obs_timestamps[:n_steps]
                # episode["action"] = actions[:n_steps]
                episode["action"] = delta_actions[:n_steps]
                episode["stage"] = stages[:n_steps]
                for key, value in obs_data.items():
                    episode[key] = value[:n_steps]
                self.replay_buffer.add_episode(episode, compressors="disk")
                episode_id = self.replay_buffer.n_episodes - 1
                print(f"Episode {episode_id} saved!")

            self.obs_accumulator = None
            self.action_accumulator = None
            self.delta_action_accumulator = None
            self.stage_accumulator = None
        self.realsense.stop_recording()

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f"Episode {episode_id} dropped!")
