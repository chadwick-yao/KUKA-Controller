"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip> --robot_port <port>
e.g python demo_real_robot.py -o "data/test_data" --robot_ip "172.31.1.147" --robot_port 30001 --frequency 20 -ps 2.0 

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

import time
import click
import cv2
import tqdm
import numpy as np
import copy
from termcolor import colored, cprint
import scipy.spatial.transform as st
from multiprocessing.managers import SharedMemoryManager
from codebase.real_world.real_env import RealEnv
from common.spacemouse_shared_memory import Spacemouse
from common.precise_sleep import precise_wait
from common.keystroke_counter import KeystrokeCounter, Key, KeyCode
from common.sec_time_counter import SecTimeCounter
from utils.data_utils import pose_euler2quat


@click.command()
@click.option(
    "--output", "-o", required=True, help="Directory to save demonstration dataset."
)
@click.option(
    "--robot_ip", "-ri", required=True, help="IIWA's IP address e.g. 172.31.1.147"
)
@click.option(
    "--robot_port", "-rp", required=True, type=int, help="IIWA's port e.g. 30001"
)
@click.option(
    "--vis_camera_idx", default=0, type=int, help="Which RealSense camera to visualize."
)
@click.option(
    "--frequency", "-f", default=10, type=float, help="Control frequency in Hz."
)
@click.option(
    "--pos_sensitivity",
    "-ps",
    default=1.0,
    type=float,
    help="Position control sensitivity. [0.0, 1.0] (The less value it is, the smoother it gets but slower.)",
)
@click.option(
    "--rot_sensitivity",
    "-rs",
    default=1.0,
    type=float,
    help="Rotation control sensitivity. [0.0, 1.0] (The less value it is, the smoother it gets but slower.)",
)
@click.option(
    "--command_latency",
    "-cl",
    default=0.01,
    type=float,
    help="Latency between receiving SapceMouse command to executing on Robot in Sec.",
)
@click.option(
    "--time_limit_mode",
    "-tlm",
    default=12,
    type=int,
    help="Record a limited period of time.",
)
def main(
    output,
    robot_ip,
    robot_port,
    vis_camera_idx,
    frequency,
    command_latency,
    pos_sensitivity,
    rot_sensitivity,
    time_limit_mode,
):
    dt = 1 / frequency

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, Spacemouse(
            shm_manager=shm_manager,
            get_max_k=30,
            frequency=200,
            deadzone=(0, 0, 0, 0.1, 0.1, 0.1),
        ) as sm, RealEnv(
            output_dir=output,
            robot_ip=robot_ip,
            robot_port=robot_port,
            n_obs_steps=2,
            # recording resolution
            obs_image_resolution=(1280, 720),
            frequency=frequency,
            enable_multi_cam_vis=True,
            record_raw_video=True,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=4,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager,
            max_pos_speed=128,
            max_rot_speed=0.75,
        ) as env:
            cv2.setNumThreads(1)

            # realsense exposure
            env.realsense.set_exposure(exposure=300, gain=10)
            # realsense white balance
            # env.realsense.set_white_balance(white_balance=5900)

            time.sleep(3.0)
            cprint("Ready!", on_color="on_red")
            state = env.get_robot_state()
            target_pose = np.append(state["EEFpos"], state["EEFrot"])

            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            last_button = [False, False]
            G_target_pose = 0  # open

            timer: SecTimeCounter = SecTimeCounter(time_limit_mode)
            while not stop and env.robot.ready_servo.is_set():
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()

                # handle timer process
                if timer.stop_record_event.is_set():
                    # stop recording
                    env.end_episode()
                    is_recording = False

                    timer.stop()
                    cprint("Timer stop!", on_color="on_blue")

                    cprint("Stopped.", on_color="on_red")
                    timer = SecTimeCounter(time_limit_mode)

                # handle key process
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char="q"):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char="c"):
                        # Start recording
                        env.start_episode(
                            t_start
                            + (iter_idx + 2) * dt
                            - time.monotonic()
                            + time.time()
                        )
                        key_counter.clear()
                        is_recording = True
                        cprint("Recording!", on_color="on_green")

                        timer.start()
                        cprint("Timer start!", on_color="on_blue")
                    elif key_stroke == KeyCode(char="s"):
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False

                        timer.stop()
                        cprint("Timer stop!", on_color="on_blue")

                        cprint("Stopped.", on_color="on_red")
                        timer = SecTimeCounter(time_limit_mode)
                    elif key_stroke == KeyCode(char="r"):
                        env.robot.reset_robot()
                        target_pose = copy.deepcopy(env.robot.init_eef_pose)

                        target_pose[:3] += np.clip(
                            np.random.normal(0, 5, size=3), -5, 5
                        )
                        target_pose[3:] += np.clip(
                            np.random.normal(0, 1, size=3), -0.05, 0.05
                        )
                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm("Are you sure to drop an episode?"):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False

                stage = key_counter[Key.space]

                # set target_pose with latest value
                # target_pose[:3] = obs["robot_eef_pos"][-1]
                # target_pose[3:] = obs["robot_eef_rot"][-1]

                # visualize
                vis_img = obs[f"camera_{vis_camera_idx}"][-1, :, :, ::-1].copy()
                episode_id = env.replay_buffer.n_episodes
                text = f"Episode: {episode_id}, Stage: {stage}"
                if is_recording:
                    text += ", Recording!"
                cv2.putText(
                    vis_img,
                    text,
                    (10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255, 255, 255),
                )

                cv2.imshow("default", vis_img)
                cv2.pollKey()

                precise_wait(t_sample)
                # get teleop command
                sm_state = sm.get_motion_state_transformed()
                # print(sm_state)

                dpos = sm_state[:3] * (env.max_pos_speed / frequency) * pos_sensitivity
                drot_xyz = (
                    sm_state[3:]
                    * (env.max_rot_speed / frequency)
                    * np.array([-1, 1, -1])
                    * rot_sensitivity
                )

                # ------------- Button Features -------------
                current_button = [sm.is_button_pressed(0), sm.is_button_pressed(1)]
                if not is_recording and current_button[0]:
                    # Start recording
                    env.start_episode(
                        t_start + (iter_idx + 2) * dt - time.monotonic() + time.time()
                    )
                    key_counter.clear()
                    is_recording = True
                    cprint("Recording!", on_color="on_green")

                    timer.start()
                    cprint("Timer start!", on_color="on_blue")

                if current_button[1] and not last_button[1]:
                    G_target_pose = 1 ^ G_target_pose
                last_button = current_button

                # pose transformation
                drot = st.Rotation.from_euler("xyz", drot_xyz)
                target_pose[:3] += dpos
                target_pose[3:] = (
                    drot * st.Rotation.from_euler("zyx", target_pose[3:])
                ).as_euler("zyx")

                # cprint(f"Target to {target_pose}", "yellow")
                # execute teleop command
                env.exec_actions(
                    actions=[np.append(pose_euler2quat(target_pose), G_target_pose)],
                    timestamps=[t_command_target - time.monotonic() + time.time()],
                    delta_actions=[
                        np.append(np.concatenate((dpos, drot_xyz)), G_target_pose)
                    ],
                    stages=[stage],
                )
                precise_wait(t_cycle_end)
                iter_idx += 1


if __name__ == "__main__":
    main()
