"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip> --robot_port <port>
e.g python demo_real_robot.py -o "data/demo_data" --robot_ip "172.31.1.147" --robot_port 30001

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
import numpy as np
import scipy.spatial.transform as st
from multiprocessing.managers import SharedMemoryManager
from codebase.real_world.real_env import RealEnv
from common.spacemouse_shared_memory import Spacemouse
from common.precise_sleep import precise_wait
from common.keystroke_counter import KeystrokeCounter, Key, KeyCode


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
    "--command_latency",
    "-cl",
    default=0.01,
    type=float,
    help="Latency between receiving SapceMouse command to executing on Robot in Sec.",
)
def main(output, robot_ip, robot_port, vis_camera_idx, frequency, command_latency):
    dt = 1 / frequency

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, Spacemouse(
            shm_manager=shm_manager,
            get_max_k=30,
            frequency=200,
        ) as sm, RealEnv(
            output_dir=output,
            robot_ip=robot_ip,
            robot_port=robot_port,
            # recording resolution
            obs_image_resolution=(1280, 720),
            frequency=frequency,
            enable_multi_cam_vis=True,
            record_raw_video=True,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager,
        ) as env:
            cv2.setNumThreads(1)

            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)

            time.sleep(1.0)
            print("Ready!")
            state = env.get_robot_state()
            target_pose = state["EEFPos"]

            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False

            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()

                # handle key presses
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
                        print("Recording!")
                    elif key_stroke == KeyCode(char="s"):
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print("Stopped.")
                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm("Are you sure to drop an episode?"):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False

                stage = key_counter[Key.space]

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
                dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)

                if not sm.is_button_pressed(0):
                    # translation mode
                    drot_xyz[:] = 0
                else:
                    dpos[:] = 0
                if not sm.is_button_pressed(1):
                    # TODO: gripper control
                    dpos[2] = 0

                drot = st.Rotation.from_euler("xyz", drot_xyz)
                target_pose[:3] += dpos
                target_pose[3:] = (
                    drot * st.Rotation.from_rotvec(target_pose[3:])
                ).as_rotvec()

                # execute teleop command
                env.exec_actions(
                    actions=[target_pose],
                    timestamps=[t_command_target - time.monotonic() + time.time()],
                    stages=[stage],
                )
                precise_wait(t_cycle_end)
                iter_idx += 1


if __name__ == "__main__":
    main()
