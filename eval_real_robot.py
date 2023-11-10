import time
import click
import cv2
import torch
import dill
import hydra
import pathlib
import skvideo.io

import numpy as np
import scipy.spatial.transform as st

from ipdb import set_trace
from omegaconf import OmegaConf
from termcolor import colored, cprint
from multiprocessing.managers import SharedMemoryManager

from common.spacemouse_shared_memory import Spacemouse
from common.precise_sleep import precise_sleep, precise_wait

from codebase.real_world.real_env import RealEnv
from codebase.diffusion_policy.workspace.base_workspace import BaseWorkspace
from codebase.diffusion_policy.policy.base_image_policy import BaseImagePolicy
from codebase.diffusion_policy.common.pytorch_util import dict_apply

from utils.cv2_utils import get_image_transform
from utils.real_inference_utils import get_real_obs_dict, get_real_obs_resolution

OmegaConf.register_new_resolver("eval", eval, replace=True)
"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""


@click.command()
@click.option(
    "--input_path",
    "-ip",
    default="/media/shawn/Yiu1/19.58.34_train_diffusion_transformer_hybrid_real_lift_image/checkpoints/latest.ckpt",
    required=True,
    help="Path to checkpoint",
)
@click.option(
    "--output_path",
    "-op",
    default="/home/shawn/Documents/pyspacemouse-coppeliasim/data/demo_data1",
    required=True,
    help="Directory to save recording",
)
@click.option(
    "--robot_ip",
    "-ri",
    default="172.31.1.147",
    required=True,
    help="Robot's IP address. e.g. 172.31.1.147",
)
@click.option(
    "--frequency", "-f", default=10, type=int, help="Control frequency in Hz."
)
@click.option(
    "--command_latency",
    "-cl",
    default=0.01,
    type=float,
    help="Latency between receiving SapceMouse command to executing on Robot in Sec.",
)
@click.option(
    "--max_duration", "-md", default=60, help="Max duration for each epoch in seconds."
)
@click.option(
    "--steps_per_inference",
    "-si",
    default=6,
    type=int,
    help="Action horizon for inference.",
)
@click.option(
    "--vis_camera_idx", default=0, type=int, help="Which RealSense camera to visualize."
)
@click.option(
    "--match_episode",
    "-me",
    default=None,
    type=int,
    help="Match specific episode from the match dataset",
)
@click.option(
    "--match_dataset",
    "-m",
    default=None,
    help="Dataset used to overlay and adjust initial condition",
)
@click.option(
    "--pos_sensitivity",
    "-ps",
    default=8.0,
    type=float,
    help="Position control sensitivity.",
)
@click.option(
    "--rot_sensitivity",
    "-rs",
    default=1.0,
    type=float,
    help="Rotation control sensitivity.",
)
def main(
    input_path,
    output_path,
    robot_ip,
    frequency,
    command_latency,
    max_duration,
    steps_per_inference,
    vis_camera_idx,
    match_episode,
    match_dataset,
    pos_sensitivity,
    rot_sensitivity,
):
    # load match_dataset

    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath("videos")
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f"{match_camera_idx}.mp4")
            if match_video_path.exists():
                frames = skvideo.io.vread(str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

    # load chechpoint
    ckpt_path = input_path
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cfg._target_ = "codebase.diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace.TrainDiffusionTransformerHybridWorkspace"
    cfg.policy._target_ = "codebase.diffusion_policy.policy.diffusion_transformer_hybrid_image_policy.DiffusionTransformerHybridImagePolicy"
    cfg.ema._target_ = "codebase.diffusion_policy.model.diffusion.ema_model.EMAModel"

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    # policy
    action_offset = 0
    delta_action = False

    policy: BaseImagePolicy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    device = torch.device("cuda:0")
    policy.eval().to(device)

    ## set inference params
    policy.num_inference_steps = 16  # DDIM inference iterations
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    # setup robot
    dt = 1 / frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm, RealEnv(
            output_dir=output_path,
            robot_ip=robot_ip,
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=True,
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
            env.realsense.set_exposure(exposure=200, gain=10)
            # realsense white balance
            # env.realsense.set_white_balance(white_balance=5900)

            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            # get current observation
            obs = env.get_obs()

            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs,
                    shape_meta=cfg.task.shape_meta,
                )
                obs_dict = dict_apply(
                    obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
                )
                for k, v in obs_dict.items():
                    if len(v.shape) == 2:
                        obs_dict[k] = torch.unsqueeze(v, 2)
                result = policy.predict_action(obs_dict)
                action = result["action"][0].detach().to("cpu").numpy()
                print(f"actions shape: {action.shape}")
                assert action.shape[-1] == 7
                del result

            cprint("Ready!", on_color="on_green")
            while True:
                cprint("Human in control!", color="yellow")
                state = env.get_robot_state()
                target_pose = np.append(state["EEFpos"], state["EEFrot"])
                t_start = time.monotonic()
                iter_idx = 0
                last_button = [False, False]
                G_target_pose = 0  # open
                while True:
                    # caculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f"camera_{vis_camera_idx}"][-1]
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), output_res=(ow, oh), bgr_to_rgb=False
                        )
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = np.minimum(vis_img, match_img)

                    text = f"Episode: {episode_id}"
                    cv2.putText(
                        vis_img,
                        text,
                        (10, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255, 255, 255),
                    )

                    cv2.imshow("default", vis_img[..., ::-1])
                    key_stroke = cv2.pollKey()
                    if key_stroke == ord("q"):
                        # Exit program
                        env.end_episode()
                        exit(0)
                    elif key_stroke == ord("c"):
                        # Exit human control loop
                        # hand control over to the policy
                        break

                    precise_wait(t_sample)
                    # get teleop command
                    sm_state = sm.get_motion_state_transformed()
                    # print(sm_state)
                    dpos = (
                        sm_state[:3] * (env.max_pos_speed / frequency) * pos_sensitivity
                    )
                    drot_xyz = (
                        sm_state[3:]
                        * (env.max_rot_speed / frequency)
                        * np.array([1, 1, -1])
                        * rot_sensitivity
                    )

                    # ------------- Button Features -------------
                    current_button = [sm.is_button_pressed(0), sm.is_button_pressed(1)]
                    if not current_button[0]:
                        # translation mode
                        drot_xyz[:] = 0
                    else:
                        dpos[:] = 0
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
                        actions=[np.append(target_pose, G_target_pose)],
                        timestamps=[t_command_target - time.monotonic() + time.time()],
                    )
                    precise_wait(t_cycle_end)
                    iter_idx += 1

                # ========== policy control loop ==============
                try:
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1 / 30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    cprint("Started!", "yellow")

                    iter_idx = 0
                    term_area_start_timestamp = float("inf")
                    perv_target_pose = None
                    while True:
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get observations
                        cprint("Get Obs!", color="blue")
                        obs = env.get_obs()
                        obs_timestamps = obs["timestamp"]
                        print(f"Obs latency {time.time() - obs_timestamps[-1]}")

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta
                            )
                            obs_dict = dict_apply(
                                obs_dict_np,
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device),
                            )
                            for k, v in obs_dict.items():
                                if len(v.shape) == 2:
                                    obs_dict[k] = torch.unsqueeze(v, 2)
                            result = policy.predict_action(obs_dict)
                            # this action starts from the first obs step
                            action = result["action"][0].detach().to("cpu").numpy()
                            print("Inference latency:", time.time() - s)

                        # convert policy action to env actions
                        action = action[:steps_per_inference, :]
                        if delta_action:
                            assert len(action) == 1
                            if perv_target_pose is None:
                                perv_target_pose = np.append(
                                    obs["robot_eef_pos"][-1],
                                    obs["robot_eef_rot"][-1],
                                    obs["gripper_pose"][-1],
                                )
                            this_target_pose = perv_target_pose.copy()
                            this_target_pose[:6] += action[-1][:6]
                            this_target_pose[6] = action[-1][6]
                            perv_target_pose = this_target_pose
                            this_target_poses = np.expand_dims(this_target_pose, axis=0)
                        else:
                            this_target_poses = np.zeros(
                                (action.shape), dtype=np.float64
                            )
                            this_target_poses[:] = action

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (
                            np.arange(len(action), dtype=np.float64) + action_offset
                        ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(
                                np.ceil((curr_time - eval_t_start) / dt)
                            )
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print("Over budget", action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # clip actions
                        # this_target_poses[:, :2] = np.clip(
                        #     this_target_poses[:, :2], [0.25, -0.45], [0.77, 0.40]
                        # )

                        this_target_poses[:, 6] = np.clip(
                            this_target_poses[:, 6], 0.0, 1.0
                        )
                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses, timestamps=action_timestamps
                        )
                        print(f"Submitted action shape: {this_target_poses.shape}")
                        print(f"Submitted action: {this_target_poses}")
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = obs[f"camera_{vis_camera_idx}"][-1]
                        text = "Episode: {}, Time: {:.1f}".format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255, 255, 255),
                        )
                        cv2.imshow("default", vis_img[..., ::-1])

                        key_stroke = cv2.pollKey()
                        if key_stroke == ord("s"):
                            # Stop episode
                            # Hand control back to human
                            env.end_episode()
                            print("Stopped.")
                            break

                        # auto termination
                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print("Terminated by the timeout!")

                        term_pose = np.array(
                            [
                                3.40948500e-01,
                                2.17721816e-01,
                                4.59076878e-02,
                                2.22014183e00,
                                -2.22184883e00,
                                -4.07186655e-04,
                            ]
                        )
                        curr_pose = np.concatenate(
                            (obs["robot_eef_pos"][-1], obs["robot_eef_rot"][-1])
                        )
                        dist = np.linalg.norm((curr_pose - term_pose)[:2], axis=-1)
                        if dist < 0.03:
                            # in termination area
                            curr_timestamp = obs["timestamp"][-1]
                            if term_area_start_timestamp > curr_timestamp:
                                term_area_start_timestamp = curr_timestamp
                            else:
                                term_area_time = (
                                    curr_timestamp - term_area_start_timestamp
                                )
                                if term_area_time > 0.5:
                                    terminate = True
                                    print("Terminated by the policy!")
                        else:
                            # out of the area
                            term_area_start_timestamp = float("inf")

                        if terminate:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()

                print("Stopped.")


if __name__ == "__main__":
    main()
