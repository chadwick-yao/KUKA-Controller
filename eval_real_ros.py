import time
import copy
import click
import cv2
import torch
import dill
import hydra
import pathlib
import skvideo.io

import numpy as np
import multiprocessing as ml
import scipy.spatial.transform as st

from einops import repeat
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
from utils.data_utils import pose_euler2quat

import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from arm_control.msg import JointInformation
from arm_control.msg import JointControl
from arm_control.msg import PosCmd
from arm_control.msg import PosCmdWithHeader
from sensor_msgs.msg import Image
from threading import Lock
from cv_bridge import CvBridge

OmegaConf.register_new_resolver("eval", eval, replace=True)
np.set_printoptions(suppress=True)
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
    required=True,
    help="Path to checkpoint",
)
@click.option(
    "--output_path",
    "-op",
    required=True,
    help="Directory to save recording",
)
@click.option(
    "--max_duration", "-md", default=5, help="Max duration for each epoch in seconds."
)
@click.option(
    "--steps_per_inference",
    "-si",
    default=6,
    type=int,
    help="Action horizon for inference.",
)
# @profile
def main(
    input_path,
    output_path,
    frequency,
    max_duration,
    steps_per_inference,
):

    # load checkpoint
    ckpt_path = input_path
    payload = torch.load(open(ckpt_path, "rb"), map_location='cpu', pickle_module=dill)
    cfg = payload["cfg"]
    cfg._target_ = "codebase." + cfg._target_
    cfg.policy._target_ = "codebase." + cfg.policy._target_
    cfg.ema._target_ = "codebase." + cfg.ema._target_

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # policy
    policy: BaseImagePolicy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    device = torch.device("cuda:0")
    policy.eval().to(device)
    policy.reset()

    ## set inference params
    policy.num_inference_steps = 16  # DDIM inference iterations
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)

    bridge = CvBridge()
    
    rospy.init_node("eval_real_ros")
    obs_robot1 = Subscriber("follow1_pos_back", PosCmdWithHeader)
    # obs_robot2 = Subscriber("follow1_pos_back", PosCmdWithHeader)
    # image_global = Subscriber("mid_camera", Image)
    image_robot1 = Subscriber("left_camera", Image)
    # image_robot2 = Subscriber("right_camera", Image)
    control_robot1 = rospy.Publisher("follow_pos_cmd_1", PosCmd, queue_size=10)
    # control_robot2 = rospy.Publisher("follow_pos_cmd_2", PosCmd, queue_size=10)
    msg_queue = ml.Queue()
    
    ats = ApproximateTimeSynchronizer([obs_robot1, image_robot1, msg_queue], queue_size=10, slop=0.1)
    ats.registerCallback(callback)
    rate = rospy.Rate(frequency)

    while not rospy.is_shutdown():
        obs_data = get_observations(msg_queue, n_obs_steps, obs_res)

        if obs_data:
            while not msg_queue.empty():
                msg_queue.get()

            # run inference
            with torch.no_grad():
                s = time.time()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs_data, shape_meta=cfg.task.shape_meta
                )
                obs_dict = dict_apply(
                    obs_dict_np,
                    lambda x: torch.from_numpy(x)
                    .unsqueeze(0)
                    .to(device),
                )

                for k, v in obs_dict.items():
                    if len(v.shape) == 2:
                        obs_dict[k] = torch.unsqueeze(v, 2)
                result = policy.predict_action(obs_dict)
                # this action starts from the first obs step
                action = (
                    result["action"][0].detach().to("cpu").numpy()
                )  # 1 n_acts 7 -> n_acts 7

            for i in range(action.shape[0]):
                control_msg = PosCmd()
                control_msg.x = action[i][0]
                control_msg.y = action[i][1]
                control_msg.z = action[i][2]
                control_msg.roll = action[i][3]
                control_msg.pitch = action[i][4]
                control_msg.yaw = action[i][5]
                control_msg.gripper = action[i][6]
                control_robot1.publish(control_msg)   
                rate.sleep()

        rate.sleep()
                
                
def get_observations(msg_queue, obs_dim, obs_res):
    """
    Fetches and processes observations from a message queue.

    :param msg_queue: The message queue to fetch observations from.
    :param obs_dim: The number of observations to fetch.
    :param bridge: The image conversion bridge.
    :param obs_res: The resolution of the camera image.
    :return: A dictionary with processed observations.
    """
    bridge = CvBridge()
    obs_data = dict()
    if msg_queue.qsize() >= obs_dim:
        eef_pose = np.empty((0, 6))
        gripper_pose = np.empty((0, 1))
        camera_rgb = np.empty((0,) + obs_res + (3,))
        obs_timestamp = np.empty((0, 1))

        for _ in range(obs_dim):
            obs_msg = msg_queue.get()
            obs_robot1_msg = obs_msg['obs_robot1']
            image_robot1_msg = obs_msg['image_robot1']
            timestamp = obs_msg['time']
            
            # Append the end-effector pose
            eef_pose = np.vstack((eef_pose, np.array(
                [obs_robot1_msg.x, obs_robot1_msg.y, obs_robot1_msg.z, 
                 obs_robot1_msg.roll, obs_robot1_msg.pitch, obs_robot1_msg.yaw], 
                dtype=np.float32
            )))
            
            # Append the gripper pose
            gripper_pose = np.vstack((gripper_pose, np.array([obs_robot1_msg.gripper], dtype=np.float32)))
            
            # Convert and append the camera image
            image_robot1_cv2 = bridge.imgmsg_to_cv2(image_robot1_msg, "bgr8")
            camera_rgb = np.stack((camera_rgb, image_robot1_cv2))
            
            # Append the timestamp
            obs_timestamp = np.vstack((obs_timestamp, timestamp))
            
        # Update the observation data dictionary
        obs_data.update(
            {
                "eef_pose": eef_pose,
                "gripper_pose": gripper_pose,
                "camera_rgb": camera_rgb,
                "timestamp": obs_timestamp,
            }
        )
        
    return obs_data

def callback(obs_robot1, image_robot1, msg):

    time = rospy.Time.now()
    obs_data = dict()
    obs_data['obs_robot1'] = obs_robot1
    # obs_data['obs_robot2'] = obs_robot2
    # obs_data['image_global'] = image_global
    obs_data['image_robot1'] = image_robot1
    # obs_data['image_robot2'] = image_robot2
    obs_data['time'] = time
    
    msg.put(obs_data)

if __name__ == "__main__":
    main()
