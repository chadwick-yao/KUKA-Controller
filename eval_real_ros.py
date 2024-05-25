import sys
sys.path.append("/home/dc/mambaforge/envs/robodiff/lib/python3.9/site-packages")
sys.path.append("/home/dc/Desktop/dp_ycw/follow_control/follow1/src/arm_control/scripts/KUKA-Controller")
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
from common.precise_sleep import precise_sleep, precise_wait

from codebase.diffusion_policy.diffusion_policy.workspace.base_workspace import BaseWorkspace
from codebase.diffusion_policy.diffusion_policy.policy.base_image_policy import BaseImagePolicy
from codebase.diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply

from utils.cv2_utils import get_image_transform
from utils.real_inference_utils import get_real_obs_dict, get_real_obs_resolution
from utils.data_utils import pose_euler2quat

import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from arm_control.msg import JointInformation
from arm_control.msg import JointControl
from arm_control.msg import PosCmd
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
    "--max_duration", "-md", default=500, help="Max duration for each epoch in seconds."
)
@click.option(
    "--steps_per_inference",
    "-si",
    default=8,
    type=int,
    help="Action horizon for inference.",
)
# @profile
def main(
    input_path,
    max_duration,
    steps_per_inference,
):
    global obs_ls, cnt, cfg, control_robot2, policy, max_steps, all_time_actions, ts, horizon
    ts = 0
    max_steps = max_duration
    # load checkpoint
    ckpt_path = input_path
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    cfg._target_ = "codebase.diffusion_policy." + cfg._target_
    cfg.policy._target_ = "codebase.diffusion_policy." + cfg.policy._target_
    cfg.ema._target_ = "codebase.diffusion_policy." + cfg.ema._target_

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # policy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    device = torch.device("cuda:0")
    policy.eval().to(device)
    policy.reset()
    all_time_actions = torch.zeros([max_duration, max_duration+policy.horizon, policy.action_dim]).cuda()
    ## set inference params
    policy.num_inference_steps = 12  # DDIM inference iterations
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    horizon = policy.n_action_steps

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    obs_ls = list()
    cnt = n_obs_steps

    rospy.init_node("eval_real_ros")
    eef_qpos = Subscriber("follow2_pos_back", PosCmd)
    qpos = Subscriber("joint_information2",JointInformation)
    mid = Subscriber("mid_camera", Image)
    right = Subscriber("right_camera", Image)
    control_robot2 = rospy.Publisher("test_right", JointControl, queue_size=10)
    ats = ApproximateTimeSynchronizer(
        [eef_qpos, qpos, mid, right], queue_size=10, slop=0.1
    )
    ats.registerCallback(callback)

    rospy.spin()

def callback(eef_qpos, qpos, image_mid,image_right):
    global obs_ls, cnt, cfg, control_robot2, policy, max_steps, all_time_actions, ts, horizon
    st=time.time()
    en=time.time()
    if ts < max_steps:
        if len(obs_ls) < cnt:
            print("Get Observation!")
            st=time.time()
            obs_data = dict()
            # prepreocess low-dim info
            obs_data["eef_qpos"] = np.array([eef_qpos.x, eef_qpos.y, eef_qpos.z, eef_qpos.roll, eef_qpos.pitch, eef_qpos.yaw, eef_qpos.gripper])
            obs_data["qpos"] = qpos.joint_pos

            mid = image_mid
            right = image_right
            # process images observation
            bridge = CvBridge()
            mid = bridge.imgmsg_to_cv2(mid, "bgr8")
            right = bridge.imgmsg_to_cv2(right, "bgr8")
            
            obs_data["mid"] = mid
            obs_data["right"] = right

            # canvas = np.zeros((480, 1280, 3), dtype=np.uint8)

            # # 将图像复制到画布的特定位置
            # canvas[:, :640, :] = mid
            # canvas[:, 640:1280, :] = right

            # # 在一个窗口中显示排列后的图像
            # cv2.imshow('Multi Camera Viewer', canvas)
            # if cv2.waitKey(1) == ord("q"):
            #     pass

            obs_ls.append(obs_data)
        else:
            print("Exe Actions")
            obs = merge_obs_dict(obs_ls)
            with torch.no_grad():
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta
                )
                obs_dict = dict_apply(
                    obs_dict_np,
                    lambda x: torch.from_numpy(x)
                    .unsqueeze(0)
                    .to(torch.device("cuda:0")),
                )

                result = policy.predict_action(obs_dict)
                action = (
                    result["action"][0].detach()
                )

            right_control = JointControl()
            all_time_actions[[ts], ts:ts+horizon] = action
            # actions_for_curr_step = all_time_actions[:, ts]
            # actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            # actions_for_curr_step = actions_for_curr_step[actions_populated]
            # k = 0.01
            # exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            # exp_weights = exp_weights / exp_weights.sum()
            # exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
            # raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            # print(raw_action[0].to("cpu").numpy())
            right_control.joint_pos = action[0].to("cpu").numpy()
            control_robot2.publish(right_control)
            print(time.time() - st)
            obs_ls = []


def merge_obs_dict(dict_ls):
    eef_ls = []
    qpos_ls = []
    mid_ls = []
    right_ls = []
    for item in dict_ls:
        eef_ls.append(item["eef_qpos"])
        qpos_ls.append(item["qpos"])
        mid_ls.append(item["mid"])
        right_ls.append(item["right"])

    obs_dict = {
        "eef_qpos": np.array(eef_ls),
        "qpos": np.array(qpos_ls),
        "mid": np.array(mid_ls),
        "right": np.array(right_ls),
    }

    return obs_dict

if __name__ == "__main__":
    main()
