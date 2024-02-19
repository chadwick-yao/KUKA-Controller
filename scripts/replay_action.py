import hydra
import time
import numpy as np
import scipy.spatial.transform as st
from tqdm import tqdm
from omegaconf import OmegaConf

import pathlib
import sys
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from codebase.real_world.iiwaPy3 import IIWAPositionalController
from codebase.real_world.robotiq85 import Robotiq85
from multiprocessing.managers import SharedMemoryManager

np.set_printoptions(precision=2, suppress=True, linewidth=100)
OmegaConf.register_new_resolver("eval", eval, replace=True)

with hydra.initialize("../codebase/diffusion_policy/config"):
    cfg = hydra.compose("train_diffusion_transformer_hybrid_workspace")
    OmegaConf.resolve(cfg)
    dataset = hydra.utils.instantiate(cfg.task.dataset)

print(dataset.replay_buffer.root.tree())
episode_end = dataset.replay_buffer.root["meta"]["episode_ends"]
actions_set = dataset.replay_buffer.root["data"]["action"][
    episode_end[2] : episode_end[3]
]

shm_manager = SharedMemoryManager()
shm_manager.start()

REMOTER = IIWAPositionalController(
    shm_manager=shm_manager,
    host="172.31.1.147",
    port=30001,
    receive_keys=None,
    max_pos_speed=128,
    max_rot_speed=0.8,
)
gripper = Robotiq85(shm_manager=shm_manager, frequency=100, receive_keys=None)
Open = True
last = 0

time.sleep(1)

REMOTER.reset_initial_state()
init_eef_pos = REMOTER.getEEFPos()
REMOTER.realTime_startDirectServoCartesian()
time.sleep(3)
try:
    for delta_act in actions_set:
        dpos, drot_xyz = delta_act[:3], delta_act[3:6] * np.array([1, 1, 1])
        drot = st.Rotation.from_euler("xyz", drot_xyz)
        init_eef_pos[:3] += dpos
        init_eef_pos[3:] = (
            drot * st.Rotation.from_euler("zyx", init_eef_pos[3:])
        ).as_euler("zyx")

        if last != delta_act[6]:
            if Open:
                Open = False
                gripper.close()
            else:
                Open = True
                gripper.open()
        _ = REMOTER.sendEEfPositionGetActualEEFpos(init_eef_pos)
        time.sleep(1 / 10)
        last = delta_act[6]
        print(delta_act[6])
finally:
    REMOTER.realTime_stopDirectServoCartesian()
    REMOTER.reset_initial_state()
    REMOTER.close()
