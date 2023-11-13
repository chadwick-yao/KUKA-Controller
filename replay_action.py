import hydra
import time
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from codebase.real_world.iiwaPy3 import IIWAPositionalController
from multiprocessing.managers import SharedMemoryManager

np.set_printoptions(precision=2, suppress=True, linewidth=100)
OmegaConf.register_new_resolver("eval", eval, replace=True)

with hydra.initialize("./codebase/diffusion_policy/config"):
    cfg = hydra.compose("train_diffusion_transformer_hybrid_workspace")
    OmegaConf.resolve(cfg)
    dataset = hydra.utils.instantiate(cfg.task.dataset)

print(dataset.replay_buffer.root.tree())
episode_end = dataset.replay_buffer.root["meta"]["episode_ends"][0]
actions_set = dataset.replay_buffer.root["data"]["action"][:episode_end]
eef_pos = dataset.replay_buffer.root["data"]["robot_eef_pos"][:episode_end]
eef_rot = dataset.replay_buffer.root["data"]["robot_eef_rot"][:episode_end]

eef_pose = np.concatenate((eef_pos, eef_rot), axis=1)

for idx in range(episode_end):
    print(actions_set[idx][:6], " | ", (actions_set[idx][:6] - eef_pose[idx]))
# shm_manager = SharedMemoryManager()
# shm_manager.start()

# REMOTER = IIWAPositionalController(
#     shm_manager=shm_manager, host="172.31.1.147", port=30001, receive_keys=None
# )
# REMOTER.reset_initial_state()
# init_eef_pos = REMOTER.getEEFPos()
# try:
#     REMOTER.realTime_startDirectServoCartesian()

#     # for next_eef_pos in actions_set:
#     #     print(f"Next EEF pose: {next_eef_pos[:6]}")
#     #     user_input = np.array(list(map(float, input("delta pos: ").split())))
#     #     init_eef_pos += user_input
#     #     print(f"Target Pos: {init_eef_pos}")
#     #     _ = REMOTER.sendEEfPositionGetActualEEFpos(init_eef_pos)
#     #     # _ = REMOTER.sendEEfPositionGetActualEEFpos(next_eef_pos[:6])
#     #     time.sleep(1 / 100)
# finally:
#     REMOTER.realTime_stopDirectServoCartesian()
#     REMOTER.reset_initial_state()
#     REMOTER.close()
