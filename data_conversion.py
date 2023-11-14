from codebase.real_world.real_data_conversion import real_data_to_replay_buffer

import imageio
from einops import rearrange
import numpy as np

np.set_printoptions(precision=2, suppress=True, linewidth=100)

if __name__ == "__main__":
    out_replay_buffer = real_data_to_replay_buffer(
        dataset_path="/media/shawn/Yiu1/side_view",
        out_resolutions=(640, 480),
        lowdim_keys=[
            "action",
            "gripper_pose",
            "robot_eef_pos",
            "robot_eef_rot",
            "robot_joint",
        ],
    )
    print(out_replay_buffer.root.tree())
    episode_end = out_replay_buffer.root["meta"]["episode_ends"][:]
    # delta_actions_set = out_replay_buffer.root["data"]["action"][
    #     episode_end[0] : episode_end[1]
    # ]
    # for item in delta_actions_set:
    #     print(item)

    imageio.mimsave(
        "output_0.mp4",
        rearrange(
            out_replay_buffer.root["data"]["camera_0"][: episode_end[0]],
            "b w h c -> b w h c",
        ),
    )
    imageio.mimsave(
        "output_1.mp4",
        rearrange(
            out_replay_buffer.root["data"]["camera_1"][: episode_end[0]],
            "b w h c -> b w h c",
        ),
    )
