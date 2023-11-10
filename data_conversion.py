from codebase.real_world.real_data_conversion import real_data_to_replay_buffer

import imageio
from einops import rearrange

if __name__ == "__main__":
    out_replay_buffer = real_data_to_replay_buffer(
        dataset_path="data/test_data",
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

    # imageio.mimsave(
    #     "output.mp4",
    #     rearrange(out_replay_buffer.root["data"]["camera_0"][:], "b w h c -> b c h w"),
    # )
