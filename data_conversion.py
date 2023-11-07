from codebase.real_world.real_data_conversion import real_data_to_replay_buffer


if __name__ == "__main__":
    out_replay_buffer = real_data_to_replay_buffer(
        dataset_path="data/demo_data",
        out_resolutions=(480, 640),
        lowdim_keys=["gripper_pose", "robot_eef_pose", "robot_joint"],
    )
    print(out_replay_buffer.root.tree())
