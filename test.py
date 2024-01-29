from common.replay_buffer import ReplayBuffer

zarr_path = "data/eval_pick_12_28_2/replay_buffer.zarr"
replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode="a")
print(replay_buffer.root.tree())