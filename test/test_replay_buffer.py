import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)


import zarr
from common.replay_buffer import ReplayBuffer


def test():
    import numpy as np

    buff = ReplayBuffer.create_empty_numpy()
    buff.add_episode({"obs": np.zeros((100, 10), dtype=np.float16)})
    buff.add_episode({"obs": np.ones((50, 10)), "action": np.ones((50, 2))})
    # buff.rechunk(256)
    obs = buff.get_episode(0)

    import numpy as np

    buff = ReplayBuffer.create_empty_zarr()
    buff.add_episode({"obs": np.zeros((100, 10), dtype=np.float16)})
    buff.add_episode({"obs": np.ones((50, 10)), "action": np.ones((50, 2))})
    obs = buff.get_episode(0)
    buff.set_chunks({"obs": (100, 10), "action": (100, 2)})


def test_real():
    dist_group = zarr.open(
        os.path.expanduser(
            "/home/shawn/Documents/pyspacemouse-coppeliasim/data/pusht/pusht_cchi_v7_replay.zarr"
        ),
        "r",
    )

    buff = ReplayBuffer.create_empty_numpy()
    key, group = next(iter(dist_group.items()))
    for key, group in dist_group.items():
        buff.add_episode(group)

    # out_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht_cchi2_v2_replay.zarr')
    out_path = os.path.expanduser(
        "/home/shawn/Documents/pyspacemouse-coppeliasim/data/pusht/test.zarr"
    )
    out_store = zarr.DirectoryStore(out_path)
    buff.save_to_store(out_store)

    buff = ReplayBuffer.copy_from_path(out_path, store=zarr.MemoryStore())
    buff.pop_episode()


def test_pop():
    buff = ReplayBuffer.create_from_path(
        "/home/shawn/Documents/pyspacemouse-coppeliasim/data/pusht/pusht_cchi_v7_replay.zarr",
        mode="rw",
    )


def test_save():
    RB = ReplayBuffer.create_empty_numpy()
    output = "/home/shawn/Documents/pyspacemouse-coppeliasim/data/pusht/test_save.zarr"
    with zarr.DirectoryStore(output) as store:
        RB.save_to_store(store=store)


if __name__ == "__main__":
    test_save()
