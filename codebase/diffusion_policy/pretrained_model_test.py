import torch
import hydra
import dill
import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from codebase.diffusion_policy.workspace.base_workspace import BaseWorkspace

if __name__ == "__main__":
    try:
        ckpt_path = "/media/shawn/Yiu1/19.58.34_train_diffusion_transformer_hybrid_real_lift_image/checkpoints/latest.ckpt"
        payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cfg._target_ = "codebase.diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace.TrainDiffusionTransformerHybridWorkspace"
        cfg.policy._target_ = "codebase.diffusion_policy.policy.diffusion_transformer_hybrid_image_policy.DiffusionTransformerHybridImagePolicy"
        cfg.ema._target_ = (
            "codebase.diffusion_policy.model.diffusion.ema_model.EMAModel"
        )
        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        device = torch.device("cuda:0")
        policy.eval().to(device)
        obs_dict = {
            "robot_eef_pos": torch.randn(32, 2, 3),
            "robot_eef_rot": torch.randn(32, 2, 3),
            "gripper_pose": torch.randn(32, 2, 1),
            "camera_0": torch.randn(32, 2, 3, 64, 64),
        }

        import hydra
        from tqdm import tqdm
        from omegaconf import OmegaConf
        from torch.utils.data import DataLoader

        OmegaConf.register_new_resolver("eval", eval, replace=True)

        with hydra.initialize("./config"):
            cfg = hydra.compose("train_diffusion_transformer_hybrid_workspace")
            OmegaConf.resolve(cfg)
            dataset = hydra.utils.instantiate(cfg.task.dataset)

        obs_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        first_batch = next(iter(obs_dataloader))
        with torch.no_grad():
            result = policy.predict_action(first_batch["obs"])
            action = result["action"][0].detach().to("cpu").numpy()
            print(action)
        print(first_batch["action"][0])
    except:
        raise RuntimeError
