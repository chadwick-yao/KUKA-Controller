from typing import Dict, Tuple, List
import math
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from codebase.diffusion_policy.model.common.normalizer import LinearNormalizer
from codebase.diffusion_policy.model.ACT.detr_vae import DETRVAE, build
from codebase.diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from codebase.diffusion_policy.policy.base_image_policy import BaseImagePolicy
from codebase.diffusion_policy.model.ACT.backbone import Joiner
from codebase.diffusion_policy.model.ACT.transformer import (
    Transformer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from codebase.diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from codebase.diffusion_policy.common.robomimic_config_util import get_robomimic_config

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import codebase.diffusion_policy.model.vision.crop_randomizer as dmvc

import logging

logger = logging.getLogger(__name__)


# %%
class ActionChunkTransformerPolicy(BaseImagePolicy):
    def __init__(
        self,
        # backbone
        joiner: Joiner,
        use_dp_vis: bool,
        # DETR arch
        num_encoder_layers: int,
        trans_encoder_layer: TransformerEncoderLayer,
        transformer: Transformer,
        # hyper params
        num_queries: int,
        kl_weight: float,
        latent_dim: int,
        hidden_dim: int,
        ## obs encoder params
        shape_meta: Dict,
        n_action_steps: int,
        n_obs_steps: int,
        crop_shape: tuple = (76, 76),
        obs_encoder_group_norm: bool = False,
        eval_fixed_crop: bool = False,
        temporal_agg: bool = False,
        **kwargs,
    ):
        super().__init__()
        backbones = []
        joiner.num_channels = joiner[0].num_channels
        backbones.append(joiner)
        encoder_norm = (
            nn.LayerNorm(trans_encoder_layer.d_model)
            if trans_encoder_layer.normalize_before
            else None
        )
        transformerEncoder = TransformerEncoder(
            trans_encoder_layer, num_encoder_layers, encoder_norm
        )

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        obs_shape_meta = shape_meta["obs"]
        obs_config = {
            "low_dim": [],
            "rgb": [],
            "depth": [],
            "scan": [],
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = shape

            obs_type = attr.get("type", "low_dim")

            if obs_type in obs_config.keys():
                obs_config[obs_type].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type {obs_type}.")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name="bc_rnn", hdf5_type="image", task_name="square", dataset_type="ph"
        )

        with config.unlocked():
            # set config with shape_meta
            obs_config_only_rgb = copy.deepcopy(obs_config)
            for key in obs_config.keys():
                if key != "rgb":
                    obs_config_only_rgb[key] = []
            config.observation.modalities.obs = obs_config_only_rgb

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device="cpu",
        )

        obs_encoder = policy.nets["policy"].nets["encoder"].nets["obs"]

        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets

        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc,
                ),
            )
        obs_feature_dim = obs_encoder.output_shape()[0]
        cond_dim = obs_feature_dim

        model = DETRVAE(
            backbones=None if use_dp_vis else backbones,
            transformer=transformer,
            encoder=transformerEncoder,
            num_queries=num_queries,
            cond_dim=cond_dim,
            low_dim=sum([obs_key_shapes[key][0] for key in obs_config["low_dim"]]),
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            n_obs_steps=n_obs_steps,
            latent_dim=latent_dim,
        )

        self.obs_encoder = obs_encoder if use_dp_vis else None
        self.model: DETRVAE = model
        self.obs_config = obs_config
        self.kl_weight = kl_weight
        self.num_queries = num_queries
        self.action_dim = action_dim
        self.temporal_agg = temporal_agg
        self.normalizer = LinearNormalizer()
        self.query_frequency = 1 if self.temporal_agg else self.num_queries
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.use_dp_vis = use_dp_vis

        if use_dp_vis:
            logger.info(
                "number of obs encoder parameters: %e",
                sum(p.numel() for p in self.obs_encoder.parameters()),
            )
        else:
            logger.info(
                "number of obs encoder parameters: %e",
                sum(p.numel() for p in self.model.backbones.parameters()),
            )

        logger.info(
            "number of cvae encoder parameters: %e",
            sum(p.numel() for p in self.model.encoder.parameters()),
        )

        logger.info(
            "number of cvae decoder parameters: %e",
            sum(p.numel() for p in self.model.transformer.parameters()),
        )

    def predict_action(self, obs_dict):
        nobs = self.normalizer.normalize(obs_dict)

        To = self.n_obs_steps
        Ta = self.n_action_steps
        value = next(iter(nobs.values()))
        B = value.shape[0]

        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))

        # img
        if self.use_dp_vis:
            img_data = dict()
            for item in self.obs_config["rgb"]:
                img_data[item] = this_nobs[item]

            # extract img features
            nobs_features = self.obs_encoder(img_data)
            img_data = nobs_features.reshape(B, To, -1)
        else:
            img_data = list()
            for item in self.obs_config["rgb"]:
                img_data.append(this_nobs[item])
            img_data = torch.stack(img_data, dim=1)  # bs*To, 2, ...
        # low
        low_data = list()
        for item in self.obs_config["low_dim"]:
            low_data.append(this_nobs[item])

        low_data = torch.cat(low_data, dim=1)  # bs*To, 9
        low_data = low_data.reshape(B, To, -1)  # bs, To, 9

        a_hat, _, (_, _) = self.model(low_data, img_data)
        action_pred = self.normalizer["action"].unnormalize(a_hat)
        start = To - 1
        end = start + Ta
        action = action_pred[:, start:end]

        result = {
            "action": action,
            "action_pred": action_pred,
        }
        return result

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        this_nobs = dict_apply(
            nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
        )  # bs*To, ...
        # img
        # img_data = dict()
        # for item in self.obs_config["rgb"]:
        #     img_data[item] = this_nobs[item]

        # # extract img features
        # nobs_features = self.obs_encoder(img_data)
        # img_cond = nobs_features.reshape(batch_size, To, -1)
        img_data = list()
        for item in self.obs_config["rgb"]:
            img_data.append(this_nobs[item])
        img_data = torch.stack(img_data, dim=1)  # bs*To, 2, ...
        # low
        low_data = list()
        for item in self.obs_config["low_dim"]:
            low_data.append(this_nobs[item])
        low_data = torch.cat(low_data, dim=1)  # bs*To, 9
        low_data = low_data.reshape(batch_size, To, -1)  # bs, To, 9

        is_pad = torch.zeros(nactions.shape[1]).bool()
        is_pad = is_pad.to(nactions.device)
        is_pad = torch.unsqueeze(is_pad, axis=0).repeat(batch_size, 1)

        a_hat, is_pad_hat, (mu, logvar) = self.model(
            low_data, img_data, nactions, is_pad
        )
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        loss_dict = dict()
        all_l1 = F.l1_loss(nactions, a_hat, reduction="none")
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        loss_dict["l1"] = l1
        loss_dict["kl"] = total_kld[0]
        loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
        return loss_dict

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
        self, weight_decay: float, learning_rate: float, lr_backbone: float
    ) -> torch.optim.Optimizer:
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=learning_rate, weight_decay=weight_decay
        )

        return optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
