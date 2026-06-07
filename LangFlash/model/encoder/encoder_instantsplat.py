from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from .type_def import LangGaussians,Gaussians

inf = float('inf')

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int

@dataclass
class EncoderNoPoSplatCfg:
    name:  str
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True
    predict_opacity: bool=False




def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderInstantSplat(Encoder[EncoderNoPoSplatCfg:]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderNoPoSplatCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity

        self.gs_params_head_type = cfg.gs_params_head_type

        self.set_center_head(output_mode='pts3d', head_type='dpt', landscape_only=True,
                           depth_mode=('exp', -inf, inf), conf_mode=None,)
        self.set_gs_params_head(cfg, cfg.gs_params_head_type)

    def set_center_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode):
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))

        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def set_gs_params_head(self, cfg, head_type):
        if head_type == 'linear':
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.backbone.dec_embed_dim,
                    cfg.num_surfaces * self.patch_size ** 2 * self.raw_gs_dim,
                ),
            )

            self.gaussian_param_head2 = deepcopy(self.gaussian_param_head)
        elif head_type == 'dpt':
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)  # for view1 3DGS
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)  # for view2 3DGS

        elif head_type == 'dpt_gs':
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
            self.scratch_head1 = head_factory('scratch', 'scratch', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
            self.scratch_head2 = head_factory('scratch', 'scratch', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
        else:
            raise NotImplementedError(f"unexpected {head_type=}")

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape, ray_embedding=ray_embedding)

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # Encode the context images.
        dec1, dec2, shape1, shape2, view1, view2 = self.backbone(context, return_views=True)
        with torch.amp.autocast(device_type='cuda',enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

            # for the 3DGS heads
            if self.gs_params_head_type == 'linear':
                GS_res1 = rearrange_head(self.gaussian_param_head(dec1[-1]), self.patch_size, h, w)
                GS_res2 = rearrange_head(self.gaussian_param_head2(dec2[-1]), self.patch_size, h, w)
            elif self.gs_params_head_type == 'dpt':
                GS_res1 = self.gaussian_param_head([tok.float() for tok in dec1], shape1[0].cpu().tolist())
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                GS_res2 = self.gaussian_param_head2([tok.float() for tok in dec2], shape2[0].cpu().tolist())
                GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
            elif self.gs_params_head_type == 'dpt_gs':
                GS_res1,scratch1_gs = self.gaussian_param_head([tok.float() for tok in dec1], res1['pts3d'].permute(0, 3, 1, 2), view1['img'][:, :3], shape1[0].cpu().tolist())
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                GS_res2,scratch2_gs = self.gaussian_param_head2([tok.float() for tok in dec2], res2['pts3d'].permute(0, 3, 1, 2), view2['img'][:, :3], shape2[0].cpu().tolist())
                GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
                if context['sams'] is not None:
                    scratch1,add_1 = self.scratch_head1([tok.float() for tok in dec1], res1['pts3d'].permute(0, 3, 1, 2), view1['img'][:, :3], shape1[0].cpu().tolist(),sams = context['sams'][:,0,...])
                    scratch2,add_2 = self.scratch_head2([tok.float() for tok in dec2], res2['pts3d'].permute(0, 3, 1, 2), view2['img'][:, :3], shape2[0].cpu().tolist(),sams = context['sams'][:,1,...])
                else:
                    scratch1,add_1 = self.scratch_head1([tok.float() for tok in dec1], res1['pts3d'].permute(0, 3, 1, 2), view1['img'][:, :3], shape1[0].cpu().tolist(),sams = context['sams'])
                    scratch2,add_2 = self.scratch_head2([tok.float() for tok in dec2], res2['pts3d'].permute(0, 3, 1, 2), view2['img'][:, :3], shape2[0].cpu().tolist(),sams = context['sams'])
                scratch1 = rearrange(scratch1,"b d h w -> b (h w) d")
                scratch2 = rearrange(scratch2,"b d h w -> b (h w) d")
        scratch_all = torch.stack((scratch1, scratch2), dim=1)
        pts3d1 = res1['pts3d']
        pts3d1 = rearrange(pts3d1, "b h w d -> b (h w) d")
        pts3d2 = res2['pts3d']
        pts3d2 = rearrange(pts3d2, "b h w d -> b (h w) d")
        pts_all = torch.stack((pts3d1, pts3d2), dim=1)
        pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces

        depths = pts_all[..., -1].unsqueeze(-1)

        gaussians = torch.stack([GS_res1, GS_res2], dim=1)
        gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
        densities = gaussians[..., 0].sigmoid().unsqueeze(-1)

        # Convert the features and depths into Gaussians.
        if self.pose_free:
            gaussians = self.gaussian_adapter.forward(
                pts_all.unsqueeze(-2),
                depths,
                self.map_pdf_to_opacity(densities, global_step),
                rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
            )
        else:
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)

            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                self.map_pdf_to_opacity(densities, global_step),
                rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
                (h, w),
            )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
            visualization_dump["means"] = rearrange(
                gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
            )
            visualization_dump['opacities'] = rearrange(
                gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            
        add_dict = {}
        add_dict["coord"] = rearrange(pts_all,"b v r srf xyz -> b (v r srf) xyz")
        add_dict["org_feat"] = rearrange(scratch_all, "b v n f -> b (v n) f")
        add_dict["emb"] = scratch_all
        add_dict["dec1"] = [tok.float() for tok in dec1]
        add_dict["dec2"] = [tok.float() for tok in dec2]
        add_dict["add1"] = add_1
        add_dict["add2"] = add_2
        add_dict["add1gs"] = scratch1_gs
        add_dict["add2gs"] = scratch2_gs

        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        ), add_dict

from pathlib import Path
from omegaconf import OmegaConf

from LangFlash.model.encoder.backbone.backbone_croco import BackboneCrocoCfg
from LangFlash.model.encoder.backbone.backbone_dino import BackboneDinoCfg
from LangFlash.model.encoder.backbone.backbone_resnet import BackboneResnetCfg


def load_encoder_cfg(cfg_path: str):
    cfg_path = Path(cfg_path)
    cfg = OmegaConf.load(cfg_path)

    merged = OmegaConf.create()

    for item in cfg.get("defaults", []):
        for key, value in item.items():
            sub_cfg_path = cfg_path.parent / key / f"{value}.yaml"
            sub_cfg = OmegaConf.load(sub_cfg_path)
            merged[key] = sub_cfg

    cfg = OmegaConf.merge(merged, cfg)

    if "defaults" in cfg:
        del cfg["defaults"]

    return OmegaConf.to_container(cfg, resolve=True)


def build_encoder_cfg(cfg_dict: dict):
    cfg_dict = dict(cfg_dict)

    # ---------- simple nested dataclass ----------
    cfg_dict["opacity_mapping"] = OpacityMappingCfg(**cfg_dict["opacity_mapping"])
    cfg_dict["gaussian_adapter"] = GaussianAdapterCfg(**cfg_dict["gaussian_adapter"])
    cfg_dict["visualizer"] = EncoderVisualizerEpipolarCfg(**cfg_dict["visualizer"])

    # ---------- backbone dispatch ----------
    backbone_dict = cfg_dict["backbone"]
    backbone_name = backbone_dict["name"]

    backbone_map = {
        "croco": BackboneCrocoCfg,
        "croco_lang": BackboneCrocoCfg,
        "resnet": BackboneResnetCfg,
        "dino": BackboneDinoCfg,
    }

    if backbone_name not in backbone_map:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    cfg_dict["backbone"] = backbone_map[backbone_name](**backbone_dict)

    return EncoderNoPoSplatCfg(**cfg_dict)


def load_model_from_yaml(config_path: str):
    cfg_dict = load_encoder_cfg(config_path)
    encoder_cfg = build_encoder_cfg(cfg_dict)
    model = EncoderInstantSplat(encoder_cfg)
    return model, encoder_cfg


if __name__ == "__main__":
    config_path = "LangFlash/config/encoder/instantsplat.yaml"

    model, cfg = load_model_from_yaml(config_path)

    print("model created successfully")
    print(type(cfg.backbone))
    print(cfg.backbone.name)
    print(model.__class__.__name__)


