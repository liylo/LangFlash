import json
from pathlib import Path
from typing import Tuple, Optional

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
from torchvision.utils import save_image

from .semantic.utils import generate_1d_sine_pos_encoding, postprocess_sam_masks
from .encoder.encoder_instantsplat import EncoderInstantSplat
from .semantic.seg import Fussion, Fussionx
from ..visualization.validation_in_3d import render_projections
from .decoder.decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg

from LangFlash.model.encoder.backbone.backbone_croco import BackboneCrocoCfg
from LangFlash.model.encoder.backbone.backbone_dino import BackboneDinoCfg
from LangFlash.model.encoder.backbone.backbone_resnet import BackboneResnetCfg
from LangFlash.model.encoder.encoder_instantsplat import (
    EncoderNoPoSplatCfg,
    OpacityMappingCfg,
    GaussianAdapterCfg,
    EncoderVisualizerEpipolarCfg,
)
from omegaconf import OmegaConf
import imageio
import numpy as np
import os

color = torch.rand((256, 3), device="cuda")
IND = "demo"

# -----------------------------
# Utils
# -----------------------------
def hstack_video_frames(left_frames: np.ndarray, right_frames: np.ndarray) -> np.ndarray:
    num_frames = min(len(left_frames), len(right_frames))
    left_frames = left_frames[:num_frames]
    right_frames = right_frames[:num_frames]

    left_h, left_w = left_frames.shape[1:3]
    right_h, right_w = right_frames.shape[1:3]

    if left_h != right_h:
        new_right_w = int(round(right_w * left_h / right_h))
        resized_right = []
        for frame in right_frames:
            frame_img = Image.fromarray(frame)
            frame_img = frame_img.resize((new_right_w, left_h), Image.BILINEAR)
            resized_right.append(np.asarray(frame_img, dtype=np.uint8))
        right_frames = np.stack(resized_right, axis=0)

    return np.concatenate([left_frames, right_frames], axis=2)


def process_pil(img: Image.Image, size=(256, 256)) -> torch.Tensor:
    img = img.convert("RGB")
    w, h = img.size
    crop_size = min(w, h)
    img = TF.center_crop(img, [crop_size, crop_size])
    img = TF.resize(img, size)
    img = TF.to_tensor(img) * 2 - 1
    return img


def tensor_to_pil(img):
    img = img.detach().cpu()
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def parse_intrinsics_from_inputs(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_w: int,
    image_h: int,
    normalized: bool = True,
) -> torch.Tensor:
    K = torch.eye(3, dtype=torch.float32)
    if normalized:
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        K = center_crop_then_resize_intrinsics(K, image_h, image_w, 256, 256)
    else:
        K[0, 0] = fx / image_w
        K[1, 1] = fy / image_h
        K[0, 2] = cx / image_w
        K[1, 2] = cy / image_h
        K = center_crop_then_resize_intrinsics(K, image_h, image_w, 256, 256)
    return K


def center_crop_then_resize_intrinsics(
    K: torch.Tensor,
    h_in: int,
    w_in: int,
    out_h: int = 256,
    out_w: int = 256,
) -> torch.Tensor:
    K = torch.as_tensor(K, dtype=torch.float32).clone()
    crop_size = min(h_in, w_in)
    top = (h_in - crop_size) // 2
    left = (w_in - crop_size) // 2
    fx = K[0, 0] * w_in
    fy = K[1, 1] * h_in
    cx = K[0, 2] * w_in
    cy = K[1, 2] * h_in
    cx -= left
    cy -= top
    sx = out_w / crop_size
    sy = out_h / crop_size
    fx *= sx
    fy *= sy
    cx *= sx
    cy *= sy
    K[0, 0] = fx / out_w
    K[1, 1] = fy / out_h
    K[0, 2] = cx / out_w
    K[1, 2] = cy / out_h
    return K


def parse_extrinsics(text: str) -> torch.Tensor:
    arr = json.loads(text)
    if len(arr) == 16:
        ext = torch.tensor(arr, dtype=torch.float32).reshape(4, 4)
        return ext
    if len(arr) == 12:
        w2c = torch.eye(4, dtype=torch.float32)
        w2c[:3, :] = torch.tensor(arr, dtype=torch.float32).reshape(3, 4)
        return torch.linalg.inv(w2c)
    raise ValueError("Extrinsics must contain 12 or 16 numbers.")


# -----------------------------
# Config & model loader
# -----------------------------
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
    cfg_dict["opacity_mapping"] = OpacityMappingCfg(**cfg_dict["opacity_mapping"])
    cfg_dict["gaussian_adapter"] = GaussianAdapterCfg(**cfg_dict["gaussian_adapter"])
    cfg_dict["visualizer"] = EncoderVisualizerEpipolarCfg(**cfg_dict["visualizer"])

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
    return build_encoder_cfg(cfg_dict)


# -----------------------------
# LangFlash wrapper
# -----------------------------
class LangFlash(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        decoder_cfg = DecoderSplattingCUDACfg(
            name="splatting_cuda",
            make_scale_invariant=False,
            background_color=[0.0, 0.0, 0.0],
        )
        self.decoder = DecoderSplattingCUDA(decoder_cfg)
        self.encoder = EncoderInstantSplat(cfg).to(self.device)
        self.encoder2 = EncoderInstantSplat(cfg).to(self.device)
        self.fuse = Fussion(depth=6, num_cond=1, dim=256).to(self.device)
        self.prefuse = Fussionx(depth=6, num_cond=1, dim=256).to(self.device)

        num_q = 256
        self.query = nn.Embedding(num_q, 256).to(self.device)
        self.query_pos = nn.Embedding(num_q, 256).to(self.device)
        self.pos_emb = []
        for lent in [512, 2048, 8192]:
            self.pos_emb.append(generate_1d_sine_pos_encoding(lent, 256).to(self.device))

        self.eval()

    def from_pretrain(self, ckpt: str):
        re10k = torch.load(f"{ckpt}/re10k.ckpt", map_location="cpu")["state_dict"]
        self.load_state_dict(re10k, strict=False)
        self.to(self.device)

        geo = torch.load(f"{ckpt}/mixRe10kDl3dv.ckpt", map_location="cpu")["state_dict"]
        geo = {k.removeprefix("encoder."): v for k, v in geo.items()}
        self.encoder2.load_state_dict(geo, strict=False)
        self.encoder2.to(self.device)

    @torch.inference_mode()
    def infer_pair(self, batch):
        b, v, _, h, w = batch["context"]["image"].shape
        batch["context"]["sams"] = None

        visualization_dump = {}
        gaussians, add_dict = self.encoder(batch["context"], 0, visualization_dump=visualization_dump)

        nopo_map = rearrange(add_dict["org_feat"], "b (v h w) c -> b v c h w", c=256, v=2, h=256, w=256)
        map1 = nopo_map[:, 0, ...]
        map2 = nopo_map[:, 1, ...]
        query = self.query.weight.repeat(b, 1, 1)
        all_ff = []
        all_pos = []
        for i in range(3):
            f1 = rearrange(add_dict["add1"]["inner"][i], "b c h w-> b (h w) c")
            f2 = rearrange(add_dict["add2"]["inner"][i], "b c h w-> b (h w) c")
            ff = torch.cat([f1, f2], dim=1)
            pos = self.pos_emb[i].repeat(ff.shape[0], 1, 1).to(self.device)
            all_ff.append(ff)
            all_pos.append(pos)

        (_, iou_scores, _), mask1s, mask2s = self.prefuse(
            query,
            [[all_ff[0]], [all_ff[0]], [all_ff[1]], [all_ff[1]], [all_ff[2]], [all_ff[2]]],
            map1,
            map2,
            self.query_pos.weight.repeat(b, 1, 1),
            [all_pos[0], all_pos[0], all_pos[1], all_pos[1], all_pos[2], all_pos[2]],
        )

        mask11 = F.interpolate(mask1s[-1], (256, 256))
        mask22 = F.interpolate(mask2s[-1], (256, 256))
        total_masks1 = torch.cat([mask11[:, None, ...], mask22[:, None, ...]], dim=1)
        reshape_mask1 = rearrange(total_masks1, "b v n h w -> n (b v h w)")
        sigmoided_iou = torch.sigmoid(iou_scores[-1][0])

        def postprocess(pred_iou_thresh=0.15, nms_thresh=0.6, stability_score_thresh=0.8,
                        stability_score_offset=0.5, min_mask_area=200):
            rr = postprocess_sam_masks(
                sigmoided_iou,
                reshape_mask1,
                pred_iou_thresh=pred_iou_thresh,
                nms_thresh=nms_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_area=min_mask_area,
                stability_score_offset=stability_score_offset,
            )
            sel_m = rr["masks"].float()
            masks = sel_m.clone()
            area = masks.sum(dim=1)
            order = torch.argsort(area)
            masks = masks[order]
            merged = torch.zeros(masks.shape[1], device=masks.device)
            for ind, m in enumerate(masks):
                merged[m == 1] = ind
            tt = rearrange(merged, "(v h w) -> v h w", v=2, h=256, w=256).int()
            ww = color[tt].permute(0, 3, 1, 2)
            return ww, tt

        fed, _ = postprocess()
        dc = rearrange(fed, "v c h w -> (v h w) c")[None, ...][..., None]

        self.global_step = 0
        feat_path = f"LangFlash/feat{IND}.mp4"
        gaussians, _ = self.encoder2(batch["context"], self.global_step, visualization_dump=visualization_dump)
        gaussians.harmonics = dc
        aa = render_projections(gaussians, resolution=1024, extra_label="", batch=batch)
        feat_frames = (aa[:, 0].clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        imageio.mimsave(feat_path, feat_frames, fps=12)

        rgb_path = f"LangFlash/rgb{IND}.mp4"
        gaussians, _ = self.encoder2(batch["context"], self.global_step, visualization_dump=visualization_dump)
        aa = render_projections(gaussians, resolution=1024, extra_label="", batch=batch)
        rgb_frames = (aa[:, 0].clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        imageio.mimsave(rgb_path, rgb_frames, fps=12)

        combined_path = f"LangFlash/combined{IND}.mp4"
        combined_frames = hstack_video_frames(feat_frames, rgb_frames)
        imageio.mimsave(combined_path, combined_frames, fps=12)

        return combined_path


# -----------------------------
# Demo: scene selection
# -----------------------------
path_base = "sample/"
SCENE_OPTIONS = [
    "1c7b4ce91be07717",
    "7f7bc8e76a0585d5",
    "40d3f6ff7584a8a2",
    "446ad6cc882512a6",
    "e3259a700f11ef9d",
    "ed460e6a0a9e9c3b",
]
DEFAULT_SCENE = "e3259a700f11ef9d"

def get_scene_defaults(scene_name: str):
    scene_dir = os.path.join(path_base, scene_name)
    img1_path = os.path.join(scene_dir, "images", "000000.png")
    img2_path = os.path.join(scene_dir, "images", "000020.png")
    cameras = torch.load(os.path.join(scene_dir, "cameras.pt"), map_location="cpu")
    cam1 = cameras[0]
    cam2 = cameras[20] if cameras.shape[0] > 20 else cameras[-1]
    return (
        img1_path,
        img2_path,
        float(cam1[0]), float(cam1[1]), float(cam1[2]), float(cam1[3]),
        float(cam2[0]), float(cam2[1]), float(cam2[2]), float(cam2[3]),
    )


@torch.inference_mode()
def run_demo(
    img1: Image.Image, img2: Image.Image,
    fx1, fy1, cx1, cy1, fx2, fy2, cx2, cy2,
    normalized_intrinsics, near, far,
):
    if img1 is None or img2 is None:
        raise gr.Error("image first")

    img1_t = process_pil(img1)
    img2_t = process_pil(img2)
    h1, w1 = img1.size[1], img1.size[0]
    h2, w2 = img2.size[1], img2.size[0]

    K1 = parse_intrinsics_from_inputs(fx1, fy1, cx1, cy1, w1, h1, normalized_intrinsics)
    K2 = parse_intrinsics_from_inputs(fx2, fy2, cx2, cy2, w2, h2, normalized_intrinsics)

    batch = {
        "context": {
            "image": torch.stack([img1_t, img2_t])[None, ...].to(model.device),
            "intrinsics": torch.stack([K1, K2])[None, ...].to(model.device),
            "extrinsics": None,
            "far": torch.tensor([[far, far]], dtype=torch.float32, device=model.device),
            "near": torch.tensor([[near, near]], dtype=torch.float32, device=model.device),
        }
    }

    combined_vid = model.infer_pair(batch)
    return combined_vid


# -----------------------------
# Launch Gradio demo
# -----------------------------
if __name__ == "__main__":
    config_path = "LangFlash/config/encoder/ptsplat.yaml"
    cfg = load_model_from_yaml(config_path)
    model = LangFlash(cfg)
    model.from_pretrain("ckpt")

    (
        default_img1, default_img2,
        default_fx1, default_fy1, default_cx1, default_cy1,
        default_fx2, default_fy2, default_cx2, default_cy2,
    ) = get_scene_defaults(DEFAULT_SCENE)

    with gr.Blocks(title="LangFlash Demo") as demo:
        gr.Markdown("# LangFlash Demo")

        scene_dropdown = gr.Dropdown(
            choices=SCENE_OPTIONS,
            value=DEFAULT_SCENE,
            label="Choose a predefined scene",
        )

        with gr.Row():
            img1 = gr.Image(value=default_img1, type="pil", label="Image 1")
            img2 = gr.Image(value=default_img2, type="pil", label="Image 2")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Image 1 intrinsics")
                fx1 = gr.Number(value=default_fx1, label="fx1")
                fy1 = gr.Number(value=default_fy1, label="fy1")
                cx1 = gr.Number(value=default_cx1, label="cx1")
                cy1 = gr.Number(value=default_cy1, label="cy1")
            with gr.Column():
                gr.Markdown("### Image 2 intrinsics")
                fx2 = gr.Number(value=default_fx2, label="fx2")
                fy2 = gr.Number(value=default_fy2, label="fy2")
                cx2 = gr.Number(value=default_cx2, label="cx2")
                cy2 = gr.Number(value=default_cy2, label="cy2")

        with gr.Row():
            normalized_intrinsics = gr.Checkbox(value=True, label="Intrinsics are normalized")
            near = gr.Number(value=0.1, label="near")
            far = gr.Number(value=100.0, label="far")

        btn = gr.Button("Run")
        out_video = gr.Video(label="Semantic projection | RGB projection")

        def update_scene(scene_name):
            return get_scene_defaults(scene_name)

        scene_dropdown.change(
            fn=update_scene,
            inputs=scene_dropdown,
            outputs=[img1, img2, fx1, fy1, cx1, cy1, fx2, fy2, cx2, cy2],
        )

        btn.click(
            fn=run_demo,
            inputs=[img1, img2, fx1, fy1, cx1, cy1, fx2, fy2, cx2, cy2, normalized_intrinsics, near, far],
            outputs=out_video,
        )

    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)