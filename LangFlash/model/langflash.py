import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from .semantic.utils import  generate_1d_sine_pos_encoding,postprocess_sam_masks
from .encoder.encoder_instantsplat import EncoderInstantSplat
from .semantic.seg import Fussion,Fussionx
from torchvision.utils import make_grid, save_image
from ..visualization.validation_in_3d import render_projections
from .decoder.decoder_splatting_cuda import DecoderSplattingCUDA,DecoderSplattingCUDACfg
IND = 0
import imageio
class LangFlash(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.device = "cuda"
        decoder_cfg = DecoderSplattingCUDACfg(name="splatting_cuda",make_scale_invariant=False,background_color=[0.0, 0.0, 0.0]) 
        self.decoder = DecoderSplattingCUDA(decoder_cfg)
        self.encoder = EncoderInstantSplat(cfg).cuda()
        self.encoder2 = EncoderInstantSplat(cfg).cuda()
        self.fuse = Fussion(depth=6,num_cond=1,dim=256).cuda()
        self.prefuse = Fussionx(depth=6,num_cond=1,dim=256).cuda()
        num_q = 256
        self.query = nn.Embedding(num_q,256).cuda()
        self.query_pos = nn.Embedding(num_q,256).cuda()
        self.pos_emb = []
        for lent in [512,2048,8192]:
            self.pos_emb.append(generate_1d_sine_pos_encoding(lent,256).to(self.device))
        self.eval()
            
    def from_pretrain(self,ckpt):
        re10k = torch.load(f"{ckpt}/re10k.ckpt")["state_dict"]
        self.load_state_dict(re10k,strict=False)
        self.to(self.device)
      
      
        geo = torch.load(f"/home/liylo/mixRe10kDl3dv.ckpt")['state_dict']
        geo = {k.removeprefix("encoder."): v for k, v in geo.items()}
        self.encoder2.load_state_dict(geo,strict=False)
        self.encoder2.to(self.device)
        
        # encoder_state = {
        #     k[len("encoder."):] : v
        #     for k, v in re10k.items()
        #     if k.startswith("encoder.")
        # }
        # fuse_state = {
        #     k[len("fuse."):] : v
        #     for k, v in re10k.items()
        #     if k.startswith("fuse.")
        # }
        # prefuse_state = {
        #     k[len("prefuse."):] : v
        #     for k, v in re10k.items()
        #     if k.startswith("prefuse.")
        # }
        # self.encoder.load_state_dict(encoder_state)
        # self.fuse.load_state_dict(fuse_state)
        # self.prefuse.load_state_dict(prefuse_state)
        # self.encoder.to(self.device)
        # self.fuse.to(self.device)
        # self.prefuse.to(self.device)
        # query = {
        #     k[len("query."):] : v
        #     for k, v in re10k.items()
        #     if k.startswith("query.")
        # }
        # query_pos = {
        #     k[len("query_pos."):] : v
        #     for k, v in re10k.items()
        #     if k.startswith("query_pos.")
        # }
        # self.query.load_state_dict(query)
        # self.query_pos.load_state_dict(query_pos)
        # self.query.to(self.device)
        # self.query_pos.to(self.device)
        
            
    def data_shim(self, batch):
        return batch
      
    @torch.inference_mode()
    def vis_step(self, batch):
            batch = self.data_shim(batch)
            b, v, _, h, w = batch["context"]["image"].shape
            
            batch["context"]["sams"] = None
                

            visualization_dump = {}
            
            gaussians,add_dict = self.encoder(batch["context"], 0, visualization_dump=visualization_dump)

            nopo_map = rearrange(add_dict['org_feat'],"b (v h w) c -> b v c h w",c=256,v=2,h=256,w=256)

            map1 = nopo_map[:,0,...]
            map2 = nopo_map[:,1,...]
            
            query = self.query.weight.repeat(b,1,1)

            all_ff = []
            all_pos = []
            for i in range(3):
                f1 = rearrange(add_dict['add1']['inner'][i],"b c h w-> b (h w) c")
                f2 = rearrange(add_dict['add2']['inner'][i],"b c h w-> b (h w) c")
                ff = torch.cat([f1,f2],dim=1)
                pos = self.pos_emb[i].repeat(ff.shape[0],1,1).to(self.device)
                all_ff.append(ff)
                all_pos.append(pos)

            (_,iou_scoress,lang_scoress),mask1s,mask2s = self.prefuse(query,[[all_ff[0]],[all_ff[0]],[all_ff[1]],[all_ff[1]],[all_ff[2]],[all_ff[2]]],map1,map2,self.query_pos.weight.repeat(b,1,1),[all_pos[0],all_pos[0],all_pos[1],all_pos[1],all_pos[2],all_pos[2]])
            mask1 = torch.sigmoid(F.interpolate(mask1s[-1],(256,256)))
            mask2 = torch.sigmoid(F.interpolate(mask2s[-1],(256,256)))
            totoal_masks = torch.cat([mask1[:,None,...],mask2[:,None,...]],dim=1) # B V N H W
            reshape_mask = rearrange(totoal_masks,"b v n h w ->  n (b v h w)")
            
            
            mask11 = (F.interpolate(mask1s[-1],(256,256)))
            mask22 = (F.interpolate(mask2s[-1],(256,256)))
            totoal_masks1 = torch.cat([mask11[:,None,...],mask22[:,None,...]],dim=1) # B V N H W
            reshape_mask1 = rearrange(totoal_masks1,"b v n h w ->  n (b v h w)")
            sigmoided_iou = F.sigmoid(iou_scoress[-1][0])
            
            color = torch.rand((256,3)).cuda()
            color[0] = torch.tensor([0,0,0]).cuda()
            color[8] = torch.tensor([1,1,1]).cuda()
            def vv(pred_iou_thresh=0.3,nms_thresh=0.7,stability_score_thresh=0.5,stability_score_offset=1.0,min_mask_area=100):
                rr = postprocess_sam_masks(sigmoided_iou,reshape_mask1,pred_iou_thresh=pred_iou_thresh,nms_thresh=nms_thresh,stability_score_thresh=stability_score_thresh,min_mask_area=min_mask_area,stability_score_offset=stability_score_offset)
                sel_m = rr['masks'].float()
                masks = sel_m.clone()
                area = masks.sum(dim=1)
                order = torch.argsort(area)  # small -> large
                masks = masks[order]
                H = masks.shape[1]
                merged = torch.zeros(H, device=masks.device)
                for ind,m in enumerate(masks):
                    merged[m == 1] = ind
                tt = rearrange(merged,"(v h w) -> v h w",v=2,h=256,w=256)
                tt = tt.int()
                ww = color[tt].permute(0,3,1,2)
                print(ind)
                return ww,tt
            fed,tt = vv(0.15,nms_thresh=0.6,stability_score_thresh=0.8,stability_score_offset=0.5,min_mask_area=200)
            
            
            dc = rearrange(fed,"v c h w -> (v h w) c")[None,...][...,None]
  
            self.global_step = 0
            gaussians,add_dict = self.encoder2(batch["context"], self.global_step, visualization_dump=visualization_dump)
            gaussians.harmonics = dc

            aa = render_projections(
                                    gaussians,
                                    resolution = 1024,
                                    extra_label="",
                                    batch = batch
                                )
            frames = aa[:, 0]  # [T, 3, H, W]

            frames = (frames.clamp(0, 1) * 255).byte()
            frames = frames.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, 3]

            imageio.mimsave(
                f"/home/liylo/LangFlash/feat{IND}.mp4",
                frames,
                fps=12
            )
            # save_image(aa[:,0,...],f"/home/liylo/LangFlash/feat{IND}.png") # 72 3 256 256
            
            gaussians,add_dict = self.encoder2(batch["context"], self.global_step, visualization_dump=visualization_dump)

            aa = render_projections(
                                    gaussians,
                                    resolution = 1024,
                                    extra_label="",
                                    batch = batch
                                )
            # save_image(aa[:,0,...],f"/home/liylo/LangFlash/rgb{IND}.png") # 72 3 256 256
            frames = aa[:, 0]  # [T, 3, H, W]

            frames = (frames.clamp(0, 1) * 255).byte()
            frames = frames.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, 3]

            imageio.mimsave(
                f"/home/liylo/LangFlash/rgb{IND}.mp4",
                frames,
                fps=12
            )

            
            # output = self.decoder.forward(
            #     gaussians2,
            #     batch["context"]["extrinsics"],
            #     batch["context"]["intrinsics"],
            #     batch["context"]["near"],
            #     batch["context"]["far"],
            #     (h, w),
            #     depth_mode=False,
            # )
            
            # rgb_pred = output.color[0]
            # save_image(rgb_pred,"/home/liylo/LangFlash/rgb_vis.png")
            

from pathlib import Path
from omegaconf import OmegaConf

from LangFlash.model.encoder.backbone.backbone_croco import BackboneCrocoCfg
from LangFlash.model.encoder.backbone.backbone_dino import BackboneDinoCfg
from LangFlash.model.encoder.backbone.backbone_resnet import BackboneResnetCfg
from LangFlash.model.encoder.encoder_instantsplat import EncoderNoPoSplatCfg,OpacityMappingCfg,GaussianAdapterCfg,GaussianAdapterCfg,EncoderVisualizerEpipolarCfg
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

    return encoder_cfg

from PIL import Image
import torchvision.transforms.functional as TF

def process(img_path, size=(256, 256)):
    """
    center square crop -> resize -> tensor -> [-1, 1]
    """
    img = Image.open(img_path).convert("RGB")

    # 取中心正方形
    w, h = img.size
    crop_size = min(w, h)
    img = TF.center_crop(img, [crop_size, crop_size])

    # resize 到目标尺寸
    img = TF.resize(img, size)

    # 转 tensor: [0,1]
    img = TF.to_tensor(img)

    # 映射到 [-1,1]
    img = img * 2 - 1

    return img

def parse_ins_ext(pose_list):
    """
    输入长度为 18 的列表:
    [fx, fy, cx, cy, 0, 0, 3x4外参展开...]

    返回:
        ins: 3x3 intrinsic matrix
        ext: 4x4 extrinsic matrix (C2W, 因为原代码最后做了 inverse)
    """
    poses = torch.tensor(pose_list, dtype=torch.float32).unsqueeze(0)  # [1, 18]

    # intrinsics
    ins = torch.eye(3, dtype=torch.float32).unsqueeze(0)  # [1, 3, 3]
    fx, fy, cx, cy = poses[:, :4].T
    ins[:, 0, 0] = fx
    ins[:, 1, 1] = fy
    ins[:, 0, 2] = cx
    ins[:, 1, 2] = cy

    # extrinsics
    w2c = torch.eye(4, dtype=torch.float32).unsqueeze(0)  # [1, 4, 4]
    w2c[:, :3, :] = poses[:, 6:].reshape(1, 3, 4)

    ext = torch.linalg.inv(w2c)  # 和你代码里的 w2c.inverse() 一样
    return ins[0], ext[0]
  
def center_crop_then_resize_intrinsics(
    K: torch.Tensor,
    h_in: int = 360,
    w_in: int = 640,
    out_h: int = 256,
    out_w: int = 256,
) -> torch.Tensor:
    """
    适用于 K 已经按原图尺寸归一化的情况:
        fx, cx 以 w_in 归一化
        fy, cy 以 h_in 归一化

    对应你的 process():
        center_crop -> resize
    """
    K = torch.as_tensor(K, dtype=torch.float32).clone()

    crop_size = min(h_in, w_in)  # 360
    top = (h_in - crop_size) // 2
    left = (w_in - crop_size) // 2

    # 1) 先从 normalized K 转回像素单位
    fx = K[0, 0] * w_in
    fy = K[1, 1] * h_in
    cx = K[0, 2] * w_in
    cy = K[1, 2] * h_in

    # 2) center crop
    cx -= left
    cy -= top

    # 3) resize 到 out_h x out_w
    sx = out_w / crop_size
    sy = out_h / crop_size
    fx *= sx
    fy *= sy
    cx *= sx
    cy *= sy

    # 4) 再归一化到输出尺寸
    K[0, 0] = fx / out_w
    K[1, 1] = fy / out_h
    K[0, 2] = cx / out_w
    K[1, 2] = cy / out_h

    return K


import os
if __name__ == "__main__":
    config_path = "LangFlash/config/encoder/ptsplat.yaml"

    cfg = load_model_from_yaml(config_path)

    model = LangFlash(cfg)
    
    model.from_pretrain("/home/liylo/LangFlash")
    
    for name in os.listdir("/home/liylo/data/re10k_subset/train_unpacked1"):
      base = f"/home/liylo/data/re10k_subset/train_unpacked1/{name}"
      
      img1_path = f"{base}/images/000000.png"
      img2_path = f"{base}/images/000020.png"
      
      import json
      
      # jf = json.load(open(f"{base}/meta.json","r"))
      cam = {"cameras":torch.load(f"{base}/cameras.pt")}
      
      img1 = process(img1_path)
      img2 = process(img2_path)
      
      save_image(img1,"i1.png")
      save_image(img2,"i2.png")
      
      
      K,E = parse_ins_ext(cam["cameras"][0])
      K2,E2 = parse_ins_ext(cam["cameras"][20])
      
      K = center_crop_then_resize_intrinsics(K,360,640,256,256)
      K2 = center_crop_then_resize_intrinsics(K2,360,640,256,256)

      # baseline
      scale = torch.norm(E[:3, 3] - E2[:3, 3])
      # 可选：避免除零
      eps = 1e-8
      scale = torch.clamp(scale, min=eps)

      # 归一化：让两帧之间的 baseline 变成 1
      E[:3, 3]  /= scale
      E2[:3, 3] /= scale
      
      near = torch.tensor(0.1)
      far = torch.tensor(100)
      
      batch = {'context':{'image':None,"intrinsics":None,"extrinsics":None}}
      batch['context']['image'] = torch.stack([img1,img2])[None,...].cuda()
      batch['context']['intrinsics'] = torch.stack([K,K2])[None,...].cuda()
      batch['context']['extrinsics'] = torch.stack([E,E2])[None,...].cuda()
      batch["context"]["far"] = torch.stack([far,far])[None,...].cuda()
      batch["context"]["near"] = torch.stack([near,near])[None,...].cuda()
      model.vis_step(batch)
      
      IND+=1
      break
    