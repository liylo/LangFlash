import math
import torch
def generate_1d_sine_pos_encoding(seq_len, d_model):
    """
    生成1D正余弦位置编码
    Args:
        seq_len: 序列长度（如512、2048、8192）
        d_model: 特征维度（这里为256）
    Returns:
        pos_encoding: 形状为 (1, seq_len, d_model) 的位置编码
    """
    # 初始化位置编码矩阵
    pos_encoding = torch.zeros(seq_len, d_model)
    
    # 生成位置索引 i (0 ~ seq_len-1)，形状 (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    
    # 计算缩放因子：10000^(2k/d_model)，k从0到(d_model//2 - 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    # 填充偶数维和奇数维
    pos_encoding[:, 0::2] = torch.sin(position * div_term)  # 偶数维：0,2,4,...
    pos_encoding[:, 1::2] = torch.cos(position * div_term)  # 奇数维：1,3,5,...
    
    # 增加batch维度 (1, seq_len, d_model)，适配输入特征的batch_size=1
    pos_encoding = pos_encoding.unsqueeze(0)
    
    return pos_encoding

def postprocess_sam_masks(
    sigmoided_iou: torch.Tensor,          # shape (N,1) or (N,)
    mask_logits_flat: torch.Tensor,       # shape (N, HW) -- logits (not necessarily sigmoided)
    original_size: tuple | None = None,   # (H, W) optional; if None the function will try to infer
    pred_iou_thresh: float = 0.,
    stability_score_thresh: float = 0.,
    mask_threshold: float = 0.0,
    stability_score_offset: float = 1.0,
    nms_thresh: float = 0.8,
    min_mask_area: int = 0,
    device= 'cuda',
):
    assert mask_logits_flat.dim() == 2, "mask_logits_flat must be (N, HW)"
    N, HW = mask_logits_flat.shape
    if device is None:
        device = mask_logits_flat.device if mask_logits_flat.device != torch.device('cpu') else torch.device('cpu')

    # normalize iou shape
    sig_iou = sigmoided_iou.view(-1).to(device)  # (N,)
    assert sig_iou.shape[0] == N, "sigmoided_iou length must match number of masks"

    # 1) filter by predicted IoU
    keep_iou_mask = sig_iou >= pred_iou_thresh
    if keep_iou_mask.sum() == 0:
        return {"kept_inds": torch.tensor([], dtype=torch.long),
                "masks": torch.zeros((0, HW), dtype=torch.bool),
                "scores": torch.tensor([], dtype=torch.float32),
                "stability": torch.tensor([], dtype=torch.float32)}
    idx_iou_kept = torch.nonzero(keep_iou_mask, as_tuple=False).view(-1)
    masks_iou_kept = mask_logits_flat[idx_iou_kept].to(device)  # (M, HW)
    scores_iou_kept = sig_iou[idx_iou_kept]

    # 2) calculate stability score (ultralytics / SAM style)
    # stability = IoU( mask > (mask_threshold + offset), mask > (mask_threshold - offset) )
    high_thresh = mask_threshold + stability_score_offset
    low_thresh = mask_threshold - stability_score_offset
    high_bin = (masks_iou_kept > high_thresh)    # (M, HW)
    low_bin  = (masks_iou_kept > low_thresh)     # (M, HW)

    # intersections / unions with safe divide
    intersections = (high_bin & low_bin).sum(dim=1).to(torch.int32)  # actually equals high_bin.sum but keep generality
    unions = low_bin.sum(dim=1).to(torch.int32)
    # avoid div by 0
    unions_safe = unions.clone().float()
    unions_safe[unions_safe == 0] = 1.0
    stability_scores = intersections.float() / unions_safe  # (M,)

    # 3) filter by stability score
    keep_stab_mask = stability_scores >= stability_score_thresh
    if keep_stab_mask.sum() == 0:
        return {"kept_inds": torch.tensor([], dtype=torch.long),
                "masks": torch.zeros((0, HW), dtype=torch.bool),
                "scores": torch.tensor([], dtype=torch.float32),
                "stability": torch.tensor([], dtype=torch.float32)}

    kept_rel_idx = torch.nonzero(keep_stab_mask, as_tuple=False).view(-1)
    final_indices = idx_iou_kept[kept_rel_idx]           # indices relative to original masks (N)
    masks_kept_logits = masks_iou_kept[kept_rel_idx]     # (K, HW)
    scores_kept = scores_iou_kept[kept_rel_idx]
    stability_kept = stability_scores[kept_rel_idx]

    # 4) binarize masks with mask_threshold
    bin_masks = (masks_kept_logits > mask_threshold)   # (K, HW) boolean

    # 4.5) remove tiny masks by area if requested
    areas = bin_masks.sum(dim=1)
    keep_area_mask = areas >= min_mask_area
    if keep_area_mask.sum() == 0:
        return {"kept_inds": torch.tensor([], dtype=torch.long),
                "masks": torch.zeros((0, HW), dtype=torch.bool),
                "scores": torch.tensor([], dtype=torch.float32),
                "stability": torch.tensor([], dtype=torch.float32)}

    keep_inds_area_rel = torch.nonzero(keep_area_mask, as_tuple=False).view(-1)
    final_indices = final_indices[keep_inds_area_rel]
    bin_masks = bin_masks[keep_inds_area_rel]
    scores_kept = scores_kept[keep_inds_area_rel]
    stability_kept = stability_kept[keep_inds_area_rel]
    areas = areas[keep_inds_area_rel]

    # 5) run mask NMS (IoU-based)
    # We'll use a standard greedy NMS using mask IoU. Sort by score descending.
    order = torch.argsort(scores_kept, descending=True)
    keep = []
    masks_bool = bin_masks # operate on CPU for easier matmul if GPU memory tight
    areas_cpu = areas.float()
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        # compute intersection between mask i and rest
        # masks are bool -> convert to uint8->float for matmul
        mi = masks_bool[i].to(torch.uint8).float()  # (HW,)
        mj = masks_bool[rest].to(torch.uint8).float()  # (R, HW)
        # intersection: dot product mi @ mj.T
        inter = torch.matmul(mj, mi)  # (R,)
        union = areas_cpu[i] + areas_cpu[rest] - inter
        # safe divide
        union[union == 0] = 1.0
        ious = inter / union
        keep_mask = ious <= nms_thresh
        order = rest[keep_mask]

    keep = torch.tensor(keep, dtype=torch.long)
    # map keep back to original mask indices
    kept_original_inds = final_indices[keep].cpu()
    kept_masks = bin_masks[keep].cpu()  # (K_keep, HW) boolean
    kept_scores = scores_kept[keep].cpu()
    kept_stabilities = stability_kept[keep].cpu()

    # optionally reshape masks to (K, H, W) if original_size provided or inferrable
    masks_out = kept_masks
    masks_2d = None

    return {
        "kept_inds": kept_original_inds,
        "masks": masks_out,          # (K_keep, HW) boolean flattened
        "scores": kept_scores,
        "stability": kept_stabilities,
    }