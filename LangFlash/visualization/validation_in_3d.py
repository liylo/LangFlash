import torch
from jaxtyping import Float, Shaped
from torch import Tensor

from ..model.decoder.cuda_splatting import render_cuda_orthographic,render_cuda
from ..model.encoder.type_def import Gaussians
from ..visualization.annotation import add_label
from ..visualization.drawing.cameras import draw_cameras
from .drawing.cameras import compute_equal_aabb_with_margin


def pad(images: list[Shaped[Tensor, "..."]]) -> list[Shaped[Tensor, "..."]]:
    shapes = torch.stack([torch.tensor(x.shape) for x in images])
    padded_shape = shapes.max(dim=0)[0]
    results = [
        torch.ones(padded_shape.tolist(), dtype=x.dtype, device=x.device)
        for x in images
    ]
    for image, result in zip(images, results):
        slices = [slice(0, x) for x in image.shape]
        result[slices] = image[slices]
    return results
import math
def make_view_trajectory_extrinsics_batch(
    scene_minima: torch.Tensor,   # [B, 3] 或 [3]
    scene_maxima: torch.Tensor,   # [B, 3] 或 [3]
    n_hold: int = 8,              # 完全正 Z 视角保持几帧
    n_warmup: int = 16,           # 从正 Z 平滑偏到小角度
    n_orbit: int = 48,            # 开始转圈
    warmup_angle_deg: float = 12.0,
    orbit_radius_ratio: float = 0.12,
    z_distance_ratio: float = 1.10,
):
    """
    返回:
        extrinsics_traj: [T, 4, 4]
    约定:
        与你的 render_cuda_orthographic 一致：
        extrinsics[:3, 0] = right
        extrinsics[:3, 1] = down
        extrinsics[:3, 2] = look
        extrinsics[:3, 3] = camera position
    """
    if scene_minima.dim() == 2:
        scene_minima = scene_minima[0]
    if scene_maxima.dim() == 2:
        scene_maxima = scene_maxima[0]

    device = scene_minima.device
    dtype = scene_minima.dtype

    center = 0.5 * (scene_minima + scene_maxima)   # [3]
    extents = scene_maxima - scene_minima          # [3]

    cam_dist = extents[2] * z_distance_ratio
    orbit_r = max(extents[0].item(), extents[1].item()) * orbit_radius_ratio

    warmup_theta = math.radians(warmup_angle_deg)

    # 初始：严格沿 +Z 看
    start_pos = center + torch.tensor([0.0, 0.0, -cam_dist], device=device, dtype=dtype)

    # 过渡结束：偏到一个很小角度
    end_pos = center + torch.tensor(
        [
            orbit_r * math.cos(warmup_theta),
            orbit_r * math.sin(warmup_theta),
            -cam_dist,
        ],
        device=device,
        dtype=dtype,
    )

    up_ref = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=dtype)

    def build_extrinsic(cam_pos: torch.Tensor):
        look = center - cam_pos
        look = look / (look.norm() + 1e-8)

        right = torch.cross(look, up_ref, dim=0)
        right = right / (right.norm() + 1e-8)

        down = torch.cross(look, right, dim=0)
        down = down / (down.norm() + 1e-8)

        E = torch.eye(4, device=device, dtype=dtype)
        E[:3, 0] = right
        E[:3, 1] = down
        E[:3, 2] = look
        E[:3, 3] = cam_pos
        return E

    traj = []

    # 1) 保持正 Z 视角
    for _ in range(n_hold):
        traj.append(build_extrinsic(start_pos))

    # 2) 平滑偏离到小角度
    for i in range(n_warmup):
        t = (i + 1) / n_warmup
        cam_pos = (1.0 - t) * start_pos + t * end_pos
        traj.append(build_extrinsic(cam_pos))

    # 3) 围绕中心小半径转圈
    for i in range(n_orbit):
        theta = warmup_theta + 2.0 * math.pi * i / n_orbit
        cam_pos = center + torch.tensor(
            [
                orbit_r * math.cos(theta),
                orbit_r * math.sin(theta),
                -cam_dist,
            ],
            device=device,
            dtype=dtype,
        )
        traj.append(build_extrinsic(cam_pos))

    return torch.stack(traj, dim=0)  # [T, 4, 4]

def render_projections(
    gaussians: Gaussians,
    resolution: int,
    margin: float = 0.1,
    draw_label: bool = True,
    extra_label: str = "",
    batch=None,
):
    device = gaussians.means.device
    b, _, _ = gaussians.means.shape  # b=1（批次大小）

    # 计算场景边界
    minima = gaussians.means.min(dim=1).values
    maxima = gaussians.means.max(dim=1).values
    scene_minima, scene_maxima = compute_equal_aabb_with_margin(
        minima, maxima, margin=margin
    )

    projections = []
    look_axis = 2  # 沿Z轴观察（视线方向为Z轴）
    right_axis = (look_axis + 1) % 3
    down_axis = (look_axis + 2) % 3

    # 定义相机外参
    extrinsics = torch.zeros((b, 4, 4), dtype=torch.float32, device=device)
    extrinsics[:, right_axis, 0] = 1
    extrinsics[:, down_axis, 1] = 1
    extrinsics[:, look_axis, 2] = 1
    extrinsics[:, right_axis, 3] = 0.5 * (scene_minima[:, right_axis] + scene_maxima[:, right_axis])
    extrinsics[:, down_axis, 3] = 0.5 * (scene_minima[:, down_axis] + scene_maxima[:, down_axis])
    extrinsics[:, look_axis, 3] = scene_minima[:, look_axis]
    extrinsics[:, 3, 3] = 1

    # 定义相机内参相关参数
    extents = scene_maxima - scene_minima
    far_plane = extents[:, look_axis]  # 远平面（场景沿视线方向的最大长度）
    near_plane = torch.zeros_like(far_plane)  # 近平面
    width = extents[:, right_axis]
    height = extents[:, down_axis]
    
    # 计算每个高斯沿视线方向的深度（相机坐标系下）
    camera_pos = extrinsics[:, :3, 3]  # 相机位置 [b, 3]
    look_dir = extrinsics[:, :3, 2].unsqueeze(1)  # 视线方向 [b, 1, 3]
    gaussian_to_cam = gaussians.means - camera_pos.unsqueeze(1)  # 高斯到相机的向量 [b, N, 3]
    depth = torch.sum(gaussian_to_cam * look_dir, dim=-1)  # 沿视线方向的深度 [b, N]

    # 步骤1：筛选出在[near_plane, far_plane]范围内的有效高斯
    valid_mask = (depth >= near_plane) & (depth <= far_plane)  # [b, N]
    # 取第0个批次（因为b=1）
    valid_mask_0 = valid_mask[0]  # [N]
    valid_depth = depth[0][valid_mask_0]  # 有效高斯的深度 [M]，M为有效数量
    valid_indices = valid_mask_0.nonzero().squeeze(1)  # 有效高斯的原始索引 [M]

    # 步骤2：在有效高斯中，去除最近的20%
    if len(valid_depth) == 0:
        # 无有效高斯，直接返回空（避免报错）
        return torch.tensor([], device=device)
    
    # 计算“最近20%”的深度阈值（第20%分位数）
    # 例如：100个有效高斯，取第20个最小深度作为阈值，剔除前20个
    threshold = torch.quantile(valid_depth, 0.05)  # 0.2表示20%分位数

    # 筛选出深度 >= 阈值的高斯（保留80%，去除最近的20%）
    keep_mask = valid_depth >= threshold  # [M]
    final_indices = valid_indices[keep_mask]  # 最终保留的高斯索引 [K]，K ≈ M*0.8

    # 步骤3：用最终索引更新高斯参数
    gaussians.means = gaussians.means[:, final_indices, :]
    gaussians.covariances = gaussians.covariances[:, final_indices, ...]
    gaussians.harmonics = gaussians.harmonics[:, final_indices, ...]
    gaussians.opacities = gaussians.opacities[:, final_indices]
    
    extrinsics_traj = make_view_trajectory_extrinsics_batch(
        scene_minima=scene_minima,
        scene_maxima=scene_maxima,
        n_hold=8,
        n_warmup=16,
        n_orbit=48,
    )

    imgs = []
    T = extrinsics_traj.shape[0]

    for i in range(T):
        proj_i = render_cuda_orthographic(
            extrinsics_traj[i:i+1],          # [1, 4, 4]
            width,                    # [1]
            height,                   # [1]
            near_plane,               # [1]
            far_plane,                # [1]
            (resolution, resolution),
            torch.zeros((1, 3), dtype=torch.float32, device=device),  # [1, 3]
            gaussians.means,                 # [1, G, 3]，保持原样即可
            gaussians.covariances,           # [1, G, 3, 3]
            gaussians.harmonics,             # [1, G, 3, d_sh]
            gaussians.opacities,             # [1, G]
            fov_degrees=0.1,
        )
        imgs.append(proj_i)

    projection = torch.cat(imgs, dim=0)  # [T, 3, H, W]
    # if draw_label:
    #     right_axis_name = "XYZ"[right_axis]
    #     down_axis_name = "XYZ"[down_axis]
    #     label = f"{right_axis_name}{down_axis_name} Projection {extra_label}"
    #     projection = torch.stack([add_label(x, label) for x in projection])

    projections.append(projection)

    return torch.stack(pad(projections), dim=1)


def render_cameras(batch: dict, resolution: int) -> Float[Tensor, "3 3 height width"]:
    # Define colors for context and target views.
    num_context_views = batch["context"]["extrinsics"].shape[1]
    num_target_views = batch["target"]["extrinsics"].shape[1]
    color = torch.ones(
        (num_target_views + num_context_views, 3),
        dtype=torch.float32,
        device=batch["target"]["extrinsics"].device,
    )
    color[num_context_views:, 1:] = 0

    return draw_cameras(
        resolution,
        torch.cat(
            (batch["context"]["extrinsics"][0], batch["target"]["extrinsics"][0])
        ),
        torch.cat(
            (batch["context"]["intrinsics"][0], batch["target"]["intrinsics"][0])
        ),
        color,
        torch.cat((batch["context"]["near"][0], batch["target"]["near"][0])),
        torch.cat((batch["context"]["far"][0], batch["target"]["far"][0])),
    )
