import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .metrics import build_ms_ssim, masked_photometric_loss
from .renderer import render_rgb_alpha


@dataclass(frozen=True)
class OptimizerConfig:
    rotation_lr: float = 0.01
    translation_lr: float = 5.0
    # The epsilon_* limits form a trust region: they bound the total deviation
    # of the refined pose from the coarse pose, not each iteration's update.
    epsilon_rotation_rad: float = np.radians(10.0)
    # Silhouette IoU is a good lateral (xy) signal but a poor depth (z) signal —
    # the optimizer exploits z to match mask scale rather than correcting true pose.
    epsilon_translation_xy: float = 10.0  # ±10mm lateral correction
    # Depth: the Chamfer loss constrains z directly when geo_weight > 0.
    # Set to 0.0 for photo-only mode (silhouette/photo exploit z).
    epsilon_translation_z: float = 10.0
    max_iterations: int = 200
    rendered_mask_threshold: float = 1e-3
    interleave_steps: int = 10
    photo_weight: float = 1.0
    silhouette_weight: float = 0.0
    # Weight for 3D Chamfer loss (model.ply → query depth point cloud).
    # Set to 0.0 for photo-only mode; tune alongside epsilon_translation_z.
    geo_weight: float = 0.001
    early_stopping_patience: int = 30
    # "rot6d": continuous 6D rotation (Zhou et al. 2019), Gram-Schmidt delta (default).
    # "axis_angle": so(3) exp-map delta (previous behavior).
    # Both use the same delta-from-coarse structure and the same angle trust region.
    rotation_parametrization: str = "rot6d"


def _skew_symmetric(vec: torch.Tensor) -> torch.Tensor:
    x, y, z = vec
    zero = torch.zeros((), dtype=vec.dtype, device=vec.device)
    row0 = torch.stack([zero, -z, y])
    row1 = torch.stack([z, zero, -x])
    row2 = torch.stack([-y, x, zero])
    return torch.stack([row0, row1, row2], dim=0)


def so3_exp_map(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    theta = torch.linalg.vector_norm(omega)
    omega_hat = _skew_symmetric(omega)
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device)
    if float(theta.item()) < eps:
        return eye + omega_hat
    omega_hat2 = omega_hat @ omega_hat
    a = torch.sin(theta) / theta
    b = (1.0 - torch.cos(theta)) / (theta * theta)
    return eye + a * omega_hat + b * omega_hat2


def rot6d_to_matrix(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Gram-Schmidt on two 3-vectors; [1,0,0,0,1,0] maps to identity.
    b1 = x[0:3] / torch.clamp(torch.linalg.vector_norm(x[0:3]), min=eps)
    b2_raw = x[3:6] - (b1 * x[3:6]).sum() * b1
    b2 = b2_raw / torch.clamp(torch.linalg.vector_norm(b2_raw), min=eps)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack([b1, b2, b3], dim=1)


def so3_log_map(rotation: torch.Tensor) -> torch.Tensor:
    skew = rotation - rotation.T
    half_vee = torch.stack([skew[2, 1], skew[0, 2], skew[1, 0]]) / 2.0
    cos_theta = torch.clamp((torch.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    if float(sin_theta.item()) < 1e-8:
        return half_vee
    return half_vee * (theta / sin_theta)


def _clamp_rot6d_angle(x: torch.Tensor, max_angle: float) -> None:
    # In-place: shrink the rotation encoded by x to at most max_angle.
    # Rewrites x as the first two columns of the clamped matrix.
    rot = rot6d_to_matrix(x)
    theta = torch.acos(torch.clamp((torch.trace(rot) - 1.0) / 2.0, -1.0, 1.0))
    if float(theta.item()) <= max_angle:
        return
    clamped = so3_exp_map(so3_log_map(rot) * (max_angle / theta))
    x.copy_(torch.cat([clamped[:, 0], clamped[:, 1]]))


def optimize_pose(
    *,
    splats: dict[str, torch.Tensor],
    coarse_rotation: torch.Tensor,
    translation: torch.Tensor,
    target_image_bchw: torch.Tensor,
    query_mask_bchw: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    config: OptimizerConfig,
    seed: int,
    model_pts_obj: torch.Tensor | None = None,
    query_pts_cam: torch.Tensor | None = None,
    debug_dir: Path | None = None,
) -> dict[str, object]:
    torch.manual_seed(seed)
    if config.rotation_parametrization not in ("axis_angle", "rot6d"):
        raise ValueError(
            "Unknown rotation_parametrization: %r" % config.rotation_parametrization
        )
    use_rot6d = config.rotation_parametrization == "rot6d"
    if use_rot6d:
        init_omega = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            dtype=coarse_rotation.dtype,
            device=coarse_rotation.device,
        )
    else:
        init_omega = torch.zeros(
            3, dtype=coarse_rotation.dtype, device=coarse_rotation.device
        )
    init_translation = torch.zeros(
        3, dtype=translation.dtype, device=translation.device
    )
    omega = torch.nn.Parameter(init_omega)
    delta_t = torch.nn.Parameter(init_translation)

    # weight_decay=0: decay would pull the deltas back toward the coarse pose.
    rot_optimizer = torch.optim.AdamW(
        [omega], lr=config.rotation_lr, weight_decay=0.0
    )
    trans_optimizer = torch.optim.AdamW(
        [delta_t], lr=config.translation_lr, weight_decay=0.0
    )
    ms_ssim_fn = build_ms_ssim(3, coarse_rotation.device)
    step_log: list[dict] = []
    n_steps = 0
    best_loss = float("inf")
    best_iteration = 0
    steps_without_improvement = 0
    _acc_render = 0.0
    _acc_photo_loss = 0.0
    _acc_backward = 0.0
    _acc_step = 0.0

    best = {
        "loss": float("inf"),
        "rotation": coarse_rotation.detach().clone(),
        "translation": translation.detach().clone(),
        "photometric_metrics": {
            "ssim": 0.0,
            "ms_ssim": 0.0,
            "loss": float("inf"),
        },
    }

    for _step in range(config.max_iterations):
        rot_optimizer.zero_grad()
        trans_optimizer.zero_grad()

        delta_rotation = rot6d_to_matrix(omega) if use_rot6d else so3_exp_map(omega)
        current_rotation = delta_rotation @ coarse_rotation
        current_translation = translation + delta_t
        bottom_row = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]],
            dtype=current_rotation.dtype,
            device=current_rotation.device,
        )
        w2c = torch.cat(
            [
                torch.cat([current_rotation, current_translation.unsqueeze(1)], dim=1),
                bottom_row,
            ],
            dim=0,
        )

        _t0 = time.perf_counter()
        rgb, alpha = render_rgb_alpha(
            splats=splats,
            world_to_camera=w2c,
            intrinsics=intrinsics,
            width=width,
            height=height,
        )
        _acc_render += time.perf_counter() - _t0
        render_bchw = rgb.permute(0, 3, 1, 2)
        alpha_bchw = alpha.permute(0, 3, 1, 2)
        rendered_mask_bchw = (alpha_bchw > config.rendered_mask_threshold).to(
            query_mask_bchw.dtype
        )

        # Photo loss: compare rendered pixels against the real scene only within
        # the region the splat currently occupies.  CNOS masks are too noisy to
        # use as a photometric window — their shape often belongs to a different
        # object, so the composite intersection mask was measuring the wrong pixels.
        _t0 = time.perf_counter()
        photo_loss, photometric_metrics = masked_photometric_loss(
            predicted_bchw=render_bchw,
            target_bchw=target_image_bchw,
            composite_mask_bchw=rendered_mask_bchw,
            ms_ssim_fn=ms_ssim_fn,
        )
        _acc_photo_loss += time.perf_counter() - _t0

        # Silhouette: very small weight — provides a weak location prior from the
        # CNOS centroid without letting the noisy mask shape dominate.
        intersection = (alpha_bchw * query_mask_bchw).sum()
        union = (alpha_bchw + query_mask_bchw - alpha_bchw * query_mask_bchw).sum()
        silhouette_loss = 1.0 - intersection / (union + 1e-6)

        # Geometric Chamfer loss: transform model.ply points to camera frame and
        # measure one-directional distance to the query depth point cloud.
        # This signal is appearance-independent and directly constrains depth (z).
        geo_loss = torch.zeros((), dtype=photo_loss.dtype, device=photo_loss.device)
        if (
            config.geo_weight > 0
            and model_pts_obj is not None
            and query_pts_cam is not None
        ):
            model_pts_cam = (current_rotation @ model_pts_obj.T).T + current_translation
            dists = torch.cdist(
                model_pts_cam.unsqueeze(0),
                query_pts_cam.unsqueeze(0),
            )[0]  # [N, M]
            # Bidirectional Chamfer: backward term (query → template) penalizes
            # the model drifting away from query points, preventing z-exploit.
            fwd = dists.min(dim=1).values.mean()  # template → query
            bwd = dists.min(dim=0).values.mean()  # query → template
            geo_loss = (fwd + bwd) * 0.5

        loss = (
            config.photo_weight * photo_loss
            + config.silhouette_weight * silhouette_loss
            + config.geo_weight * geo_loss
        )

        n_steps += 1
        loss_value = float(loss.detach().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_iteration = _step
            best["loss"] = loss_value
            best["rotation"] = current_rotation.detach().clone()
            best["translation"] = current_translation.detach().clone()
            best["photometric_metrics"] = photometric_metrics
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        _t0 = time.perf_counter()
        loss.backward()
        _acc_backward += time.perf_counter() - _t0

        if debug_dir is not None:
            step_log.append(
                {
                    "step": _step,
                    "phase": "rot"
                    if (
                        config.interleave_steps <= 0
                        or (_step // config.interleave_steps) % 2 == 0
                    )
                    else "trans",
                    "loss": float(loss.detach()),
                    "photo_loss": float(photo_loss.detach()),
                    "silhouette_loss": float(silhouette_loss.detach()),
                    "geo_loss": float(geo_loss.detach()),
                    "ssim": photometric_metrics["ssim"],
                    "ms_ssim": photometric_metrics["ms_ssim"],
                    "omega_grad_norm": float(omega.grad.norm())
                    if omega.grad is not None
                    else 0.0,
                    "delta_t_grad_norm": float(delta_t.grad.norm())
                    if delta_t.grad is not None
                    else 0.0,
                    "omega_norm": float(omega.detach().norm()),
                    "delta_t_norm": float(delta_t.detach().norm()),
                    "delta_t_x": float(delta_t[0].detach()),
                    "delta_t_y": float(delta_t[1].detach()),
                    "delta_t_z": float(delta_t[2].detach()),
                }
            )

        _t0 = time.perf_counter()
        if config.interleave_steps > 0:
            optimize_rotation = (_step // config.interleave_steps) % 2 == 0
            if optimize_rotation:
                rot_optimizer.step()
            else:
                trans_optimizer.step()
        else:
            rot_optimizer.step()
            trans_optimizer.step()
        _acc_step += time.perf_counter() - _t0

        with torch.no_grad():
            if use_rot6d:
                _clamp_rot6d_angle(omega, config.epsilon_rotation_rad)
            else:
                omega_norm = torch.linalg.vector_norm(omega)
                if omega_norm > config.epsilon_rotation_rad:
                    omega.mul_(config.epsilon_rotation_rad / omega_norm)
            delta_t[0].clamp_(
                min=-config.epsilon_translation_xy, max=config.epsilon_translation_xy
            )
            delta_t[1].clamp_(
                min=-config.epsilon_translation_xy, max=config.epsilon_translation_xy
            )
            delta_t[2].clamp_(
                min=-config.epsilon_translation_z, max=config.epsilon_translation_z
            )

        if steps_without_improvement >= config.early_stopping_patience:
            break

    if debug_dir is not None and step_log:
        csv_path = debug_dir / "steps.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=step_log[0].keys())
            writer.writeheader()
            writer.writerows(step_log)

    return {
        "rotation": best["rotation"],
        "translation": best["translation"],
        "photometric_metrics": best["photometric_metrics"],
        "iterations": n_steps,
        "best_iteration": best_iteration,
        "t_render": _acc_render,
        "t_photo_loss": _acc_photo_loss,
        "t_backward": _acc_backward,
        "t_step": _acc_step,
    }
