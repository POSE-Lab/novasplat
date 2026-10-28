import torch
from fused_ssim import fused_ssim
from pytorch_msssim import MS_SSIM


def build_ms_ssim(channels: int, device: torch.device) -> MS_SSIM:
    # Use 3 scales (default is 5) so MS-SSIM works for objects as small as
    # 11 * 2^2 = 44px — all rendered objects in this dataset are ≥ 67px.
    # The 5-scale default requires ≥ 176px and silently fails for 14/18 objects.
    weights = [0.0448, 0.2856, 0.3001]
    return MS_SSIM(
        data_range=1.0,
        size_average=True,
        channel=channels,
        spatial_dims=2,
        weights=weights,
    ).to(device)


def masked_photometric_loss(
    predicted_bchw: torch.Tensor,
    target_bchw: torch.Tensor,
    composite_mask_bchw: torch.Tensor,
    ms_ssim_fn: MS_SSIM | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_masked = predicted_bchw * composite_mask_bchw
    tgt_masked = target_bchw * composite_mask_bchw

    ssim_tensor = fused_ssim(pred_masked, tgt_masked, padding="valid")

    try:
        if ms_ssim_fn is None:
            ms_ssim_fn = build_ms_ssim(pred_masked.shape[1], pred_masked.device)
        ms_ssim_tensor = ms_ssim_fn(pred_masked, tgt_masked)
    except (RuntimeError, ValueError):
        ms_ssim_tensor = torch.ones_like(ssim_tensor)

    loss = (1.0 - ssim_tensor) + (1.0 - ms_ssim_tensor)

    with torch.no_grad():
        metrics = {
            "ssim": float(ssim_tensor.item()),
            "ms_ssim": float(ms_ssim_tensor.item()),
            "loss": float(loss.item()),
        }
    return loss, metrics
