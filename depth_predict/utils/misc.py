import torch

def _ensure_4d(t):
    """(B,H,W) → (B,1,H,W) 로 맞춰 줌"""
    return t.unsqueeze(1) if t.dim() == 3 else t

def median_scaling(pred, gt, mask):
    """
    배치 단위 median-scale 정합
    pred, gt : (B,1,H,W) 또는 (B,H,W)
    mask     : 동일 크기, 0/1
    """
    pred = _ensure_4d(pred)
    gt   = _ensure_4d(gt)
    mask = _ensure_4d(mask)

    valid = mask > 0.5
    scale = torch.median(gt[valid]) / (torch.median(pred[valid]) + 1e-6)
    return pred * scale            # (B,1,H,W)
