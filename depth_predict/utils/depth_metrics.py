"""
Depth 평가 지표 모음
"""
import torch
import torch.nn.functional as F

__all__ = [
    "masked_l1",
    "rmse",
    "silog",
    "delta_accuracy",
]

# ---------------------------------------------
def masked_l1(pred, gt, mask):
    diff = (pred - gt).abs() * mask
    return diff.sum() / (mask.sum() + 1e-6)

def rmse(pred, gt, mask):
    diff = ((pred - gt) ** 2) * mask
    return torch.sqrt(diff.sum() / (mask.sum() + 1e-6))

def silog(pred, gt, mask, eps=1e-6):
    """
    Scale-Invariant Log MSE  (Eigen et al.)
    """
    pred = pred.clamp(min=eps)
    gt   = gt.clamp(min=eps)
    g = (torch.log(pred) - torch.log(gt)) * mask
    n = mask.sum()
    return torch.sqrt(((g ** 2).sum() / n) - ((g.sum() / n) ** 2))

def delta_accuracy(pred, gt, mask, thr=1.25):
    """
    δ = max(pred/gt, gt/pred) < thr
    """
    eps = 1e-6
    ratio = torch.max(pred / (gt + eps), gt / (pred + eps))
    valid = mask > 0.5
    return (ratio[valid] < thr).float().mean()
