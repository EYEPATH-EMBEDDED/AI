"""
MiDaS fine-tune on ZED depth
"""
import os, yaml, math, argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import DPTImageProcessor

from dataset.zed_depth_dataset import ZEDDepthDataset
from models.midas_dpt_loader import load_midas_model
from utils.depth_metrics import masked_l1, rmse, silog, delta_accuracy
from utils.misc import median_scaling        # ← 패치된 버전 사용

# ---------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/paths.yaml")
    p.add_argument("--out", default="runs/midas_zed")
    return p.parse_args()

def as_float(cfg, k, d=None): return float(cfg.get(k, d))
def as_int  (cfg, k, d=None): return int(  cfg.get(k, d))

# ---------------------------------------------
def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.cfg))

    cfg["learning_rate"] = as_float(cfg, "learning_rate", 1e-4)
    cfg["batch_size"]    = as_int(  cfg, "batch_size",    4)
    cfg["epochs"]        = as_int(  cfg, "epochs",       20)
    cfg["max_depth"]     = as_float(cfg, "max_depth",   10.0)
    cfg["val_split"]     = as_float(cfg, "val_split",   0.1)
    cfg["num_workers"]   = as_int(  cfg, "num_workers",  4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Dataset ----------
    full_ds  = ZEDDepthDataset(cfg["data_root"], split="all",
                               max_depth=cfg["max_depth"],
                               val_ratio=cfg["val_split"])
    vlen     = int(len(full_ds) * cfg["val_split"])
    tlen     = len(full_ds) - vlen
    train_ds, val_ds = random_split(full_ds, [tlen, vlen],
                                    generator=torch.Generator().manual_seed(42))

    train_ld = DataLoader(train_ds, batch_size=cfg["batch_size"],
                          shuffle=True,  num_workers=cfg["num_workers"],
                          pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                          shuffle=False, num_workers=cfg["num_workers"],
                          pin_memory=True)

    # ---------- Model ----------
    model     = load_midas_model().to(device)
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    opt       = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    writer    = SummaryWriter(args.out)
    Path("checkpoints").mkdir(exist_ok=True)

    best_rmse = math.inf

    # ---------- Epoch loop ----------
    for ep in range(cfg["epochs"]):
        for phase, loader in (("train", train_ld), ("val", val_ld)):
            model.train(phase == "train")
            agg = {"l1":0,"rmse":0,"silog":0,"delta":0,"n":0}

            for it, (img, gt, mask) in enumerate(loader):
                B = img.size(0)
                img_np = [x.permute(1,2,0).numpy() for x in img]
                inp = processor(images=img_np, return_tensors="pt",
                                do_rescale=False).to(device)
                gt, mask = gt.to(device), mask.to(device)

                with torch.set_grad_enabled(phase=="train"):
                    pred = model(**inp).predicted_depth
                    pred = median_scaling(pred, gt, mask)     # (B,1,H,W)
                    loss = masked_l1(pred, gt, mask)
                    if phase=="train":
                        opt.zero_grad(); loss.backward(); opt.step()

                # ---- metrics ----
                agg["l1"]   += loss.item() * B
                agg["rmse"] += rmse(pred, gt, mask).item()   * B
                agg["silog"]+= silog(pred, gt, mask).item()  * B
                agg["delta"]+= delta_accuracy(pred, gt, mask).item()* B
                agg["n"]    += B

                # ---- visualize first val batch ----
                if phase == "val" and it == 0:
                    import torchvision
                    grid = lambda x: torchvision.utils.make_grid(
                        x, nrow=min(x.size(0), 2), normalize=True, scale_each=True
                    )
                    writer.add_image("val/input", grid(img.float() / 255.), ep)   # ← 수정
                    writer.add_image("val/gt",    grid(gt),          ep)
                    writer.add_image("val/pred",  grid(pred),        ep)



            for k in ("l1","rmse","silog","delta"):
                agg[k] /= agg["n"]

            print(f"[{ep:02d}] {phase:5} "
                  f"L1 {agg['l1']:.3f}  RMSE {agg['rmse']:.3f}  "
                  f"SIlog {agg['silog']:.3f}  δ1 {agg['delta']:.3f}")

            writer.add_scalar(f"{phase}/L1",    agg["l1"],   ep)
            writer.add_scalar(f"{phase}/RMSE",  agg["rmse"], ep)
            writer.add_scalar(f"{phase}/SIlog", agg["silog"],ep)
            writer.add_scalar(f"{phase}/delta", agg["delta"],ep)

            # ---- checkpoint ----
            if phase=="val" and agg["rmse"] < best_rmse:
                best_rmse = agg["rmse"]
                p = f"checkpoints/best_ep{ep:03d}_rmse{best_rmse:.3f}.pth"
                torch.save({"epoch":ep,"model":model.state_dict(),
                            "opt":opt.state_dict(),"cfg":cfg}, p)
                print(f"  ↳  best model saved to {p}")

        scheduler.step()
    writer.close()

if __name__ == "__main__":
    main()
