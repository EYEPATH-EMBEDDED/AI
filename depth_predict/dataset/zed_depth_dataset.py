"""
ZEDDepthDataset
--------------
• Depth_***/ 폴더 내  8종 파일(설명서 참고) 중
   - *_left.png / *_L.png           : 단안 RGB 입력
   - *_disp16.png (or *_disp.png)   : disparity 원본(우선 disp16 사용)
   - *_confidence.png (선택)        : binary 0/255 mask, 유효 픽셀 검사용
• .conf  파일에서 fx,  baseline(mm)  추출 → depth(m) GT 생성
"""
import os
import glob
import random
import configparser
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["ZEDDepthDataset"]

# ------------------------------------------------------------
def _load_conf(conf_path: str) -> Tuple[float, float]:
    """fx, baseline(m) 반환 – LEFT_CAM_HD 사용"""
    parser = configparser.ConfigParser()
    parser.read(conf_path)
    fx        = float(parser["LEFT_CAM_HD"]["fx"])
    baseline  = float(parser["STEREO"]["BaseLine"]) / 1000.0  # mm → m
    return fx, baseline

# ------------------------------------------------------------
class ZEDDepthDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        val_ratio: float = 0.1,
        img_size: int = 384,
        max_depth: float = 10.0,
        seed: int = 42,
        augment: bool = True,
    ):
        super().__init__()
        self.root_dir  = root_dir
        self.split     = split
        self.img_size  = img_size
        self.max_depth = max_depth
        self.augment   = augment and split == "train"
        random.seed(seed)

        self.samples: List[Dict] = []
        self._scan_folders()      # ──> self.samples 채움
        self._train_val_split(val_ratio, seed)

    # --------------------
    def _scan_folders(self):
        depth_folders = sorted(glob.glob(os.path.join(self.root_dir, "Depth_*")))
        for folder in depth_folders:
            conf_path = os.path.join(folder, os.path.basename(folder) + ".conf")
            if not os.path.exists(conf_path):
                continue
            fx, baseline = _load_conf(conf_path)

            left_imgs = (
                glob.glob(os.path.join(folder, "*_left.png"))
                + glob.glob(os.path.join(folder, "*_L.png"))
            )
            for left in sorted(left_imgs):
                root_name = (
                    left.replace("_left.png", "")
                    .replace("_L.png", "")
                    .replace("_left.jpg", "")
                )
                disp16 = root_name + "_disp16.png"
                disp   = root_name + "_disp.png"
                conf   = root_name + "_confidence.png"

                if not os.path.exists(disp16) and not os.path.exists(disp):
                    continue

                self.samples.append(
                    {
                        "left": left,
                        "disp": disp16 if os.path.exists(disp16) else disp,
                        "conf": conf if os.path.exists(conf) else None,
                        "fx": fx,
                        "baseline": baseline,
                    }
                )

    # --------------------
    def _train_val_split(self, val_ratio: float, seed: int):
        if self.split not in ("train", "val", "all"):
            raise ValueError("split must be 'train'|'val'|'all'")

        if self.split == "all":
            return

        random.Random(seed).shuffle(self.samples)
        val_len = int(len(self.samples) * val_ratio)
        if self.split == "val":
            self.samples = self.samples[:val_len]
        else:  # train
            self.samples = self.samples[val_len:]

    # --------------------
    def __len__(self):
        return len(self.samples)

    # --------------------
    def _augment_hflip(self, img, depth, mask):
        """동일 random horizontal flip"""
        if random.random() < 0.5:
            img   = np.fliplr(img).copy()
            depth = np.fliplr(depth).copy()
            mask  = np.fliplr(mask).copy() if mask is not None else None
        return img, depth, mask

    def _resize(self, arr, interp=cv2.INTER_LINEAR):
        return cv2.resize(arr, (self.img_size, self.img_size), interpolation=interp)

    # --------------------
    def __getitem__(self, idx: int):
        item = self.samples[idx]
        fx, baseline = item["fx"], item["baseline"]

        # ---------- 1) RGB 이미지 ----------
        img = cv2.imread(item["left"], cv2.IMREAD_COLOR)  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # RGB  uint8 → (H,W,3)

        # ---------- 2) disparity ----------
        disp = cv2.imread(
            item["disp"],
            cv2.IMREAD_UNCHANGED,
        )  # 16-bit or 8-bit
        if disp is None:
            raise FileNotFoundError(item["disp"])
        disp = disp.astype(np.float32)
        if disp.dtype == np.uint16 or disp.max() > 255:
            disp /= 16.0  # ZED disp16 은 소수점을 ×16 해서 저장

        # ---------- 3) confidence mask ----------
        mask = None
        if item["conf"] and os.path.exists(item["conf"]):
            mask = cv2.imread(item["conf"], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.float32)  # 1(valid)/0(invalid)

        # disparity → depth(m)
        disp[disp < 0.1] = 0.1  # division zero 방지
        depth = (fx * baseline) / disp
        depth = np.clip(depth, 0.0, self.max_depth)

        # ---------- Augmentation ----------
        if self.augment:
            img, depth, mask = self._augment_hflip(img, depth, mask)

        # ---------- 리사이즈 ----------
        img   = self._resize(img,  cv2.INTER_AREA)
        depth = self._resize(depth, cv2.INTER_NEAREST)
        if mask is not None:
            mask = self._resize(mask, cv2.INTER_NEAREST)

        # ---------- Tensor 변환 ----------
        img_t   = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # uint8 [3,H,W]
        depth_t = torch.from_numpy(depth).unsqueeze(0).float()         # float32 [1,H,W]
        mask_t  = (
            torch.from_numpy(mask).unsqueeze(0).float()
            if mask is not None
            else torch.ones_like(depth_t)
        )

        return img_t, depth_t, mask_t
