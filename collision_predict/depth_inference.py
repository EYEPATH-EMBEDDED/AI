
#!/usr/bin/env python
"""
depth_inference.py
──────────────────
1280×720 RGB 프레임 → MiDaS depth  → *.npy 저장  +  CSV 갱신
(설정은 파일 상단 CONFIG 블록에서 직접 수정)
"""
from pathlib import Path
import re, csv, sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import DPTForDepthEstimation, DPTImageProcessor

# ───────────── CONFIG ────────────────────────────────────────────────
CKPT_PATH  = Path("preprocess_output/depth.pth")
IMAGES_DIR = Path("images")                            # src/images/...
IN_CSV     = Path("preprocess_output/final_collision.csv")
OUT_CSV    = Path("preprocess_output/final_collision_depth.csv")
DEPTH_DIR  = Path("preprocess_output/depth_pred")
MAX_DEPTH  = 10.0                                     # m
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────


def load_model(ckpt: Path, device: str):
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
    model.to(device).eval()

    if not ckpt.exists():
        sys.exit(f"❌ checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    return model, processor


@torch.no_grad()
def infer_depth(img_bgr: np.ndarray, model, processor, device) -> np.ndarray:
    """
    입력: 1280×720 BGR uint8
    출력: 동일 해상도 float32 depth (0–MAX_DEPTH m)
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt",
                       do_rescale=False).to(device)        # → 384×384
    inv = model(**inputs).predicted_depth                   # (1,H,W) 3-D
    inv = inv.unsqueeze(1)                                  # (1,1,H,W)

    # frame-wise 0-1 정규화 후 선형 매핑
    inv = inv / (inv.max() + 1e-6)
    depth = MAX_DEPTH * (1.0 - inv)                         # (1,1,H,W)

    # 720×1280 로 업샘플
    depth = F.interpolate(depth,
                          size=(img_bgr.shape[0], img_bgr.shape[1]),
                          mode="bilinear", align_corners=False)
    return depth.squeeze().cpu().numpy().astype("float32")   # (H,W)


def list_frames(root: Path):
    pat = re.compile(r"(\d+)[/_](\d+)\.jpg$")
    frames = []
    for jpg in root.rglob("*.jpg"):
        m = pat.search(jpg.as_posix())
        if m:
            vid, frm = m.group(1), int(m.group(2))
            frames.append((vid, frm, jpg))
    return frames


def main():
    device = torch.device(DEVICE)
    model, processor = load_model(CKPT_PATH, device)

    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    frames = list_frames(Path(__file__).parent / IMAGES_DIR)
    print(f"🔎 {len(frames)} frames found — starting depth inference")

    cache = {}   # (vid,frame) ➜ path
    for vid, frm, jpg_path in frames:
        out_dir  = DEPTH_DIR / vid
        out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = out_dir / f"{vid}_{frm}.npy"
        cache[(vid, frm)] = str(npy_path)

        if npy_path.exists():
            continue

        img = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
        depth = infer_depth(img, model, processor, device)
        np.save(npy_path, depth)

    print("✅ depth maps saved")

    # ---------- CSV update ----------
    with open(IN_CSV, newline="") as fi, open(OUT_CSV, "w", newline="") as fo:
        rdr = csv.DictReader(fi)
        fields = rdr.fieldnames + ["depth_path"]
        wtr = csv.DictWriter(fo, fieldnames=fields)
        wtr.writeheader()

        for row in rdr:
            vid   = row["video"].replace(".mp4", "")
            frm   = int(row["frame"])
            row["depth_path"] = cache.get((vid, frm), "")
            wtr.writerow(row)

    print(f"✅ CSV updated → {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()




# #!/usr/bin/env python
# """
# depth_inference.py  (src 폴더에 위치)

# ● src/images/<vid>/<vid>_<frame>.jpg   =>   depth_pred/<vid>/<vid>_<frame>.npy
# ● preprocess_output/final_collision_depth.csv  생성( depth_path 열 추가 )

# ※ 스크립트 내부 CONFIG 변수만 고치면 되며, argparse/CLI 없이 실행합니다.
# """

# from pathlib import Path
# import csv, re, sys

# import cv2
# import numpy as np
# import torch
# import torch.nn.functional as F
# from transformers import DPTForDepthEstimation, DPTImageProcessor


# # ---------------------------------------------------------------------
# # CONFIG ───────────────────────────────────────────────────────────────
# CKPT_PATH   = Path("preprocess_output/depth.pth")
# IMAGES_DIR  = Path("images")                      # src 기준
# IN_CSV      = Path("preprocess_output/final_collision.csv")
# OUT_CSV     = Path("preprocess_output/final_collision_depth.csv")
# DEPTH_DIR   = Path("preprocess_output/depth_pred")
# MAX_DEPTH   = 10.0                               # m
# DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# # ---------------------------------------------------------------------


# def load_finetuned_midas(ckpt: Path, device: torch.device):
#     base = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
#     base.to(device).eval()

#     if not ckpt.is_file():
#         sys.exit(f"❌ checkpoint not found: {ckpt}")
#     ckpt_data = torch.load(ckpt, map_location=device)
#     base.load_state_dict(ckpt_data["model"], strict=False)

#     proc = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
#     return base, proc


# @torch.no_grad()
# def predict_depth(img_bgr, model, processor, device, max_depth=10.0):
#     rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     inp   = processor(images=rgb, return_tensors="pt", do_rescale=False).to(device)
#     inv   = model(**inp).predicted_depth.squeeze(0)          # (1,h,w) inverse depth
#     inv   = inv / (inv.max() + 1e-6)
#     depth = max_depth * (1.0 - inv)

#     depth = F.interpolate(depth.unsqueeze(0),
#                            size=img_bgr.shape[:2],  # (H,W)=720,1280
#                            mode="bilinear", align_corners=False)
#     return depth.squeeze().cpu().numpy().astype("float32")


# def scan_frames(root: Path):
#     """
#     반환: [(vid, frame, jpg_path), ...]
#     """
#     pat = re.compile(r"(\d+)[/_](\d+)\.jpg$")
#     items = []
#     for jpg in root.rglob("*.jpg"):
#         m = pat.search(jpg.as_posix())
#         if not m:
#             continue
#         vid, frame = m.group(1), int(m.group(2))
#         items.append((vid, frame, jpg))
#     return items


# def main():
#     device = torch.device(DEVICE)
#     model, processor = load_finetuned_midas(CKPT_PATH, device)

#     DEPTH_DIR.mkdir(parents=True, exist_ok=True)
#     frames = scan_frames(Path(__file__).parent / IMAGES_DIR)
#     print(f"🔎 총 {len(frames)} 프레임 depth 추론…")

#     depth_cache = {}  # (vid,frame) -> str path

#     for vid, frame, jpg in frames:
#         out_dir  = DEPTH_DIR / vid
#         out_dir.mkdir(parents=True, exist_ok=True)
#         npy_path = out_dir / f"{vid}_{frame}.npy"
#         depth_cache[(vid, frame)] = str(npy_path)

#         if npy_path.exists():           # 이미 존재하면 건너뜀
#             continue
#         img   = cv2.imread(str(jpg), cv2.IMREAD_COLOR)
#         depth = predict_depth(img, model, processor, device, MAX_DEPTH)
#         np.save(npy_path, depth)

#     print("✅ depth 저장 완료")

#     # ---- CSV 업데이트 --------------------------------------------------
#     with open(IN_CSV, newline="") as f_in, open(OUT_CSV, "w", newline="") as f_out:
#         reader = csv.DictReader(f_in)
#         fields = reader.fieldnames + ["depth_path"]
#         writer = csv.DictWriter(f_out, fieldnames=fields)
#         writer.writeheader()

#         for row in reader:
#             vid   = row["video"].replace(".mp4", "")
#             frame = int(row["frame"])
#             row["depth_path"] = depth_cache.get((vid, frame), "")
#             writer.writerow(row)

#     print(f"✅ CSV 갱신 → {OUT_CSV.resolve()}")


# if __name__ == "__main__":
#     main()
