# ────────────────────────────────────────────────────────────
# 충돌 가능 영역(depth) Crop & 저장  (resize X, float16)
# ────────────────────────────────────────────────────────────
import os, csv, math, pathlib, numpy as np, cv2
from tqdm import tqdm

# ====== 수정할 것 ======
CSV_IN   = 'preprocess_output/final_collision_depth.csv'   # 원본 CSV
CSV_OUT  = 'preprocess_output/final_collision_depth_crop.csv'
DST_DIR  = 'preprocess_output/depth_crop'                  # 잘라낸 npy 저장 폴더
# 카메라·사람 파라미터
HEIGHT_M = 1.70        # 사람 시야 높이   (h)
TORSO_M  = 0.45        # 몸통 너비         (a)
THETA_V  = 52.0        # 수직 FOV(deg) (θ₁)
THETA_H  = 65.0        # 수평 FOV(deg) (θ₂)
IMG_W, IMG_H = 1280, 720
# =====================

# ───────────────────── 충돌 직사각형 좌/우 X 계산 ────────────
def collision_left_right_x():
    d = HEIGHT_M * math.tan(math.radians(THETA_V) / 2)  # 그림2의 d
    ratio = TORSO_M / (2 * d * math.tan(math.radians(THETA_H) / 2))
    half  = (ratio * IMG_W) / 2                         # pixel
    cx_left  = int(round(IMG_W / 2 - half))
    cx_right = int(round(IMG_W / 2 + half))
    return cx_left, cx_right                             # inclusive 시작, exclusive 끝

LEFT_X, RIGHT_X = collision_left_right_x()
CROP_W = RIGHT_X - LEFT_X   # 최종 폭

print(f"[INFO] crop range x ∈ [{LEFT_X}, {RIGHT_X})  ->  shape = (360, {CROP_W})")

# ───────────────────── 1) CSV 읽기 ──────────────────────────
with open(CSV_IN, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

# ───────────────────── 2) depth 파일 처리 ───────────────────
for r in tqdm(rows, desc='Cropping depth'):
    src = r['depth_path']               # 원본 npy (720×1280 float32)
    rel = pathlib.Path(src).relative_to(pathlib.Path(src).anchor)  # 보존할 하위구조
    dst = pathlib.Path(DST_DIR) / rel
    dst = dst.with_suffix('').with_name(dst.stem + '_c16.npy')     # *_c16.npy

    if not dst.exists():
        # load (mmap = False => 바로 배열)  / mmap_mode='r' 도 OK
        arr = np.load(src)                       # (720,1280)
        crop = arr[0:360, LEFT_X:RIGHT_X]        # (360, CROP_W)
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, crop.astype(np.float16), allow_pickle=False)

    r['depth_path'] = str(dst)                   # CSV 경로 갱신

# ───────────────────── 3) 새 CSV 저장 ───────────────────────
with open(CSV_OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

print(f"[DONE] {len(rows)} rows  ->  {CSV_OUT}")
