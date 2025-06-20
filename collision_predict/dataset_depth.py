import os, math, csv, random, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from shapely.geometry import Polygon
import cv2                           # depth crop-resize 용
from collections import defaultdict

############  충돌 삼각형(좌표) ################################################
def _collision_poly(height_m=1.70, torso_m=0.45,
                    theta_v=52.0, theta_h=65.0):
    d = height_m * math.tan(math.radians(theta_v)/2)
    ratio = torso_m / (2*d*math.tan(math.radians(theta_h)/2))
    half_base = (ratio * 1280.0) / 2          # px
    left_x  = int(640 - half_base)
    right_x = int(640 + half_base)
    return left_x, right_x                    # (430 ~ 850 정도), 고정

LEFT_X, RIGHT_X = _collision_poly()           # 전역 캐시
CROP_H, CROP_W  = 180, 360                   # 그림에서 지정한 최종 해상도

####################  DataSet ##################################################
class DepthCollisionDataset(Dataset):
    """
    각 sample:
        depth_tensor : (1 , 1200 , 180 , 360)  ← float32, meter 단위
        feat_tensor  : (1200 , 6)              ← time,x1,y1,x2,y2,overlap
        label        :  0 / 1
    """
    FEAT_LIST = ['time','x1','y1','x2','y2','overlap_ratio']   # 6D

    def __init__(self, csv_rows, queue_size=1200,
                 after_start=50, after_finish=400):
        self.q  = queue_size
        self.a1 = after_start
        self.a2 = after_finish
        self.vd = self._group(csv_rows)
        self.samples = self._make_samples()

    # ---------- util ---------------------------------------------------------
    def _group(self, rows):
        vd = defaultdict(list)
        for r in rows: vd[r['video']].append(r)
        for v in vd: vd[v].sort(key=lambda x:int(x['frame']))
        return vd

    def _load_depth(self, npy_path):
        arr = np.load(npy_path)               # (720,1280)
        crop = arr[0:360, LEFT_X:RIGHT_X]     # (360, 약420)
        crop = cv2.resize(crop, (CROP_W, CROP_H),
                          interpolation=cv2.INTER_AREA)
        return crop.astype('float32')         # (180,360)

    # ---------- sample build --------------------------------------------------
    def _make_samples(self):
        sp = []
        for vid, rows in self.vd.items():
            N = len(rows)
            labels = [int(r['label']) for r in rows]
            minT   = min(float(r['time']) for r in rows)

            for st in range(N-self.q):
                ed = st + self.q
                # ---- label (future window) ------------------
                fs = ed + self.a1
                fe = min(ed + self.a2, N-1)
                y  = int(any(labels[fs:fe+1])) if fs < N else 0

                # ---- depth tensor ---------------------------
                depth_list = [self._load_depth(rows[i]['depth_path'])
                              for i in range(st, ed)]
                depth_arr  = np.stack(depth_list, axis=0)      # (T,180,360)
                depth_arr  = depth_arr.clip(0,10)/10.0         # 0~1 정규화
                depth_arr  = depth_arr[None, ...]              # (1,T,H,W)

                # ---- feature tensor -------------------------
                feats=[]
                T_last = float(rows[ed-1]['time'])
                for i in range(st, ed):
                    r   = rows[i]
                    dt  = (T_last - float(r['time']))/max(1e-6,
                            (T_last-minT))
                    feats.append([
                        dt,
                        float(r['x1'])/1280.0,
                        float(r['y1'])/720.0,
                        float(r['x2'])/1280.0,
                        float(r['y2'])/720.0,
                        float(r['overlap_ratio'])
                    ])
                feat_arr = np.asarray(feats, dtype='float32')   # (T,6)

                sp.append(dict(
                    video=vid,
                    rep_frame=int(rows[ed-1]['frame']),
                    depth=depth_arr,
                    feat =feat_arr,
                    label=y))
        return sp
    # -------------------------------------------------------------------------
    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return torch.from_numpy(s['depth']), \
               torch.from_numpy(s['feat']),  \
               torch.tensor(s['label'], dtype=torch.long)
    def get_frame(self,idx): return self.samples[idx]['rep_frame']
    def get_video(self,idx): return self.samples[idx]['video']
