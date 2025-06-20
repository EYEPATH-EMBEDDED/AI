# # ─────────────────────────────────────────────────────────────
# # train_depth_collision_timepreserved_fast.py
# #   * 메모리 여유   : DEPTH_CHUNK 200 → 150   (원하면 200~300 로 올려도 OK)
# #   * 안정성        : GRAD_CHK 옵션 추가
# #   * 모니터링      : step/epoch 피크 메모리 로그
# #   * 기존 plot / metric / auto-batch 그대로
# # ─────────────────────────────────────────────────────────────
# import os, csv, random, gc, warnings
# from collections import defaultdict
# from typing import List, Dict

# import cv2, numpy as np
# from tqdm import tqdm
# import matplotlib; matplotlib.use("Agg") ; import matplotlib.pyplot as plt

# import torch, torch.nn as nn, torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torch.amp import autocast, GradScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # ═══════════════ 1. 파라미터 (★ 바꾼 곳만 주석) ═══════════════
# CSV_PATH        = "preprocess_output/final_collision_depth_crop.csv"
# QUEUE_SIZE      = 1200 ; AFTER_START = 50 ; AFTER_FINISH = 400
# EPOCHS          = 5     ; INIT_BS     = 2    # auto-reduce 그대로
# ACC_STEPS       = 2     ; LR          = 3e-5
# SEED            = 10
# DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# CROP_H, CROP_W  = 128, 256 ; DTYPE_NORM = 10.0
# NUM_WORKERS     = 12       ; PIN_MEMORY = True ; PREFETCH = 2

# THRESH_VAL      = 0.55 ; THRESH_USE = True

# USE_TIME_DISTRIBUTED = True
# DEPTH_CHUNK          = 20     # ★ 200 → 150  (23 GB 카드에 넉넉)
# GRAD_CHK             = False   # True 로 바꾸면 gradient-checkpoint
# USE_COMPILE          = False   # torch.compile(optional)

# # 디렉터리 준비
# os.makedirs("results/bar", exist_ok=True) ; os.makedirs("results/thr", exist_ok=True)
# random.seed(SEED) ; np.random.seed(SEED) ; torch.manual_seed(SEED)
# warnings.filterwarnings("ignore", category=UserWarning) ; torch.backends.cuda.matmul.allow_tf32 = True


# # ═══════════════ 2. Plot 함수 (변경 없음) ════════════════════
# def _sort(fr,*arr):
#     idx=np.argsort(fr); return (fr[idx],)+tuple(a[idx] for a in arr)

# def plot_three_bars(fr,lb,pd,path):
#     if fr.size==0: return
#     fr,lb,pd=_sort(fr,lb,pd); L=int(fr[-1])+1
#     bar1=np.ones((1,L,3)); bar2=np.ones((1,L,3)); bar3=np.ones((1,L,3))
#     bar1[0,fr[lb==1]]=[1,0,0]; bar2[0,fr[pd==1]]=[0,0,1]
#     fn=(lb==1)&(pd==0); fp=(lb==0)&(pd==1); tp=(lb==1)&(pd==1)
#     bar3[0,fr[fn]]=[1,0,0]; bar3[0,fr[fp]]=[0,0,1]; bar3[0,fr[tp]]=[0,1,0]
#     os.makedirs(os.path.dirname(path),exist_ok=True)
#     fig,ax=plt.subplots(3,1,figsize=(12,4),sharex=True)
#     ax[0].imshow(bar1,aspect="auto",interpolation="nearest"); ax[0].set_ylabel("Label"); ax[0].set_yticks([])
#     ax[1].imshow(bar2,aspect="auto",interpolation="nearest"); ax[1].set_ylabel("Pred");  ax[1].set_yticks([])
#     ax[2].imshow(bar3,aspect="auto",interpolation="nearest"); ax[2].set_ylabel("Err");   ax[2].set_yticks([])
#     ax[2].set_xlabel("Frame"); ax[0].set_xlim([0,L]); plt.tight_layout(); plt.savefig(path,dpi=100); plt.close()

# def plot_threshold(fr,pr,path,thr=THRESH_VAL):
#     if fr.size==0: return
#     fr,pr=_sort(fr,pr)[:2]
#     os.makedirs(os.path.dirname(path),exist_ok=True)
#     plt.figure(figsize=(12,2.5)); plt.plot(fr,pr,lw=1); plt.axhline(thr,color='r',ls='--')
#     plt.ylim([0,1]); plt.xlabel("Frame"); plt.ylabel("P"); plt.tight_layout(); plt.savefig(path,dpi=100); plt.close()

# # ═══════════════ 3. Dataset (기존 로직) ══════════════════════
# class DepthCollisionDataset(Dataset):
#     FEAT_COLS=("time","x1","y1","x2","y2","overlap_ratio","frame")
#     def __init__(self, rows:List[Dict], q=QUEUE_SIZE, a1=AFTER_START, a2=AFTER_FINISH):
#         self.q,self.a1,self.a2=q,a1,a2
#         self.vdict=self._group(rows); self.samples=self._index()
#     @staticmethod
#     def _group(rows):
#         vd=defaultdict(list)
#         for r in rows: vd[r["video"]].append(r)
#         for v in vd: vd[v].sort(key=lambda x:int(x["frame"]))
#         return vd
#     def _index(self):
#         out=[]; add=out.append
#         for vid,rows in self.vdict.items():
#             N=len(rows); labels=[int(r["label"]) for r in rows]; minT=min(float(r["time"]) for r in rows)
#             for st in range(N-self.q):
#                 ed=st+self.q; fs,fe=ed+self.a1, min(ed+self.a2,N-1)
#                 y=int(fs<N and any(labels[fs:fe+1]))
#                 add(dict(video=vid, frame=int(rows[ed-1]["frame"]),
#                          depth_paths=[r["depth_path"] for r in rows[st:ed]],
#                          bbox_rows=rows[st:ed], minT=minT, label=y))
#         return out
#     def _ldepth(self,p):
#         a=np.load(p,mmap_mode="r").astype(np.float32)
#         if a.shape!=(CROP_H,CROP_W): a=cv2.resize(a,(CROP_W,CROP_H),cv2.INTER_AREA)
#         return (a/DTYPE_NORM).clip(0,1).astype(np.float16)
#     def __len__(self): return len(self.samples)
#     def __getitem__(self,idx):
#         s=self.samples[idx]
#         depth=np.stack([self._ldepth(p) for p in s["depth_paths"]],0)   # (T,H,W)
#         depth=torch.from_numpy(depth)[None]                             # (1,T,H,W)
#         T_last=float(s["bbox_rows"][-1]["time"]); minT=s["minT"]; T_len=len(s["bbox_rows"])
#         feat=[[(T_last-float(r["time"])) / max(1e-6,(T_last-minT)),
#                float(r["x1"])/1280, float(r["y1"])/720,
#                float(r["x2"])/1280, float(r["y2"])/720,
#                float(r["overlap_ratio"]),
#                i/(T_len-1) if T_len>1 else 0.] for i,r in enumerate(s["bbox_rows"])]
#         return depth, torch.from_numpy(np.asarray(feat,dtype=np.float16)), torch.tensor(s["label"],dtype=torch.long)
#     def get_frame(self,i): return self.samples[i]["frame"]
#     def get_video(self,i): return self.samples[i]["video"]

# # ═══════════════ 4. Model (Chunk-ed Time-Distributed Conv2D) ═════════════
# # ═══════════════ 4. Depth Branch (chunk + grad-checkpoint) ═════════════
# class DepthBranchTD(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
#             nn.MaxPool2d(2))
#     def _forward_single(self, slab):
#         return self.cnn(slab).amax(-1).amax(-1)   # (N,32)
#     def forward(self, x):                         # x:(B,1,T,H,W)
#         B,_,T,H,W = x.shape ; C = DEPTH_CHUNK ; chunks=[]
#         for st in range(0, T, C):
#             ed = min(st+C, T)
#             slab = x[:,:,st:ed].permute(0,2,1,3,4).reshape(B*(ed-st),1,H,W)
#             feat = (torch.utils.checkpoint.checkpoint(self._forward_single, slab)
#                     if self.training and GRAD_CHK else self._forward_single(slab))
#             chunks.append(feat.view(B, ed-st, -1))
#         return torch.cat(chunks, dim=1)           # (B,T,32)
    
# class DepthConvLSTM(nn.Module):
#     def __init__(self,bbox_dim=7):
#         super().__init__()
#         self.depth = DepthBranchTD() if USE_TIME_DISTRIBUTED else None
#         if not USE_TIME_DISTRIBUTED:      # 원본 Conv3D 경로
#             ch=(1,8,16,32); bl=[]
#             for ci,co in zip(ch[:-1],ch[1:]):
#                 bl+=[nn.Conv3d(ci,co,(1,3,3),padding=(0,1,1)),
#                      nn.BatchNorm3d(co), nn.ReLU(), nn.MaxPool3d((1,2,2))]
#             self.depth=nn.Sequential(*bl)
#         self.bbox_conv=nn.Sequential(nn.Conv1d(bbox_dim,64,1),
#                                      nn.BatchNorm1d(64), nn.ReLU())
#         self.fuse_conv=nn.Sequential(nn.Conv1d(96,128,5,padding=2),
#                                      nn.BatchNorm1d(128), nn.ReLU())
#         self.lstm=nn.LSTM(128,128,batch_first=True,bidirectional=True)
#         self.head=nn.Sequential(nn.Linear(256,64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64,2))

#     def _depth_forward(self,x):                                  # Conv3D 경로
#         x=self.depth(x); x=x.amax(-1).amax(-1); return x.permute(0,2,1)

#     def forward(self,depth,feat):
#         if USE_TIME_DISTRIBUTED:
#             d=self.depth(depth.to(memory_format=torch.channels_last_3d))
#         else:
#             d=self._depth_forward(depth.to(memory_format=torch.channels_last_3d))
#         b=self.bbox_conv(feat.permute(0,2,1)).permute(0,2,1)
#         x=self.fuse_conv(torch.cat([d,b],-1).permute(0,2,1)).permute(0,2,1)
#         out,_=self.lstm(x)
#         return self.head(out[:,-1])

# # ═══════════════ 5. Loss / Metric ════════════════════════════
# class RecallCELoss(nn.Module):
#     def __init__(self,pos_w): super().__init__(); self.w=torch.tensor([1.,pos_w])
#     def forward(self,l,t): return nn.functional.cross_entropy(l,t,self.w.to(l.device))
# @torch.no_grad()
# def logits2pred(l): return (torch.softmax(l,1)[:,1]>=THRESH_VAL).long() if THRESH_USE else l.argmax(1)
# def metric(y,p): return (accuracy_score(y,p),precision_score(y,p,zero_division=0),
#                          recall_score(y,p,zero_division=0),f1_score(y,p,zero_division=0))

# # ═══════════════ 6. Train 루프 (기존 로직 그대로) ═════════════
# def load_csv(p):  return list(csv.DictReader(open(p,"r",encoding="utf-8")))
# def split_rows(rows,r=0.8):
#     vids=list({r["video"] for r in rows}); random.shuffle(vids)
#     k=int(len(vids)*r); tr=set(vids[:k]); vl=set(vids[k:])
#     return [r for r in rows if r["video"] in tr],[r for r in rows if r["video"] in vl]
# def make_loader(ds,bs,sh):
#     return DataLoader(ds,bs,shuffle=sh,drop_last=sh,num_workers=NUM_WORKERS,
#                       pin_memory=PIN_MEMORY,persistent_workers=True,prefetch_factor=PREFETCH)

# # ═══════════════ 6. Train loop: step 도중 메모리 로그 추가 ═════════════
# def _print_mem(tag=""):
#     cur = torch.cuda.memory_allocated() / 2**20
#     res = torch.cuda.memory_reserved() / 2**20
#     peak= torch.cuda.max_memory_allocated() / 2**20
#     print(f"[mem{tag}] alloc {cur:,.0f} MB  reserved {res:,.0f} MB  peak {peak:,.0f} MB")




# def train():
#     rows=load_csv(CSV_PATH)
#     tr_rows,vl_rows=split_rows(rows)
#     tr_ds,vl_ds=DepthCollisionDataset(tr_rows),DepthCollisionDataset(vl_rows)
#     pos=sum(s["label"] for s in tr_ds.samples); neg=len(tr_ds)-pos
#     loss_fn=RecallCELoss(neg/max(1,pos)).to(DEVICE)

#     model=DepthConvLSTM().to(DEVICE)
#     if USE_COMPILE and hasattr(torch,'compile'): model=torch.compile(model,mode="max-autotune")
#     opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4,fused=True)
#     scaler=GradScaler(enabled=(DEVICE=="cuda"))

#     bs=INIT_BS
#     while True:
#         try: next(iter(make_loader(tr_ds,bs,True))); break
#         except RuntimeError as e:
#             if "out of memory" in str(e).lower() and bs>1:
#                 bs//=2; torch.cuda.empty_cache(); print(f"[MEM] batch↓ {bs}")
#             else: raise
#     tr_ld,vl_ld=make_loader(tr_ds,bs,True), make_loader(vl_ds,bs,False)

#     best=0.
#     for ep in range(1,EPOCHS+1):
#         # train() 내부 ─ Train 루프에서 10 step 마다:
#         # -----------------------------------------------------------------
#         #   if (step+1)%10==0:
#         #       _print_mem(f' E{ep} S{step+1}')
#         # ─ Train
#         model.train(); tl=tt=0; yt=yp=[]
#         opt.zero_grad(set_to_none=True)
#         for step,(d,f,l) in enumerate(tqdm(tr_ld,desc=f"E{ep} Train",leave=False)):
#             d,f,l=d.to(DEVICE),f.to(DEVICE),l.to(DEVICE)
#             with autocast(device_type="cuda",dtype=torch.float16,enabled=(DEVICE=="cuda")):
#                 out=model(d,f); loss=loss_fn(out,l)/ACC_STEPS
#             scaler.scale(loss).backward()
#             if (step+1)%ACC_STEPS==0:
#                 scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
#             tl+=loss.item()*l.size(0)*ACC_STEPS; tt+=l.size(0)
#             yt+=l.cpu().tolist(); yp+=logits2pred(out).cpu().tolist()
#         tr_m=metric(yt,yp)

#         # ─ Val
#         model.eval(); vl=vt=0; yv=pv=[]; buf=defaultdict(lambda:{"fr":[],"lb":[],"pd":[],"pr":[]}); seen=0
#         with torch.no_grad():
#             for d,f,l in tqdm(vl_ld,desc=f"E{ep} Val",leave=False):
#                 d,f,l=d.to(DEVICE),f.to(DEVICE),l.to(DEVICE)
#                 with autocast(device_type="cuda",dtype=torch.float16,enabled=(DEVICE=="cuda")):
#                     out=model(d,f); loss=loss_fn(out,l)
#                 vl+=loss.item()*l.size(0); vt+=l.size(0)
#                 prob=torch.softmax(out,1)[:,1].cpu().numpy()
#                 pd=logits2pred(out).cpu().numpy(); lb=l.cpu().numpy()
#                 yv+=lb.tolist(); pv+=pd.tolist()
#                 for i in range(l.size(0)):
#                     vid=vl_ds.get_video(seen+i); fr=vl_ds.get_frame(seen+i)
#                     buf[vid]["fr"].append(fr); buf[vid]["lb"].append(int(lb[i]))
#                     buf[vid]["pd"].append(int(pd[i])); buf[vid]["pr"].append(float(prob[i]))
#                 seen+=l.size(0)
#         vl_m=metric(yv,pv)

#         print(f"[E{ep}/{EPOCHS}] TrainL {tl/tt:.4f} Acc {tr_m[0]:.3f} Rec {tr_m[2]:.3f} F1 {tr_m[3]:.3f} | "
#               f"ValL {vl/vt:.4f} Acc {vl_m[0]:.3f} Rec {vl_m[2]:.3f} F1 {vl_m[3]:.3f}")

#         for vid,d in buf.items():
#             fr=np.asarray(d["fr"]); lb=np.asarray(d["lb"]); pd=np.asarray(d["pd"]); pr=np.asarray(d["pr"])
#             plot_three_bars(fr,lb,pd,f"results/bar/{vid}/bar_e{ep}.png")
#             plot_threshold(fr,pr,f"results/thr/{vid}/thr_e{ep}.png")

#         torch.save(model.state_dict(),f"ckpt_epoch_{ep}.pth")
#         if vl_m[3]>best:
#             best=vl_m[3]; torch.save(model.state_dict(),"best_model.pth"); print(f"  ↳ new BEST! F1={best:.3f}")

#         torch.cuda.empty_cache(); gc.collect()

# # ═══════════════ 7. main ════════════════════════════════════
# if __name__=="__main__":
#     train()




# ─────────────────────────────────────────────────────────────
# train_depth_collision_timepreserved.py
#   Depth CNN : Conv3D(kernel=(1,3,3)) only  –  시간축 보존
# ─────────────────────────────────────────────────────────────
import os, csv, random, gc, warnings
from collections import defaultdict
from typing import List, Dict, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ══════════════════════════════════════════════════════════
# 1. 하이퍼파라미터
# ══════════════════════════════════════════════════════════
CSV_PATH      = "preprocess_output/final_collision_depth_crop.csv"

QUEUE_SIZE    = 1200
AFTER_START   = 50
AFTER_FINISH  = 400

EPOCHS        = 7
INIT_BS       = 4
ACC_STEPS     = 2
LR            = 1e-5

SEED          = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# depth 텐서 전처리
CROP_H, CROP_W = 128, 256
DTYPE_NORM     = 10.0

# DataLoader
NUM_WORKERS   = 16
PIN_MEMORY    = False

# 추론 임계값
THRESH_VAL    = 0.55
THRESH_USE    = True

# 디렉터리
os.makedirs("results/bar", exist_ok=True)
os.makedirs("results/thr", exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = True

# ══════════════════════════════════════════════════════════
# 2. 시각화 함수
# ══════════════════════════════════════════════════════════
def _sort(frames, *arrays):
    idx = np.argsort(frames)
    return (frames[idx],) + tuple(a[idx] for a in arrays)

def plot_three_bars(frames: np.ndarray,
                    labels: np.ndarray,
                    preds:  np.ndarray,
                    save_path: str) -> None:
    """LabelBar / PredBar / ErrorBar (pixel 단위, 번짐 없음)"""
    if frames.size == 0:
        return

    frames, labels, preds = _sort(frames, labels, preds)
    length = int(frames[-1]) + 1

    bar1 = np.ones((1, length, 3), float)
    bar2 = np.ones((1, length, 3), float)
    bar3 = np.ones((1, length, 3), float)

    bar1[0, frames[labels == 1]] = [1, 0, 0]      # GT 빨강
    bar2[0, frames[preds  == 1]] = [0, 0, 1]      # Pred 파랑

    fn = (labels == 1) & (preds == 0)
    fp = (labels == 0) & (preds == 1)
    tp = (labels == 1) & (preds == 1)
    bar3[0, frames[fn]] = [1, 0, 0]
    bar3[0, frames[fp]] = [0, 0, 1]
    bar3[0, frames[tp]] = [0, 1, 0]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(12, 4), sharex=True)

    ax[0].imshow(bar1, aspect="auto", interpolation="nearest")
    ax[0].set_ylabel("LabelBar"); ax[0].set_yticks([])

    ax[1].imshow(bar2, aspect="auto", interpolation="nearest")
    ax[1].set_ylabel("PredBar");  ax[1].set_yticks([])

    ax[2].imshow(bar3, aspect="auto", interpolation="nearest")
    ax[2].set_ylabel("ErrorBar"); ax[2].set_yticks([])
    ax[2].set_xlabel("Frame Index")

    ax[0].set_xlim([0, length])
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

def plot_threshold_curve(frames: np.ndarray,
                         probs:  np.ndarray,
                         save_path: str,
                         thresh: float = THRESH_VAL) -> None:
    if frames.size == 0:
        return
    frames, probs = _sort(frames, probs)[0:2]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 2.5))
    plt.plot(frames, probs, lw=1)
    plt.axhline(thresh, color="r", ls="--")
    plt.ylim([0, 1])
    plt.xlabel("Frame"); plt.ylabel("P(collision)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

# ══════════════════════════════════════════════════════════
# 3. Dataset
# ══════════════════════════════════════════════════════════
class DepthCollisionDataset(Dataset):
    FEAT_COLS = ("time","x1","y1","x2","y2","overlap_ratio","frame")

    def __init__(self,
                 rows: List[Dict],
                 queue_size: int = QUEUE_SIZE,
                 after_start: int = AFTER_START,
                 after_finish: int = AFTER_FINISH):
        self.q, self.a1, self.a2 = queue_size, after_start, after_finish
        self.vdict  = self._group(rows)
        self.samples= self._index()

    @staticmethod
    def _group(rows):
        vd = defaultdict(list)
        for r in rows: vd[r["video"]].append(r)
        for v in vd: vd[v].sort(key=lambda x:int(x["frame"]))
        return vd

    def _index(self):
        out=[]
        for vid, rows in self.vdict.items():
            N=len(rows)
            labels=[int(r["label"]) for r in rows]
            minT=min(float(r["time"]) for r in rows)
            for st in range(N-self.q):
                ed = st+self.q
                fs, fe = ed+self.a1, min(ed+self.a2, N-1)
                y = int(fs<N and any(labels[fs:fe+1]))
                out.append(dict(
                    video=vid,
                    frame=int(rows[ed-1]["frame"]),
                    depth_paths=[r["depth_path"] for r in rows[st:ed]],
                    bbox_rows  = rows[st:ed],
                    minT=minT,
                    label=y))
        return out

    def _load_depth(self, path:str)->np.ndarray:
        arr=np.load(path, mmap_mode="r").astype(np.float32) # (360,540) 등
        if (arr.shape[0],arr.shape[1])!=(CROP_H,CROP_W):
            arr=cv2.resize(arr,(CROP_W,CROP_H),cv2.INTER_AREA)
        return (arr/DTYPE_NORM).clip(0,1).astype(np.float16)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s=self.samples[idx]
        depth=np.stack([self._load_depth(p) for p in s["depth_paths"]],0)
        depth=torch.from_numpy(depth)[None]                     # (1,T,H,W)

        T_last=float(s["bbox_rows"][-1]["time"])
        minT   = s["minT"];  T_len=len(s["bbox_rows"])
        feat=[]
        for i,r in enumerate(s["bbox_rows"]):
            feat.append([
                (T_last-float(r["time"])) / max(1e-6,(T_last-minT)),
                float(r["x1"])/1280, float(r["y1"])/720,
                float(r["x2"])/1280, float(r["y2"])/720,
                float(r["overlap_ratio"]),
                i/(T_len-1) if T_len>1 else 0.0
            ])
        feat=torch.from_numpy(np.asarray(feat,dtype=np.float16)) # (T,7)
        label=torch.tensor(s["label"],dtype=torch.long)
        return depth, feat, label

    def get_frame(self,i): return self.samples[i]["frame"]
    def get_video(self,i): return self.samples[i]["video"]

# ══════════════════════════════════════════════════════════
# 4. Model
# ══════════════════════════════════════════════════════════
class DepthConv3DLSTM(nn.Module):
    def __init__(self, bbox_dim=7):
        super().__init__()
        # Depth branch
        ch=(1,8,16,32)
        blocks=[]
        for ci,co in zip(ch[:-1],ch[1:]):
            blocks+=[
                nn.Conv3d(ci,co,(1,3,3),padding=(0,1,1)),
                nn.BatchNorm3d(co),
                nn.ReLU(inplace=True),
                nn.MaxPool3d((1,2,2))
            ]
        self.depth_cnn=nn.Sequential(*blocks)          # (B,32,T,H',W')

        # BBox branch
        self.bbox_conv=nn.Sequential(
            nn.Conv1d(bbox_dim,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        # Fuse
        self.fuse_conv=nn.Sequential(
            nn.Conv1d(96,128,5,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))

        self.lstm=nn.LSTM(128,128,batch_first=True,bidirectional=True)

        self.head=nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64,2))

    def _depth_forward(self,x):
        x=self.depth_cnn(x)           # (B,32,T,H,W)
        x=x.amax(-1).amax(-1)         # (B,32,T)
        return x.permute(0,2,1)       # (B,T,32)

    def forward(self,depth,feat):
        d=self._depth_forward(depth.to(memory_format=torch.channels_last_3d))
        b=self.bbox_conv(feat.permute(0,2,1)).permute(0,2,1)   # (B,T,64)
        x=torch.cat([d,b],-1)                                  # (B,T,96)
        x=self.fuse_conv(x.permute(0,2,1)).permute(0,2,1)      # (B,T,128)
        out,_=self.lstm(x)
        return self.head(out[:,-1])

# ══════════════════════════════════════════════════════════
# 5. Loss & Metric
# ══════════════════════════════════════════════════════════
class RecallCELoss(nn.Module):
    def __init__(self,pos_w):
        super().__init__()
        self.ce=nn.CrossEntropyLoss(weight=torch.tensor([1.,pos_w]))
    def forward(self,l,t): return self.ce(l,t)

@torch.no_grad()
def logits_to_pred(logits: torch.Tensor)->torch.Tensor:
    if THRESH_USE:
        prob=torch.softmax(logits,1)[:,1]
        return (prob>=THRESH_VAL).long()
    return logits.argmax(1)

def evaluate_metrics(y_true: List[int],
                     y_pred: List[int]) -> Tuple[float,float,float,float]:
    acc  = accuracy_score(y_true,y_pred)
    prec = precision_score(y_true,y_pred,zero_division=0)
    rec  = recall_score(y_true,y_pred,zero_division=0)
    f1   = f1_score(y_true,y_pred,zero_division=0)
    return acc,prec,rec,f1

# ══════════════════════════════════════════════════════════
# 6. Train loop
# ══════════════════════════════════════════════════════════
def load_csv(p:str)->List[Dict]:
    with open(p,"r",encoding="utf-8") as f: return list(csv.DictReader(f))

def split_rows(rows:List[Dict],ratio=0.8):
    vids=list({r["video"] for r in rows}); random.shuffle(vids)
    k=int(len(vids)*ratio); train=set(vids[:k]); val=set(vids[k:])
    return [r for r in rows if r["video"] in train], [r for r in rows if r["video"] in val]

def make_loader(ds:Dataset,bs:int,shuf:bool)->DataLoader:
    return DataLoader(ds,bs,shuffle=shuf,drop_last=shuf,
                      num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,
                      persistent_workers=True)

def train():
    rows=load_csv(CSV_PATH)
    train_rows,val_rows=split_rows(rows)

    train_ds, val_ds = DepthCollisionDataset(train_rows), DepthCollisionDataset(val_rows)

    pos=sum(s["label"] for s in train_ds.samples); neg=len(train_ds)-pos
    loss_fn=RecallCELoss(neg/max(1,pos)).to(DEVICE)

    model=DepthConv3DLSTM().to(DEVICE)
    optimizer=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    scaler=GradScaler(enabled=(DEVICE=="cuda"))

    # OOM-safe batch size
    bs=INIT_BS
    while True:
        try:
            _=next(iter(make_loader(train_ds,bs,True)))
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs>1:
                bs//=2; torch.cuda.empty_cache()
                print(f"[MEM] batch size ↓ {bs}")
            else: raise

    train_loader=make_loader(train_ds,bs,True)
    val_loader  =make_loader(val_ds,bs,False)

    best_f1=0.
    for epoch in range(1,EPOCHS+1):
        # ─ Train ─
        model.train(); tl=tt=0; yT,yP=[], []
        optimizer.zero_grad(set_to_none=True)
        for step,(d,f,l) in enumerate(tqdm(train_loader,desc=f"E{epoch} Train",leave=False)):
            d,f,l=d.to(DEVICE),f.to(DEVICE),l.to(DEVICE)
            with autocast(device_type="cuda" if DEVICE=="cuda" else "cpu",
                          dtype=torch.float16,enabled=(DEVICE=="cuda")):
                out=model(d,f); loss=loss_fn(out,l)/ACC_STEPS
            scaler.scale(loss).backward()
            if (step+1)%ACC_STEPS==0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            tl+=loss.item()*l.size(0)*ACC_STEPS; tt+=l.size(0)
            yT+=l.cpu().tolist(); yP+=logits_to_pred(out).cpu().tolist()
        acc_t,pre_t,rec_t,f1_t=evaluate_metrics(yT,yP)

       # ──── Validation ────
        model.eval()
        vl_loss = vl_total = 0                 # 손실 합, 샘플 수
        y_val_true, y_val_pred = [], []        # ⬅️ 분리된 리스트

        vid_buf = defaultdict(lambda: {"fr":[], "lab":[], "pred":[], "prob":[]})
        seen = 0

        with torch.no_grad():
            for d, f, l in tqdm(val_loader, desc=f"E{epoch} Val", leave=False):
                d, f, l = d.to(DEVICE), f.to(DEVICE), l.to(DEVICE)

                with autocast(device_type="cuda" if DEVICE=="cuda" else "cpu",
                            dtype=torch.float16, enabled=(DEVICE=="cuda")):
                    out  = model(d, f)
                    loss = loss_fn(out, l)

                vl_loss  += loss.item() * l.size(0)
                vl_total += l.size(0)

                prob = torch.softmax(out, 1)[:, 1].cpu().numpy()
                pred = logits_to_pred(out).cpu().numpy()
                lab  = l.cpu().numpy()

                y_val_true.extend(lab.tolist())
                y_val_pred.extend(pred.tolist())

                # ── bar/threshold plot용 버퍼 유지 ──
                for i in range(l.size(0)):
                    idx = seen + i
                    vid = val_ds.get_video(idx)
                    fr  = val_ds.get_frame(idx)

                    vid_buf[vid]["fr"  ].append(fr)
                    vid_buf[vid]["lab" ].append(int(lab[i]))
                    vid_buf[vid]["pred"].append(int(pred[i]))
                    vid_buf[vid]["prob"].append(float(prob[i]))
                seen += l.size(0)

        # ── metric 계산 ──
        acc_v, pre_v, rec_v, f1_v = evaluate_metrics(y_val_true, y_val_pred)

        print(f"[Epoch {epoch}/{EPOCHS}] "
            f"TrainL {tl/tt:.4f} Acc {acc_t:.3f} Rec {rec_t:.3f} F1 {f1_t:.3f} | "
            f"ValL {vl_loss/vl_total:.4f} Acc {acc_v:.3f} Rec {rec_v:.3f} F1 {f1_v:.3f}")


        # plots
        for vid,dic in vid_buf.items():
            fr=np.asarray(dic["fr"]); lab=np.asarray(dic["lab"])
            pred=np.asarray(dic["pred"]); prob=np.asarray(dic["prob"])
            plot_three_bars(fr,lab,pred,f"results/bar/{vid}/bar_e{epoch}.png")
            plot_threshold_curve(fr,prob,f"results/thr/{vid}/thr_e{epoch}.png")

        # ckpt
        torch.save(model.state_dict(),f"ckpt_epoch_{epoch}.pth")
        if f1_v>best_f1:
            best_f1=f1_v; torch.save(model.state_dict(),"best_model.pth")
            print(f"  ↳ new BEST! F1={best_f1:.3f}")

        torch.cuda.empty_cache(); gc.collect()

# ══════════════════════════════════════════════════════════
# 7. main
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    train()



# # ──────────────────────────────────────────────────────────────
# # train_proposed_memopt.py  (rev-F + frame + no-crop)
# # ──────────────────────────────────────────────────────────────
# import os, csv, random, gc, warnings, cv2
# from collections import defaultdict
# from typing import List, Tuple, Dict
# import numpy as np
# from tqdm import tqdm
# import matplotlib; matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# import torch, torch.nn as nn, torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torch.amp import autocast, GradScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # ══════════════════════ 하이퍼파라미터 ══════════════════════
# CSV_PATH      = "preprocess_output/final_collision_depth_crop.csv"
# QUEUE_SIZE    = 1200
# T_STRIDE      = 4
# AFTER_START   = 50
# AFTER_FINISH  = 400
# EPOCHS        = 5
# INIT_BS       = 4
# ACC_STEPS     = 2
# LR            = 3e-5
# SEED          = 10
# DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# # ══════════════════════ 메모리 관련 ═════════════════════════
# # depth 파일은 이미 crop 되어 있으므로 가로 자르기(l,r) 제거
# CROP_H, CROP_W = 128, 256           # down-scale 용 (유지)
# DTYPE_NORM     = 10.0               # 깊이 정규화 최대(m)
# NUM_WORKERS    = 16
# PIN_MEMORY     = False
# THRESH_VAL     = 0.55; THRESH_USE = True
# BBOX_DIM       = 7                  # ▲ time,x1,y1,x2,y2,overlap,frame
# USE_CHECKPOINT = False

# # ─────────────────────────────────────────────────────────────
# os.makedirs("results/bar", exist_ok=True)
# os.makedirs("results/thr", exist_ok=True)
# torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
# torch.backends.cuda.matmul.allow_tf32 = True
# warnings.filterwarnings("ignore", category=UserWarning)

# # ══════════════════════ Plot utils ══════════════════════════
# def _sort(fr, *arrs):
#     idx = np.argsort(fr)
#     return (fr[idx],) + tuple(a[idx] for a in arrs)

# def plot_three_bars_separated(frames, labels, preds, out_path):
#     if frames.size == 0: return
#     frames, labels, preds = _sort(frames, labels, preds)
#     L = int(frames[-1]) + 1
#     bar = np.ones((3, L, 3), dtype=float)
#     bar[0, frames[labels==1]] = [1,0,0]
#     bar[1, frames[preds==1]]  = [0,0,1]
#     fn=(labels==1)&(preds==0); fp=(labels==0)&(preds==1); tp=(labels==1)&(preds==1)
#     bar[2, frames[fn]]=[1,0,0]; bar[2, frames[fp]]=[0,0,1]; bar[2, frames[tp]]=[0,1,0]
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.figure(figsize=(12,3)); plt.imshow(bar,aspect='auto'); plt.axis("off")
#     plt.tight_layout(); plt.savefig(out_path,dpi=100); plt.close()

# def plot_threshold_bar(frames, probs, out_path, thresh=THRESH_VAL):
#     if frames.size == 0: return
#     frames, probs = _sort(frames, probs)[0:2]
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.figure(figsize=(12,2.5))
#     plt.plot(frames, probs, lw=1); plt.axhline(thresh,color='r',ls='--')
#     plt.ylim([0,1]); plt.xlabel("Frame"); plt.ylabel("P(collision)")
#     plt.tight_layout(); plt.savefig(out_path,dpi=100); plt.close()

# # ══════════════════════ Dataset ═════════════════════════════
# class DepthCollisionDataset(Dataset):
#     FEAT_COLS = ("time","x1","y1","x2","y2","overlap_ratio","frame")   # ▲ frame 추가
#     def __init__(self, rows: List[Dict], q=QUEUE_SIZE,
#                  a1=AFTER_START, a2=AFTER_FINISH, stride=T_STRIDE):
#         self.q,self.a1,self.a2,self.stride = q,a1,a2,stride
#         self.vdict = self._group(rows)
#         self.samples = self._index()

#     @staticmethod
#     def _group(rows):
#         vd=defaultdict(list)
#         for r in rows: vd[r["video"]].append(r)
#         for v in vd: vd[v].sort(key=lambda x:int(x["frame"]))
#         return vd

#     def _index(self):
#         out=[]
#         for vid,rows in self.vdict.items():
#             N=len(rows)
#             labels=[int(r["label"]) for r in rows]
#             minT=min(float(r["time"]) for r in rows)
#             for st in range(N-self.q):
#                 ed=st+self.q
#                 fs,fe=ed+self.a1, min(ed+self.a2, N-1)
#                 y=int(fs<N and any(labels[fs:fe+1]))

#                 depth_paths=[rows[i]["depth_path"] for i in range(st,ed)][::self.stride]
#                 bbox_rows  = rows[st:ed:self.stride]

#                 out.append(dict(
#                     video=vid,
#                     frame=int(rows[ed-1]["frame"]),
#                     depth_paths=depth_paths,
#                     bbox_rows=bbox_rows,
#                     minT=minT,
#                     label=y))
#         return out

#     def _load_depth(self, path:str)->np.ndarray:
#         # depth 파일은 이미 crop 되어 있으므로 그대로 읽고 ↓ down-scale 만
#         arr = np.load(path, mmap_mode="r").astype(np.float32)    # ex) (H,W) = 360×540
#         if (arr.shape[0],arr.shape[1]) != (CROP_H,CROP_W):
#             arr = cv2.resize(arr, (CROP_W, CROP_H), cv2.INTER_AREA)
#         return (arr/DTYPE_NORM).clip(0,1).astype(np.float16)

#     def __len__(self): return len(self.samples)

#     def __getitem__(self, idx):
#         s=self.samples[idx]
#         depth=np.stack([self._load_depth(p) for p in s["depth_paths"]],0) # (T',H,W)
#         depth=torch.from_numpy(depth)[None]                                # (1,T',H,W)

#         T_last=float(s["bbox_rows"][-1]["time"]); minT=s["minT"]
#         feat=[]
#         for i,r in enumerate(s["bbox_rows"]):
#             rel_frame = i / (len(s["bbox_rows"])-1) if len(s["bbox_rows"])>1 else 0.0
#             feat.append([
#                 (T_last-float(r["time"])) / max(1e-6,(T_last-minT)),
#                 float(r["x1"])/1280, float(r["y1"])/720,
#                 float(r["x2"])/1280, float(r["y2"])/720,
#                 float(r["overlap_ratio"]),
#                 rel_frame                                           # ▲ frame feature
#             ])
#         feat=torch.from_numpy(np.asarray(feat,dtype=np.float16))           # (T',7)

#         return depth, feat, torch.tensor(s["label"],dtype=torch.long)

#     def get_frame(self,i): return self.samples[i]["frame"]
#     def get_video(self,i): return self.samples[i]["video"]

# # ══════════════════════ Model ═════════════════════════════════
# class DepthConv3DLSTM(nn.Module):
#     def __init__(self, bbox_dim=BBOX_DIM):          # ▲ bbox_dim=7
#         super().__init__()
#         ch=(1,8,16,32,64)
#         blocks=[]
#         for ci,co in zip(ch[:-1],ch[1:]):
#             blocks.extend([nn.Conv3d(ci,co,3,padding=1),
#                            nn.BatchNorm3d(co), nn.ReLU(inplace=True),
#                            nn.MaxPool3d((1,2,2))])
#         self.backbone=nn.Sequential(*blocks)

#         self.bbox_conv=nn.Sequential(
#             nn.Conv1d(bbox_dim,64,1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))

#         self.fuse_conv=nn.Sequential(
#             nn.Conv1d(128,128,5,padding=2), nn.BatchNorm1d(128), nn.ReLU(inplace=True))

#         self.lstm=nn.LSTM(128,128,batch_first=True,bidirectional=True)

#         self.head=nn.Sequential(
#             nn.Linear(256,64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64,2))

#     def _depth_forward(self,x):
#         if USE_CHECKPOINT:
#             x=torch.utils.checkpoint.checkpoint_sequential(self.backbone,4,x)
#         else: x=self.backbone(x)
#         x=x.amax(-1).amax(-1)          # (B,64,T')
#         return x.permute(0,2,1)        # (B,T',64)

#     def forward(self,depth,feat):
#         depth=depth.to(memory_format=torch.channels_last_3d)
#         d=self._depth_forward(depth)
#         b=self.bbox_conv(feat.permute(0,2,1)).permute(0,2,1)
#         x=self.fuse_conv(torch.cat([d,b],-1).permute(0,2,1)).permute(0,2,1)
#         out,_=self.lstm(x)
#         return self.head(out[:,-1])

# # ══════════════════════ Helpers ══════════════════════════════
# class RecallCELoss(nn.Module):
#     def __init__(self,pos_w): super().__init__(); self.ce=nn.CrossEntropyLoss(
#         weight=torch.tensor([1.,pos_w]))
#     def forward(self,l,t): return self.ce(l,t)

# @torch.no_grad()
# def logits2pred(l): return ((torch.softmax(l,1)[:,1]>=THRESH_VAL).long()
#                             if THRESH_USE else l.argmax(1))

# def metric(y,p): return (accuracy_score(y,p),
#                          precision_score(y,p,zero_division=0),
#                          recall_score(y,p,zero_division=0),
#                          f1_score(y,p,zero_division=0))

# # ══════════════════════ Train Loop (전체) ══════════════════════
# def load_csv(p:str)->List[Dict]:
#     with open(p,"r",encoding="utf-8") as f: return list(csv.DictReader(f))

# def split_rows(rows,ratio=0.8):
#     vids=sorted({r["video"] for r in rows}); random.shuffle(vids)
#     k=int(len(vids)*ratio); tr=set(vids[:k]); vl=set(vids[k:])
#     print(f"Train videos ({len(tr)}): {sorted(tr)}")
#     print(f"Val   videos ({len(vl)}): {sorted(vl)}")
#     return [r for r in rows if r["video"] in tr], [r for r in rows if r["video"] in vl]

# def make_loader(ds,bs,shuf):
#     return DataLoader(ds,bs,shuffle=shuf,drop_last=shuf,
#                       num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,
#                       persistent_workers=True)

# def train():                                               # ◀── 항상 전체 제공
#     rows=load_csv(CSV_PATH)
#     train_rows,val_rows=split_rows(rows)

#     train_ds, val_ds = DepthCollisionDataset(train_rows), DepthCollisionDataset(val_rows)

#     pos=sum(s["label"] for s in train_ds.samples)
#     neg=len(train_ds)-pos
#     loss_fn=RecallCELoss(neg/max(1,pos)).to(DEVICE)

#     model=DepthConv3DLSTM().to(DEVICE)
#     optimizer=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
#     scaler=GradScaler(enabled=(DEVICE=="cuda"))

#     # ── batch size OOM auto-reduce ────────────────────────────
#     bs=INIT_BS
#     while True:
#         try:
#             train_loader=make_loader(train_ds,bs,True)
#             val_loader  =make_loader(val_ds,bs,False)
#             depth,feat,_=next(iter(train_loader))
#             with autocast(device_type="cuda" if DEVICE=="cuda" else "cpu",
#                           dtype=torch.float16,enabled=(DEVICE=="cuda")):
#                 _=model(depth.to(DEVICE),feat.to(DEVICE))
#             break
#         except RuntimeError as e:
#             if "out of memory" in str(e).lower() and bs>1:
#                 bs//=2; torch.cuda.empty_cache(); print(f"[MEM] batch↓ {bs}")
#             else: raise

#     best_f1=0.0
#     for ep in range(1,EPOCHS+1):
#         # ── Train ────────────────────────────────────────────
#         model.train(); tr_loss=tr_total=0; yT=yP=[]
#         optimizer.zero_grad(set_to_none=True)
#         for step,(d,f,l) in enumerate(tqdm(train_loader,desc=f"E{ep}-train",leave=False)):
#             d,f,l=d.to(DEVICE),f.to(DEVICE),l.to(DEVICE)
#             with autocast(device_type="cuda" if DEVICE=="cuda" else "cpu",
#                           dtype=torch.float16,enabled=(DEVICE=="cuda")):
#                 out=model(d,f); loss=loss_fn(out,l)/ACC_STEPS
#             scaler.scale(loss).backward()
#             if (step+1)%ACC_STEPS==0:
#                 scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
#             tr_loss+=loss.item()*l.size(0)*ACC_STEPS; tr_total+=l.size(0)
#             yT+=l.cpu().tolist(); yP+=logits2pred(out).cpu().tolist()
#         tr_metrics=metric(np.array(yT),np.array(yP))

#         # ── Validation & Plot ───────────────────────────────
#         model.eval(); vl_loss=vl_total=0; yT=yP=[]
#         vid_buf=defaultdict(lambda:{"fr":[],"lab":[],"pred":[],"prob":[]}); seen=0
#         with torch.no_grad():
#             for d,f,l in tqdm(val_loader,desc=f"E{ep}-val  ",leave=False):
#                 bsz=l.size(0); d,f,l=d.to(DEVICE),f.to(DEVICE),l.to(DEVICE)
#                 with autocast(device_type="cuda" if DEVICE=="cuda" else "cpu",
#                               dtype=torch.float16,enabled=(DEVICE=="cuda")):
#                     out=model(d,f); loss=loss_fn(out,l)
#                 vl_loss+=loss.item()*bsz; vl_total+=bsz
#                 prob=torch.softmax(out,1)[:,1].cpu().numpy()
#                 pred=logits2pred(out).cpu().numpy(); lab=l.cpu().numpy()
#                 yT+=lab.tolist(); yP+=pred.tolist()
#                 for i in range(bsz):
#                     idx=seen+i; vid=val_ds.get_video(idx); fr=val_ds.get_frame(idx)
#                     vid_buf[vid]["fr"].append(fr)
#                     vid_buf[vid]["lab"].append(int(lab[i]))
#                     vid_buf[vid]["pred"].append(int(pred[i]))
#                     vid_buf[vid]["prob"].append(float(prob[i]))
#                 seen+=bsz
#         vl_metrics=metric(np.array(yT),np.array(yP))
#         print(f"[E{ep:02d}] TrainL={tr_loss/tr_total:.4f} Acc={tr_metrics[0]:.3f} "
#               f"Rec={tr_metrics[2]:.3f} F1={tr_metrics[3]:.3f} | "
#               f"ValL={vl_loss/vl_total:.4f} Acc={vl_metrics[0]:.3f} "
#               f"Rec={vl_metrics[2]:.3f} F1={vl_metrics[3]:.3f}")

#         # ── Plot 저장 ───────────────────────────────────────
#         for vid,dic in vid_buf.items():
#             fr=np.asarray(dic["fr"]); lab=np.asarray(dic["lab"])
#             pred=np.asarray(dic["pred"]); prob=np.asarray(dic["prob"])
#             plot_three_bars_separated(fr,lab,pred,f"results/bar/{vid}/bar_e{ep}.png")
#             plot_threshold_bar(fr,prob,f"results/thr/{vid}/thr_e{ep}.png")

#         # ── Checkpoint ─────────────────────────────────────
#         torch.save(model.state_dict(),f"ckpt_epoch_{ep}.pth")
#         if vl_metrics[3]>best_f1:
#             best_f1=vl_metrics[3]; torch.save(model.state_dict(),"best_model.pth")
#             print(f"  ↳ new BEST! F1={best_f1:.3f}")

#         torch.cuda.empty_cache(); gc.collect()

# if __name__=="__main__":
#     train()






# # ──────────────────────────────────────────────────────────────
# # train_proposed_memopt.py  (ALL-IN ver.)
# #   ▷ Depth fp16 128×256  +  T-stride 4  +  MaxPool3d((1,2,2))
# #   ▷ OOM-safe (auto bs↓, checkpoint), AMP 최신 API
# #   ▷ epoch-wise metrics + bar/threshold plots + ckpt save
# # ──────────────────────────────────────────────────────────────
# import os, csv, random, gc, warnings, cv2, math
# from collections import defaultdict
# import numpy as np
# from tqdm import tqdm
# import matplotlib, matplotlib.pyplot as plt; matplotlib.use("Agg")

# import torch, torch.nn as nn, torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torch.amp import autocast, GradScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # ══════════════════════ 고정 하이퍼파라미터 ══════════════════════
# CSV_PATH      = "preprocess_output/final_collision_depth_crop.csv"
# QUEUE_SIZE    = 1200
# AFTER_START   = 50
# AFTER_FINISH  = 400
# EPOCHS        = 5
# INIT_BS       = 4             # out-of-memory 시 자동 ½↓
# ACC_STEPS     = 2
# LR            = 3e-5
# SEED          = 10
# DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# # ══════════════════════ 메모리 절약용 파라미터 ═══════════════════
# T_STRIDE      = 4                 # 1200 → 300
# CROP_H, CROP_W = 128, 256         # 180×360 → 128×256
# LEFT_X, RIGHT_X = 367, 913
# DTYPE_NORM    = 10.0              # depth 0~10 m → 0~1
# NUM_WORKERS   = 16
# PIN_MEMORY    = False
# THRESH_VAL    = 0.55; THRESH_USE = True
# BBOX_DIM      = 6
# USE_CHECKPOINT = False            # True → extra mem ↓ (속도↓)

# # ───────────────────────────────────────────────────────────────
# os.makedirs("results/bar", exist_ok=True)
# os.makedirs("results/thr", exist_ok=True)
# torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
# torch.backends.cuda.matmul.allow_tf32 = True
# warnings.filterwarnings("ignore", category=UserWarning)

# # ══════════════════════ Plot utils ═════════════════════════════
# def plot_three_bars_separated(frames, labels, preds, out_path):
#     if frames.size == 0: return
#     L = frames.max()+1
#     bar = np.ones((3, L, 3), dtype=float)
#     bar[0, frames[labels==1]] = [1,0,0]          # GT (red)
#     bar[1, frames[preds==1]]  = [0,0,1]          # Pred (blue)
#     fn = (labels==1)&(preds==0); fp=(labels==0)&(preds==1); tp=(labels==1)&(preds==1)
#     bar[2, frames[fn]]=[1,0,0]; bar[2, frames[fp]]=[0,0,1]; bar[2, frames[tp]]=[0,1,0]
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.figure(figsize=(12,3)); plt.imshow(bar,aspect='auto'); plt.axis("off")
#     plt.tight_layout(); plt.savefig(out_path,dpi=100); plt.close()

# def plot_threshold_bar(frames, probs, out_path, thresh=THRESH_VAL):
#     if frames.size == 0: return
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.figure(figsize=(12,2.5))
#     plt.plot(frames, probs, lw=1)
#     plt.axhline(thresh, color='r', ls='--')
#     plt.ylim([0,1]); plt.xlabel("Frame"); plt.ylabel("P")
#     plt.tight_layout(); plt.savefig(out_path,dpi=100); plt.close()

# # ══════════════════════ Dataset ═══════════════════════════════
# class DepthCollisionDataset(Dataset):
#     FEAT_COLS = ("time","x1","y1","x2","y2","overlap_ratio")
#     def __init__(self, rows, q=QUEUE_SIZE, a1=AFTER_START, a2=AFTER_FINISH):
#         self.q,self.a1,self.a2=q,a1,a2
#         self.vdict=self._group(rows)
#         self.samples=self._index()
#     def _group(self,rows):
#         vd=defaultdict(list)
#         for r in rows: vd[r["video"]].append(r)
#         for v in vd: vd[v].sort(key=lambda x:int(x["frame"]))
#         return vd
#     def _index(self):
#         out=[]
#         for vid,rows in self.vdict.items():
#             N=len(rows); labels=[int(r["label"]) for r in rows]
#             minT=min(float(r["time"]) for r in rows)
#             for st in range(N-self.q):
#                 ed=st+self.q; fs,fe=ed+self.a1, min(ed+self.a2,N-1)
#                 y=int(fs<N and any(labels[fs:fe+1]))
#                 out.append(dict(
#                     video=vid,
#                     frame=int(rows[ed-1]["frame"]),
#                     depth_paths=[rows[i]["depth_path"] for i in range(st,ed)][::T_STRIDE],
#                     bbox_rows = rows[st:ed:T_STRIDE],
#                     minT=minT, label=y))
#         return out
#     def _load_depth(self,path):
#         arr=np.load(path,mmap_mode="r").squeeze()
#         H,W=arr.shape; l,r=LEFT_X,min(RIGHT_X,W)
#         if r-l<=0: l,r=0,W
#         arr=cv2.resize(arr[:,l:r].astype(np.float32),(CROP_W,CROP_H),cv2.INTER_AREA)
#         return (arr/DTYPE_NORM).astype(np.float16)
#     def __len__(self): return len(self.samples)
#     def __getitem__(self,idx):
#         s=self.samples[idx]
#         depth=np.stack([self._load_depth(p) for p in s["depth_paths"]],0)[None] # (1,T',H,W)
#         depth=torch.from_numpy(depth)
#         T_last=float(s["bbox_rows"][-1]["time"]); minT=s["minT"]
#         feat=[[(T_last-float(r["time"])) / max(1e-6,(T_last-minT)),
#                float(r["x1"])/1280, float(r["y1"])/720,
#                float(r["x2"])/1280, float(r["y2"])/720,
#                float(r["overlap_ratio"])] for r in s["bbox_rows"]]
#         feat=torch.from_numpy(np.asarray(feat,dtype=np.float16))
#         return depth, feat, torch.tensor(s["label"],dtype=torch.long)
#     def get_frame(self,i):  return self.samples[i]["frame"]
#     def get_video(self,i):  return self.samples[i]["video"]

# # ══════════════════════ Model ═════════════════════════════════
# class DepthConv3DLSTM(nn.Module):
#     def __init__(self,bbox_dim=BBOX_DIM):
#         super().__init__()
#         ch=(1,8,16,32,64); blocks=[]
#         for cin,cout in zip(ch[:-1],ch[1:]):
#             blocks.extend([nn.Conv3d(cin,cout,3,padding=1),
#                            nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
#                            nn.MaxPool3d((1,2,2))])
#         self.backbone=nn.Sequential(*blocks)
#         self.bbox_conv=nn.Sequential(
#             nn.Conv1d(bbox_dim,64,1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
#         self.fuse_conv=nn.Sequential(
#             nn.Conv1d(128,128,5,padding=2), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
#         self.lstm=nn.LSTM(128,128,batch_first=True,bidirectional=True)
#         self.head=nn.Sequential(
#             nn.Linear(256,64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64,2))
#     def _depth(self,x):
#         if USE_CHECKPOINT: x=torch.utils.checkpoint.checkpoint_sequential(self.backbone,4,x)
#         else: x=self.backbone(x)
#         x=x.amax(-1).amax(-1)       # (B,64,T)
#         return x.permute(0,2,1)     # (B,T,64)
#     def forward(self,depth,feat):
#         depth=depth.to(memory_format=torch.channels_last_3d)
#         d=self._depth(depth)
#         b=self.bbox_conv(feat.permute(0,2,1)).permute(0,2,1)
#         x=self.fuse_conv(torch.cat([d,b],-1).permute(0,2,1)).permute(0,2,1)
#         out,_=self.lstm(x)
#         return self.head(out[:,-1])

# # ══════════════════════ Helpers ═══════════════════════════════
# class RecallCELoss(nn.Module):
#     def __init__(self,pos_w):
#         super().__init__(); self.ce=nn.CrossEntropyLoss(weight=torch.tensor([1.,pos_w]))
#     def forward(self,l,t): return self.ce(l,t)
# @torch.no_grad()
# def logits2pred(l): return ((torch.softmax(l,1)[:,1]>=THRESH_VAL).long() if THRESH_USE else l.argmax(1))
# def metric(y,p): return (accuracy_score(y,p),precision_score(y,p,zero_division=0),
#                          recall_score(y,p,zero_division=0),f1_score(y,p,zero_division=0))

# # ══════════════════════ Train Loop ════════════════════════════
# def load_csv(p):  
#     with open(p,"r",encoding="utf-8") as f: return list(csv.DictReader(f))
# def split_rows(rows,ratio=0.8):
#     vids=sorted({r["video"] for r in rows}); random.shuffle(vids)
#     k=int(len(vids)*ratio); tr=set(vids[:k]); vl=set(vids[k:])
#     print("Train vids:",sorted(tr)); print("Val vids:",sorted(vl))
#     return [r for r in rows if r["video"] in tr], [r for r in rows if r["video"] in vl]
# def make_loader(ds,bs,shuf):
#     return DataLoader(ds,bs,shuffle=shuf,drop_last=shuf,
#                       num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,
#                       persistent_workers=True)

# def train():
#     rows=load_csv(CSV_PATH); tr_r,vl_r=split_rows(rows)
#     tr_ds,vl_ds=DepthCollisionDataset(tr_r),DepthCollisionDataset(vl_r)
#     pos=sum(s["label"] for s in tr_ds.samples); neg=len(tr_ds)-pos
#     loss_fn=RecallCELoss(neg/max(1,pos)).to(DEVICE)

#     model=DepthConv3DLSTM().to(DEVICE)
#     opt=optim.Adam(model.parameters(),lr=LR)
#     scaler=GradScaler()

#     bs=INIT_BS
#     while True:
#         try:
#             tr_ld,vl_ld=make_loader(tr_ds,bs,True),make_loader(vl_ds,bs,False)
#             next(iter(tr_ld)); next(iter(vl_ld)); break
#         except RuntimeError as e:
#             if "out of memory" in str(e).lower() and bs>1:
#                 bs//=2; torch.cuda.empty_cache(); print(f"[MEM] batch↓ {bs}")
#             else: raise

#     best_f1=0
#     for ep in range(1,EPOCHS+1):
#         # ── Train ──────────────────────────────────────
#         model.train(); tl=tt=0; yT=yP=[]
#         opt.zero_grad(set_to_none=True)
#         for step,(d,f,l) in enumerate(tqdm(tr_ld,desc=f"E{ep}-train",leave=False)):
#             d,f,l=d.to(DEVICE),f.to(DEVICE),l.to(DEVICE)
#             with autocast(device_type="cuda", dtype=torch.float16):
#                 out=model(d,f); loss=loss_fn(out,l)/ACC_STEPS
#             scaler.scale(loss).backward()
#             if (step+1)%ACC_STEPS==0:
#                 scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
#             tl+=loss.item()*l.size(0)*ACC_STEPS; tt+=l.size(0)
#             yT+=l.cpu().tolist(); yP+=logits2pred(out).cpu().tolist()
#         tr_metrics=metric(np.array(yT),np.array(yP))

#         # ── Val ( + plot ) ─────────────────────────────
#         model.eval(); vl=vt=0; yT=yP=[]
#         vid_buf=defaultdict(lambda:{"fr":[],"lab":[],"pred":[],"prob":[]})
#         seen=0
#         with torch.no_grad():
#             for d,f,l in tqdm(vl_ld,desc=f"E{ep}-val  ",leave=False):
#                 bs_cur=l.size(0)
#                 d,f,l=d.to(DEVICE),f.to(DEVICE),l.to(DEVICE)
#                 with autocast(device_type="cuda", dtype=torch.float16):
#                     out=model(d,f); loss=loss_fn(out,l)
#                 vl+=loss.item()*bs_cur; vt+=bs_cur
#                 prob=torch.softmax(out,1)[:,1].cpu().numpy()
#                 pred=logits2pred(out).cpu().numpy(); lab=l.cpu().numpy()
#                 yT+=lab.tolist(); yP+=pred.tolist()
#                 for i in range(bs_cur):
#                     idx=seen+i; vid=vl_ds.get_video(idx); fr=vl_ds.get_frame(idx)
#                     vid_buf[vid]["fr"].append(fr)
#                     vid_buf[vid]["lab"].append(int(lab[i]))
#                     vid_buf[vid]["pred"].append(int(pred[i]))
#                     vid_buf[vid]["prob"].append(float(prob[i]))
#                 seen+=bs_cur
#         val_metrics=metric(np.array(yT),np.array(yP))
#         print(f"[{ep:02d}] "
#               f"trL={tl/tt:.4f} Acc={tr_metrics[0]:.3f} F1={tr_metrics[-1]:.3f} | "
#               f"vlL={vl/vt:.4f} Acc={val_metrics[0]:.3f} F1={val_metrics[-1]:.3f}")

#         # ── video-wise plots ──────────────────────────
#         for vid,dic in vid_buf.items():
#             fr=np.array(dic["fr"]); lab=np.array(dic["lab"])
#             pr=np.array(dic["pred"]); pb=np.array(dic["prob"])
#             plot_three_bars_separated(fr,lab,pr,f"results/bar/{vid}/bar_e{ep}.png")
#             plot_threshold_bar      (fr,pb,f"results/thr/{vid}/thr_e{ep}.png")

#         # ── Save ckpts every epoch + best ─────────────
#         torch.save(model.state_dict(), f"ckpt_epoch_{ep}.pth")
#         if val_metrics[-1] > best_f1:
#             best_f1=val_metrics[-1]
#             torch.save(model.state_dict(),"best_model.pth")
#             print(f"  ↳ new best! F1={best_f1:.3f}")
#         torch.cuda.empty_cache(); gc.collect()

# if __name__=="__main__":
#     train()
