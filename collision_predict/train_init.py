import csv
import os
import math
import numpy as np
from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

########################################
# 전역 설정
########################################
MODEL_TYPE = "convLSTM"   # "FFN", "LSTM", "XGBOOST", "CATBOOST", plus "struct1"~"struct6", "convLSTM", "transformer"
AFTER_LABEL_START = 50        # ===== NEW ===== (frame 단위) 10프레임 뒤부터
AFTER_LABEL_FINISH = 400      # ===== NEW ===== (frame 단위) 200프레임 까지 안에 1이 있으면 양성
QUEUE_SIZE = 1200
FEATURE_COLS = ['frame', 'time', 'x1', 'y1', 'x2', 'y2','overlap_ratio'] #'in_zone'
OVERLAP_BIN_THRESH = 0.05
LEARNING_RATE = 3e-5          # 트랜스포머는 살짝 높은 lr와 warm‑up이 유리
EPOCHS = 30
RANDOM_SEED = 10
# LSTM 확장 파라미터
NUM_LAYERS = 2
BIDIRECTIONAL = True

# ===== NEW ===== Threshold 튜닝 스위치
THRESH_USE = True
THRESH_VAL = 0.55             # 0.35 ~ 0.5 사이 값을 validation ROC/PR로 찾아서 넣으세요

# ===== NEW ===== Queue 패딩 허용 (초반부라도 곧바로 예측)
PADDING_ALLOWED = True

# Activation 설정: "relu" or "gelu" etc.
ACTIVATION_FUNC = "relu"


# RANDOM SEED
#55 : 3 10 
#60 : 9 4 
#100 : 7 2

def get_activation():
    if ACTIVATION_FUNC.lower() == "relu":
        return nn.ReLU()
    elif ACTIVATION_FUNC.lower() == "gelu":
        return nn.GELU()
    else:
        return nn.ReLU()

########################################
# 1) Dataset
########################################
class CollisionVideoDataset(Dataset):
    """
    final_collision.csv를 video별로 그룹 -> 비디오별 (frame순 정렬) -> 큐(QUEUE_SIZE) 시계열 생성.
    AFTER_LABEL_START..FINISH 구간 내에 label=1 => 현재 시점=1

    + 정규화 로직:
      - time => (T - time) / (T - minTime) (optional)
      - x => x/1280, y => y/720
      - overlap_ratio => 그대로
      - frame => 무시 or 0
    """
    def __init__(self, rows,
                 queue_size=QUEUE_SIZE,
                 after_label_start=100,
                 after_label_finish=700,
                 feature_cols=FEATURE_COLS):
        super().__init__()
        self.queue_size = queue_size
        self.after_label_start = after_label_start
        self.after_label_finish = after_label_finish
        self.feature_cols = feature_cols

        self.video_dict = self.group_by_video(rows)
        self.samples = self.create_samples()

    def group_by_video(self, rows):
        video_dict = defaultdict(list)
        for r in rows:
            video_dict[r['video']].append(r)
        for v in video_dict:
            video_dict[v].sort(key=lambda x: int(x['frame']))
        return video_dict

    def create_samples(self):
        all_samples = []
        for video, rowlist in self.video_dict.items():
            N = len(rowlist)
            labels = [int(r['label']) for r in rowlist]
            frames = [int(r['frame']) for r in rowlist]
            times  = [float(r['time']) for r in rowlist]
            minTime = min(times)
            maxTime = max(times)

            for start_t in range(N - self.queue_size):
                end_t = start_t + self.queue_size
                # future range -> label
                future_start = end_t + self.after_label_start
                future_end   = end_t + self.after_label_finish
                if future_end >= N:
                    future_end = N - 1
                y_val = 0
                if future_start < N:
                    future_chunk = labels[future_start:future_end+1]
                    if any(l==1 for l in future_chunk):
                        y_val = 1

                # 대표 프레임
                rep_frame = frames[end_t-1]

                # 정규화 + build x_array
                T = float(rowlist[end_t-1]['time'])  # 마지막 프레임 시간
                x_array = []
                for idx in range(start_t, end_t):
                    r = rowlist[idx]
                    f_time = float(r['time'])
                    x1 = float(r['x1'])
                    y1 = float(r['y1'])
                    x2 = float(r['x2'])
                    y2 = float(r['y2'])
                    overlap = float(r['overlap_ratio'])

                    dt = T - f_time
                    denom = T - minTime if (T>minTime) else 1e-6
                    dt_scaled = dt/denom

                    x1n = x1/1280.0
                    x2n = x2/1280.0
                    y1n = y1/720.0
                    y2n = y2/720.0

                    feats = []
                    for c in self.feature_cols:
                        if c=='frame':
                            feats.append(0.0)  # or skip
                        elif c=='time':
                            feats.append(dt_scaled)
                        elif c=='x1':
                            feats.append(x1n)
                        elif c=='y1':
                            feats.append(y1n)
                        elif c=='x2':
                            feats.append(x2n)
                        elif c=='y2':
                            feats.append(y2n)
                        elif c=='overlap_ratio':
                            feats.append(overlap)
                        elif c=='in_zone':                     
                            feats.append(float(overlap > OVERLAP_BIN_THRESH))
                        else:
                            feats.append(0.0)
                    x_array.append(feats)

                x_array = np.array(x_array)  # shape (queue_size, len(feature_cols))

                all_samples.append({
                    'video': video,
                    'rep_frame': rep_frame,
                    'x_array': x_array,
                    'label': y_val
                })
        return all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dic = self.samples[idx]
        x_array = dic['x_array']
        y_val   = dic['label']
        return x_array, y_val

    def get_rep_frame(self, idx):
        return self.samples[idx]['rep_frame']

    def get_video(self, idx):
        return self.samples[idx]['video']


def load_final_collision(csv_path='preprocess_output/final_collision.csv'):
    with open(csv_path, 'r', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    return rows


########################################
# 기존 FFN / LSTM (주석 or 유지)
########################################
# ... (이미 구현된 FFNModel와 CustomLSTMModel)

########################################
# 2) 새 모델들 (struct1 ~ struct6)
########################################


class ConvLSTMClassifier(nn.Module):
    """
    Conv1D → BiLSTM → FC
    - Conv1D: local temporal features
    - LSTM: sequential modeling
    - FC: final binary classification
    """
    def __init__(self, input_dim=7, conv_channels=32, lstm_hidden=64, num_classes=2, lstm_layers=1, bidirectional=True):
        super().__init__()
        self.input_dim = input_dim
        self.conv_channels = conv_channels
        self.lstm_hidden = lstm_hidden
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.act = get_activation()

        # Conv1D: input (B, input_dim, 1200) -> output (B, conv_channels, 1200)
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=conv_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(conv_channels)

        # LSTM: input (B, 1200, conv_channels)
        self.lstm = nn.LSTM(input_size=conv_channels,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            self.act,
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, 1200, input_dim)
        x = x.permute(0, 2, 1)  # → (B, input_dim, 1200)
        x = self.conv1d(x)      # → (B, conv_channels, 1200)
        x = self.bn(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)  # → (B, 1200, conv_channels)

        out, _ = self.lstm(x)   # → (B, 1200, lstm_hidden*2)
        last_hidden = out[:, -1, :]  # 마지막 타임스텝 → (B, lstm_hidden*2)

        logits = self.fc(last_hidden)  # → (B, 2)
        return logits


# ===== NEW ===== Conv1D + BiLSTM + Multi-Head Self-Attention
class ConvLSTMAttentionClassifier(nn.Module):
    """
    (B, T, F=7)
        → Conv1D(kernel=5, ch=32)
        → Bi-LSTM(hidden=64)
        → Multi-Head Attention(d_model=128, heads=4)
        → FC(64→2)
    """
    def __init__(self, input_dim=len(FEATURE_COLS),
                 conv_channels=32, lstm_hidden=64,
                 attn_heads=4, num_classes=2):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, conv_channels, kernel_size=5, padding=2)
        self.bn   = nn.BatchNorm1d(conv_channels)
        self.act  = get_activation()

        # Bi-LSTM → output dim = hidden*2
        self.lstm = nn.LSTM(conv_channels, lstm_hidden,
                            num_layers=1, batch_first=True,
                            bidirectional=True)

        d_model = lstm_hidden * 2
        self.attn = nn.MultiheadAttention(embed_dim=d_model,
                                          num_heads=attn_heads,
                                          batch_first=True)

        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            self.act,
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):                 # x:(B,T,F)
        B, T, F = x.shape
        h = self.conv(x.permute(0,2,1))   # (B, C, T)
        h = self.act(self.bn(h))
        h = h.permute(0,2,1)              # (B,T,C)

        h, _ = self.lstm(h)               # (B,T,d_model)

        # Self-Attention (query = CLS token)
        cls = h[:, -1:, :]                # 마지막 타임스텝을 CLS로 활용
        attn_out, _ = self.attn(query=cls, key=h, value=h)  # (B,1,d_model)
        logits = self.fc(attn_out.squeeze(1))
        return logits



# ===== NEW ===== Conv + LSTM + Self-Attention Fusion
class ConvLSTMAttentionClassifier(nn.Module):
    """
    (B, T, F)
      └─Conv1D(ch=32)─BN─GELU                  # local
        └─Bi-LSTM(hidden=64)                   # mid-range
          └─Multi-Head Attn(heads=4, d=128)    # long-range
            └─Gate(fusion) → FC(2)
    """
    def __init__(self, input_dim=len(FEATURE_COLS),
                 conv_channels=32, lstm_hidden=64,
                 attn_heads=4, num_classes=2):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, conv_channels,
                              kernel_size=5, padding=2)
        self.bn   = nn.BatchNorm1d(conv_channels)
        self.act  = get_activation()

        self.lstm = nn.LSTM(conv_channels, lstm_hidden,
                            batch_first=True, bidirectional=True)

        d_model = lstm_hidden * 2
        self.attn = nn.MultiheadAttention(d_model, attn_heads,
                                          batch_first=True)
        # 게이팅 파라미터
        self.gate = nn.Parameter(torch.tensor(0.5))

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            self.act,
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):               # x:(B,T,F)
        h = self.conv(x.permute(0,2,1)) # (B,C,T)
        h = self.act(self.bn(h)).permute(0,2,1)   # (B,T,C)

        h,_ = self.lstm(h)              # (B,T,d_model)

        attn_out,_ = self.attn(h, h, h) # (B,T,d_model)

        fused = self.gate * h + (1-self.gate) * attn_out
        cls_vec = fused[:, -1]          # 마지막 타임스텝 사용
        return self.head(cls_vec)





########################################
# 4) 시각화 함수 (3개의 Bar)
########################################
def plot_three_bars_separated(frames, labels, preds, output_path='testplot.png'):
    if len(frames) == 0:
        print("No data to plot.")
        return
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True) 

    length = np.max(frames) + 1  # 가로 길이

    label_bar = np.ones((1, length, 3), dtype=float)
    pred_bar  = np.ones((1, length, 3), dtype=float)
    error_bar = np.ones((1, length, 3), dtype=float)

    # Bar1: label=1 => f만 빨강
    for i in range(len(frames)):
        f = frames[i]
        if labels[i] == 1:
            label_bar[0, f] = [1.0, 0.0, 0.0] # red

    # Bar2: pred=1 => 파랑
    for i in range(len(frames)):
        f = frames[i]
        if preds[i] == 1:
            pred_bar[0, f] = [0.0, 0.0, 1.0]

    # Bar3: FN => 빨강, FP => 파랑, (TP => green optional)
    for i in range(len(frames)):
        f = frames[i]
        if labels[i]==1 and preds[i]==0:
            error_bar[0, f] = [1.0, 0.0, 0.0]
        elif labels[i]==0 and preds[i]==1:
            error_bar[0, f] = [0.0, 0.0, 1.0]
        elif labels[i]==1 and preds[i]==1:
            error_bar[0, f] = [0.0, 1.0, 0.0]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12,4), sharex=True)
    axes[0].imshow(label_bar, aspect='auto', interpolation='nearest')
    axes[0].set_yticks([])
    axes[0].set_ylabel("LabelBar")

    axes[1].imshow(pred_bar, aspect='auto', interpolation='nearest')
    axes[1].set_yticks([])
    axes[1].set_ylabel("PredBar")

    axes[2].imshow(error_bar, aspect='auto', interpolation='nearest')
    axes[2].set_yticks([])
    axes[2].set_ylabel("ErrorBar")

    axes[2].set_xlabel("Frame Index")
    axes[0].set_xlim([0,length])

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"[plot] saved => {output_path}")


########################################
# 5) Train Loop
########################################
def evaluate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1


# ===== NEW =====  softmax probs → label 변환 util
@torch.no_grad()
def logits_to_pred(logits):
    if THRESH_USE:
        probs = torch.softmax(logits, dim=1)[:, 1]  # 양성 클래스 확률
        return (probs >= THRESH_VAL).long()
    else:
        return logits.argmax(dim=1)


########################################
# 5) Train Loop  (Validation: video별 plot 포함)
########################################
def train_pytorch_model(model, train_loader, val_loader,
                        epochs=EPOCHS, lr=LEARNING_RATE, device='cpu'):
    # --------- ① Recall Loss 준비 ----------
    train_ds = train_loader.dataset
    pos_cnt  = sum(s['label'] for s in train_ds.samples)
    neg_cnt  = len(train_ds) - pos_cnt
    pos_w    = neg_cnt / max(1, pos_cnt)          # >1 이면 양성 가중 ↑
    criterion = RecallCELoss(pos_w, device)

    # --------- ② Optimizer ----------
    if MODEL_TYPE == "transformer":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(epochs):
        # =================== Train ===================
        model.train()
        tr_loss = tr_total = 0
        tr_pred = []; tr_label = []

        for x_seq, y_label in train_loader:
            x_seq   = x_seq.to(device, dtype=torch.float32)
            y_label = y_label.to(device, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(x_seq)
            loss   = criterion(logits, y_label)
            loss.backward(); optimizer.step()

            tr_loss  += loss.item() * x_seq.size(0)
            tr_total += x_seq.size(0)
            tr_pred  += logits_to_pred(logits).cpu().tolist()
            tr_label += y_label.cpu().tolist()

        acc_t, pre_t, rec_t, f1_t = evaluate_metrics(tr_label, tr_pred)

        # =================== Validation ===================
        model.eval()
        vl_loss = vl_total = 0
        vl_pred = []; vl_label = []

        ds         = val_loader.dataset
        vid_buf    = defaultdict(lambda: {'frame':[], 'prob':[],
                                          'label':[], 'pred':[]})
        seen_idx   = 0   # dataset 인덱스 추적용

        with torch.no_grad():
            for x_seq, y_label in val_loader:
                batch = x_seq.size(0)
                logits = model(x_seq.to(device, dtype=torch.float32))
                loss   = criterion(logits, y_label.to(device))

                vl_loss  += loss.item() * batch
                vl_total += batch

                probs = torch.softmax(logits, 1)[:,1].cpu().numpy()
                preds = logits_to_pred(logits).cpu().numpy()
                labs  = y_label.cpu().numpy()

                # ---- video / frame별 누적 ----
                for i in range(batch):
                    ds_idx = seen_idx + i
                    vid    = ds.get_video(ds_idx)
                    frm    = ds.get_rep_frame(ds_idx)

                    vid_buf[vid]['frame'].append(frm)
                    vid_buf[vid]['prob' ].append(probs[i])
                    vid_buf[vid]['label'].append(int(labs[i]))
                    vid_buf[vid]['pred' ].append(int(preds[i]))

                vl_pred  += preds.tolist()
                vl_label += labs.tolist()
                seen_idx += batch

        acc_v, pre_v, rec_v, f1_v = evaluate_metrics(vl_label, vl_pred)
        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss={tr_loss/tr_total:.4f} Acc={acc_t:.4f} "
              f"Rec={rec_t:.4f} Pre={pre_t:.4f}| "
              f"Val Loss={vl_loss/vl_total:.4f} Acc={acc_v:.4f} "
              f"Rec={rec_v:.4f} Pre={pre_v:.4f} F1={f1_v:.4f}")

        # --------- ③ video 단위 plot 저장 ---------
        for vid, d in vid_buf.items():
            f  = np.array(d['frame'])
            lb = np.array(d['label'])
            pr = np.array(d['pred'])
            pb = np.array(d['prob'])
            
                    # 디렉토리 생성
            bar_dir = f"results/bar/{vid}"
            thr_dir = f"results/thr/{vid}"
            os.makedirs(bar_dir, exist_ok=True)
            os.makedirs(thr_dir, exist_ok=True)

                # 파일 경로
            bar_path = os.path.join(bar_dir, f"bar_{epoch}.png")
            thr_path = os.path.join(thr_dir, f"thr_{epoch}.png")    
            
            plot_three_bars_separated(f, lb, pr, output_path=bar_path)
            plot_threshold_bar(f, pb, thresh=THRESH_VAL, output_path=thr_path)

    # --------- ④ miss-classified 보고 ---------
    mis = [i for i,(p,l) in enumerate(zip(vl_pred, vl_label)) if p!=l]
    print(f"Mis-classified: {len(mis)}/{len(vl_pred)}   "
          f"(showing first 10)")
    for idx in mis[:10]:
        print(f" idx={idx}  pred={vl_pred[idx]}  true={vl_label[idx]}")

    return model



# ===== NEW =====  (x축: frame, y축: prob) + threshold 선
# ===== NEW =====  (x축: frame, y축: prob) + threshold 선
def plot_threshold_bar(frames, probs, thresh=THRESH_VAL, output_path='results/plot.png'):
    if len(frames) == 0:
        return
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12,2.5))
    plt.plot(frames, probs, linewidth=1)
    plt.axhline(thresh, color='r', linestyle='--')
    plt.ylim([0,1])
    plt.xlabel('Frame')
    plt.ylabel('P(collision)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"[plot] saved => {output_path}")



########################################
#  Λ  Recall-중심 Loss -- helper
########################################
class RecallCELoss(nn.Module):
    """
    CrossEntropy + 클래스 가중치(neg/pos) ⇒ FN 패널티 ↑
    """
    def __init__(self, pos_weight, device):
        super().__init__()
        w = torch.tensor([1.0, pos_weight], dtype=torch.float32,
                         device=device)
        self.ce = nn.CrossEntropyLoss(weight=w)

    def forward(self, logits, target):
        return self.ce(logits, target)



# ===== NEW ===== Dual-branch:  (xy 좌표) + (IoU / in-zone)  ===================
class DualBranchOverlapClassifier(nn.Module):
    """
    Branch-A : 좌표   (6D)  → Conv1D → Bi-LSTM
    Branch-B : IoU/in-zone(2D) → Conv1D → Bi-LSTM
    두 branch 출력을 concat → Self-Attention → FC
    """
    def __init__(self,
                 coord_dim=6,          # x1,y1,x2,y2,time,frame(0)
                 ov_dim=2,            # overlap_ratio, in_zone
                 conv_ch=32, lstm_h=64,
                 attn_heads=4, num_classes=2):
        super().__init__()
        act = get_activation()

        # --- Branch-A (coords) ---
        self.convA = nn.Conv1d(coord_dim, conv_ch, 5, padding=2)
        self.bnA   = nn.BatchNorm1d(conv_ch)
        self.lstmA = nn.LSTM(conv_ch, lstm_h, batch_first=True,
                             bidirectional=True)

        # --- Branch-B (overlap) ---
        self.convB = nn.Conv1d(ov_dim, conv_ch//2, 3, padding=1)
        self.bnB   = nn.BatchNorm1d(conv_ch//2)
        self.lstmB = nn.LSTM(conv_ch//2, lstm_h//2, batch_first=True,
                             bidirectional=True)

        d_model = lstm_h*2 + lstm_h   #  (branch A 128) + (branch B 64) = 192
        self.attn = nn.MultiheadAttention(d_model, attn_heads,
                                          batch_first=True)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            act,
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):        # x:(B,T,F=8)
        coord, ov = x[..., :6], x[..., 6:]        # split
        # -------- Branch-A --------
        hA = self.convA(coord.permute(0,2,1))
        hA = self.bnA(hA).permute(0,2,1)
        hA,_ = self.lstmA(hA)          # (B,T,128)

        # -------- Branch-B --------
        hB = self.convB(ov.permute(0,2,1))
        hB = self.bnB(hB).permute(0,2,1)
        hB,_ = self.lstmB(hB)          # (B,T,64)

        h   = torch.cat([hA, hB], dim=-1)        # (B,T,192)
        attn_out,_ = self.attn(h, h, h)          # (B,T,192)
        cls = attn_out[:, -1]                    # 마지막 step
        return self.head(cls)




########################################
# 6) 메인
########################################
def dataset_to_numpy(ds):
    X_list = []
    y_list = []
    for i in range(len(ds)):
        x_seq, y_label = ds[i]
        X_list.append(x_seq.reshape(-1))
        y_list.append(y_label)
    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int32)
    return X_arr, y_arr

def main():
    
    
    # 1) CSV 로드
    rows = load_final_collision('preprocess_output/final_collision.csv')

    # 2) 비디오 목록
    video_set = sorted(list(set(r['video'] for r in rows)))
    print("Total videos:", len(video_set))

    random.seed(RANDOM_SEED) 
    random.shuffle(video_set)
    train_ratio = 0.8
    n_videos = len(video_set)
    n_train = int(n_videos * train_ratio)
    train_videos = set(video_set[:n_train])
    val_videos   = set(video_set[n_train:])

    print("Train videos:", train_videos)
    print("Val videos:", val_videos)

    # 3) Train/Val 분할
    train_rows = [r for r in rows if r['video'] in train_videos]
    val_rows   = [r for r in rows if r['video'] in val_videos]
    print(f"Train rows: {len(train_rows)}, Val rows: {len(val_rows)}")

    # 4) Dataset
    train_ds = CollisionVideoDataset(
        train_rows,
        queue_size=QUEUE_SIZE,
        after_label_start=AFTER_LABEL_START,
        after_label_finish=AFTER_LABEL_FINISH,
        feature_cols=FEATURE_COLS
    )
    val_ds = CollisionVideoDataset(
        val_rows,
        queue_size=QUEUE_SIZE,
        after_label_start=AFTER_LABEL_START,
        after_label_finish=AFTER_LABEL_FINISH,
        feature_cols=FEATURE_COLS
    )

    print("Train DS samples:", len(train_ds))
    print("Val DS samples:", len(val_ds))

    # DataLoader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8 if MODEL_TYPE == "transformer" else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # 모델 선택
    
    if MODEL_TYPE == "convLSTM_attn":          # 원하는 문자열로 지정
        model = ConvLSTMAttentionClassifier()    
    elif MODEL_TYPE == "transformer":
        model = TemporalTransformerClassifier(
            input_dim=len(FEATURE_COLS),
            d_model=64,
            nhead=4,
            num_layers=2,
            num_classes=2,
            queue_size=QUEUE_SIZE,
            dropout=0.1
        )
    elif MODEL_TYPE == "convLSTM_attn":
        model = ConvLSTMAttentionClassifier()
    elif MODEL_TYPE == "dual_branch":
        model = DualBranchOverlapClassifier()      
    elif MODEL_TYPE == "convLSTM":
        model = ConvLSTMClassifier(
        input_dim=len(FEATURE_COLS),
        conv_channels=32,
        lstm_hidden=64,
        num_classes=2,
        lstm_layers=1,
        bidirectional=True
    )
    else:
        print("Unknown MODEL_TYPE=", MODEL_TYPE)
        return

    # 학습
    train_pytorch_model(model, train_loader, val_loader,
                        epochs=EPOCHS, lr=LEARNING_RATE, device=device)


if __name__ == "__main__":
    main()
