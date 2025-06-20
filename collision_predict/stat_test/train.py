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

try:
    import xgboost as xgb
    from catboost import CatBoostClassifier
except ImportError:
    xgboost = None
    CatBoostClassifier = None

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
                        
                # ─── ①-A : 100-프레임 충돌 비율 계산 ★추가★ ──
                ratio_end = min(end_t + 100, N-1)               # 최신 프레임 기준 +100
                future_chunk_100 = labels[end_t:ratio_end+1]     # 길이 ≤101
                future_ratio_100 = sum(future_chunk_100) / len(future_chunk_100)
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

                # all_samples 에 저장
                all_samples.append({
                    'video'            : video,
                    'rep_frame'        : rep_frame,
                    'x_array'          : x_array,
                    'label'            : y_val,
                    'future_ratio_100' : future_ratio_100      # ★새 키★
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





class TemporalTransformerClassifier(nn.Module):
    """Transformer Encoder 기반 시계열 분류기 (Self‑Attention)"""
    def __init__(self, input_dim=len(FEATURE_COLS), d_model=64, nhead=4,
                 num_layers=2, num_classes=2, queue_size=QUEUE_SIZE, dropout=0.1):
        super().__init__()
        self.queue_size = queue_size
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, queue_size, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        # init
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x : (B, queue_size, input_dim)
        b, t, _ = x.shape
        h = self.input_fc(x)               # (B, T, d_model)
        h = h + self.pos_embedding[:, :t]  # positional encoding
        cls = self.cls_token.expand(b, -1, -1)  # (B,1,d)
        h = torch.cat([cls, h], dim=1)     # prepend [CLS]
        enc = self.encoder(h)              # (B,1+T,d)
        cls_h = enc[:, 0]                  # (B,d)
        logits = self.head(self.norm(cls_h))
        return logits





#######################
# struct2
#######################
class Struct2Model(nn.Module):
    """
    2) 1200x7 -> bounding box xy combine ->??
       actually "xy" combine => doc says "여러 객체 x,y를 하나의 값" => we can interpret e.g. sqrt((x2-x1)^2+(y2-y1)^2)? or x1,y1,x2,y2 =>?? 
       Then we produce 1200x4 => fc layer =>1200x1 => 
       Then =>1200->256->64->2
    """
    def __init__(self, queue_size=1200):
        super().__init__()
        self.queue_size= queue_size
        self.act = get_activation()

        # step1: 7->4 (we combine x1,y1,x2,y2 => 1 scalar?), let's do something
        # but let's do a small FC => 7->4
        self.fc_combine= nn.Linear(7,4)

        # step2: => 1200x4 -> 1200x1
        self.fc_step= nn.Linear(4,1)

        # step3: 1200->256->64->2
        self.fcA= nn.Linear(1200,256)
        self.fcB= nn.Linear(256,64)
        self.fcC= nn.Linear(64,2)

    def forward(self,x):
        # x: (b,1200,7)
        b,qs,fd= x.shape
        x2d= x.view(b*qs, fd) # (b*qs,7)
        out2d= self.fc_combine(x2d) # =>(b*qs,4)
        out2d= self.act(out2d)

        out2d2= self.fc_step(out2d) # =>(b*qs,1)
        out3d= out2d2.view(b,qs)

        # next =>(b,qs=1200)-> (b,256)->(b,64)->(b,2)
        h= self.fcA(out3d)
        h= self.act(h)
        h= self.fcB(h)
        h= self.act(h)
        logits= self.fcC(h)
        return logits


#######################
# struct3
#######################
class Struct3Model(nn.Module):
    """
    3) 1200x7->1200x1(FC)
       1200x1에서 max pooling을 10크기 슬라이딩 => 5개? 
       "1..10=>max, 2..11=>max, ..., 6..15 =>??" 
       Or doc says "이렇게5개" => so we get shape(b,1200-9=1192?), 
       but question states "120x 5" ??? 
       Possibly means chunk=10 with stride=?
       Let's interpret doc: "10개씩 pooling, 한칸씩 이동 => 1200-10+1=1191 => 5" ???

       We'll do: (b,1200) -> sliding window=10 => total 5 times => or doc says 120? 
       The doc text is ambiguous. We'll interpret "we produce 5 windows"? Then 120 => ???

       We'll do a simpler: "take 5 distinct pools"? We'll do an example:
         i=0 => pool(0..9), i=1 => pool(1..10), i=2 => pool(2..11), i=3=>..., i=4=>... 
         => 5 total. => shape= (b,5)
       Then flatten => 5 => (b,5)->(b,600)? doc is uncertain. We'll do a direct approach.

    doc says "120 x 5 => flatten->600->128->32->2"
    Possibly means from (b,1200) => chunk=10 => 120 segments => each segment is max => (b,120)
    Then we do "5 repeated"? It's not fully clear. We'll follow the doc: 
    "이렇게 5개를 하면 120 x 5 => 600"
    We'll do a direct interpretation:
     step1: "1200->(120,10)" -> each chunk => max => (120)
     step2: we "slide" 5 times offset? => we get shape(120,5)? => => flatten=600

    We'll do: offset=0..4 => chunk(0..9,1..10,...?), 5 different 120 arrays => cat => (120,5) => flatten=600
    """
    def __init__(self, queue_size=1200):
        super().__init__()
        self.act= get_activation()

        # step1: fc => 7->1
        self.fc_step= nn.Linear(7,1)

        # final mlp 600->128->32->2
        self.fcA= nn.Linear(600,128)
        self.fcB= nn.Linear(128,32)
        self.fcC= nn.Linear(32,2)

    def forward(self,x):
        # x:(b,1200,7)->(b,1200)
        b,qs,fd= x.shape
        x2d= x.view(b*qs,fd)
        out2d= self.fc_step(x2d) # =>(b*qs,1)
        out3d= out2d.view(b,qs).squeeze(-1) # (b,1200)

        # chunk=10 => we produce (b,120) for offset=0 => chunk i*(10)..i*(10)+9
        # but we want sliding offset 0..4 => each produce (b,120)
        # let's define a function:
        def chunk_pooling(in2d, offset):
            # offset from 0..9 => step=10 => total 120 chunks
            # in2d shape(b,1200)
            # for i in range(120): chunk from offset+ i*10 .. offset+ i*10+9
            # max
            out_list=[]
            for i in range(120):
                start= offset + i*10
                end= start+10
                # if end>1200 => break
                if end> qs: 
                    # break or fill 0
                    break
                block= in2d[:, start:end] # (b,10)
                blkmax= block.max(dim=1)[0] # (b,)
                out_list.append(blkmax)
            # out shape =>(b,120) if we got exactly 120 blocks
            # if the last chunk is incomplete, might produce <120
            # let's do safe approach => min(120,(qs-offset)//10)
            # but doc says it's fine => we'll assume exactly 120
            out_res= torch.stack(out_list, dim=1) # (b, #chunks)
            return out_res

        # do offset=0..4 => we get 5 arrays => cat => shape(b,120,5)
        pool_list=[]
        for offset in range(5):
            p= chunk_pooling(out3d, offset) # (b,120) 
            pool_list.append(p)
        # cat along dim=2 => shape(b,120,5)
        # then flatten => (b,120*5=600)
        cat_3d= torch.stack(pool_list,dim=2) # shape(b,120,5)
        cat_2d= cat_3d.view(b,120*5) # (b,600)

        # final mlp
        h= self.fcA(cat_2d)
        h= self.act(h)
        h= self.fcB(h)
        h= self.act(h)
        logits= self.fcC(h)
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




###################
# struct4 => struct1 + Residual
###################
class ResidualStruct1Model(nn.Module):
    """
    struct1 => 1200x7->1200x1-> chunk=10 =>120 => 32 =>2
    but inside a residual style MLP can be used.
    We'll do final MLP with residual blocks
    """
    def __init__(self, feature_dim=7, chunk_size=10, blocks=2):
        super().__init__()
        self.act= get_activation()
        self.feature_dim= feature_dim
        self.chunk_size= chunk_size
        self.fc_step= nn.Linear(feature_dim,1)

        # final => we get (b,120) => pass into a small residual MLP => out
        self.fc_in= nn.Linear(120,32)
        self.resblocks= nn.ModuleList([ResidualBlock(32) for _ in range(blocks)])
        self.fc_out= nn.Linear(32,2)

    def forward(self,x):
        b,qs,fd= x.shape
        x2d= x.view(b*qs, fd)
        out2d= self.fc_step(x2d)
        out3d= out2d.view(b,qs).squeeze(-1) # (b,qs=1200)

        # chunk=10 => produce (b,120)
        # simple: reshape(b,120,10)->max =>(b,120)
        resh= out3d.view(b, 120, self.chunk_size)
        pooled= resh.max(dim=2)[0] # (b,120)

        # residual MLP
        h= self.fc_in(pooled) # (b,32)
        h= nn.BatchNorm1d(32).to(h.device)(h)
        h= self.act(h)

        for block in self.resblocks:
            h= block(h)

        logits= self.fc_out(h)
        return logits


###################
# struct5 => struct2 + Residual
###################
class ResidualStruct2Model(nn.Module):
    """
    struct2:
      step1: 7->4
      step2: =>1200x4->1200x1
      =>(b,1200)->(b,256)->(b,64)->(b,2)
    + residual in final part
    """
    def __init__(self, queue_size=1200):
        super().__init__()
        self.act= get_activation()

        self.fc_combine= nn.Linear(7,4)
        self.fc_step= nn.Linear(4,1)

        # final =>(b,1200)-> residual MLP or partial
        self.fcA= nn.Linear(1200,256)
        self.resblock= ResidualBlock(256)
        self.fcB= nn.Linear(256,64)
        self.fcC= nn.Linear(64,2)

    def forward(self,x):
        b,qs,fd= x.shape
        x2d= x.view(b*qs, fd)
        out2d= self.fc_combine(x2d)
        out2d= self.act(out2d)
        out2d2= self.fc_step(out2d)
        out3d= out2d2.view(b,qs)

        # =>(b,1200)-> fcA->res->fcB->fcC
        h= self.fcA(out3d)
        # bn? let's do inline
        bn= nn.BatchNorm1d(256).to(h.device)
        h= bn(h)
        h= self.act(h)

        h= self.resblock(h) # shape(b,256)
        h= self.fcB(h)
        h= self.act(h)
        logits= self.fcC(h)
        return logits


###################
# struct6 => struct3 + Residual
###################
class ResidualStruct3Model(nn.Module):
    """
    struct3:
      1200x7->1200x1-> sliding maxpool =>(b,600)->128->32->2
    + residual blocks in final MLP
    """
    def __init__(self):
        super().__init__()
        self.act= get_activation()
        self.fc_step= nn.Linear(7,1)

        self.fcA= nn.Linear(600,128)
        self.resblock= ResidualBlock(128)
        self.fcB= nn.Linear(128,32)
        self.fcC= nn.Linear(32,2)

    def chunk_pooling(self, in2d, offset, qs=1200):
        # same logic as struct3
        out_list=[]
        for i in range(120):
            start= offset + i*10
            end= start+10
            if end> qs:
                break
            block= in2d[:, start:end]
            blkmax= block.max(dim=1)[0]
            out_list.append(blkmax)
        out_res= torch.stack(out_list, dim=1)
        return out_res

    def forward(self,x):
        # x:(b,1200,7)->(b,1200)
        b,qs,fd= x.shape
        x2d= x.view(b*qs,fd)
        out2d= self.fc_step(x2d)
        out3d= out2d.view(b,qs).squeeze(-1)

        # sliding offset=0..4 => cat => (b,120,5)->(b,600)
        pool_list=[]
        for offset in range(5):
            p= self.chunk_pooling(out3d, offset, qs=qs) # (b,120)
            pool_list.append(p)
        cat_3d= torch.stack(pool_list, dim=2) # (b,120,5)
        cat_2d= cat_3d.view(b,120*5) # (b,600)

        # final MLP w/ residual
        h= self.fcA(cat_2d)
        h= self.act(h)
        h= self.resblock(h) # (b,128)
        h= self.fcB(h)
        h= self.act(h)
        logits= self.fcC(h)
        return logits


########################################
# 3) XGBoost / CatBoost Train 함수
########################################
def train_xgboost(X_train, y_train, X_val, y_val):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'eta': 0.1,
        'max_depth': 6
    }
    watchlist = [(dtrain,'train'), (dval,'val')]
    bst = xgb.train(params, dtrain, num_boost_round=50, evals=watchlist)
    return bst

def train_catboost(X_train, y_train, X_val, y_val):
    model = CatBoostClassifier(iterations=50, learning_rate=0.1, depth=6, verbose=10)
    model.fit(X_train, y_train, eval_set=(X_val,y_val))
    return model


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


# version 1
# def train_pytorch_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, device='cpu'):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     model.to(device)
#     for epoch in range(epochs):
#         # ------------------- Train -------------------
#         model.train()
#         train_losses = 0.0
#         train_total = 0
#         all_preds_train = []
#         all_labels_train = []

#         for x_seq, y_label in train_loader:
#             x_seq = x_seq.to(device, dtype=torch.float32)
#             y_label = y_label.to(device, dtype=torch.long)

#             optimizer.zero_grad()
#             logits = model(x_seq)
#             loss = criterion(logits, y_label)
#             loss.backward()
#             optimizer.step()

#             train_losses += loss.item() * x_seq.size(0)
#             train_total  += x_seq.size(0)

#             preds = logits.argmax(dim=1).cpu().numpy()
#             all_preds_train.extend(preds)
#             all_labels_train.extend(y_label.cpu().numpy())

#         train_loss = train_losses / train_total
#         acc_t, prec_t, rec_t, f1_t = evaluate_metrics(all_labels_train, all_preds_train)

#         # ------------------- Validation -------------------
#         model.eval()
#         val_losses = 0.0
#         val_total = 0
#         all_preds_val = []
#         all_labels_val = []

#         ds = val_loader.dataset
#         sample_indices_in_epoch = []

#         with torch.no_grad():
#             for batch_idx, (x_seq, y_label) in enumerate(val_loader):
#                 batch_size = x_seq.shape[0]
#                 sample_indices_in_epoch.extend(range(val_total, val_total+batch_size))

#                 logits = model(x_seq.to(device,dtype=torch.float32))
#                 loss = criterion(logits, y_label.to(device,dtype=torch.long))

#                 val_losses += loss.item() * batch_size
#                 val_total  += batch_size

#                 preds = logits.argmax(dim=1).cpu().numpy()
#                 y_label_np = y_label.cpu().numpy()
#                 all_preds_val.extend(preds)
#                 all_labels_val.extend(y_label_np)

#         val_loss = val_losses / val_total
#         acc_v, prec_v, rec_v, f1_v = evaluate_metrics(all_labels_val, all_preds_val)

#         print(f"[Epoch {epoch+1}/{epochs}] "
#               f"Train Loss={train_loss:.4f} Acc={acc_t:.4f} Prec={prec_t:.4f} Rec={rec_t:.4f} F1={f1_t:.4f} | "
#               f"Val Loss={val_loss:.4f} Acc={acc_v:.4f} Prec={prec_v:.4f} Rec={rec_v:.4f} F1={f1_v:.4f}")

#         # 시각화
#         frames_for_plot = []
#         labels_for_plot = []
#         preds_for_plot  = []

#         for i, sample_idx in enumerate(sample_indices_in_epoch):
#             rep_f = ds.get_rep_frame(sample_idx)
#             frames_for_plot.append(rep_f)
#             labels_for_plot.append(all_labels_val[i])
#             preds_for_plot.append(all_preds_val[i])

#         frames_for_plot = np.array(frames_for_plot)
#         labels_for_plot = np.array(labels_for_plot)
#         preds_for_plot  = np.array(preds_for_plot)

#         outname = f"results/val_epoch_{epoch+1}.png"
#         plot_three_bars_separated(frames_for_plot, labels_for_plot, preds_for_plot, output_path=outname)

#     print("Training done.\n")

#     y_pred = np.array(all_preds_val)
#     y_true = np.array(all_labels_val)
#     mis_idx = np.where(y_pred != y_true)[0]
#     print(f"Mis-classified Count: {len(mis_idx)}/{len(y_true)}")
#     max_show = min(10, len(mis_idx))
#     if max_show>0:
#         print("Examples of misclassified (index: pred/true):")
#         for i in range(max_show):
#             idx_m = mis_idx[i]
#             print(f"  Idx={idx_m}: Pred={y_pred[idx_m]}, True={y_true[idx_m]}")

#     return model

# # version 2
# def train_pytorch_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, device='cpu'):
#     criterion = nn.CrossEntropyLoss()

#     # ===== NEW ===== Transformer warm‑up optimizer 설정 (옵션)
#     if MODEL_TYPE == "transformer":
#         optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
#     else:
#         optimizer = optim.Adam(model.parameters(), lr=lr)

#     model.to(device)
#     for epoch in range(epochs):
#         # ------------------- Train -------------------
#         model.train()
#         train_losses = 0.0
#         train_total = 0
#         all_preds_train = []
#         all_labels_train = []

#         for x_seq, y_label in train_loader:
#             x_seq = x_seq.to(device, dtype=torch.float32)
#             y_label = y_label.to(device, dtype=torch.long)

#             optimizer.zero_grad()
#             logits = model(x_seq)
#             loss = criterion(logits, y_label)
#             loss.backward()
#             optimizer.step()

#             train_losses += loss.item() * x_seq.size(0)
#             train_total  += x_seq.size(0)

#             preds = logits_to_pred(logits).cpu().numpy()
#             all_preds_train.extend(preds)
#             all_labels_train.extend(y_label.cpu().numpy())

#         train_loss = train_losses / train_total
#         acc_t, prec_t, rec_t, f1_t = evaluate_metrics(all_labels_train, all_preds_train)

#         # ------------------- Validation -------------------
#         model.eval()
#         val_losses = 0.0
#         val_total = 0
#         all_preds_val = []
#         all_labels_val = []

#         ds = val_loader.dataset
#         sample_indices_in_epoch = []
#         frame_buf, prob_buf, vid_buf = [], [], []

#         with torch.no_grad():
#             for batch_idx, (x_seq, y_label) in enumerate(val_loader):
#                 batch_size = x_seq.shape[0]
#                 sample_indices_in_epoch.extend(range(val_total, val_total+batch_size))

#                 logits = model(x_seq.to(device,dtype=torch.float32))
#                 loss = criterion(logits, y_label.to(device,dtype=torch.long))

#                 val_losses += loss.item() * batch_size
#                 val_total  += batch_size
                
#                 probs  = torch.softmax(logits,1)[:,1].cpu().numpy()
#                 preds = logits_to_pred(logits).cpu().numpy()
#                 y_label_np = y_label.cpu().numpy()
#                 all_preds_val.extend(preds)
#                 all_labels_val.extend(y_label_np)

#         val_loss = val_losses / val_total
#         acc_v, prec_v, rec_v, f1_v = evaluate_metrics(all_labels_val, all_preds_val)

#         print(f"[Epoch {epoch+1}/{epochs}] "
#               f"Train Loss={train_loss:.4f} Acc={acc_t:.4f} Prec={prec_t:.4f} Rec={rec_t:.4f} F1={f1_t:.4f} | "
#               f"Val Loss={val_loss:.4f} Acc={acc_v:.4f} Prec={prec_v:.4f} Rec={rec_v:.4f} F1={f1_v:.4f}")

#         # 시각화 (원본 로직 유지)
#         frames_for_plot = []
#         labels_for_plot = []
#         preds_for_plot  = []

#         for i, sample_idx in enumerate(sample_indices_in_epoch):
#             rep_f = ds.get_rep_frame(sample_idx)
#             frames_for_plot.append(rep_f)
#             labels_for_plot.append(all_labels_val[i])
#             preds_for_plot.append(all_preds_val[i])

#         frames_for_plot = np.array(frames_for_plot)
#         labels_for_plot = np.array(labels_for_plot)
#         preds_for_plot  = np.array(preds_for_plot)

#         outname = f"results/val_epoch_{epoch+1}.png"
#         plot_three_bars_separated(frames_for_plot, labels_for_plot, preds_for_plot, output_path=outname)

#     print("Training done.\n")

#     y_pred = np.array(all_preds_val)
#     y_true = np.array(all_labels_val)
#     mis_idx = np.where(y_pred != y_true)[0]
#     print(f"Mis-classified Count: {len(mis_idx)}/{len(y_true)}")
#     max_show = min(10, len(mis_idx))
#     if max_show>0:
#         print("Examples of misclassified (index: pred/true):")
#         for i in range(max_show):
#             idx_m = mis_idx[i]
#             print(f"  Idx={idx_m}: Pred={y_pred[idx_m]}, True={y_true[idx_m]}")

#     return model



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
    elif MODEL_TYPE == "struct2":
        from_struct2 = Struct2Model(queue_size=QUEUE_SIZE)
        model = from_struct2
    elif MODEL_TYPE == "struct3":
        from_struct3 = Struct3Model()
        model = from_struct3
    elif MODEL_TYPE == "struct4":
        from_struct4 = ResidualStruct1Model(feature_dim=len(FEATURE_COLS), chunk_size=10, blocks=2)
        model = from_struct4
    elif MODEL_TYPE == "struct5":
        from_struct5 = ResidualStruct2Model(queue_size=QUEUE_SIZE)
        model = from_struct5
    elif MODEL_TYPE == "struct6":
        from_struct6 = ResidualStruct3Model()
        model = from_struct6
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
    elif MODEL_TYPE == "XGBOOST":
        if xgboost is None:
            print("XGBoost not installed!")
            return
        X_train, y_train = dataset_to_numpy(train_ds)
        X_val, y_val = dataset_to_numpy(val_ds)
        bst = train_xgboost(X_train, y_train, X_val, y_val)
        print("XGBoost model trained.")
        return
    elif MODEL_TYPE == "CATBOOST":
        if CatBoostClassifier is None:
            print("CatBoost not installed!")
            return
        X_train, y_train = dataset_to_numpy(train_ds)
        X_val, y_val = dataset_to_numpy(val_ds)
        model = train_catboost(X_train, y_train, X_val, y_val)
        print("CatBoost model trained.")
        return
    else:
        print("Unknown MODEL_TYPE=", MODEL_TYPE)
        return

    # 학습
    train_pytorch_model(model, train_loader, val_loader,
                        epochs=EPOCHS, lr=LEARNING_RATE, device=device)


if __name__ == "__main__":
    main()
