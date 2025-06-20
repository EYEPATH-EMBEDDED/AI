import torch, torch.nn as nn
from dataset_depth import DepthCollisionDataset   # 같은 패키지라고 가정

class DepthConv3DLSTM(nn.Module):
    """
    Depth 3D CNN  +  bbox 1D CNN  →  concat  →  Conv1D + Bi-LSTM
    """
    def __init__(self, q_len=1200, bbox_dim=6,
                 d_ch=(1,8,16,32,64), lstm_h=128):
        super().__init__()
        # --- ① 3D CNN 블록 ------------------------------
        blocks=[]
        for cin,cout in zip(d_ch[:-1], d_ch[1:]):
            blocks.append(nn.Conv3d(cin, cout, 3, padding=1))
            blocks.append(nn.BatchNorm3d(cout))
            blocks.append(nn.ReLU(inplace=True))
        self.conv3d = nn.Sequential(*blocks)          # 4개 층

        # --- ② 1D CNN (bbox) ----------------------------
        self.conv1d_bbox = nn.Sequential(
            nn.Conv1d(bbox_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # --- ③ Fusion + Conv1D --------------------------
        self.conv1d_fuse = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

        # --- ④ Bi-LSTM + FC -----------------------------
        self.lstm = nn.LSTM(128, lstm_h,
                            batch_first=True,
                            bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_h*2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    # ----------------------------------------------------
    def forward(self, depth, feat):
        """
        depth : (B,1,T,H,W)
        feat  : (B,T,6)
        """
        # depth branch
        d = self.conv3d(depth)               # (B,64,T,H,W)
        d = d.max(-1)[0].max(-1)[0]          # GMP → (B,64,T)
        d = d.permute(0,2,1)                 # (B,T,64)

        # bbox branch
        b = feat.permute(0,2,1)              # (B,6,T)
        b = self.conv1d_bbox(b)              # (B,64,T)
        b = b.permute(0,2,1)                 # (B,T,64)

        # concat & fuse
        x = torch.cat([d,b], dim=-1)         # (B,T,128)
        x = self.conv1d_fuse(x.permute(0,2,1))\
                    .permute(0,2,1)          # 유지 (B,T,128)

        # LSTM
        lstm_out,_ = self.lstm(x)            # (B,T,256)
        cls = lstm_out[:,-1]                 # 마지막 step
        return self.head(cls)
