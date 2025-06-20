# analyze_iou_distribution.py
import os, csv, math, random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, pointbiserialr
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

# ─────────────────────────────────────────────────────────────
# ❶  ▶▶ ❷ ❸ : 지금까지 쓰던 유틸 그대로 재-사용
# ─────────────────────────────────────────────────────────────
QUEUE_SIZE         = 1200
AFTER_LABEL_START  = 50
AFTER_LABEL_FINISH = 400
FEATURE_COLS       = ['frame','time','x1','y1','x2','y2','overlap_ratio']   # ← 반드시 overlap_ratio 포함
CSV_PATH           = 'final_collision.csv'
SAVE_FIG           = 'results/iou_mean_distribution.png'
RANDOM_SEED        = 10            # 데이터 split과 무관, 단지 de 동일성

# ※ 아래 두 클래스/함수는 <train.py> 안에 이미 정의돼 있으므로
#   "import train" 처럼 불러와도 되고, 여기로 copy-paste 해도 OK.
from train import CollisionVideoDataset, load_final_collision   # 경로 맞게 수정

# ─────────────────────────────────────────────────────────────
# ❹  데이터셋 → 샘플별 ‘mean IOU’ 수집
# ─────────────────────────────────────────────────────────────
rows = load_final_collision(CSV_PATH)

ds = CollisionVideoDataset(
        rows,
        queue_size        = QUEUE_SIZE,
        after_label_start = AFTER_LABEL_START,
        after_label_finish= AFTER_LABEL_FINISH,
        feature_cols      = FEATURE_COLS
)

iou_idx = FEATURE_COLS.index('overlap_ratio')     # 열 위치
pos_means, neg_means = [], []                    # 1 / 0

for i in range(len(ds)):
    x, y = ds[i]                                 # x:(1200,7)  y:0/1
    mean_iou = float(np.mean(x[:, iou_idx]))     # 큐 내부 평균
    (pos_means if y==1 else neg_means).append(mean_iou)

pos_means = np.array(pos_means)
neg_means = np.array(neg_means)

print(f"Samples 1(cls)={len(pos_means)}   0(cls)={len(neg_means)}")
print(f"  ▸ mean(IOU) 1 = {pos_means.mean():.4f}   0 = {neg_means.mean():.4f}")

# ─────────────────────────────────────────────────────────────
# ❺  히스토그램(또는 KDE) + 통계검정
# ─────────────────────────────────────────────────────────────
plt.figure(figsize=(8,4))
bins = np.linspace(0, 1, 40)

plt.hist(neg_means, bins=bins, alpha=.6,
         label=f'Non-collision (N={len(neg_means)})', color='skyblue')
plt.hist(pos_means, bins=bins, alpha=.6,
         label=f'Collision (N={len(pos_means)})',  color='salmon')

plt.axvline(pos_means.mean(),  color='red',   ls='--', lw=1.2)
plt.axvline(neg_means.mean(),  color='blue',  ls='--', lw=1.2)
plt.xlabel('Mean IOU within 1 × 1200 queue')
plt.ylabel('Count')
plt.legend()
plt.title('Distribution of queue-level mean IOU')

os.makedirs(os.path.dirname(SAVE_FIG), exist_ok=True)
plt.tight_layout()
plt.savefig(SAVE_FIG, dpi=120)
plt.close()
print(f"[✓] Histogram saved → {SAVE_FIG}")

# ─────────────────────────────────────────────────────────────
# ❻  p-value : 두 분포 차이가 유의한가?
#     (SciPy 가 있으면 Mann-Whitney U, 없으면 간단 t-test)
# ─────────────────────────────────────────────────────────────
try:
    from scipy.stats import mannwhitneyu
    u, p_val = mannwhitneyu(pos_means, neg_means, alternative='two-sided')
    print(f"Mann-Whitney U={u:.1f},  p = {p_val:.3e}")
except ImportError:
    # fallback : Welch-t
    from math import sqrt
    m1, m0 = pos_means.mean(),  neg_means.mean()
    v1, v0 = pos_means.var(ddof=1), neg_means.var(ddof=1)
    n1, n0 = len(pos_means), len(neg_means)
    t = (m1-m0) / sqrt(v1/n1 + v0/n0)
    print("(SciPy 없음) Welch-t ≈ ", t)

# ──────────────────────────────────
# ★ NEW ❶ :  단측 Mann-Whitney  (μ₁ > μ₀ 인지)
# ──────────────────────────────────
from scipy.stats import mannwhitneyu, pointbiserialr
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

u_two,  p_two  = mannwhitneyu(pos_means, neg_means,
                              alternative='two-sided')
u_g,    p_one  = mannwhitneyu(pos_means, neg_means,
                              alternative='greater')

def pretty_p(p):
    return f"p < 1e-300" if p < 1e-300 else f"p = {p:.3e}"

print(f"[Mann-Whitney] two-sided U={u_two:.0f},  {pretty_p(p_two)}")
print(f"[Mann-Whitney] H1: μ₁>μ₀  U={u_g:.0f},   {pretty_p(p_one)}")

# ──────────────────────────────────
# ★ NEW ❷ :  IOU – 충돌 상관/회귀
# ──────────────────────────────────
all_iou  = np.concatenate([pos_means, neg_means])
all_lab  = np.concatenate([np.ones_like(pos_means),
                           np.zeros_like(neg_means)])

r_pb, p_pb = pointbiserialr(all_lab, all_iou)
print(f"[Point-biserial] r = {r_pb:.3f},  {pretty_p(p_pb)}")

X   = sm.add_constant(all_iou)          # (N,2) : [1, meanIOU]
logit = sm.Logit(all_lab, X).fit(disp=False)
beta1  = logit.params[1]
p_b1   = logit.pvalues[1]
OR     = math.exp(beta1)
print(f"[Logistic] β₁ = {beta1:.3f}  (OR = {OR:.2f}),  {pretty_p(p_b1)}")

auc = roc_auc_score(all_lab, all_iou)
print(f"[AUC]  AUROC(mean-IOU) = {auc:.3f}")

# ──────────────────────────────────
# ★ NEW ❸ :  히스토그램 → density & x축 0-0.2  (기존 ❺ 대신)
# ──────────────────────────────────
plt.figure(figsize=(8,4))
bins = np.linspace(0, 0.07, 60)
plt.hist(neg_means, bins=bins, density=True, alpha=.6,
         label=f'Non-collision (N={len(neg_means)})', color='skyblue')
plt.hist(pos_means, bins=bins, density=True, alpha=.6,
         label=f'Collision (N={len(pos_means)})',  color='salmon')
plt.axvline(neg_means.mean(), color='blue', ls='--', lw=1.2)
plt.axvline(pos_means.mean(), color='red',  ls='--', lw=1.2)
plt.xlabel('Mean IOU within queue')
plt.ylabel('Probability Density')
plt.title('Distribution of queue-level mean IOU (density)')
plt.legend()
plt.tight_layout()
plt.savefig(SAVE_FIG, dpi=120)
plt.close()
print(f"[✓] Density-hist saved → {SAVE_FIG}")

# ──────────────────────────────────
# ★ NEW ❹ :  산점도  (x=mean-IOU, y=future label ratio)
# ──────────────────────────────────
scatter_path = 'results/iou_vs_future_ratio.png'
mx, my = [], []
for s in ds.samples:
    mx.append(s['x_array'][:, iou_idx].mean())
    my.append(s['future_ratio_100'])   # ← 방금 저장한 값
plt.figure(figsize=(7,4))
plt.scatter(mx, my, s=6, alpha=.15, color='purple')
plt.xlim(0, 0.2); plt.ylim(-0.02, 1.02)
plt.xlabel('Queue mean IOU'); plt.ylabel('Future frame collision ratio')
plt.title('IOU & future collision avg ratio')
os.makedirs(os.path.dirname(scatter_path), exist_ok=True)
plt.tight_layout(); plt.savefig(scatter_path, dpi=120); plt.close()
print(f"[✓] Scatter saved → {scatter_path}")


# ...
bins = np.linspace(0, 0.07, 60)

def plot_prob(hist_data, color, label):
    # 각 집단을 0~100 %로 맞춤
    weights = np.ones_like(hist_data) / len(hist_data) * 100   # 100% 스케일
    plt.hist(hist_data, bins=bins, weights=weights,
             alpha=.6, color=color, label=f'{label} (N={len(hist_data)})')
dist_dir = 'results/iou_collision.png'
plt.figure(figsize=(8,4))
plot_prob(neg_means, 'skyblue', 'Non-collision')
plot_prob(pos_means, 'salmon',  'Collision')

plt.axvline(neg_means.mean(), color='blue', ls='--', lw=1.2)
plt.axvline(pos_means.mean(), color='red',  ls='--', lw=1.2)
plt.xlabel('Mean IOU within queue')
plt.ylabel('Percentage per bin (%)')   # ← y축 의미 변경
plt.title('IOU Distribution (each group sums to 100 %)')
plt.legend()
plt.tight_layout(); plt.savefig(dist_dir, dpi=120); plt.close()

