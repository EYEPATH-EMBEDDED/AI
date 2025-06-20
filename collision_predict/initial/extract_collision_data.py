import math
import csv
from shapely.geometry import Polygon

###################################################
# 사용자 파라미터
###################################################
theta_v_deg = 52.0   # 수직 시야각
theta_h_deg = 65.0   # 수평 시야각
height_m    = 1.70   # 키 (카메라 높이)
torso_m     = 0.45   # 몸통 너비
IMAGE_WIDTH  = 1280
IMAGE_HEIGHT = 720

# 가중치 함수 파라미터
ALPHA = 5.0
BETA  = 5.0

###################################################
# 1) 충돌 영역 삼각형 정의
###################################################
import math
from shapely.geometry import Polygon

def compute_distance_from_vertical_fov(h, theta_v_deg):
    """
    그림2 공식: d = h * tan((theta_v)/2)
    """
    theta_v_rad = math.radians(theta_v_deg)
    return h * math.tan(theta_v_rad / 2)

def compute_collision_ratio(a, d, theta_h_deg):
    """
    그림3 공식: ratio = a / [2 * d * tan((theta_h)/2)]
    """
    theta_h_rad = math.radians(theta_h_deg)
    denom = 2 * d * math.tan(theta_h_rad / 2)
    return a / denom  # 0~1 (이상으로 클 수도 있음)

def define_collision_triangle():
    d = compute_distance_from_vertical_fov(height_m, theta_v_deg)
    ratio = compute_collision_ratio(torso_m, d, theta_h_deg)

    base_px = ratio * IMAGE_WIDTH
    half_base = base_px / 2

    apex_x = IMAGE_WIDTH / 2  # 640
    apex_y = IMAGE_HEIGHT / 2 # 360
    left_pt  = (apex_x - half_base, IMAGE_HEIGHT)  # (640-half_base, 720)
    right_pt = (apex_x + half_base, IMAGE_HEIGHT)  # (640+half_base, 720)
    apex_pt  = (apex_x, apex_y)                    # (640, 360)

    poly = Polygon([apex_pt, left_pt, right_pt])
    return poly

collision_zone_poly = define_collision_triangle()
collision_zone_area = collision_zone_poly.area

###################################################
# 2) Overlap ratio
###################################################
def compute_overlap_ratio(x1, y1, x2, y2):
    X1, X2 = sorted([x1, x2])
    Y1, Y2 = sorted([y1, y2])
    bbox_poly = Polygon([(X1,Y1),(X2,Y1),(X2,Y2),(X1,Y2)])
    inter_area = collision_zone_poly.intersection(bbox_poly).area
    ratio = inter_area / collision_zone_area if collision_zone_area>0 else 0
    return ratio

###################################################
# 3) Weight function (수정: y=0 근처 가중치↑)
###################################################
def weight_function(x1, y1, x2, y2):
    """
    x 가중치: x=640 중심
        wx = exp(-ALPHA * ((xc-640)/(base/2))^2)
    y 가중치: y=0(화면 상단)에 가까울수록 큼
        wy = exp(-BETA * (yc/360))  # yc=0 => wy=1, yc=360 => e^-BETA
    """
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2

    base_px = collision_zone_poly.bounds[2] - collision_zone_poly.bounds[0]
    half_base = base_px / 2

    # x weight
    x_rel = (xc - 640.0) / half_base
    wx = math.exp(-ALPHA * (x_rel**2))

    # y weight (yc=0 => wy=1, yc=360 => wy=e^-BETA)
    y_rel = yc / 360.0
    wy = math.exp(-BETA * y_rel)

    return wx * wy

###################################################
# 4) CSV 처리
###################################################
def process_bboxes(
    input_csv='bboxes.csv',
    output_csv1='bbox_overlap.csv',
    output_csv2='bbox_overlap_weight.csv'
):
    with open(input_csv, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)

    # (1) overlap_ratio
    with open(output_csv1, 'w', newline='', encoding='utf-8') as f1:
        w1 = csv.writer(f1)
        w1.writerow(['video','frame','box_idx','x1','y1','x2','y2','overlap_ratio'])
        for row in rows:
            video = row['video']
            frame = row['frame']
            box_idx = row['box_idx']
            x1 = float(row['x1'])
            y1 = float(row['y1'])
            x2 = float(row['x2'])
            y2 = float(row['y2'])

            overlap = compute_overlap_ratio(x1,y1,x2,y2)
            w1.writerow([video, frame, box_idx, x1, y1, x2, y2, f"{overlap:.4f}"])

    # (2) overlap_ratio + weight
    with open(output_csv2, 'w', newline='', encoding='utf-8') as f2:
        w2 = csv.writer(f2)
        w2.writerow(['video','frame','box_idx','x1','y1','x2','y2','overlap_ratio','weight_value'])
        for row in rows:
            video = row['video']
            frame = row['frame']
            box_idx = row['box_idx']
            x1 = float(row['x1'])
            y1 = float(row['y1'])
            x2 = float(row['x2'])
            y2 = float(row['y2'])

            overlap = compute_overlap_ratio(x1,y1,x2,y2)
            wval = weight_function(x1,y1,x2,y2)
            w2.writerow([video, frame, box_idx, x1, y1, x2, y2, 
                         f"{overlap:.4f}", f"{wval:.4f}"])

    print(f"[DONE] CSV created:\n 1) {output_csv1}\n 2) {output_csv2}")

if __name__=="__main__":
    process_bboxes()
