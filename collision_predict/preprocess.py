import os
import csv
import math
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon

#############################################
# 0) 유틸: 진행 상황 표시 (간단한 퍼센트 출력)
#############################################
def print_progress(current, total, prefix="Progress"):
    percent = (current / total) * 100
    print(f"{prefix}: {current}/{total} ({percent:.1f}%)")


#############################################
# 1) Preprocess: 비디오 -> frame_labeled_data.csv
#############################################

def parse_time_string(t_str):
    """
    'MM:SS.xx' 또는 'HH:MM:SS.xx' 형식의 문자열을 초(float)로 변환.
    예: '00:09.96' -> 9.96, '01:14.90' -> 74.90
    """
    parts = t_str.split(':')
    if len(parts) == 2:  # MM:SS.xx
        mm = int(parts[0])
        ss = float(parts[1])
        total_seconds = mm * 60 + ss
    elif len(parts) == 3:  # HH:MM:SS.xx
        hh = int(parts[0])
        mm = int(parts[1])
        ss = float(parts[2])
        total_seconds = hh * 3600 + mm * 60 + ss
    else:
        raise ValueError(f"Time format not recognized: {t_str}")
    return total_seconds

def load_toggle_data(csv_path):
    """
    CSV 컬럼: Label, Time, video
    - video : "1.mp4" 같은 파일 이름
    - Time  : "MM:SS.xx" (또는 "HH:MM:SS.xx")
    - Label : 정수 (1,2,3...)
      * 홀수이면 OFF->ON 토글
      * 짝수이면 ON->OFF 토글
    
    반환값: dict 형태
      { "1.mp4": [(time_sec, label), (time_sec, label), ...], ... }
    """
    toggle_dict = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_str = row['Label']
            time_str = row['Time']
            video_str = row['video']

            label = int(label_str)
            time_sec = parse_time_string(time_str)

            if video_str not in toggle_dict:
                toggle_dict[video_str] = []
            toggle_dict[video_str].append((time_sec, label))

    # 시간 순 정렬
    for v in toggle_dict:
        toggle_dict[v].sort(key=lambda x: x[0])
    return toggle_dict

def get_onoff_label(time_sec, toggle_list):
    """
    토글 목록 기반: time_sec 이전의 토글을 count하여 0/1 상태 결정
    초기 state=0
    """
    state = 0
    for (t, lbl) in toggle_list:
        if time_sec < t:
            break
        # 토글 발생 -> state 뒤집기
        state = 1 - state
    return state

def process_video(video_path, toggle_list, images_dir, sampling_fps=2):
    """
    - 비디오를 sampling_fps로 샘플링
    - 각 프레임 (video, frame, time, label)을 반환
    - 이미지도 저장 (images/<video_basename>/<video_basename>_<frameidx>.jpg)
    """
    base_name = os.path.basename(video_path)        # ex) "1.mp4"
    name_only, _ = os.path.splitext(base_name)      # ex) "1"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: cannot open {video_path}")
        return []
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / original_fps if original_fps>0 else 0

    step_sec = 1.0 / sampling_fps
    results = []

    save_subdir = os.path.join(images_dir, name_only)
    os.makedirs(save_subdir, exist_ok=True)

    t = 0.0
    while t <= duration_sec:
        f_idx = int(round(t * original_fps))
        if f_idx >= frame_count:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        label = get_onoff_label(t, toggle_list)

        img_filename = f"{name_only}_{f_idx}.jpg"
        img_save_path = os.path.join(save_subdir, img_filename)
        cv2.imwrite(img_save_path, frame)

        results.append((base_name, f_idx, round(t,2), label))
        t += step_sec

    cap.release()
    return results

def step1_preprocess(
    toggle_csv = 'eyepath_data3.csv',
    video_dir = 'videos',
    images_dir = 'images',
    output_csv = 'preprocess_output/frame_labeled_data.csv',
    sampling_fps = 20
):
    """
    1단계: 토글 CSV를 로드 -> 각 비디오별로 샘플링 -> frame_labeled_data.csv 생성
    """
    print("[Step1] Preprocess Start")
    toggle_dict = load_toggle_data(toggle_csv)
    video_list = list(toggle_dict.items())
    total_videos = len(video_list)
    all_rows = []

    for i, (video_file, toggles) in enumerate(video_list, start=1):
        print_progress(i, total_videos, prefix="Preprocess Video")
        video_path = os.path.join(video_dir, video_file)
        rows = process_video(video_path, toggles, images_dir, sampling_fps)
        all_rows.extend(rows)

    # 최종 CSV 저장
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['video','frame','time','label'])
        for row in all_rows:
            w.writerow(row)

    print(f"[Step1 Done] Created {output_csv} with {len(all_rows)} rows.")


#############################################
# 2) BBox 추출 -> bboxes.csv
#############################################
def step2_extract_bboxes(
    input_csv='preprocess_output/frame_labeled_data.csv',
    images_dir='images',
    yolo_weights='last.pt',
    max_boxes=30,
    output_csv='preprocess_output/bboxes.csv'
):
    """
    2단계: frame_labeled_data.csv를 읽어 -> YOLO로 bbox 추출 -> bboxes.csv
    """
    print("[Step2] Extract BBoxes Start")
    model = YOLO(yolo_weights)
    model.conf = 0.25
    model.iou = 0.45

    with open(input_csv, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)

    total_rows = len(rows)
    count = 0

    with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        w = csv.writer(f_out)
        # (video, frame, box_idx, x1,y1,x2,y2)
        # label,time은 여기선 빼고, 다음 단계에서 merge
        w.writerow(['video','frame','box_idx','x1','y1','x2','y2'])

        for i, row in enumerate(rows, start=1):
            print_progress(i, total_rows, prefix="ExtractBBoxes")
            video = row['video']
            frame_idx = int(row['frame'])

            base_name = os.path.splitext(video)[0]
            img_filename = f"{base_name}_{frame_idx}.jpg"
            img_path = os.path.join(images_dir, base_name, img_filename)

            frame_bgr = cv2.imread(img_path)
            if frame_bgr is None:
                continue

            # YOLO 추론
            results = model(frame_bgr, verbose=False)
            boxes = results[0].boxes

            num_b = min(len(boxes), max_boxes)
            for b_idx in range(num_b):
                x1, y1, x2, y2 = boxes[b_idx].xyxy[0]
                # GPU -> float
                x1 = float(x1.detach().cpu().numpy())
                y1 = float(y1.detach().cpu().numpy())
                x2 = float(x2.detach().cpu().numpy())
                y2 = float(y2.detach().cpu().numpy())

                w.writerow([video, frame_idx, b_idx, x1,y1,x2,y2])
                count += 1

    print(f"[Step2 Done] Created {output_csv} with {count} bboxes.")


#############################################
# 3) 충돌영역 계산 -> 최종 CSV
#############################################
# (a) 충돌영역 정의 (수직/수평시야각)
def compute_distance_from_vertical_fov(h, theta_v_deg):
    # d = h * tan((theta_v)/2)
    theta_v_rad = math.radians(theta_v_deg)
    return h * math.tan(theta_v_rad/2)

def compute_collision_ratio(a, d, theta_h_deg):
    # ratio = a / [2*d*tan(theta_h/2)]
    theta_h_rad = math.radians(theta_h_deg)
    denom = 2 * d * math.tan(theta_h_rad/2)
    return a/denom

def define_collision_triangle(height_m=1.70, torso_m=0.45, theta_v_deg=52.0, theta_h_deg=65.0):
    d = compute_distance_from_vertical_fov(height_m, theta_v_deg)
    ratio = compute_collision_ratio(torso_m, d, theta_h_deg)

    base_px = ratio * 1280.0
    half_base = base_px/2.0

    apex_x = 640
    apex_y = 360
    left_pt  = (apex_x - half_base, 720)
    right_pt = (apex_x + half_base, 720)
    apex_pt  = (apex_x, apex_y)

    poly = Polygon([apex_pt, left_pt, right_pt])
    return poly

collision_zone_poly = None
collision_zone_area = None

def compute_overlap_ratio(x1,y1,x2,y2):
    # BBox vs collision_zone_poly
    X1,X2 = sorted([x1,x2])
    Y1,Y2 = sorted([y1,y2])
    bbox_poly = Polygon([(X1,Y1),(X2,Y1),(X2,Y2),(X1,Y2)])
    inter_area = collision_zone_poly.intersection(bbox_poly).area
    ratio = inter_area/collision_zone_area if collision_zone_area>0 else 0
    return ratio

## 공식 확인 필요
def weight_function(x1,y1,x2,y2, alpha=5.0, beta=5.0):
    # 기존 로직: x=640 중심, y=0 상단 가중치↑ or y=360가중치↓등
    # 여기서는 예시로: x=640 근처, y=0 근처 => 가중치↑
    xc = (x1+x2)/2.0
    yc = (y1+y2)/2.0

    base_px = collision_zone_poly.bounds[2] - collision_zone_poly.bounds[0]
    half_base = base_px/2.0

    # x weight
    x_rel = (xc - 640.0)/half_base
    wx = math.exp(-alpha*(x_rel**2))

    # y weight (y=0 근처면 wy=1, y=360 => e^-beta)
    y_rel = yc/360.0
    wy = math.exp(-beta*y_rel)

    return wx*wy

def step3_extract_collision(
    labeled_csv='preprocess_output/frame_labeled_data.csv',
    bbox_csv='preprocess_output/bboxes.csv',
    output_csv='preprocess_output/final_collision.csv',
    height_m=1.70,
    torso_m=0.45,
    theta_v=52.0,
    theta_h=65.0,
    alpha=5.0,
    beta=5.0
):
    """
    3단계: bboxes.csv + frame_labeled_data.csv -> (video, frame, time, label, box_idx, x1,y1,x2,y2, overlap, weight) -> final_collision.csv
    """
    print("[Step3] Collision & Merge Start")

    # 1) frame_labeled_data.csv -> dict {(video, frame) : (time, label)}
    label_dict = {}
    with open(labeled_csv,'r',encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        rows_label = list(rdr)
    for r in rows_label:
        v = r['video']
        fr = int(r['frame'])
        tm = r['time']   # float str
        lb = r['label']  # int str
        label_dict[(v,fr)] = (tm, lb)

    # 2) 충돌 영역 polygon 준비
    global collision_zone_poly, collision_zone_area
    collision_zone_poly = define_collision_triangle(height_m, torso_m, theta_v, theta_h)
    collision_zone_area = collision_zone_poly.area

    # 3) bboxes.csv 읽어서 overlap, weight 계산 + label merge
    with open(bbox_csv, 'r', encoding='utf-8') as f_in:
        rdr = csv.DictReader(f_in)
        bboxes = list(rdr)

    total_bboxes = len(bboxes)

    with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        wout = csv.writer(f_out)
        # 헤더
        wout.writerow([
            'video','frame','time','label','box_idx','x1','y1','x2','y2','overlap_ratio','weight_value'
        ])

        for i, b in enumerate(bboxes, start=1):
            print_progress(i, total_bboxes, prefix="CollisionCalc")

            video = b['video']
            frame = int(b['frame'])
            box_idx = b['box_idx']
            x1 = float(b['x1'])
            y1 = float(b['y1'])
            x2 = float(b['x2'])
            y2 = float(b['y2'])

            # label, time merge
            tm_lb = label_dict.get((video, frame), ("0","0"))
            time_str, label_str = tm_lb

            # overlap, weight
            overlap = compute_overlap_ratio(x1,y1,x2,y2)
            wval = weight_function(x1,y1,x2,y2, alpha, beta)

            wout.writerow([
                video, frame, time_str, label_str, box_idx,
                f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
                f"{overlap:.4f}", f"{wval:.4f}"
            ])

    print(f"[Step3 Done] Created {output_csv}. Rows={total_bboxes}")

#############################################
# 4) 종합 실행
#############################################
def run_entire_pipeline():
    """
    하나의 함수에서 전부 순차적으로 실행:
    1) Preprocess -> frame_labeled_data.csv
    2) Extract BBoxes -> bboxes.csv
    3) Extract Collision -> final_collision.csv
    """
    # --------------------------------
    # 1) Preprocess
    # --------------------------------
    step1_preprocess(
        toggle_csv='eyepath_data3.csv',
        video_dir='videos',
        images_dir='images',
        output_csv='preprocess_output/frame_labeled_data.csv',
        sampling_fps=20
    )

    # --------------------------------
    # 2) BBox
    # --------------------------------
    step2_extract_bboxes(
        input_csv='preprocess_output/frame_labeled_data.csv',
        images_dir='images',
        yolo_weights='last.pt',
        max_boxes=30,
        output_csv='preprocess_output/bboxes.csv'
    )

    # --------------------------------
    # 3) Collision
    # --------------------------------
    step3_extract_collision(
        labeled_csv='preprocess_output/frame_labeled_data.csv',
        bbox_csv='preprocess_output/bboxes.csv',
        output_csv='preprocess_output/final_collision.csv',
        height_m=1.70,
        torso_m=0.45,
        theta_v=52.0,
        theta_h=65.0,
        alpha=5.0,
        beta=5.0
    )
    


if __name__ == "__main__":
    run_entire_pipeline()
