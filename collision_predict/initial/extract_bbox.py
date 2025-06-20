# extract_bboxes.py

import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO

CSV_IN = 'frame_labeled_data.csv'     # 이미지 목록+라벨이 들어있는 CSV
IMAGES_DIR = 'images'                 # 전처리된 이미지가 들어있는 상위 폴더
YOLO_WEIGHTS = 'last.pt'              # YOLO 가중치
MAX_BOXES = 30                        # 한 이미지에서 최대 몇 개의 bbox까지 저장할지
OUTPUT_CSV = 'bboxes.csv'             # YOLO 결과(좌표) 저장용 CSV


def main():
    # 2) YOLO 모델 로드
    #    (num_workers 문제 방지 위해, multiprocessing 비활성화가 권장됨)
    model = YOLO(YOLO_WEIGHTS)
    model.conf = 0.25
    model.iou = 0.45

    # 3) CSV_IN 열어서 (video, frame) 별 이미지 경로를 순회
    with open(CSV_IN, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        
        # bboxes.csv 헤더: video, frame, box_idx, x1, y1, x2, y2
        writer.writerow(['video', 'frame', 'box_idx', 'x1', 'y1', 'x2', 'y2'])

        count = 0
        for row in reader:
            video = row['video']          # ex) "1.mp4"
            frame_idx = int(row['frame']) # ex) 334

            # 이미지 경로 구성
            base_name, _ = os.path.splitext(video)  # "1"
            img_filename = f"{base_name}_{frame_idx}.jpg"
            img_path = os.path.join(IMAGES_DIR, base_name, img_filename)

            frame_bgr = cv2.imread(img_path)
            if frame_bgr is None:
                # 이미지 파일이 없으면 skip
                continue
            
            # YOLO 추론
            results = model(frame_bgr, verbose=False)
            boxes = results[0].boxes  # => [N, 4 or 6...], ultralytics 8.x

            # 최대 MAX_BOXES개만 저장
            num_b = min(len(boxes), MAX_BOXES)
            for b_idx in range(num_b):
                # ultralytics 8.x => boxes[b_idx].xyxy[0] == [x1,y1,x2,y2]
                x1, y1, x2, y2 = boxes[b_idx].xyxy[0]
                writer.writerow([video, frame_idx, b_idx, x1, y1, x2, y2])

            count += 1
            if count % 500 == 0:
                print(f"[INFO] Processed {count} frames...")

    print(f"[DONE] Created {OUTPUT_CSV} with bounding box info.")

import csv
import cv2
import torch
from ultralytics import YOLO

def extract_bboxes_to_csv(input_csv, output_csv, images_dir, yolo_weights, max_boxes=30):
    """
    input_csv  : 'frame_labeled_data.csv' 등, (video, frame, ...) 정보가 들어있는 CSV
    output_csv : 이 스크립트가 생성할 bboxes.csv
    images_dir : 'images/' 상위 폴더
    yolo_weights : YOLO 가중치 파일 (ex: 'last.pt')
    max_boxes : 한 이미지에서 최대 몇 개 bbox를 저장할지
    """
    # 1) YOLO 로드
    model = YOLO(yolo_weights)
    model.conf = 0.25
    model.iou = 0.45

    with open(input_csv, 'r', encoding='utf-8') as f_in, \
         open(output_csv, 'w', newline='', encoding='utf-8') as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        
        # CSV 헤더: [video, frame, box_idx, x1, y1, x2, y2]
        writer.writerow(['video', 'frame', 'box_idx', 'x1', 'y1', 'x2', 'y2'])
        
        count = 0
        for row in reader:
            video = row['video']
            frame_idx = int(row['frame'])
            
            # 이미지 경로 구성
            base_name = video.rsplit('.', 1)[0]  # ex) "1.mp4" -> "1"
            img_filename = f"{base_name}_{frame_idx}.jpg"
            img_path = f"{images_dir}/{base_name}/{img_filename}"

            frame_bgr = cv2.imread(img_path)
            if frame_bgr is None:
                continue

            # 2) YOLO 추론
            results = model(frame_bgr, verbose=False)
            boxes = results[0].boxes  # ultralytics 8.x => N개의 Box
            
            # 3) 최대 max_boxes개만 취급
            num_b = min(len(boxes), max_boxes)
            for b_idx in range(num_b):
                # (x1,y1,x2,y2) = GPU 텐서일 수도 있음
                x1, y1, x2, y2 = boxes[b_idx].xyxy[0]
                
                # 4) CPU float 변환
                x1 = float(x1.detach().cpu().numpy())
                y1 = float(y1.detach().cpu().numpy())
                x2 = float(x2.detach().cpu().numpy())
                y2 = float(y2.detach().cpu().numpy())

                # CSV 기록
                writer.writerow([video, frame_idx, b_idx, x1, y1, x2, y2])

            count += 1
            if count % 500 == 0:
                print(f"[INFO] Processed {count} frames...")

    print(f"[DONE] {count} frames processed. Bboxes saved to {output_csv}.")
    
    
    
    CSV_IN = 'frame_labeled_data.csv'     # 이미지 목록+라벨이 들어있는 CSV
    IMAGES_DIR = 'images'                 # 전처리된 이미지가 들어있는 상위 폴더
    YOLO_WEIGHTS = 'last.pt'              # YOLO 가중치
    MAX_BOXES = 30                        # 한 이미지에서 최대 몇 개의 bbox까지 저장할지
    OUTPUT_CSV = 'bboxes.csv'             # YOLO 결과(좌표) 저장용 CSV

if __name__ == "__main__":
    extract_bboxes_to_csv(CSV_IN,OUTPUT_CSV,IMAGES_DIR,YOLO_WEIGHTS,MAX_BOXES)