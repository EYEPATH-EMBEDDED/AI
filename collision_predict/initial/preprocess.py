import os
import csv
import math
import cv2

# ------------------------------------------------------------------
# 1. CSV(토글 정보)에서 (Time, Label) 읽어오기
# ------------------------------------------------------------------
def parse_time_string(t_str):
    """
    'MM:SS.xx' 또는 'HH:MM:SS.xx' 형식의 문자열을 초(float)로 변환.
    예: '00:09.96' -> 9.96, '01:14.90' -> 74.90
    """
    parts = t_str.split(':')
    if len(parts) == 2:
        # MM:SS.xx
        mm = int(parts[0])
        ss = float(parts[1])
        total_seconds = mm * 60 + ss
    elif len(parts) == 3:
        # HH:MM:SS.xx
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
      { "1.mp4": [(time_sec1, label1), (time_sec2, label2), ...],
        "2.mp4": [...],
        ...
      }
    시간 오름차순 정렬된 상태
    """
    toggle_dict = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
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
    toggle_list = [(t1, lbl1), (t2, lbl2), ...] (시간 오름차순)
    초기에 state=0 (OFF) 가정.
    - 홀수 label(1,3,5...) -> OFF->ON 혹은 ON->OFF로 전환
      (실제로는 OFF->ON 이지만, 간단히 state=1-state 로 구현)
    - 짝수 label(2,4,6...) -> ON->OFF 혹은 OFF->ON 전환
    
    => 여기서는 "홀수면 state 뒤집기, 짝수도 state 뒤집기" 로 동일하게 볼 수 있음.
       실제 의미는 '번갈아 가며 0→1, 1→0'.
    
    time_sec 시점 이전에 등장한 모든 토글을 count하여 0/1 결정.
    """
    state = 0  # OFF
    for (t, lbl) in toggle_list:
        if time_sec < t:
            break
        # 토글 발생 -> state 뒤집기
        state = 1 - state
    return state

# ------------------------------------------------------------------
# 2. 비디오를 step 간격으로 샘플링 -> 이미지 파일 생성, CSV용 행(row) 리턴
# ------------------------------------------------------------------
def process_video(video_path, toggle_list, images_dir, sampling_fps=2):
    """
    - toggle_list: 해당 비디오의 토글 시점 목록 [(time_sec, label), ...]
    - sampling_fps: 초당 몇 장을 샘플링할지 (디폴트=2 -> 0.5초 간격)
    - images_dir: 최종 이미지들이 저장될 폴더 (예: "images/")
    
    반환: [ (video_name, frame_idx, time_sec, label), ... ] 리스트
          여기서 video_name은 원본 비디오 파일명 (예: "1.mp4")
    """
    base_name = os.path.basename(video_path)        # ex) "1.mp4"
    name_only, _ = os.path.splitext(base_name)      # ex) "1"
    
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: cannot open {video_path}")
        return []
    
    # 원본 영상의 FPS와 전체 길이를 구한다
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / original_fps if original_fps > 0 else 0
    
    # step (초 단위) = 1 / sampling_fps
    step_sec = 1.0 / sampling_fps
    
    # 샘플링 결과를 담을 리스트
    results = []
    
    # 저장할 디렉토리(images/1) 만들기
    save_subdir = os.path.join(images_dir, name_only)
    os.makedirs(save_subdir, exist_ok=True)
    
    # 0초부터 duration까지 step_sec 간격으로 샘플링
    t = 0.0
    while t <= duration_sec:
        # 해당 시점의 원본 프레임 인덱스 (반올림)
        f_idx = int(round(t * original_fps))
        if f_idx >= frame_count:
            break
        
        # 비디오 캡처 위치 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        # time_sec -> ON/OFF 라벨 구하기
        label = get_onoff_label(t, toggle_list)
        
        # 이미지 파일명 ex) "1_334.jpg"
        img_filename = f"{name_only}_{f_idx}.jpg"
        img_save_path = os.path.join(save_subdir, img_filename)
        
        # 이미지 저장 (jpg)
        cv2.imwrite(img_save_path, frame)
        
        # CSV용 행 구성
        results.append((base_name, f_idx, round(t, 2), label))
        
        t += step_sec
    
    cap.release()
    return results

# ------------------------------------------------------------------
# 3. 메인 함수
# ------------------------------------------------------------------
def main():
    # (1) 사용자 설정
    # 토글 CSV 경로
    TOGGLE_CSV = 'eyepath_data3.csv'        # 토글 CSV 경로
    VIDEO_DIR = 'videos'                    # 비디오가 들어있는 폴더
    IMAGES_DIR = 'images'                   # 샘플링 이미지들을 저장할 폴더
    OUTPUT_CSV = 'frame_labeled_data.csv'   # 최종 csv 이름
    SAMPLING_FPS = 20                       # (0.05초 간격) 기본 fps -> 2 (즉, 0.5초 간격)

    # -----------------------------
    # (2) CSV 로드 -> 비디오별 토글 정보
    # -----------------------------
    toggle_dict = load_toggle_data(TOGGLE_CSV)
    # toggle_dict = {
    #   "1.mp4": [(time_sec, label), ...],
    #   "2.mp4": ...
    # }

    # -----------------------------
    # (3) 각 비디오를 순회하며 샘플링 수행
    # -----------------------------
    all_rows = []  # 최종 CSV에 쓸 (video, frame, time, label) 묶음
    
    # 비디오 목록과 개수
    video_list = list(toggle_dict.items())
    total_videos = len(video_list)
    
    # 비디오별 진행상황 표시
    for idx, (video_file, toggle_list) in enumerate(video_list, start=1):
        print(f"[INFO] Processing video {idx}/{total_videos}: {video_file}")
        
        video_path = os.path.join(VIDEO_DIR, video_file)
        # 비디오를 step 간격으로 처리
        rows = process_video(video_path, toggle_list, IMAGES_DIR, SAMPLING_FPS)
        
        # 결과 누적
        all_rows.extend(rows)
        
        # 현재 비디오에서 몇 개 프레임이 추출되었는지 출력
        print(f"       -> Sampled {len(rows)} frames from {video_file}")

    # -----------------------------
    # (4) CSV로 저장
    # -----------------------------
    # 열 순서: video, frame, time, label
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['video', 'frame', 'time', 'label'])
        for row in all_rows:
            writer.writerow(row)
    
    print(f"[INFO] Done! Created {len(all_rows)} rows in '{OUTPUT_CSV}'.")
    print(f"[INFO] Images saved under '{IMAGES_DIR}'.")


# 이 스크립트를 직접 실행했을 때만 메인 함수를 호출하도록 한다.
if __name__ == "__main__":
    main()
