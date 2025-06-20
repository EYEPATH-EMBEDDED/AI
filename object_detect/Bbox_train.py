#from ..utils import PARAMS
from ultralytics import YOLO


model = YOLO("yolo11x.pt")  # load an official model


# 데이터셋 및 학습 파라미터 설정

# First Params
# model.train(
#     data='/mnt/dataset/dataset.yaml',  # dataset.yaml 파일 경로
#     epochs=3,                         # 학습 에포크 수
#     imgsz=1920,                         # 입력 이미지 크기
#     batch=16,                          # 배치 크기
#     optimizer='Adam',                  # 최적화 알고리즘 (SGD 또는 Adam)
#     device='0',                        # GPU ID, '0'은 첫 번째 GPU를 의미
#     lr0=0.001,                         # 초기 학습률
#     workers=4                          # 데이터 로딩에 사용할 워커 수
# )

# # Second Params
model.train(
    data='/mnt/dataset_bbox/dataset.yaml',  # dataset.yaml 파일 경로
    epochs=20,                         # 학습 에포크 수
    imgsz= 960,                         # 입력 이미지 크기
    batch=0.80,                          # 배치 크기
    warmup_epochs = 2,
    save_period = 5,
    optimizer='Adam',                  # 최적화 알고리즘 (SGD 또는 Adam)
    device='0',                        # GPU ID, '0'은 첫 번째 GPU를 의미
    lr0=0.001,                         # 초기 학습률
    workers=8,                          # 데이터 로딩에 사용할 워커 수
    single_cls = True
)


# third Params
# model.train(
#     data='/mnt/dataset/dataset.yaml',  # dataset.yaml 파일 경로
#     epochs=20,                         # 학습 에포크 수
#     imgsz= [1920,1080],                         # 입력 이미지 크기
#     batch=0.70,                          # 배치 크기
#     warmup_epochs = 2,
#     save_period = 4,
#     optimizer='Adam',                  # 최적화 알고리즘 (SGD 또는 Adam)
#     single_cls = True,
#     device='0',                        # GPU ID, '0'은 첫 번째 GPU를 의미
#     lr0=0.001,                         # 초기 학습률
#     workers=8                          # 데이터 로딩에 사용할 워커 수
# )