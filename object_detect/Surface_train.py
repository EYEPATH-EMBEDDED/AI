from ultralytics import YOLO


#model = YOLO("yolo11n-seg.pt")  # load an official model
#model = YOLO("yolo11s-seg.pt")  # load an official model
#model = YOLO("yolo11m-seg.pt")  # load an official model
#model = YOLO("yolo11l-seg.pt")  # load an official model
model = YOLO("yolo11x-seg.pt")  # load an official model


# # Second Params
# model.train(
#     data='/mnt/dataset_surface/dataset.yaml',  # dataset.yaml 파일 경로
#     epochs=20,                         # 학습 에포크 수
#     imgsz= 1280,                         # 입력 이미지 크기
#     batch=0.80,                          # 배치 크기
#     warmup_epochs = 3,
#     save_period = 4,
#     mask_ratio=2,
#     auto_augment="autoaugment",
#     cos_lr=True,
#     overlap_mask=False,
#     weight_decay=0.0001,
#     close_mosaic=5,
#     momentum=0.95,
#     optimizer='Adam',                  # 최적화 알고리즘 (SGD 또는 Adam)
#     device='0',                        # GPU ID, '0'은 첫 번째 GPU를 의미
#     lr0=0.0005,                         # 초기 학습률
#     workers=8                          # 데이터 로딩에 사용할 워커 수
# )

model.train(
    data='/mnt/dataset_surface/dataset.yaml',  # dataset.yaml 파일 경로
    epochs=40,                         # 학습 에포크 수
    imgsz= 640,                         # 입력 이미지 크기
    batch=0.80,                          # 배치 크기
    warmup_epochs = 3,
    save_period = 4,
    mask_ratio=2,
    auto_augment="autoaugment",
    cos_lr=True,
    overlap_mask=False,
    weight_decay=0.0002,
    #close_mosaic=5,
    momentum=0.95,
    optimizer='Adam',                  # 최적화 알고리즘 (SGD 또는 Adam)
    device='0',                        # GPU ID, '0'은 첫 번째 GPU를 의미
    lr0=0.0006,                         # 초기 학습률
    workers=8                          # 데이터 로딩에 사용할 워커 수
)