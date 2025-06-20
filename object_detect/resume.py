from ultralytics import YOLO

# 이미 학습을 마친 모델(또는 중간 체크포인트)을 로드
model = YOLO("runs/detect/train3/weights/last.pt")  

# train() 메서드에 resume=True 옵션을 추가
results = model.train(
    resume=True, 
    epochs=20, 
    device="0"
    # 필요한 다른 파라미터들(data, batch, imgsz 등)을 그대로 넣으면 됨
)
