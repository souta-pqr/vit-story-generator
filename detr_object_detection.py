import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from transformers import DetrFeatureExtractor, DetrForObjectDetection

# DETRモデルとフィーチャーエクストラクターのロード
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# デバイスの設定（GPUがある場合はGPUを使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# カメラの初期化
cap = cv2.VideoCapture(0)  # 0は通常、最初のカメラを指します

# 画像の前処理関数
def preprocess_image(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = F.to_tensor(image)
    return image.unsqueeze(0).to(device)

# 物体検出関数
def detect_objects(image, threshold=0.7):
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # 予測結果の後処理
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    # 検出結果のフィルタリングと変換
    bboxes_scaled = outputs.pred_boxes[0, keep]
    probas = probas[keep]
    
    return bboxes_scaled, probas

# 検出結果の描画関数
def draw_detections(image, boxes, labels, scores):
    for box, label, score in zip(boxes, labels, scores):
        box = box.cpu().numpy()
        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
        x, y, w, h = int(x * image.shape[1]), int(y * image.shape[0]), int(w * image.shape[1]), int(h * image.shape[0])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

while True:
    # フレームの取得
    ret, frame = cap.read()
    if not ret:
        break

    # 物体検出の実行
    image_tensor = preprocess_image(frame)
    boxes, probas = detect_objects(image_tensor)

    # 検出結果の解釈
    labels = [model.config.id2label[i.item()] for i in probas.max(-1).indices]
    scores = probas.max(-1).values

    # 結果の描画
    draw_detections(frame, boxes, labels, scores)

    # 結果の表示
    cv2.imshow('DETR Object Detection', frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break