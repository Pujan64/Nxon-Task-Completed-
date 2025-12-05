# size_estimation.py
import os
from ultralytics import YOLO
import cv2
import pandas as pd
from utils import centroid_from_box, xyxy_to_xywh, save_csv

MODEL_NAME = "yolov8n.pt"
IMAGE_DIR = "data/images"
OUT_CSV = "outputs/size_estimation.csv"
CONF_THRESH = 0.25

os.makedirs("outputs", exist_ok=True)
model = YOLO(MODEL_NAME)

rows = []
for fname in sorted(os.listdir(IMAGE_DIR)):
    if not fname.lower().endswith(('.jpg','.jpeg','.png')):
        continue
    img_path = os.path.join(IMAGE_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    res = model.predict(img, conf=CONF_THRESH, verbose=False)
    r = res[0]
    for b in r.boxes:
        xyxy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy[0], 'cpu') else b.xyxy[0].numpy()
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        x1,y1,x2,y2 = map(float, xyxy)
        area = (x2-x1)*(y2-y1)
        norm_area = area / (w*h)  
        if norm_area < 0.005:
            size = "small"
        elif norm_area < 0.02:
            size = "medium"
        else:
            size = "large"
        rows.append({
            "image": fname,
            "class_id": cls,
            "confidence": conf,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "area": area,
            "norm_area": norm_area,
            "size_label": size
        })

# save CSV
save_csv(rows, OUT_CSV)
