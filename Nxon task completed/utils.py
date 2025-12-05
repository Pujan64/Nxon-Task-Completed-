# utils.py
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2
    cy = y1 + h/2
    return [int(x1), int(y1), int(w), int(h), int(cx), int(cy)]

def draw_bbox(img, box, cls_name='obj', score=0.0, color=(0,255,0), thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    label = f"{cls_name} {score:.2f}"
    t_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 4, y1), color, -1)
    cv2.putText(img, label, (x1+2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

def centroid_from_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2)/2.0, (y1 + y2)/2.0)

def save_csv(rows: List[Dict], out_path: str):
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved CSV: {out_path}")
