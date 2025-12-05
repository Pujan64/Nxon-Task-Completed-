import cv2
from ultralytics import YOLO


VIDEO_IN = "data/videos/video.mp4"     # input video
VIDEO_OUT = "outputs/line_cross_out.mp4"   # annotated video
MODEL_NAME = "yolov8n.pt"    
CONF_THRESH = 0.4             

LINE_Y = 300                  


model = YOLO(MODEL_NAME)

# Open video
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    print("ERROR: Cannot open input video!")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (width, height))


previous_y = {}

print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    cv2.line(frame, (0, LINE_Y), (width, LINE_Y), (0, 0, 255), 3)

  
    results = model.predict(frame, conf=CONF_THRESH, verbose=False)
    detections = results[0].boxes

    for det in detections:
        cls = int(det.cls[0])
        conf = float(det.conf[0])
        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()

    
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        obj_id = len(previous_y) + cls  

        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 5, (255,0,0), -1)
        cv2.putText(frame, f"ID {obj_id}", (cx-20, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

       
        if obj_id in previous_y:
            old_y = previous_y[obj_id]

            # If object moved from above LINE_Y to below LINE_Y
            if old_y < LINE_Y and cy > LINE_Y:
                cv2.putText(frame, "CROSSED ↓", (cx+10, cy+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)

            # If object moved from below to above
            elif old_y > LINE_Y and cy < LINE_Y:
                cv2.putText(frame, "CROSSED ↑", (cx+10, cy+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 3)

        
        previous_y[obj_id] = cy

    
    out.write(frame)

cap.release()
out.release()

print(f"✅ DONE! Output saved to: {VIDEO_OUT}")
