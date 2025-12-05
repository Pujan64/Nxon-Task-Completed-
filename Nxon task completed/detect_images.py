from ultralytics import YOLO
import os
import glob

# Load pretrained YOLO model
model = YOLO('yolov8n.pt')  # nano version for speed

# Create output directory
os.makedirs('outputs', exist_ok=True)

# Get image files (adjust path as needed)
image_files = glob.glob('data/images/*.jpg') + glob.glob('data/images/*.png') + glob.glob('data/images/*.jpeg') + glob.glob('data/images/*.webp')

if not image_files:
    print("No images found in 'data/images/' directory")
    exit()

# Limit to 10-20 images
image_files = image_files[:20]

print(f"Processing {len(image_files)} images...")

# Process each image
for img_path in image_files:
    # Run inference
    results = model(img_path)
    
    # Save image with bounding boxes
    img_name = os.path.basename(img_path)
    output_path = f'outputs/{img_name}'
    results[0].save(output_path)
    
    # Count objects
    detections = results[0].boxes
    object_count = len(detections) if detections is not None else 0
    
    print(f"{img_name}: {object_count} objects detected")

print("Processing complete. Check 'outputs/' directory for results.")