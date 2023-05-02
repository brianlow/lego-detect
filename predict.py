import os
import sys
from ultralytics import YOLO
import cv2
from PIL import Image

os.makedirs("tmp", exist_ok=True)

# model = YOLO('detect-06-10-classes-2k-images-paperspace2.pt')
# model = YOLO('runs/detect/detect-05-sample-real3/weights/best.pt')
# model = YOLO('detect-07-4k-real-and-renders.pt')
# model = YOLO('detect-08-4k-real-and-renders-small.pt')
model = YOLO('detect-10-4k-real-and-renders-nano-1024-image-size2.pt')

# loop through files in the /samples folder
for filename in os.listdir("./samples"):
    print()
    print(f"-------- {filename}")

    img = Image.open(os.path.join("./samples", filename))
    results = model(img)

    boxes = list(map(lambda box: { "x": box.xyxy[0][0].item(), "y": box.xyxy[0][1].item(), "x2": box.xyxy[0][2].item(), "y2": box.xyxy[0][3].item()}, results[0].cpu().boxes))
    print(boxes)

    cv2.imwrite(os.path.join("tmp/", filename), results[0].plot())
