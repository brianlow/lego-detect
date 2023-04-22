import os
import sys
sys.path.append('../ultralytics')
from ultralytics import YOLO
import cv2
from PIL import Image

# model = YOLO('detect-06-10-classes-2k-images-paperspace2.pt')
model = YOLO('runs/detect/detect-05-sample-real3/weights/best.pt')

# loop through files in the /samples folder
for filename in os.listdir("./samples"):
    img = Image.open(os.path.join("./samples", filename))
    results = model(img)

    boxes = list(map(lambda box: { "x": box.xyxy[0][0].item(), "y": box.xyxy[0][1].item(), "x2": box.xyxy[0][2].item(), "y2": box.xyxy[0][3].item()}, results[0].cpu().boxes))
    print("--------")
    print(boxes)

    cv2.imwrite(os.path.join("tmp/", filename), results[0].plot())
