import sys
sys.path.append('../ultralytics')
from ultralytics import YOLO
import cv2
from PIL import Image

model = YOLO('runs/detect/detect-04-10-classes-2k-images2/weights/best.pt')

# Predict with the model (grid of lego)
img = Image.open('/Users/brian/Downloads/lego-datasets/lego_grid.jpeg')
results = model(img)
cv2.imwrite("tmp/lego_grid_prediction.png", results[0].plot())

# Predict with the model (one lego)
img = Image.open('../lego-rendering/renders/dataset/train/images/60592_42.png')
results = model(img)
cv2.imwrite("tmp/3001_2_prediction.png", results[0].plot())
