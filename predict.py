import os
import sys
sys.path.append('../ultralytics')
from ultralytics import YOLO
import cv2
from PIL import Image

model = YOLO('detect-06-10-classes-2k-images-paperspace2.pt')

# loop through files in the /samples folder
for filename in os.listdir("./samples"):
  img = Image.open(os.path.join("./samples", filename))
  results = model(img)
  cv2.imwrite(os.path.join("tmp/", filename), results[0].plot())
