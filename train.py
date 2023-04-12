#
# Yolo v8 training script
#

# Paperspace
# !pip uninstall --yes protobuf tensorboard tensorflow
# !pip install ultralytics protobuf onnx comet_ml

import comet_ml
comet_ml.init(project_name='detect-04-10-classes-2k-images')


import sys
sys.path.append('../ultralytics')
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data=f"../lego-rendering/renders/dataset.yaml", name="detect-04-10-classes-2k-images", epochs=300, close_mosaic=10, imgsz=244, hsv_h=0.25, shear=0)
