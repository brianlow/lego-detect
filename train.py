#
# Yolo v8 training script
#

# Paperspace
# !pip uninstall --yes protobuf tensorboard tensorflow
# !pip install ultralytics protobuf onnx comet_ml

# import comet_ml
# comet_ml.init(project_name='yolov8')

import sys
sys.path.append('../ultralytics')
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data=f"../lego-rendering/renders/dataset.yaml", name="detect-03-50images-hue-shift", epochs=30, close_mosaic=7, imgsz=244, hsv_h=1.0, shear=0)
