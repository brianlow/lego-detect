#
# Yolo v8 training script
#

# Paperspace
# !pip uninstall --yes protobuf tensorboard tensorflow
# !pip install ultralytics protobuf onnx comet_ml

experiment_name = "detect-05-sample-real"

import comet_ml
comet_ml.init(project_name=experiment_name)

import sys
# sys.path.append('../ultralytics')
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data=f"./data/dataset.yml", name=experiment_name, epochs=300, hsv_h=0.25, shear=0)
