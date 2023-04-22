import os
import comet_ml
import sys
from ultralytics import YOLO
from pathlib import Path

#
# Yolo v8 training script
#

# Paperspace
# !pip uninstall --yes protobuf tensorboard tensorflow
# !pip install ultralytics protobuf onnx comet_ml

experiment_name = "detect-05-sample-real"

comet_ml.init(project_name=experiment_name)

# Determine where the data will be stored. Either
#  ./data   - when running locally
#  /storage - when running on Paperspace
is_paperspace = os.environ.get('PAPERSPACE_CLUSTER_ID') is not None
data_dir = Path('/storage' if is_paperspace else './data')
if not os.path.exists(data_dir):
  os.makedirs(data_dir)

# Load a model
model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data=f"./data/dataset.yaml", name=experiment_name, epochs=300, hsv_h=0.25, shear=0)
