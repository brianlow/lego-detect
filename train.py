import os
import comet_ml
import sys
from ultralytics import YOLO
from pathlib import Path

#
# Yolo v8 training script
#

# Paperspace setup
# export COMET_API_KEY=<your key if you want to report to comet.com>
# pip install -r requirements.txt!

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
model = YOLO('yolov8n.pt')  # yolo8s and yolo8m achieved similar accuracy

# Train the model
model.train(data=f"./data/dataset.yaml", name=experiment_name, epochs=300, hsv_h=0.25, shear=0)
