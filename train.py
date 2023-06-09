import os
import comet_ml
import sys
import zipfile
from ultralytics import YOLO
from pathlib import Path

#
# Yolo v8 training script
#

# Paperspace setup
# export COMET_API_KEY=<your key if you want to report to comet.com>
# pip install -r requirements.txt!

experiment_name = "lego-detect-13-7k-more-negatives"
dataset_name = "lego-detect-13-7k-more-negatives"

comet_ml.init(project_name=experiment_name)

# Determine where the data will be stored. Either
#  ./dataset  - when running locally
#  /storage   - when running on Paperspace
is_paperspace = os.environ.get('PAPERSPACE_CLUSTER_ID') is not None
data_dir = Path('/storage' if is_paperspace else './datasets')
data_dir = str(Path(data_dir).resolve())

# Download dataset if the data_dir doesn't exist
dataset_dir = os.path.join(data_dir, dataset_name)
if not os.path.exists(dataset_dir):
    dataset_zip = f"{dataset_name}.zip"
    dataset_zip_path = os.path.join(data_dir, dataset_zip)
    if not os.path.exists(dataset_zip_path):
        dataset_url = f"https://brian-lego-public.s3.us-west-1.amazonaws.com/lego-detect/{dataset_zip}"
        print(f"Downloading dataset {dataset_url}...")
        os.system(f"wget {dataset_url} -O {dataset_zip_path}")
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

# Load a model
model = YOLO('yolov8n.pt')  # yolo8s and yolo8m achieved similar accuracy

# Train the model
model.train(data=f"{data_dir}/{dataset_name}.yaml", name=experiment_name, epochs=300, batch=-1, hsv_h=0.25, shear=0)
