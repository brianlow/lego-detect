# Create a dataset for training a YOLO detection model
#
# Before running this script:
#    - Get the zawora dataset
#      - Download from https://brian-lego-public.s3.us-west-1.amazonaws.com/lego-detect/zawora-dataset.zip
#      - Unzip to ./datasets/zawora-dataset
#    - Generate renders
#      - Clone the brianlow/lego-renders repo on commit `c05e053`
#      - `./run.sh render-detect-dataset.py`
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import shutil

dataset_name = "lego-detect-12-4k-new-rendering"

dataset_path = os.path.join("./datasets", dataset_name)
dataset_yaml = f"datasets/{dataset_name}.yaml"

# Dataset of real photos
zawora_path = "datasets/zawora-dataset"

# Dataset of renders from brianlow/lego-rendering
renders_path = "../lego-rendering/renders/lego-detect-12-4k-new-rendering"

# Dataset of background images with no lego
negatives_path = "./src"

# Make the train/val split repeatable for easier debugging
random.seed(1)

def copy_and_split_into_train_and_val(source_path, dest_path, val_split):
    os.makedirs(dest_path, exist_ok=True)
    os.makedirs(os.path.join(dest_path, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_path, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(dest_path, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_path, "val", "labels"), exist_ok=True)

    # Add zawora dataset of real photos
    val_split = 0.2
    for image in Path(os.path.join(source_path, "images")).glob("*"):
        if image.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image_basename = os.path.basename(image)
            label_basename = f"{image.stem}.txt"
            label = os.path.join(source_path, "labels", label_basename)

            if not os.path.exists(image):
                raise Exception(f"image doesn't exist {image}")

            if not os.path.exists(label):
                raise Exception(f"label doesn't exist {label}")

            train_or_val = 'val' if random.random() < val_split else 'train'

            image_dest = os.path.join(dest_path, train_or_val, "images", image_basename)
            label_dest = os.path.join(dest_path, train_or_val, "labels", label_basename)

            print(f"Copying {image} to {image_dest}")
            shutil.copy(image, image_dest)

            print(f"Copying {label} to {label_dest}")
            shutil.copy(label, label_dest)


# Add files from the various datasets
# For datasets with real photo, reserve a larger portion for the validation set
copy_and_split_into_train_and_val(zawora_path, dataset_path, val_split = 0.4)
copy_and_split_into_train_and_val(renders_path, dataset_path, val_split = 0.1)
copy_and_split_into_train_and_val(negatives_path, dataset_path, val_split = 0.4)


# Output a dataset yaml file
with open(dataset_yaml, 'w') as f:
  f.write(f"# Lego Detection Dataset\n")
  f.write(f"# \n")
  f.write(f"# Includes:\n")
  f.write(f"#     - roughly 2k real photos with 1 to 20 parts per picture, from:\n")
  f.write(f"#       https://mostwiedzy.pl/en/open-research-data/tagged-images-with-lego-bricks,209111650250426-0\n")
  f.write(f"#     - roughly 4.5k rendered parts of the 1k most common parts, 1 part per photo from:\n")
  f.write(f"#       brianlow/lego-rendering repo\n")
  f.write(f"#     - manual photos of backgrounds without lego\n")
  f.write(f"#\n")
  f.write(f"# Path must be an absolute path unless it is Ultralytics standard location\n")
  f.write(f"path: {os.path.abspath(dataset_path)}\n")
  f.write(f"train: train/images\n")
  f.write(f"val: val/images\n")
  f.write(f"\n")
  f.write(f"names:\n")
  f.write(f"  0: lego\n")
