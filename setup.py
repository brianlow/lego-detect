import os
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import zipfile

def convert_voc_to_yolo(voc_root, yolo_output, val_split):
    # Create output directories
    yolo_output = Path(yolo_output)
    (yolo_output / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (yolo_output / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
    (yolo_output / 'val' / 'images').mkdir(parents=True, exist_ok=True)
    (yolo_output / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

    annotations_dir = Path(voc_root)
    images_dir = Path(voc_root)

    for xml_file in annotations_dir.glob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        image_filename = root.find('filename').text
        img_path = images_dir / image_filename
        img_name = img_path.stem

        if not os.path.exists(img_path):
            print(f"#{xml_file} points to a non-existing image: {img_path}")
            continue

        yolo_annotation = []
        for obj in root.findall('object'):
            class_idx = 0 # single class

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)

            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            yolo_annotation.append(f"{class_idx} {x_center} {y_center} {box_width} {box_height}")

        train_or_val = 'val' if random.random() < val_split else 'train'

        # Save YOLO annotation file
        yolo_label_path = yolo_output / train_or_val / 'labels' / f"{img_name}.txt"
        if not os.path.exists(yolo_label_path):
            with open(yolo_label_path, 'w') as f:
                f.write('\n'.join(yolo_annotation))

        # Copy the image to the YOLO output directory
        yolo_image_path = yolo_output / train_or_val / 'images' / img_path.name
        if not os.path.exists(yolo_image_path):
            img_path.rename(yolo_image_path)

# Start of script

# 80% train, 20% val
val_split = 0.2

# Make the train/val split repeatable for easier debugging
random.seed(1)

# Determine where the data will be stored. Either
#  ./data   - when running locally
#  /storage - when running on Paperspace
is_paperspace = os.environ.get('PAPERSPACE_CLUSTER_ID') is not None
data_dir = Path('/storage' if is_paperspace else './data')
if not os.path.exists(data_dir):
  os.makedirs(data_dir)

# Download the dataset, have to do this manually right now!
# from https://mostwiedzy.pl/en/open-research-data/tagged-images-with-lego-bricks,209111650250426-0
# download https://mostwiedzy.pl/en/open-research-data/tagged-images-with-lego-bricks,209111650250426-0/download to a zip

# Unzip the dataset
with zipfile.ZipFile(data_dir / 'final_dataset_lego_detection.zip', 'r') as zip_ref:
    zip_ref.extractall(data_dir)

# Convert the dataset to YOLO format
voc_paths = [
    os.path.join(data_dir, "final_dataset_lego_detection/photos/1"),
    os.path.join(data_dir, "final_dataset_lego_detection/photos/2"),
    os.path.join(data_dir, "final_dataset_lego_detection/photos/3"),
    os.path.join(data_dir, "final_dataset_lego_detection/photos/4"),
    os.path.join(data_dir, "final_dataset_lego_detection/photos/5"),
    os.path.join(data_dir, "final_dataset_lego_detection/photos/6"),
    os.path.join(data_dir, "final_dataset_lego_detection/photos/7"),
    os.path.join(data_dir, "final_dataset_lego_detection/photos/8"),
    os.path.join(data_dir, "final_dataset_lego_detection/photos/9"),
    os.path.join(data_dir, "final_dataset_lego_detection/photos/10"),
]
for voc_path in voc_paths:
    yolo_path = os.path.join(data_dir, "dataset")
    convert_voc_to_yolo(voc_path, yolo_path, val_split)

# Output a dataset yaml file
with open(data_dir / 'dataset.yaml', 'w') as f:
  f.write(f"# Path must be an absolute path unless it is Ultralytics standard location\n")
  f.write(f"path: {os.path.abspath(data_dir / 'dataset')}\n")
  f.write(f"train: train/images\n")
  f.write(f"val: val/images\n")
  f.write(f"\n")
  f.write(f"names:\n")
  f.write(f"  0: lego\n")
