# Prepares this dataset for training:
#     Zawora, K., Zaraziński, S., Śledź, B., Łobacz, B., & Boiński, T. M. (2021).
#     Tagged images with LEGO bricks [Data set]. Gdańsk University of Technology.
#     https://doi.org/10.34808/2dbx-6a16
#
# With the these changes:
#     - YOLO format (no train/val split though)
#     - resize down to 1024x1024 images
#     - only a portion of the real photos (images with <= 10 objects in them)
#     - no renders
#
# Before running this script:
#     - download from https://mostwiedzy.pl/en/open-research-data/tagged-images-with-lego-bricks,209111650250426-0
#     - unzip to ./dataset/final_dataset_lego_detection

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import zipfile
from PIL import Image

def convert_voc_to_yolo(voc_root, yolo_output):
    # Create output directories
    yolo_output = Path(yolo_output)
    (yolo_output / 'images').mkdir(parents=True, exist_ok=True)
    (yolo_output / 'labels').mkdir(parents=True, exist_ok=True)

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

        # Save YOLO annotation file
        yolo_label_path = yolo_output / 'labels' / f"{img_name}.txt"
        if not os.path.exists(yolo_label_path):
            with open(yolo_label_path, 'w') as f:
                f.write('\n'.join(yolo_annotation))

        # Move the image to the YOLO output directory
        yolo_image_path = yolo_output / 'images' / img_path.name
        if not os.path.exists(yolo_image_path):
            img_path.rename(yolo_image_path)

def resize_images(directory, max_size=1024):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"Checking {file}... ", end="")
                filepath = os.path.join(root, file)
                img = Image.open(filepath)
                if max(img.size) > max_size:
                    print(f"RESIZE from {img.size}")
                    img.thumbnail((max_size, max_size), Image.LANCZOS)
                    img.save(filepath)
                else:
                    print("")

# Start of script

data_dir = Path()
dataset_name = "zawora-dataset"
output_dir = Path(f'./datasets/{dataset_name}')

input_dirs = [
    # "./datasets/final_dataset_lego_detection/photos/1",
    # "./datasets/final_dataset_lego_detection/photos/2",
    # "./datasets/final_dataset_lego_detection/photos/3",
    # "./datasets/final_dataset_lego_detection/photos/4",
    # "./datasets/final_dataset_lego_detection/photos/5",
    # "./datasets/final_dataset_lego_detection/photos/6",
    # "./datasets/final_dataset_lego_detection/photos/7",
    # "./datasets/final_dataset_lego_detection/photos/8",
    # "./datasets/final_dataset_lego_detection/photos/9",
    # "./datasets/final_dataset_lego_detection/photos/10",
    "./datasets/final_dataset_lego_detection/photos/11",
    "./datasets/final_dataset_lego_detection/photos/12",
    "./datasets/final_dataset_lego_detection/photos/13",
    "./datasets/final_dataset_lego_detection/photos/14",
    "./datasets/final_dataset_lego_detection/photos/15",
    "./datasets/final_dataset_lego_detection/photos/16",
    "./datasets/final_dataset_lego_detection/photos/17",
    "./datasets/final_dataset_lego_detection/photos/18",
    "./datasets/final_dataset_lego_detection/photos/20",
]
for input_dir in input_dirs:
    convert_voc_to_yolo(input_dir, output_dir)

# Resize the images to 1024x1024
# I found detection rate was just as good but training time decreased
# from 3:30 minutes down to a few seconds per epoch
resize_images(output_dir)

os.chdir('datasets')
os.system(f'zip -r {dataset_name}.zip {dataset_name}')
os.chdir('..')

# Save to AWS
os.system(f'aws s3 cp datasets/{dataset_name}.zip s3://brian-lego-public/lego-detect/')
