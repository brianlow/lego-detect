# Adds to the existing dataset
# This is a temporary script until can rework dataset.py

import os
from pathlib import Path


dataset_name = "lego-detect-11-aruco"



dataset_folder = f"./datasets/{dataset_name}"
dataset_yaml = f"./datasets/{dataset_name}.yaml"

os.makedirs(dataset_folder, exist_ok=True)

os.system(f"cp -r ../lego-rendering/renders/lego-detect-01-250-transparent/* {dataset_folder}/")
os.system(f"cp -r ./src/images/* {dataset_folder}/train/images/")
os.system(f"cp -r ./src/labels/* {dataset_folder}/train/labels/")
os.system(f"cp -r ./datasets/dataset/* {dataset_folder}/")
os.system(f"cp ./datasets/dataset.yaml {dataset_yaml}")


os.chdir('datasets')
os.system(f'zip -r {dataset_name}.zip {dataset_name} {dataset_yaml}')
os.chdir('..')

print('')
print(f"Created dataset at {dataset_folder}")
print(f"Zipped to {dataset_name}.zip")
print('')
print("Done")
