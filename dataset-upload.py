import os

dataset_name = "lego-detect-12-4k-new-rendering"

os.chdir('datasets')
os.system(f'zip -r {dataset_name}.zip {dataset_name}.yaml {dataset_name}')
os.chdir('..')

# Save to AWS
os.system(f'aws s3 cp datasets/{dataset_name}.zip s3://brian-lego-public/lego-detect/')
