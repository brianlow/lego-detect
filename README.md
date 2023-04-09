Training and sample webpage for recognizing brick


## Setup

```
install miniconda but don't use the conda envs

pip install torch torchvision

# https://pytorch-lightning.readthedocs.io/en/stable/starter/installation.html
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
python -m pip install -U lightning

https://lightning-flash.readthedocs.io/en/stable/installation.html
pip install lightning-flash
pip install 'lightning-flash[image]'
pip install 'lightning-flash[serve]'
pip install flask
```

What didn't work
* asdf
* conda envs

## Training

* Dataset
  * 477 lego parts
  * https://mostwiedzy.pl/en/open-research-data/lego-bricks-for-training-classification-network,618104539639776-0
* Data prep:
  * python ./data-prep/square.py
  * python ./data-prep/extract-subset.py
* Current best classification model
  * Yolo v8, xl size
  * trained on 50 classes, ~96% accurate
  * Paperspace.com, $8 plan, NVidia A4000 free gpu
  * Yolo notebook: https://console.paperspace.com/brian22/notebook/r5qgvpedkjbw78v


## Running

```
python serve.py
./ngrok http 8000
```


## Other

Use ./download_bricks.sh to pull icons for each brick type and save to ./static/images
