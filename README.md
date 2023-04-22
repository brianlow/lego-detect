Training a detection model for identifying lego pieces on a conveyor

## Instructions

```
asdf install
python -m pip install -r requirements.txt

# need to manually download the dataset for the time being (see setup.py)
python setup.py

# see results in ./runs, if good copy path to weights/best.pt
python train.py

# update predict.py with best weights, put images in ./samples, run script and see results in ./tmp
python predict.py
```


## Results

Training on 1800k rendered images -> only good at predicting rendered images
Training on 10 real photos with 10 blocks each -> pretty good!
