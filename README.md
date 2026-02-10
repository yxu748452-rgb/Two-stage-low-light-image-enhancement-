# A two-stage low-light image enhancement network combining spatial and frequency information

## Introduction
In this project, we use Python 3.10.12, Pytorch 2.0.0 and one NVIDIA RTX 4090 GPU.

##Testing

If you want to evaluate using our provided pretrained model, please download the [LOL datsets](https://daooshee.github.io/BMVC2018website/ "LOL datsets").

The pretrained model is in the ./experiments/24.32.
Check the model and image pathes in test.py, and then run:
```javascript
python test.py
```

##Training
To train the model, you need to prepare our training dataset.

Check the dataset path in train.py, and then run:
```javascript
python train.py
```

