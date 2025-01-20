# Robust Conformal Prediction with Levy-Prokhorov Distribution Shift

This repository contains the implementation of our robust conformal prediction algorithm using Levy-Prokhorov distance.

## Overview

The code implements three conformal prediction approaches:
- Standard conformal prediction
- LP-robust conformal prediction (our method)
- f-divergence robust conformal prediction

## Experiments

We conduct experiments on two datasets:

### ImageNet
Located in the `imgnet/` folder:
- `imgnet_exp.py`:  ImageNet experiments under data-space distribution shift 
- `imgnet_score_exp.py`: ImageNet experiments under score-space distribution shift
The trained predictor we use is the torch pretrained ResNet152 model

### MNIST 
Located in the `mnist/` folder:
- `mnist_exp.py`:  MNIST experiments under data-space distribution shift
- `mnist_score_exp.py`: MNIST experiments under score-space distribution shift
The trained predictor we use is a simple ResNet trained using:
- `train_mnist.py`: ImageNet experiments under score-space distribution shift

### iWildCam  
Located in the `wilds/` folder:
- `wilds_exp.py`:  iWildCam experiments under natural data-space distribution shift
The trained predictor we use is a simple ResNet trained using a trained 'best_model.pth' using IRM scheme:
https://worksheets.codalab.org/bundles/0x466af6d839d64982bbe4271098b358c4

## Requirements

- PyTorch 
- torchvision
- numpy
- pandas
- matplotlib

