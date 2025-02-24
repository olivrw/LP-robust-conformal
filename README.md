# Robust Conformal Prediction with Levy-Prokhorov Distribution Shift

This repository implements a conformal prediction algorithm that is robust to Levy-Prokhorov type distribution shifts.

Paper: "Conformal Prediction under Lévy-Prokhorov Distribution Shifts: Robustness to Local and Global Perturbations" 
https://arxiv.org/abs/2502.14105

## Overview

The code implements three conformal prediction approaches:
- Standard conformal prediction
- LP robust conformal prediction (our method)
- f-divergence robust conformal prediction

## Experiments

We conduct experiments on 3 classification datasets:

### ImageNet
Located in the `imgnet/` folder:
- `imgnet_exp.py`:  ImageNet experiments under data-space distribution shift 
The trained predictor we use is the torch pretrained ResNet152 model

### MNIST 
Located in the `mnist/` folder:
- `mnist_exp.py`:  MNIST experiments under data-space distribution shift
- `train_mnist.py`: Training script for a simple NN

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

