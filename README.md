# Robust Conformal Prediction with Levy-Prokhorov Distribution Shift

This repository implements a conformal prediction algorithm that is robust to Levy-Prokhorov type distribution shifts.

Paper: "Conformal Prediction under Lévy-Prokhorov Distribution Shifts: Robustness to Local and Global Perturbations" 
https://arxiv.org/abs/2502.14105

To faithfully reproduce results for comparative experiements, we borrowed implementations from the referenced methods' 
repositories:
- f-divergence robust conformal prediction: Code https://github.com/suyashgupta28/robust-validation.git, Paper https://www.tandfonline.com/doi/full/10.1080/01621459.2023.2298037
- Fine-grain robust conformal prediction: Code https://github.com/zhimeir/finegrained-conformal-paper.git, Paper https://proceedings.mlr.press/v235/ai24a.html
- Random smoothing conformal prediction: Code https://github.com/Asafgendler/RSCP.git Paper https://openreview.net/pdf?id=9L1BsI4wP1H

## Overview

This repository, besides the referenced methods for benchmarking, implements
- Standard conformal prediction
- LP robust conformal prediction (our method)
- (eps, rho) estimation algorithm
- Weighted conformal prediction (https://papers.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html)

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


