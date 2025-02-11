import PIL
import argparse
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet152, ResNet152_Weights
from src.utils import *
from src.lp_robust_cp import LPRobustCP
from mnist.train_mnist import SimpleCNN
ImageFile.LOAD_TRUNCATED_IMAGES = True

# specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# argument parser
parser = argparse.ArgumentParser('Robust-CP')
parser.add_argument('--dataset',        type=str,   default='imgnet', help="dataset_name, imgnet or mnist")
parser.add_argument('--batch_size',     type=int,   default=1024, help="batch size for loading data")
parser.add_argument('--corrupt_ratio',  type=float, default=0.05, help="percent of data label being rolled")
parser.add_argument('--noise_upper',    type=float, default=1.,   help="std used for noising images")
parser.add_argument('--noise_lower',    type=float, default=-1.,  help="std used for noising images")
parser.add_argument('--data_dir',       type=str,   default='../LP-robust-conformal/datasets/ImageNet/val', help="dir to imagenet val data")
args = parser.parse_args()


"""
Set-up Stage
"""

if args.dataset == 'imgnet':
    # load pretrained model
    weights = ResNet152_Weights.DEFAULT
    model = resnet152()
    state_dict = torch.load('pretrained_models/resnet152-f82ba261.pth')
    preprocess = weights.transforms()
    # load dataset
    val_dataset = datasets.ImageFolder(root='../LP-robust-conformal/datasets/ImageNet/val', transform=preprocess)
elif args.dataset == 'mnist':
    model = SimpleCNN()
    state_dict = torch.load("pretrained_models/mnist_cnn.pth")
    
    mean, std = (0.1307,), (0.3081,)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # load dataset
    val_dataset = datasets.MNIST(root='../LP-robust-conformal/datasets', train=False, download=False, transform=preprocess)
model.load_state_dict(state_dict)
model.to(device)

# load data
id_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
od_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

lp_robust_cp = LPRobustCP(model, nll_score, 0.1)


"""
Conformal Prediction Stage
"""
# obtain calibration and test scores
id_scores, id_labels, od_scores, od_labels = lp_robust_cp.get_scores(id_loader, od_loader,
                                                                     corrupt_ratio=args.corrupt_ratio,
                                                                     noise_upper=args.noise_upper,
                                                                     noise_lower=args.noise_lower)
id_scores = id_scores.cpu().numpy()
id_labels = id_labels.cpu().numpy()
od_scores = od_scores.cpu().numpy()
od_labels = od_labels.cpu().numpy()

# obtain scores
id_scores = id_scores[np.arange(id_scores.shape[0]), id_labels]
od_scores = od_scores[np.arange(od_scores.shape[0]), od_labels]

# save scores
np.savez(f'{args.dataset}_scores_{args.corrupt_ratio}_{args.noise_upper}.npz', id_scores=id_scores, od_scores=od_scores)
