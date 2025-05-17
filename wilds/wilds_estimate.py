import torch
import argparse
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader
import torchvision.transforms as transforms
from torchvision import models
from src.utils import *
from src.lp_robust_cp import LPRobustCP

# specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# argument parser
parser = argparse.ArgumentParser('Robust-CP')
parser.add_argument('--num_trials',     type=int,   default=30,   help="number of experiment runs")
parser.add_argument('--alpha',          type=float, default=0.1,  help="user prescribed confidence (1-alpha)")
parser.add_argument('--cal_ratio',      type=float, default=0.1, help="percent of data used for calibration")
parser.add_argument('--batch_size',     type=int,   default=256, help="batch size for loading data")
parser.add_argument('--corrupt_ratio',  type=float, default=0.05, help="percent of data label being rolled")
parser.add_argument('--noise_upper',    type=float, default=1.,   help="std used for noising images")
parser.add_argument('--noise_lower',    type=float, default=-1.,  help="std used for noising images")
parser.add_argument('--eps_upper',      type=float, default=1.2,   help="std used for noising images")
parser.add_argument('--eps_lower',      type=float, default=0.2,  help="std used for noising images")
parser.add_argument('--k',              type=float, default=2,   help="std used for noising images")
parser.add_argument('--reg',            type=float, default=0.05,   help="std used for noising images")
parser.add_argument('--num_grid',       type=int,   default=25,  help="std used for noising images")
parser.add_argument('--worst_case',     type=int,   default=0,    help="boolean for considering w.c. distribution or not")
parser.add_argument('--save',           type=str,   default='experiments/wilds', help="define the save directory")
args = parser.parse_args()


"""
Set-up Stage
"""

# Load model
print('Loading model ...')
model = models.resnet50(weights=None, progress=False)
model.fc = nn.Linear(2048, 182)
model.eval()

# Load parameters
print('Loading parameters ...')

checkpoint = torch.load('../pretrained_models/wilds_model.pth')

state_dict = checkpoint['algorithm']
new_state_dict = {}
for k, v in state_dict.items():
    name = k[6:]
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.to(device)

# instantiate robust cp class
lp_robust_cp = LPRobustCP(model, nll_score, args.alpha)

columns = ["eps", "rho", "quantile", "coverage", "size"]
result_hist = pd.DataFrame(columns=columns)

# load dataset
dataset = get_dataset(dataset="iwildcam", download=False)

# Get the test set
test_data = dataset.get_subset(
    "test",
    transform=transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
)
# Prepare the evaluation data loader
id_test_data = dataset.get_subset(
    "id_test",
    transform=transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
)
test_loader = get_eval_loader("standard", test_data, batch_size=args.batch_size)
calib_loader = get_eval_loader("standard", id_test_data, batch_size=args.batch_size)

cache_file = "cached_scores.npz"     # change path/name if you like

if os.path.isfile(cache_file):
    # Fast path: pull scores/labels from disk
    cache = np.load(cache_file)
    calib_scores = cache["calib_scores"]
    calib_labels = cache["calib_labels"]
    tst_scores   = cache["tst_scores"]
    tst_labels   = cache["tst_labels"]
    print(f"[cache] loaded calibration/test scores from {cache_file}")

else:
    with torch.no_grad():
        cal_scoreslist, cal_labelslist = [], []
        for _, batch in tqdm(enumerate(calib_loader), total=len(calib_loader)):
            cal_features, cal_labels, _ = batch
            cal_features, cal_labels = cal_features.to(device), cal_labels.to(device)
            cal_scoreslist.append(nll_score(model, cal_features, cal_labels))
            cal_labelslist.append(cal_labels)
            torch.cuda.empty_cache()

        test_scoreslist, test_labelslist = [], []
        for _, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            test_features, test_labels, _ = batch
            test_features, test_labels = test_features.to(device), test_labels.to(device)
            test_scoreslist.append(nll_score(model, test_features, test_labels))
            test_labelslist.append(test_labels)
            torch.cuda.empty_cache()

        calib_scores = torch.cat(cal_scoreslist, dim=0).cpu().numpy()
        calib_labels = torch.cat(cal_labelslist).cpu().numpy()
        tst_scores   = torch.cat(test_scoreslist,  dim=0).cpu().numpy()
        tst_labels   = torch.cat(test_labelslist).cpu().numpy()

    # Save for next time (compressed ≈ 1/3 the size of .npy files)
    np.savez_compressed(
        cache_file,
        calib_scores=calib_scores,
        calib_labels=calib_labels,
        tst_scores=tst_scores,
        tst_labels=tst_labels,
    )
    print(f"[cache] saved scores to {cache_file}")

# obtain calibration scores
cal_scores = calib_scores[np.arange(calib_scores.shape[0]), calib_labels]
tst_scores_for_ot = tst_scores
tst_scores_for_ot = tst_scores_for_ot[np.arange(tst_scores_for_ot.shape[0]), tst_labels]
tst_scores_for_ot = np.random.choice(tst_scores_for_ot, size=1200, replace=False)

shuffled_indices = np.random.permutation(len(cal_scores))
half = len(cal_scores) // 2
cal_scores_1 = cal_scores
cal_scores_2 = cal_scores
cal_scores_1 = cal_scores_1[shuffled_indices[:half]]
cal_scores_2 = cal_scores_2[shuffled_indices[half:]]

# create a grid of eps
k = args.k
eps = np.linspace(args.eps_lower, args.eps_upper, args.num_grid)
for i in tqdm(range(args.num_grid), desc="Processing grid"):
    eps_i = eps[i]
    rho_i, _, _ = indicator_cost_plan(cal_scores_1, tst_scores_for_ot, k*eps_i, reg=args.reg)
    print(f'Estimated rho: {rho_i}')
    qhat_i = lp_robust_cp.lp_robust_quantile(cal_scores_2, rho=rho_i, epsilon=eps_i, k=k)
    # conformal
    lp_prediction_est_sets = tst_scores <= qhat_i
    lp_robust_coverage_est = lp_prediction_est_sets[np.arange(lp_prediction_est_sets.shape[0]), tst_labels].mean()
    lp_robust_avgwidth_est = np.mean(np.sum(lp_prediction_est_sets, axis=1))
    result_hist.loc[len(result_hist.index)] = [k*eps_i, rho_i, qhat_i, lp_robust_coverage_est, lp_robust_avgwidth_est]

# save the results
if not os.path.exists(args.save):
    os.makedirs(args.save)
result_hist.to_csv(os.path.join(args.save, f'%s_gridresult_hist_{args.corrupt_ratio}_{args.noise_upper}_{args.noise_lower}.csv' % 'reg'))

