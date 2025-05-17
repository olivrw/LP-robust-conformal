import os
import ot
import argparse
import random
import pandas as pd
import torch
import torchvision.transforms as transforms
from src.utils import *
from src.lp_robust_cp import LPRobustCP
from train_mnist import SimpleCNN
ImageFile.LOAD_TRUNCATED_IMAGES = True

# specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# argument parser
parser = argparse.ArgumentParser('Robust-CP')
parser.add_argument('--num_trials',     type=int,   default=30,   help="number of experiment runs")
parser.add_argument('--alpha',          type=float, default=0.1,  help="user prescribed confidence (1-alpha)")
parser.add_argument('--cal_ratio',      type=float, default=0.1, help="percent of data used for calibration")
parser.add_argument('--batch_size',     type=int,   default=1024, help="batch size for loading data")
parser.add_argument('--fdiv_radius',    type=float, default=2.5,  help="radius for f-divergence ball")
parser.add_argument('--corrupt_ratio',  type=float, default=0.05, help="percent of data label being rolled")
parser.add_argument('--noise_upper',    type=float, default=1.,   help="std used for noising images")
parser.add_argument('--noise_lower',    type=float, default=-1.,  help="std used for noising images")
parser.add_argument('--eps_upper',      type=float, default=0.,   help="std used for noising images")
parser.add_argument('--eps_lower',      type=float, default=1.5,  help="std used for noising images")
parser.add_argument('--num_grid',       type=int,   default=25,  help="std used for noising images")
parser.add_argument('--worst_case',     type=int,   default=0,    help="boolean for considering w.c. distribution or not")
parser.add_argument('--data_dir',       type=str,   default='/home/gridsan/zwang1/LP-Robust-CP/datasets', help="dir to mnist val data")
parser.add_argument('--save',           type=str,   default='experiments/mnist_grid', help="define the save directory")
args = parser.parse_args()


"""
Set-up Stage
"""

# load pretrained model
model = SimpleCNN()
model.load_state_dict(torch.load("../pretrained_models/mnist_cnn.pth"))
model.to(device)

# load data transforms
mean, std = (0.1307,), (0.3081,)
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# instantiate robust cp class
lp_robust_cp = LPRobustCP(model, nll_score, args.alpha)

# sample for random seeds
seed_range = 100000
num_seeds = args.num_trials
unique_seeds = random.sample(range(seed_range), num_seeds)

columns = ["eps", "rho", "quantile", "coverage", "size"]
result_hist = pd.DataFrame(columns=columns)

# load dataset
cal_loader, test_loader = load_mnist_valdata(args.data_dir, test_transforms, cal_ratio=args.cal_ratio,
                                             batch_size=args.batch_size, seed=42)
"""
Conformal Prediction Stage
"""
# obtain calibration and test scores
calib_scores, calib_labels, tst_scores, tst_labels = lp_robust_cp.get_scores(cal_loader, test_loader,
                                                                             corrupt_ratio=args.corrupt_ratio,
                                                                             noise_upper=args.noise_upper,
                                                                             noise_lower=args.noise_lower,
                                                                             worst_case=bool(args.worst_case))
calib_scores = calib_scores.cpu().numpy()
calib_labels = calib_labels.cpu().numpy()
tst_scores = tst_scores.cpu().numpy()
tst_labels = tst_labels.cpu().numpy()

# obtain calibration scores
cal_scores = calib_scores[np.arange(calib_scores.shape[0]), calib_labels]
tst_scores_for_ot = tst_scores
tst_scores_for_ot = tst_scores_for_ot[np.arange(tst_scores_for_ot.shape[0]), tst_labels]
tst_scores_for_ot = np.random.choice(tst_scores_for_ot, size=600, replace=False)

shuffled_indices = np.random.permutation(len(cal_scores))
half = len(cal_scores) // 2
cal_scores_1 = cal_scores
cal_scores_2 = cal_scores
cal_scores_1 = cal_scores_1[shuffled_indices[:half]]
cal_scores_2 = cal_scores_2[shuffled_indices[half:]]

# create a grid of eps
eps = np.linspace(args.eps_upper, args.eps_lower, args.num_grid)
for i in range(args.num_grid):
    eps_i = eps[i]
    rho_i, _, _ = indicator_cost_plan(cal_scores_1, tst_scores_for_ot, 2*eps_i)
    qhat_i = lp_robust_cp.lp_robust_quantile(cal_scores_2, rho=rho_i, epsilon=eps_i, k=2.)
    # conformal
    lp_prediction_est_sets = tst_scores <= qhat_i
    lp_robust_coverage_est = lp_prediction_est_sets[np.arange(lp_prediction_est_sets.shape[0]), tst_labels].mean()
    lp_robust_avgwidth_est = np.mean(np.sum(lp_prediction_est_sets, axis=1))
    result_hist.loc[len(result_hist.index)] = [eps_i, rho_i, qhat_i, lp_robust_coverage_est, lp_robust_avgwidth_est]

# save results
if not os.path.exists(args.save):
    os.makedirs(args.save)
if args.worst_case == 1:
    result_hist.to_csv(os.path.join(args.save, f'%s_gridresult_hist_{args.corrupt_ratio}_{args.noise_upper}_{args.noise_lower}.csv' % 'wc'))
else:
    result_hist.to_csv(os.path.join(args.save, f'%s_gridresult_hist_{args.corrupt_ratio}_{args.noise_upper}_{args.noise_lower}.csv' % 'reg'))
