import os
import argparse
import random
import pandas as pd
import torch
import torchvision.transforms as transforms
from src.utils import *
from src.lp_robust_cp import LPRobustCP
from src.fdiv_robust_cp import FDivRobustCP
from train_mnist import SimpleCNN
ImageFile.LOAD_TRUNCATED_IMAGES = True

# specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# argument parser
parser = argparse.ArgumentParser('Robust-CP')
parser.add_argument('--num_trials',     type=int,   default=20,   help="number of experiment runs")
parser.add_argument('--alpha',          type=float, default=0.1,  help="user prescribed confidence (1-alpha)")
parser.add_argument('--cal_ratio',      type=float, default=0.1, help="percent of data used for calibration")
parser.add_argument('--batch_size',     type=int,   default=1024, help="batch size for loading data")
parser.add_argument('--fdiv_radius',    type=float, default=2.5,  help="radius for f-divergence ball")
parser.add_argument('--corrupt_ratio',  type=float, default=0.05, help="percent of data label being rolled")
parser.add_argument('--noise_upper',    type=float, default=1.,   help="std used for noising images")
parser.add_argument('--noise_lower',    type=float, default=-1.,  help="std used for noising images")
parser.add_argument('--worst_case',     type=int,   default=0,    help="boolean for considering w.c. distribution or not")
parser.add_argument('--data_dir',       type=str,   default='../LP-Conformal/datasets', help="dir to mnist val data")
parser.add_argument('--save',           type=str,   default='experiments/mnist', help="define the save directory")
args = parser.parse_args()


# define function for f-divergence
def f(x):
    return (x - 1) ** 2


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
fdiv_robust_cp = FDivRobustCP(rho=args.fdiv_radius, tol=1e-12, f=f, is_chisq=True)

# sample for random seeds
seed_range = 100000
num_seeds = args.num_trials
unique_seeds = random.sample(range(seed_range), num_seeds)

columns = ["standard_coverage", "lp_robust_coverage", "fdiv_robust_coverage", 
           "standard_avgsize", "lp_robust_avgsize", "fdiv_robust_avgsize", "seed"]
result_hist = pd.DataFrame(columns=columns)

for seed in unique_seeds:
    # load dataset
    cal_loader, test_loader = load_mnist_valdata(args.data_dir, test_transforms, cal_ratio=args.cal_ratio,
                                                  batch_size=args.batch_size, seed=seed)
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

    # standard cp quantile
    qhat = lp_robust_cp.standard_quantile(cal_scores)
    # lp robust cp quantile
    rho = args.corrupt_ratio
    epsilon = np.max(np.abs((args.noise_upper, args.noise_lower)))
    lp_robust_qhat = lp_robust_cp.lp_robust_quantile(cal_scores, rho=rho, epsilon=epsilon)
    # f-div robust cp quantile
    fdiv_robust_qhat = fdiv_robust_cp.adjusted_quantile(cal_scores, cal_scores.shape[0], args.alpha)

    # form prediction sets
    prediction_sets = tst_scores <= qhat
    lp_prediction_sets = tst_scores <= lp_robust_qhat
    fdiv_prediction_sets = tst_scores <= fdiv_robust_qhat

    """
    Evaluation Stage
    """
    # compute empirical coverage
    empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]), tst_labels].mean()
    lp_robust_coverage = lp_prediction_sets[np.arange(lp_prediction_sets.shape[0]), tst_labels].mean()
    fdiv_robust_coverage = fdiv_prediction_sets[np.arange(fdiv_prediction_sets.shape[0]), tst_labels].mean()
    print(f"The empirical coverage is: {empirical_coverage: .3f}")
    print(f"The LP robust coverage is: {lp_robust_coverage: .3f}")
    print(f"The f-div robust coverage is: {fdiv_robust_coverage: .3f}")

    # compute average prediction set width
    avg_width = np.mean(np.sum(prediction_sets, axis=1))
    lp_robust_avgwidth = np.mean(np.sum(lp_prediction_sets, axis=1))
    fdiv_robust_avgwidth = np.mean(np.sum(fdiv_prediction_sets, axis=1))
    print(f"The average width is: {avg_width: .3f}")
    print(f"The average LP robust width is: {lp_robust_avgwidth: .3f}")
    print(f"The average f-div robust width is: {fdiv_robust_avgwidth: .3f}")

    result_hist.loc[len(result_hist.index)] = [empirical_coverage, lp_robust_coverage, fdiv_robust_coverage,
                                               avg_width, lp_robust_avgwidth, fdiv_robust_avgwidth, seed]

# save the results
if not os.path.exists(args.save):
    os.makedirs(args.save)
    
if args.worst_case == 1:
    result_hist.to_csv(os.path.join(args.save, f'%s_result_hist_{args.corrupt_ratio}_{args.noise_upper}_{args.noise_lower}.csv' % 'wc'))
else:
    result_hist.to_csv(os.path.join(args.save, f'%s_result_hist_{args.corrupt_ratio}_{args.noise_upper}_{args.noise_lower}.csv' % 'reg'))

# plotting
results_file = os.path.join(args.save, f'%s_result_hist_{args.corrupt_ratio}_{args.noise_upper}_{args.noise_lower}.csv' % 'reg')
reg_results = pd.read_csv(results_file).to_numpy()
coverage_results = [reg_results[:, i] for i in range(1, 4)]
size_results = [reg_results[:, j] for j in range(4, 7)]
plot_cp(coverage_results, plt_type='Coverage', plt_name=f'mnist_{args.corrupt_ratio}_{args.noise_upper}_{args.noise_lower}_cover.png', save_dir='figures')
plot_cp(size_results, plt_type='Size', plt_name=f'mnist_{args.corrupt_ratio}_{args.noise_upper}_{args.noise_lower}_size.png', save_dir='figures')

