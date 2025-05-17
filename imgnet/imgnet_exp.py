import os
import ot
from tqdm.auto import tqdm
import argparse
import random
import torch.nn as nn
import pandas as pd
from torchvision.models import resnet152, ResNet152_Weights
from src.utils import *
from src.lp_robust_cp import LPRobustCP
from src.weighted_cp import weighted_conformal_prediction
ImageFile.LOAD_TRUNCATED_IMAGES = True

# fdiv imports
import sys

HERE       = os.path.dirname(os.path.abspath(__file__))    # …/LP-Robust-CP/imgnet
PROJECT    = os.path.dirname(HERE)                         # …/LP-Robust-CP
FDIV_BACK  = os.path.join(PROJECT, "fdiv_code", "backend") # …/LP-Robust-CP/fdiv_code/backend

sys.path.insert(0, FDIV_BACK)

import tensorflow as tf
import tensorflow.keras.backend as K
from tf_backend.tf_utils import *

import np_backend.conformal_utils as cf_utils
from np_backend.dro_conformal import *
import cvxpy as cp

# RSCP imports
import RSCP.Score_Functions as scores
from RSCP.utils import evaluate_predictions, get_scores, calibration, prediction

# Fine Grain imports
from finegrain_code.utils_torch import ConformalPredictionTorch
from scipy.stats import norm

# specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# argument parser
parser = argparse.ArgumentParser('Robust-CP')
parser.add_argument('--num_trials',     type=int,   default=20,   help="number of experiment runs")
parser.add_argument('--alpha',          type=float, default=0.1,  help="user prescribed confidence (1-alpha)")
parser.add_argument('--cal_ratio',      type=float, default=0.02, help="percent of data used for calibration")
parser.add_argument('--batch_size',     type=int,   default=1024, help="batch size for loading data")
parser.add_argument('--corrupt_ratio',  type=float, default=0.05, help="percent of data label being rolled")
parser.add_argument('--noise_upper',    type=float, default=1.,   help="std used for noising images")
parser.add_argument('--noise_lower',    type=float, default=-1.,  help="std used for noising images")
parser.add_argument('--rho_est',        type=float, default=-1.,  help="estimated rho")
parser.add_argument('--eps_est',        type=float, default=-1.,  help="estimated eps")
parser.add_argument('--worst_case',     type=int,   default=0,    help="boolean for considering w.c. distribution or not")
parser.add_argument('--data_dir',       type=str,   default='/home/olivrw/LP-Robust-CP/datasets/ImageNet/val', help="dir to imagenet val data")
parser.add_argument('--save',           type=str,   default='experiments/imgnet', help="define the save directory")
# fdiv arguments
parser.add_argument('--n_slabs_directions', default=1000, type=int)
parser.add_argument('--delta_slab', default=0.1, type=float)
parser.add_argument('--alpha_slab', default=0.1, type=float)
# RSCP arguments
parser.add_argument('--ratio', default=3.5, type=float,
                    help='Ratio between adversarial noise bound to smoothing noise')
parser.add_argument('--n_s', default=64, type=int, help='Number of samples used for estimating smoothed score')
parser.add_argument('--gpu_cap',        type=int,   default=2048, help="batch size for RSCP")
parser.add_argument('--coverage_on_label', action='store_true', help='True for getting coverage and size for each label')
args = parser.parse_args()


"""
Set-up Stage
"""

# load pretrained model
weights = ResNet152_Weights.DEFAULT
model = resnet152()
state_dict = torch.load('../pretrained_models/resnet152-f82ba261.pth')
model.load_state_dict(state_dict)
model.to(device).eval()

# parameters for RSCP
epsilon = np.max(np.abs((args.noise_upper, args.noise_lower)))  # L2 bound on the adversarial noise
ratio = args.ratio  # ratio between adversarial noise bound to smoothed noise
sigma_smooth = ratio * epsilon # sigma used fro smoothing
n_smooth = args.n_s
GPU_CAPACITY = args.gpu_cap
coverage_on_label = args.coverage_on_label
model_norm = model

# feature extractor for fdiv
n_features = model.fc.in_features
model_preprocessing = nn.Sequential(
    *list(model.children())[:-1],  
    nn.Flatten(1)               
).to(device)

# load data transforms
preprocess = weights.transforms()

# instantiate robust cp class
lp_robust_cp = LPRobustCP(model, nll_score, args.alpha)

# sample for random seeds
seed_range = 100000
num_seeds = args.num_trials
unique_seeds = random.sample(range(seed_range), num_seeds)

columns = ["standard_coverage", "lp_robust_coverage", "lp_est_coverage", "fdiv_robust_coverage", "fg_covergae", "rscp_coverage", "weight_coverage",
           "standard_avgsize", "lp_robust_avgsize", "lp_est_avgsize", "fdiv_robust_avgsize", "fg_avgwidth", "rscp_avgsize", "weight_avgsize", "seed"]

result_hist = pd.DataFrame(columns=columns)

for seed in unique_seeds:
    # load dataset
    cal_loader, test_loader = load_imgnet_valdata(args.data_dir, preprocess, cal_ratio=args.cal_ratio,
                                                  batch_size=args.batch_size, seed=seed)
    # obtain image embeddings for FDiv
    all_imgs_feature = []
    with torch.no_grad():
        for _, batch in tqdm(enumerate(cal_loader), total=len(cal_loader)):
            features, _ = batch
            all_imgs_feature.append(model_preprocessing(features.to(device)))
    data_features = torch.cat(all_imgs_feature, dim=0)
    data_features = data_features.cpu().numpy()

    # obtain test embeddings for FG-CP
    tst_imgs_feature = []
    with torch.no_grad():
        for _, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            features, _ = batch
            tst_imgs_feature.append(model_preprocessing(features.to(device)))
    test_features = torch.cat(tst_imgs_feature, dim=0)
    test_features = test_features.cpu().numpy() 
 
    """
    Conformal Prediction Stage
    """
    # obtain calibration and test scores
    calib_scores, calib_labels, tst_scores, tst_labels, x_test_adv, y_test_adv = lp_robust_cp.get_scores(cal_loader, test_loader,
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
    # obtain test score for FG-CP
    tst_scores_fg = tst_scores[np.arange(tst_scores.shape[0]), tst_labels]

    # Vanilla CP
    qhat = lp_robust_cp.standard_quantile(cal_scores)

    # LP Robust CP
    rho = args.corrupt_ratio
    epsilon = np.max(np.abs((args.noise_upper, args.noise_lower)))
    lp_robust_qhat = lp_robust_cp.lp_robust_quantile(cal_scores, rho=rho, epsilon=epsilon, k=2.)

    # LP Robust CP with EST 
    lp_robust_qhat_est = lp_robust_cp.lp_robust_quantile(cal_scores, rho=args.rho_est, epsilon=args.eps_est, k=2.)
    
    # F-Div
    slab_quantiles = np.zeros(args.n_slabs_directions)
    for slab_idx in range(args.n_slabs_directions):
        if slab_idx % 10 == 0:
            # Might want to try out different sampling mechanisms for direction
            direction = np.random.randn(data_features.shape[1])
            direction = direction / np.linalg.norm(direction)
            slab_quantiles[slab_idx] = find_worst_case_slab_quantile(
                        direction, data_features, cal_scores, args.alpha, args.delta_slab)
    fdiv_robust_qhat = np.quantile(slab_quantiles, 1-args.alpha_slab, interpolation="higher")

    cal_examples = enumerate(cal_loader)
    _, (_, y_cal) = next(cal_examples)
    
    # Fine Grain CP
    samples = np.concatenate([data_features, y_cal.reshape(-1,1)],axis=1)
    obj = ConformalPredictionTorch(samples, args.alpha, rho, 'kl', "cmr")
    
    k = min(1000, test_features.shape[0])
    perm = torch.randperm(test_features.shape[0])
    idx_sub, idx_rest = perm[:k], perm[k:]
    x_sub,  x_rest  = test_features[idx_sub], test_features[idx_rest]

    tst_scores_rest = tst_scores_fg[idx_rest] 

    obj.initial_torch(data_features, x_sub, y_cal, cal_scores, model, 'random_forest', 5)
    shiftsamples=np.concatenate([x_rest, y_test_adv[idx_rest].reshape(-1,1)],axis=1)
    
    count=0
    lenth=0 
    for i, shiftsample in enumerate(shiftsamples):
        boolin, quant = obj.one_test(shiftsample[:-1], cal_scores, tst_scores_rest[i], '3')
        if boolin:
            count+=1
        lenth += quant
    fg_coverage = count/shiftsamples.shape[0]
    fg_avgwidth = 2*lenth/shiftsamples.shape[0] 

    # RSCP
    num_of_classes = 1000
    cal_indices = torch.arange(50000*args.cal_ratio)
    tst_indices = torch.arange(50000*(1-args.cal_ratio))
    cal_examples = enumerate(cal_loader)
    cal_batch_idx, (x_cal, y_cal) = next(cal_examples)
    scores_list = []
    scores_list.append(scores.class_probability_score) 
    
    correction = float(epsilon) / float(sigma_smooth) 
  
    smoothed_scores_clean_cal, scores_smoothed_clean_cal = get_scores(model_norm, x_cal, cal_indices, n_smooth, sigma_smooth, num_of_classes, 
                                                                      scores_list, base=False, device=device, GPU_CAPACITY=GPU_CAPACITY) 
    smoothed_scores_adv_tst, scores_smoothed_adv_tst = get_scores(model_norm, x_test_adv, tst_indices, n_smooth, sigma_smooth, num_of_classes,                                                                  scores_list, base=False, device=device, GPU_CAPACITY=GPU_CAPACITY)
    
    # calibrate base model with the desired scores and get the thresholds
    thresholds, bounds = calibration(scores_smoothed=scores_smoothed_clean_cal, smoothed_scores=smoothed_scores_clean_cal, 
                                     alpha=args.alpha, num_of_scores=len(scores_list), correction=correction, base=False)
    predicted_adv_sets = prediction(scores_smoothed=scores_smoothed_adv_tst, smoothed_scores=smoothed_scores_adv_tst, 
                                    num_of_scores=len(scores_list), thresholds=thresholds, correction=correction, base=False)
    res = evaluate_predictions(predicted_adv_sets[0][1], None, y_test_adv.numpy(), conditional=True, 
                               coverage_on_label=coverage_on_label, num_of_classes=1000) 

    # Weighted Conformal
    w_coverage, w_avgsize, _, _ = weighted_conformal_prediction(
        model, cal_loader, test_loader, alpha=0.1, device=device, dataset='imgnet'
    )

    # form prediction sets
    prediction_sets = tst_scores <= qhat
    lp_prediction_sets = tst_scores <= lp_robust_qhat
    lp_prediction_est_sets = tst_scores <= lp_robust_qhat_est
    fdiv_prediction_sets = tst_scores <= fdiv_robust_qhat

    """
    Evaluation Stage
    """
    # compute empirical coverage
    empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]), tst_labels].mean()
    lp_robust_coverage = lp_prediction_sets[np.arange(lp_prediction_sets.shape[0]), tst_labels].mean()
    lp_robust_coverage_est = lp_prediction_est_sets[np.arange(lp_prediction_est_sets.shape[0]), tst_labels].mean()
    fdiv_robust_coverage = fdiv_prediction_sets[np.arange(fdiv_prediction_sets.shape[0]), tst_labels].mean()
    rscp_coverage = res.loc[0, 'Coverage']
    print(f"Vanilla CP coverage under rho={rho}, eps={epsilon}: {empirical_coverage: .3f}")
    print(f"LP coverage under rho={rho}, eps={epsilon}: {lp_robust_coverage: .3f}")
    print(f"LP EST coverage under rho={rho}, eps={epsilon}: {lp_robust_coverage_est: .3f}")
    print(f"F-Div coverage under rho={rho}, eps={epsilon}: {fdiv_robust_coverage: .3f}")
    print(f"Fine Grain CP coverage under rho={rho}, eps={epsilon}: {fg_coverage: .3f}")
    print(f"RSCP coverage under rho={rho}, eps={epsilon}: {rscp_coverage: .3f}")
    print(f"Weighted coverage under rho={rho}, eps={epsilon}: {w_coverage: .3f}")

    # compute average prediction set width
    avg_width = np.mean(np.sum(prediction_sets, axis=1))
    lp_robust_avgwidth = np.mean(np.sum(lp_prediction_sets, axis=1))
    lp_robust_avgwidth_est = np.mean(np.sum(lp_prediction_est_sets, axis=1))
    fdiv_robust_avgwidth = np.mean(np.sum(fdiv_prediction_sets, axis=1))
    rscp_avgwidth = res.loc[0, 'Size']
    print(f"Vanilla CP width under rho={rho}, eps={epsilon}: {avg_width: .3f}")
    print(f"LP width under rho={rho}, eps={epsilon}: {lp_robust_avgwidth: .3f}")
    print(f"LP EST width under rho={rho}, eps={epsilon}: {lp_robust_avgwidth_est: .3f}")
    print(f"F-Div width under rho={rho}, eps={epsilon}: {fdiv_robust_avgwidth: .3f}")
    print(f"Fine Grain CP width under rho={rho}, eps={epsilon}: {fg_avgwidth: .3f}")
    print(f"RSCP width under rho={rho}, eps={epsilon}: {rscp_avgwidth: .3f}")
    print(f"Weighted width under rho={rho}, eps={epsilon}: {w_avgsize: .3f}")
    
    result_hist.loc[len(result_hist.index)] = [empirical_coverage, lp_robust_coverage, lp_robust_coverage_est, fdiv_robust_coverage, fg_coverage, rscp_coverage, w_coverage,
                                               avg_width, lp_robust_avgwidth, lp_robust_avgwidth_est, fdiv_robust_avgwidth, fg_avgwidth, rscp_avgwidth, w_avgsize, seed]

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
coverage_results = [reg_results[:, i] for i in range(1, 8)]
size_results = [reg_results[:, j] for j in range(8, 15)]

plot_cp(coverage_results, plt_type='Coverage', plt_name=f'imgnet_{args.corrupt_ratio}_{args.noise_upper}_{args.noise_lower}_cover.png', 
        save_dir='figures', )
plot_cp(size_results, plt_type='Size', plt_name=f'imgnet_{args.corrupt_ratio}_{args.noise_upper}_{args.noise_lower}_size.png', save_dir='figures')

