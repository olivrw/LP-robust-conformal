import torch
import argparse
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
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
parser.add_argument('--alpha',          type=float, default=0.1,  help="user prescribed confidence (1-alpha)")
parser.add_argument('--batch_size',     type=int,   default=256, help="batch size for loading data")
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

# sample for random seeds
unique_seed = random.sample(range(100000), 1)[0]

# pick ambiguity set parameter space
eps_list = torch.arange(0, 2.05, 0.2) 
rho_list = torch.arange(0, 0.022, 0.002)

columns = ["lp_robust_coverage", "lp_robust_avgsize", "eps", "rho"]
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

plot_cover_list = []
plot_size_list = []
for i, eps in enumerate(eps_list):
    for j, rho in enumerate(rho_list):
        with torch.no_grad():
            cal_scoreslist = []
            cal_labelslist = []
            for _, batch in tqdm(enumerate(calib_loader), total=len(calib_loader)):
                cal_features, cal_labels, _ = batch
                cal_features.to(device)
                cal_labels.to(device)
                cal_scoreslist.append(nll_score(model, cal_features, cal_labels))
                cal_labelslist.append(cal_labels)
                torch.cuda.empty_cache()

            test_scoreslist = []
            test_labelslist = []
            for _, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
                test_features, test_labels, _ = batch
                test_features.to(device)
                test_labels.to(device)
                test_scoreslist.append(nll_score(model, test_features, test_labels))
                test_labelslist.append(test_labels)
                torch.cuda.empty_cache()

            calib_scores = torch.cat(cal_scoreslist, dim=0)
            calib_labels = torch.cat(cal_labelslist)
            tst_scores = torch.cat(test_scoreslist, dim=0)
            tst_labels = torch.cat(test_labelslist)
        calib_scores = calib_scores.cpu().numpy()
        calib_labels = calib_labels.cpu().numpy()
        tst_scores = tst_scores.cpu().numpy()
        tst_labels = tst_labels.cpu().numpy()
        
        """
        Conformal Prediction Stage
        """
        # obtain true label scores
        cal_scores = calib_scores[np.arange(calib_scores.shape[0]), calib_labels]
        # compute quantile
        lp_robust_qhat = lp_robust_cp.lp_robust_quantile(cal_scores, rho=rho, epsilon=eps).cpu().numpy()                        
        # form prediction sets
        lp_prediction_sets = tst_scores <= lp_robust_qhat

        """
        Evaluation Stage
        """
        # compute empirical coverage
        lp_robust_coverage = lp_prediction_sets[np.arange(lp_prediction_sets.shape[0]), tst_labels].mean()                        
        print(f"The LP robust coverage under rho={rho}, eps={eps} is: {lp_robust_coverage: .3f}")
        # compute average prediction set width
        lp_robust_avgwidth = np.mean(np.sum(lp_prediction_sets, axis=1))
        print(f"The average LP robust width under rho={rho}, eps={eps} is: {lp_robust_avgwidth: .3f}")
        result_hist.loc[len(result_hist.index)] = [lp_robust_coverage, lp_robust_avgwidth, eps.item(), rho.item()]
        
        plot_cover_list.append((lp_robust_coverage, eps, rho))
        plot_size_list.append((lp_robust_avgwidth, eps, rho))


if not os.path.exists(args.save):
    os.makedirs(args.save)
# save result tab
result_hist.to_csv(os.path.join(args.save, 'wilds_result_hist.csv'))
# plotting
cov_pltname = 'wilds_cover_plt.png'
size_pltname = 'wilds_size_plt.png'
cov_savepth = os.path.join(args.save, cov_pltname)
size_savepth =  os.path.join(args.save, size_pltname)

# obtain set size giving coverage closest to 0.9
diffs = np.abs(result_hist['lp_robust_coverage'] - 0.9)
closest_idx = np.argmin(diffs)
closest_row = result_hist.iloc[closest_idx]
highlight_val = closest_row[2]

eps_rho_plot(np.array(plot_cover_list), plt_type='Coverage', savefig_path=cov_pltname)
eps_rho_plot(np.array(plot_size_list), plt_type='Size', highlight_val=highlight_val, savefig_path=size_savepth)
