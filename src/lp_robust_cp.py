import torch
import numpy as np
from tqdm import tqdm
from typing import Callable
import torch.nn as nn
from src.utils import *
ImageFile.LOAD_TRUNCATED_IMAGES = True
# specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LPRobustCP:
    def __init__(self, model: nn.Module, score_func: Callable, alpha: float) -> None:
        super(LPRobustCP, self).__init__()
        self.model = model
        self.score_func = score_func
        self.alpha = alpha

    def get_scores(self, calib_loader, test_loader, corrupt_ratio=0.05, noise_upper=1., 
            noise_lower=-1., worst_case=False, perturb=True, save=False):
        # get calibration scores
        self.model.eval()
        with torch.no_grad():
            cal_scoreslist = []
            cal_labelslist = []
            for _, batch in tqdm(enumerate(calib_loader), total=len(calib_loader)):
                cal_features, cal_labels = batch
                cal_features.to(device)
                cal_labels.to(device)
                cal_scoreslist.append(self.score_func(self.model, cal_features, cal_labels))
                cal_labelslist.append(cal_labels)
                torch.cuda.empty_cache()

            test_scoreslist = []
            test_labelslist = []
            for _, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
                test_features, test_labels = batch
                test_features.to(device)
                test_labels.to(device)
                if perturb is True:
                    test_features_pert, test_labels_pert = perturb_test_data(test_features, test_labels,
                                                                   corrupt_ratio=corrupt_ratio,
                                                                   noise_upper=noise_upper,
                                                                   noise_lower=noise_lower,
                                                                   worst_case=worst_case)
                else:
                    test_features_pert = test_features
                    test_labels_pert = test_labels
                test_scoreslist.append(self.score_func(self.model, test_features_pert, test_labels_pert))
                test_labelslist.append(test_labels_pert)
                torch.cuda.empty_cache()

            calib_scores = torch.cat(cal_scoreslist, dim=0)
            calib_labels = torch.cat(cal_labelslist)
            tst_scores = torch.cat(test_scoreslist, dim=0)
            tst_labels = torch.cat(test_labelslist)

        if save is True:
            torch.save(calib_scores, 'imgnet_cal_scores.pt')
            torch.save(calib_labels, 'imgnet_cal_labels.pt')
            torch.save(tst_scores, 'imgnet_tst_scores.pt')
            torch.save(tst_labels, 'imgnet_tst_labels.pt')

        return calib_scores, calib_labels, tst_scores, tst_labels

    # Compute the robust empirical quantile
    def lp_robust_quantile(self, calib_scores, rho: float, epsilon: float, k=1.):
        n = calib_scores.shape[0]
        alpha_prime = (n + 1) * self.alpha / n - rho / n
        q_level = 1. - alpha_prime + rho
        return np.quantile(calib_scores, q_level, method='higher') + k*epsilon

    def standard_quantile(self, calib_scores):
        n = calib_scores.shape[0]
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        return np.quantile(calib_scores, q_level, method='higher')



