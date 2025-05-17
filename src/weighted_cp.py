import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List


@torch.no_grad()
def _get_logits_and_features(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        dataset
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    model.eval()

    if dataset == 'imgnet':
        feature_extractor = torch.nn.Sequential(
            *list(model.children())[:-1],
            torch.nn.Flatten(1)
        ).to(device)
    else:
        feature_extractor = nn.Sequential(
            model.conv1,
            nn.ReLU(),
            model.pool,
            model.conv2,
            nn.ReLU(),
            model.pool,
            nn.Flatten(1),
            model.fc1,
            nn.ReLU()
        ).to(device) 

    probs, scores, feats = [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        p = torch.softmax(logits, dim=1)
        z = feature_extractor(x)

        probs.append(p.cpu().numpy())
        feats.append(z.cpu().numpy())
        scores.append((1.0 - p.gather(1, y.view(-1, 1))).cpu().numpy())

    probs   = np.concatenate(probs,   axis=0)
    feats   = np.concatenate(feats,   axis=0)
    scores  = np.concatenate(scores,  axis=0).ravel()

    return probs, scores, feats


def _weighted_quantile(scores: np.ndarray,
                       weights: np.ndarray,
                       alpha: float) -> float:

    weights /= (weights.sum() + 1.0)
    order          = np.argsort(scores)
    cum_weights    = np.cumsum(weights[order])
    idx            = np.searchsorted(cum_weights, 1.0 - alpha, side="right")
    return scores[order[min(idx, len(scores) - 1)]]



def weighted_conformal_prediction(
        model: torch.nn.Module,
        calib_loader: DataLoader,
        test_loader:  DataLoader,
        dataset,
        alpha: float = 0.1,
        device: torch.device = torch.device("cuda:0")
) -> Tuple[float, float, List[np.ndarray], float]:

    P_cal, s_cal, φ_cal = _get_logits_and_features(model, calib_loader, device, dataset)
    P_tst, _,    φ_tst  = _get_logits_and_features(model, test_loader,  device, dataset)

    y_cal = np.concatenate([y.numpy() for _, y in calib_loader], axis=0)
    y_tst = np.concatenate([y.numpy() for _, y in test_loader],  axis=0)

    X_joint = np.vstack([φ_cal, φ_tst])
    y_joint = np.hstack([np.zeros(len(φ_cal)), np.ones(len(φ_tst))])
    scaler  = StandardScaler().fit(X_joint)
    X_norm  = scaler.transform(X_joint)

    lr = LogisticRegression(
        solver='lbfgs', 
        max_iter=200,
        tol=1e-2,
        multi_class='auto',
        n_jobs=-1
    )
    lr.fit(X_norm, y_joint)
    p_test_joint = lr.predict_proba(X_norm)[:, 1]

    w = p_test_joint[:len(φ_cal)] / (1.0 - p_test_joint[:len(φ_cal)])

    q_hat = _weighted_quantile(s_cal, w.copy(), alpha)

    sets = [np.where(p >= 1.0 - q_hat)[0] for p in P_tst]         

    coverage = np.mean([y in S for y, S in zip(y_tst, sets)])
    avg_size = np.mean([len(S) for S in sets])

    return coverage, avg_size, sets, q_hat

