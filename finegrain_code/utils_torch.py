# conformal_torch.py
import warnings
import math
warnings.filterwarnings("ignore",
                        category=FutureWarning,
                        message="`max_features='auto'` has been deprecated*")

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.linear_model     import LogisticRegressionCV
from sklearn.ensemble         import RandomForestClassifier
from finegrain_code.qosa     import base_forest   # unchanged third‑party tree package

# ---------------------------------------------------------------------
#   Helper: safe softmax‑based predict_proba for an arbitrary torch model
# ---------------------------------------------------------------------
@torch.no_grad()
def torch_predict_proba(model, x, device):
    """
    x: torch.Tensor on *any* device or NumPy array
    Returns a NumPy array of shape (n, C) with class probabilities
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float().to(device)
    else:
        x = x.to(device)
    model.eval()
    logits = model(x)
    if logits.dim() == 1:               # regression head
        # probit‐style mapping to (0,1); caller should know what it wants
        return torch.sigmoid(logits).unsqueeze(1).cpu().numpy()
    probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()


# ---------------------------------------------------------------------
#   f‑divergence primitives  (torch‑friendly)
# --------------------------------------------------------------------

def f_kl(x):
    if torch.is_tensor(x):
        return x * torch.log(x)
    else:                        # x is float
        return x * math.log(x)

def f_tv(x):
    return torch.abs(x - 1) / 2 if torch.is_tensor(x) else abs(x - 1) / 2

def f_chi(x):
    return (x - 1) ** 2          # works for both float and tensor


class ConformalPredictionTorch:
    """
    A GPU‑enabled re‑implementation of the original Conformal_Prediction class.
    Only the public surface changes:
        * pass `device` (e.g. "cuda:0") in the ctor
        * call .initial_torch(X_P, X_Q, y_P, model, classifier_type=...)
            where `model` is any pretrained torch.nn.Module
    All other APIs and output semantics are preserved.
    """

    def __init__(self,
                 samples: np.ndarray | torch.Tensor,
                 alpha: float,
                 rho: float,
                 f_type: str,
                 score_type: str = "cmr",
                 device: str | torch.device = "cuda:0"):

        self.device     = torch.device(device)
        self.alpha      = alpha
        self.rho        = rho
        self.score_type = score_type

        # f‑divergence selector ------------------------------------------------
        div_map = {"kl": f_kl, "tv": f_tv, "chi_square": f_chi}
        if f_type not in div_map:
            raise ValueError("f_type must be one of kl/tv/chi_square")
        self.f      = div_map[f_type]
        self.f_type = f_type

        # score selector -------------------------------------------------------
        if score_type == "cmr":
            self.score = self._cmr_score
            self.smax  = torch.tensor(float("inf"), device=self.device)
        elif score_type == "aps":
            self.score = self._aps_score
            self.smax  = torch.tensor(1.0, device=self.device)
        else:
            raise ValueError("score_type must be cmr or aps")

        # keep a torch copy of the calibration sample --------------------------
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples).float()
        self.sample = samples.to(self.device)


    # ---------------------------------------------------------------------
    #   Public initialisation (weight model + user‑supplied NN + m(·))
    # ---------------------------------------------------------------------
    def initial_torch(self,
                      X_P: np.ndarray,
                      X_Q: np.ndarray,
                      y_P: np.ndarray,
                      calib_scores,
                      model: torch.nn.Module,
                      classifier_type: str = "logistic",
                      tree_depth: int | None = None):
        """
        Fit covariate‑shift weights **on CPU with scikit‑learn**,
        register a *pre‑trained* torch model as self.model_u,
        and learn the conditional CDF m(·).
        """
        self.model_u = model.to(self.device).eval()          # ← main change
        self._fit_weights(X_P, X_Q, classifier_type, tree_depth)
        print("w done.")
        self._fit_m(X_P, y_P, calib_scores, X_Q)                           # uses base_forest
        print("m done.")


    # ---------------------------------------------------------------------
    #   1) Covariate‑shift weights  (still CPU / NumPy for sklearn)
    # ---------------------------------------------------------------------
    def _fit_weights(self, X_P, X_Q,
                     classifier_type="logistic",
                     tree_depth=None):

        if classifier_type == "logistic":
            clf = LogisticRegressionCV(penalty="l1", solver="liblinear")
        elif classifier_type == "random_forest":
            clf = RandomForestClassifier(max_depth=tree_depth)
        elif classifier_type == "xgb":
            from whyshift import fetch_model
            clf = fetch_model("xgb")
        else:
            raise ValueError("Unsupported classifier_type")

        P0, P1 = X_P.shape[0], X_Q.shape[0]
        # merge and fit ----------------------------------------------------
        X_merge   = np.concatenate([X_P, X_Q], axis=0)
        y_merge   = np.concatenate([np.zeros(P0), np.ones(P1)])
        clf.fit(X_merge, y_merge)

        # handy references -------------------------------------------------
        sample_np = X_P
        proba_PQ  = clf.predict_proba(X_merge)
        proba_PQ  = np.maximum(proba_PQ, 1e-10)

        # estimate shifted‑rho for robust CP (KL case) ---------------------
        if self.f_type == "kl":
            ratio = (P0 / P1) * (proba_PQ[P0:, 1] / proba_PQ[P0:, 0])
            self.shiftrho = self.rho + np.log(ratio).mean()

        # store weight‑function that can accept torch OR NumPy -------------
        def w(x):
            """
            x : array‑like [n, d] – torch tensor or NumPy
            returns: NumPy weights of shape (n,)
            """
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x
            X_all = np.concatenate([sample_np, x_np], axis=0)
            p_all = np.maximum(clf.predict_proba(X_all), 1e-10)
            return (P0 / P1) * (p_all[:, 1] / p_all[:, 0])

        self.w       = w
        self.shift_x = X_Q


    # ---------------------------------------------------------------------
    #   2) Conditional CDF m(·) via QuantileRegressionForest (CPU)
    # ---------------------------------------------------------------------
    """ 
    def _fit_m(self, X_train, y_train, calib_scores, X_Q):

        # one draw of U for EVERY calibration point -----------------------
        U_cal = np.random.uniform(size=X_train.shape[0])

        # compute scores with torch model_u (→ NumPy)
        S_train = calib_scores  
        S_train_np = S_train if isinstance(S_train, np.ndarray) else S_train.cpu().numpy()

        # fit forest -------------------------------------------------------
        qrf = base_forest.QuantileRegressionForest()
        qrf.fit(X_train, S_train_np)               # CPU

        # cache for fast CDF lookup ---------------------------------------
        y_sorted = np.sort(S_train_np)
        N        = X_train.shape[0] - 1

        def conditional_cdf(x, t):
            C = qrf.predict_C_CDF(x)               # (n, N+1)
            # binary search on sorted y’s
            l, r = 0, N
            while r - l > 1:
                mid = (l + r) // 2
                if y_sorted[mid] <= t:
                    l = mid
                else:
                    r = mid
            return C[:, l]                         # choose lower index

        self.m = conditional_cdf

        # -----   q for DR‑weighted robust CP   ---------------------------
        S_torch   = torch.from_numpy(S_train_np).to(self.device)
        W         = torch.from_numpy(self.w(X_train)[:-1]).to(self.device)
        inv_g_val = self._invg(1 - self.alpha, self.rho)

        self.q = self._dr_quant(S_torch,
                                W,
                                self.m,
                                X_train,
                                X_Q,
                                inv_g_val)

        # stash U as torch for later use
        self.u = torch.from_numpy(U_cal).to(self.device)
    """

    def _fit_m(self, X_train, y_train, calib_scores, X_Q):
        """Fit m(t|X) and pre‑compute DR‑quantile with dtype‑consistent float64."""
        # -----------------------------------------------------------
        U_cal = np.random.uniform(size=X_train.shape[0]).astype(np.float64)

        S_train_np = (calib_scores if isinstance(calib_scores, np.ndarray)
                    else calib_scores.cpu().numpy()).astype(np.float64)

        X_train64 = X_train.astype(np.float64)
        X_Q64     = X_Q.astype(np.float64)
        # -----------------------------------------------------------
        qrf = base_forest.QuantileRegressionForest()
        qrf.fit(X_train64, S_train_np)        # all float64 now
        # -----------------------------------------------------------
        y_sorted = np.sort(S_train_np)
        N        = len(y_sorted) - 1

        def conditional_cdf(x_raw, t):
            C = qrf.predict_C_CDF(x_raw.astype(np.float64))
            l, r = 0, N
            while r - l > 1:
                mid = (l + r) // 2
                if y_sorted[mid] <= t:
                    l = mid
                else:
                    r = mid
            return C[:, l]

        self.m = conditional_cdf
        # -----------------------------------------------------------
        S_torch = torch.from_numpy(S_train_np).to(self.device)
        W_np    = self.w(X_train64)[:-1].astype(np.float64)
        W       = torch.from_numpy(W_np).to(self.device)

        inv_g_val = self._invg(1 - self.alpha, self.rho)

        # self.q = self._dr_quant(S_torch, W, self.m, X_train64, X_Q64, inv_g_val)
        self.u = torch.from_numpy(U_cal).to(self.device)

    # ---------------------------------------------------------------------
    #   Divergence inverse g_{f,ρ}^{‑1}
    # ---------------------------------------------------------------------
    def _invg(self, r, rho):
        eps = 1e-10
        if r > 1:
            return torch.tensor(1.0, device=self.device)
        left, right = r, 1.0
        f = self.f
        while right - left > eps:
            mid = (left + right) / 2
            ans = mid * f(r / mid) + (1 - mid) * f((1 - r) / (1 - mid))
            if ans <= rho:
                left = mid
            else:
                right = mid
        return torch.tensor((left + right) / 2, device=self.device)


    # ---------------------------------------------------------------------
    #   Interval construction
    # ---------------------------------------------------------------------
    def get_interval(self, X_q: np.ndarray | torch.Tensor, calib_scores, interval_type: str):
        """
        interval_type ∈ {"0","1","2","3","4"}
        """

        # calibration scores S_i and corresponding weights ---------------
        X_dim  = self.sample.shape[1] - 1
        S_cal  = calib_scores 
                            
        # ensure torch tensor on device
        S_cal = (S_cal if isinstance(S_cal, torch.Tensor)
                 else torch.from_numpy(S_cal)).to(self.device)
        
        S_all = torch.cat([S_cal, self.smax.unsqueeze(0)])

        if interval_type == "0":        # standard CP
            w_all = torch.ones_like(S_all, device=self.device)
            q = self._quantile_weighted(S_all, w_all, 1 - self.alpha)

        elif interval_type == "1":      # weighted CP
            w_np  = self.w(X_q)                           # NumPy
            w_all = torch.from_numpy(
                     np.concatenate([w_np, [0.0]])).to(self.device)
            w_all = w_all / w_all.sum()
            q = self._quantile_weighted(S_all, w_all, 1 - self.alpha)

        elif interval_type == "2":      # robust CP
            w_all = torch.ones_like(S_all, device=self.device)
            q = self._quantile_weighted(S_all,
                                        w_all,
                                        self._invg(1 - self.alpha,
                                                   self.shiftrho))

        elif interval_type == "3":      # weighted robust CP
            w_np  = self.w(X_q)
            w_all = torch.from_numpy(
                    np.concatenate([w_np, [0.0]])).to(self.device)
            w_all = w_all / w_all.sum()
            q = self._quantile_weighted(S_all,
                                        w_all,
                                        self._invg(1 - self.alpha,
                                                   self.rho))

        elif interval_type == "4":      # DR weighted robust CP
            q = torch.tensor(self.q, device=self.device)

        else:
            raise ValueError("interval_type must be '0'..'4'")

        return q.item()


    # ---------------------------------------------------------------------
    #   One‑shot coverage / length evaluation
    # ---------------------------------------------------------------------
    def one_test(self, shift_sample, calib_scores, test_scores, interval_type="0"):

        q_val = self.get_interval(np.expand_dims(shift_sample, axis=0), calib_scores, interval_type)

        U_new = torch.rand(1, device=self.device)
        s_new = test_scores 
        cover = int(s_new <= q_val)

        if self.score_type == "aps":
            length = int(self.score(X_q, 0, self.model_u, U_new)[0] <= q_val) \
                   + int(self.score(X_q, 1, self.model_u, U_new)[0] <= q_val)
        else:
            length = 2 * q_val

        return cover, length


    # =====================================================================
    #                 -------  low‑level utilities  -------
    # =====================================================================
    @staticmethod
    def _cmr_score(x, y, model, u=None):
        if isinstance(x, torch.Tensor):
            x_t = x
        else:
            x_t = torch.from_numpy(x).float().to(next(model.parameters()).device)
        pred = model(x_t).squeeze()
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        return np.abs(y - pred)

    @staticmethod
    def _aps_score(x, y, model, u):
        probs = torch_predict_proba(model, x, next(model.parameters()).device)
        return (probs[:, 0] > probs[:, 1]) * ((1 - y) * (1 - u) * probs[:, 0] +
                                              y * (1 - u * probs[:, 1]))   \
             + (probs[:, 0] <= probs[:, 1]) * (y * (1 - u) * probs[:, 1] +
                                              (1 - y) * (1 - u * probs[:, 0]))

    @staticmethod
    def _quantile_weighted(s, w, beta):
        """
        s, w : torch tensors on the same device
        """
        sort_idx = torch.argsort(s)
        s_sorted = s[sort_idx]
        w_sorted = w[sort_idx]
        cumsum   = torch.cumsum(w_sorted, 0)
        thresh   = beta * cumsum[-1]
        idx      = torch.searchsorted(cumsum, thresh)
        return s_sorted[idx]

    # ---------------------------------------------------------------------
    def _dr_quant(self, S, W, m_func, X_P, X_Q, q_target):
        """
        Vectorised DR‑weighted robust CP quantile
        (simple scalar search – remains on CPU for the base_forest calls).
        """
        # move to CPU NumPy for m_func (base_forest) ----------------------
        S_np  = S.cpu().numpy()
        W_np  = W.cpu().numpy()

        def est_cov(t):
            if W_np.sum() <= 1e-5:
                return m_func(X_Q, t).mean()
            else:
                term1 = np.sum(W_np * ((S_np <= t) - m_func(X_P, t))) / W_np.sum()
                term2 = m_func(X_Q, t).mean()
                return term1 + term2

        t_grid = np.linspace(S_np.max(), 0, 1000)
        best_t = t_grid[-1]
        for t in t_grid:
            if est_cov(t) < q_target:
                break
            best_t = t
        return best_t

