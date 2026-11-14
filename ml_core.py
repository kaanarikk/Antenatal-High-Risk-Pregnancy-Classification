import numpy as np
import pandas as pd
from scipy import optimize, stats

# ==============================================================================
# 1. METRICS, CALIBRATION & DECISION CURVE ANALYSIS
# ==============================================================================

def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = stats.rankdata(y_prob, method="average")
    u1 = np.sum(ranks[y_true == 1]) - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n0))

def roc_curve_points(y_true: np.ndarray, y_prob: np.ndarray):
    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[order]
    p_sorted = y_prob[order]
    distinct_idx = np.where(np.diff(p_sorted))[0]
    thresh_idx = np.r_[distinct_idx, y_sorted.size - 1]
    tps = np.cumsum(y_sorted)[thresh_idx]
    fps = (1 + thresh_idx) - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[p_sorted[0] + 1e-6, p_sorted[thresh_idx]]
    fpr = fps / max(fps[-1], 1)
    tpr = tps / max(tps[-1], 1)
    return fpr, tpr, thresholds

def pr_curve_points(y_true: np.ndarray, y_prob: np.ndarray):
    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[order]
    p_sorted = y_prob[order]
    distinct_idx = np.where(np.diff(p_sorted))[0]
    thresh_idx = np.r_[distinct_idx, y_sorted.size - 1]
    tps = np.cumsum(y_sorted)[thresh_idx]
    fps = (1 + thresh_idx) - tps
    prec = tps / np.maximum(tps + fps, 1)
    rec = tps / max(np.sum(y_true == 1), 1)
    prec = np.r_[1.0, prec]
    rec = np.r_[0.0, rec]
    return rec, prec

def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    rec, prec = pr_curve_points(y_true, y_prob)
    # Step-wise average precision
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))

def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_true - y_prob) ** 2))

def sens_spec_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5):
    pred = (y_prob >= thresh).astype(int)
    tp = np.sum((pred == 1) & (y_true == 1))
    tn = np.sum((pred == 0) & (y_true == 0))
    fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    acc = (tp + tn) / max(len(y_true), 1)
    return float(sens), float(spec), float(ppv), float(npv), float(acc)

def calibration_intercept_slope(y_true: np.ndarray, y_prob: np.ndarray):
    p = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    lp = np.log(p / (1.0 - p))
    if np.std(lp) < 1e-8 or len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    def nll(params):
        a, b = params
        z = np.clip(a + b * lp, -30, 30)
        q = 1.0 / (1.0 + np.exp(-z))
        return -np.sum(y_true * np.log(np.clip(q, 1e-12, 1)) + (1 - y_true) * np.log(np.clip(1 - q, 1e-12, 1))) + 1e-6 * (a**2 + b**2)
    res = optimize.minimize(nll, x0=[0.0, 1.0], method="BFGS")
    return float(res.x[0]), float(res.x[1])

def decision_curve_analysis(y_true: np.ndarray, y_prob: np.ndarray, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.90, 86)
    n = len(y_true)
    prev = np.mean(y_true)
    nb_model = []
    nb_all = []
    nb_none = np.zeros_like(thresholds)
    for pt in thresholds:
        pred = (y_prob >= pt).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        w = pt / (1.0 - pt)
        nb_model.append((tp / n) - (fp / n) * w)
        nb_all.append(prev - (1.0 - prev) * w)
    return thresholds, np.array(nb_model), np.array(nb_all), nb_none

# ==============================================================================
# 2. LOGISTIC REGRESSION (L2, MLE, AND FIRTH PENALIZED) & VIF
# ==============================================================================

def compute_vif(X: np.ndarray, feature_names):
    rows = []
    n, p = X.shape
    for j in range(p):
        y_j = X[:, j]
        X_other = np.delete(X, j, axis=1)
        X_design = np.column_stack([np.ones(n), X_other])
        beta, _, _, _ = np.linalg.lstsq(X_design, y_j, rcond=None)
        y_hat = X_design @ beta
        ss_res = np.sum((y_j - y_hat) ** 2)
        ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        vif = 1.0 / max(1.0 - r2, 1e-12)
        rows.append({"feature": feature_names[j], "R2": r2, "VIF": vif})
    return pd.DataFrame(rows)

def fit_firth_logistic(X: np.ndarray, y: np.ndarray, max_iter: int = 200, tol: float = 1e-6):
    """Firth's bias-reduced penalized logistic regression (handles complete separation)."""
    n, p = X.shape
    Xd = np.column_stack([np.ones(n), X])
    beta = np.zeros(p + 1)

    for _ in range(max_iter):
        eta = np.clip(Xd @ beta, -20, 20)
        pi = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(pi * (1.0 - pi), 1e-10)
        XtW = Xd.T * w
        I = XtW @ Xd + 1e-8 * np.eye(p + 1)
        try:
            I_inv = np.linalg.inv(I)
        except np.linalg.LinAlgError:
            I_inv = np.linalg.pinv(I)
        # Hat matrix diagonal h_i = w_i * x_i^T I^-1 x_i
        h = np.sum((Xd @ I_inv) * Xd, axis=1) * w
        # Modified score U* = X^T (y - pi + h * (0.5 - pi))
        U_star = Xd.T @ (y - pi + h * (0.5 - pi))
        step = I_inv @ U_star
        # Damped step if large
        step_norm = np.linalg.norm(step)
        if step_norm > 5.0:
            step = step * (5.0 / step_norm)
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    eta = np.clip(Xd @ beta, -20, 20)
    pi = 1.0 / (1.0 + np.exp(-eta))
    w = np.maximum(pi * (1.0 - pi), 1e-10)
    I = (Xd.T * w) @ Xd + 1e-8 * np.eye(p + 1)
    cov = np.linalg.pinv(I)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    z = beta / se
    pvals = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se
    return {
        "beta": beta,
        "se": se,
        "z": z,
        "pval": pvals,
        "or": np.exp(np.clip(beta, -20, 20)),
        "or_low": np.exp(np.clip(ci_low, -20, 20)),
        "or_high": np.exp(np.clip(ci_high, -20, 20)),
    }

class LogisticRegressionL2:
    def __init__(self, C: float = 1.0):
        self.C = C
        self.coef_ = None
        self.intercept_ = 0.0
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = np.where(X.std(axis=0) > 1e-8, X.std(axis=0), 1.0)
        Xs = (X - self.mean_) / self.std_
        n, p = Xs.shape
        reg = 1.0 / max(self.C, 1e-6)

        def loss_grad(w):
            b, beta = w[0], w[1:]
            z = np.clip(b + Xs @ beta, -30, 30)
            q = 1.0 / (1.0 + np.exp(-z))
            nll = -np.sum(y * np.log(np.clip(q, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-12, 1)))
            loss = nll + 0.5 * reg * np.sum(beta ** 2)
            diff = q - y
            gb = np.sum(diff)
            gbeta = Xs.T @ diff + reg * beta
            return loss, np.r_[gb, gbeta]

        res = optimize.minimize(loss_grad, np.zeros(p + 1), jac=True, method="L-BFGS-B")
        self.intercept_ = float(res.x[0])
        self.coef_ = res.x[1:].copy()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, dtype=float) - self.mean_) / self.std_
        z = np.clip(self.intercept_ + Xs @ self.coef_, -30, 30)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])

# ==============================================================================
# 3. CART DECISION TREE & RANDOM FOREST
# ==============================================================================

class _TreeNode:
    __slots__ = ("feature", "threshold", "left", "right", "prob", "n_samples", "n_pos", "impurity")
    def __init__(self, prob, n_samples, n_pos, impurity, feature=None, threshold=None, left=None, right=None):
        self.prob = prob
        self.n_samples = n_samples
        self.n_pos = n_pos
        self.impurity = impurity
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

class DecisionTreeClassifierScratch:
    def __init__(self, max_depth=4, min_samples_leaf=5, max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root_ = None
        self.n_features_ = None

    def _gini(self, n_pos, n_tot):
        if n_tot == 0:
            return 0.0
        p = n_pos / n_tot
        return 2.0 * p * (1.0 - p)

    def _best_split(self, X, y, rng):
        n, p = X.shape
        n_pos_tot = np.sum(y)
        parent_gini = self._gini(n_pos_tot, n)
        if parent_gini <= 1e-12 or n < 2 * self.min_samples_leaf:
            return None, None, 0.0

        if self.max_features is None or self.max_features >= p:
            feat_idx = np.arange(p)
        else:
            feat_idx = rng.choice(p, size=self.max_features, replace=False)

        best_gain = 0.0
        best_feat = None
        best_thresh = None

        for j in feat_idx:
            xj = X[:, j]
            order = np.argsort(xj, kind="mergesort")
            xs = xj[order]
            ys = y[order]
            # Vectorized prefix sums
            cum_pos = np.cumsum(ys)
            # Valid split indices where xs[i] < xs[i+1] and leaf size >= min_samples_leaf
            valid = np.where(
                (xs[:-1] < xs[1:]) &
                (np.arange(1, n) >= self.min_samples_leaf) &
                ((n - np.arange(1, n)) >= self.min_samples_leaf)
            )[0]
            if len(valid) == 0:
                continue
            n_left = valid + 1
            n_right = n - n_left
            pos_left = cum_pos[valid]
            pos_right = n_pos_tot - pos_left
            p_l = pos_left / n_left
            p_r = pos_right / n_right
            gini_l = 2.0 * p_l * (1.0 - p_l)
            gini_r = 2.0 * p_r * (1.0 - p_r)
            weighted_gini = (n_left * gini_l + n_right * gini_r) / n
            gains = parent_gini - weighted_gini
            idx_max = np.argmax(gains)
            if gains[idx_max] > best_gain:
                best_gain = float(gains[idx_max])
                best_feat = int(j)
                k = valid[idx_max]
                best_thresh = float(0.5 * (xs[k] + xs[k + 1]))
        return best_feat, best_thresh, best_gain

    def _build(self, X, y, depth, rng):
        n = len(y)
        n_pos = int(np.sum(y))
        prob = n_pos / max(n, 1)
        imp = self._gini(n_pos, n)
        if self.max_depth is not None and depth >= self.max_depth:
            return _TreeNode(prob, n, n_pos, imp)
        feat, thresh, gain = self._best_split(X, y, rng)
        if feat is None or gain <= 1e-9:
            return _TreeNode(prob, n, n_pos, imp)
        left_mask = X[:, feat] <= thresh
        left = self._build(X[left_mask], y[left_mask], depth + 1, rng)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1, rng)
        return _TreeNode(prob, n, n_pos, imp, feat, thresh, left, right)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_features_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        self.root_ = self._build(X, y, 0, rng)
        return self

    def _pred_row(self, node, x):
        while node.feature is not None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prob

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p1 = np.array([self._pred_row(self.root_, X[i]) for i in range(len(X))])
        return np.column_stack([1.0 - p1, p1])

    def extract_rules(self, feature_names):
        rules = []
        def dfs(node, conds):
            if node.feature is None:
                pred = "Yes" if node.prob >= 0.5 else "No"
                rules.append({
                    "rule": " AND ".join(conds) if conds else "ALL",
                    "pred": pred,
                    "prob_yes": node.prob,
                    "n": node.n_samples,
                    "n_yes": node.n_pos,
                    "purity": max(node.prob, 1.0 - node.prob),
                })
                return
            fname = feature_names[node.feature]
            dfs(node.left, conds + [f"{fname} <= {node.threshold:.2f}"])
            dfs(node.right, conds + [f"{fname} > {node.threshold:.2f}"])
        dfs(self.root_, [])
        return rules

class RandomForestClassifierScratch:
    def __init__(self, n_estimators=60, max_depth=8, min_samples_leaf=2, max_features="sqrt", random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, p = X.shape
        mf = max(1, int(np.sqrt(p))) if self.max_features == "sqrt" else self.max_features
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        for t in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=mf,
                random_state=int(self.random_state + t + 1)
            )
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.mean([t.predict_proba(X)[:, 1] for t in self.trees_], axis=0)
        return np.column_stack([1.0 - preds, preds])

# ==============================================================================
# 4. SECOND-ORDER GRADIENT BOOSTED TREES (XGBOOST-EQUIVALENT)
# ==============================================================================

class _GBDTNode:
    __slots__ = ("feature", "threshold", "left", "right", "weight")
    def __init__(self, weight=0.0, feature=None, threshold=None, left=None, right=None):
        self.weight = weight
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

class XGBoostScratch:
    def __init__(self, n_estimators=60, max_depth=4, learning_rate=0.1, reg_lambda=1.0, min_child_weight=1.0, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        self.trees_ = []
        self.base_score_ = 0.0

    def _build_tree(self, X, g, h, depth):
        G = np.sum(g)
        H = np.sum(h)
        weight = -G / (H + self.reg_lambda)
        if depth >= self.max_depth or len(g) < 2:
            return _GBDTNode(weight=float(weight))

        n, p = X.shape
        parent_score = (G ** 2) / (H + self.reg_lambda)
        best_gain = 0.0
        best_feat = None
        best_thresh = None

        for j in range(p):
            xj = X[:, j]
            order = np.argsort(xj, kind="mergesort")
            xs = xj[order]
            gs = g[order]
            hs = h[order]
            cum_g = np.cumsum(gs)
            cum_h = np.cumsum(hs)
            valid = np.where(
                (xs[:-1] < xs[1:]) &
                (cum_h[:-1] >= self.min_child_weight) &
                ((H - cum_h[:-1]) >= self.min_child_weight)
            )[0]
            if len(valid) == 0:
                continue
            GL = cum_g[valid]
            HL = cum_h[valid]
            GR = G - GL
            HR = H - HL
            gains = 0.5 * ((GL ** 2) / (HL + self.reg_lambda) + (GR ** 2) / (HR + self.reg_lambda) - parent_score)
            idx_max = np.argmax(gains)
            if gains[idx_max] > best_gain:
                best_gain = float(gains[idx_max])
                best_feat = int(j)
                k = valid[idx_max]
                best_thresh = float(0.5 * (xs[k] + xs[k + 1]))

        if best_feat is None or best_gain <= 1e-6:
            return _GBDTNode(weight=float(weight))

        mask = X[:, best_feat] <= best_thresh
        left = self._build_tree(X[mask], g[mask], h[mask], depth + 1)
        right = self._build_tree(X[~mask], g[~mask], h[~mask], depth + 1)
        return _GBDTNode(weight=float(weight), feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _pred_tree(self, node, X):
        out = np.empty(len(X), dtype=float)
        for i in range(len(X)):
            cur = node
            x = X[i]
            while cur.feature is not None:
                cur = cur.left if x[cur.feature] <= cur.threshold else cur.right
            out[i] = cur.weight
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        p0 = np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)
        self.base_score_ = float(np.log(p0 / (1.0 - p0)))
        f = np.full(len(y), self.base_score_, dtype=float)
        self.trees_ = []
        for _ in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-np.clip(f, -20, 20)))
            g = p - y
            h = np.maximum(p * (1.0 - p), 1e-6)
            tree = self._build_tree(X, g, h, 0)
            self.trees_.append(tree)
            f = f + self.learning_rate * self._pred_tree(tree, X)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        f = np.full(len(X), self.base_score_, dtype=float)
        for tree in self.trees_:
            f = f + self.learning_rate * self._pred_tree(tree, X)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(f, -20, 20)))
        return np.column_stack([1.0 - p1, p1])

# ==============================================================================
# 5. CYCLIC BOOSTED GAM / GA2M (EBM-EQUIVALENT)
# ==============================================================================

class EBMScratch:
    """Cyclic gradient-boosted Generalized Additive Model (main effects + top pairwise interactions)."""
    def __init__(self, n_rounds=40, learning_rate=0.08, reg_lambda=5.0, n_interactions=5, random_state=42):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.n_interactions = n_interactions
        self.random_state = random_state
        self.base_score_ = 0.0
        self.bins_ = {}
        self.main_effects_ = {}
        self.pair_effects_ = {}

    def _digitize(self, x, edges):
        if len(edges) == 0:
            return np.zeros(len(x), dtype=int)
        return np.digitize(x, edges, right=False)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        p0 = np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)
        self.base_score_ = float(np.log(p0 / (1.0 - p0)))
        f = np.full(n, self.base_score_, dtype=float)

        self.bins_ = {}
        self.main_effects_ = {}
        binned_X = np.zeros((n, p), dtype=int)
        for j in range(p):
            u = np.unique(X[:, j])
            if len(u) <= 12:
                edges = 0.5 * (u[:-1] + u[1:])
            else:
                qs = np.linspace(0, 100, 11)[1:-1]
                edges = np.unique(np.percentile(X[:, j], qs))
            self.bins_[j] = edges
            binned_X[:, j] = self._digitize(X[:, j], edges)
            self.main_effects_[j] = np.zeros(len(edges) + 1, dtype=float)

        # Cyclic boosting on 1D main effects
        for _ in range(self.n_rounds):
            for j in range(p):
                prob = 1.0 / (1.0 + np.exp(-np.clip(f, -20, 20)))
                g = prob - y
                h = np.maximum(prob * (1.0 - prob), 1e-6)
                b_idx = binned_X[:, j]
                n_b = len(self.main_effects_[j])
                step = np.zeros(n_b)
                for b in range(n_b):
                    mask = (b_idx == b)
                    if np.any(mask):
                        step[b] = -self.learning_rate * np.sum(g[mask]) / (np.sum(h[mask]) + self.reg_lambda)
                self.main_effects_[j] += step
                f += step[b_idx]

        # Select top pairwise interactions by residual chi-square/variance reduction
        self.pair_effects_ = {}
        if self.n_interactions > 0:
            prob = 1.0 / (1.0 + np.exp(-np.clip(f, -20, 20)))
            res = y - prob
            pair_scores = []
            for j1 in range(p):
                for j2 in range(j1 + 1, p):
                    # Fast binned interaction score
                    b1, b2 = binned_X[:, j1], binned_X[:, j2]
                    n2 = len(self.main_effects_[j2])
                    combo = b1 * n2 + b2
                    score = 0.0
                    for c in np.unique(combo):
                        m = (combo == c)
                        if np.sum(m) >= 5:
                            score += (np.sum(res[m]) ** 2) / np.sum(m)
                    pair_scores.append((score, j1, j2))
            pair_scores.sort(reverse=True)
            top_pairs = [(j1, j2) for _, j1, j2 in pair_scores[:self.n_interactions]]
            for (j1, j2) in top_pairs:
                n1 = len(self.main_effects_[j1])
                n2 = len(self.main_effects_[j2])
                self.pair_effects_[(j1, j2)] = np.zeros((n1, n2), dtype=float)
            for _ in range(max(10, self.n_rounds // 2)):
                for (j1, j2), grid in self.pair_effects_.items():
                    prob = 1.0 / (1.0 + np.exp(-np.clip(f, -20, 20)))
                    g = prob - y
                    h = np.maximum(prob * (1.0 - prob), 1e-6)
                    b1, b2 = binned_X[:, j1], binned_X[:, j2]
                    step = np.zeros_like(grid)
                    for i1 in range(grid.shape[0]):
                        m1 = (b1 == i1)
                        if not np.any(m1):
                            continue
                        for i2 in range(grid.shape[1]):
                            m = m1 & (b2 == i2)
                            if np.sum(m) >= 3:
                                step[i1, i2] = -self.learning_rate * np.sum(g[m]) / (np.sum(h[m]) + self.reg_lambda)
                    grid += step
                    f += step[b1, b2]
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        f = np.full(n, self.base_score_, dtype=float)
        binned_X = np.zeros((n, X.shape[1]), dtype=int)
        for j, edges in self.bins_.items():
            b = self._digitize(X[:, j], edges)
            binned_X[:, j] = b
            f += self.main_effects_[j][b]
        for (j1, j2), grid in self.pair_effects_.items():
            f += grid[binned_X[:, j1], binned_X[:, j2]]
        p1 = 1.0 / (1.0 + np.exp(-np.clip(f, -20, 20)))
        return np.column_stack([1.0 - p1, p1])

# ==============================================================================
# 6. REPEATED GROUP-KFOLD & KERNEL SHAPLEY VALUES
# ==============================================================================

def make_group_kfold(groups: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42):
    """Balanced GroupKFold: assigns unique groups to n_splits folds balancing size & positives."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    # Sort groups by size descending (with random tie-breaking via prior shuffle)
    grp_indices = {g: np.where(groups == g)[0] for g in unique_groups}
    sorted_groups = sorted(unique_groups, key=lambda g: len(grp_indices[g]), reverse=True)

    fold_sizes = np.zeros(n_splits, dtype=int)
    fold_pos = np.zeros(n_splits, dtype=int)
    fold_groups = [[] for _ in range(n_splits)]

    for g in sorted_groups:
        idx = grp_indices[g]
        # Pick fold with smallest current size (tie-break by smallest positives)
        best_f = min(range(n_splits), key=lambda f: (fold_sizes[f], fold_pos[f]))
        fold_groups[best_f].append(g)
        fold_sizes[best_f] += len(idx)
        fold_pos[best_f] += int(np.sum(y[idx]))

    splits = []
    for f in range(n_splits):
        test_mask = np.isin(groups, fold_groups[f])
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        splits.append((train_idx, test_idx))
    return splits

def compute_shapley_values(model, X_explain: np.ndarray, X_background: np.ndarray, n_samples: int = 80, seed: int = 42):
    """Monte Carlo Shapley values on predicted probability (exact marginal expectation)."""
    rng = np.random.default_rng(seed)
    n_exp, p = X_explain.shape
    n_bg = len(X_background)
    shap_vals = np.zeros((n_exp, p), dtype=float)
    base_val = float(np.mean(model.predict_proba(X_background)[:, 1]))

    for i in range(n_exp):
        x = X_explain[i]
        # Antithetic / stratified random permutations
        for _ in range(n_samples):
            perm = rng.permutation(p)
            bg_row = X_background[rng.integers(0, n_bg)]
            # Build path of p+1 points
            path = np.tile(bg_row, (p + 1, 1))
            for step, feat in enumerate(perm):
                path[step + 1:, feat] = x[feat]
            preds = model.predict_proba(path)[:, 1]
            diffs = np.diff(preds)
            shap_vals[i, perm] += diffs
        shap_vals[i] /= n_samples
    return shap_vals, base_val
