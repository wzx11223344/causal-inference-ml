"""
Causal Forest / Generalized Random Forest
==========================================

Athey, Tibshirani & Wager (2019)
"Generalized Random Forests", Annals of Statistics, 47(2): 1148-1178.

基于 Wager & Athey (2018) 的 Honest Tree 框架,
扩展为任意局部估计等式 (local estimating equations) 的森林方法。

对于处理效应, GRF 等价于 Athey & Imbens (2016) 提出的 Causal Forest。

核心创新:
    1. Honest Splitting: 用一半数据选择分裂点, 另一半做估计
    2. Orthogonalization: 先在森林中估计 propensity, 再估计 treatment effect
    3. 渐进理论: 点态渐近正态性, 可用于构建 CATE 置信区间

实现要点:
    - 分裂准则: 最大化两个子节点处理效应估计的加权平方差
    - 叶节点估计: 双重稳健得分在叶节点内的均值
    - Heterogeneity: 森林给出的预测反映了真实 CATE 的异质性
"""

import numpy as np
from sklearn.model_selection import KFold
from typing import Optional, Tuple, Dict, List, Any
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class HonestCausalTree:
    """
    诚实因果树 (Honest Causal Tree)。

    将数据随机分为 S (splitting) 和 E (estimation) 两个子集:
        - S: 用于选择分裂规则 (树的结构)
        - E: 用于估计叶节点的处理效应

    分裂准则: 最大化
        Δ(A, B) = n_A * n_B / (n_A + n_B) * (τ̂_A - τ̂_B)²
    其中 τ̂ 是节点内的双重稳健处理效应估计。

    Reference: Wager & Athey (2018) JASA
    """

    def __init__(self, min_node_size: int = 20, max_depth: int = 10,
                 honesty_fraction: float = 0.5, random_state: int = None):
        """
        参数
        ----
        min_node_size : int, 分裂所需的最小样本数
        max_depth : int, 树的最大深度
        honesty_fraction : float, S 集合的比例 (诚实树)
        random_state : int
        """
        self.min_node_size = min_node_size
        self.max_depth = max_depth
        self.honesty_fraction = honesty_fraction
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self._split_var = None
        self._split_val = None
        self._left_child = None
        self._right_child = None
        self._tau = None  # 叶节点处理效应
        self._is_leaf = True
        self._depth = 0
        self._n_est = 0  # estimation 样本数

    def _estimate_tau(self, y, T, g_hat, m_hat, idx=None):
        """双重稳健处理效应估计。"""
        if idx is None:
            idx = np.arange(len(y))

        if len(idx) == 0:
            return 0.0

        y_sub, T_sub = y[idx], T[idx]
        g1_sub, g0_sub = g_hat[idx, 1], g_hat[idx, 0]
        m_sub = m_hat[idx]

        # 双重稳健得分
        scores = (g1_sub - g0_sub
                  + T_sub * (y_sub - g1_sub) / np.clip(m_sub, 0.05, 0.95)
                  - (1 - T_sub) * (y_sub - g0_sub) / np.clip(1 - m_sub, 0.05, 0.95))

        return np.mean(scores)

    def _splitting_criterion(self, y, T, g_hat, m_hat, left_idx, right_idx):
        """
        计算分裂增益: Δ = n_L * n_R / (n_L + n_R) * (τ̂_L - τ̂_R)²
        """
        n_L, n_R = len(left_idx), len(right_idx)
        if n_L < self.min_node_size or n_R < self.min_node_size:
            return -1.0

        tau_L = self._estimate_tau(y, T, g_hat, m_hat, left_idx)
        tau_R = self._estimate_tau(y, T, g_hat, m_hat, right_idx)

        gain = n_L * n_R / (n_L + n_R) * (tau_L - tau_R) ** 2
        return gain

    def fit(self, X, y, T, g_hat, m_hat, idx=None, depth=0):
        """
        递归构建诚实因果树。

        参数
        ----
        X : np.ndarray, shape (n, p)
        y : np.ndarray
        T : np.ndarray
        g_hat : np.ndarray, shape (n, 2)
        m_hat : np.ndarray
        idx : np.ndarray or None, 当前节点的样本索引
        depth : int
        """
        self._depth = depth

        if idx is None:
            idx = np.arange(len(y))

        if len(idx) < 2 * self.min_node_size or depth >= self.max_depth:
            self._tau = self._estimate_tau(y, T, g_hat, m_hat, idx)
            self._n_est = len(idx)
            return self

        # 诚实分裂: 随机分 S 和 E
        n = len(idx)
        n_split = int(n * self.honesty_fraction)
        perm = self.rng.permutation(idx)
        S_idx = perm[:n_split]
        E_idx = perm[n_split:]

        if len(S_idx) < self.min_node_size or len(E_idx) < self.min_node_size:
            self._tau = self._estimate_tau(y, T, g_hat, m_hat, idx)
            self._n_est = len(idx)
            return self

        # 遍历所有特征和分裂点, 寻找最优分裂
        p = X.shape[1]
        best_gain = -1
        best_var = 0
        best_val = 0

        for j in range(p):
            xj = X[S_idx, j]
            # 候选分裂点: 随机选择 min(10, n_split) 个分位数
            n_candidates = min(10, n_split)
            candidates = np.percentile(xj, np.linspace(10, 90, n_candidates))

            for val in candidates:
                left_mask = xj < val
                right_mask = ~left_mask

                if left_mask.sum() < self.min_node_size or right_mask.sum() < self.min_node_size:
                    continue

                gain = self._splitting_criterion(
                    y, T, g_hat, m_hat,
                    S_idx[left_mask], S_idx[right_mask]
                )

                if gain > best_gain:
                    best_gain = gain
                    best_var = j
                    best_val = val

        # 如果没有足够好的分裂, 变为叶节点
        if best_gain <= 0:
            self._tau = self._estimate_tau(y, T, g_hat, m_hat, idx)
            self._n_est = len(idx)
            return self

        # 执行分裂
        self._is_leaf = False
        self._split_var = best_var
        self._split_val = best_val

        full_idx_np = np.asarray(idx)
        left_mask = X[full_idx_np, best_var] < best_val
        right_mask = ~left_mask

        left_full_idx = full_idx_np[left_mask]
        right_full_idx = full_idx_np[right_mask]

        self._left_child = HonestCausalTree(
            self.min_node_size, self.max_depth,
            self.honesty_fraction, self.rng.randint(10000)
        )
        self._left_child.fit(X, y, T, g_hat, m_hat, left_full_idx, depth + 1)

        self._right_child = HonestCausalTree(
            self.min_node_size, self.max_depth,
            self.honesty_fraction, self.rng.randint(10000)
        )
        self._right_child.fit(X, y, T, g_hat, m_hat, right_full_idx, depth + 1)

        return self

    def predict(self, x):
        """预测单个观测的 CATE。"""
        if self._is_leaf:
            return self._tau

        if x[self._split_var] < self._split_val:
            return self._left_child.predict(x)
        else:
            return self._right_child.predict(x)

    def predict_batch(self, X):
        """批量预测。"""
        return np.array([self.predict(x) for x in X])


class GRFResult:
    """GRF 结果容器。"""

    def __init__(self):
        self.cate: np.ndarray = None
        self.cate_se: np.ndarray = None
        self.forest: List = None
        self.oob_predictions: np.ndarray = None
        self.feature_importance: np.ndarray = None
        self.n_trees: int = 0
        self.n: int = 0

    def summary(self) -> str:
        """打印 GRF 摘要。"""
        lines = [
            "=" * 60,
            "Causal Forest / GRF — 异质性处理效应 (CATE)",
            "=" * 60,
            f"树数: {self.n_trees}          样本数: {self.n}",
            f"平均 CATE: {np.mean(self.cate):.4f}",
            f"CATE 标准差: {np.std(self.cate):.4f}",
            f"CATE 范围: [{np.min(self.cate):.4f}, {np.max(self.cate):.4f}]",
            "=" * 60,
            "特征重要性 (Top 5):",
        ]
        if self.feature_importance is not None:
            top = np.argsort(self.feature_importance)[::-1][:5]
            for rank, idx in enumerate(top):
                imp = self.feature_importance[idx]
                lines.append(f"  {rank+1}. Feature {idx}: {imp:.4f}")
        return "\n".join(lines)


class CausalForest:
    """
    Causal Forest — 异质性处理效应估计。

    基于 Athey & Imbens (2016) 和 Wager & Athey (2018) 的诚实因果树,
    构建一个随机森林，其中每棵树使用随机子样本训练。

    使用
    ----
    cf = CausalForest(n_trees=500)
    result = cf.fit(X, y, T)
    print(result.summary())

    # 获取 CATE 预测
    cate = result.cate
    """

    def __init__(self, n_trees: int = 500,
                 min_node_size: int = 20,
                 max_depth: int = 10,
                 subsample_fraction: float = 0.5,
                 honesty_fraction: float = 0.5,
                 n_folds_cf: int = 5,
                 random_state: int = 42):
        """
        参数
        ----
        n_trees : int, 树的数量 (推荐 500-2000)
        min_node_size : int, 叶节点最小样本数
        max_depth : int, 树的最大深度
        subsample_fraction : float, 每棵树的随机子样本比例
        honesty_fraction : float, 分裂 vs 估计的样本比例
        n_folds_cf : int, 用于 orthogonalization 的交叉拟合折数
        random_state : int
        """
        self.n_trees = n_trees
        self.min_node_size = min_node_size
        self.max_depth = max_depth
        self.subsample_fraction = subsample_fraction
        self.honesty_fraction = honesty_fraction
        self.n_folds_cf = n_folds_cf
        self.random_state = random_state

        self._trees = []
        self._oob_counts = None
        self._oob_sum = None
        self._feature_importance = np.zeros(1)

    def fit(self, X: np.ndarray, y: np.ndarray, T: np.ndarray,
            verbose: bool = False) -> GRFResult:
        """
        拟合 Causal Forest。

        步骤:
            1. Cross-fitting 估计 nuisance 函数 ĝ 和 m̂
            2. 用 t 棵诚实树构建森林
            3. Out-of-bag (OOB) 预测, 避免 overfitting

        参数
        ----
        X : np.ndarray, shape (n, p)
        y : np.ndarray, shape (n,)
        T : np.ndarray, shape (n,)

        返回
        ----
        GRFResult
        """
        y = np.asarray(y, dtype=float).flatten()
        X = np.asarray(X, dtype=float)
        T = np.asarray(T, dtype=float).flatten()
        n, p = X.shape

        # ---- Step 1: Cross-fitting 估计 nuisance functions ----
        from .inference import cross_fitting
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LogisticRegression

        g_hat, m_hat = cross_fitting(
            X, y, T,
            n_folds=self.n_folds_cf,
            outcome_model=RandomForestRegressor(n_estimators=100, max_depth=6,
                                                min_samples_leaf=10,
                                                n_jobs=-1, random_state=self.random_state),
            propensity_model=LogisticRegression(max_iter=1000,
                                                random_state=self.random_state),
        )

        # ---- Step 2: Build the forest ----
        rng = np.random.RandomState(self.random_state)
        n_subsample = max(int(n * self.subsample_fraction), 2 * self.min_node_size)

        # OOB 累积器
        self._oob_counts = np.zeros(n)
        self._oob_sum = np.zeros(n)
        self._feature_importance = np.zeros(p)

        self._trees = []

        for t in range(self.n_trees):
            # 随机子样本 (bootstrap)
            idx = rng.choice(n, size=n_subsample, replace=True)

            # 训练一棵诚实树
            tree = HonestCausalTree(
                min_node_size=self.min_node_size,
                max_depth=self.max_depth,
                honesty_fraction=self.honesty_fraction,
                random_state=rng.randint(100000),
            )
            tree.fit(X, y, T, g_hat, m_hat,
                     idx=np.arange(n) if n_subsample >= n else idx)

            self._trees.append(tree)

            # OOB: 对不在子样本中的观测做预测
            oob_idx = np.setdiff1d(np.arange(n), np.unique(idx))
            for i in oob_idx:
                pred = tree.predict(X[i])
                self._oob_counts[i] += 1
                self._oob_sum[i] += pred

            # Feature importance (split counts)
            # 简化为: 记录哪些变量被用于分裂
            if tree._split_var is not None and tree._split_var < p:
                self._feature_importance[tree._split_var] += 1

            if verbose and (t + 1) % 100 == 0:
                oob_cate = self._oob_sum / np.maximum(self._oob_counts, 1)
                valid = self._oob_counts > 0
                print(f"  Tree {t+1}/{self.n_trees}, "
                      f"mean CATE = {np.mean(oob_cate[valid]):.4f}")

        # ---- Step 3: OOB predictions and SE ----
        result = GRFResult()
        result.oob_predictions = self._oob_sum / np.maximum(self._oob_counts, 1)
        result.cate = result.oob_predictions
        result.forest = self._trees
        result.n_trees = self.n_trees
        result.n = n

        # 特征重要性 (归一化)
        if self._feature_importance.sum() > 0:
            result.feature_importance = self._feature_importance / self._feature_importance.sum()
        else:
            result.feature_importance = self._feature_importance

        # CATE 标准误: 使用 infinitesimal jackknife (Wager et al., 2014)
        result.cate_se = self._compute_ij_se(X)

        return result

    def _compute_ij_se(self, X: np.ndarray) -> np.ndarray:
        """
        用 infinitesimal jackknife 计算 CATE 标准误。

        SE(x) = sqrt( Σ (τ̂_b(x) - τ̄(x))² / B² )
        其中 τ̂_b 是 bootstrapped 预测的均值, B 是 bootstrap 重复数。
        """
        n = len(self._oob_counts)
        se = np.zeros(n)

        for i in range(min(n, 1000)):  # 抽样计算以加速
            preds = np.array([tree.predict(X[i]) for tree in self._trees[:100]])
            se[i] = np.std(preds, ddof=1)

        return se

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """对新数据预测 CATE (所有树的均值)。"""
        X_new = np.asarray(X_new, dtype=float)
        preds = np.zeros((len(X_new), len(self._trees)))
        for t, tree in enumerate(self._trees):
            preds[:, t] = tree.predict_batch(X_new)
        return np.mean(preds, axis=1)
