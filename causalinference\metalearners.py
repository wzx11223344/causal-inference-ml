"""
Meta-Learners: S-Learner, T-Learner, X-Learner
===============================================

Künzel, Sekhon, Bickel & Yu (2019)
"Metalearners for estimating heterogeneous treatment effects
using machine learning", PNAS, 116(10): 4156-4165.

三种使用任意 ML 模型估计处理效应的方法:

1. S-Learner (Single-model):
   用单一模型预测 E[Y | T, X], 处理效应 = f(1, X) - f(0, X)
   优点: 简单; 缺点: 当 T 的效应被淹没时表现差

2. T-Learner (Two-model):
   分别在处理组和对照组训练两个模型, τ̂(x) = μ̂₁(x) - μ̂₀(x)
   优点: 灵活; 缺点: 处理组和对照组样本量失衡时不稳定

3. X-Learner (Cross-model):
   用交叉信息增强小样本组的估计
   步骤:
       a. 估计 μ̂₀ 和 μ̂₁ (T-Learner)
       b. 计算"伪效应" D̂_i^0 = μ̂₁(X_i) - Y_i (for control) etc.
       c. 在伪效应上训练 τ̂₀(x) 和 τ̂₁(x)
       d. CATE = g(x) τ̂₀(x) + (1-g(x)) τ̂₁(x)
   优点: 在小样本处理组中表现优异
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from typing import Any, Optional, Dict
from .inference import clone_model, cross_fitting


class SLearner:
    """
    S-Learner: 单一模型学习器。

    训练一个模型 f(T, X) = E[Y | T, X],
    处理效应: τ̂(x) = f(1, x) - f(0, x)
    """

    def __init__(self, base_model: Any = None, random_state: int = 42):
        self.base_model = base_model or RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=random_state, n_jobs=-1)
        self.random_state = random_state
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, T: np.ndarray) -> "SLearner":
        """
        在 [T|X] 上训练单一模型。

        参数
        ----
        X : np.ndarray, shape (n, p)
        y : np.ndarray, shape (n,)
        T : np.ndarray, shape (n,)
        """
        X = np.asarray(X, dtype=float)
        T_flat = np.asarray(T, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float).flatten()

        XT = np.hstack([X, T_flat])

        self._model = clone_model(self.base_model)
        self._model.fit(XT, y)
        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """预测 CATE。"""
        X_new = np.asarray(X_new, dtype=float)
        n = len(X_new)

        XT1 = np.hstack([X_new, np.ones((n, 1))])
        XT0 = np.hstack([X_new, np.zeros((n, 1))])

        return self._model.predict(XT1) - self._model.predict(XT0)


class TLearner:
    """
    T-Learner: 双模型学习器。

    分别训练 μ̂₁(x) = E[Y(1) | X] 和 μ̂₀(x) = E[Y(0) | X],
    处理效应: τ̂(x) = μ̂₁(x) - μ̂₀(x)
    """

    def __init__(self, treatment_model: Any = None,
                 control_model: Any = None,
                 random_state: int = 42):
        self.treatment_model = treatment_model or RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=random_state, n_jobs=-1)
        self.control_model = control_model or RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=random_state + 1, n_jobs=-1)
        self.random_state = random_state
        self._model_1 = None
        self._model_0 = None

    def fit(self, X: np.ndarray, y: np.ndarray, T: np.ndarray) -> "TLearner":
        """
        分别对处理组和对照组拟合模型。

        参数
        ----
        X : np.ndarray, shape (n, p)
        y : np.ndarray, shape (n,)
        T : np.ndarray, shape (n,)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()
        T_flat = np.asarray(T, dtype=float).flatten()

        mask_1 = T_flat == 1
        mask_0 = T_flat == 0

        self._model_1 = clone_model(self.treatment_model)
        if mask_1.sum() > 10:
            self._model_1.fit(X[mask_1], y[mask_1])

        self._model_0 = clone_model(self.control_model)
        if mask_0.sum() > 10:
            self._model_0.fit(X[mask_0], y[mask_0])

        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """预测 CATE。"""
        mu1 = self._model_1.predict(X_new)
        mu0 = self._model_0.predict(X_new)
        return mu1 - mu0


class XLearner:
    """
    X-Learner: 交叉模型学习器。

    特别适合处理组样本量很小的情况。

    步骤:
        1. 估计 μ̂₀ 和 μ̂₁ (同 T-Learner)
        2. 为对照组计算伪效应: D̃ᵢ = μ̂₁(Xᵢ) - Yᵢ, 训练 τ̂₀(x)
           为处理组计算伪效应: D̃ᵢ = Yᵢ - μ̂₀(Xᵢ), 训练 τ̂₁(x)
        3. CATE(x) = g(x) τ̂₀(x) + (1-g(x)) τ̂₁(x)
           其中 g(x) 是倾向得分

    Reference: Künzel et al. (2019), PNAS
    """

    def __init__(self,
                 outcome_model: Any = None,
                 tau_model: Any = None,
                 propensity_model: Any = None,
                 random_state: int = 42):
        self.outcome_model = outcome_model or RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=random_state, n_jobs=-1)
        self.tau_model = tau_model or RandomForestRegressor(
            n_estimators=200, max_depth=6, random_state=random_state, n_jobs=-1)
        self.propensity_model = propensity_model or LogisticRegression(
            max_iter=1000, random_state=random_state)
        self.random_state = random_state

        self._mu1_model = None
        self._mu0_model = None
        self._tau1_model = None
        self._tau0_model = None
        self._prop_model = None

    def fit(self, X: np.ndarray, y: np.ndarray, T: np.ndarray) -> "XLearner":
        """
        拟合 X-Learner。

        参数
        ----
        X : np.ndarray, shape (n, p)
        y : np.ndarray, shape (n,)
        T : np.ndarray, shape (n,)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()
        T_flat = np.asarray(T, dtype=float).flatten()

        mask_1 = T_flat == 1
        mask_0 = T_flat == 0

        # Step 1: 训练 outcome 模型 (T-Learner)
        self._mu1_model = clone_model(self.outcome_model)
        self._mu1_model.fit(X[mask_1], y[mask_1])

        self._mu0_model = clone_model(self.outcome_model)
        self._mu0_model.fit(X[mask_0], y[mask_0])

        # Step 2a: 对照组伪效应 D̃ = μ̂₁(X) - Y (对照组中)
        pseudo_0 = self._mu1_model.predict(X[mask_0]) - y[mask_0]
        self._tau0_model = clone_model(self.tau_model)
        self._tau0_model.fit(X[mask_0], pseudo_0)

        # Step 2b: 处理组伪效应 D̃ = Y - μ̂₀(X) (处理组中)
        pseudo_1 = y[mask_1] - self._mu0_model.predict(X[mask_1])
        self._tau1_model = clone_model(self.tau_model)
        self._tau1_model.fit(X[mask_1], pseudo_1)

        # Step 3: 倾向得分模型 g(x) = P(T=1 | x)
        self._prop_model = clone_model(self.propensity_model)
        self._prop_model.fit(X, T_flat)

        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """预测 CATE。"""
        g = self._prop_model.predict_proba(X_new)[:, 1]

        tau0 = self._tau0_model.predict(X_new)
        tau1 = self._tau1_model.predict(X_new)

        # 加权组合
        return g * tau0 + (1 - g) * tau1
