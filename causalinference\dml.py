"""
Double/Debiased Machine Learning (DML)
======================================

Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins (2018)
"Double/Debiased Machine Learning for Treatment and Structural Parameters"
The Econometrics Journal, 21(1): C1-C68.

核心直觉:
    1. 用 ML 估计 nuisance 函数 g(X) = E[Y|T,X] 和 m(X) = P(T=1|X)
    2. 构造 Neyman 正交得分，消除 ML 偏差的一阶效应
    3. 用 K-fold 交叉拟合避免 overfitting
    4. 结果: √n-一致的渐近正态估计量，允许 ML 以非参数速率收敛

这是过去十年因果推断最重要的方法论突破之一。
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold
from typing import Optional, Any, Dict
from .inference import (
    cross_fitting, doubly_robust_score, ate_inference,
    bootstrap_ci, permutation_test, summary_table,
    clone_model, causal_bootstrap
)


class DMLResult:
    """DML 估计结果容器。"""

    def __init__(self):
        self.ate: float = None
        self.se: float = None
        self.t_stat: float = None
        self.p_value: float = None
        self.ci_lower: float = None
        self.ci_upper: float = None
        self.n: int = 0
        self.psi: np.ndarray = None
        self.bootstrap: Dict = None
        self.perm_test: Dict = None
        self.dr_scores: np.ndarray = None  # CATE proxies
        self.model_type: str = "DML"

    def summary(self) -> str:
        return summary_table({
            "ate": self.ate, "se": self.se, "t_stat": self.t_stat,
            "p_value": self.p_value, "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper, "n": self.n,
        }, method_name=self.model_type)


class DoubleML:
    r"""
    Double/Debiased Machine Learning 估计器。

    估计目标: ATE = E[Y(1) - Y(0)]

    三步法:
        1. K-fold 交叉拟合: 估计 g(T, X) 和 m(X)
        2. Neyman 正交得分: ψ = g(1,X) - g(0,X) + (T(Y-g(1,X))/m) - ((1-T)(Y-g(0,X))/(1-m))
        3. 推断: θ̂ = (1/n) Σ ψ_i ~ N(θ, Var(ψ)/n)

    使用
    ----
    dml = DoubleML()
    result = dml.fit(X, y, T)
    print(result.summary())
    """

    def __init__(self,
                 outcome_model: Any = None,
                 propensity_model: Any = None,
                 n_folds: int = 5,
                 treatment_is_ate: bool = True,
                 random_state: int = 42):
        """
        参数
        ----
        outcome_model : sklearn estimator, optional
            E[Y | T, X] 的预测模型。默认 RandomForestRegressor。
        propensity_model : sklearn classifier, optional
            P(T=1 | X) 的预测模型。默认 LogisticRegression。
        n_folds : int, 交叉拟合折数 (推荐 2 或 5)
        treatment_is_ate : bool
            True: 估计 ATE (平均处理效应)。未来可扩展 ATT。
        random_state : int
        """
        self.outcome_model = outcome_model or RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            random_state=random_state, n_jobs=-1)
        self.propensity_model = propensity_model or LogisticRegression(
            max_iter=2000, random_state=random_state)
        self.n_folds = n_folds
        self.ate_type = treatment_is_ate
        self.random_state = random_state

        self._g_hat = None
        self._m_hat = None
        self._psi = None

    def fit(self, X: np.ndarray, y: np.ndarray, T: np.ndarray,
            do_bootstrap: bool = True, n_bootstrap: int = 2000,
            do_permutation_test: bool = False) -> DMLResult:
        """
        拟合 DML 模型。

        参数
        ----
        X : np.ndarray, shape (n, p)
            协变量矩阵。
        y : np.ndarray, shape (n,)
            结果变量。
        T : np.ndarray, shape (n,)
            二值处理变量 (0/1)。
        do_bootstrap : bool
            是否运行 Bootstrap 以获得稳健标准误。
        n_bootstrap : int
        do_permutation_test : bool
            是否运行排列检验。

        返回
        ----
        DMLResult
        """
        y = np.asarray(y, dtype=float).flatten()
        X = np.asarray(X, dtype=float)
        T = np.asarray(T, dtype=float).flatten()
        n = len(y)

        # ---- Step 1: K-fold Cross-Fitting ----
        self._g_hat, self._m_hat = cross_fitting(
            X, y, T,
            n_folds=self.n_folds,
            outcome_model=clone_model(self.outcome_model),
            propensity_model=clone_model(self.propensity_model),
        )

        # ---- Step 2: Neyman 正交得分 ----
        self._psi = doubly_robust_score(y, T, self._g_hat, self._m_hat)

        # ---- Step 3: 推断 ----
        inf = ate_inference(self._psi)

        result = DMLResult()
        result.ate = inf["ate"]
        result.se = inf["se"]
        result.t_stat = inf["t_stat"]
        result.p_value = inf["p_value"]
        result.ci_lower = inf["ci_lower"]
        result.ci_upper = inf["ci_upper"]
        result.n = n
        result.psi = self._psi
        result.dr_scores = self._psi  # 逐观测 CATE proxy

        # Bootstrap
        if do_bootstrap:
            result.bootstrap = causal_bootstrap(None, self._psi,
                                                n_bootstrap=n_bootstrap,
                                                seed=self.random_state)

        # Permutation test
        if do_permutation_test:
            def ate_estimator(y_i, T_i, X_i):
                dm = DoubleML(outcome_model=clone_model(self.outcome_model),
                              propensity_model=clone_model(self.propensity_model),
                              n_folds=self.n_folds,
                              random_state=self.random_state)
                res = dm.fit(X_i, y_i, T_i, do_bootstrap=False)
                return res.ate

            result.perm_test = permutation_test(
                y, T, X, ate_estimator, seed=self.random_state)

        return result

    def cate(self, X_new: Optional[np.ndarray] = None) -> np.ndarray:
        """
        条件平均处理效应 (CATE) 的代理估计。

        CATE(x) ≈ g(1, x) - g(0, x)

        注意: 这是 "plug-in" CATE，可能包含了 ML 偏差。
        更严格的 CATE 需要使用 Causal Forest。

        返回
        ----
        np.ndarray
        """
        raise NotImplementedError(
            "CATE 估计请使用 CausalForest。"
            "DML 适用于 ATE，Causal Forest 适用于 CATE。"
        )
