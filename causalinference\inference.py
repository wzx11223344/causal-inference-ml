"""
推断与评分函数模块
==================

实现了因果推断的核心统计工具:

1. Cross-Fitting — K-fold 样本分割，消除 overfitting 偏差
2. Neyman 正交得分 — DML 的数学基础
3. 双重稳健得分 (Doubly Robust)
4. Bootstrap 置信区间
5. 排列检验 (Permutation Test)

核心理论:
    - Neyman 正交性: E[∂ψ/∂η] = 0, 其中 η 为 nuisance 函数
    - 交叉拟合使 Donsker 条件不再需要
    - 双重稳健: 只要倾向得分模型 OR 结果模型正确, ATE 就是一致的
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from typing import Callable, Optional, Tuple, Any
from functools import partial


# ============================================================
# 1. Cross-Fitting
# ============================================================

def cross_fitting(X: np.ndarray, y: np.ndarray, T: np.ndarray,
                  n_folds: int = 5,
                  outcome_model: Any = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
                  propensity_model: Any = LogisticRegression(max_iter=1000, random_state=42)) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-Fold 交叉拟合: 为每个观测生成 out-of-fold 预测。

    这是 DML 的核心步骤。通过对每个 fold 使用其他 fold 训练的模型
    进行预测，避免了 overfitting 引入的偏差。

    参数
    ----
    X : np.ndarray, shape (n, p)
    y : np.ndarray, shape (n,)
    T : np.ndarray, shape (n,)
        二值处理变量 (0/1)。
    n_folds : int
    outcome_model : sklearn estimator
        预测 y 的模型。
    propensity_model : sklearn estimator
        预测倾向得分 P(T=1|X) 的模型 (二分类器)。

    返回
    ----
    g_hat : np.ndarray, shape (n, 2)
        每列的 out-of-fold outcome 预测。列 0 = g(0, X), 列 1 = g(1, X)。
    m_hat : np.ndarray, shape (n,)
        out-of-fold 倾向得分预测。
    """
    n = len(y)
    X = np.asarray(X, dtype=float)
    T_flat = np.asarray(T, dtype=float).flatten()

    g_hat = np.zeros((n, 2))
    m_hat = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        T_train = T_flat[train_idx]

        # ---- 1. 预测 E[Y | T=t, X] ----
        try:
            model_0 = clone_model(outcome_model)
            mask_0 = T_train == 0
            if mask_0.sum() > 5:
                model_0.fit(X_train[mask_0], y_train[mask_0])
                g_hat[test_idx, 0] = model_0.predict(X_test)
            else:
                g_hat[test_idx, 0] = np.mean(y_train[mask_0])

            model_1 = clone_model(outcome_model)
            mask_1 = T_train == 1
            if mask_1.sum() > 5:
                model_1.fit(X_train[mask_1], y_train[mask_1])
                g_hat[test_idx, 1] = model_1.predict(X_test)
            else:
                g_hat[test_idx, 1] = np.mean(y_train[mask_1])
        except Exception:
            g_hat[test_idx, 0] = np.mean(y_train[T_train == 0]) if (T_train == 0).sum() > 0 else 0
            g_hat[test_idx, 1] = np.mean(y_train[T_train == 1]) if (T_train == 1).sum() > 0 else 0

        # ---- 2. 预测倾向得分 P(T=1|X) ----
        try:
            pm = clone_model(propensity_model)
            pm.fit(X_train, T_train)
            m_hat[test_idx] = pm.predict_proba(X_test)[:, 1]
        except Exception:
            m_hat[test_idx] = np.mean(T_train)

    # 截断倾向得分，避免极端值
    m_hat = np.clip(m_hat, 0.025, 0.975)

    return g_hat, m_hat


def clone_model(model):
    """Produce a fresh clone of an sklearn model."""
    from sklearn.base import clone
    return clone(model)


# ============================================================
# 2. Neyman 正交得分 / 双重稳健
# ============================================================

def doubly_robust_score(y: np.ndarray, T: np.ndarray,
                        g_hat: np.ndarray, m_hat: np.ndarray) -> np.ndarray:
    r"""
    双重稳健 (Doubly Robust / AIPW) 得分函数。

    基于 Neyman 正交的 ATE 估计量:
        ψ_i = g(1, X_i) - g(0, X_i)
            + T_i * (Y_i - g(1, X_i)) / m(X_i)
            - (1 - T_i) * (Y_i - g(0, X_i)) / (1 - m(X_i))

    估计量: θ̂ = (1/n) Σ ψ_i

    参数
    ----
    y : np.ndarray, shape (n,)
    T : np.ndarray, shape (n,)
    g_hat : np.ndarray, shape (n, 2)
        两个 outcome 模型的预测。
    m_hat : np.ndarray, shape (n,)
        倾向得分。

    返回
    ----
    np.ndarray, shape (n,) — 逐观测的 DR 得分。
    """
    T_flat = np.asarray(T, dtype=float).flatten()

    # 直接效应: g(1,X) - g(0,X)
    direct = g_hat[:, 1] - g_hat[:, 0]

    # 残差修正项
    correction = (T_flat * (y - g_hat[:, 1]) / m_hat
                  - (1 - T_flat) * (y - g_hat[:, 0]) / (1 - m_hat))

    return direct + correction


def ate_inference(psi: np.ndarray, alpha: float = 0.05) -> dict:
    """
    基于 Neyman 正交得分的推断。

    返回
    ----
    dict with ate, se, t_stat, p_value, ci_lower, ci_upper
    """
    n = len(psi)
    ate = np.mean(psi)
    se = np.sqrt(np.var(psi, ddof=1) / n)
    t_stat = ate / se
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(np.abs(t_stat)))
    z_alpha = norm.ppf(1 - alpha / 2)

    return {
        "ate": float(ate),
        "se": float(se),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ci_lower": float(ate - z_alpha * se),
        "ci_upper": float(ate + z_alpha * se),
        "n": n,
    }


# ============================================================
# 3. Bootstrap & Permutation
# ============================================================

def bootstrap_ci(estimator_fn: Callable, data_args: tuple,
                 n_bootstrap: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> dict:
    """
    非参数 Bootstrap 置信区间。

    参数
    ----
    estimator_fn : callable, 返回标量
        估计函数 estimator_fn(y, T, X) -> float。
    data_args : tuple, (y, T, X)
    n_bootstrap : int
    alpha : float

    返回
    ----
    dict
    """
    y, T, X = data_args
    n = len(y)

    estimates = np.zeros(n_bootstrap)
    rng = np.random.RandomState(seed)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        estimates[b] = estimator_fn(y[idx], T[idx], X[idx])

    return {
        "mean": float(np.mean(estimates)),
        "se": float(np.std(estimates, ddof=1)),
        "ci_lower": float(np.percentile(estimates, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(estimates, 100 * (1 - alpha / 2))),
    }


def causal_bootstrap(dr_score_fn: Callable, psi: np.ndarray,
                     n_bootstrap: int = 2000, alpha: float = 0.05,
                     seed: int = 42) -> dict:
    """
    因果推断专用 Bootstrap: 对 Neyman 正交得分进行重抽样。

    参数
    ----
    dr_score_fn : callable
        双重稳健得分函数。
    psi : np.ndarray
        原始 Neyman 正交得分。
    n_bootstrap, alpha : int, float
    """
    n = len(psi)
    estimates = np.zeros(n_bootstrap)
    rng = np.random.RandomState(seed)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        estimates[b] = np.mean(psi[idx])

    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)
    se = np.std(estimates, ddof=1)

    return {
        "ate": float(np.mean(estimates)),
        "se": float(se),
        "ci_lower": float(np.percentile(estimates, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(estimates, 100 * (1 - alpha / 2))),
        "ci_normal_lower": float(np.mean(estimates) - z_alpha * se),
        "ci_normal_upper": float(np.mean(estimates) + z_alpha * se),
    }


def permutation_test(y: np.ndarray, T: np.ndarray, X: np.ndarray,
                     estimator_fn: Callable, n_perm: int = 2000,
                     seed: int = 42) -> dict:
    """
    排列检验: 通过随机置换处理分配构建虚无分布。

    H0: ATE = 0 (无处理效应)

    参数
    ----
    estimator_fn : callable(y, T, X) -> float
    n_perm : int
    """
    n = len(y)
    T_flat = np.asarray(T, dtype=float).flatten()
    obs_effect = estimator_fn(y, T_flat, X)

    null_effects = np.zeros(n_perm)
    rng = np.random.RandomState(seed)

    for p in range(n_perm):
        T_perm = T_flat.copy()
        rng.shuffle(T_perm)
        null_effects[p] = estimator_fn(y, T_perm, X)

    p_value = np.mean(np.abs(null_effects) >= np.abs(obs_effect))

    return {
        "observed_effect": float(obs_effect),
        "p_value": float(p_value),
        "null_mean": float(np.mean(null_effects)),
        "null_std": float(np.std(null_effects)),
        "significant_at_5pct": bool(p_value < 0.05),
    }


# ============================================================
# 4. Summary Table
# ============================================================

def summary_table(results: dict, method_name: str = "DML") -> str:
    """生成 ATE 估计结果汇总表。"""
    lines = [
        "=" * 56,
        f"  {method_name} — 平均处理效应 (ATE) 估计",
        "=" * 56,
        f"  ATE     : {results.get('ate', float('nan')):>10.4f}",
        f"  Std.Err.: {results.get('se', float('nan')):>10.4f}",
        f"  t-stat  : {results.get('t_stat', float('nan')):>10.4f}",
        f"  p-value : {results.get('p_value', float('nan')):>10.4f}",
        f"  95% CI  : [{results.get('ci_lower', 0):>8.4f}, "
        f"{results.get('ci_upper', 0):>8.4f}]",
        f"  N       : {results.get('n', 0):>10d}",
        "=" * 56,
    ]
    if results.get('p_value', 1) < 0.01:
        lines.append("  *** p < 0.01 — 极强的统计显著")
    elif results.get('p_value', 1) < 0.05:
        lines.append("  **  p < 0.05 — 统计显著")
    elif results.get('p_value', 1) < 0.10:
        lines.append("  *   p < 0.10 — 边缘显著")
    else:
        lines.append("      不显著")

    return "\n".join(lines)
