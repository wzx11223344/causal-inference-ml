"""
Causal ML 综合示例
===================

演示前沿因果推断方法:
1. Double/Debiased Machine Learning (ATE 估计)
2. Causal Forest (CATE 异质性)
3. S-Learner / T-Learner / X-Learner 对比
4. Bootstrap & Permutation Test 推断
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

np.random.seed(42)

# ==================== 生成模拟数据 ====================
# 数据生成过程包含:
#   - 异质性处理效应: τ(x) = 2 + 0.5 * x1  (x1 越大, 效应越大)
#   - 非线性混淆: x2 影响 T 和 Y
#   - 多维噪声: x3-p 不相关

n = 2000
p = 8

X = np.random.randn(n, p)

# 处理分配: T ~ Bernoulli(logistic(0.5*x2))
propensity = 1 / (1 + np.exp(-0.5 * X[:, 1]))
T = (np.random.rand(n) < propensity).astype(float)

# 异质性处理效应: τ(x) = 2.0 + 0.5 * x1
tau = 2.0 + 0.5 * X[:, 0]

# 结果: Y = baseline + τ * T + noise
baseline = 1.0 + 0.8 * X[:, 1] + 0.3 * X[:, 2]
y = baseline + tau * T + np.random.randn(n) * 0.5

print(f"生成数据: n={n}, p={p}")
print(f"真实 ATE = {np.mean(tau):.4f}")
print(f"CATE 范围: [{np.min(tau):.4f}, {np.max(tau):.4f}]")
print(f"处理组比例: {np.mean(T):.3f}")

# ==================== 1. Double ML ====================
print("\n" + "=" * 60)
print("1. Double/Debiased Machine Learning (Chernozhukov 2018)")
print("=" * 60)

from causalinference import DoubleML

dml = DoubleML(n_folds=5)
result_dml = dml.fit(X, y, T, do_bootstrap=True)
print(result_dml.summary())

print(f"\nBootstrap CI: [{result_dml.bootstrap['ci_lower']:.4f}, "
      f"{result_dml.bootstrap['ci_upper']:.4f}]")

# ==================== 2. Causal Forest ====================
print("\n" + "=" * 60)
print("2. Causal Forest / GRF (Athey et al. 2019)")
print("=" * 60)

from causalinference import CausalForest

cf = CausalForest(n_trees=50, min_node_size=50, max_depth=6)
result_cf = cf.fit(X, y, T)
print(result_cf.summary())

# 检验 CATE 预测质量: CATE 应随 x1 递增
cate_cf = result_cf.cate
x1_bins = np.digitize(X[:, 0], bins=np.linspace(-2, 2, 6))
for b in range(1, 6):
    mask = x1_bins == b
    if mask.sum() > 0:
        print(f"  x1 bin {b}: mean CATE = {np.mean(cate_cf[mask]):.4f} "
              f"(true ≈ {2.0 + 0.5 * np.mean(X[mask, 0]):.4f})")

# ==================== 3. Meta-Learners 对比 ====================
print("\n" + "=" * 60)
print("3. Meta-Learners 对比")
print("=" * 60)

from causalinference import SLearner, TLearner, XLearner
from causalinference.inference import summary_table

# S-Learner
s_learner = SLearner()
s_learner.fit(X, y, T)
cate_s = s_learner.predict(X)

# T-Learner
t_learner = TLearner()
t_learner.fit(X, y, T)
cate_t = t_learner.predict(X)

# X-Learner
x_learner = XLearner()
x_learner.fit(X, y, T)
cate_x = x_learner.predict(X)

print(f"\n{'Method':>12s}  {'Mean CATE':>10s}  {'CATE SD':>10s}  {'Corr with True':>15s}")
print("-" * 52)
for name, cate in [("S-Learner", cate_s), ("T-Learner", cate_t),
                    ("X-Learner", cate_x), ("CF (OOB)", cate_cf)]:
    corr = np.corrcoef(cate, tau)[0, 1]
    print(f"{name:>12s}  {np.mean(cate):>10.4f}  {np.std(cate):>10.4f}  {corr:>15.4f}")

# ==================== 4. Permutation Test ====================
print("\n" + "=" * 60)
print("4. 排列检验 (Null: ATE = 0)")
print("=" * 60)

from causalinference.inference import permutation_test

perm = permutation_test(y, T, X,
    lambda y_i, T_i, X_i: DoubleML(n_folds=2).fit(X_i, y_i, T_i, do_bootstrap=False).ate,
    n_perm=100)
print(f"  Observed ATE: {perm['observed_effect']:.4f}")
print(f"  p-value: {perm['p_value']:.4f}")
print(f"  Significant: {perm['significant_at_5pct']}")
