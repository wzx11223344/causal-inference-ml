# CausalInference — 前沿因果推断与机器学习融合引擎

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

实现了近年计量经济学/因果推断**最前沿**的方法论，将机器学习与经典计量理论深度融合。

> "Double/Debiased Machine Learning" 是 Chernozhukov et al. (2018) 提出的框架，
> 被广泛认为是 **过去十年因果推断最重要的方法论突破**。

---

## 🧬 核心算法

| 方法 | 论文 | 期刊 | 功能 |
|------|------|------|------|
| **Double ML** | Chernozhukov et al. (2018) | *Econometrics Journal* | √n-一致的 ATE 估计 |
| **Causal Forest** | Athey et al. (2019) | *Annals of Statistics* | 异质性处理效应 (CATE) |
| **S/T/X-Learner** | Künzel et al. (2019) | *PNAS* | Meta-Learner 框架 |

### 技术亮点

- 🎯 **Neyman 正交得分** — 消除 ML 偏差的一阶效应
- 🔀 **K-fold 交叉拟合** — 去除 Donsker 条件, 允许任意 ML 模型
- 🌲 **诚实因果树** — Half-sampling: 分裂与估计在不同数据上
- 🛡️ **双重稳健** — 倾向得分 OR 结果模型任一正确即可
- 📊 **异质性 CATE** — 发现谁从处理中获益最大
- 🎪 **排列检验** — Fisher 精确检验, 无需渐近近似

---

## 🚀 快速开始

```python
from causalinference import DoubleML, CausalForest
from causalinference import SLearner, TLearner, XLearner

# 1. Double ML — 稳健的 ATE 估计
dml = DoubleML(n_folds=5)
result = dml.fit(X, y, T, do_bootstrap=True)
print(result.summary())
# ATE = 2.05, p < 0.001, Neyman-orthogonal, Double-robust

# 2. Causal Forest — 异质性处理效应
cf = CausalForest(n_trees=500)
result = cf.fit(X, y, T)
cate = result.cate  # 每个观测的个性化处理效应

# 3. Meta-Learners 对比
s = SLearner().fit(X, y, T).predict(X)
t = TLearner().fit(X, y, T).predict(X)
x = XLearner().fit(X, y, T).predict(X)
```

## 🔬 算法原理

### Double/Debiased ML

```
Step 1: K-fold Cross-fitting → ĝ(X), m̂(X)  (nuisance functions)
Step 2: Neyman Orthogonal Score
    ψ = g(1,X) - g(0,X) + T(Y-g(1,X))/m(X) - (1-T)(Y-g(0,X))/(1-m(X))
Step 3: ATE = mean(ψ),  SE = std(ψ)/√n
```

### Honest Causal Tree

```
For each split candidate:
    分裂增益 Δ = n_L·n_R/(n_L+n_R) · (τ̂_L - τ̂_R)²
    其中 τ̂ 是节点内双重稳健处理效应
```

---

## 📖 参考文献

1. **Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018).** Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.

2. **Athey, S., Tibshirani, J., & Wager, S. (2019).** Generalized random forests. *Annals of Statistics*, 47(2), 1148-1178.

3. **Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019).** Metalearners for estimating heterogeneous treatment effects using machine learning. *PNAS*, 116(10), 4156-4165.

4. **Wager, S., & Athey, S. (2018).** Estimation and inference of heterogeneous treatment effects using random forests. *JASA*, 113(523), 1228-1242.

## 📄 许可证

MIT License © 2024 wzx11223344
