"""
CausalInference — 前沿因果推断与机器学习融合引擎
================================================

实现了近年计量经济学最前沿的因果推断方法，将机器学习
与经典计量理论深度融合。

核心组件:
    - Double/Debiased Machine Learning (Chernozhukov et al., 2018)
    - Causal Forest / Generalized Random Forest (Athey et al., 2019)
    - Meta-Learners: S/T/X-Learner (Künzel et al., 2019)
    - Neyman 正交得分 + K-fold 交叉拟合 + 双重稳健推断
    - 异质性处理效应 (HTE) + 条件平均处理效应 (CATE)

论文来源:
    - Chernozhukov et al. (2018) Double/Debiased ML, Econometrics Journal
    - Athey, Tibshirani & Wager (2019) Generalized Random Forests, AoS
    - Künzel et al. (2019) Metalearners, PNAS
    - Chernozhukov et al. (2022) DML with Riesz Representers
"""

from .dml import DoubleML, DMLResult
from .causal_forest import CausalForest, GRFResult
from .metalearners import SLearner, TLearner, XLearner
from .inference import (
    bootstrap_ci, doubly_robust_score, cross_fitting,
    causal_bootstrap, permutation_test, summary_table
)

__version__ = "1.0.0"
__author__ = "wzx11223344"
