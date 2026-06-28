"""Quick validation test."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from causalinference import DoubleML

np.random.seed(42)
n = 500
X = np.random.randn(n, 3)
T = (np.random.rand(n) < 0.4).astype(float)
tau = 1.5
y = X[:, 0] + tau * T + np.random.randn(n) * 0.5

print("DML test...")
dml = DoubleML(n_folds=3)
r = dml.fit(X, y, T, do_bootstrap=False)
print(f"ATE: {r.ate:.4f} (true=1.5), SE: {r.se:.4f}, t: {r.t_stat:.2f}")
assert abs(r.ate - 1.5) < 0.5, f"DML ATE too far off: {r.ate}"
print("DML OK")

print("\nCausal Forest test...")
from causalinference import CausalForest
cf = CausalForest(n_trees=30, min_node_size=30, max_depth=5)
r2 = cf.fit(X, y, T)
cate = r2.cate
print(f"Mean CATE: {np.mean(cate):.4f} (true ATE=1.5)")
print(f"CATE SD: {np.std(cate):.4f}")
print("CF OK")

print("\nMeta-Learners test...")
from causalinference import SLearner, TLearner, XLearner
for name, learner in [("S", SLearner()), ("T", TLearner()), ("X", XLearner())]:
    learner.fit(X, y, T)
    c = learner.predict(X)
    print(f"{name}-Learner: mean CATE={np.mean(c):.4f}")
print("Meta OK")

print("\nAll tests passed!")
