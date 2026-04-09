import numpy as np
from pyitlib import discrete_random_variable as drv


def H(X: np.ndarray, base: int = 2):
    return tuple(float(h) for h in drv.entropy(X, base=base))


def H_joint(vars: np.ndarray, base: int = 2):
    return float(drv.entropy_joint(vars, base=base))


def H_cond(Y_vars: np.ndarray, X_vars: np.ndarray, base: int = 2):
    XY = np.concatenate([X_vars, Y_vars])
    return H_joint(XY, base=base) - H_joint(X_vars, base=base)


def H_mutual(vars: np.ndarray, base: int = 2):
    return float(drv.information_co(vars, base=base))
