import numpy as np


def info_content(P: np.ndarray) -> np.ndarray:
    return -np.log2(P)


def I_sample(P: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    idx = np.random.choice(len(P), size=n, p=P)
    return values[idx]


def artificial_sample(
    values: list[tuple[int, int]], prob: np.ndarray, n_samples: int = 100
) -> list[tuple[int, int]]:
    sample = []
    for val, p in zip(values, prob, strict=True):
        sample.extend(int(n_samples * p) * [val])
    return sample


def entropy(P: np.ndarray, base: float = 2.0) -> float:
    P_pos = P[P > 0]
    return -((P_pos * np.log(P_pos)).sum() / np.log(base)).item()


def entropy_conditional(P_XY: np.ndarray, conditioned_idx: int) -> np.ndarray:
    # Marginal distribution of the conditioning variable
    P_cond = P_XY.sum(axis=conditioned_idx, keepdims=True)

    # Mask of valid entries
    mask = P_XY > 0

    # H(A|B) = - sum p(a,b) log p(a|b)
    return -np.sum(P_XY[mask] * np.log2((P_XY / P_cond)[mask]))


def mutual_info(P_XY: np.ndarray, base: float = 2.0) -> float:
    P_X = P_XY.sum(axis=1)
    P_Y = P_XY.sum(axis=0)
    P_XP_Y = np.outer(P_X, P_Y)

    # Mask of valid entries
    mask = P_XY > 0

    # H(X;Y) = sum p(x,y) log( p(x,y) / (p(x)p(y)) )
    return (np.sum(P_XY[mask] * np.log((P_XY / P_XP_Y)[mask])) / np.log(base)).item()


def cross_entropy(P: np.ndarray, Q: np.ndarray, base: float = 2.0) -> float:
    P_pos = P[P > 0]
    Q_pos = Q[P > 0]
    return -((P_pos * np.log(Q_pos)).sum() / np.log(base)).item()


def kl_divergence(P: np.ndarray, Q: np.ndarray, base: float = 2.0) -> float:

    assert P.shape == Q.shape, "P and Q must have the same shape"
    assert np.isclose(np.sum(P), 1), "P must sum to 1"
    assert np.isclose(np.sum(Q), 1), "Q must sum to 1"
    assert np.all(P[Q == 0] == 0), "Violation: Q==0 implies P must also be 0"

    # Mask of valid entries
    mask = P > 0

    # I(X;Y) = sum p(x,y) log( p(x,y) / (p(x)p(y)) )
    return (np.sum(P[mask] * np.log((P / Q)[mask])) / np.log(base)).item()
