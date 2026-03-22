# evaluation/robustness.py
import numpy as np

def apply_sensor_dropout(X: np.ndarray, p_drop: float = 0.05, seed: int = 42) -> np.ndarray:
    """
    X: (N, L, d) -> randomly set entries to 0 (or np.nan if you prefer)
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(X.shape) < p_drop
    X2 = X.copy()
    X2[mask] = 0.0
    return X2

def apply_gaussian_noise(X: np.ndarray, sigma: float = 0.01, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return X + rng.normal(0.0, sigma, size=X.shape)