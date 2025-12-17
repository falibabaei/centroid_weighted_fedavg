import numpy as np
from typing import Optional, Tuple
from nvflare.apis.fl_constant import FLMetaKey

def _global_centroid(C: np.ndarray, sizes: Optional[np.ndarray] = None, eps: float = 1e-8) -> np.ndarray:
    """
    Compute global centroid c_bar from client centroids C [M, d].
    If sizes provided, use size-weighted mean; else uniform.
    """
    C = np.asarray(C, dtype=np.float32)
    M = C.shape[0]
    if M == 0:
        raise ValueError("Empty centroid matrix C.")
    if sizes is None:
        w = np.ones(M, dtype=np.float32) / M
    else:
        w = np.asarray(sizes, dtype=np.float32)
        w = np.maximum(w, 0.0)
        s = float(w.sum())
        w = (w / (s + eps)) if s > 0 else np.ones(M, dtype=np.float32) / M
    return (C * w[:, None]).sum(axis=0)

def _distances_to_global(
    C: np.ndarray,
    sizes: Optional[np.ndarray] = None,
    mode: str = "euclid",
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute distances di from each client centroid to the global centroid.
    mode: 'euclid' (L2) or 'cosine'
    Returns (d, c_bar)
    """
    C = np.asarray(C, dtype=np.float32)
    c_bar = _global_centroid(C, sizes)
    if mode == "euclid":
        d = np.linalg.norm(C - c_bar[None, :], axis=1)
    elif mode == "cosine":
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + eps)
        cb = c_bar / (np.linalg.norm(c_bar) + eps)
        sim = Cn @ cb  # cosine similarity in [-1,1]
        d = 1.0 - sim  # cosine distance in [0,2]
    else:
        raise ValueError("mode must be 'euclid' or 'cosine'")
    return d.astype(np.float32), c_bar

def _huber_weight(d: np.ndarray, delta: float, eps: float = 1e-8) -> np.ndarray:
    """
    Robust IRLS-style Huber weight:
      w(d) = 1                  if d <= delta
           = delta / max(d,eps) if d > delta
    """
    d = np.asarray(d, dtype=np.float32)
    w = np.ones_like(d, dtype=np.float32)
    mask = d > delta
    w[mask] = float(delta) / (d[mask] + eps)
    return w

def trimming_weights(
    C: np.ndarray,
    sizes: Optional[np.ndarray] = None,
    percentile: float = 80.0,
    mode: str = "euclid",              # 'euclid' or 'cosine'
    scheme: str = "hard",              # 'hard' | 'soft_inv' | 'soft_huber'
    gamma: float = 1.0,                # size exponent for base weights
    delta: Optional[float] = None,     # Huber threshold; if None uses tau
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute aggregation weights with trimming.

    Inputs:
      C: centroids matrix [M, d]
      sizes: sample counts [M] or None (uniform base)
      percentile: percentile to set tau (threshold) from distances (e.g., 80.0)
      mode: distance mode ('euclid' or 'cosine')
      scheme:
        - 'hard': exclude clients with d_i > tau
        - 'soft_inv': weight_i ∝ size_w_i / (1 + d_i / tau)
        - 'soft_huber': weight_i ∝ size_w_i * HuberWeight(d_i; delta)
      gamma: base size weight exponent (size_w ∝ sizes^gamma)
      delta: Huber threshold; if None, uses tau
    Returns:
      weights: normalized weights [M] summing to 1 (excluded clients get 0)
      distances: distances d_i [M]
      tau: threshold used
    """
    C = np.asarray(C, dtype=np.float32)
    M = C.shape[0]
    if M == 0:
        raise ValueError("Empty centroid matrix C.")

    # Base size weights
    if sizes is None:
        size_w = np.ones(M, dtype=np.float32)
    else:
        size_w = np.asarray(sizes, dtype=np.float32)
        size_w = np.maximum(size_w, 0.0)
    size_w = (size_w ** float(gamma)).astype(np.float32)

    # Distances to global centroid and threshold tau
    d, _ = _distances_to_global(C, sizes=sizes, mode=mode, eps=eps)
    # Guard: if all distances are zero, make tau positive to avoid divide-by-zero
    tau = float(np.percentile(d, percentile)) if d.size > 0 else 0.0
    if tau <= 0.0:
        tau = float(np.max(d)) + 1e-6

    weights = np.zeros(M, dtype=np.float32)

    if scheme == "hard":
        # Include only clients with d_i <= tau
        mask = d <= tau
        if np.any(mask):
            w = size_w.copy()
            w[~mask] = 0.0
            s = float(w.sum())
            weights = (w / (s + eps)) if s > 0 else np.zeros_like(w)
        else:
            # Fallback: include the closest client
            k = int(np.argmin(d))
            weights[k] = 1.0

    elif scheme == "soft_inv":
        # weights ∝ size_w / (1 + d / tau)
        denom = 1.0 + (d / (tau + eps))
        w = size_w / denom
        s = float(w.sum())
        weights = (w / (s + eps)) if s > 0 else np.ones(M, dtype=np.float32) / M

    elif scheme == "soft_huber":
        # weights ∝ size_w * huber_weight(d; delta)
        delt = float(delta) if (delta is not None and delta > 0) else tau
        hw = _huber_weight(d, delt, eps=eps)
        w = size_w * hw
        s = float(w.sum())
        weights = (w / (s + eps)) if s > 0 else np.ones(M, dtype=np.float32) / M

    else:
        raise ValueError("scheme must be one of: 'hard', 'soft_inv', 'soft_huber'")

    return weights.astype(np.float32), d, tau

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Mock centroids for 5 clients in 3D
    C = np.array([
        [0.1, 0.2, 0.3],
        [0.11, 0.19, 0.31],
        [2.0, -1.0, 0.5],   # an outlier
        [0.12, 0.18, 0.29],
        [0.09, 0.21, 0.28],
    ], dtype=np.float32)
    sizes = np.array([5000, 4000, 6000, 3000, 2000], dtype=np.float32)

    # Hard trimming at 80th percentile
    w_hard, d_hard, tau_hard = trimming_weights(C, sizes, percentile=80.0, mode="euclid", scheme="hard", gamma=1.0)
    print("Hard weights:", w_hard, "tau:", tau_hard, "d:", d_hard)

    # Soft inverse trimming
    w_soft_inv, d_inv, tau_inv = trimming_weights(C, sizes, percentile=80.0, mode="euclid", scheme="soft_inv", gamma=1.0)
    print("Soft-inv weights:", w_soft_inv, "tau:", tau_inv, "d:", d_inv)

    # Soft Huber trimming
    w_soft_huber, d_huber, tau_huber = trimming_weights(C, sizes, percentile=80.0, mode="euclid", scheme="soft_huber", gamma=1.0, delta=None)
    print("Soft-huber weights:", w_soft_huber, "tau:", tau_huber, "d:", d_huber)

import numpy as np

def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def compute_weights_from_centroid_drift(models, gamma_size: float = 0.5, min_share: float = 0.01, max_share: float = 0.20):
    """
    models: list of FLModel-like objects with:
      - m.params: dict(layer_name -> weights list/np arrays)  (not used here)
      - m.meta: dict with keys:
          'centroid_delta': list/np.ndarray (feature drift vector),
          'num_examples': int (client data size)
    Returns: list of weights alpha_i (sum to 1)
    """
    drifts = []
    nis = []
    weights=[]

    for m in models:
        d = m.meta.get("centroid_delta", None)
        n = float(m.meta.get("num_examples", 0))
        w=m.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND,1)
        if d is None:
            # if we already have a reference shape, use zeros_like; else default to 1-dim zero
            d_arr = np.zeros_like(drifts[0]) if len(drifts) > 0 else np.zeros((1,), dtype=np.float64)
        else:
            d_arr = np.asarray(d, dtype=np.float64).reshape(-1)

        drifts.append(d_arr)
        nis.append(n)
       
        

    # Ensure all drift vectors have the same dimensionality
    max_dim = max(d.shape[0] for d in drifts) if drifts else 1
    drifts = [np.pad(d, (0, max_dim - d.shape[0])) if d.shape[0] != max_dim else d for d in drifts]

    drifts = [np.asarray(d, dtype=np.float64) for d in drifts]
    nis = np.asarray(nis, dtype=np.float64)

    # Magnitudes and Gaussian kernel (sigma = median magnitude, guarded from zero)
    mags = np.asarray([np.linalg.norm(d) for d in drifts], dtype=np.float64)
    sigma = float(np.median(mags))
    if sigma <= 0.0:
        sigma = 1e-12
    kernels = np.exp(-(mags ** 2) / (2.0 * sigma * sigma))

    # Alignment with leave-one-out mean drift
    sum_drifts = np.sum(drifts, axis=0) if len(drifts) > 0 else np.zeros((max_dim,), dtype=np.float64)
    alignments = []
    M = len(drifts)
    for i, d in enumerate(drifts):
        if M > 1:
            loo_mean = (sum_drifts - d) / (M - 1)
        else:
            loo_mean = sum_drifts
        s = _cosine(d, loo_mean)
        alignments.append(max(0.0, s))  # gate negative alignment
    alignments = np.asarray(alignments, dtype=np.float64)

    # Raw weights
    raw = (nis ** gamma_size) * kernels * alignments
    if np.all(raw <= 0):
        raw = (nis ** gamma_size)

    # Normalize
    total = float(raw.sum())
    if total > 0:
        alphas = raw / total
    else:
        alphas = np.ones_like(raw) / max(len(raw), 1)

    # Clip shares and renormalize
    alphas = np.clip(alphas, min_share, max_share)
    
    s = float(alphas.sum())
    if s > 0:
        alphas = (alphas / s) 
    else:
        alphas = (np.ones_like(alphas) / max(len(alphas), 1)) 
    
    return alphas.tolist()    