from nvflare.fuel.utils.log_utils import center_message
import numpy as np
from typing import List, Optional, Tuple
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants
from nvflare.apis.fl_constant import FLMetaKey
from  .weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.workflows.base_fedavg import BaseFedAvg
from src.soft_hard_trimming import trimming_weights, compute_weights_from_centroid_drift

# ----------------------------
# Helpers: centroids & weights
# ----------------------------

def extract_centroids_and_sizes(results: List[FLModel]) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Return (valid_idx, C, sizes) for results that have a valid centroid.

    sizes: will use meta["FLMetaKey.NUM_STEPS_CURRENT_ROUND"] if present; otherwise fallback to 1.0
    """
    valid_idx: List[int] = []
    centroids: List[np.ndarray] = []
    sizes: List[float] = []

    for i, r in enumerate(results):
        meta = r.meta or {}
        c = meta.get("centroid", None)
        if c is None:
            continue
        c = np.asarray(c, dtype=np.float32).ravel()
        if c.size == 0 or not np.isfinite(c).all():
            continue

        n = meta.get('num_examples', 1.0)
        try:
            n = float(n)
        except Exception:
            n = 1.0

        valid_idx.append(i)
        centroids.append(c)
        sizes.append(n)

    if not centroids:
        return [], np.array([]), np.array([])

    d = min(c.shape[0] for c in centroids)  # ensure consistent dim
    C = np.stack([c[:d] for c in centroids], axis=0)  # [M, d]
    sizes_arr = np.asarray(sizes, dtype=np.float32)   # [M]
    return valid_idx, C, sizes_arr


def choose_sigma(distances: np.ndarray, mode: str = "median", eps: float = 1e-8) -> float:
    distances = np.asarray(distances, dtype=np.float32)
    if distances.size == 0:
        return 1.0
    if mode == "median":
        return float(np.median(distances) + eps)
    elif mode == "std":
        return float(np.std(distances) + eps)
    elif mode == "silverman":
        std = float(np.std(distances))
        q75, q25 = np.percentile(distances, [75, 25])
        iqr = float(q75 - q25)
        s = min(std, iqr / 1.34)
        n = max(len(distances), 2)
        return float(1.06 * s * (n ** (-1.0 / 5.0)) + eps)
    else:
        return float(np.median(distances) + eps)

''' 
def compute_alphas(
    C: np.ndarray,
    sizes: np.ndarray,
    mode: str = "euclid",
    gamma: float = 1.0,
    sigma: Optional[float] = None,
    lambda_blend: Optional[float] = None,#if not None, blend size-based and kernel-based weights
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute normalized centroid-based weights alpha (sum to 1)."""
    M = C.shape[0]
    if M == 0:
        return np.array([], dtype=np.float32)

    sizes = np.asarray(sizes, dtype=np.float32)
    sizes = np.maximum(sizes, 1.0)
    size_w = sizes ** float(gamma)
    size_w = size_w / (size_w.sum() + eps)

    c_bar = (C * size_w[:, None]).sum(axis=0)

    if mode == "euclid":
        d = np.linalg.norm(C - c_bar[None, :], axis=1)
        if sigma is None:
            sigma = choose_sigma(d, mode="median", eps=eps)
        k = np.exp(-(d ** 2) / (2.0 * (sigma ** 2)))
        kernel_w = k / (k.sum() + eps)
    elif mode == "cosine":
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + eps)
        cb = c_bar / (np.linalg.norm(c_bar) + eps)
        s = (Cn @ cb)
        beta = 5.0
        e = np.exp(beta * s)
        kernel_w = e / (e.sum() + eps)
    else:
        raise ValueError("mode must be 'euclid' or 'cosine'")

    if lambda_blend is None:
        raw = size_w * kernel_w #αi ∝ ni · ki
        alpha = raw / (raw.sum() + eps)
    else:
        lam = float(max(0.0, min(1.0, lambda_blend)))
        alpha = (1.0 - lam) * size_w + lam * kernel_w # (1−λ) wi + λ (ki / Σj kj) 
        alpha = alpha / (alpha.sum() + eps)

    return alpha.astype(np.float32)
'''    


# ----------------------------
# Controller with corrected aggregate_fn
# ----------------------------

class FedAvg(BaseFedAvg):
    """FedAvg controller that injects centroid-based weights into WeightedAggregationHelper."""

    def __init__(self, *args, mode="euclid", gamma: float = 1.0, sigma: Optional[float] = None, lambda_blend: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.gamma = float(gamma)
        self.sigma = sigma
        self.lambda_blend = lambda_blend

    def run(self) -> None:
        self.info(center_message("Start FedAvg."))
        model = self.load_model()
        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            self.info(center_message(message=f"Round {self.current_round} started.", boarder_str="-"))
            model.current_round = self.current_round

            clients = self.sample_clients(self.num_clients)
            results = self.send_model_and_wait(targets=clients, data=model)

            aggregate_results = self.aggregate(results, aggregate_fn=self.aggregate_fn)
            model = self.update_model(model, aggregate_results)
            self.save_model(model)

        self.info(center_message("Finished FedAvg."))

    def aggregate_fn(self, results: List[FLModel]) -> FLModel:
        if not results:
            raise ValueError("received empty results for aggregation.")

        # Compute centroid-based weights if all clients have valid centroids
        valid_idx, C, sizes = extract_centroids_and_sizes(results)
        all_have_centroids = (len(valid_idx) == len(results)) and (len(valid_idx) >= 2)
         
        #weights: List[float] = []
        if all_have_centroids:
          #  weights, d_hard, tau_hard = trimming_weights(C, sizes, percentile=75.0, mode='cosine', scheme="soft_inv", gamma=1.0)
          weights= compute_weights_from_centroid_drift(results)
         #   alphas = compute_alphas(
         #       C=C,
           #     sizes=sizes,
           #     mode=self.mode,
           #     gamma=self.gamma,
           #     sigma=self.sigma,
           #     lambda_blend=self.lambda_blend,
           # )
            # Map alphas to results order
           # alpha_map = {valid_idx[k]: float(alphas[k]) for k in range(len(valid_idx))}
           # weights = [alpha_map[i] for i in range(len(results))]
        else:
            # Fallback to size-based weights using num_examples or num_steps
            sizes_all = []
            for r in results:
                meta = r.meta or {}
                n = meta.get("num_examples", meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0))
                try:
                    n = float(n)
                except Exception:
                    n = 1.0
                sizes_all.append(n)
            sizes_all = np.asarray(sizes_all, dtype=np.float32)
            w = sizes_all / (sizes_all.sum() + 1e-8)
            weights = [float(x) for x in w]

        # Aggregate params
        aggr_helper = WeightedAggregationHelper()
        # Aggregate metrics (optional)
        aggr_metrics_helper = WeightedAggregationHelper()
        all_metrics = True

        for i, r in enumerate(results):
            meta = r.meta or {}
            contributor_name = meta.get("client_name", AppConstants.CLIENT_UNKNOWN)
            aggr_helper.add(
                data=r.params,
                weight=weights[i],
                contributor_name=contributor_name,
                contribution_round=r.current_round,
            )
            if not r.metrics:
                all_metrics = False
            if all_metrics:
                aggr_metrics_helper.add(
                    data=r.metrics,
                    weight=weights[i],
                    contributor_name=contributor_name,
                    contribution_round=r.current_round,
                )

        aggr_params = aggr_helper.get_result()
        aggr_metrics = aggr_metrics_helper.get_result() if all_metrics else None

        aggr_result = FLModel(
            params=aggr_params,
            params_type=results[0].params_type,
            metrics=aggr_metrics,
            meta={"nr_aggregated": len(results), "current_round": results[0].current_round},
        )
        return aggr_result