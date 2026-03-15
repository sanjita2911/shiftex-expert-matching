import numpy as np
import torch
from alibi_detect.cd import MMDDrift


def alibi_mmd_predict(x_ref: np.ndarray, x_test: np.ndarray, alpha: float = 0.05) -> dict:
    x_ref = np.asarray(x_ref, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)

    cd = MMDDrift(
        x_ref=x_ref,
        backend="pytorch",
        p_val=alpha,
        configure_kernel_from_x_ref=True
    )
    return cd.predict(x_test)


def compute_mmd_distance(x_ref: np.ndarray, x_test: np.ndarray, alpha: float = 0.05) -> float:
    res = alibi_mmd_predict(x_ref, x_test, alpha=alpha)
    return float(res["data"]["distance"])


def median_heuristic_sigma(
    arrays: list[np.ndarray],
    max_points: int = 1024,
    seed: int = 0,
) -> float:
    """
    Compute one global RBF bandwidth from pooled embeddings.
    """
    rng = np.random.default_rng(seed)
    pooled = np.concatenate(arrays, axis=0)
    n = pooled.shape[0]
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        pooled = pooled[idx]

    x = torch.from_numpy(np.asarray(pooled, dtype=np.float32))
    dists = torch.cdist(x, x, p=2.0)
    tri = dists[torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)]
    med = float(torch.median(tri).item())
    if (not np.isfinite(med)) or med <= 0:
        return 1.0
    return med


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    d2 = torch.cdist(x, y, p=2.0) ** 2
    return torch.exp(-d2 / (2.0 * sigma * sigma))


def mmd_rbf_unbiased(
    x_ref: np.ndarray,
    x_test: np.ndarray,
    sigma: float,
    device: str = "cpu",
) -> float:
    """
    Unbiased MMD^2 with fixed RBF sigma. Returns sqrt(clamp(MMD^2, 0)).
    """
    x = torch.from_numpy(np.asarray(x_ref, dtype=np.float32)).to(device)
    y = torch.from_numpy(np.asarray(x_test, dtype=np.float32)).to(device)

    n = int(x.shape[0])
    m = int(y.shape[0])
    if n < 2 or m < 2:
        return float("nan")

    k_xx = _rbf_kernel(x, x, sigma)
    k_yy = _rbf_kernel(y, y, sigma)
    k_xy = _rbf_kernel(x, y, sigma)

    sum_kxx = (k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1))
    sum_kyy = (k_yy.sum() - k_yy.diag().sum()) / (m * (m - 1))
    sum_kxy = k_xy.mean()

    mmd2 = sum_kxx + sum_kyy - 2.0 * sum_kxy
    mmd2 = torch.clamp(mmd2, min=0.0)
    return float(torch.sqrt(mmd2).detach().cpu().item())
