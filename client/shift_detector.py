import numpy as np
from typing import Optional, Tuple
from common.mmd import alibi_mmd_predict


class ShiftDetector:
    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        self.baseline_embeddings: Optional[np.ndarray] = None
        self.current_expert_id: Optional[str] = None

    def detect_shift(self, new_embeddings: np.ndarray) -> Tuple[bool, float, float]:
        """
        Returns:
            (is_drift, p_val, distance)
        """
        if self.baseline_embeddings is None:
            return False, 1.0, 0.0

        x_ref = np.asarray(self.baseline_embeddings, dtype=np.float32)
        x_test = np.asarray(new_embeddings, dtype=np.float32)

        res = alibi_mmd_predict(x_ref, x_test, alpha=self.alpha)

        is_drift = bool(res["data"]["is_drift"])
        p_val = float(res["data"]["p_val"])
        dist = float(res["data"]["distance"])
        return is_drift, p_val, dist

    def update_baseline(self, embeddings: np.ndarray):
        self.baseline_embeddings = np.asarray(
            embeddings, dtype=np.float32).copy()

    def set_expert(self, expert_id: str):
        self.current_expert_id = expert_id

    def get_expert(self) -> Optional[str]:
        return self.current_expert_id

    def get_alpha(self) -> float:
        return self.alpha

    def set_alpha(self, alpha: float):
        self.alpha = float(alpha)

    def reset(self):
        self.baseline_embeddings = None
        self.current_expert_id = None

    def has_baseline(self) -> bool:
        return self.baseline_embeddings is not None

    def get_baseline_shape(self) -> Optional[tuple]:
        return None if self.baseline_embeddings is None else self.baseline_embeddings.shape
