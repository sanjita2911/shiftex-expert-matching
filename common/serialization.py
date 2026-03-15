import io
from typing import Dict, Any, List, Tuple

import numpy as np
import torch


def serialize_state_dict(state_dict: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


def deserialize_state_dict(b: bytes) -> Dict[str, Any]:
    buf = io.BytesIO(b)
    return torch.load(buf, map_location="cpu")


def serialize_ndarray(arr: np.ndarray) -> Tuple[bytes, List[int]]:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue(), list(arr.shape)


def deserialize_ndarray(b: bytes) -> np.ndarray:
    buf = io.BytesIO(b)
    return np.load(buf, allow_pickle=False)
