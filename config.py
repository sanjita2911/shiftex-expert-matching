"""
config.py

Single source of truth for all project constants.
Every script imports from here — no more hardcoded values scattered across files.

Usage:
    from config import DEVICE, CORRUPTIONS_4, CIFAR10C, TINYIMAGENETC
"""

import os
import torch

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# ---------------------------------------------------------------------------
# Corruptions
# ---------------------------------------------------------------------------

CORRUPTIONS_4 = [
    "gaussian_noise",
    "fog",
    "frost",
    "snow",
]

CORRUPTIONS_15 = [
    # Noise
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    # Blur
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    # Weather
    "fog",
    "frost",
    "snow",
    "brightness",
    # Digital
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

CIFAR10C = {
    "name":           "cifar10c",
    "router_ckpt":    "checkpoints/router_resnet50_cifar10.pt",
    "storage_dir":    "expert_storage",
    "data_root":      "data/cifar10c",
    "frost_dir":      "data/frost_images",
    "num_classes":    10,
    "image_size":     32,
    "severity":       5,
    "train_size":     40_000,
    "val_size":       10_000,
    # Training hyperparams
    "epochs":         50,
    "lr":             3e-4,
    "weight_decay":   5e-4,
    "label_smoothing": 0.1,
    "patience":       5,
}

TINYIMAGENETC = {
    "name":           "tinyimagenetc",
    "router_ckpt":    "checkpoints/router_resnet50_tinyimagenet.pt",
    "storage_dir":    "expert_storage_tinyimagenet",
    "data_root":      "data/tiny-imagenet-200",
    "frost_dir":      "data/frost_images",
    "num_classes":    200,
    "image_size":     64,
    "severity":       5,
    "train_size":     80_000,
    "val_size":       20_000,
    # Training hyperparams (router uses different lr)
    "epochs":         50,
    "lr":             3e-4,
    "weight_decay":   5e-4,
    "label_smoothing": 0.1,
    "patience":       5,
    "router_epochs":  15,
    "router_lr":      1e-4,
    "router_weight_decay": 1e-4,
}

DATASETS = {
    "cifar10c":      CIFAR10C,
    "tinyimagenetc": TINYIMAGENETC,
}

# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

BATCH_SIZE = 128
NUM_WORKERS = 2

# ---------------------------------------------------------------------------
# Embedding / MMD
# ---------------------------------------------------------------------------

MAX_SIG_SAMPLES = 10_000   # signature embeddings per expert
MAX_TEST_SAMPLES = 10_000   # test embeddings per corruption
SIGMA_POOL_PTS = 2048     # points used for median heuristic
N_SLICES = 1000     # slices for sliced Wasserstein

# ---------------------------------------------------------------------------
# Shift detection
# ---------------------------------------------------------------------------

SHIFT_ALPHA = 0.05          # p-value threshold for MMD drift test

# ---------------------------------------------------------------------------
# gRPC
# ---------------------------------------------------------------------------

GRPC_MAX_MSG = 200 * 1024 * 1024   # 200 MB
SERVER_HOST = "0.0.0.0"           # server listens on all interfaces
SERVER_PORT = 50051

# Clients use SERVER_ADDRESS to connect.
# - Local testing : localhost:50051        (default)
# - On Nautilus   : shiftex-server:50051  (injected via env var in k8s yaml)
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "localhost:50051")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def get_dataset_config(name: str) -> dict:
    """
    Returns the config dict for a dataset by name.

    Args:
        name: "cifar10c" or "tinyimagenetc"

    Returns:
        config dict

    Raises:
        ValueError if name is unknown
    """
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Choose from: {list(DATASETS.keys())}"
        )
    return DATASETS[name]
