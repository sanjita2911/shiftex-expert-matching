"""
data/cifar10c.py

Unified CIFAR-10 corruption data utilities.

This module merges the two historical code paths in this repo:
1) CIFAR-10-C (precomputed .npy files, 10k images per severity level)
   - Previously: data/cifar10c_loader.py
2) In-memory corruption of torchvision CIFAR-10 via our corruption functions
   - Previously: data/cifar10_corrupt_dataset.py

Keeping both is useful:
- CIFAR-10-C .npy is convenient for fast experiments over many corruptions.
- In-memory corruption matches the "4 corruption" expert setup and provides
  explicit train/val/test splits for early stopping.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from data.corruptions import apply_corruption

# ---------------------------------------------------------------------------
# CIFAR-10-C (.npy) loader
# ---------------------------------------------------------------------------


class CIFAR10CDataset(Dataset):
    """
    Loads one CIFAR-10-C corruption at a specific severity from .npy files.

    The CIFAR-10-C format stores 50k images per corruption:
    - 10k images for each severity in [1..5]
    - aligned with the CIFAR-10 test labels (labels.npy)
    """

    def __init__(
        self,
        root: str,
        corruption: str,
        severity: int,
        split: str,
        train_size: int = 8000,
        test_size: int = 2000,
        seed: int = 0,
        transform=None,
    ):
        if not (1 <= severity <= 5):
            raise ValueError("Severity must be in the range [1, 5]")
        if split not in {"train", "test"}:
            raise ValueError("Split must be 'train' or 'test'")

        self.transform = transform

        x_path = os.path.join(root, f"{corruption}.npy")
        y_path = os.path.join(root, "labels.npy")

        if not os.path.exists(x_path):
            raise FileNotFoundError(f"Missing corruption file: {x_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Missing labels file: {y_path}")

        X = np.load(x_path)
        y = np.load(y_path)

        start = (severity - 1) * 10000
        end = severity * 10000
        X = X[start:end]
        y = y[start:end]

        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))

        if split == "train":
            idx = perm[:train_size]
        else:
            idx = perm[train_size: train_size + test_size]

        self.X = X[idx]
        self.y = y[idx]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        img = self.X[idx]
        label = int(self.y[idx])

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, label


def make_loader(
    root: str,
    corruption: str,
    severity: int,
    split: str,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
    num_workers: int = 0,
    train_size: int = 8000,
    test_size: int = 2000,
    transform=None,
) -> DataLoader:
    """
    Builds a DataLoader for the CIFAR-10-C .npy dataset.
    Kept for backward compatibility with the existing scripts.
    """

    dataset = CIFAR10CDataset(
        root=root,
        corruption=corruption,
        severity=severity,
        split=split,
        train_size=train_size,
        test_size=test_size,
        seed=seed,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


# ---------------------------------------------------------------------------
# In-memory corruption loader (train/val/test splits)
# ---------------------------------------------------------------------------


DEFAULT_SEVERITY = 5
DEFAULT_TRAIN_SIZE = 40_000
DEFAULT_VAL_SIZE = 10_000


class CorruptedCIFAR10(Dataset):
    """
    Applies a corruption in memory at construction time and caches the result.

    Args:
        images     : uint8 numpy array, shape (N, 32, 32, 3)
        labels     : list or array of int labels, length N
        corruption : corruption name supported by data.corruptions.apply_corruption
        severity   : 1-5
        transform  : torchvision transform applied after corruption
        frost_dir  : path to frost overlay images (frost only)
        seed       : numpy random seed for reproducible corruptions
    """

    def __init__(
        self,
        images: np.ndarray,
        labels,
        corruption: str,
        severity: int = DEFAULT_SEVERITY,
        transform=None,
        frost_dir: str = "data/frost_images",
        seed: int = 0,
    ):
        super().__init__()
        self.transform = transform
        self.labels = list(labels)

        print(
            f"[CorruptedCIFAR10] Applying '{corruption}' severity={severity} "
            f"to {len(images)} images ..."
        )

        rng_state = np.random.get_state()
        np.random.seed(seed)

        corrupted = []
        for i, img in enumerate(images):
            from PIL import Image as PILImage

            pil = PILImage.fromarray(img)
            corrupted_img = apply_corruption(
                pil, corruption=corruption, severity=severity, frost_dir=frost_dir
            )
            corrupted.append(corrupted_img)

            if (i + 1) % 5000 == 0:
                print(f"  ... {i + 1}/{len(images)}")

        np.random.set_state(rng_state)

        self.data = np.stack(corrupted, axis=0).astype(np.uint8)
        print(f"[CorruptedCIFAR10] Done. Cached shape: {self.data.shape}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.data[idx]

        if self.transform is not None:
            from PIL import Image as PILImage

            img = self.transform(PILImage.fromarray(img))
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, self.labels[idx]


def make_loaders(
    corruption: str,
    cifar10_root: str = "data",
    batch_size: int = 128,
    seed: int = 0,
    num_workers: int = 4,
    frost_dir: str = "data/frost_images",
    splits: Optional[list] = None,
    severity: int = DEFAULT_SEVERITY,
    train_size: int = DEFAULT_TRAIN_SIZE,
    val_size: int = DEFAULT_VAL_SIZE,
) -> dict:
    """
    Build DataLoaders for one corruption type, optionally including train/val/test.

    This is the "expert training" path: corrupts CIFAR-10 in memory and returns
    explicit splits. When requesting only ["test"], it avoids corrupting the
    CIFAR-10 train images.
    """
    if splits is None:
        splits = ["train", "val", "test"]

    base_tfm = T.ToTensor()
    train_tfm = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )

    need_train_src = "train" in splits or "val" in splits
    need_test_src = "test" in splits

    raw_train = (
        torchvision.datasets.CIFAR10(root=cifar10_root, train=True, download=True)
        if need_train_src
        else None
    )
    raw_test = (
        torchvision.datasets.CIFAR10(root=cifar10_root, train=False, download=True)
        if need_test_src
        else None
    )

    result = {}

    if "train" in splits:
        train_ds = CorruptedCIFAR10(
            images=raw_train.data[:train_size],
            labels=raw_train.targets[:train_size],
            corruption=corruption,
            severity=severity,
            transform=train_tfm,
            frost_dir=frost_dir,
            seed=seed,
        )
        result["train"] = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        result["train_dataset"] = train_ds
        print(f"  [make_loaders] train : {len(train_ds):>6} images")

    if "val" in splits:
        val_ds = CorruptedCIFAR10(
            images=raw_train.data[train_size: train_size + val_size],
            labels=raw_train.targets[train_size: train_size + val_size],
            corruption=corruption,
            severity=severity,
            transform=base_tfm,
            frost_dir=frost_dir,
            seed=seed + 1,
        )
        result["val"] = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        result["val_dataset"] = val_ds
        print(f"  [make_loaders] val   : {len(val_ds):>6} images")

    if "test" in splits:
        test_ds = CorruptedCIFAR10(
            images=raw_test.data,
            labels=raw_test.targets,
            corruption=corruption,
            severity=severity,
            transform=base_tfm,
            frost_dir=frost_dir,
            seed=seed + 2,
        )
        result["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        result["test_dataset"] = test_ds
        print(f"  [make_loaders] test  : {len(test_ds):>6} images")

    return result


__all__ = [
    "CIFAR10CDataset",
    "make_loader",
    "CorruptedCIFAR10",
    "make_loaders",
]

