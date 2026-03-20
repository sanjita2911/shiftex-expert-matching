from __future__ import annotations

import os
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image as PILImage

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from data.corruptions import apply_corruption


DEFAULT_SEVERITY = 5
DEFAULT_TRAIN_SIZE = 80_000
DEFAULT_VAL_SIZE = 20_000
IMG_SIZE = 64

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _list_train_classes(train_dir: str) -> List[str]:

    entries = []
    for name in sorted(os.listdir(train_dir)):
        path = os.path.join(train_dir, name)
        if os.path.isdir(path):
            entries.append(name)
    return entries


def _load_train_images(data_root: str) -> Tuple[np.ndarray, List[int]]:
    train_dir = os.path.join(data_root, "train")
    classes = _list_train_classes(train_dir)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    images: List[np.ndarray] = []
    labels: List[int] = []

    for cls in classes:
        img_dir = os.path.join(train_dir, cls, "images")
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            img = PILImage.open(os.path.join(img_dir, fname)).convert("RGB")
            images.append(np.array(img, dtype=np.uint8))
            labels.append(class_to_idx[cls])

    print(
        f"[_load_train_images] Loaded {len(images)} train images, {len(classes)} classes")
    return np.stack(images, axis=0), labels


def _load_val_images(data_root: str) -> Tuple[np.ndarray, List[int]]:
    val_dir = os.path.join(data_root, "val")
    img_dir = os.path.join(val_dir, "images")
    annot_file = os.path.join(val_dir, "val_annotations.txt")

    fname_to_synset = {}
    with open(annot_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                fname_to_synset[parts[0]] = parts[1]

    train_dir = os.path.join(data_root, "train")
    classes = _list_train_classes(train_dir)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    images: List[np.ndarray] = []
    labels: List[int] = []

    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
            continue
        synset = fname_to_synset.get(fname)
        if synset is None or synset not in class_to_idx:
            continue
        img = PILImage.open(os.path.join(img_dir, fname)).convert("RGB")
        images.append(np.array(img, dtype=np.uint8))
        labels.append(class_to_idx[synset])

    print(f"[_load_val_images] Loaded {len(images)} val images")
    return np.stack(images, axis=0), labels


class CorruptedTinyImageNet(Dataset):
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
            f"[CorruptedTinyImageNet] Applying '{corruption}' severity={severity} "
            f"to {len(images)} images ..."
        )

        rng_state = np.random.get_state()
        np.random.seed(seed)

        corrupted = []
        for i, img in enumerate(images):
            pil = PILImage.fromarray(img)
            corrupted_img = apply_corruption(
                pil, corruption=corruption, severity=severity, frost_dir=frost_dir
            )
            corrupted.append(corrupted_img)

            if (i + 1) % 10_000 == 0:
                print(f"  ... {i + 1}/{len(images)}")

        np.random.set_state(rng_state)

        self.data = np.stack(corrupted, axis=0).astype(np.uint8)
        print(f"[CorruptedTinyImageNet] Done. Cached shape: {self.data.shape}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.data[idx]

        if self.transform is not None:
            img = self.transform(PILImage.fromarray(img))
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, self.labels[idx]


def make_loaders(
    corruption: str,
    data_root: str = "data/tiny-imagenet-200",
    batch_size: int = 128,
    seed: int = 0,
    num_workers: int = 4,
    frost_dir: str = "data/frost_images",
    splits: Optional[list] = None,
    severity: int = DEFAULT_SEVERITY,
    train_size: int = DEFAULT_TRAIN_SIZE,
    val_size: int = DEFAULT_VAL_SIZE,
) -> dict:
    if splits is None:
        splits = ["train", "val", "test"]

    base_tfm = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    train_tfm = T.Compose(
        [
            T.RandomCrop(IMG_SIZE, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    need_train_src = "train" in splits or "val" in splits
    need_test_src = "test" in splits

    train_images, train_labels = _load_train_images(
        data_root) if need_train_src else (None, None)

    if need_train_src:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(train_labels))
        train_images = train_images[idx]
        train_labels = [train_labels[i] for i in idx]

    test_images, test_labels = _load_val_images(
        data_root) if need_test_src else (None, None)

    result = {}

    if "train" in splits:
        train_ds = CorruptedTinyImageNet(
            images=train_images[:train_size],
            labels=train_labels[:train_size],
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
        val_ds = CorruptedTinyImageNet(
            images=train_images[train_size: train_size + val_size],
            labels=train_labels[train_size: train_size + val_size],
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
        test_ds = CorruptedTinyImageNet(
            images=test_images,
            labels=test_labels,
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
    "CorruptedTinyImageNet",
    "make_loaders",
]
