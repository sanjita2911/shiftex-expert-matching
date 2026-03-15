import argparse
import os
import sys
from typing import Optional

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.trainer import train_model
from common.models import ResNet50CIFAR, ResNet50TinyImageNet
from config import DEVICE, BATCH_SIZE, NUM_WORKERS, get_dataset_config


class TinyImageNetValDataset(Dataset):
    """
    Tiny-ImageNet validation loader.

    The val split stores all images in a flat directory with labels in
    val/val_annotations.txt. We map synset IDs to class indices using the same
    mapping as ImageFolder(train/).
    """

    def __init__(self, val_dir: str, class_to_idx: dict, transform=None):
        self.transform = transform
        self.samples = []

        img_dir = os.path.join(val_dir, "images")
        ann_file = os.path.join(val_dir, "val_annotations.txt")

        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"val_annotations.txt not found at {ann_file}")

        fname_to_synset = {}
        with open(ann_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    fname_to_synset[parts[0]] = parts[1]

        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            synset = fname_to_synset.get(fname)
            if synset is None:
                continue
            if synset not in class_to_idx:
                continue
            label = int(class_to_idx[synset])
            self.samples.append((os.path.join(img_dir, fname), label))

        print(f"[TinyImageNetValDataset] Loaded {len(self.samples)} val images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _build_cifar10_loaders(
    *,
    batch_size: int,
    num_workers: int,
):
    train_tfm = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )
    val_tfm = T.Compose([T.ToTensor()])

    train_ds = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=train_tfm
    )
    val_ds = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=val_tfm
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def _build_tinyimagenet_loaders(
    *,
    data_root: str,
    batch_size: int,
    num_workers: int,
):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Tiny-ImageNet train directory not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Tiny-ImageNet val directory not found: {val_dir}")

    train_tfm = T.Compose(
        [
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tfm = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = ImageFolder(root=train_dir, transform=train_tfm)
    val_ds = TinyImageNetValDataset(
        val_dir=val_dir,
        class_to_idx=train_ds.class_to_idx,
        transform=val_tfm,
    )

    print(f"[TinyImageNet] Train set: {len(train_ds)} images, {len(train_ds.classes)} classes")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Train a frozen router (ResNet-50) for ShiftEx.")
    ap.add_argument("--dataset", type=str, default="cifar10c", choices=["cifar10c", "tinyimagenetc"])
    ap.add_argument("--device", type=str, default=DEVICE)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--data_root", type=str, default="")
    ap.add_argument("--ckpt_path", type=str, default="")
    ap.add_argument("--no_pretrained", action="store_true", help="TinyImageNet only: start from random weights")
    args = ap.parse_args(argv)

    ds = get_dataset_config(args.dataset)
    ckpt_path = args.ckpt_path or ds["router_ckpt"]

    epochs = args.epochs or int(ds.get("router_epochs", ds.get("epochs", 50)))
    lr = args.lr or float(ds.get("router_lr", ds.get("lr", 1e-3)))
    weight_decay = args.weight_decay or float(ds.get("router_weight_decay", ds.get("weight_decay", 1e-4)))
    patience = args.patience or int(ds.get("patience", 5))

    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    print(f"Dataset     : {args.dataset}")
    print(f"Device      : {args.device}")
    print(f"Checkpoint  : {ckpt_path}")
    print(f"Epochs      : {epochs}")
    print(f"LR          : {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Patience    : {patience}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Num workers : {args.num_workers}")

    if args.dataset == "cifar10c":
        train_loader, val_loader = _build_cifar10_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        model = ResNet50CIFAR(num_classes=int(ds.get("num_classes", 10)))
    else:
        data_root = args.data_root or ds.get("data_root", "data/tiny-imagenet-200")
        print(f"Data root   : {data_root}")
        train_loader, val_loader = _build_tinyimagenet_loaders(
            data_root=data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        # Historically the router is fine-tuned from ImageNet weights.
        pretrained = not bool(args.no_pretrained)
        model = ResNet50TinyImageNet(
            num_classes=int(ds.get("num_classes", 200)),
            pretrained=pretrained,
        )

    model = train_model(
        model,
        train_loader,
        val_loader,
        device=args.device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=float(ds.get("label_smoothing", 0.0)),
        patience=patience,
    )

    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved router to: {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
