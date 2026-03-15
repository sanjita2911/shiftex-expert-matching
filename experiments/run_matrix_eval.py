import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.mmd import median_heuristic_sigma, mmd_rbf_unbiased
from common.models import ResNet50CIFAR, ResNet50TinyImageNet, get_model
from client.embedding_extractor import extract_embeddings
from client.trainer import evaluate
from config import (
    CORRUPTIONS_4,
    CORRUPTIONS_15,
    DEVICE,
    MAX_SIG_SAMPLES,
    MAX_TEST_SAMPLES,
    SIGMA_POOL_PTS,
    BATCH_SIZE,
    NUM_WORKERS,
    get_dataset_config,
)


def _load_frozen_router(checkpoint_path: str, dataset_name: str, device: str) -> torch.nn.Module:
    name = dataset_name.strip().lower()
    if name in {"cifar10c", "cifar10"}:
        model = ResNet50CIFAR(num_classes=10)
    elif name in {"tinyimagenetc", "tinyimagenet", "tiny-imagenet"}:
        model = ResNet50TinyImageNet(num_classes=200, pretrained=False)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    sd = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _load_expert_model(storage_dir: str, dataset_name: str, expert_id: str, device: str) -> torch.nn.Module:
    path = os.path.join(storage_dir, "models", f"{expert_id}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expert model not found: {path}")
    state_dict = torch.load(path, map_location="cpu")

    model = get_model(dataset_name, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _load_expert_signature(storage_dir: str, expert_id: str) -> np.ndarray:
    path = os.path.join(storage_dir, "signatures", f"{expert_id}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expert signature not found: {path}")
    sig = np.load(path, allow_pickle=False).astype(np.float32)
    if sig.ndim != 2:
        raise RuntimeError(f"Signature expected shape (N, D), got {sig.shape}")
    return sig


def _build_test_loader(
    *,
    dataset_name: str,
    corruption: str,
    severity: int,
    batch_size: int,
    num_workers: int,
    data_root: str,
    frost_dir: str,
) -> DataLoader:
    name = dataset_name.strip().lower()
    if name in {"cifar10c", "cifar10"}:
        from data.cifar10c import make_loaders

        loaders = make_loaders(
            corruption=corruption,
            cifar10_root="data",
            batch_size=batch_size,
            seed=42,
            num_workers=num_workers,
            frost_dir=frost_dir,
            splits=["test"],
            severity=severity,
        )
        return loaders["test"]

    if name in {"tinyimagenetc", "tinyimagenet", "tiny-imagenet"}:
        from data.tinyimagenetc import make_loaders

        loaders = make_loaders(
            corruption=corruption,
            data_root=data_root,
            batch_size=batch_size,
            seed=42,
            num_workers=num_workers,
            frost_dir=frost_dir,
            splits=["test"],
            severity=severity,
        )
        return loaders["test"]

    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def _diagonal_summary(acc_mat: pd.DataFrame, mmd_mat: pd.DataFrame, corruptions: List[str]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("ACCURACY MATRIX (rows=test, cols=expert)")
    lines.append("=" * 70)
    lines.append(acc_mat.to_string(float_format="{:.4f}".format))

    lines.append("\n" + "=" * 70)
    lines.append("MMD MATRIX (rows=test, cols=expert)")
    lines.append("=" * 70)
    lines.append(mmd_mat.to_string(float_format="{:.6f}".format))

    lines.append("\n" + "=" * 70)
    lines.append("DIAGONAL ANALYSIS")
    lines.append("=" * 70)

    diag_accs = [float(acc_mat.loc[c, c]) for c in corruptions]
    offdiag_accs = [float(acc_mat.loc[r, c]) for r in corruptions for c in corruptions if r != c]
    lines.append("Accuracy — diagonal vs off-diagonal:")
    lines.append(f"  Mean diagonal    : {np.mean(diag_accs):.4f}")
    lines.append(f"  Mean off-diagonal: {np.mean(offdiag_accs):.4f}")
    lines.append(f"  Diagonal gain    : {np.mean(diag_accs) - np.mean(offdiag_accs):+.4f}")

    diag_mmds = [float(mmd_mat.loc[c, c]) for c in corruptions]
    offdiag_mmds = [float(mmd_mat.loc[r, c]) for r in corruptions for c in corruptions if r != c]
    lines.append("\nMMD — diagonal vs off-diagonal:")
    lines.append(f"  Mean diagonal    : {np.mean(diag_mmds):.6f}")
    lines.append(f"  Mean off-diagonal: {np.mean(offdiag_mmds):.6f}")

    lines.append("\nPer-corruption: best expert by MMD vs best expert by accuracy")
    lines.append(f"  {'Corruption':<18} {'Best MMD expert':<20} {'Best Acc expert':<20} {'Match?'}")
    lines.append(f"  {'-' * 74}")
    all_match = True
    for test_corr in corruptions:
        best_by_mmd = mmd_mat.loc[test_corr].idxmin()
        best_by_acc = acc_mat.loc[test_corr].idxmax()
        match = "YES" if best_by_mmd == best_by_acc else "NO"
        if best_by_mmd != best_by_acc:
            all_match = False
        lines.append(f"  {test_corr:<18} {best_by_mmd:<20} {best_by_acc:<20} {match}")

    lines.append(f"\nMMD routing correct for all corruptions: {'YES' if all_match else 'NO'}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Run NxN expert-vs-corruption matrix evaluation (accuracy + MMD).")
    ap.add_argument("--dataset", type=str, default="cifar10c", choices=["cifar10c", "tinyimagenetc"])
    ap.add_argument("--use_15", action="store_true", help="Use CORRUPTIONS_15 instead of CORRUPTIONS_4.")
    ap.add_argument("--corruptions", type=str, default="", help="Comma-separated corruption list (overrides defaults).")
    ap.add_argument("--device", type=str, default=DEVICE)
    ap.add_argument("--storage_dir", type=str, default="")
    ap.add_argument("--router_ckpt", type=str, default="")
    ap.add_argument("--data_root", type=str, default="")
    ap.add_argument("--frost_dir", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--severity", type=int, default=0)
    ap.add_argument("--n_sig", type=int, default=MAX_SIG_SAMPLES)
    ap.add_argument("--n_test", type=int, default=MAX_TEST_SAMPLES)
    ap.add_argument("--sigma_pts", type=int, default=SIGMA_POOL_PTS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    args = ap.parse_args(argv)

    ds = get_dataset_config(args.dataset)

    if args.corruptions:
        corruptions = [x.strip() for x in args.corruptions.split(",") if x.strip()]
    else:
        corruptions = list(CORRUPTIONS_15 if args.use_15 else CORRUPTIONS_4)

    storage_dir = args.storage_dir or ds["storage_dir"]
    router_ckpt = args.router_ckpt or ds["router_ckpt"]
    data_root = args.data_root or ds.get("data_root", "")
    frost_dir = args.frost_dir or ds.get("frost_dir", "data/frost_images")
    severity = args.severity or int(ds.get("severity", 5))

    n = len(corruptions)
    out_dir = args.out_dir or os.path.join("results", f"{args.dataset}_{n}x{n}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Dataset     : {args.dataset}")
    print(f"Device      : {args.device}")
    print(f"Storage dir : {storage_dir}")
    print(f"Router ckpt : {router_ckpt}")
    if args.dataset == "tinyimagenetc":
        print(f"Data root   : {data_root}")
    print(f"Frost dir   : {frost_dir}")
    print(f"Corruptions : {corruptions}")
    print(f"Severity    : {severity}")
    print(f"Out dir     : {out_dir}")

    # Step 1: load router
    print(f"\nLoading frozen router ...")
    router = _load_frozen_router(router_ckpt, args.dataset, args.device)
    print("Router loaded.")

    # Step 2: load expert models and signatures
    print("\nLoading experts ...")
    experts: Dict[str, torch.nn.Module] = {}
    signatures: Dict[str, np.ndarray] = {}
    for corr in corruptions:
        experts[corr] = _load_expert_model(storage_dir, args.dataset, corr, args.device)
        sig = _load_expert_signature(storage_dir, corr)[: args.n_sig]
        signatures[corr] = sig
        print(f"  [{corr}] model + signature loaded | sig={sig.shape}")

    # Step 3: build test loaders + extract test embeddings
    print("\nExtracting test embeddings ...")
    test_loaders: Dict[str, DataLoader] = {}
    test_embeds: Dict[str, np.ndarray] = {}
    for corr in corruptions:
        test_loader = _build_test_loader(
            dataset_name=args.dataset,
            corruption=corr,
            severity=severity,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_root=data_root,
            frost_dir=frost_dir,
        )
        test_loaders[corr] = test_loader

        emb = extract_embeddings(
            router,
            test_loader,
            device=args.device,
            max_samples=args.n_test,
        )
        test_embeds[corr] = emb
        print(f"  [{corr}] test embeddings: {emb.shape}")

    # Step 4: global sigma for fair comparison
    print("\nComputing global RBF bandwidth (median heuristic) ...")
    sigma = median_heuristic_sigma(
        list(signatures.values()) + list(test_embeds.values()),
        max_points=args.sigma_pts,
        seed=0,
    )
    print(f"Global sigma = {sigma:.6f}")

    # Step 5: fill matrices
    print(f"\nComputing {n}x{n} matrices ...")
    acc_mat = pd.DataFrame(index=corruptions, columns=corruptions, dtype=float)
    mmd_mat = pd.DataFrame(index=corruptions, columns=corruptions, dtype=float)
    rows = []

    for expert_corr in corruptions:
        model = experts[expert_corr]
        sig = signatures[expert_corr]
        for test_corr in corruptions:
            acc = evaluate(model, test_loaders[test_corr], device=args.device)
            dist = mmd_rbf_unbiased(sig, test_embeds[test_corr], sigma=sigma, device=args.device)

            acc_mat.loc[test_corr, expert_corr] = float(acc)
            mmd_mat.loc[test_corr, expert_corr] = float(dist)

            rows.append(
                {
                    "expert_corruption": expert_corr,
                    "test_corruption": test_corr,
                    "accuracy": float(acc),
                    "mmd": float(dist),
                    "match": expert_corr == test_corr,
                    "n_sig": int(sig.shape[0]),
                    "n_test": int(test_embeds[test_corr].shape[0]),
                    "embed_dim": int(sig.shape[1]),
                    "sigma": float(sigma),
                    "dataset": args.dataset,
                    "severity": int(severity),
                }
            )

            tag = " ← diagonal" if expert_corr == test_corr else ""
            print(f"  [{test_corr:18s} -> {expert_corr:18s}] acc={acc:.4f} mmd={dist:.6f}{tag}")

    # Save outputs
    acc_path = os.path.join(out_dir, "accuracy_matrix.csv")
    mmd_path = os.path.join(out_dir, "mmd_matrix.csv")
    long_path = os.path.join(out_dir, "results_long.csv")
    summary_path = os.path.join(out_dir, "summary.txt")

    acc_mat.to_csv(acc_path)
    mmd_mat.to_csv(mmd_path)
    pd.DataFrame(rows).to_csv(long_path, index=False)

    summary = _diagonal_summary(acc_mat, mmd_mat, corruptions)
    with open(summary_path, "w") as f:
        f.write(summary)

    print("\n" + summary)
    print("\nSaved:")
    print(f"  {acc_path}")
    print(f"  {mmd_path}")
    print(f"  {long_path}")
    print(f"  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

