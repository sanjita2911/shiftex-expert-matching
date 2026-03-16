import argparse
import os
from typing import Optional, Tuple

import numpy as np

from client.client import FederatedClient
from client.embedding_extractor import extract_embeddings
from client.trainer import evaluate
from common.serialization import serialize_ndarray
from config import (
    DEVICE,
    SERVER_ADDRESS,
    MAX_TEST_SAMPLES,
    BATCH_SIZE,
    NUM_WORKERS,
    get_dataset_config,
)
from proto import expert_matching_pb2 as pb2


def _build_test_loader(
    *,
    dataset_name: str,
    corruption: str,
    severity: int,
    batch_size: int,
    num_workers: int,
    data_root: str,
    frost_dir: str,
):
    ds = dataset_name.strip().lower()
    if ds in {"cifar10c", "cifar10"}:
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

    if ds in {"tinyimagenetc", "tinyimagenet", "tiny-imagenet"}:
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


def _match_expert(client: FederatedClient, embeddings: np.ndarray) -> Tuple[str, float]:
    emb_bytes, emb_shape = serialize_ndarray(embeddings)
    resp = client.stub.MatchExpert(
        pb2.ExpertMatchRequest(
            client_id=client.client_id,
            test_embeddings=emb_bytes,
            embedding_shape=emb_shape,
        )
    )
    return resp.best_expert_id, float(resp.best_mmd)


def _ensure_dataset(dataset_name: str, data_root: str) -> None:
    """Download dataset if not already present on this pod."""
    if dataset_name == "cifar10c":
        marker = os.path.join("data", "cifar10c", "labels.npy")
        if not os.path.exists(marker):
            print("[setup] CIFAR-10-C not found. Downloading via kagglehub...")
            import kagglehub
            import shutil
            src = kagglehub.dataset_download("harshadakhatu/cifar-10-c")
            src_data = os.path.join(src, "CIFAR-10-C")
            os.makedirs("data/cifar10c", exist_ok=True)
            for f in os.listdir(src_data):
                if f.endswith(".npy"):
                    shutil.copy(os.path.join(src_data, f),
                                os.path.join("data/cifar10c", f))
            print("[setup] CIFAR-10-C ready.")
        else:
            print("[setup] CIFAR-10-C already present.")

    elif dataset_name == "tinyimagenetc":
        marker = os.path.join("data", "tiny-imagenet-200", "train")
        if not os.path.exists(marker):
            print("[setup] Tiny-ImageNet not found. Downloading...")
            os.makedirs("data", exist_ok=True)
            os.system(
                "wget -q http://cs231n.stanford.edu/tiny-imagenet-200.zip -P data/")
            os.system("unzip -q data/tiny-imagenet-200.zip -d data/")
            print("[setup] Tiny-ImageNet ready.")
        else:
            print("[setup] Tiny-ImageNet already present.")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="ShiftEx client pod entrypoint.")
    ap.add_argument("--mode", type=str, default=os.getenv("MODE",
                    "full"), choices=["train", "infer", "full"])
    ap.add_argument("--client_id", type=str,
                    default=os.getenv("CLIENT_ID", "client"))
    ap.add_argument("--server_address", type=str,
                    default=os.getenv("SERVER_ADDRESS", SERVER_ADDRESS))
    ap.add_argument("--dataset", type=str, default=os.getenv("DATASET",
                    "cifar10c"), choices=["cifar10c", "tinyimagenetc"])
    ap.add_argument("--device", type=str, default=os.getenv("DEVICE", DEVICE))
    ap.add_argument("--train_corruption", type=str,
                    default=os.getenv("TRAIN_CORRUPTION", ""))
    ap.add_argument("--seed", type=int, default=int(os.getenv("SEED", "0")))
    ap.add_argument("--batch_size", type=int,
                    default=int(os.getenv("BATCH_SIZE", str(BATCH_SIZE))))
    ap.add_argument("--num_workers", type=int,
                    default=int(os.getenv("NUM_WORKERS", str(NUM_WORKERS))))
    ap.add_argument("--max_test_samples", type=int,
                    default=int(os.getenv("MAX_TEST_SAMPLES", str(MAX_TEST_SAMPLES))))
    args = ap.parse_args(argv)

    # ------------------------------------------------------------------
    # mode=infer requires a pre-existing shift-detector baseline.
    # That baseline is set during train_and_register() (mode=train/full).
    # Running infer without a prior training phase means no baseline exists,
    # so detect_shift() will always return (False, 1.0, 0.0) and expert
    # matching will never trigger — the client would just run inference
    # with an untrained model and report wrong results silently.
    #
    # On Nautilus every pod starts fresh (no persistent state between runs),
    # so mode=infer alone is never valid there. Always use mode=full.
    #
    # mode=infer is only meaningful if you re-use a long-lived process that
    # already called train_and_register() earlier in the same session
    # (e.g. interactive local testing). Guard against accidental misuse here.
    # ------------------------------------------------------------------
    if args.mode == "infer" and not args.train_corruption:
        # No training corruption supplied — warn clearly rather than silently
        # producing wrong results.
        print(
            "[warning] mode=infer with no prior training session: "
            "shift_detector has no baseline, so shift will never be detected "
            "and MatchExpert will not be called. "
            "Use mode=full to train and infer in the same run, "
            "or supply --train_corruption and run mode=train first."
        )

    ds = get_dataset_config(args.dataset)
    frost_dir = ds.get("frost_dir", "data/frost_images")
    data_root = ds.get("data_root", "")

    _ensure_dataset(args.dataset, data_root)

    print(f"client_id       : {args.client_id}")
    print(f"mode            : {args.mode}")
    print(f"dataset         : {args.dataset}")
    print(f"device          : {args.device}")
    print(f"server_address  : {args.server_address}")
    print(f"train_corruption: {args.train_corruption or '(none)'}")

    client = FederatedClient(
        client_id=args.client_id,
        server_address=args.server_address,
        device=args.device,
        dataset_name=args.dataset,
    )

    try:
        if args.mode in {"train", "full"}:
            if not args.train_corruption:
                raise SystemExit(
                    "TRAIN_CORRUPTION is required for mode=train/full.")
            client.train_and_register(
                args.train_corruption, dataset_name=args.dataset, seed=args.seed)

        if args.mode in {"infer", "full"}:
            assigned_corr, severity = client.request_test_data_assignment()
            if not assigned_corr or severity <= 0:
                raise SystemExit(
                    "AssignTestData failed (empty corruption or severity).")

            print(
                f"[{client.client_id}] Assigned test set: corruption={assigned_corr} severity={severity}")

            test_loader = _build_test_loader(
                dataset_name=args.dataset,
                corruption=assigned_corr,
                severity=severity,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                data_root=data_root,
                frost_dir=frost_dir,
            )

            print(f"[{client.client_id}] Extracting test embeddings ...")
            test_embeddings = extract_embeddings(
                client.router,
                test_loader,
                device=args.device,
                max_samples=args.max_test_samples,
            )
            print(
                f"[{client.client_id}] Test embeddings shape={test_embeddings.shape}")

            shift_detected, p_val, drift_dist = client.shift_detector.detect_shift(
                test_embeddings)
            print(
                f"[{client.client_id}] Shift test: shift={shift_detected} p={p_val:.6f} dist={drift_dist:.6f}")

            # If no baseline exists (mode=infer without prior training),
            # detect_shift() returns (False, 1.0, 0.0) — skip the MMD
            # gating entirely and go straight to server matching so the
            # experiment still produces a meaningful result.
            no_baseline = not client.shift_detector.has_baseline()
            if no_baseline:
                print(
                    f"[{client.client_id}] No shift-detector baseline — "
                    "skipping local MMD test, calling MatchExpert directly."
                )

            matched_expert = client.current_expert_id or ""
            best_mmd = float(drift_dist)

            if shift_detected or no_baseline:
                matched_expert, best_mmd = _match_expert(
                    client, test_embeddings)
                print(
                    f"[{client.client_id}] Server match: expert={matched_expert} mmd={best_mmd:.6f}")
                if matched_expert and matched_expert != client.current_expert_id:
                    client.download_and_load_expert(matched_expert)

            acc = evaluate(client.model, test_loader, device=args.device)
            correct = bool(matched_expert == assigned_corr)
            print(
                f"[{client.client_id}] Final: matched={matched_expert} assigned={assigned_corr} acc={acc:.4f} correct={correct}")

            status = client.report_result(
                assigned_corruption=assigned_corr,
                matched_expert=matched_expert,
                accuracy=float(acc),
                mmd_distance=float(best_mmd),
                shift_detected=bool(shift_detected),
                correct_match=bool(correct),
            )
            print(f"[{client.client_id}] ReportResult status: {status}")

    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
