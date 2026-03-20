from config import (
    CORRUPTIONS_4,
    CORRUPTIONS_15,
    DEVICE,
    SERVER_ADDRESS,
)
from client.client import FederatedClient
import argparse
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_corruptions(arg: Optional[str], use_15: bool) -> List[str]:
    if arg:
        items = [x.strip() for x in arg.split(",")]
        return [x for x in items if x]
    return list(CORRUPTIONS_15 if use_15 else CORRUPTIONS_4)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Train corruption experts and register them to the ShiftEx gRPC server."
    )
    ap.add_argument("--dataset", type=str, default="cifar10c",
                    choices=["cifar10c", "tinyimagenetc"])
    ap.add_argument("--server_address", type=str, default=SERVER_ADDRESS)
    ap.add_argument("--device", type=str, default=DEVICE)
    ap.add_argument("--client_id", type=str, default="client")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--corruptions",
        type=str,
        default="",
        help="Comma-separated corruption list. Default is CORRUPTIONS_4 unless --use_15 is set.",
    )
    ap.add_argument(
        "--use_15",
        action="store_true",
        help="Train the 15-corruption set from config.CORRUPTIONS_15 instead of the 4-corruption set.",
    )
    args = ap.parse_args(argv)

    corruptions = _parse_corruptions(args.corruptions, args.use_15)
    if not corruptions:
        raise SystemExit("No corruptions specified.")

    print(f"Dataset       : {args.dataset}")
    print(f"Server address: {args.server_address}")
    print(f"Device        : {args.device}")
    print(f"Seed          : {args.seed}")
    print(f"Corruptions   : {corruptions}")

    client = FederatedClient(
        client_id=args.client_id,
        server_address=args.server_address,
        device=args.device,
        dataset_name=args.dataset,
    )

    try:
        for i, corr in enumerate(corruptions):
            client.client_id = f"{args.client_id}_{corr}"
            print("\n" + "=" * 80)
            print(
                f"Training + registering expert: {corr} ({i+1}/{len(corruptions)})")
            print("=" * 80)
            client.train_and_register(
                corr, dataset_name=args.dataset, seed=args.seed)
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
