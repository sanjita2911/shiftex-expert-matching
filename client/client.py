from proto import expert_matching_pb2_grpc as pb2_grpc
from proto import expert_matching_pb2 as pb2
import torch
import grpc
import numpy as np
from typing import Optional, Tuple, Dict, Any

from common.models import ResNet50CIFAR, ResNet50TinyImageNet, get_model
from common.serialization import serialize_ndarray, deserialize_ndarray, serialize_state_dict, deserialize_state_dict
from client.shift_detector import ShiftDetector
from client.embedding_extractor import extract_embeddings
from client.trainer import evaluate, train_model
from config import (
    DEVICE,
    SHIFT_ALPHA,
    SERVER_ADDRESS,
    GRPC_MAX_MSG,
    MAX_SIG_SAMPLES,
    BATCH_SIZE,
    NUM_WORKERS,
    get_dataset_config,
)


class FederatedClient:

    def __init__(
        self,
        client_id: str,
        server_address: str = SERVER_ADDRESS,
        device: str = DEVICE,
        alpha: float = SHIFT_ALPHA,
        dataset_name: str = "cifar10c",
    ):

        self.client_id = client_id
        self.device = device
        self.dataset_name = dataset_name

        ds = get_dataset_config(dataset_name)
        self._dataset_config = ds
        self.router = self._load_frozen_router(ds["router_ckpt"], dataset_name)

        # Initialize model
        self.model = get_model(
            dataset_name,
            num_classes=int(ds["num_classes"]),
            pretrained=False,
        )
        self.model.to(device)

        # Initialize shift detector
        self.shift_detector = ShiftDetector(alpha=alpha)

        # gRPC connection
        self.server_address = server_address
        self.channel = None
        self.stub = None
        self._connect_to_server()

        # Track current state
        self.current_expert_id: Optional[str] = None
        self.window_count = 0

    def _load_frozen_router(self, checkpoint_path: str, dataset_name: str):
        """
        Load a router checkpoint and freeze it.

        We keep this local to avoid relying on historical helper signatures.
        """
        name = dataset_name.strip().lower()
        if name in {"cifar10c", "cifar10"}:
            model = ResNet50CIFAR(num_classes=10)
        elif name in {"tinyimagenetc", "tinyimagenet", "tiny-imagenet"}:
            model = ResNet50TinyImageNet(num_classes=200, pretrained=False)
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        sd = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(sd)
        model.to(self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def _connect_to_server(self):
        """Establish gRPC connection to server."""
        self.channel = grpc.insecure_channel(
            self.server_address,
            options=[
                ("grpc.max_receive_message_length", GRPC_MAX_MSG),
                ("grpc.max_send_message_length", GRPC_MAX_MSG),
            ],
        )
        self.stub = pb2_grpc.ExpertMatchingServiceStub(self.channel)
        print(f"[{self.client_id}] Connected to server at {self.server_address}")

    def train_and_register(
        self,
        corruption: str,
        *,
        dataset_name: Optional[str] = None,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """
        Phase 1: train an expert locally, build a signature, register both to the server.

        Returns:
            dict with training + registration metadata
        """
        ds_name = self.dataset_name if dataset_name is None else dataset_name
        ds = get_dataset_config(ds_name)

        batch_size = BATCH_SIZE
        num_workers = NUM_WORKERS

        if ds_name.strip().lower() in {"cifar10c", "cifar10"}:
            from data.cifar10c import make_loaders as make_cifar_loaders

            loaders = make_cifar_loaders(
                corruption=corruption,
                cifar10_root="data",
                batch_size=batch_size,
                seed=seed,
                num_workers=num_workers,
                frost_dir=ds.get("frost_dir", "data/frost_images"),
                splits=["train", "val"],
                severity=int(ds.get("severity", 5)),
                train_size=int(ds.get("train_size", 40_000)),
                val_size=int(ds.get("val_size", 10_000)),
            )
        else:
            from data.tinyimagenetc import make_loaders as make_tiny_loaders

            loaders = make_tiny_loaders(
                corruption=corruption,
                data_root=ds.get("data_root", "data/tiny-imagenet-200"),
                batch_size=batch_size,
                seed=seed,
                num_workers=num_workers,
                frost_dir=ds.get("frost_dir", "data/frost_images"),
                splits=["train", "val"],
                severity=int(ds.get("severity", 5)),
                train_size=int(ds.get("train_size", 80_000)),
                val_size=int(ds.get("val_size", 20_000)),
            )

        # Fresh expert model for training
        self.model = get_model(
            ds_name,
            num_classes=int(ds["num_classes"]),
            pretrained=False,
        )
        self.model.to(self.device)

        print(f"[{self.client_id}] Training expert for corruption='{corruption}' on dataset='{ds_name}'")
        self.model = train_model(
            self.model,
            loaders["train"],
            loaders["val"],
            device=self.device,
            epochs=int(ds.get("epochs", 5)),
            lr=float(ds.get("lr", 1e-3)),
            weight_decay=float(ds.get("weight_decay", 1e-4)),
            label_smoothing=float(ds.get("label_smoothing", 0.0)),
            patience=int(ds.get("patience", 5)),
        )

        print(f"[{self.client_id}] Building signature embeddings (max={MAX_SIG_SAMPLES})...")
        sig_embeddings = extract_embeddings(
            self.router,
            loaders["train"],
            device=self.device,
            max_samples=int(ds.get("max_sig_samples", 0) or MAX_SIG_SAMPLES),
        )
        print(f"[{self.client_id}] Signature embeddings shape={sig_embeddings.shape}")

        self.register_as_expert(expert_id=corruption, signature_embeddings=sig_embeddings)
        self.shift_detector.update_baseline(sig_embeddings)

        return {
            "status": "OK",
            "client_id": self.client_id,
            "dataset": ds_name,
            "expert_id": corruption,
            "signature_shape": tuple(sig_embeddings.shape),
        }

    def request_test_data_assignment(self) -> Tuple[str, int]:
        """
        Phase 2: ask server for a hidden test corruption assignment.
        """
        try:
            resp = self.stub.AssignTestData(pb2.TestDataRequest(client_id=self.client_id))
            return resp.corruption, int(resp.severity)
        except grpc.RpcError as e:
            print(f"[{self.client_id}] ERROR: AssignTestData failed: {e}")
            return "", 0

    def report_result(
        self,
        *,
        assigned_corruption: str,
        matched_expert: str,
        accuracy: float,
        mmd_distance: float,
        shift_detected: bool,
        correct_match: bool,
    ) -> str:
        """
        Phase 4: report result to server for aggregation.
        """
        try:
            resp = self.stub.ReportResult(
                pb2.ResultRequest(
                    client_id=self.client_id,
                    assigned_corruption=assigned_corruption,
                    matched_expert=matched_expert,
                    accuracy=float(accuracy),
                    mmd_distance=float(mmd_distance),
                    shift_detected=bool(shift_detected),
                    correct_match=bool(correct_match),
                )
            )
            return resp.status
        except grpc.RpcError as e:
            print(f"[{self.client_id}] ERROR: ReportResult failed: {e}")
            return "ERROR"

    def process_new_window(
        self,
        data_loader,
        corruption_type: str,
        evaluate_baseline: bool = False,
    ) -> Dict[str, float]:

        self.window_count += 1
        print(f"\n{'='*60}")
        print(
            f"[{self.client_id}] Window {self.window_count}: {corruption_type} data")
        print(f"{'='*60}")

        # Step 1: Extract embeddings from new data
        print(f"[{self.client_id}] Extracting embeddings...")
        new_embeddings = extract_embeddings(
            self.router,
            data_loader,
            device=self.device,
            max_samples=2000
        )
        print(
            f"[{self.client_id}] Extracted embeddings: shape={new_embeddings.shape}")

        # Step 2: Detect shift
        shift_detected, p_val, dist = self.shift_detector.detect_shift(
            new_embeddings)
        print(f"[{self.client_id}] p-value: {p_val:.6f}")

        print(f"[{self.client_id}] alpha: {self.shift_detector.get_alpha()}")
        print(f"[{self.client_id}] MMD distance: {dist:.6f}")
        print(f"[{self.client_id}] Drift detected: {shift_detected}")

        # Store original expert for baseline comparison
        original_expert = self.current_expert_id

        # Step 3: Handle shift if detected
        if shift_detected:
            print(
                f"[{self.client_id}] ⚠️ SHIFT DETECTED - Requesting expert match...")
            matched_expert_id = self.request_expert_match(new_embeddings)

            if matched_expert_id != self.current_expert_id:
                print(
                    f"[{self.client_id}] Switching from '{self.current_expert_id}' to '{matched_expert_id}'")
                self.download_and_load_expert(matched_expert_id)
            else:
                print(
                    f"[{self.client_id}] Server confirmed: Keep current expert '{matched_expert_id}'")
        else:
            print(
                f"[{self.client_id}] ✓ No shift - Continuing with expert '{self.current_expert_id}'")
            matched_expert_id = self.current_expert_id

        # Step 4: Evaluate with matched expert
        accuracy = evaluate(self.model, data_loader, device=self.device)
        print(
            f"[{self.client_id}] Accuracy with '{matched_expert_id}': {accuracy:.4f}")

        # Optional: Evaluate with baseline for comparison
        baseline_accuracy = None
        if evaluate_baseline and original_expert and original_expert != matched_expert_id:
            print(
                f"[{self.client_id}] Evaluating baseline ('{original_expert}') for comparison...")
            # Temporarily load original expert
            original_model_state = {k: v.detach().cpu().clone()
                                    for k, v in self.model.state_dict().items()}
            self.download_and_load_expert(original_expert, verbose=False)
            baseline_accuracy = evaluate(
                self.model, data_loader, device=self.device)
            print(f"[{self.client_id}] Baseline accuracy: {baseline_accuracy:.4f}")
            print(
                f"[{self.client_id}] Improvement: {(accuracy - baseline_accuracy)*100:.2f} pp")
            # Restore matched expert
            self.model.load_state_dict(original_model_state)
            self.model.to(self.device)

        # Step 5: Update baseline for next window
        self.shift_detector.update_baseline(new_embeddings)

        # Return metrics
        return {
            "window": self.window_count,
            "corruption_type": corruption_type,
            "mmd_distance": dist,
            "p_val": p_val,
            "shift_detected": shift_detected,
            "matched_expert": matched_expert_id,
            "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
        }

    def request_expert_match(self, test_embeddings: np.ndarray) -> str:

        # Serialize embeddings
        emb_bytes, emb_shape = serialize_ndarray(test_embeddings)

        # Call server
        try:
            response = self.stub.MatchExpert(
                pb2.ExpertMatchRequest(
                    client_id=self.client_id,
                    test_embeddings=emb_bytes,
                    embedding_shape=emb_shape,
                )
            )

            print(f"[{self.client_id}] Server response:")
            print(
                f"[{self.client_id}]   Best match: {response.best_expert_id} (MMD: {response.best_mmd:.6f})")
            print(
                f"[{self.client_id}]   All scores: {dict(response.all_mmd_scores)}")

            return response.best_expert_id

        except grpc.RpcError as e:
            print(f"[{self.client_id}] ERROR: Server communication failed: {e}")
            # Fallback: keep current expert
            return self.current_expert_id or ""

    def download_and_load_expert(self, expert_id: str, verbose: bool = True):

        if verbose:
            print(f"[{self.client_id}] Downloading expert '{expert_id}'...")

        try:
            response = self.stub.DownloadExpertModel(
                pb2.ModelDownloadRequest(expert_id=expert_id)
            )

            # Deserialize and load model weights
            state_dict = deserialize_state_dict(response.model_weights)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)

            # Update tracking
            self.current_expert_id = expert_id
            self.shift_detector.set_expert(expert_id)

            if verbose:
                print(f"[{self.client_id}] ✓ Loaded expert '{expert_id}'")

        except grpc.RpcError as e:
            print(f"[{self.client_id}] ERROR: Failed to download expert: {e}")

    def register_as_expert(
        self,
        expert_id: str,
        signature_embeddings: np.ndarray,
    ):

        print(f"[{self.client_id}] Registering as expert '{expert_id}'...")

        # Serialize model and embeddings
        model_bytes = serialize_state_dict(self.model.state_dict())
        emb_bytes, emb_shape = serialize_ndarray(signature_embeddings)

        # Register with server
        response = self.stub.RegisterExpert(
            pb2.ExpertRegistrationRequest(
                client_id=self.client_id,
                expert_id=expert_id,
                model_weights=model_bytes,
                signature_embeddings=emb_bytes,
                embedding_shape=emb_shape,
            )
        )

        print(f"[{self.client_id}] Registration status: {response.status}")

        # Track as current expert
        self.current_expert_id = expert_id
        self.shift_detector.set_expert(expert_id)

    def set_model(self, model: torch.nn.Module):

        self.model = model
        self.model.to(self.device)

    def get_model(self) -> torch.nn.Module:
        """Get the current model."""
        return self.model

    def close(self):
        """Close gRPC connection."""
        if self.channel:
            self.channel.close()
            print(f"[{self.client_id}] Closed connection to server")
