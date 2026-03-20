import os
import json
import time
import numpy as np
from typing import Dict, Optional
from common.serialization import serialize_state_dict, deserialize_state_dict, serialize_ndarray, deserialize_ndarray


class PersistentExpertRegistry:
    def __init__(self, storage_dir: str = "expert_storage"):
        self.storage_dir = storage_dir
        self._experts = {}  # expert_id -> dict

        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(os.path.join(storage_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(storage_dir, "signatures"), exist_ok=True)

        self._load_all_experts()

    def register(self, client_id: str, expert_id: str, model_weights: bytes,
                 sig_bytes: bytes, shape: list):

        self._experts[expert_id] = {
            "client_id": client_id,
            "model_weights": model_weights,
            "signature_embeddings": sig_bytes,
            "embedding_shape": list(shape),
            "ts": time.time(),
        }

        # Save to disk
        self._save_expert(expert_id)
        print(f"[registry] Saved expert '{expert_id}' to disk")

    def _save_expert(self, expert_id: str):
        if expert_id not in self._experts:
            return

        expert_data = self._experts[expert_id]

        # Save model weights
        model_path = os.path.join(
            self.storage_dir, "models", f"{expert_id}.pt")
        with open(model_path, "wb") as f:
            f.write(expert_data["model_weights"])

        # Save signature embeddings
        sig_path = os.path.join(
            self.storage_dir, "signatures", f"{expert_id}.npy")
        with open(sig_path, "wb") as f:
            f.write(expert_data["signature_embeddings"])

        # Save metadata
        metadata = {
            "client_id": expert_data["client_id"],
            "embedding_shape": expert_data["embedding_shape"],
            "timestamp": expert_data["ts"],
        }
        metadata_path = os.path.join(
            self.storage_dir, "models", f"{expert_id}.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_all_experts(self):
        models_dir = os.path.join(self.storage_dir, "models")

        if not os.path.exists(models_dir):
            print("[registry] No existing experts found")
            return

        metadata_files = [f for f in os.listdir(
            models_dir) if f.endswith(".json")]

        if not metadata_files:
            print("[registry] No existing experts found")
            return

        print(f"[registry] Loading {len(metadata_files)} existing experts...")

        for metadata_file in metadata_files:
            expert_id = metadata_file.replace(".json", "")

            try:
                # Load metadata
                metadata_path = os.path.join(models_dir, metadata_file)
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                # Load model weights
                model_path = os.path.join(models_dir, f"{expert_id}.pt")
                with open(model_path, "rb") as f:
                    model_weights = f.read()

                # Load signature embeddings
                sig_path = os.path.join(
                    self.storage_dir, "signatures", f"{expert_id}.npy")
                with open(sig_path, "rb") as f:
                    sig_bytes = f.read()

                # Store in memory
                self._experts[expert_id] = {
                    "client_id": metadata["client_id"],
                    "model_weights": model_weights,
                    "signature_embeddings": sig_bytes,
                    "embedding_shape": metadata["embedding_shape"],
                    "ts": metadata["timestamp"],
                }

                print(f"[registry] ✓ Loaded expert '{expert_id}'")

            except Exception as e:
                print(f"[registry] ✗ Failed to load expert '{expert_id}': {e}")
                continue

        print(f"[registry] Successfully loaded {len(self._experts)} experts")

    def list_ids(self):
        return list(self._experts.keys())

    def get(self, expert_id: str) -> Optional[Dict]:
        return self._experts.get(expert_id)

    def delete(self, expert_id: str):
        if expert_id not in self._experts:
            return

        del self._experts[expert_id]

        model_path = os.path.join(
            self.storage_dir, "models", f"{expert_id}.pt")
        sig_path = os.path.join(
            self.storage_dir, "signatures", f"{expert_id}.npy")
        metadata_path = os.path.join(
            self.storage_dir, "models", f"{expert_id}.json")

        for path in [model_path, sig_path, metadata_path]:
            if os.path.exists(path):
                os.remove(path)

        print(f"[registry] Deleted expert '{expert_id}'")

    def clear_all(self):
        expert_ids = list(self._experts.keys())
        for expert_id in expert_ids:
            self.delete(expert_id)
        print(f"[registry] Cleared all {len(expert_ids)} experts")
