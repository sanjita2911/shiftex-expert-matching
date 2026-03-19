from concurrent import futures
import os
import random
import threading

import grpc

from proto import expert_matching_pb2 as pb2
from proto import expert_matching_pb2_grpc as pb2_grpc

# Import MMD computation and serialization utilities
from common.mmd import median_heuristic_sigma, mmd_rbf_unbiased
from common.serialization import deserialize_ndarray
from server.persistent_expert_registry import PersistentExpertRegistry


class ExpertMatchingService(pb2_grpc.ExpertMatchingServiceServicer):
    def __init__(self, storage_dir: str = "expert_storage"):
        """
        Initialize with persistent storage.

        Args:
            storage_dir: directory to store expert models and signatures
        """
        self.registry = PersistentExpertRegistry(storage_dir=storage_dir)

        # Phase 2: server-side hidden test assignment
        # - client_id -> {"corruption": str, "severity": int}
        self._assign_lock = threading.Lock()
        self._assignments = {}
        self._assignment_pool = None  # lazily initialized

        # Phase 4: results aggregation
        self._results_lock = threading.Lock()
        self._results = {}  # client_id -> dict
        self._expected_clients = int(os.getenv("EXPECTED_CLIENTS", "4"))

        # Barrier: MatchExpert blocks until all expected experts are registered.
        self._all_experts_ready = threading.Event()
        expert_ids = self.registry.list_ids()
        if len(expert_ids) >= self._expected_clients:
            self._all_experts_ready.set()
            print(f"[server] All {self._expected_clients} experts already present — barrier open")

        # Log loaded experts
        if expert_ids:
            print(
                f"[server] Loaded {len(expert_ids)} existing experts: {expert_ids}")
        else:
            print(f"[server] No existing experts found - starting fresh")

    def RegisterExpert(self, request, context):
        self.registry.register(
            request.client_id,
            request.expert_id,
            request.model_weights,
            request.signature_embeddings,
            request.embedding_shape,
        )
        print(
            f"[server] registered expert={request.expert_id} from client={request.client_id}")
        # Open barrier once all expected experts are present
        if len(self.registry.list_ids()) >= self._expected_clients:
            self._all_experts_ready.set()
            print(f"[server] All {self._expected_clients} experts registered — barrier open")
        return pb2.ExpertRegistrationResponse(status="OK")

    def AssignTestData(self, request, context):
        """
        Assign one hidden (corruption, severity) pair to a client.

        The server keeps assignments stable per client_id across retries.
        """
        from config import CORRUPTIONS_4

        client_id = request.client_id.strip()
        if not client_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("client_id is required")
            return pb2.TestDataResponse(corruption="", severity=0)

        # Severity injected via env var — keeps server dataset-agnostic.
        # Defaults to 5 (matches all our experiments).
        severity = int(os.getenv("SEVERITY", "5"))

        with self._assign_lock:
            existing = self._assignments.get(client_id)
            if existing is not None:
                return pb2.TestDataResponse(
                    corruption=existing["corruption"],
                    severity=int(existing["severity"]),
                )

            if self._assignment_pool is None:
                # One-time shuffle: deterministic within a server run.
                seed = int(os.getenv("ASSIGNMENT_SEED", "0"))
                rng = random.Random(seed)
                pool = list(CORRUPTIONS_4)
                rng.shuffle(pool)
                self._assignment_pool = pool

            if not self._assignment_pool:
                # More clients than corruptions: reshuffle and reuse.
                seed = int(os.getenv("ASSIGNMENT_SEED", "0"))
                rng = random.Random(seed + len(self._assignments) + 1)
                pool = list(CORRUPTIONS_4)
                rng.shuffle(pool)
                self._assignment_pool = pool

            corruption = self._assignment_pool.pop(0)
            self._assignments[client_id] = {"corruption": corruption, "severity": severity}

        print(f"[server] assigned test data to client={client_id}: {corruption} severity={severity}")
        return pb2.TestDataResponse(corruption=corruption, severity=severity)

    def MatchExpert(self, request, context):
        """
        Match client's test embeddings to the best expert using MMD.

        Blocks until all expected experts are registered (avoids race where
        fast clients call MatchExpert before slow experts exist).
        """
        timeout = int(os.getenv("MATCH_EXPERT_WAIT_TIMEOUT", "3600"))
        if not self._all_experts_ready.wait(timeout=timeout):
            print(
                f"[server] WARNING: MatchExpert timed out after {timeout}s waiting for "
                f"{self._expected_clients} experts. Proceeding with {len(self.registry.list_ids())} available."
            )

        expert_ids = self.registry.list_ids()

        # Check if we have any experts registered
        if not expert_ids:
            print("[server] ERROR: No experts registered yet!")
            return pb2.ExpertMatchResponse(
                best_expert_id="",
                best_mmd=float("inf"),
                all_mmd_scores={}
            )

        # Deserialize test embeddings from client
        try:
            test_embeddings = deserialize_ndarray(request.test_embeddings)
            print(
                f"[server] Received test embeddings: shape={test_embeddings.shape}")
        except Exception as e:
            print(f"[server] ERROR deserializing test embeddings: {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Failed to deserialize embeddings: {e}")
            return pb2.ExpertMatchResponse(
                best_expert_id="",
                best_mmd=float("inf"),
                all_mmd_scores={}
            )

        # Collect all expert signatures
        expert_signatures = {}
        for expert_id in expert_ids:
            expert_record = self.registry.get(expert_id)

            try:
                # Deserialize signature embeddings
                signature = deserialize_ndarray(
                    expert_record["signature_embeddings"])
                expert_signatures[expert_id] = signature
                print(
                    f"[server]   Expert '{expert_id}': signature shape={signature.shape}")
            except Exception as e:
                print(
                    f"[server] WARNING: Failed to load signature for expert '{expert_id}': {e}")
                continue

        # Check if we successfully loaded any signatures
        if not expert_signatures:
            print("[server] ERROR: No valid expert signatures loaded!")
            return pb2.ExpertMatchResponse(
                best_expert_id="",
                best_mmd=float("inf"),
                all_mmd_scores={}
            )

        # Compute one global sigma so all expert-vs-test distances are on the same scale.
        sigma_global = median_heuristic_sigma(
            [test_embeddings] + list(expert_signatures.values()),
            max_points=1024,
            seed=0,
        )

        # Compute MMD between test embeddings and all expert signatures
        print(
            f"[server] Computing fixed-kernel MMD with {len(expert_signatures)} experts (sigma={sigma_global:.6f})...")
        mmd_scores = {}
        for expert_id, signature in expert_signatures.items():
            try:
                mmd = mmd_rbf_unbiased(
                    signature,
                    test_embeddings,
                    sigma=sigma_global,
                    device="cpu",
                )
                mmd_scores[expert_id] = float(mmd)
            except Exception as e:
                print(
                    f"[server] WARNING: MMD failed for expert '{expert_id}': {e}")

        if not mmd_scores:
            print("[server] ERROR: MMD failed for all experts!")
            return pb2.ExpertMatchResponse(
                best_expert_id="",
                best_mmd=float("inf"),
                all_mmd_scores={}
            )

        # Find best match (lowest MMD)
        best_expert_id = min(mmd_scores, key=mmd_scores.get)
        best_mmd = mmd_scores[best_expert_id]

        # Log results
        print(f"[server] Match request from client={request.client_id}")
        print(f"[server] MMD Scores:")
        for expert_id in sorted(mmd_scores.keys(), key=lambda x: mmd_scores[x]):
            score = mmd_scores[expert_id]
            marker = " ← BEST MATCH" if expert_id == best_expert_id else ""
            print(f"[server]   {expert_id}: {score:.6f}{marker}")

        # Return response
        return pb2.ExpertMatchResponse(
            best_expert_id=best_expert_id,
            best_mmd=best_mmd,
            all_mmd_scores=mmd_scores
        )

    def DownloadExpertModel(self, request, context):
        rec = self.registry.get(request.expert_id)
        if rec is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Unknown expert_id: {request.expert_id}")
            return pb2.ModelDownloadResponse(model_weights=b"")
        print(f"[server] download request expert={request.expert_id}")
        return pb2.ModelDownloadResponse(model_weights=rec["model_weights"])

    def ListExperts(self, request, context):
        ids = self.registry.list_ids()
        return pb2.ListExpertsResponse(expert_ids=ids)

    def ReportResult(self, request, context):
        """
        Receive per-client evaluation results and print an aggregate table.
        """
        client_id = request.client_id.strip()
        if not client_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("client_id is required")
            return pb2.ResultResponse(status="ERROR")

        row = {
            "client_id": client_id,
            "assigned_corruption": request.assigned_corruption,
            "shift_detected": bool(request.shift_detected),
            "matched_expert": request.matched_expert,
            "mmd_distance": float(request.mmd_distance),
            "accuracy": float(request.accuracy),
            "correct_match": bool(request.correct_match),
        }

        with self._results_lock:
            self._results[client_id] = row
            n = len(self._results)

        print(
            "[server] result received: "
            f"client={client_id} assigned={row['assigned_corruption']} "
            f"shift={row['shift_detected']} matched={row['matched_expert']} "
            f"mmd={row['mmd_distance']:.6f} acc={row['accuracy']:.4f} correct={row['correct_match']}"
        )

        if n >= self._expected_clients:
            self._print_results_table()

        return pb2.ResultResponse(status="OK")

    def _print_results_table(self):
        with self._results_lock:
            rows = [self._results[k] for k in sorted(self._results.keys())]

        if not rows:
            return

        print("\n" + "=" * 90)
        print("[server] FINAL RESULTS")
        print("=" * 90)
        print(
            f"  {'Client':<12} {'Assigned':<18} {'Shift?':<8} "
            f"{'Matched':<18} {'MMD':>8} {'Accuracy':>9} {'Correct?':<8}"
        )
        print(f"  {'-'*86}")

        correct_total = 0
        for r in rows:
            correct = r["correct_match"]
            correct_total += int(correct)
            marker = "✓" if correct else "✗"
            print(
                f"  {r['client_id']:<12} {r['assigned_corruption']:<18} "
                f"{str(r['shift_detected']):<8} {r['matched_expert']:<18} "
                f"{r['mmd_distance']:>8.6f} {r['accuracy']:>9.4f} {marker}"
            )

        print(f"  {'-'*86}")
        print(f"  Routing correct: {correct_total}/{len(rows)} ({100*correct_total/len(rows):.1f}%)")
        print("=" * 90 + "\n")


def serve(host: str = "0.0.0.0", port: int = 50051, storage_dir: str = "expert_storage"):
    from config import GRPC_MAX_MSG

    MAX_MSG = GRPC_MAX_MSG
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_receive_message_length", MAX_MSG),
            ("grpc.max_send_message_length", MAX_MSG),
        ],
    )
    pb2_grpc.add_ExpertMatchingServiceServicer_to_server(
        ExpertMatchingService(storage_dir=storage_dir), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"[server] listening on {host}:{port}")
    print(f"[server] storage directory: {storage_dir}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
