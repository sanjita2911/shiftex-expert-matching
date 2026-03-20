#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "ShiftEx Single-Pod Launcher (4 corruptions, 4 GPUs)"
echo "========================================"

SERVER_READY_WAIT=15

echo "[launcher] Starting server..."
python3 -m server.server &
SERVER_PID=$!
echo "[launcher] Server PID: $SERVER_PID"

echo "[launcher] Waiting ${SERVER_READY_WAIT}s for server to be ready..."
sleep $SERVER_READY_WAIT

echo "[launcher] Starting 4 clients, 1 GPU each..."

CORRUPTIONS=(
    "gaussian_noise"
    "fog"
    "frost"
    "snow"
)

PIDS=()

for i in "${!CORRUPTIONS[@]}"; do
    client_num=$((i + 1))
    gpu=$i
    corr="${CORRUPTIONS[$i]}"

    echo "[launcher] Client-${client_num} GPU=${gpu} corruption=${corr}"

    CUDA_VISIBLE_DEVICES=$gpu python3 run_client.py \
        --client_id "client-${client_num}" \
        --train_corruption "$corr" \
        --mode full \
        --dataset cifar10c \
        --device cuda \
        --server_address localhost:50051 &

    PIDS+=($!)
done

echo "[launcher] All 4 clients started. Waiting for completion..."
echo "========================================"

# Wait for all clients to finish
FAILED=0
for i in "${!PIDS[@]}"; do
    client_num=$((i + 1))
    if wait "${PIDS[$i]}"; then
        echo "[launcher] Client-${client_num} done"
    else
        echo "[launcher] Client-${client_num} FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo "========================================"
echo "[launcher] All clients finished. Failed: $FAILED/4"
echo "[launcher] Shutting down server..."
kill $SERVER_PID 2>/dev/null || true

# Upload results to HuggingFace automatically
echo "[launcher] Uploading results to HuggingFace..."
python3 -c "
import os
from huggingface_hub import HfApi

token = os.environ.get('HF_TOKEN')
if not token or token == 'REPLACE_WITH_YOUR_HF_TOKEN':
    print('[launcher] ERROR: HF_TOKEN not set, skipping upload')
    exit(0)

api = HfApi()

print('[launcher] Uploading expert_storage...')
api.upload_folder(
    folder_path='/app/expert_storage',
    repo_id='user_id/shiftex-results',
    repo_type='model',
    path_in_repo='expert_storage',
    token=token,
)
print('[launcher] Upload complete!')
print('[launcher] Results at: https://huggingface.co/user_id/shiftex-results')
"

echo "[launcher] Experiment complete."
echo "========================================"
