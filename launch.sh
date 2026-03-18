#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "ShiftEx Single-Pod Launcher"
echo "========================================"

# Wait for server to be ready before starting clients
SERVER_READY_WAIT=15

# Start server in background (CPU only)
echo "[launcher] Starting server..."
python3 -m server.server &
SERVER_PID=$!
echo "[launcher] Server PID: $SERVER_PID"

# Give server time to start up before clients connect
echo "[launcher] Waiting ${SERVER_READY_WAIT}s for server to be ready..."
sleep $SERVER_READY_WAIT

# Start 4 clients in parallel, each pinned to a different GPU
echo "[launcher] Starting 4 clients..."

CUDA_VISIBLE_DEVICES=0 python3 run_client.py \
    --client_id client-1 \
    --train_corruption gaussian_noise \
    --mode full \
    --dataset cifar10c \
    --device cuda \
    --server_address localhost:50051 &
CLIENT1_PID=$!
echo "[launcher] Client-1 PID: $CLIENT1_PID (GPU 0, gaussian_noise)"

CUDA_VISIBLE_DEVICES=1 python3 run_client.py \
    --client_id client-2 \
    --train_corruption fog \
    --mode full \
    --dataset cifar10c \
    --device cuda \
    --server_address localhost:50051 &
CLIENT2_PID=$!
echo "[launcher] Client-2 PID: $CLIENT2_PID (GPU 1, fog)"

CUDA_VISIBLE_DEVICES=2 python3 run_client.py \
    --client_id client-3 \
    --train_corruption frost \
    --mode full \
    --dataset cifar10c \
    --device cuda \
    --server_address localhost:50051 &
CLIENT3_PID=$!
echo "[launcher] Client-3 PID: $CLIENT3_PID (GPU 2, frost)"

CUDA_VISIBLE_DEVICES=3 python3 run_client.py \
    --client_id client-4 \
    --train_corruption snow \
    --mode full \
    --dataset cifar10c \
    --device cuda \
    --server_address localhost:50051 &
CLIENT4_PID=$!
echo "[launcher] Client-4 PID: $CLIENT4_PID (GPU 3, snow)"

echo "[launcher] All processes started. Waiting for completion..."
echo "========================================"

# Wait for all clients to finish
wait $CLIENT1_PID && echo "[launcher] Client-1 done" || echo "[launcher] Client-1 FAILED"
wait $CLIENT2_PID && echo "[launcher] Client-2 done" || echo "[launcher] Client-2 FAILED"
wait $CLIENT3_PID && echo "[launcher] Client-3 done" || echo "[launcher] Client-3 FAILED"
wait $CLIENT4_PID && echo "[launcher] Client-4 done" || echo "[launcher] Client-4 FAILED"

echo "========================================"
echo "[launcher] All clients finished."
echo "[launcher] Shutting down server..."
kill $SERVER_PID 2>/dev/null || true

echo "[launcher] Experiment complete. Results saved in /app/expert_storage"
echo "========================================"
