#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NAMESPACE="${NAMESPACE:-default}"
IMAGE="${IMAGE:-shiftex:latest}"
DATASET="${DATASET:-cifar10c}"

echo "Namespace: $NAMESPACE"
echo "Image    : $IMAGE"
echo "Dataset  : $DATASET"

echo
echo "Applying server..."
kubectl -n "$NAMESPACE" apply -f k8s/server.yaml

echo
echo "Waiting for server rollout..."
kubectl -n "$NAMESPACE" rollout status deployment/shiftex-server

echo
echo "Creating 4 client pods..."

declare -a CORRUPTIONS=("gaussian_noise" "fog" "frost" "snow")

for i in "${!CORRUPTIONS[@]}"; do
  cid="client-$((i+1))"
  corr="${CORRUPTIONS[$i]}"
  pod="shiftex-client-$((i+1))"

  echo "  - $pod (CLIENT_ID=$cid, TRAIN_CORRUPTION=$corr)"

  kubectl -n "$NAMESPACE" delete pod "$pod" --ignore-not-found

  kubectl -n "$NAMESPACE" apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${pod}
  labels:
    app: shiftex-client
spec:
  restartPolicy: Never
  containers:
    - name: client
      image: ${IMAGE}
      imagePullPolicy: IfNotPresent
      command: ["python3", "run_client.py"]
      env:
        - name: SERVER_ADDRESS
          value: "shiftex-server:50051"
        - name: MODE
          value: "full"
        - name: DATASET
          value: "${DATASET}"
        - name: DEVICE
          value: "cuda"
        - name: CLIENT_ID
          value: "${cid}"
        - name: TRAIN_CORRUPTION
          value: "${corr}"
      resources:
        limits:
          nvidia.com/gpu: "1"
          cpu: "4"
          memory: "16Gi"
        requests:
          nvidia.com/gpu: "1"
          cpu: "2"
          memory: "8Gi"
EOF
done

echo
echo "Deployed. Watch logs with:"
echo "  kubectl -n $NAMESPACE logs -f deploy/shiftex-server"
echo "  kubectl -n $NAMESPACE logs -f pod/shiftex-client-1"

