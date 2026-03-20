# ShiftEx — Quick Setup

## One-time Setup

```bash
# 1. Setup the repo
cd shiftex-expert-matching

# 2. Build and push image
docker buildx build --platform linux/amd64 -t YOUR_DOCKERHUB/shiftex:latest --push .

# 3. Edit k8s/single_pod.yaml — replace these two placeholders:
#    image: xyz/shiftex:latest   ← your image
#    HF_TOKEN: REPLACE_WITH_YOUR_HF_TOKEN  ← your HuggingFace token

#4. Launch.sh
# Update your HuggingFace user_name
```

## Run Experiment

```bash
# Deploy
kubectl -n YOUR_NAMESPACE apply -f k8s/single_pod.yaml

# Watch logs
kubectl -n YOUR_NAMESPACE logs -f job/shiftex-experiment

# Auto-save results when done
kubectl -n YOUR_NAMESPACE wait --for=condition=complete job/shiftex-experiment --timeout=10800s && \
kubectl -n YOUR_NAMESPACE logs \
  $(kubectl -n YOUR_NAMESPACE get pods --selector=job-name=shiftex-experiment \
  -o jsonpath='{.items[0].metadata.name}') > experiment.log

# Clean up
kubectl -n YOUR_NAMESPACE delete job shiftex-experiment
```

## Results
Results are automatically uploaded to your HuggingFace repo at:
`https://huggingface.co/YOUR_HF_USERNAME/shiftex-results`

