import numpy as np
import torch
from common.models import ResNet50CIFAR, ResNet50TinyImageNet


def load_router(checkpoint_path: str, device: str, model_class=None):
    if model_class is None:
        model_class = ResNet50CIFAR
    model = model_class(pretrained=False)
    sd = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_embeddings(model, loader, device: str, max_samples: int = 2000) -> np.ndarray:
    model.eval()
    feats = []
    seen = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            f = model.forward_features(x).detach().cpu().numpy()
            feats.append(f)
            seen += f.shape[0]
            if seen >= max_samples:
                break

    arr = np.concatenate(feats, axis=0)[:max_samples]
    return arr.astype(np.float32)
