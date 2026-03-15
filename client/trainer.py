import torch
import torch.nn as nn
from tqdm import tqdm


def evaluate(model, loader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().cpu())
            total += int(y.numel())
    return correct / max(total, 1)


def train_model(
    model,
    train_loader,
    val_loader,
    device: str,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.0,
    patience: int = 5,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs, 1)
    )
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = 0.0
    best_state = None
    epochs_without_improvement = 0

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {ep}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        acc = evaluate(model, val_loader, device)
        cur_lr = opt.param_groups[0]["lr"]
        print(f"  val_acc: {acc:.4f} | lr: {cur_lr:.6g}")
        scheduler.step()

        if acc >= best_val:
            best_val = acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"  no improvement: {epochs_without_improvement}/{patience}")
            if epochs_without_improvement >= patience:
                print(
                    f"  early stopping at epoch {ep} — best val_acc: {best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"  best_val_acc: {best_val:.4f}")

    return model
