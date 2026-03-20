import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from typing import Optional


class ResNet50CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        m = resnet50(weights=None)

        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                            padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(2048, num_classes)

        self.m = m

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        m = self.m
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)

        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)

        x = m.avgpool(x)
        x = torch.flatten(x, 1)  # [B, 2048]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.m.fc(feats)


class ResNet50TinyImageNet(nn.Module):

    def __init__(self, num_classes: int = 200, pretrained: bool = False):
        super().__init__()

        if pretrained:
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            print("[ResNet50TinyImageNet] Loaded pretrained IMAGENET1K_V2 weights")
        else:
            m = resnet50(weights=None)
            print("[ResNet50TinyImageNet] Initialized with random weights")

        m.fc = nn.Linear(2048, num_classes)

        self.m = m

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:

        m = self.m
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)

        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)

        x = m.avgpool(x)
        x = torch.flatten(x, 1)  # [B, 2048]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.m.fc(feats)


def get_model(
    dataset_name: str,
    *,
    num_classes: Optional[int] = None,
    pretrained: bool = False,
) -> nn.Module:

    name = dataset_name.strip().lower()

    if name in {"cifar10c", "cifar10"}:
        return ResNet50CIFAR(num_classes=10 if num_classes is None else num_classes)

    if name in {"tinyimagenetc", "tinyimagenet", "tiny-imagenet"}:
        return ResNet50TinyImageNet(
            num_classes=200 if num_classes is None else num_classes,
            pretrained=pretrained,
        )

    raise ValueError(
        f"Unknown dataset '{dataset_name}'. "
        "Expected 'cifar10c' or 'tinyimagenetc'."
    )
