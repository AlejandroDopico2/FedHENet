"""Helpers for constructing FedHENet split models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch.nn as nn
from torchvision import models

from .base import FeatureExtractorClassifierModel


@dataclass(frozen=True)
class SplitModelConfig:
    """Configuration for constructing the canonical extractor/classifier setup."""

    num_classes: int
    backbone: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = True
    trainable_backbone_layers: Optional[Sequence[str]] = None
    classifier_bias: bool = True
    classifier_dropout: float = 0.0
    device: str = "cpu"


def build_split_model(config: SplitModelConfig) -> FeatureExtractorClassifierModel:
    """Return a split model honoring the provided configuration."""

    feature_extractor, in_features = _build_feature_extractor(
        backbone=config.backbone,
        pretrained=config.pretrained,
    )

    if config.freeze_backbone:
        _freeze_module(feature_extractor)

    if config.trainable_backbone_layers:
        _unfreeze_layers(feature_extractor, config.trainable_backbone_layers)

    classifier = _build_classifier(
        in_features=in_features,
        num_classes=config.num_classes,
        bias=config.classifier_bias,
        dropout=config.classifier_dropout,
    )

    model = FeatureExtractorClassifierModel(
        feature_extractor=feature_extractor,
        classifier=classifier,
        flatten_features=False,
    )
    model.to(config.device)
    model.eval()
    model.get_classifier().train()
    return model


def _build_feature_extractor(
    backbone: str, pretrained: bool
) -> tuple[nn.Module, int]:
    builders = {"resnet18": _build_resnet18_extractor}
    if backbone not in builders:
        raise ValueError(f"Unsupported backbone '{backbone}'")
    return builders[backbone](pretrained=pretrained)


def _build_resnet18_extractor(pretrained: bool) -> tuple[nn.Module, int]:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    backbone = models.resnet18(weights=weights)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, in_features


def _build_classifier(
    *,
    in_features: int,
    num_classes: int,
    bias: bool,
    dropout: float,
) -> nn.Module:
    layers = []
    if dropout and dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(in_features, num_classes, bias=bias))
    if len(layers) == 1:
        return layers[0]
    return nn.Sequential(*layers)


def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _unfreeze_layers(module: nn.Module, layer_prefixes: Sequence[str]) -> None:
    prefixes = tuple(layer_prefixes)
    if not prefixes:
        return
    for name, param in module.named_parameters():
        if any(_matches_prefix(name, prefix) for prefix in prefixes):
            param.requires_grad = True


def _matches_prefix(name: str, prefix: str) -> bool:
    if prefix.endswith("*"):
        prefix = prefix[:-1]
    return name.startswith(prefix)

