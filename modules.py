"""
modules.py
Model module definitions shared across training scripts.
Contains: MindEyeModule, RidgeRegression, and LoRA/Skip-LoRA helper utilities.
"""

import torch
import torch.nn as nn
from typing import List


class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()

    def forward(self, x):
        return x


class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer to enable regularization
    def __init__(self, input_sizes, out_features):
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(input_size, out_features) for input_size in input_sizes
        ])

    def forward(self, x, subj_idx):
        out = self.linears[subj_idx](x[:, 0]).unsqueeze(1)
        return out


def freeze_module_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def count_trainable_params(module: nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def trainable_params(module: nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def print_trainable_params_breakdown(model: nn.Module):
    """Print trainable parameter counts for ridge, backbone, and prior.
    Also breaks down backbone adapters into LoRA-only and Skip-LoRA-only if present.
    """
    import utils  # imported here to avoid circular dependency at module load time

    ridge_trainable = count_trainable_params(model.ridge) if hasattr(model, 'ridge') else 0
    backbone_trainable = count_trainable_params(model.backbone) if hasattr(model, 'backbone') else 0
    prior_trainable = count_trainable_params(model.diffusion_prior) if hasattr(model, 'diffusion_prior') else 0

    backbone_lora_only = 0
    backbone_skip_only = 0
    if hasattr(model, 'backbone'):
        for n, p in model.backbone.named_parameters():
            if not p.requires_grad:
                continue
            if 'lora_' in n:
                backbone_lora_only += p.numel()
            elif ('_skip_adapter' in n) or ('_skip_adapters' in n):
                backbone_skip_only += p.numel()

    prior_lora_only = 0
    prior_skip_only = 0
    if hasattr(model, 'diffusion_prior'):
        for n, p in model.diffusion_prior.named_parameters():
            if not p.requires_grad:
                continue
            if 'lora_' in n:
                prior_lora_only += p.numel()
            elif ('_skip_adapter' in n) or ('_skip_adapters' in n):
                prior_skip_only += p.numel()

    total_trainable = ridge_trainable + backbone_trainable + prior_trainable
    total_params_all = utils.count_params(model)

    print(
        "Trainable params breakdown (M):\n"
        f"  - MLP backbone (trainable): {backbone_trainable/1e6:.2f}M\n"
        f"      • LoRA-only (backbone): {backbone_lora_only/1e6:.2f}M\n"
        f"      • Skip-LoRA-only (backbone): {backbone_skip_only/1e6:.2f}M\n"
        f"  - ridge: {ridge_trainable/1e6:.2f}M\n"
        f"  - prior (trainable): {prior_trainable/1e6:.2f}M\n"
        f"      • LoRA-only (prior): {prior_lora_only/1e6:.2f}M\n"
        f"      • Skip-LoRA-only (prior): {prior_skip_only/1e6:.2f}M\n"
        f"  = TOTAL trainable: {total_trainable/1e6:.2f}M\n"
        f"  = TOTAL (all params): {total_params_all/1e6:.2f}M"
    )
