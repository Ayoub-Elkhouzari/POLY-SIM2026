import torch
import torch.nn as nn


class CMSBlock(nn.Module):
    """Single residual FFN block from the Continuum Memory System (Nested Learning)."""

    def __init__(self, dim, hidden_multiplier=4, activation='gelu', grad_clip=1.0):
        super().__init__()
        hidden = dim * hidden_multiplier
        act = {'relu': nn.ReLU(), 'silu': nn.SiLU()}.get(activation, nn.GELU())
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            act,
            nn.Linear(hidden, dim),
        )
        self.grad_clip = grad_clip

    def forward(self, x):
        delta = self.net(x)
        if self.training and self.grad_clip > 0:
            with torch.no_grad():
                scale = torch.clamp(
                    delta.norm(dim=-1, keepdim=True) / self.grad_clip, min=1.0)
            delta = delta / scale
        return x + delta


class CMS(nn.Module):
    """
    Continuum Memory System: stack of CMSBlocks with multi-level residual processing.

    Used here as a cross-modal imputer: maps available-modality features to an
    approximation of a missing modality's post-attention embedding.
    """

    def __init__(self, dim, n_levels=3, hidden_multiplier=4, activation='gelu'):
        super().__init__()
        self.blocks = nn.ModuleList([
            CMSBlock(dim, hidden_multiplier=hidden_multiplier, activation=activation)
            for _ in range(n_levels)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
