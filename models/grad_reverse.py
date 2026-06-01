import torch
import torch.nn as nn
from torch.autograd import Function


class _GRLFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    """Gradient Reversal Layer — passes forward unchanged, flips gradient sign backward."""

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return _GRLFunction.apply(x, self.alpha)
