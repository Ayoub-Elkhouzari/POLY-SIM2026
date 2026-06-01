import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """Additive Angular Margin (ArcFace) classification head.

    At training (labels provided): adds margin m to the target class angle.
    At inference (labels=None): returns cosine * s — behaves like a cosine classifier.
    """

    def __init__(self, in_dim, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.s     = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels=None):
        cosine = F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))
        if labels is None:
            return cosine * self.s
        sine   = (1.0 - cosine.pow(2)).clamp(0, 1).sqrt()
        phi    = cosine * self.cos_m - sine * self.sin_m        # cos(θ + m)
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.s
