import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).
    Pulls together audio_e from the same identity (across EN and UR),
    pushes apart embeddings from different identities.
    features: (N, D) L2-normalized embeddings
    labels:   (N,)  integer class labels
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        N = features.shape[0]

        sim = (features @ features.T) / self.temperature   # (N, N)

        # numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # positive mask: same label, different index
        label_col = labels.view(-1, 1)
        pos_mask = label_col.eq(label_col.T).float()
        pos_mask.fill_diagonal_(0)

        # denominator: all pairs except self
        all_mask = 1.0 - torch.eye(N, device=device)

        log_prob = sim - torch.log((torch.exp(sim) * all_mask).sum(dim=1, keepdim=True) + 1e-8)

        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -(pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)
        return loss[valid].mean()


class CrossLingualTripletLoss(nn.Module):
    """
    Hard-negative triplet loss across EN and UR audio embeddings.
    anchor  : EN audio_e  (N_a, D) L2-normalized
    pool    : UR audio_e  (N_p, D) L2-normalized
    labels_a: EN identity labels (N_a,)
    labels_p: UR identity labels (N_p,)
    For each EN sample, pulls closest same-identity UR sample closer
    and pushes closest different-identity UR sample further away.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, pool, labels_a, labels_p):
        device = anchor.device
        cos = anchor @ pool.T                                     # (N_a, N_p)
        same = labels_a.unsqueeze(1) == labels_p.unsqueeze(0)    # (N_a, N_p) bool

        pos_cos = cos.clone(); pos_cos[~same] = 2.0              # invalid → large
        neg_cos = cos.clone(); neg_cos[same]  = -2.0             # invalid → small

        pos_sim = pos_cos.min(dim=1).values   # hardest positive per anchor
        neg_sim = neg_cos.max(dim=1).values   # hardest negative per anchor

        valid = same.any(dim=1) & (~same).any(dim=1)
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = F.relu(neg_sim - pos_sim + self.margin)
        return loss[valid].mean()


class OrthogonalProjectionLoss(nn.Module):
    def forward(self, feats, labels):
        feats = F.normalize(feats, dim=1)
        labels = labels.unsqueeze(1)

        mask = labels.eq(labels.T)
        eye = torch.eye(len(labels), device=labels.device).bool()

        pos = (mask & ~eye).float()
        neg = (~mask).float()

        dot = feats @ feats.T

        pos_mean = (pos * dot).sum() / (pos.sum() + 1e-6)
        neg_mean = (neg * dot).abs().sum() / (neg.sum() + 1e-6)

        loss = (1 - pos_mean) + 0.7 * neg_mean
        return loss
