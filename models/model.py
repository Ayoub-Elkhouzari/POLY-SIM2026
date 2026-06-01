import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# Utility blocks
# --------------------------------------------------

def fc_block(in_dim, out_dim, p=0.5):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p),
    )


class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, emb_dim):
        super().__init__()
        self.fc = fc_block(feat_dim, emb_dim)

    def forward(self, x):
        x = self.fc(x)
        return F.normalize(x, dim=1)


class DeepEmbedBranch(nn.Module):
    """Two-layer version of EmbedBranch: feat_dim → emb_dim → emb_dim."""
    def __init__(self, feat_dim, emb_dim):
        super().__init__()
        self.fc1 = fc_block(feat_dim, emb_dim)
        self.fc2 = fc_block(emb_dim, emb_dim)

    def forward(self, x):
        return F.normalize(self.fc2(self.fc1(x)), dim=1)


# --------------------------------------------------
# Linear fusion
# --------------------------------------------------

class LinearFusion(nn.Module):
    """
    Learnable weighted sum fusion
    """
    def __init__(self):
        super().__init__()
        self.w_face = nn.Parameter(torch.rand(1))
        self.w_audio = nn.Parameter(torch.rand(1))

    def forward(self, face, audio):
        fused = self.w_face * face + self.w_audio * audio
        return fused, face, audio


# --------------------------------------------------
# Adaptive gate fusion (element-wise, lightweight)
# --------------------------------------------------

class AdaptiveGateFusion(nn.Module):
    """
    Per-sample, per-dimension gate learned from [face_e; audio_e].
    gate_i → 1: trust face on dim i; gate_i → 0: trust audio.
    No projection heads — gates the already L2-normalised embeddings directly.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.gate_fc = nn.Linear(emb_dim * 2, emb_dim)

    def forward(self, face, audio):
        gate  = torch.sigmoid(self.gate_fc(torch.cat([face, audio], dim=1)))
        fused = F.normalize(gate * face + (1.0 - gate) * audio, dim=1)
        return fused, face, audio


# --------------------------------------------------
# Gated fusion
# --------------------------------------------------

class ForwardBlock(nn.Module):
    def __init__(self, in_dim, out_dim, p=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
        )

    def forward(self, x):
        return self.block(x)


class AudioMemoryBank(nn.Module):
    """
    Lightweight associative memory bank for the audio branch.

    Inspired by Nested Learning (NeurIPS 2025): learns M prototype keys/values
    that act as a language-agnostic identity bank.  At inference on unseen-
    language (Urdu) audio the L2-similarity lookup retrieves the closest stored
    identity pattern, bridging the cross-lingual gap.

    Starts as a pure passthrough (gate initialised to -3 ≈ sigmoid → 0.05) so
    it cannot hurt early training, and gradually learns to use the memory.
    """
    def __init__(self, dim: int, memory_size: int = 70, beta: float = 5.0):
        super().__init__()
        self.beta = beta
        self.keys   = nn.Parameter(torch.randn(memory_size, dim) * 0.02)
        self.values = nn.Parameter(torch.randn(memory_size, dim) * 0.02)
        # scalar gate: sigmoid(-3) ≈ 0.047 → nearly pure passthrough at init
        self.gate = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) — already L2-normalised by EmbedBranch
        keys_n = F.normalize(self.keys, dim=1)           # (M, D)
        sims   = x @ keys_n.T                            # (B, M)
        attn   = F.softmax(sims * self.beta, dim=-1)     # (B, M)
        retrieved = attn @ self.values                   # (B, D)
        alpha  = torch.sigmoid(self.gate)
        out    = alpha * retrieved + (1.0 - alpha) * x
        return F.normalize(out, dim=1)


class GatedFusion(nn.Module):
    """
    Gated multimodal fusion
    """
    def __init__(self, emb_dim, mid_dim=128):
        super().__init__()

        self.attention = nn.Sequential(
            ForwardBlock(emb_dim * 2, mid_dim),
            nn.Linear(mid_dim, emb_dim),
        )

        self.face_proj = nn.Linear(emb_dim, emb_dim)
        self.audio_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, face, audio):
        concat = torch.cat([face, audio], dim=1)
        gate = torch.sigmoid(self.attention(concat))

        face_t = torch.tanh(self.face_proj(face))
        audio_t = torch.tanh(self.audio_proj(audio))

        fused = gate * face_t + (1.0 - gate) * audio_t
        return fused, face_t, audio_t
