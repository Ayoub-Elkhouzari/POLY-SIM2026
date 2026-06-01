import torch
import torch.nn as nn

from .model import EmbedBranch, DeepEmbedBranch, LinearFusion, GatedFusion, AdaptiveGateFusion, AudioMemoryBank
from .grad_reverse import GradientReversal
from .arc_face import ArcFaceHead


class MaskedFOP(nn.Module):
    """
    TMSE-style masked FOP for missing-modality speaker identification.

    Audio is mandatory (always present).
    Face is optional: mask_face=1 → face available, mask_face=0 → face absent.

    Two classification heads:
        head_av : fusion(face + audio) → used when mask_face=1
        head_a  : audio only           → used when mask_face=0
    """

    def __init__(self, config, face_dim, audio_dim):
        super().__init__()

        emb = config.embedding_dim
        num_classes = config.resolved_num_classes

        AudioBranch = DeepEmbedBranch if getattr(config, "use_deep_audio", False) else EmbedBranch
        self.face_branch  = EmbedBranch(face_dim, emb)
        self.audio_branch = AudioBranch(audio_dim, emb)

        use_mem = getattr(config, "use_audio_memory", False)
        mem_size = getattr(config, "audio_memory_size", 70)
        self.audio_memory = AudioMemoryBank(emb, memory_size=mem_size) if use_mem else None

        if config.fusion == "linear":
            self.fusion = LinearFusion()
            fusion_dim = emb
        elif config.fusion == "gated":
            self.fusion = GatedFusion(emb)
            fusion_dim = emb
        elif config.fusion == "adaptive_gate":
            self.fusion = AdaptiveGateFusion(emb)
            fusion_dim = emb
        elif config.fusion == "concat":
            self.fusion = None
            fusion_dim = emb * 2
        else:
            raise ValueError(f"Unknown fusion type: {config.fusion}")

        self.use_arcface = getattr(config, "use_arcface", False)
        arc_s = getattr(config, "arcface_s", 64.0)
        arc_m = getattr(config, "arcface_m", 0.5)
        if self.use_arcface:
            self.head_av = ArcFaceHead(fusion_dim, num_classes, s=arc_s, m=arc_m)
            self.head_a  = ArcFaceHead(emb,        num_classes, s=arc_s, m=arc_m)
        else:
            self.head_av = nn.Linear(fusion_dim, num_classes)
            self.head_a  = nn.Linear(emb, num_classes)

        # Domain adversarial branch (audio_e → GRL → 2-class domain classifier)
        if getattr(config, "use_domain_adv", False):
            self.grl        = GradientReversal(alpha=1.0)
            self.domain_clf = nn.Sequential(
                nn.Linear(emb, 256), nn.ReLU(), nn.Linear(256, 2)
            )
        else:
            self.grl        = None
            self.domain_clf = None

    def forward(self, face, audio, mask_face=1, labels=None):
        audio_e = self.audio_branch(audio)
        if self.audio_memory is not None:
            audio_e = self.audio_memory(audio_e)
        audio_logits = self.head_a(audio_e, labels) if self.use_arcface else self.head_a(audio_e)

        domain_logits = self.domain_clf(self.grl(audio_e)) if self.domain_clf is not None else None

        if mask_face == 1:
            face_e = self.face_branch(face)
            if self.fusion is None:
                fused = torch.cat([face_e, audio_e], dim=1)
            else:
                fused, face_e, audio_e = self.fusion(face_e, audio_e)
            av_logits = self.head_av(fused, labels) if self.use_arcface else self.head_av(fused)
            return {
                "logits":        av_logits,
                "audio_logits":  audio_logits,
                "domain_logits": domain_logits,
                "face_e":        face_e,
                "audio_e":       audio_e,
                "fused":         fused,
                "mask_face":     1,
            }
        else:
            return {
                "logits":        audio_logits,
                "audio_logits":  audio_logits,
                "domain_logits": domain_logits,
                "face_e":        None,
                "audio_e":       audio_e,
                "fused":         None,
                "mask_face":     0,
            }
