from dataclasses import dataclass
import logging
import os

_BASE = os.path.dirname(os.path.abspath(__file__))

@dataclass
class ExperimentConfig:
    # Paths — set DATA_ROOT env var to override, or edit these defaults
    data_dir:  str = os.path.join(_BASE, "Data")
    home_dir:  str = os.path.join(_BASE, "Data/Features/train")
    val_feats_dir: str = os.path.join(_BASE, "Data/Features")

    seed: int = 1
    device: str = "cuda"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    batch_size: int = 32
    max_epochs: int = 300
    num_workers: int = 0
    alpha: float = 0.5
    embedding_dim: int = 512

    model_type: str = "masked_fop"   # "fop" | "multibranch" | "masked_fop"
    fusion: str = "linear"           # "linear" | "gated" | "concat"

    loss_face: float = 1.0
    loss_audio: float = 1.0
    loss_fusion: float = 1.0

    # TMSE-style modality dropout: probability to drop face modality per sample during training
    modality_dropout_prob: float = 0.5

    # Concatenate ECAPA-TDNN + WavLM-SV audio features (disabled: wavlmsv not extracted locally)
    use_multi_audio: bool = False

    # Include unseen-language training data (forced audio-only) to improve P5/P6
    # NOTE: must remain False for protocol compliance — only English training data is permitted
    train_unseen_lang: bool = False

    # Cosine annealing LR schedule
    lr_schedule: bool = True

    # Cross-modal distillation: audio_e learns to mimic fused (no masking)
    use_crossmodal_distill: bool = False
    crossmodal_distill_weight: float = 1.0

    # AV consistency: KL(head_av || head_a.detach()) on face-present samples
    use_av_consistency: bool = False
    av_consistency_weight: float = 0.5

    # Gaussian noise augmentation on audio features during training
    noise_std: float = 0.0

    # Scheduled adversarial λ: ramp domain_adv_weight from 0 → target over warmup epochs
    lambda_schedule: bool = False
    lambda_warmup_epochs: int = 30

    # Deeper audio branch: two fc_blocks instead of one
    use_deep_audio: bool = False

    # Audio associative memory bank (Nested Learning)
    use_audio_memory: bool = False
    audio_memory_size: int = 70   # one prototype per class

    # Domain adversarial training — forces audio_e to be language-agnostic
    use_domain_adv: bool = False
    domain_adv_weight: float = 0.1

    # ArcFace classification heads — tighter embedding clusters for transductive inference
    use_arcface: bool = False
    arcface_s: float = 64.0
    arcface_m: float = 0.5

    # Cross-lingual triplet loss — pulls same-identity EN/UR audio_e closer
    use_triplet: bool = False
    triplet_weight: float = 0.1
    triplet_margin: float = 0.3

    # Cross-lingual centroid alignment — cosine distance between EN/UR EMA prototypes per class
    use_cl_align: bool = False
    cl_align_weight: float = 0.3
    cl_ema_momentum: float = 0.99

    # Supervised Contrastive Loss on audio_e (cross-lingual positive pairs)
    use_supcon: bool = False
    supcon_weight: float = 0.1
    supcon_temp: float = 0.1

    # M3 optimizer (Multi-scale Momentum Muon from Nested Learning)
    use_m3: bool = False
    m3_slow_chunk: int = 500   # sweep: [100, 250, 500, 1000]

    version: str = "v1"
    seen_lang: str = "English"

    # Fraction of train data held out for local validation monitoring
    val_frac: float = 0.2

    debug: bool = False
    log_level: int = logging.DEBUG if debug else logging.INFO

    early_stop: bool = True
    early_stop_patience: int = 15
    early_stop_min_delta: float = 0.1
    # Monitor mean of all 4 proxy metrics (P3/P4/P5/P6-like)
    early_stop_metric: str = "mean"   # "seen" | "unseen" | "mean"

    @property
    def resolved_num_classes(self):
        if self.version == "v1":
            return 70
        elif self.version == "v2":
            return 84
        elif self.version == "v3":
            return 36
        else:
            raise ValueError(f"Unknown version '{self.version}'")

    @property
    def unseen_lang(self):
        mapping = {
            ("v1", "English"): "Urdu",
            ("v1", "Urdu"):    "English",
            ("v2", "English"): "Hindi",
            ("v2", "Hindi"):   "English",
            ("v3", "English"): "German",
            ("v3", "German"):  "English",
        }
        key = (self.version, self.seen_lang)
        if key not in mapping:
            raise ValueError(f"Invalid version '{self.version}' or seen_lang '{self.seen_lang}'.")
        return mapping[key]
