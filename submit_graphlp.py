"""
Graph Label Propagation inference for POLY-SIM 2026.

For P4/P6 (audio-only) the current transductive approach cascades:
  P3 pseudo-labels → test audio prototypes → P4 nearest prototype

This script replaces that cascade with a global label spreading step:
  Training audio prototypes (70 labeled nodes)
  + all test audio embeddings (unlabeled nodes)
  → K-NN affinity graph → label spreading → P4/P6 labels

P4/P6 labels emerge from graph geometry — no discrete P3/P5 pseudo-labels involved.
P3/P5 still use multimodal fused-NN (face available, ~99% accurate).

Reference: Chen et al. "Graph-based Label Propagation for Semi-Supervised
Speaker Identification", Interspeech 2021 / Amazon Science.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ExperimentConfig
from models.masked_fop import MaskedFOP
from utils.featLoader import LoadData
from submit_transductive import (
    load_val_feats, extract_embeddings, build_train_proto_av, cosine_nn,
    build_test_audio_proto,
)


def extract_face_embs(model, face, audio, device):
    """Return L2-normalized face_e for the English val set (mask_face=1 pass)."""
    face, audio = face.to(device), audio.to(device)
    with torch.no_grad():
        out = model(face, audio, mask_face=1)
    return F.normalize(out["face_e"], dim=1).cpu()


def build_train_face_proto(model, config, device, audio_encoder="ecappa_feats_path"):
    """L2-normalized face_e per-class centroids from all training data."""
    seen_df   = pd.read_csv(f"{config.data_dir}/Csv_files/{config.version}_train_{config.seen_lang}.csv")
    unseen_df = pd.read_csv(f"{config.data_dir}/Csv_files/{config.version}_train_{config.unseen_lang}.csv")
    train_df  = pd.concat([seen_df, unseen_df], ignore_index=True)


def build_train_face_proto_en_only(model, config, device, audio_encoder="ecappa_feats_path"):
    """L2-normalized face_e per-class centroids from English training data only."""
    seen_df = pd.read_csv(f"{config.data_dir}/Csv_files/{config.version}_train_{config.seen_lang}.csv")
    loader  = DataLoader(
        LoadData(config, df=seen_df, audio_encoder=audio_encoder),
        batch_size=256, shuffle=False, num_workers=0,
    )
    C, D = config.resolved_num_classes, config.embedding_dim
    acc  = torch.zeros(C, D, device=device)
    with torch.no_grad():
        for audio, face, labels in loader:
            audio, face, labels = audio.to(device), face.to(device), labels.to(device)
            fe = F.normalize(model(face, audio, mask_face=1)["face_e"], dim=1)
            for i, lbl in enumerate(labels):
                acc[lbl.item()] += fe[i]
    return F.normalize(acc, dim=1).cpu()   # (C, D)
    loader    = DataLoader(
        LoadData(config, df=train_df, audio_encoder=audio_encoder),
        batch_size=256, shuffle=False, num_workers=0,
    )
    C, D = config.resolved_num_classes, config.embedding_dim
    acc  = torch.zeros(C, D, device=device)
    with torch.no_grad():
        for audio, face, labels in loader:
            audio, face, labels = audio.to(device), face.to(device), labels.to(device)
            fe = F.normalize(model(face, audio, mask_face=1)["face_e"], dim=1)
            for i, lbl in enumerate(labels):
                acc[lbl.item()] += fe[i]
    return F.normalize(acc, dim=1).cpu()   # (C, D)


def build_train_audio_proto(model, config, device,
                             audio_encoder="ecappa_feats_path", audio_encoder2=None):
    """L2-normalized audio_e per-class centroids from all training data."""
    seen_df   = pd.read_csv(f"{config.data_dir}/Csv_files/{config.version}_train_{config.seen_lang}.csv")
    unseen_df = pd.read_csv(f"{config.data_dir}/Csv_files/{config.version}_train_{config.unseen_lang}.csv")
    train_df  = pd.concat([seen_df, unseen_df], ignore_index=True)
    loader    = DataLoader(
        LoadData(config, df=train_df, audio_encoder=audio_encoder, audio_encoder2=audio_encoder2),
        batch_size=256, shuffle=False, num_workers=0,
    )
    C, D = config.resolved_num_classes, config.embedding_dim
    acc  = torch.zeros(C, D, device=device)
    with torch.no_grad():
        for audio, face, labels in loader:
            audio, face, labels = audio.to(device), face.to(device), labels.to(device)
            ae = F.normalize(model(face, audio, mask_face=0)["audio_e"], dim=1)
            for i, lbl in enumerate(labels):
                acc[lbl.item()] += ae[i]
    return F.normalize(acc, dim=1).cpu()   # (C, D)


def build_train_proto_av_face(model, config, device, audio_encoder="ecappa_feats_path"):
    """L2-normalized fused per-class centroids using mask_face=1 (face features enabled)."""
    seen_df   = pd.read_csv(f"{config.data_dir}/Csv_files/{config.version}_train_{config.seen_lang}.csv")
    unseen_df = pd.read_csv(f"{config.data_dir}/Csv_files/{config.version}_train_{config.unseen_lang}.csv")
    train_df  = pd.concat([seen_df, unseen_df], ignore_index=True)
    loader    = DataLoader(
        LoadData(config, df=train_df, audio_encoder=audio_encoder),
        batch_size=256, shuffle=False, num_workers=0,
    )
    C, D = config.resolved_num_classes, config.embedding_dim
    acc  = torch.zeros(C, D, device=device)
    with torch.no_grad():
        for audio, face, labels in loader:
            audio, face, labels = audio.to(device), face.to(device), labels.to(device)
            fused = F.normalize(model(face, audio, mask_face=1)["fused"], dim=1)
            for i, lbl in enumerate(labels):
                acc[lbl.item()] += fused[i]
    return F.normalize(acc, dim=1).cpu()   # (C, D)


def _build_lp_graph(embs, k, mutual=False, doubly_stochastic=False, ds_iters=20,
                    precomp_sim=None):
    """Build normalised K-NN affinity matrix S from L2-normalised embeddings."""
    n = embs.shape[0]
    sim = precomp_sim if precomp_sim is not None else embs @ embs.T
    np.fill_diagonal(sim, 0.0)
    W = np.zeros((n, n), dtype=np.float32)
    top_k = np.argsort(sim, axis=1)[:, -k:]
    rows  = np.arange(n)[:, None].repeat(k, axis=1)
    W[rows, top_k] = np.maximum(sim[rows, top_k], 0.0)
    if mutual:
        W = W * (W.T > 0).astype(np.float32)    # keep only bidirectional edges
    W = (W + W.T) * 0.5
    if doubly_stochastic:
        # Sinkhorn: iterate row/col normalisation until doubly stochastic
        for _ in range(ds_iters):
            W = W / (W.sum(axis=1, keepdims=True) + 1e-8)
            W = W / (W.sum(axis=0, keepdims=True) + 1e-8)
        return W                                  # already normalised
    deg = W.sum(axis=1)
    d   = 1.0 / (np.sqrt(deg) + 1e-8)
    return W * d[:, None] * d[None, :]           # D^{-1/2} W D^{-1/2}


def graph_lp(labeled_embs, labeled_labels, unlabeled_embs, n_classes, k=15, alpha=0.99, n_iters=100, mutual=False, doubly_stochastic=False, precomp_sim=None):
    """
    Classic semi-supervised LP: small set of labeled training anchors + large unlabeled test set.
    labeled_embs   : (n_l, D) — training audio prototypes
    unlabeled_embs : (n_u, D) — test audio embeddings
    Returns        : (n_u,) hard labels
    """
    n_l      = labeled_embs.shape[0]
    all_embs = np.vstack([labeled_embs, unlabeled_embs]).astype(np.float32)
    S        = _build_lp_graph(all_embs, k, mutual=mutual, doubly_stochastic=doubly_stochastic, precomp_sim=precomp_sim)
    Y = np.zeros((all_embs.shape[0], n_classes), dtype=np.float32)
    for i, c in enumerate(labeled_labels):
        Y[i, c] = 1.0
    F = Y.copy()
    for _ in range(n_iters):
        F = alpha * (S @ F) + (1.0 - alpha) * Y
    return F[n_l:].argmax(axis=1)


def graph_lp_p3_smooth(test_embs, p3_labels, n_classes, k=10, alpha=0.7, n_iters=50, mutual=False, doubly_stochastic=False, precomp_sim=None):
    """
    P3-initialised LP: all test nodes start with their P3 (multimodal) label;
    LP smooths these labels through the audio K-NN graph.

    Corrects hard P4 cases where audio_e is far from its class's global prototype
    but surrounded by correctly P3-labelled audio-similar neighbours.

    test_embs  : (N, D) L2-normalised audio embeddings for all test samples
    p3_labels  : (N,) int — multimodal predictions (used as initial labels)
    alpha      : propagation weight; 0 → pure P3, →1 → pure graph
    Returns    : (N,) refined labels for P4
    """
    S = _build_lp_graph(test_embs.astype(np.float32), k, mutual=mutual, doubly_stochastic=doubly_stochastic, precomp_sim=precomp_sim)
    Y = np.zeros((test_embs.shape[0], n_classes), dtype=np.float32)
    for i, c in enumerate(p3_labels):
        Y[i, c] = 1.0
    F = Y.copy()
    for _ in range(n_iters):
        F = alpha * (S @ F) + (1.0 - alpha) * Y
    return F.argmax(axis=1)


def graph_lp_soft_init(test_embs, p3_sims, n_classes, k=5, alpha=0.6, temp=0.1, n_iters=50, mutual=False, doubly_stochastic=False):
    """
    Soft-initialised LP: initial label matrix Y is softmax(p3_sims / temp).

    Uncertain P3 nodes get a flatter prior → LP can more easily correct them
    via audio-similar neighbours. High-confidence P3 nodes stay near their label.

    test_embs : (N, D) L2-normalised audio embeddings
    p3_sims   : (N, C) cosine similarities to each class prototype (raw scores)
    temp      : softmax temperature; lower = sharper (closer to hard labels)
    """
    import scipy.special
    S  = _build_lp_graph(test_embs.astype(np.float32), k, mutual=mutual, doubly_stochastic=doubly_stochastic)
    Y  = scipy.special.softmax(p3_sims.astype(np.float32) / temp, axis=1)
    F  = Y.copy()
    for _ in range(n_iters):
        F = alpha * (S @ F) + (1.0 - alpha) * Y
    return F.argmax(axis=1)


def adapt_bn_to_test(model, all_face, all_audio, device, chunk=256):
    """
    AdaBN: replace BN running stats with test-set statistics.
    Only BN layers are set to train mode; dropout stays off.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.momentum = None   # cumulative moving average → exact batch stats
            m.training = True   # BN train mode; dropout untouched (stays eval)
    with torch.no_grad():
        for i in range(0, len(all_audio), chunk):
            a = all_audio[i:i+chunk].to(device)
            f = all_face[i:i+chunk].to(device)
            model(f, a, mask_face=0)
            model(f, a, mask_face=1)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.training = False
    print(f"  AdaBN: updated BN stats from {len(all_audio)} test samples")


def enforce_cross_lang_consistency(seen_csv, unseen_csv, p3, p4, p5, p6, en_conf, ur_conf):
    """
    For keys shared between EN and UR val sets, enforce that predictions agree.
    Conflict resolution: pick the prediction with higher fused-embedding confidence.
    Applies to both multimodal (P3/P5) and audio-only (P4/P6) predictions.
    """
    en_key_to_idx = {k: i for i, k in enumerate(seen_csv["key"])}
    p3, p4 = p3.copy(), p4.copy()
    p5, p6 = p5.copy(), p6.copy()
    n_mm_agree = n_mm_resolve = 0
    n_ao_agree = n_ao_resolve = 0
    for j, key in enumerate(unseen_csv["key"]):
        if key not in en_key_to_idx:
            continue
        i = en_key_to_idx[key]
        use_en = en_conf[i] >= ur_conf[j]
        # multimodal
        if p3[i] == p5[j]:
            n_mm_agree += 1
        else:
            winner = p3[i] if use_en else p5[j]
            p3[i] = p5[j] = winner
            n_mm_resolve += 1
        # audio-only
        if p4[i] == p6[j]:
            n_ao_agree += 1
        else:
            winner = p4[i] if use_en else p6[j]
            p4[i] = p6[j] = winner
            n_ao_resolve += 1
    print(f"  Cross-lang consistency: multimodal {n_mm_agree} agree / {n_mm_resolve} resolved; "
          f"audio-only {n_ao_agree} agree / {n_ao_resolve} resolved")
    return p3, p4, p5, p6


def collect_train_embeddings(model, config, device, audio_encoder="ecappa_feats_path", audio_encoder2=None):
    """Per-sample L2-normalised fused embeddings + class labels from all training data."""
    seen_df   = pd.read_csv(f"{config.data_dir}/Csv_files/{config.version}_train_{config.seen_lang}.csv")
    unseen_df = pd.read_csv(f"{config.data_dir}/Csv_files/{config.version}_train_{config.unseen_lang}.csv")
    train_df  = pd.concat([seen_df, unseen_df], ignore_index=True)
    loader    = DataLoader(
        LoadData(config, df=train_df, audio_encoder=audio_encoder, audio_encoder2=audio_encoder2),
        batch_size=256, shuffle=False, num_workers=0,
    )
    all_embs, all_labels = [], []
    with torch.no_grad():
        for audio, face, labels in loader:
            audio, face, labels = audio.to(device), face.to(device), labels.to(device)
            fused = F.normalize(model(face, audio, mask_face=0)["audio_e"], dim=1)
            all_embs.append(fused.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.vstack(all_embs), np.concatenate(all_labels)


def collect_train_fused_embs(model, config, device, audio_encoder="ecappa_feats_path"):
    """Per-sample L2-normalised fused embeddings (mask_face=1) + class labels from all training data."""
    seen_df   = pd.read_csv(f"{config.data_dir}/Csv_files/{config.version}_train_{config.seen_lang}.csv")
    unseen_df = pd.read_csv(f"{config.data_dir}/Csv_files/{config.version}_train_{config.unseen_lang}.csv")
    train_df  = pd.concat([seen_df, unseen_df], ignore_index=True)
    loader    = DataLoader(
        LoadData(config, df=train_df, audio_encoder=audio_encoder),
        batch_size=256, shuffle=False, num_workers=0,
    )
    all_embs, all_labels = [], []
    with torch.no_grad():
        for audio, face, labels in loader:
            audio, face, labels = audio.to(device), face.to(device), labels.to(device)
            fused = F.normalize(model(face, audio, mask_face=1)["fused"], dim=1)
            all_embs.append(fused.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.vstack(all_embs), np.concatenate(all_labels)


class LDASpeakerScorer:
    """LDA discriminative projection + cosine scoring (simplified PLDA backend)."""

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.W_  = None
        self.mu_ = None

    def fit(self, embs, labels):
        from scipy.linalg import eigh
        classes  = np.unique(labels)
        mu       = embs.mean(axis=0)
        D        = embs.shape[1]
        Sw       = np.zeros((D, D), dtype=np.float64)
        Sb       = np.zeros((D, D), dtype=np.float64)
        embs64   = embs.astype(np.float64)
        mu64     = mu.astype(np.float64)
        for c in classes:
            Xc  = embs64[labels == c]
            mc  = Xc.mean(axis=0)
            diff = Xc - mc
            Sw  += diff.T @ diff
            db   = (mc - mu64).reshape(-1, 1)
            Sb  += len(Xc) * (db @ db.T)
        n_comp = self.n_components or min(len(classes) - 1, D)
        lo = D - n_comp
        _, eigvecs = eigh(Sb, Sw + 1e-6 * np.eye(D), subset_by_index=[lo, D - 1])
        self.W_  = eigvecs[:, ::-1].astype(np.float32)
        self.mu_ = mu.astype(np.float32)
        return self

    def transform(self, embs):
        return (embs.astype(np.float32) - self.mu_) @ self.W_

    def score(self, test_embs, proto_embs):
        te = self.transform(test_embs)
        pr = self.transform(proto_embs)
        te = te / (np.linalg.norm(te, axis=1, keepdims=True) + 1e-8)
        pr = pr / (np.linalg.norm(pr, axis=1, keepdims=True) + 1e-8)
        return te @ pr.T


def as_norm_scores(sims, top_n=10):
    """AS-Norm: adaptive score normalisation using top-N impostor cohort statistics."""
    out = np.empty_like(sims)
    for i in range(len(sims)):
        row    = sims[i]
        cohort = np.argsort(row)[-top_n:]
        mu     = row[cohort].mean()
        sig    = row[cohort].std() + 1e-8
        out[i] = (row - mu) / sig
    return out


def k_reciprocal_rerank(embs: np.ndarray, k1: int = 20, k2: int = 6,
                        lambda_val: float = 0.3) -> np.ndarray:
    """
    Zhong et al. CVPR 2017: Re-ranking Person Re-ID with k-reciprocal Encoding.
    Produces a Jaccard-smoothed affinity matrix more robust than raw cosine sim for LP.
    embs      : (N, D) L2-normalised embeddings
    k1        : size of initial k-reciprocal neighbourhood
    k2        : expansion neighbourhood size
    lambda_val: blend weight  (0 = pure cosine, 1 = pure Jaccard)
    Returns   : (N, N) refined affinity (non-negative, higher = more similar)
    """
    n = len(embs)
    sim = (embs @ embs.T).astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    ranks = np.argsort(-sim, axis=1)   # descending rank indices (N, N)

    # Build k-reciprocal sets: j ∈ R(i,k1)  iff  i ∈ NN_k1(j)
    k_rec = []
    for i in range(n):
        nn_i = set(ranks[i, :k1].tolist())
        rec  = {j for j in nn_i if i in set(ranks[j, :k1].tolist())}
        k_rec.append(rec)

    # Expand each k-reciprocal set with k2-NN of compatible members
    k_rec_exp = []
    for i in range(n):
        R_i = set(k_rec[i])
        for j in list(R_i):
            R_j = k_rec[j]
            if len(R_j & R_i) >= 2 / 3 * len(R_i):
                R_i |= set(ranks[j, :k2].tolist())
        k_rec_exp.append(R_i)

    # Build weight matrix V then normalise → Jaccard affinity
    V = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in k_rec_exp[i]:
            V[i, j] = max(sim[i, j], 0.0)
    row_sum = V.sum(axis=1, keepdims=True) + 1e-8
    V_norm  = V / row_sum
    jaccard = V_norm @ V_norm.T   # (N, N)

    return (1.0 - lambda_val) * np.maximum(sim, 0.0) + lambda_val * jaccard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",           default=None)
    parser.add_argument("--audio_encoder",  default="ecappa_feats_path")
    parser.add_argument("--audio_encoder2", default=None)
    parser.add_argument("--mode",   default="p3_smooth",
                        choices=["train_anchors", "p3_smooth", "soft_init"],
                        help="train_anchors: 70 training protos as labeled anchors; "
                             "p3_smooth: hard P3 labels as initial node labels [DEFAULT]; "
                             "soft_init: softmax(P3 cosine sims) as initial label matrix")
    parser.add_argument("--k",              type=int,   default=5,    help="K-NN neighbours (default 5)")
    parser.add_argument("--alpha",          type=float, default=0.6,  help="label spreading weight for both stages (default 0.6)")
    parser.add_argument("--fused_alpha",    type=float, default=None, help="override alpha for fused-embedding LP (P3/P5 refinement)")
    parser.add_argument("--audio_alpha",    type=float, default=None, help="override alpha for audio-embedding LP (P4/P6)")
    parser.add_argument("--fused_k",        type=int,   default=None, help="override k for fused-embedding LP (P3/P5); falls back to --k")
    parser.add_argument("--n_iters",        type=int,   default=50,   help="LP iterations (default 50)")
    parser.add_argument("--soft_temp",      type=float, default=0.1,  help="softmax temperature for soft_init mode (default 0.1)")
    parser.add_argument("--refine_p3p5",        action="store_true", help="also run LP on fused embeddings to refine P3/P5 predictions")
    parser.add_argument("--mutual_knn",          action="store_true", help="use mutual K-NN graph (keep only bidirectional edges)")
    parser.add_argument("--adabn",               action="store_true", help="adapt BN running stats to test data before extracting embeddings")
    parser.add_argument("--cross_lang_consist",  action="store_true", help="enforce same-key EN/UR prediction agreement after LP")
    parser.add_argument("--val_audio_feats_dir", default=None,        help="override val audio feats root (default: config.val_feats_dir)")
    parser.add_argument("--doubly_stochastic",  action="store_true",  help="use doubly stochastic LP affinity (Sinkhorn normalisation)")
    parser.add_argument("--plda",               action="store_true",  help="use LDA-based PLDA scoring for P3/P5 instead of raw cosine")
    parser.add_argument("--as_norm",            action="store_true",  help="apply AS-Norm to P3/P5 cosine scores before argmax")
    parser.add_argument("--as_norm_cohort",     type=int, default=10, help="AS-Norm impostor cohort size (default 10)")
    parser.add_argument("--softmax_p3p5",       action="store_true",  help="use model softmax logits for P3/P5 instead of fused-NN to training prototypes (higher quality anchor for P4/P6 LP)")
    parser.add_argument("--en_k",              type=int,   default=None, help="override k for English LP (P3 fused + P4 audio); falls back to --k")
    parser.add_argument("--ur_k",              type=int,   default=None, help="override k for Urdu LP (P5 fused + P6 audio); falls back to --k")
    parser.add_argument("--en_fused_alpha",    type=float, default=None, help="override fused_alpha for English P3 LP only (Urdu P5 keeps --fused_alpha)")
    parser.add_argument("--face_concat_p3",    action="store_true",   help="for P3 fused LP, concatenate face_e to fused_e (richer graph similarity)")
    parser.add_argument("--face_solo_p3",      action="store_true",   help="replace fused LP for P3 with face-space train_anchors LP (face centroids as labeled nodes)")
    parser.add_argument("--face_only_p3",     action="store_true",   help="use face_e alone (not fused) as LP graph features for P3 refinement")
    parser.add_argument("--face_weight",      type=float, default=1.0, help="scale face_e before concat with fused_e (default 1.0; >1 emphasises face)")
    parser.add_argument("--ckpt2",               default=None, help="second checkpoint; embeddings from both models are concatenated for richer LP graph")
    parser.add_argument("--ckpt3",               default=None, help="third checkpoint; embeddings from all three models are concatenated (triple model)")
    parser.add_argument("--val_audio_feats_dir2", default=None, help="val audio feats root for ckpt2 (use when ckpt2 was trained on different features, e.g. titanetfeats)")
    parser.add_argument("--audio_encoder_ckpt2",  default=None, help="training CSV column for ckpt2 audio features (default: same as --audio_encoder)")
    parser.add_argument("--val_audio_feats_dir3", default=None, help="val audio feats root for ckpt3 (e.g. eres2netv2feats); defaults to same as ckpt1")
    parser.add_argument("--audio_encoder_ckpt3",  default=None, help="training CSV column for ckpt3 audio features (default: same as --audio_encoder)")
    parser.add_argument("--wavlm_p4_dir",     default=None,           help="dir of raw WavLM-SV val features (.npy) to use as P4/P6 LP graph (instead of model audio_e)")
    parser.add_argument("--decouple_p4",     action="store_true",    help="seed P4/P6 LP from audio-only model logits (mask_face=0) instead of P3/P5 — allows P3≠P4")
    parser.add_argument("--use_ur_face",     action="store_true",    help="use real Urdu face features (mask_face=0) for P5 fused embedding and matching training prototypes")
    parser.add_argument("--train_sample_anchors", action="store_true",
                        help="use all per-sample training embeddings as labeled LP nodes for P3/P4/P5/P6 (replaces 70-proto approach)")
    parser.add_argument("--train_sample_p4_only", action="store_true",
                        help="use per-sample training audio_e as labeled nodes for P4/P6 only; keep current P3/P5")
    parser.add_argument("--transductive_p46", action="store_true",
                        help="P4/P6: build test audio prototypes from P3/P5 pseudo-labels then cosine NN (transductive; in-domain for both EN P4 and UR P6, avoids zeroing)")
    parser.add_argument("--avg_ckpts", nargs="+", default=None,
                        help="additional checkpoint paths whose audio_e are averaged (not concat) with ckpt1 before transductive centroid building")
    parser.add_argument("--transductive_lp", action="store_true",
                        help="P4/P6: build test audio centroids from P3/P5 pseudo-labels, then use as labeled anchors in graph LP (vs pure cosine NN in --transductive_p46)")
    parser.add_argument("--ur_face_weight", type=float, default=0.0,
                        help="face blend weight for Urdu P5 similarity: 0=fused only (default), 1=face_e only; intermediate values blend both")
    parser.add_argument("--diffusion_en",   default=None, help=".npy file of diffusion-denoised EN audio embeddings (replaces model audio_e for P4 LP graph)")
    parser.add_argument("--diffusion_ur",   default=None, help=".npy file of diffusion-denoised UR audio embeddings (replaces model audio_e for P6 LP graph)")
    parser.add_argument("--krerank",        action="store_true", help="use k-reciprocal Jaccard reranking affinity for P4/P6 LP graph (Zhong et al. CVPR 2017)")
    parser.add_argument("--krerank_k1",     type=int,   default=20,  help="k-reciprocal primary neighbourhood size (default 20)")
    parser.add_argument("--krerank_k2",     type=int,   default=6,   help="k-reciprocal expansion neighbourhood size (default 6)")
    parser.add_argument("--krerank_lambda", type=float, default=0.3, help="Jaccard blend weight (0=cosine, 1=Jaccard; default 0.3)")
    parser.add_argument("--split",          default="val", choices=["val", "test"], help="which split to run inference on (default: val)")
    args = parser.parse_args()

    config = ExperimentConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    SPLIT  = args.split

    seen_csv   = pd.read_csv(f"{config.data_dir}/Csv_files/{config.version}_{SPLIT}_{config.seen_lang}.csv")
    unseen_csv = pd.read_csv(f"{config.data_dir}/Csv_files/{config.version}_{SPLIT}_{config.unseen_lang}.csv")

    face_feats_dir  = config.val_feats_dir
    audio_feats_dir = args.val_audio_feats_dir or config.val_feats_dir
    seen_audio,   seen_face   = load_val_feats(seen_csv,   audio_feats_dir, device, face_feats_dir)
    unseen_audio, unseen_face = load_val_feats(unseen_csv, audio_feats_dir, device, face_feats_dir)

    ckpt_path = args.ckpt
    print(f"Loading checkpoint: {ckpt_path}")
    _bn = os.path.basename(ckpt_path)
    config.use_audio_memory       = "_mem"        in _bn
    config.use_domain_adv         = ("_adv"       in _bn) or ("_crossmodal" in _bn)
    config.use_arcface            = "_arcface"    in _bn
    config.use_deep_audio         = "_deepaudio"  in _bn
    config.use_crossmodal_distill = "_crossmodal" in _bn
    if "_adaptive_gate" in _bn:   config.fusion = "adaptive_gate"
    elif "_gated"       in _bn:   config.fusion = "gated"
    model = MaskedFOP(config=config, face_dim=seen_face.shape[1], audio_dim=seen_audio.shape[1])
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    # ── Optional second checkpoint for dual-embed concat ─────────────────────
    model2 = None
    seen_audio2 = seen_audio
    unseen_audio2 = unseen_audio
    audio_encoder_ckpt2 = args.audio_encoder_ckpt2 or args.audio_encoder
    if args.val_audio_feats_dir2:
        print(f"Loading val audio features for ckpt2 from: {args.val_audio_feats_dir2}")
        seen_audio2,   _ = load_val_feats(seen_csv,   args.val_audio_feats_dir2, device, face_feats_dir)
        unseen_audio2, _ = load_val_feats(unseen_csv, args.val_audio_feats_dir2, device, face_feats_dir)

    # ── Optional third-checkpoint val audio features ──────────────────────────
    seen_audio3   = seen_audio
    unseen_audio3 = unseen_audio
    audio_encoder_ckpt3 = args.audio_encoder_ckpt3 or args.audio_encoder
    if args.val_audio_feats_dir3:
        print(f"Loading val audio features for ckpt3 from: {args.val_audio_feats_dir3}")
        seen_audio3,   _ = load_val_feats(seen_csv,   args.val_audio_feats_dir3, device, face_feats_dir)
        unseen_audio3, _ = load_val_feats(unseen_csv, args.val_audio_feats_dir3, device, face_feats_dir)
    if args.ckpt2:
        print(f"Loading second checkpoint: {args.ckpt2}")
        _bn2 = os.path.basename(args.ckpt2)
        config2 = ExperimentConfig()
        config2.use_audio_memory       = "_mem"        in _bn2
        config2.use_domain_adv         = ("_adv"       in _bn2) or ("_crossmodal" in _bn2)
        config2.use_arcface            = "_arcface"    in _bn2
        config2.use_deep_audio         = "_deepaudio"  in _bn2
        config2.use_crossmodal_distill = "_crossmodal" in _bn2
        if "_adaptive_gate" in _bn2:  config2.fusion = "adaptive_gate"
        elif "_gated"       in _bn2:  config2.fusion = "gated"
        model2 = MaskedFOP(config=config2, face_dim=seen_face.shape[1], audio_dim=seen_audio2.shape[1])
        ckpt2  = torch.load(args.ckpt2, map_location=device, weights_only=False)
        model2.load_state_dict(ckpt2["model_state"])
        model2 = model2.to(device).eval()

    # ── AdaBN: adapt BN running stats to test distribution ───────────────────
    if args.adabn:
        print("Adapting BN stats to test distribution (AdaBN)...")
        all_test_audio = torch.cat([seen_audio, unseen_audio], dim=0)
        all_test_face  = torch.cat([seen_face,  unseen_face],  dim=0)
        adapt_bn_to_test(model, all_test_face, all_test_audio, device)

    # ── Optional third checkpoint for triple-embed concat ────────────────────
    model3 = None
    if args.ckpt3:
        print(f"Loading third checkpoint: {args.ckpt3}")
        _bn3 = os.path.basename(args.ckpt3)
        config3 = ExperimentConfig()
        config3.use_audio_memory       = "_mem"        in _bn3
        config3.use_domain_adv         = ("_adv"       in _bn3) or ("_crossmodal" in _bn3)
        config3.use_arcface            = "_arcface"    in _bn3
        config3.use_deep_audio         = "_deepaudio"  in _bn3
        config3.use_crossmodal_distill = "_crossmodal" in _bn3
        if "_adaptive_gate" in _bn3:  config3.fusion = "adaptive_gate"
        elif "_gated"       in _bn3:  config3.fusion = "gated"
        model3 = MaskedFOP(config=config3, face_dim=seen_face.shape[1], audio_dim=seen_audio3.shape[1])
        ckpt3  = torch.load(args.ckpt3, map_location=device, weights_only=False)
        model3.load_state_dict(ckpt3["model_state"])
        model3 = model3.to(device).eval()

    # ── Training prototypes ──────────────────────────────────────────────────
    print("Building training fused prototypes (for P3/P5)...")
    proto_av = build_train_proto_av(model, config, device, args.audio_encoder, args.audio_encoder2)
    if model2 is not None:
        proto_av2 = build_train_proto_av(model2, config, device, audio_encoder_ckpt2)
        proto_av  = F.normalize(torch.cat([proto_av, proto_av2], dim=1), dim=1)
    if model3 is not None:
        proto_av3 = build_train_proto_av(model3, config, device, audio_encoder_ckpt3)
        proto_av  = F.normalize(torch.cat([proto_av, proto_av3], dim=1), dim=1)

    # ── Test embeddings ──────────────────────────────────────────────────────
    print("Extracting test embeddings...")
    en_audio_e, en_fused = extract_embeddings(model, seen_face,   seen_audio,   device)
    ur_audio_e, ur_fused = extract_embeddings(model, unseen_face, unseen_audio, device)

    # ── Multi-checkpoint audio_e averaging (for transductive centroid quality) ─
    if args.avg_ckpts:
        en_audio_e_list = [en_audio_e]
        ur_audio_e_list = [ur_audio_e]
        for extra_ckpt in args.avg_ckpts:
            print(f"  avg_ckpts: loading {extra_ckpt}")
            _bn_ex = os.path.basename(extra_ckpt)
            config_ex = ExperimentConfig()
            config_ex.use_audio_memory       = "_mem"        in _bn_ex
            config_ex.use_domain_adv         = ("_adv"       in _bn_ex) or ("_crossmodal" in _bn_ex)
            config_ex.use_arcface            = "_arcface"    in _bn_ex
            config_ex.use_deep_audio         = "_deepaudio"  in _bn_ex
            config_ex.use_crossmodal_distill = "_crossmodal" in _bn_ex
            model_ex = MaskedFOP(config=config_ex, face_dim=seen_face.shape[1], audio_dim=seen_audio.shape[1])
            ckpt_ex  = torch.load(extra_ckpt, map_location=device, weights_only=False)
            model_ex.load_state_dict(ckpt_ex["model_state"])
            model_ex = model_ex.to(device).eval()
            en_ae_ex, _ = extract_embeddings(model_ex, seen_face, seen_audio, device)
            ur_ae_ex, _ = extract_embeddings(model_ex, unseen_face, unseen_audio, device)
            en_audio_e_list.append(en_ae_ex)
            ur_audio_e_list.append(ur_ae_ex)
            del model_ex
        en_audio_e = F.normalize(torch.stack(en_audio_e_list, dim=0).mean(dim=0), dim=1)
        ur_audio_e = F.normalize(torch.stack(ur_audio_e_list, dim=0).mean(dim=0), dim=1)
        print(f"  avg_ckpts: averaged {len(en_audio_e_list)} checkpoints, audio_e shape={en_audio_e.shape}")

    proto_av_face = None
    ur_fused_face = None
    if args.use_ur_face:
        print("use_ur_face: computing Urdu fused embeddings with face enabled (mask_face=0)...")
        with torch.no_grad():
            ur_fused_face = F.normalize(
                model(unseen_face.to(device), unseen_audio.to(device), mask_face=1)["fused"], dim=1
            ).cpu()
        print("use_ur_face: building face-informed training prototypes (mask_face=0)...")
        proto_av_face = build_train_proto_av_face(model, config, device, args.audio_encoder)

    if model2 is not None:
        en_audio_e2, en_fused2 = extract_embeddings(model2, seen_face,   seen_audio2,   device)
        ur_audio_e2, ur_fused2 = extract_embeddings(model2, unseen_face, unseen_audio2, device)
        en_audio_e = F.normalize(torch.cat([en_audio_e, en_audio_e2], dim=1), dim=1)
        en_fused   = F.normalize(torch.cat([en_fused,   en_fused2],   dim=1), dim=1)
        ur_audio_e = F.normalize(torch.cat([ur_audio_e, ur_audio_e2], dim=1), dim=1)
        ur_fused   = F.normalize(torch.cat([ur_fused,   ur_fused2],   dim=1), dim=1)
    if model3 is not None:
        en_audio_e3, en_fused3 = extract_embeddings(model3, seen_face,   seen_audio3,   device)
        ur_audio_e3, ur_fused3 = extract_embeddings(model3, unseen_face, unseen_audio3, device)
        en_audio_e = F.normalize(torch.cat([en_audio_e, en_audio_e3], dim=1), dim=1)
        en_fused   = F.normalize(torch.cat([en_fused,   en_fused3],   dim=1), dim=1)
        ur_audio_e = F.normalize(torch.cat([ur_audio_e, ur_audio_e3], dim=1), dim=1)
        ur_fused   = F.normalize(torch.cat([ur_fused,   ur_fused3],   dim=1), dim=1)

    # ── PLDA: fit LDA scorer on training embeddings ──────────────────────────
    lda_scorer = None
    if args.plda:
        print("Fitting LDA backend (PLDA) on training embeddings...")
        tr_embs, tr_labels = collect_train_embeddings(
            model, config, device, args.audio_encoder, args.audio_encoder2)
        lda_scorer = LDASpeakerScorer().fit(tr_embs, tr_labels)
        print(f"  LDA fitted: {tr_embs.shape[0]} samples → {lda_scorer.W_.shape[1]}-dim space")

    # ── P3/P5: multimodal fused nearest prototype ─────────────────────────────
    if lda_scorer is not None:
        en_sims = lda_scorer.score(en_fused.numpy(), proto_av.numpy())
        ur_sims = lda_scorer.score(ur_fused.numpy(), proto_av.numpy())
    else:
        en_sims = (en_fused @ proto_av.T).numpy()   # (N_en, C) — kept for soft_init
        if args.use_ur_face:
            ur_sims = (ur_fused_face @ proto_av_face.T).numpy()
        elif args.ur_face_weight > 0.0:
            w = args.ur_face_weight
            print(f"  Urdu P5: blending fused (w={1-w:.2f}) + face_e (w={w:.2f}) for P5 similarity...")
            with torch.no_grad():
                ur_face_e = F.normalize(
                    model(unseen_face.to(device), unseen_audio.to(device), mask_face=1)["face_e"],
                    dim=1,
                ).cpu()
            proto_face_en = build_train_face_proto_en_only(model, config, device, args.audio_encoder)
            ur_fused_w = F.normalize((1 - w) * ur_fused + w * ur_face_e,    dim=1)
            proto_av_w = F.normalize((1 - w) * proto_av  + w * proto_face_en, dim=1)
            ur_sims = (ur_fused_w @ proto_av_w.T).numpy()   # (N_ur, C)
        else:
            ur_sims = (ur_fused @ proto_av.T).numpy()   # (N_ur, C)

    if args.as_norm:
        print(f"  Applying AS-Norm (cohort={args.as_norm_cohort})...")
        en_sims = as_norm_scores(en_sims, args.as_norm_cohort)
        ur_sims = as_norm_scores(ur_sims, args.as_norm_cohort)

    p3 = en_sims.argmax(axis=1)
    p5 = ur_sims.argmax(axis=1)

    # ── Softmax-anchored override: replace fused-NN P3/P5 with direct softmax ─
    if args.softmax_p3p5:
        print("Overriding P3/P5 with model softmax logits (better anchor for P4/P6 LP)...")
        model.eval()
        with torch.no_grad():
            p3 = model(seen_face,   seen_audio,   mask_face=1)["logits"].argmax(1).cpu().numpy()
            p5 = model(unseen_face, unseen_audio, mask_face=1)["logits"].argmax(1).cpu().numpy()
        print(f"  Softmax P3: {len(set(p3.tolist()))} classes | P5: {len(set(p5.tolist()))} classes")

    C           = config.resolved_num_classes
    fused_alpha    = args.fused_alpha    if args.fused_alpha    is not None else args.alpha
    audio_alpha    = args.audio_alpha    if args.audio_alpha    is not None else args.alpha
    en_fused_alpha = args.en_fused_alpha if args.en_fused_alpha is not None else fused_alpha
    fused_k     = args.fused_k     if args.fused_k     is not None else args.k
    en_k        = args.en_k        if args.en_k        is not None else args.k
    ur_k        = args.ur_k        if args.ur_k        is not None else args.k
    en_fused_k  = args.en_k        if args.en_k        is not None else fused_k
    ur_fused_k  = args.ur_k        if args.ur_k        is not None else fused_k

    # ── Per-sample training embeddings (train_sample LP modes) ──────────────────
    tr_audio_embs = tr_audio_labels = tr_fused_embs = tr_fused_labels = None
    if args.train_sample_anchors or args.train_sample_p4_only:
        print("Collecting per-sample training audio embeddings for LP...")
        tr_audio_embs, tr_audio_labels = collect_train_embeddings(
            model, config, device, args.audio_encoder, args.audio_encoder2)
        print(f"  {len(tr_audio_embs)} training samples, {len(np.unique(tr_audio_labels))} classes")
    if args.train_sample_anchors:
        print("Collecting per-sample training fused embeddings for LP...")
        tr_fused_embs, tr_fused_labels = collect_train_fused_embs(
            model, config, device, args.audio_encoder)

    # Extract face_e for English if needed
    en_face_e = None
    if args.face_concat_p3 or args.face_solo_p3 or args.face_only_p3:
        print("Extracting English face embeddings (for face-LP)...")
        en_face_e = extract_face_embs(model, seen_face, seen_audio, device)
        if model2 is not None:
            en_face_e2 = extract_face_embs(model2, seen_face, seen_audio, device)
            en_face_e  = F.normalize(torch.cat([en_face_e, en_face_e2], dim=1), dim=1)

    if args.refine_p3p5:
        if args.face_only_p3:
            # Pure face-similarity graph for P3 LP (face_e alone, not fused)
            en_p3_embs = en_face_e.numpy()
            print(f"  Refining P3 with face-ONLY LP (k={en_fused_k}, alpha={fused_alpha}), P5 with fused LP (k={ur_fused_k})...")
        elif args.face_concat_p3:
            # Richer P3 graph: normalize(scale*face_e || fused_e)
            scaled_face = en_face_e * args.face_weight
            en_p3_embs = F.normalize(torch.cat([scaled_face, en_fused], dim=1), dim=1).numpy()
            print(f"  Refining P3 with face+fused LP (face_weight={args.face_weight}, k={en_fused_k}, alpha={en_fused_alpha}), P5 with fused LP (k={ur_fused_k}, alpha={fused_alpha})...")
        else:
            en_p3_embs = en_fused.numpy()
            print(f"  Refining P3/P5 with fused-embedding LP (en_k={en_fused_k}, ur_k={ur_fused_k}, alpha={fused_alpha})...")
        ur_p5_embs = ur_fused_face.numpy() if args.use_ur_face else ur_fused.numpy()
        p3 = graph_lp_p3_smooth(en_p3_embs,  p3, C, en_fused_k, en_fused_alpha, args.n_iters, mutual=args.mutual_knn, doubly_stochastic=args.doubly_stochastic)
        p5 = graph_lp_p3_smooth(ur_p5_embs,  p5, C, ur_fused_k, fused_alpha,    args.n_iters, mutual=args.mutual_knn, doubly_stochastic=args.doubly_stochastic)

    if args.face_solo_p3:
        print("Replacing P3 with face train_anchors LP (face centroids as labeled nodes)...")
        proto_face_train = build_train_face_proto(model, config, device, args.audio_encoder)
        p3 = graph_lp(proto_face_train.numpy(), np.arange(C), en_face_e.numpy(), C,
                      en_fused_k, fused_alpha, args.n_iters, mutual=args.mutual_knn,
                      doubly_stochastic=args.doubly_stochastic)

    if args.train_sample_anchors:
        print(f"  Train-sample P3/P5 LP ({len(tr_fused_embs)} labeled nodes, en_k={en_k}, ur_k={ur_k}, fused_alpha={fused_alpha})...")
        p3 = graph_lp(tr_fused_embs, tr_fused_labels, en_fused.numpy(), C,
                      en_k, fused_alpha, args.n_iters)
        p5 = graph_lp(tr_fused_embs, tr_fused_labels, ur_fused.numpy(), C,
                      ur_k, fused_alpha, args.n_iters)

    print(f"  P3: {len(set(p3.tolist()))} classes | P5: {len(set(p5.tolist()))} classes")

    # ── Raw WavLM-SV override for P4/P6 LP graph ────────────────────────────
    en_p4_graph = en_audio_e.numpy()
    ur_p6_graph = ur_audio_e.numpy()
    if args.diffusion_en:
        print(f"Loading diffusion-denoised EN embeddings from {args.diffusion_en}...")
        en_p4_graph = np.load(args.diffusion_en).astype("float32")
        en_p4_graph /= (np.linalg.norm(en_p4_graph, axis=1, keepdims=True) + 1e-8)
        print(f"  EN diffusion: {en_p4_graph.shape}")
    if args.diffusion_ur:
        print(f"Loading diffusion-denoised UR embeddings from {args.diffusion_ur}...")
        ur_p6_graph = np.load(args.diffusion_ur).astype("float32")
        ur_p6_graph /= (np.linalg.norm(ur_p6_graph, axis=1, keepdims=True) + 1e-8)
        print(f"  UR diffusion: {ur_p6_graph.shape}")
    if args.wavlm_p4_dir:
        print(f"Loading raw WavLM-SV features for P4/P6 LP from {args.wavlm_p4_dir}...")
        en_wavlm = np.stack([
            np.load(os.path.join(args.wavlm_p4_dir, p.replace(".wav", ".npy"))).astype("float32")
            for p in seen_csv["voices"]
        ])
        ur_wavlm = np.stack([
            np.load(os.path.join(args.wavlm_p4_dir, p.replace(".wav", ".npy"))).astype("float32")
            for p in unseen_csv["voices"]
        ])
        en_p4_graph = (en_wavlm / (np.linalg.norm(en_wavlm, axis=1, keepdims=True) + 1e-8))
        ur_p6_graph = (ur_wavlm / (np.linalg.norm(ur_wavlm, axis=1, keepdims=True) + 1e-8))
        print(f"  WavLM P4 graph: EN {en_p4_graph.shape}, UR {ur_p6_graph.shape}")

    # ── Decouple P4/P6 seed from P3/P5: use audio-only model logits ─────────────
    p4_seed, p6_seed = p3, p5
    if args.decouple_p4:
        print("Decoupling P4/P6 seeds: using audio-only model logits (mask_face=0)...")
        with torch.no_grad():
            p4_seed = model(seen_face,   seen_audio,   mask_face=0)["logits"].argmax(1).cpu().numpy()
            p6_seed = model(unseen_face, unseen_audio, mask_face=0)["logits"].argmax(1).cpu().numpy()
        agree_en = (p4_seed == p3).sum()
        agree_ur = (p6_seed == p5).sum()
        print(f"  P3/P4-seed agreement: {agree_en}/{len(p3)} ({100*agree_en/len(p3):.1f}%)")
        print(f"  P5/P6-seed agreement: {agree_ur}/{len(p5)} ({100*agree_ur/len(p5):.1f}%)")

    # ── K-reciprocal reranking: build refined affinity matrices for P4/P6 LP ─
    en_krerank_sim = ur_krerank_sim = None
    if args.krerank:
        print(f"  K-reciprocal reranking (k1={args.krerank_k1}, k2={args.krerank_k2}, lambda={args.krerank_lambda})...")
        en_krerank_sim = k_reciprocal_rerank(en_p4_graph, args.krerank_k1, args.krerank_k2, args.krerank_lambda)
        ur_krerank_sim = k_reciprocal_rerank(ur_p6_graph, args.krerank_k1, args.krerank_k2, args.krerank_lambda)
        print(f"  EN affinity: min={en_krerank_sim.min():.4f} max={en_krerank_sim.max():.4f}")

    # ── P4/P6: graph label propagation ──────────────────────────────────────
    print(f"Running graph LP for P4/P6 (en_k={en_k}, ur_k={ur_k}, audio_alpha={audio_alpha}, iters={args.n_iters})...")

    if args.transductive_p46:
        print("Building transductive test audio prototypes from P3/P5 pseudo-labels...")
        proto_a_en = build_test_audio_proto(en_audio_e, p3, C)
        proto_a_ur = build_test_audio_proto(ur_audio_e, p5, C)
        p4 = cosine_nn(en_audio_e, proto_a_en)
        p6 = cosine_nn(ur_audio_e, proto_a_ur)
        agree_en = (p4 == p3).sum()
        agree_ur = (p6 == p5).sum()
        print(f"  Transductive P4 agrees with P3: {agree_en}/{len(p3)} ({100*agree_en/len(p3):.1f}%)")
        print(f"  Transductive P6 agrees with P5: {agree_ur}/{len(p5)} ({100*agree_ur/len(p5):.1f}%)")
    elif args.transductive_lp:
        print("Building transductive test audio prototypes from P3/P5 pseudo-labels...")
        proto_a_en = build_test_audio_proto(en_audio_e, p3, C)
        proto_a_ur = build_test_audio_proto(ur_audio_e, p5, C)
        print(f"  Transductive LP: 70 test centroids as anchors, en_k={en_k}, ur_k={ur_k}, audio_alpha={audio_alpha}...")
        p4 = graph_lp(proto_a_en.numpy(), np.arange(C), en_p4_graph, C, en_k, audio_alpha, args.n_iters)
        p6 = graph_lp(proto_a_ur.numpy(), np.arange(C), ur_p6_graph, C, ur_k, audio_alpha, args.n_iters)
        agree_en = (p4 == p3).sum()
        agree_ur = (p6 == p5).sum()
        print(f"  Transductive LP P4 agrees with P3: {agree_en}/{len(p3)} ({100*agree_en/len(p3):.1f}%)")
        print(f"  Transductive LP P6 agrees with P5: {agree_ur}/{len(p5)} ({100*agree_ur/len(p5):.1f}%)")
    elif args.train_sample_anchors or args.train_sample_p4_only:
        print(f"  Train-sample LP: {len(tr_audio_embs)} labeled nodes, en_k={en_k}, ur_k={ur_k}...")
        p4 = graph_lp(tr_audio_embs, tr_audio_labels, en_p4_graph, C,
                      en_k, audio_alpha, args.n_iters)
        p6 = graph_lp(tr_audio_embs, tr_audio_labels, ur_p6_graph, C,
                      ur_k, audio_alpha, args.n_iters)
    elif args.mode == "p3_smooth":
        p4 = graph_lp_p3_smooth(en_p4_graph, p4_seed, C, en_k, audio_alpha, args.n_iters, mutual=args.mutual_knn, doubly_stochastic=args.doubly_stochastic, precomp_sim=en_krerank_sim)
        p6 = graph_lp_p3_smooth(ur_p6_graph, p6_seed, C, ur_k, audio_alpha, args.n_iters, mutual=args.mutual_knn, doubly_stochastic=args.doubly_stochastic, precomp_sim=ur_krerank_sim)
    elif args.mode == "soft_init":
        p4 = graph_lp_soft_init(en_p4_graph, en_sims, C, en_k, args.alpha, args.soft_temp, args.n_iters, mutual=args.mutual_knn, doubly_stochastic=args.doubly_stochastic)
        p6 = graph_lp_soft_init(ur_p6_graph, ur_sims, C, ur_k, args.alpha, args.soft_temp, args.n_iters, mutual=args.mutual_knn, doubly_stochastic=args.doubly_stochastic)
    else:
        # train_anchors: 70 training audio prototypes as labeled, all test as unlabeled.
        print("  Building training audio prototypes (labeled anchors)...")
        proto_a_train  = build_train_audio_proto(model, config, device, args.audio_encoder, args.audio_encoder2)
        labeled_embs   = proto_a_train.numpy()
        labeled_labels = np.arange(C)
        p4 = graph_lp(labeled_embs, labeled_labels, en_p4_graph, C, en_k, args.alpha, args.n_iters, mutual=args.mutual_knn, doubly_stochastic=args.doubly_stochastic)
        p6 = graph_lp(labeled_embs, labeled_labels, ur_p6_graph, C, ur_k, args.alpha, args.n_iters, mutual=args.mutual_knn, doubly_stochastic=args.doubly_stochastic)

    # ── Cross-language consistency enforcement ────────────────────────────────
    if args.cross_lang_consist:
        print("Enforcing cross-language consistency...")
        en_conf = en_sims.max(axis=1)   # max cosine sim to any training proto
        ur_conf = ur_sims.max(axis=1)
        p3, p4, p5, p6 = enforce_cross_lang_consistency(
            seen_csv, unseen_csv, p3, p4, p5, p6, en_conf, ur_conf
        )

    os.makedirs("./csv_files/submission", exist_ok=True)
    pd.DataFrame({"key": seen_csv["key"],   "p3": p3, "p4": p4}).to_csv(
        f"./csv_files/submission/submission_{config.version}_{SPLIT}_{config.seen_lang}_{config.seen_lang}.csv",
        index=False,
    )
    pd.DataFrame({"key": unseen_csv["key"], "p5": p5, "p6": p6}).to_csv(
        f"./csv_files/submission/submission_{config.version}_{SPLIT}_{config.seen_lang}_{config.unseen_lang}.csv",
        index=False,
    )
    print(f"Saved: {len(seen_csv)} EN rows (P3,P4), {len(unseen_csv)} UR rows (P5,P6)")


if __name__ == "__main__":
    main()
