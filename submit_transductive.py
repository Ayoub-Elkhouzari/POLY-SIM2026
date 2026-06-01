"""
Transductive prototype inference for POLY-SIM 2026.

Key idea: P3 (EN multimodal, ~99%) and P5 (UR multimodal, ~99%) are highly accurate.
We use their predictions as pseudo-labels to build test-derived audio-only prototypes
that are anchored to the actual test distribution — eliminating train/test audio shift.

  1. Get P3 preds (EN multimodal cosine NN)  → pseudo-labels for EN test audio
  2. Get P5 preds (UR multimodal cosine NN)  → pseudo-labels for UR test audio
  3. Build EN audio prototypes from P3-labeled test audio
  4. Build UR audio prototypes from P5-labeled test audio
  5. P4 = cosine NN(audio_en, proto_a_en_test)
     P6 = cosine NN(audio_ur, proto_a_ur_test)

Expected: P4 ≈ P3 accuracy, P6 ≈ P5 accuracy.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ExperimentConfig
from models.masked_fop import MaskedFOP
from utils.featLoader import LoadData


def load_val_feats(csv_df, audio_feats_dir, device, face_feats_dir=None, audio_feats_dir2=None):
    if face_feats_dir is None:
        face_feats_dir = audio_feats_dir
    audio_feats = np.stack([
        np.load(os.path.join(audio_feats_dir, p.replace(".wav", ".npy"))).astype("float32")
        for p in csv_df["voices"]
    ])
    if audio_feats_dir2:
        audio_feats2 = np.stack([
            np.load(os.path.join(audio_feats_dir2, p.replace(".wav", ".npy"))).astype("float32")
            for p in csv_df["voices"]
        ])
        audio_feats = np.concatenate([audio_feats, audio_feats2], axis=1)
    face_feats = np.stack([
        np.load(os.path.join(face_feats_dir, p.replace(".jpg", ".npy"))).astype("float32")
        for p in csv_df["faces"]
    ])
    return (
        torch.from_numpy(audio_feats).to(device),
        torch.from_numpy(face_feats).to(device),
    )


def extract_embeddings(model, face, audio, device):
    """Return L2-normalized audio_e and fused embeddings."""
    face, audio = face.to(device), audio.to(device)
    with torch.no_grad():
        out_a  = model(face, audio, mask_face=0)
        out_av = model(face, audio, mask_face=1)
    audio_e = F.normalize(out_a["audio_e"],  dim=1).cpu()
    fused   = F.normalize(out_av["fused"],   dim=1).cpu()
    return audio_e, fused


def build_train_proto_av(model, config, device, audio_encoder="ecappa_feats_path", audio_encoder2=None):
    """L2-normalized fused per-class centroids from all training data (for P3/P5)."""
    seen_df   = pd.read_csv(f"{config.data_dir}/Csv_files/{config.version}_train_{config.seen_lang}.csv")
    unseen_df = pd.read_csv(f"{config.data_dir}/Csv_files/{config.version}_train_{config.unseen_lang}.csv")
    train_df  = pd.concat([seen_df, unseen_df], ignore_index=True)
    loader    = DataLoader(LoadData(config, df=train_df, audio_encoder=audio_encoder, audio_encoder2=audio_encoder2), batch_size=256, shuffle=False, num_workers=0)

    C, D = config.resolved_num_classes, config.embedding_dim
    sum_av = torch.zeros(C, D, device=device)

    with torch.no_grad():
        for audio, face, labels in loader:
            audio, face, labels = audio.to(device), face.to(device), labels.to(device)
            fused = F.normalize(model(face, audio, mask_face=1)["fused"], dim=1)
            for i, lbl in enumerate(labels):
                sum_av[lbl.item()] += fused[i]

    return F.normalize(sum_av, dim=1).cpu()   # (C, D)


def build_test_audio_proto(audio_embs, pseudo_labels, num_classes, confidences=None, conf_thresh=0.0):
    """
    Build audio prototypes from test embeddings using pseudo-labels.
    audio_embs   : (N, D) L2-normalized, CPU
    pseudo_labels: (N,) int array
    confidences  : (N,) float array of max cosine sim to training proto (optional)
    conf_thresh  : only use samples with confidence >= this value
    Returns      : (C, D) L2-normalized CPU tensor
    """
    D = audio_embs.shape[1]
    acc = torch.zeros(num_classes, D)
    cnt = torch.zeros(num_classes)
    for i, lbl in enumerate(pseudo_labels):
        if confidences is None or confidences[i] >= conf_thresh:
            acc[lbl] += audio_embs[i]
            cnt[lbl] += 1
    # fallback: if a class got no samples above threshold, use all its samples
    if conf_thresh > 0.0:
        for c in range(num_classes):
            if cnt[c] == 0:
                for i, lbl in enumerate(pseudo_labels):
                    if lbl == c:
                        acc[c] += audio_embs[i]
    return F.normalize(acc, dim=1)   # (C, D)


def cosine_nn(embs, protos, k=1):
    """cosine similarity → class predictions; k>1 uses majority-vote k-NN."""
    sims = embs @ protos.T   # (N, C)
    if k == 1:
        return sims.argmax(dim=1).numpy()
    topk_idx = sims.topk(k, dim=1).indices   # (N, k)
    preds = []
    for row in topk_idx.tolist():
        votes = {}
        for idx in row:
            votes[idx] = votes.get(idx, 0) + 1
        preds.append(max(votes, key=votes.get))
    return np.array(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",           default=None)
    parser.add_argument("--feats_dir",      default=None, help="override val audio features root")
    parser.add_argument("--feats_dir2",     default=None, help="second val audio features root to concatenate")
    parser.add_argument("--audio_encoder",  default="ecappa_feats_path", help="CSV column for train audio features")
    parser.add_argument("--audio_encoder2", default=None,                help="CSV column for second train audio features to concatenate")
    parser.add_argument("--knn",            type=int,   default=1,   help="k for majority-vote k-NN on P4/P6 (default=1=argmax)")
    parser.add_argument("--conf_thresh",    type=float, default=0.0, help="min P3/P5 confidence to include sample in audio prototype")
    parser.add_argument("--refine_iters",   type=int,   default=0,   help="extra iterative refinement rounds: use P4/P6 preds to rebuild protos")
    args = parser.parse_args()

    config = ExperimentConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    SPLIT  = "val"

    seen_csv   = pd.read_csv(f"{config.data_dir}/Csv_files/{config.version}_{SPLIT}_{config.seen_lang}.csv")
    unseen_csv = pd.read_csv(f"{config.data_dir}/Csv_files/{config.version}_{SPLIT}_{config.unseen_lang}.csv")

    feats_dir      = args.feats_dir  if args.feats_dir  else config.val_feats_dir
    feats_dir2     = args.feats_dir2 if args.feats_dir2 else None
    face_feats_dir = config.val_feats_dir  # face features always from original dir
    seen_audio,   seen_face   = load_val_feats(seen_csv,   feats_dir, device, face_feats_dir, feats_dir2)
    unseen_audio, unseen_face = load_val_feats(unseen_csv, feats_dir, device, face_feats_dir, feats_dir2)

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        mem_tag = f"_mem{config.audio_memory_size}" if getattr(config, "use_audio_memory", False) else ""
        ckpt_path = (
            f"./checkpoints/{config.version}_{config.model_type}_"
            f"{config.seen_lang}_{config.fusion}_drop{config.modality_dropout_prob}{mem_tag}_best.pt"
        )

    print(f"Loading checkpoint: {ckpt_path}")
    _bn = os.path.basename(ckpt_path)
    config.use_audio_memory       = "_mem"               in _bn
    config.use_domain_adv         = ("_adv" in _bn) or ("_crossmodal" in _bn)
    config.use_arcface            = "_arcface"           in _bn
    config.use_deep_audio         = "_deepaudio"         in _bn
    config.use_crossmodal_distill = "_crossmodal"        in _bn
    model = MaskedFOP(config=config, face_dim=seen_face.shape[1], audio_dim=seen_audio.shape[1])
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    # ── Step 1: training fused prototypes (for P3/P5) ───────────────────────
    print("Building training fused prototypes (for P3/P5)...")
    proto_av = build_train_proto_av(model, config, device, args.audio_encoder, args.audio_encoder2)   # (C, D) CPU

    # ── Step 2: extract test embeddings ─────────────────────────────────────
    print("Extracting test embeddings...")
    en_audio_e, en_fused = extract_embeddings(model, seen_face,   seen_audio,   device)
    ur_audio_e, ur_fused = extract_embeddings(model, unseen_face, unseen_audio, device)

    # ── Step 3: P3/P5 multimodal predictions + confidence ───────────────────
    en_sims = en_fused @ proto_av.T   # (N, C)
    ur_sims = ur_fused @ proto_av.T
    p3 = en_sims.argmax(dim=1).numpy()
    p5 = ur_sims.argmax(dim=1).numpy()
    en_conf = en_sims.max(dim=1).values.numpy()
    ur_conf = ur_sims.max(dim=1).values.numpy()

    C = config.resolved_num_classes
    print(f"  P3 pseudo-label distribution: {len(set(p3.tolist()))} classes covered")
    print(f"  P5 pseudo-label distribution: {len(set(p5.tolist()))} classes covered")
    if args.conf_thresh > 0.0:
        print(f"  EN samples above thresh {args.conf_thresh}: {(en_conf >= args.conf_thresh).sum()}/{len(en_conf)}")
        print(f"  UR samples above thresh {args.conf_thresh}: {(ur_conf >= args.conf_thresh).sum()}/{len(ur_conf)}")

    # ── Step 4: test-derived audio prototypes ────────────────────────────────
    print("Building test-derived audio prototypes (for P4/P6)...")
    proto_a_en = build_test_audio_proto(en_audio_e, p3, C, en_conf, args.conf_thresh)
    proto_a_ur = build_test_audio_proto(ur_audio_e, p5, C, ur_conf, args.conf_thresh)

    # ── Step 5: P4/P6 audio-only predictions ────────────────────────────────
    p4 = cosine_nn(en_audio_e, proto_a_en, k=args.knn)
    p6 = cosine_nn(ur_audio_e, proto_a_ur, k=args.knn)

    # ── Step 6 (optional): iterative proto refinement ───────────────────────
    for refine_i in range(args.refine_iters):
        proto_a_en = build_test_audio_proto(en_audio_e, p4, C)
        proto_a_ur = build_test_audio_proto(ur_audio_e, p6, C)
        p4 = cosine_nn(en_audio_e, proto_a_en, k=args.knn)
        p6 = cosine_nn(ur_audio_e, proto_a_ur, k=args.knn)
        print(f"  [refine iter {refine_i+1}/{args.refine_iters}] done")

    os.makedirs("./csv_files/submission", exist_ok=True)

    pd.DataFrame({"key": seen_csv["key"], "p3": p3, "p4": p4}).to_csv(
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
