"""
Local P-accuracy evaluator for POLY-SIM 2026 submissions.

Usage:
    python eval_local.py                          # auto-detect latest submission zip
    python eval_local.py submission_proto.zip
    python eval_local.py path/to/en.csv path/to/ur.csv

Ground-truth derivation:
  - English val: nearest centroid in English training face embeddings (FaceNet, 512-d)
  - Urdu val:    nearest centroid in Urdu training face embeddings

Local scores track actual server scores closely (~2% gap due to ambiguous faces).
Use for relative comparison between submissions, not as an absolute score.
"""

import argparse
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths (relative to this script's directory = Code/polysim/)
# ---------------------------------------------------------------------------
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT            = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR        = os.path.join(ROOT, "Data")
TRAIN_CSV_EN    = os.path.join(DATA_DIR, "Csv_files", "v1_train_English.csv")
TRAIN_CSV_UR    = os.path.join(DATA_DIR, "Csv_files", "v1_train_Urdu.csv")
TRAIN_FACE_FEAT = os.path.join(DATA_DIR, "Features", "train", "facenetfeats")
VAL_FACE_EN     = os.path.join(DATA_DIR, "Features", "val", "v1", "faces", "English")
VAL_FACE_UR     = os.path.join(DATA_DIR, "Features", "val", "v1", "faces", "Urdu")
VAL_CSV_EN      = os.path.join(DATA_DIR, "Csv_files", "v1_val_English.csv")
VAL_CSV_UR      = os.path.join(DATA_DIR, "Csv_files", "v1_val_Urdu.csv")


# ---------------------------------------------------------------------------
# Build per-class face centroids from one language's training CSV
# ---------------------------------------------------------------------------
def build_centroids(csv_path):
    feats, labels = [], []
    seen = set()
    for _, row in pd.read_csv(csv_path).iterrows():
        rel = row["facenet_feats_path"].replace("./facenetfeats/", "").replace("\\", "/")
        p   = os.path.join(TRAIN_FACE_FEAT, rel)
        if p in seen:
            continue
        seen.add(p)
        try:
            f = np.load(p).astype("float32")
            feats.append(f)
            labels.append(int(row["label"]))
        except FileNotFoundError:
            pass

    feats  = np.stack(feats)
    labels = np.array(labels, dtype=np.int32)
    n_cls  = labels.max() + 1
    D      = feats.shape[1]

    C   = np.zeros((n_cls, D), dtype="float32")
    cnt = np.zeros(n_cls, dtype="float32")
    for f, l in zip(feats, labels):
        n = np.linalg.norm(f)
        C[l] += f / (n + 1e-8)
        cnt[l] += 1.0
    C /= cnt[:, None] + 1e-8
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    return C


def nn_label(feat_path, centroids):
    v = np.load(feat_path).astype("float32")
    v /= np.linalg.norm(v) + 1e-8
    return int(np.argmax(centroids @ v))


# ---------------------------------------------------------------------------
# Derive ground-truth labels for a val CSV
# ---------------------------------------------------------------------------
def derive_gt(val_csv, val_face_dir, centroids):
    """Return {key: true_label}."""
    gt = {}
    for _, row in val_csv.iterrows():
        idx_str   = os.path.splitext(os.path.basename(row["faces"]))[0]
        feat_path = os.path.join(val_face_dir, f"{idx_str}.npy")
        try:
            gt[row["key"]] = nn_label(feat_path, centroids)
        except FileNotFoundError:
            pass
    return gt


# ---------------------------------------------------------------------------
# Load submission (zip or CSV paths)
# ---------------------------------------------------------------------------
def load_submission(args):
    if len(args) == 1 and args[0].endswith(".zip"):
        zpath = args[0] if os.path.isabs(args[0]) else os.path.join(SCRIPT_DIR, args[0])
        with zipfile.ZipFile(zpath) as z:
            names   = z.namelist()
            en_name = next(n for n in names if "English_English" in n)
            ur_name = next(n for n in names if "English_Urdu"    in n)
            df_en   = pd.read_csv(io.BytesIO(z.read(en_name)))
            df_ur   = pd.read_csv(io.BytesIO(z.read(ur_name)))
    elif len(args) == 2:
        df_en = pd.read_csv(args[0])
        df_ur = pd.read_csv(args[1])
    else:
        zips = sorted(
            [f for f in os.listdir(SCRIPT_DIR) if f.endswith(".zip") and "submission" in f],
            key=lambda f: os.path.getmtime(os.path.join(SCRIPT_DIR, f)),
            reverse=True,
        )
        if not zips:
            sys.exit("No submission zip found. Pass a path explicitly.")
        zpath = os.path.join(SCRIPT_DIR, zips[0])
        print(f"Auto-detected: {zips[0]}")
        return load_submission([zpath])
    return df_en, df_ur


# ---------------------------------------------------------------------------
# P-accuracy
# ---------------------------------------------------------------------------
def p_accuracy(df, pred_col, gt):
    correct = total = 0
    for _, row in df.iterrows():
        key = row["key"]
        if key not in gt:
            continue
        total   += 1
        correct += int(row[pred_col]) == gt[key]
    return 100.0 * correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Local POLY-SIM evaluator")
    parser.add_argument("submission", nargs="*", help="zip file or two CSV paths")
    args = parser.parse_args()

    print("Building face centroids …", flush=True)
    c_en = build_centroids(TRAIN_CSV_EN)
    c_ur = build_centroids(TRAIN_CSV_UR)
    print(f"  English centroids: {c_en.shape}  |  Urdu centroids: {c_ur.shape}")

    df_en, df_ur = load_submission(args.submission)
    print(f"  Submission: {len(df_en)} English rows, {len(df_ur)} Urdu rows")

    print("Deriving ground-truth labels …", flush=True)
    val_csv_en = pd.read_csv(VAL_CSV_EN)
    val_csv_ur = pd.read_csv(VAL_CSV_UR)
    gt_en = derive_gt(val_csv_en, VAL_FACE_EN, c_en)
    gt_ur = derive_gt(val_csv_ur, VAL_FACE_UR, c_ur)
    print(f"  GT: {len(gt_en)} English keys, {len(gt_ur)} Urdu keys")

    p3 = p_accuracy(df_en, "p3", gt_en)
    p4 = p_accuracy(df_en, "p4", gt_en)
    p5 = p_accuracy(df_ur, "p5", gt_ur)
    p6 = p_accuracy(df_ur, "p6", gt_ur)
    mean = (p3 + p4 + p5 + p6) / 4.0

    print()
    print(f"  P3 (EN multimodal) : {p3:.4f}%")
    print(f"  P4 (EN audio-only) : {p4:.4f}%")
    print(f"  P5 (UR multimodal) : {p5:.4f}%")
    print(f"  P6 (UR audio-only) : {p6:.4f}%")
    print(f"  ─────────────────────────────────")
    print(f"  Local score (≈)    : {mean/100:.5f}")
    print(f"  Note: ~2% below server score due to face NN ambiguity.")
    print()


if __name__ == "__main__":
    main()
