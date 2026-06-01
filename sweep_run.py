"""
Single sweep run — launched by scripts/sweep.sh as a SLURM array task.
Overrides config fields via CLI, trains, and appends the result to sweep_results.csv.
"""
import argparse
import csv
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig
from models.masked_fop import MaskedFOP
from utils.earlystop import EarlyStopping
from utils.evaluator import Evaluator
from utils.featLoader import LoadData
from utils.trainer import Trainer


def make_dataset(df, config, audio_encoder="ecappa_feats_path", audio_encoder2=None):
    return LoadData(config=config, df=df, audio_encoder=audio_encoder, audio_encoder2=audio_encoder2)


def make_loader(dataset, config, shuffle=False):
    return DataLoader(
        dataset, batch_size=config.batch_size, shuffle=shuffle,
        num_workers=config.num_workers, pin_memory=True, drop_last=shuffle,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_decay",    type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--alpha",           type=float, default=None)
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--dropout_prob",    type=float, default=None)
    parser.add_argument("--use_m3",            type=int,   default=None, help="1=M3, 0=Adam")
    parser.add_argument("--m3_slow_chunk",     type=int,   default=None)
    parser.add_argument("--use_deep_audio",     type=int,   default=None, help="1=use two-layer DeepEmbedBranch for audio")
    parser.add_argument("--use_domain_adv",    type=int,   default=None, help="1=enable adversarial training")
    parser.add_argument("--domain_adv_weight", type=float, default=None, help="lambda for adv loss")
    parser.add_argument("--use_supcon",        type=int,   default=None, help="1=enable supervised contrastive loss on audio_e")
    parser.add_argument("--supcon_weight",     type=float, default=None, help="weight for SupCon loss")
    parser.add_argument("--supcon_temp",       type=float, default=None, help="SupCon temperature")
    parser.add_argument("--use_arcface",       type=int,   default=None, help="1=replace linear heads with ArcFace heads")
    parser.add_argument("--arcface_s",         type=float, default=None, help="ArcFace scale (default 64)")
    parser.add_argument("--arcface_m",         type=float, default=None, help="ArcFace margin (default 0.5)")
    parser.add_argument("--use_triplet",       type=int,   default=None, help="1=enable cross-lingual triplet loss on audio_e")
    parser.add_argument("--triplet_weight",    type=float, default=None, help="triplet loss weight (default 0.1)")
    parser.add_argument("--triplet_margin",    type=float, default=None, help="triplet margin (default 0.3)")
    parser.add_argument("--use_cl_align",      type=int,   default=None, help="1=enable cross-lingual centroid alignment loss")
    parser.add_argument("--cl_align_weight",   type=float, default=None, help="CL alignment loss weight (default 0.3)")
    parser.add_argument("--cl_ema_momentum",   type=float, default=None, help="EMA momentum for CL protos (default 0.99; use 0.9 for fast warmup)")
    parser.add_argument("--audio_encoder",     type=str,   default="ecappa_feats_path", help="CSV column for audio features")
    parser.add_argument("--audio_encoder2",    type=str,   default=None,                help="CSV column for second audio features to concatenate")
    parser.add_argument("--ckpt_name",         type=str,   default=None, help="override output checkpoint filename (no .pt)")
    parser.add_argument("--init_ckpt",         type=str,   default=None, help="path to pre-trained checkpoint to warm-start from (model weights only)")
    parser.add_argument("--max_epochs",        type=int,   default=None, help="override config.max_epochs (useful for fine-tuning)")
    parser.add_argument("--batch_size",          type=int,   default=None, help="mini-batch size (default 32)")
    parser.add_argument("--early_stop_patience",  type=int,   default=None, help="early stopping patience (default 15)")
    parser.add_argument("--noise_std",            type=float, default=None, help="std of Gaussian noise added to audio features during training (default 0)")
    parser.add_argument("--lambda_schedule",      type=int,   default=None, help="1=ramp domain_adv_weight from 0 to target over warmup epochs")
    parser.add_argument("--lambda_warmup_epochs", type=int,   default=None, help="warmup epochs for scheduled lambda (default 30)")
    parser.add_argument("--seed",               type=int,   default=None, help="random seed (default 1)")
    parser.add_argument("--val_frac",           type=float, default=None, help="fraction of train data held out for local validation (default 0.2)")
    parser.add_argument("--use_crossmodal_distill",   type=int,   default=None, help="1=replace masking with cross-modal distillation")
    parser.add_argument("--crossmodal_distill_weight", type=float, default=None, help="weight for cosine distillation loss (default 1.0)")
    parser.add_argument("--use_av_consistency",        type=int,   default=None, help="1=add KL consistency loss between head_av and head_a")
    parser.add_argument("--av_consistency_weight",     type=float, default=None, help="weight for AV consistency KL loss (default 0.5)")
    parser.add_argument("--fusion",            type=str,   default=None, help="fusion type: linear|gated|adaptive_gate|concat")
    parser.add_argument("--train_unseen_lang", type=int,   default=None, help="1=include Urdu batches during training (default True); set 0 for protocol-compliant English-only training")
    parser.add_argument("--run_id",            type=str,   default="0")
    args = parser.parse_args()

    config = ExperimentConfig()
    if args.weight_decay      is not None: config.weight_decay      = args.weight_decay
    if args.label_smoothing   is not None: config.label_smoothing   = args.label_smoothing
    if args.alpha             is not None: config.alpha             = args.alpha
    if args.lr                is not None: config.lr                = args.lr
    if args.dropout_prob      is not None: config.modality_dropout_prob = args.dropout_prob
    if args.use_m3            is not None: config.use_m3            = bool(args.use_m3)
    if args.m3_slow_chunk     is not None: config.m3_slow_chunk     = args.m3_slow_chunk
    if args.use_deep_audio     is not None: config.use_deep_audio     = bool(args.use_deep_audio)
    if args.use_domain_adv    is not None: config.use_domain_adv    = bool(args.use_domain_adv)
    if args.domain_adv_weight is not None: config.domain_adv_weight = args.domain_adv_weight
    if args.use_supcon        is not None: config.use_supcon        = bool(args.use_supcon)
    if args.supcon_weight     is not None: config.supcon_weight     = args.supcon_weight
    if args.supcon_temp       is not None: config.supcon_temp       = args.supcon_temp
    if args.use_arcface       is not None: config.use_arcface       = bool(args.use_arcface)
    if args.arcface_s         is not None: config.arcface_s         = args.arcface_s
    if args.arcface_m         is not None: config.arcface_m         = args.arcface_m
    if args.use_triplet       is not None: config.use_triplet       = bool(args.use_triplet)
    if args.triplet_weight    is not None: config.triplet_weight    = args.triplet_weight
    if args.triplet_margin    is not None: config.triplet_margin    = args.triplet_margin
    if args.use_cl_align      is not None: config.use_cl_align      = bool(args.use_cl_align)
    if args.cl_align_weight   is not None: config.cl_align_weight   = args.cl_align_weight
    if args.cl_ema_momentum   is not None: config.cl_ema_momentum   = args.cl_ema_momentum
    if args.max_epochs        is not None: config.max_epochs        = args.max_epochs
    if args.val_frac          is not None: config.val_frac          = args.val_frac
    if args.batch_size           is not None: config.batch_size           = args.batch_size
    if args.early_stop_patience   is not None: config.early_stop_patience   = args.early_stop_patience
    if args.noise_std             is not None: config.noise_std             = args.noise_std
    if args.lambda_schedule       is not None: config.lambda_schedule       = bool(args.lambda_schedule)
    if args.lambda_warmup_epochs  is not None: config.lambda_warmup_epochs  = args.lambda_warmup_epochs
    if args.seed              is not None: config.seed              = args.seed
    if args.use_crossmodal_distill    is not None: config.use_crossmodal_distill    = bool(args.use_crossmodal_distill)
    if args.crossmodal_distill_weight is not None: config.crossmodal_distill_weight = args.crossmodal_distill_weight
    if args.use_av_consistency        is not None: config.use_av_consistency        = bool(args.use_av_consistency)
    if args.av_consistency_weight     is not None: config.av_consistency_weight     = args.av_consistency_weight
    if args.fusion                    is not None: config.fusion                    = args.fusion
    if args.train_unseen_lang         is not None: config.train_unseen_lang         = bool(args.train_unseen_lang)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train_csv        = f"{config.data_dir}/Csv_files/{config.version}_train_{config.seen_lang}.csv"
    unseen_train_csv = f"{config.data_dir}/Csv_files/{config.version}_train_{config.unseen_lang}.csv"

    df_seen   = pd.read_csv(train_csv)
    df_unseen = pd.read_csv(unseen_train_csv)

    val_seen_df     = df_seen.sample(frac=config.val_frac, random_state=config.seed)
    train_df        = df_seen.drop(val_seen_df.index).reset_index(drop=True)
    val_unseen_df   = df_unseen.sample(frac=config.val_frac, random_state=config.seed)
    unseen_train_df = df_unseen.drop(val_unseen_df.index).reset_index(drop=True)

    ae, ae2 = args.audio_encoder, args.audio_encoder2
    train_dataset      = make_dataset(train_df,      config, ae, ae2)
    val_seen_dataset   = make_dataset(val_seen_df,   config, ae, ae2)
    val_unseen_dataset = make_dataset(val_unseen_df, config, ae, ae2)
    unseen_train_loader = (
        make_loader(make_dataset(unseen_train_df, config, ae, ae2), config, shuffle=True)
        if config.train_unseen_lang else None
    )

    train_loader = make_loader(train_dataset, config, shuffle=True)

    audio, face, _ = next(iter(train_loader))
    audio_dim, face_dim = audio.shape[1], face.shape[1]

    model = MaskedFOP(config=config, face_dim=face_dim, audio_dim=audio_dim)

    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"[sweep] Warm-started from {args.init_ckpt}")

    trainer   = Trainer(model, config)
    evaluator = Evaluator(model, config)

    alpha        = config.alpha
    best_metric  = -float("inf")
    best_epoch   = -1
    best_scores  = {}

    _ckpt_stem = args.ckpt_name if args.ckpt_name else f"sweep_run{args.run_id}"
    ckpt_path = f"./checkpoints/{_ckpt_stem}_best.pt"
    os.makedirs("checkpoints", exist_ok=True)

    early_stopper = EarlyStopping(
        patience=config.early_stop_patience,
        min_delta=config.early_stop_min_delta,
    )

    target_lambda = config.domain_adv_weight  # saved for scheduled λ ramp

    for epoch in range(config.max_epochs):
        if config.use_domain_adv and config.lambda_schedule:
            warmup = config.lambda_warmup_epochs
            config.domain_adv_weight = min(target_lambda, target_lambda * (epoch + 1) / warmup)
        trainer.train_epoch(train_loader, alpha, unseen_loader=unseen_train_loader, epoch=epoch)

        p3 = evaluator.accuracy(val_seen_dataset,   mask_face=1)
        p4 = evaluator.accuracy(val_seen_dataset,   mask_face=0)
        p5 = evaluator.accuracy(val_unseen_dataset, mask_face=1)
        p6 = evaluator.accuracy(val_unseen_dataset, mask_face=0)
        mean_score = (p3 + p4 + p5 + p6) / 4.0

        if mean_score > best_metric:
            best_metric = mean_score
            best_epoch  = epoch
            best_scores = {"p3": p3, "p4": p4, "p5": p5, "p6": p6, "mean": mean_score}
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "metric": mean_score}, ckpt_path)

        if config.early_stop and early_stopper.step(mean_score):
            break

    # Append result row to sweep_results.csv
    results_file = "./sweep_results.csv"
    write_header = not os.path.exists(results_file)
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_id", "weight_decay", "label_smoothing", "alpha", "lr",
            "dropout_prob", "use_m3", "m3_slow_chunk", "best_epoch",
            "p3", "p4", "p5", "p6", "mean",
        ])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "run_id":          args.run_id,
            "weight_decay":    config.weight_decay,
            "label_smoothing": config.label_smoothing,
            "alpha":           config.alpha,
            "lr":              config.lr,
            "dropout_prob":    config.modality_dropout_prob,
            "use_m3":          config.use_m3,
            "m3_slow_chunk":   config.m3_slow_chunk,
            "best_epoch":      best_epoch,
            **best_scores,
        })

    print(f"[sweep] run={args.run_id} | wd={config.weight_decay} | ls={config.label_smoothing} "
          f"| alpha={config.alpha} | best_mean={best_metric:.2f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
