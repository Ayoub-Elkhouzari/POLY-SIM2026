import logging
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig
from models.fop import FOP
from models.multibranch import MultiBranchFOP
from models.masked_fop import MaskedFOP
from utils.earlystop import EarlyStopping
from utils.evaluator import Evaluator
from utils.featLoader import LoadData
from utils.trainer import Trainer


def save_checkpoint(model, optimizer, config, epoch, metric_value, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "epoch":             epoch,
        "metric":            metric_value,
        "early_stop_metric": config.early_stop_metric,
        "model_state":       model.state_dict(),
        "optimizer_state":   optimizer.state_dict() if optimizer else None,
        "config":            vars(config),
    }, save_path)


def setup_logger(config):
    logger = logging.getLogger("Experiment")
    logger.setLevel(config.log_level)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s][%(name)s] %(message)s"))
        logger.addHandler(h)
    return logger


def make_dataset(df, config):
    return LoadData(config=config, df=df, audio_encoder="ecappa_feats_path")


def make_loader(dataset, config, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def main():
    config = ExperimentConfig()
    torch.manual_seed(config.seed)
    if config.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger = setup_logger(config)
    logger.info("=== Experiment started ===")
    logger.info(
        "Seed=%d | Device=%s | Model=%s | Fusion=%s | Version=%s | "
        "Train=%s | Unseen=%s | #Classes=%d | dropout_p=%.2f",
        config.seed, config.device, config.model_type, config.fusion,
        config.version, config.seen_lang, config.unseen_lang,
        config.resolved_num_classes, config.modality_dropout_prob,
    )

    # --------------------------------------------------
    # CSV paths
    # --------------------------------------------------
    train_csv        = f"{config.data_dir}/Csv_files/{config.version}_train_{config.seen_lang}.csv"
    unseen_train_csv = f"{config.data_dir}/Csv_files/{config.version}_train_{config.unseen_lang}.csv"

    # --------------------------------------------------
    # Train / local-val split (no held-out labels in challenge val)
    # 80% train  |  20% local proxy val (seen language)
    # 20% of unseen-language train → proxy for P5/P6
    # --------------------------------------------------
    df_seen   = pd.read_csv(train_csv)
    df_unseen = pd.read_csv(unseen_train_csv)

    val_seen_df   = df_seen.sample(frac=config.val_frac, random_state=config.seed)
    train_df      = df_seen.drop(val_seen_df.index).reset_index(drop=True)
    val_unseen_df = df_unseen.sample(frac=config.val_frac, random_state=config.seed)
    # Unseen-language training data (excluded from val monitoring to avoid leakage)
    unseen_train_df = df_unseen.drop(val_unseen_df.index).reset_index(drop=True)

    logger.info(
        "Split | Train=%d | Val-seen=%d | Unseen-train=%d | Val-unseen=%d",
        len(train_df), len(val_seen_df), len(unseen_train_df), len(val_unseen_df),
    )

    train_dataset      = make_dataset(train_df,      config)
    val_seen_dataset   = make_dataset(val_seen_df,   config)
    val_unseen_dataset = make_dataset(val_unseen_df, config)

    train_loader = make_loader(train_dataset, config, shuffle=True)

    unseen_train_loader = None
    if config.train_unseen_lang:
        unseen_train_dataset = make_dataset(unseen_train_df, config)
        unseen_train_loader  = make_loader(unseen_train_dataset, config, shuffle=True)
        logger.info("Unseen-lang training enabled | %d samples (audio-only)", len(unseen_train_df))

    # --------------------------------------------------
    # Infer feature dims and build model
    # --------------------------------------------------
    audio, face, _ = next(iter(train_loader))
    audio_dim, face_dim = audio.shape[1], face.shape[1]
    logger.info("Feature dims | Audio=%d | Face=%d", audio_dim, face_dim)

    model_type = config.model_type.lower()

    if model_type == "masked_fop":
        model = MaskedFOP(config=config, face_dim=face_dim, audio_dim=audio_dim)
    elif model_type == "fop":
        model = FOP(config=config, face_dim=face_dim, audio_dim=audio_dim)
    elif model_type == "multibranch":
        model = MultiBranchFOP(config=config, face_dim=face_dim, audio_dim=audio_dim)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    logger.info("Model | Params=%.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    trainer   = Trainer(model, config)
    evaluator = Evaluator(model, config)

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    alpha      = config.alpha
    best_metric = -float("inf")
    best_epoch  = -1

    mem_tag = f"_mem{config.audio_memory_size}" if getattr(config, "use_audio_memory", False) else ""
    adv_tag = "_adv" if getattr(config, "use_domain_adv", False) else ""
    save_path = (
        f"./checkpoints/{config.version}_{config.model_type}_"
        f"{config.seen_lang}_{config.fusion}_drop{config.modality_dropout_prob}{mem_tag}{adv_tag}_best.pt"
    )

    early_stopper = EarlyStopping(
        patience=config.early_stop_patience,
        min_delta=config.early_stop_min_delta,
    )

    for epoch in range(config.max_epochs):
        loss = trainer.train_epoch(train_loader, alpha, unseen_loader=unseen_train_loader, epoch=epoch)

        # P3-like: seen language, both modalities (mask_face=1)
        p3 = evaluator.accuracy(val_seen_dataset,   mask_face=1)
        # P4-like: seen language, audio-only (mask_face=0)
        p4 = evaluator.accuracy(val_seen_dataset,   mask_face=0)
        # P5-like: unseen language, both modalities (mask_face=1)
        p5 = evaluator.accuracy(val_unseen_dataset, mask_face=1)
        # P6-like: unseen language, audio-only (mask_face=0)
        p6 = evaluator.accuracy(val_unseen_dataset, mask_face=0)

        mean_score = (p3 + p4 + p5 + p6) / 4.0

        monitor = {
            "seen":   p3,
            "unseen": p5,
            "mean":   mean_score,
        }[config.early_stop_metric]

        if monitor > best_metric:
            best_metric = monitor
            best_epoch  = epoch
            save_checkpoint(
                model=model, optimizer=trainer.opt,
                config=config, epoch=epoch,
                metric_value=monitor, save_path=save_path,
            )

        logger.info(
            "Epoch %03d | Loss %.4f | P3 %.2f | P4 %.2f | P5 %.2f | P6 %.2f | Mean %.2f",
            epoch, loss, p3, p4, p5, p6, mean_score,
        )

        if config.early_stop and early_stopper.step(monitor):
            logger.info(
                "Early stop at epoch %d (best=%s %.2f at epoch %d)",
                epoch, config.early_stop_metric, early_stopper.best_score, best_epoch,
            )
            break

    logger.info("=== Done | Best epoch=%d | Best %s=%.2f ===",
                best_epoch, config.early_stop_metric, best_metric)


if __name__ == "__main__":
    main()
