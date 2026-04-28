from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.JRPP import JRPP
from utils.data_loader import PopularityDataset, collate_popularity_batch, read_data
from utils.metrics import regression_metrics
from utils.parsers import build_parser
from utils.runtime import (
    load_config,
    meta_fields_for,
    move_items_to_device,
    prepare_config,
    resolve_device,
    seed_everything,
    setup_logging,
)


def _batch_inputs(batch, items: Tuple[torch.Tensor, ...], text_vec=None, img_vec_cls=None):
    items_id, items_text, items_img, items_meta = items
    return (
        text_vec if text_vec is not None else batch.text_vec,
        getattr(batch, "img_vec_pool", None),
        batch.meta_features,
        img_vec_cls if img_vec_cls is not None else batch.img_vec_cls,
        batch.user_id,
        batch.image_id,
        items_id,
        items_text,
        items_img,
        items_meta,
    )


def _parse_noise_stds(value: str):
    levels = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not levels:
        raise ValueError("--tta-noise-stds must contain at least one numeric value.")
    return levels


def resolve_tta_settings(args, config):
    tta_config = config.get("tta", {})
    runs = 0 if args.no_tta else int(args.tta_runs if args.tta_runs is not None else tta_config.get("runs", 20))

    if args.tta_noise_stds is not None:
        noise_stds = _parse_noise_stds(args.tta_noise_stds)
    elif "noise_stds" in tta_config:
        noise_stds = [float(value) for value in tta_config["noise_stds"]]
    else:
        noise_stds = [0.01]

    confidence_temperature = float(
        args.tta_confidence_temperature
        if args.tta_confidence_temperature is not None
        else tta_config.get("confidence_temperature", 0.05)
    )
    if confidence_temperature <= 0:
        raise ValueError("TTA confidence temperature must be positive.")

    anchor_weight = float(
        args.tta_anchor_weight
        if args.tta_anchor_weight is not None
        else tta_config.get("anchor_weight", 0.5)
    )
    if not 0 <= anchor_weight <= 1:
        raise ValueError("TTA anchor weight must be in [0, 1].")

    return runs, noise_stds, confidence_temperature, anchor_weight


@torch.no_grad()
def predict_with_tta(
    model: JRPP,
    batch,
    items: Tuple[torch.Tensor, ...],
    runs: int,
    noise_stds,
    confidence_temperature: float,
    anchor_weight: float,
) -> torch.Tensor:
    original_pred, _ = model(*_batch_inputs(batch, items))
    original_pred = original_pred.reshape(-1)
    if runs <= 0:
        return original_pred

    preds = []
    similarities = []
    original_text = batch.text_vec
    original_image = batch.img_vec_cls
    original_joint = model._project_query(original_text, original_image, batch.meta_features)

    for step in range(runs):
        std = noise_stds[step % len(noise_stds)]
        text_aug = original_text + torch.randn_like(original_text) * std
        image_aug = original_image + torch.randn_like(original_image) * std
        pred, _ = model(*_batch_inputs(batch, items, text_vec=text_aug, img_vec_cls=image_aug))
        preds.append(pred.reshape(-1))

        augmented_joint = model._project_query(text_aug, image_aug, batch.meta_features)
        similarities.append(F.cosine_similarity(original_joint, augmented_joint, dim=1, eps=1e-6))

    stacked_preds = torch.stack(preds, dim=0)
    stacked_similarities = torch.stack(similarities, dim=0)
    confidence = F.softmax(stacked_similarities / confidence_temperature, dim=0)
    augmented_pred = (stacked_preds * confidence).sum(dim=0)
    return anchor_weight * original_pred + (1.0 - anchor_weight) * augmented_pred


def load_model_checkpoint(path: str, model: JRPP, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def main() -> None:
    args = build_parser(require_model_path=True).parse_args()
    seed_everything(args.seed)

    raw_config = load_config(args.config)
    meta_fields = meta_fields_for(raw_config, args.data_name)
    train_split, _, test_split = read_data(
        args.data_name,
        data_dir=args.data_dir,
        meta_fields=meta_fields,
        splits=("train", "val", "test"),
    )

    meta_dim = int(test_split[0].meta_features.size(-1))
    config = prepare_config(raw_config, args, meta_dim)
    device = resolve_device(args.device)
    logger = setup_logging(Path(args.save_path), args.data_name, log_name="test.log")

    model = JRPP(args=args, config=config, meta_dim=meta_dim, dropout=args.dropout).to(device)
    checkpoint = load_model_checkpoint(args.model_path, model, device)
    model.eval()

    items = move_items_to_device(train_split, device)

    tta_runs, tta_noise_stds, tta_confidence_temperature, tta_anchor_weight = resolve_tta_settings(args, config)
    logger.info(
        "TTA runs=%d noise_stds=%s confidence_temperature=%.6f anchor_weight=%.6f",
        tta_runs,
        tta_noise_stds,
        tta_confidence_temperature,
        tta_anchor_weight,
    )

    test_loader = DataLoader(
        PopularityDataset(test_split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_popularity_batch,
    )

    predictions = []
    targets = []
    for batch in tqdm(test_loader, desc=f"Testing {args.data_name}"):
        batch = batch.to(device)
        pred = predict_with_tta(
            model,
            batch,
            items,
            tta_runs,
            tta_noise_stds,
            tta_confidence_temperature,
            tta_anchor_weight,
        )
        predictions.append(pred.detach().cpu().numpy())
        targets.append(batch.y.detach().cpu().reshape(-1).numpy())

    metrics = regression_metrics(np.concatenate(predictions), np.concatenate(targets))
    logger.info("Loaded checkpoint from %s", args.model_path)
    if checkpoint.get("epoch") is not None:
        logger.info("Checkpoint epoch: %s", checkpoint["epoch"])
    logger.info("Test MSE: %.6f", metrics["mse"])
    logger.info("Test MAE: %.6f", metrics["mae"])
    logger.info("Test SRC: %.6f", metrics["src"])


if __name__ == "__main__":
    main()
