"""
Run LingBot-VLA attention heatmap with a REAL image instead of dummy tensors.

Manually patchifies the image into (num_patches, patch_dim) format that
LingBot's embed_image expects, then runs the full flow-matching inference
with attention capture hooks.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/viz_attention_real_image.py \
        --config lingbot_vla_4b/attention \
        --image /path/to/image.jpg \
        --output output/lingbot-vla-4b/attention_heatmaps_real
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def patchify_image(
    image_path: str,
    target_size: tuple = (224, 224),
    patch_size: int = 14,
    temporal_patch_size: int = 2,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load image, resize, and convert to patchified tensor for Qwen2.5-VL ViT.

    Returns:
        patches: (num_patches, patch_dim) tensor in bf16
        original_image: numpy array of the resized image (H, W, 3)
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    original_np = np.array(img)

    img_tensor = torch.from_numpy(original_np).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)

    C, H, W = img_tensor.shape
    grid_h = H // patch_size
    grid_w = W // patch_size

    # Qwen2.5-VL ViT expects temporal_patch_size=2, so we stack the image twice
    # to simulate (C * temporal * patch_h * patch_w) = 3 * 2 * 14 * 14 = 1176
    img_doubled = torch.stack([img_tensor, img_tensor], dim=0)  # (2, 3, H, W)

    # Reshape to patches: (2, 3, grid_h, patch_size, grid_w, patch_size)
    patches = img_doubled.reshape(
        temporal_patch_size, C, grid_h, patch_size, grid_w, patch_size
    )
    # -> (grid_h, grid_w, temporal, C, patch_h, patch_w)
    patches = patches.permute(2, 4, 0, 1, 3, 5)
    # -> (num_patches, temporal * C * patch_h * patch_w)
    patches = patches.reshape(grid_h * grid_w, -1)

    logger.info(
        "Patchified %s: %dx%d -> %d patches, dim=%d",
        image_path, H, W, patches.shape[0], patches.shape[1],
    )

    return patches.to(torch.bfloat16), original_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True, help="Path to real image file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--instruction", default="pick up the red cup")
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir
    from omegaconf import DictConfig, OmegaConf

    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        raw_cfg = compose(config_name=args.config)
    OmegaConf.set_struct(raw_cfg, False)
    keys = list(raw_cfg.keys())
    if len(keys) == 1 and isinstance(raw_cfg[keys[0]], DictConfig):
        cfg = raw_cfg[keys[0]]
    else:
        cfg = raw_cfg
    OmegaConf.resolve(cfg)

    from src.controllers import CONTROLLER_REGISTRY

    controller_name = cfg.controller_name
    controller_cfg = OmegaConf.to_container(cfg.controller_config, resolve=True)
    controller_cls = CONTROLLER_REGISTRY[controller_name]

    controller = controller_cls(
        model_name=cfg.model_name,
        store_type=controller_cfg.get("store_type"),
        store_layers=controller_cfg.get("store_layers"),
        store_phases=controller_cfg.get("store_phases"),
        hook_mode="analysis",
    )

    pipeline = controller.init_pipeline(cfg)
    controller.pipeline = pipeline
    controller._resolve_store_layers()
    controller.register_hooks()

    # Patchify real image
    config = pipeline.config
    img_size = config.get("resize_imgs_with_padding", [224, 224])
    patches, original_np = patchify_image(args.image, tuple(img_size))

    device = pipeline.device
    num_patches = patches.shape[0]

    # Shape: (batch=1, num_cam=1, num_patches, patch_dim)
    images = patches.unsqueeze(0).unsqueeze(0).to(device)
    img_masks = torch.ones(1, 1, dtype=torch.bool, device=device)

    max_state_dim = config.get("max_state_dim", 75)
    state = torch.zeros(1, max_state_dim, dtype=torch.bfloat16, device=device)

    policy = pipeline.policy
    tokenizer = policy.language_tokenizer
    max_len = config.get("tokenizer_max_length", 72)
    lang_encoded = tokenizer(
        args.instruction,
        return_tensors="pt",
        padding="max_length",
        max_length=max_len,
        truncation=True,
    )
    lang_tokens = lang_encoded["input_ids"].to(device)
    lang_masks = lang_encoded["attention_mask"].to(device)

    logger.info("Running inference with real image: %s", args.image)
    controller._context_started = False
    controller.reset_state()

    with torch.no_grad():
        actions = policy.model.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )

    controller.postprocess()
    logger.info("global_store keys: %s", list(controller.global_store.keys()))

    # Now generate heatmaps using the same viz functions
    import matplotlib
    matplotlib.use("Agg")
    from scripts.viz_attention_heatmap import (
        compute_attention,
        classify_tokens,
        plot_attention_matrix,
        plot_token_importance,
        plot_visual_spatial_map,
        plot_cross_modal_heatmap,
    )

    os.makedirs(args.output, exist_ok=True)

    # Save original image for reference
    Image.fromarray(original_np).save(os.path.join(args.output, "input_image.jpg"))

    # Also save an overlay version
    import matplotlib.pyplot as plt

    qk_pairs = []
    for key in controller.global_store:
        if key.endswith("_q_states"):
            layer_idx = int(key.split("_", 1)[0])
            k_key = f"{layer_idx}_k_states"
            if k_key in controller.global_store:
                qk_pairs.append((layer_idx, key, k_key))
    qk_pairs.sort(key=lambda x: x[0])
    logger.info("Found %d QK pairs", len(qk_pairs))

    patch_size = 14
    grid_h = img_size[0] // patch_size
    grid_w = img_size[1] // patch_size
    n_visual = grid_h * grid_w

    for layer_idx, q_key, k_key in qk_pairs:
        q = controller.global_store[q_key]
        k = controller.global_store[k_key]
        if isinstance(q, list):
            q = q[0]
        if isinstance(k, list):
            k = k[0]

        attn_probs = compute_attention(q, k, head_dim=args.head_dim)
        attn_mean = attn_probs.mean(dim=(0, 1)).detach().cpu().numpy()
        seq_len = attn_mean.shape[0]

        # Use actual n_visual from image patches (may differ from seq_len assumption)
        actual_n_visual = min(n_visual, seq_len)
        token_types = classify_tokens(seq_len, actual_n_visual)

        prefix = os.path.join(args.output, f"layer_{layer_idx}")

        plot_attention_matrix(attn_mean, token_types, layer_idx, f"{prefix}_attn_matrix.png")
        plot_token_importance(attn_mean, token_types, layer_idx, f"{prefix}_token_importance.png")
        plot_cross_modal_heatmap(attn_mean, actual_n_visual, layer_idx, f"{prefix}_cross_modal.png")

        # Spatial map with image overlay
        received = attn_mean.sum(axis=0)[:actual_n_visual]
        if actual_n_visual == grid_h * grid_w:
            spatial = received.reshape(grid_h, grid_w)

            # Spatial heatmap alone
            plot_visual_spatial_map(attn_mean, actual_n_visual, layer_idx, f"{prefix}_spatial_map.png")

            # Overlay on actual image
            spatial_resized = np.array(
                Image.fromarray(
                    ((spatial - spatial.min()) / (spatial.max() - spatial.min() + 1e-8) * 255).astype(np.uint8)
                ).resize((img_size[1], img_size[0]), Image.LANCZOS)
            )

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            axes[0].imshow(original_np)
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            axes[1].imshow(spatial, cmap="hot", interpolation="bilinear")
            axes[1].set_title(f"Layer {layer_idx} — Patch Importance")

            axes[2].imshow(original_np)
            axes[2].imshow(spatial_resized, cmap="hot", alpha=0.5)
            axes[2].set_title(f"Layer {layer_idx} — Attention Overlay")
            axes[2].axis("off")

            plt.tight_layout()
            fig.savefig(f"{prefix}_overlay.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved overlay: %s", f"{prefix}_overlay.png")

    controller.remove_hooks()
    logger.info("All real-image heatmaps saved to %s", args.output)


if __name__ == "__main__":
    main()
