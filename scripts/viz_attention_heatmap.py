"""
Generate attention heatmap visualizations from Q/K captures.

Produces:
1. Token-to-token attention matrix heatmap (per layer, head-averaged)
2. Per-token importance bar (received attention, colored by visual/text type)
3. Visual token spatial importance map (if image patches form a grid)

Usage:
    # On server with model loaded
    CUDA_VISIBLE_DEVICES=0 python scripts/viz_attention_heatmap.py \
        --config lingbot_vla_4b/attention \
        --output output/lingbot-vla-4b/attention_heatmaps
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_attention(q: torch.Tensor, k: torch.Tensor, head_dim: int = 128):
    if q.dim() == 3:
        num_q_heads = q.shape[-1] // head_dim
        num_k_heads = k.shape[-1] // head_dim
        b, sq, _ = q.shape
        _, sk, _ = k.shape
        q = q.view(b, sq, num_q_heads, head_dim).transpose(1, 2)
        k = k.view(b, sk, num_k_heads, head_dim).transpose(1, 2)
    else:
        num_q_heads = q.shape[1]
        num_k_heads = k.shape[1]
        head_dim = q.shape[-1]

    if num_k_heads != num_q_heads:
        k = k.repeat_interleave(num_q_heads // num_k_heads, dim=1)

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    return F.softmax(scores, dim=-1)


def classify_tokens(seq_len: int, n_visual: int):
    types = []
    for i in range(seq_len):
        if i < n_visual:
            types.append("visual")
        else:
            types.append("text")
    return types


def plot_attention_matrix(attn: np.ndarray, token_types: list, layer_idx: int, out_path: str):
    seq_len = attn.shape[0]
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(attn, cmap="viridis", aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Attention weight")

    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(f"Layer {layer_idx} — Attention Matrix (head-averaged)\nseq_len={seq_len}")

    n_vis = sum(1 for t in token_types if t == "visual")
    if n_vis > 0 and n_vis < seq_len:
        ax.axhline(y=n_vis - 0.5, color="red", linewidth=1.5, linestyle="--", alpha=0.8)
        ax.axvline(x=n_vis - 0.5, color="red", linewidth=1.5, linestyle="--", alpha=0.8)
        ax.text(n_vis // 2, -2, "visual", ha="center", color="cyan", fontsize=9, fontweight="bold")
        ax.text(n_vis + (seq_len - n_vis) // 2, -2, "text", ha="center", color="orange", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved attention matrix: %s", out_path)


def plot_token_importance(attn: np.ndarray, token_types: list, layer_idx: int, out_path: str):
    received = attn.sum(axis=0)
    seq_len = len(received)

    colors = ["#4ECDC4" if t == "visual" else "#FF6B6B" for t in token_types]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(seq_len), received, color=colors, width=1.0, edgecolor="none")

    n_vis = sum(1 for t in token_types if t == "visual")
    if n_vis > 0:
        ax.axvline(x=n_vis - 0.5, color="white", linewidth=1.5, linestyle="--", alpha=0.8)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#4ECDC4", label="Visual"), Patch(facecolor="#FF6B6B", label="Text")]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_xlabel("Token position")
    ax.set_ylabel("Total attention received")
    ax.set_title(f"Layer {layer_idx} — Per-Token Importance (attention received)")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    logger.info("Saved token importance: %s", out_path)


def plot_visual_spatial_map(attn: np.ndarray, n_visual: int, layer_idx: int, out_path: str):
    received = attn.sum(axis=0)[:n_visual]

    grid_side = int(math.sqrt(n_visual))
    if grid_side * grid_side != n_visual:
        for h in range(int(math.sqrt(n_visual)) + 1, 0, -1):
            if n_visual % h == 0:
                grid_h, grid_w = h, n_visual // h
                break
        else:
            logger.warning("Cannot reshape %d visual tokens into a grid", n_visual)
            return
    else:
        grid_h = grid_w = grid_side

    spatial = received.reshape(grid_h, grid_w)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(spatial, cmap="hot", interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="Attention received")
    ax.set_title(f"Layer {layer_idx} — Visual Patch Importance ({grid_h}x{grid_w})")
    ax.set_xlabel("Patch column")
    ax.set_ylabel("Patch row")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved spatial map: %s", out_path)


def plot_cross_modal_heatmap(attn: np.ndarray, n_visual: int, layer_idx: int, out_path: str):
    seq_len = attn.shape[0]
    n_text = seq_len - n_visual

    text_to_visual = attn[n_visual:, :n_visual]
    visual_to_text = attn[:n_visual, n_visual:]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    im0 = axes[0].imshow(text_to_visual, cmap="magma", aspect="auto", interpolation="nearest")
    axes[0].set_title(f"Layer {layer_idx} — Text→Visual Attention")
    axes[0].set_xlabel(f"Visual token (0–{n_visual-1})")
    axes[0].set_ylabel(f"Text token ({n_visual}–{seq_len-1})")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(visual_to_text, cmap="magma", aspect="auto", interpolation="nearest")
    axes[1].set_title(f"Layer {layer_idx} — Visual→Text Attention")
    axes[1].set_xlabel(f"Text token ({n_visual}–{seq_len-1})")
    axes[1].set_ylabel(f"Visual token (0–{n_visual-1})")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved cross-modal heatmap: %s", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Hydra config name (e.g. lingbot_vla_4b/attention)")
    parser.add_argument("--output", required=True, help="Output directory for heatmaps")
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir
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
    import src.tasks.attention_task  # noqa: F401

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

    inputs_list = controller.prepare_inputs(cfg)
    inp = inputs_list[0]
    logger.info("Running inference for: %s", inp.get("name", "unnamed"))

    controller.reset_state()
    controller.model_inference(controller.pipeline, cfg, inp)
    controller.postprocess()

    os.makedirs(args.output, exist_ok=True)

    logger.info("step_store keys: %s", list(controller.step_store.keys()))
    logger.info("global_store keys: %s", list(controller.global_store.keys()))

    qk_pairs = []
    for key in controller.global_store:
        if key.endswith("_q_states"):
            layer_idx = int(key.split("_", 1)[0])
            k_key = f"{layer_idx}_k_states"
            if k_key in controller.global_store:
                qk_pairs.append((layer_idx, key, k_key))
    qk_pairs.sort(key=lambda x: x[0])

    logger.info("Found %d QK pairs in global_store", len(qk_pairs))

    n_visual = 45
    if hasattr(controller, "get_vision_encoder"):
        try:
            for key in controller.global_store:
                if key.endswith("_q_states"):
                    q = controller.global_store[key]
                    if isinstance(q, list):
                        q = q[0]
                    seq_len = q.shape[1] if q.dim() == 3 else q.shape[2]
                    logger.info("Detected seq_len=%d from Q tensor", seq_len)
                    break
        except Exception:
            pass

    summary = {}

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

        token_types = classify_tokens(seq_len, n_visual)

        plot_attention_matrix(
            attn_mean, token_types, layer_idx,
            os.path.join(args.output, f"layer_{layer_idx}_attn_matrix.png"),
        )

        plot_token_importance(
            attn_mean, token_types, layer_idx,
            os.path.join(args.output, f"layer_{layer_idx}_token_importance.png"),
        )

        plot_visual_spatial_map(
            attn_mean, n_visual, layer_idx,
            os.path.join(args.output, f"layer_{layer_idx}_spatial_map.png"),
        )

        plot_cross_modal_heatmap(
            attn_mean, n_visual, layer_idx,
            os.path.join(args.output, f"layer_{layer_idx}_cross_modal.png"),
        )

        received = attn_mean.sum(axis=0)
        top5_idx = np.argsort(received)[-5:][::-1]
        summary[f"layer_{layer_idx}"] = {
            "seq_len": seq_len,
            "n_visual": n_visual,
            "top5_sinks": [
                {"pos": int(i), "type": token_types[i], "received": float(received[i])}
                for i in top5_idx
            ],
            "visual_total": float(received[:n_visual].sum()),
            "text_total": float(received[n_visual:].sum()),
        }

    with open(os.path.join(args.output, "heatmap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    controller.remove_hooks()
    logger.info("All heatmaps saved to %s", args.output)


if __name__ == "__main__":
    main()
