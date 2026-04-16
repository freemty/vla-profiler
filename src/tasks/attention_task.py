"""
Attention analysis tasks for VLM profiling.

Three tasks:
1. visual_text_attention — Gini sparsity and top-k concentration
2. sink_detection — identify attention sink tokens
3. per_layer_stats — per-layer entropy and mean attention

All tasks read Q/K tensors from controller.global_store.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from src.tasks import TASK_REGISTRY


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Shared utilities
# ------------------------------------------------------------------

def _find_qk_keys(global_store: Dict[str, Any]) -> List[Tuple[int, str, str]]:
    """
    Find all (layer_idx, q_key, k_key) tuples in global_store.

    Looks for keys matching pattern: {layer_idx}_q_states / {layer_idx}_k_states
    """
    q_keys = {}
    k_keys = {}

    for key in global_store:
        if key.endswith("_q_states"):
            parts = key.split("_", 1)
            layer_idx = int(parts[0])
            q_keys[layer_idx] = key
        elif key.endswith("_k_states"):
            parts = key.split("_", 1)
            layer_idx = int(parts[0])
            k_keys[layer_idx] = key

    result = []
    for layer_idx in sorted(set(q_keys.keys()) & set(k_keys.keys())):
        result = [*result, (layer_idx, q_keys[layer_idx], k_keys[layer_idx])]

    return result


def _compute_attention_scores(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    head_dim: int = 128,
) -> torch.Tensor:
    """
    Compute attention scores from Q and K tensors.

    Args:
        q_tensor: Query tensor [batch, seq_len, hidden] or [batch, heads, seq_len, head_dim]
        k_tensor: Key tensor [batch, seq_len, hidden] or [batch, heads, seq_len, head_dim]
        head_dim: Attention head dimension (used when reshaping 3D tensors).
                  Default 128 works for Qwen2.5-VL-7B and Llama-2-7B.

    Returns:
        Attention scores [batch, heads, seq_len, seq_len] after softmax
    """
    # If 3D (batch, seq, hidden), reshape assuming multi-head
    if q_tensor.dim() == 3:
        if q_tensor.shape[-1] % head_dim != 0:
            raise ValueError(
                f"Q hidden dim {q_tensor.shape[-1]} not divisible by head_dim {head_dim}. "
                f"Pass head_dim explicitly for this model."
            )
        if k_tensor.shape[-1] % head_dim != 0:
            raise ValueError(
                f"K hidden dim {k_tensor.shape[-1]} not divisible by head_dim {head_dim}. "
                f"Pass head_dim explicitly for this model."
            )
        num_q_heads = q_tensor.shape[-1] // head_dim
        num_k_heads = k_tensor.shape[-1] // head_dim
        batch, seq_q, _ = q_tensor.shape
        _, seq_k, _ = k_tensor.shape
        q = q_tensor.view(batch, seq_q, num_q_heads, head_dim).transpose(1, 2)
        k = k_tensor.view(batch, seq_k, num_k_heads, head_dim).transpose(1, 2)
    elif q_tensor.dim() == 4:
        q = q_tensor
        k = k_tensor
        head_dim = q.shape[-1]
        num_q_heads = q.shape[1]
        num_k_heads = k.shape[1]
    else:
        raise ValueError(f"Unexpected Q tensor dim: {q_tensor.dim()}")

    # Handle GQA: repeat K heads to match Q head count
    if num_k_heads != num_q_heads:
        repeat_factor = num_q_heads // num_k_heads
        k = k.repeat_interleave(repeat_factor, dim=1)

    scale = 1.0 / math.sqrt(head_dim)
    # [batch, heads, seq_q, seq_k]
    attn_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    attn_probs = F.softmax(attn_scores, dim=-1)
    return attn_probs


def _gini_coefficient(values: torch.Tensor) -> float:
    """Compute Gini coefficient for a 1D tensor of non-negative values."""
    if values.numel() == 0:
        return 0.0

    sorted_vals = torch.sort(values.float())[0]
    n = sorted_vals.numel()
    index = torch.arange(1, n + 1, dtype=torch.float32, device=sorted_vals.device)
    total = sorted_vals.sum()
    if total < 1e-12:
        return 0.0

    gini = (2.0 * (index * sorted_vals).sum() / (n * total)) - (n + 1) / n
    return gini.item()


def _top_k_concentration(
    attn_row: torch.Tensor,
    k: int = 5,
) -> float:
    """Fraction of attention mass in top-k positions."""
    if attn_row.numel() == 0:
        return 0.0

    k = min(k, attn_row.numel())
    topk_vals = torch.topk(attn_row.float(), k).values
    total = attn_row.float().sum()
    if total < 1e-12:
        return 0.0
    return (topk_vals.sum() / total).item()


def _attention_entropy(attn_row: torch.Tensor) -> float:
    """Shannon entropy of an attention distribution."""
    probs = attn_row.float()
    probs = probs[probs > 1e-12]
    if probs.numel() == 0:
        return 0.0
    return -(probs * probs.log()).sum().item()


# ------------------------------------------------------------------
# Task 1: Visual-Text Attention Analysis
# ------------------------------------------------------------------

def task_visual_text_attn(
    controller: Any,
    save_dir: str,
    task_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute attention sparsity between visual and text tokens.

    For each layer, computes:
    - text-to-visual Gini coefficient (how sparsely text attends to visual)
    - visual-to-text Gini coefficient (how sparsely visual attends to text)
    - top-k concentration for both directions

    Requires Q and K tensors in controller.global_store.
    """
    output_dir = os.path.join(save_dir, "visual_text_attention")
    os.makedirs(output_dir, exist_ok=True)

    global_store = controller.global_store
    qk_pairs = _find_qk_keys(global_store)

    if not qk_pairs:
        logger.warning("No QK pairs found in global_store")
        return {}

    # Config
    top_k = task_config.get("top_k", 5)
    visual_token_count = task_config.get("visual_token_count", None)
    head_dim = task_config.get("head_dim", 128)

    results: Dict[str, Any] = {}

    for layer_idx, q_key, k_key in qk_pairs:
        q_tensors = global_store[q_key]
        k_tensors = global_store[k_key]

        # Use the first step's tensors (prefill)
        q = q_tensors[0] if isinstance(q_tensors, list) else q_tensors
        k = k_tensors[0] if isinstance(k_tensors, list) else k_tensors

        attn_probs = _compute_attention_scores(q, k, head_dim=head_dim)
        # Average over batch and heads: [seq_q, seq_k]
        attn_mean = attn_probs.mean(dim=(0, 1))

        seq_len = attn_mean.shape[0]
        # Estimate visual token boundary if not provided
        n_visual = visual_token_count if visual_token_count else seq_len // 3

        # text-to-visual: text tokens attending to visual tokens
        text_to_visual = attn_mean[n_visual:, :n_visual]
        # visual-to-text: visual tokens attending to text tokens
        visual_to_text = attn_mean[:n_visual, n_visual:]

        layer_stats = {
            "text_to_visual_gini": _gini_coefficient(text_to_visual.flatten()),
            "visual_to_text_gini": _gini_coefficient(visual_to_text.flatten()),
            "text_to_visual_topk": _top_k_concentration(
                text_to_visual.flatten(), k=top_k
            ),
            "visual_to_text_topk": _top_k_concentration(
                visual_to_text.flatten(), k=top_k
            ),
            "seq_len": seq_len,
            "n_visual_tokens": n_visual,
        }
        results[f"layer_{layer_idx}"] = layer_stats

    output_path = os.path.join(output_dir, "sparsity_stats.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Visual-text attention stats saved to %s", output_path)
    return results


# ------------------------------------------------------------------
# Task 2: Sink Token Detection
# ------------------------------------------------------------------

def task_sink_detection(
    controller: Any,
    save_dir: str,
    task_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Detect attention sink tokens across layers.

    Computes total attention received per token position, identifies
    top-K sink tokens, and labels them as visual/text/boundary.
    """
    output_dir = os.path.join(save_dir, "sink_detection")
    os.makedirs(output_dir, exist_ok=True)

    global_store = controller.global_store
    qk_pairs = _find_qk_keys(global_store)

    if not qk_pairs:
        logger.warning("No QK pairs found for sink detection")
        return {}

    sink_k = task_config.get("sink_k", 10)
    visual_token_count = task_config.get("visual_token_count", None)
    head_dim = task_config.get("head_dim", 128)

    results: Dict[str, Any] = {}

    for layer_idx, q_key, k_key in qk_pairs:
        q_tensors = global_store[q_key]
        k_tensors = global_store[k_key]

        q = q_tensors[0] if isinstance(q_tensors, list) else q_tensors
        k = k_tensors[0] if isinstance(k_tensors, list) else k_tensors

        attn_probs = _compute_attention_scores(q, k, head_dim=head_dim)
        # Attention received per key token: sum over query dim
        # [batch, heads, seq_q, seq_k] -> [seq_k]
        attn_received = attn_probs.sum(dim=(0, 1, 2))
        seq_len = attn_received.shape[0]
        n_visual = visual_token_count if visual_token_count else seq_len // 3

        # Top-K sink tokens
        top_k_count = min(sink_k, seq_len)
        top_indices = torch.topk(attn_received, top_k_count).indices.tolist()

        sink_tokens = []
        for pos in top_indices:
            if pos < n_visual:
                token_type = "visual"
            elif pos == n_visual:
                token_type = "boundary"
            else:
                token_type = "text"
            sink_tokens = [
                *sink_tokens,
                {
                    "position": pos,
                    "type": token_type,
                    "attention_received": attn_received[pos].item(),
                },
            ]

        results[f"layer_{layer_idx}"] = {
            "sink_tokens": sink_tokens,
            "seq_len": seq_len,
            "n_visual_tokens": n_visual,
        }

    output_path = os.path.join(output_dir, "sink_tokens.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Sink tokens saved to %s", output_path)
    return results


# ------------------------------------------------------------------
# Task 3: Per-Layer Statistics
# ------------------------------------------------------------------

def task_per_layer_stats(
    controller: Any,
    save_dir: str,
    task_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute per-layer attention entropy and mean attention statistics.

    For each layer, reports:
    - mean entropy across all attention heads
    - mean attention score
    - max attention score
    """
    output_dir = os.path.join(save_dir, "per_layer_stats")
    os.makedirs(output_dir, exist_ok=True)

    global_store = controller.global_store
    qk_pairs = _find_qk_keys(global_store)

    if not qk_pairs:
        logger.warning("No QK pairs found for per-layer stats")
        return {}

    head_dim = task_config.get("head_dim", 128)
    results: Dict[str, Any] = {}

    for layer_idx, q_key, k_key in qk_pairs:
        q_tensors = global_store[q_key]
        k_tensors = global_store[k_key]

        q = q_tensors[0] if isinstance(q_tensors, list) else q_tensors
        k = k_tensors[0] if isinstance(k_tensors, list) else k_tensors

        attn_probs = _compute_attention_scores(q, k, head_dim=head_dim)
        # [batch, heads, seq_q, seq_k]
        attn_mean = attn_probs.mean(dim=(0, 1))  # [seq_q, seq_k]

        # Vectorized entropy: -(p * log(p)).sum(dim=-1).mean()
        probs = attn_mean.float().clamp(min=1e-12)
        row_entropies = -(probs * probs.log()).sum(dim=-1)
        mean_entropy = row_entropies.mean().item() if row_entropies.numel() > 0 else 0.0
        mean_attn = attn_mean.mean().item()
        max_attn = attn_mean.max().item()

        results[f"layer_{layer_idx}"] = {
            "mean_entropy": mean_entropy,
            "mean_attention": mean_attn,
            "max_attention": max_attn,
            "seq_len": attn_mean.shape[0],
        }

    output_path = os.path.join(output_dir, "layer_stats.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Per-layer stats saved to %s", output_path)
    return results


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

TASK_REGISTRY.register("visual_text_attention", task_visual_text_attn)
TASK_REGISTRY.register("sink_detection", task_sink_detection)
TASK_REGISTRY.register("per_layer_stats", task_per_layer_stats)
