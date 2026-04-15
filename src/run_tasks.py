"""
Hydra entry point for VLM profiling and analysis.

Modes:
    - profiling: warmup + multi-run benchmark with mean/std timing
    - analysis: single run with post-processing and task execution

Usage:
    python -m src.run_tasks --config-path ../configs --config-name base \
        +experiment=qwen_vl_7b/profiling
"""

from __future__ import annotations

import logging
import math
import os
import statistics
from typing import Any, Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf

# Import registries (triggers module-level registration)
from src.controllers import CONTROLLER_REGISTRY
import src.controllers.qwen_vl_controller  # noqa: F401 — register controller
import src.controllers.openvla_controller  # noqa: F401 — register controller
from src.tasks import TASK_REGISTRY
import src.tasks.profiling_task  # noqa: F401 — register task
import src.tasks.attention_task  # noqa: F401 — register task


logger = logging.getLogger(__name__)


def _setup_logging(debug: bool = False) -> None:
    """Configure logging based on debug flag."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _create_controller(cfg: DictConfig) -> Any:
    """Instantiate controller from registry."""
    controller_name = cfg.controller_name
    controller_cfg = OmegaConf.to_container(cfg.controller_config, resolve=True)

    controller_cls = CONTROLLER_REGISTRY[controller_name]

    controller = controller_cls(
        model_name=cfg.model_name,
        store_type=controller_cfg.get("store_type"),
        store_layers=controller_cfg.get("store_layers"),
        store_phases=controller_cfg.get("store_phases"),
        hook_mode=controller_cfg.get("mode", "analysis"),
    )

    return controller


def _init_pipeline(controller: Any, cfg: DictConfig) -> None:
    """Initialize model pipeline and resolve layers."""
    pipeline = controller.init_pipeline(cfg)
    controller.pipeline = pipeline
    controller._resolve_store_layers()


def _percentile(sorted_data: List[float], pct: float) -> float:
    """Compute percentile from pre-sorted data using linear interpolation."""
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    k = (pct / 100.0) * (n - 1)
    lo = int(math.floor(k))
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


def _run_profiling(controller: Any, cfg: DictConfig) -> None:
    """
    Run profiling mode: warmup + benchmark with timing aggregation.

    After warmup runs, performs num_benchmark_runs full inference passes
    and computes mean/std for each phase timing.
    """
    inputs_list = controller.prepare_inputs(cfg)
    num_warmup = getattr(cfg, "num_warmup_runs", 1)
    num_benchmark = getattr(cfg, "num_benchmark_runs", 3)

    logger.info("Profiling mode: %d warmup + %d benchmark runs", num_warmup, num_benchmark)

    # Register profiling hooks
    controller.register_hooks()

    # Profile each input independently
    for inp in inputs_list:
        input_name = inp.get("name", "unnamed")
        logger.info("Profiling input: %s", input_name)

        # Warmup runs
        for i in range(num_warmup):
            logger.info("  Warmup %d/%d", i + 1, num_warmup)
            controller.timer.reset()
            controller.model_inference(controller.pipeline, cfg, inp)

        # Benchmark runs: collect timing per phase per run
        all_timings: Dict[str, List[float]] = {}

        for i in range(num_benchmark):
            logger.info("  Benchmark %d/%d", i + 1, num_benchmark)
            controller.timer.reset()
            controller.model_inference(controller.pipeline, cfg, inp)

            summary = controller.timer.summary()
            for phase, ms in summary.items():
                if phase not in all_timings:
                    all_timings[phase] = [ms]
                else:
                    all_timings[phase] = [*all_timings[phase], ms]

        # Aggregate timing stats for this input
        aggregated: Dict[str, Any] = {
            "num_runs": num_benchmark,
            "input_name": input_name,
        }
        for phase, timings in all_timings.items():
            sorted_t = sorted(timings)
            mean_val = statistics.mean(sorted_t)
            std_val = statistics.stdev(sorted_t) if len(sorted_t) > 1 else 0.0
            aggregated[phase] = {
                "median_ms": statistics.median(sorted_t),
                "mean_ms": mean_val,
                "std_ms": std_val,
                "cv": std_val / mean_val if mean_val > 0 else 0.0,
                "p10_ms": _percentile(sorted_t, 10),
                "p90_ms": _percentile(sorted_t, 90),
                "p99_ms": _percentile(sorted_t, 99),
                "min_ms": sorted_t[0],
                "max_ms": sorted_t[-1],
                "all_ms": timings,
            }

        controller._aggregated_timing = aggregated

        # Execute tasks per input
        base_output = cfg.get("base_output_path", "output")
        model_short = cfg.model_name.split("/")[-1]
        save_dir = os.path.join(base_output, model_short, input_name)
        os.makedirs(save_dir, exist_ok=True)
        _execute_tasks_to_dir(controller, cfg, save_dir)

    logger.info("All inputs profiled")


def _run_analysis(controller: Any, cfg: DictConfig) -> None:
    """
    Run analysis mode: single inference pass with hook data collection.

    Registers analysis hooks, runs inference for each input,
    postprocesses, then executes configured tasks.
    """
    inputs_list = controller.prepare_inputs(cfg)

    logger.info("Analysis mode: %d inputs", len(inputs_list))

    # Register analysis hooks
    controller.register_hooks()

    for idx, inp in enumerate(inputs_list):
        logger.info("Processing input %d/%d: %s", idx + 1, len(inputs_list), inp.get("name", ""))
        controller.reset_state()

        result = controller.model_inference(controller.pipeline, cfg, inp)
        controller.postprocess(result)
        controller.save_results(inp, result, cfg)

    # Execute tasks
    _execute_tasks(controller, cfg)


def _execute_tasks(controller: Any, cfg: DictConfig) -> None:
    """Execute all configured tasks with auto-generated save_dir."""
    base_output = cfg.get("base_output_path", "output")
    model_short = cfg.model_name.split("/")[-1]
    save_dir = os.path.join(base_output, model_short)
    os.makedirs(save_dir, exist_ok=True)
    _execute_tasks_to_dir(controller, cfg, save_dir)


def _execute_tasks_to_dir(controller: Any, cfg: DictConfig, save_dir: str) -> None:
    """Execute all configured tasks, writing output to save_dir."""
    task_names = list(cfg.get("tasks", []))
    task_configs = OmegaConf.to_container(cfg.get("task_configs", {}), resolve=True) \
        if "task_configs" in cfg else {}

    for task_name in task_names:
        if task_name not in TASK_REGISTRY:
            logger.warning("Task '%s' not found in registry, skipping", task_name)
            continue

        logger.info("Executing task: %s", task_name)
        task_fn = TASK_REGISTRY[task_name]
        task_config = task_configs.get(task_name, {})
        task_fn(controller, save_dir, task_config)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(raw_cfg: DictConfig) -> None:
    """Hydra entry point."""
    # Hydra may nest config under a group key (e.g., qwen_vl_7b) when using
    # --config-name subdir/file. Unwrap if needed.
    OmegaConf.set_struct(raw_cfg, False)
    keys = list(raw_cfg.keys())
    if len(keys) == 1 and isinstance(raw_cfg[keys[0]], DictConfig):
        cfg = raw_cfg[keys[0]]
    else:
        cfg = raw_cfg
    OmegaConf.resolve(cfg)
    _setup_logging(cfg.get("debug", False))

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Set seed
    seed = cfg.get("seed", 42)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create controller
    controller = _create_controller(cfg)

    # Initialize pipeline
    _init_pipeline(controller, cfg)

    # Route to mode
    mode = cfg.controller_config.get("mode", "analysis")
    if mode == "profiling":
        _run_profiling(controller, cfg)
    else:
        _run_analysis(controller, cfg)

    # Cleanup
    controller.remove_hooks()
    logger.info("Run complete")


if __name__ == "__main__":
    main()
