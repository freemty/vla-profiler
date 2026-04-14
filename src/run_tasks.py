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
import os
import statistics
from typing import Any, Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf

# Import registries (triggers module-level registration)
from src.controllers import CONTROLLER_REGISTRY
import src.controllers.qwen_vl_controller  # noqa: F401 — register controller
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

    # Warmup runs
    for i in range(num_warmup):
        logger.info("Warmup run %d/%d", i + 1, num_warmup)
        controller.timer.reset()
        for inp in inputs_list:
            controller.model_inference(controller.pipeline, cfg, inp)
        logger.info("Warmup %d complete", i + 1)

    # Benchmark runs: collect timing per phase per run
    all_timings: Dict[str, List[float]] = {}

    for i in range(num_benchmark):
        logger.info("Benchmark run %d/%d", i + 1, num_benchmark)
        controller.timer.reset()
        for inp in inputs_list:
            controller.model_inference(controller.pipeline, cfg, inp)

        summary = controller.timer.summary()
        for phase, ms in summary.items():
            if phase not in all_timings:
                all_timings[phase] = [ms]
            else:
                all_timings[phase] = [*all_timings[phase], ms]

    # Aggregate timing stats
    aggregated: Dict[str, Any] = {"num_runs": num_benchmark}
    for phase, timings in all_timings.items():
        phase_stats = {
            "mean_ms": statistics.mean(timings),
            "std_ms": statistics.stdev(timings) if len(timings) > 1 else 0.0,
            "min_ms": min(timings),
            "max_ms": max(timings),
            "all_ms": timings,
        }
        aggregated[phase] = phase_stats

    controller._aggregated_timing = aggregated

    # Execute tasks
    _execute_tasks(controller, cfg)


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
    """Execute all configured tasks."""
    task_names = list(cfg.get("tasks", []))
    base_output = cfg.get("base_output_path", "output")
    save_dir = os.path.join(base_output, cfg.model_name.replace("/", "_"))
    os.makedirs(save_dir, exist_ok=True)

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
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
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
