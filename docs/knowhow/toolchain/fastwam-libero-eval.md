# Fast-WAM LIBERO Eval 接口

> Fast-WAM 的 LIBERO eval 用 Hydra config 而非 argparse，且 single-process 模式默认只跑 task_id=0。

## Problem

首次尝试跑 Fast-WAM LIBERO eval 时踩了三个坑：
1. `eval_libero_single.py` 用 Hydra 而非 argparse → `--checkpoint_path` 等参数不被识别
2. 默认 `EVALUATION.task_id=0` → 每个 suite 只跑 1 个 task（非全部 10 个）
3. `run_libero_manager.py` 需要 `run_libero_parallel_test.sh` 但该脚本不存在

## Solution

### 正确的单 task 调用
```bash
cd /data1/ybyang/FastWAM
conda activate fastwam
export MUJOCO_GL=egl

CUDA_VISIBLE_DEVICES=0 python experiments/libero/eval_libero_single.py \
  --config-name sim_libero \
  ckpt=checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=20 \
  EVALUATION.num_inference_steps=5 \
  EVALUATION.dataset_stats_path=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.output_dir=/path/to/output
```

### 遍历全部 tasks (循环)
```bash
for suite in libero_spatial libero_object libero_goal libero_10; do
  for tid in $(seq 0 9); do
    CUDA_VISIBLE_DEVICES=0 python experiments/libero/eval_libero_single.py \
      --config-name sim_libero \
      ckpt=... \
      EVALUATION.task_suite_name=$suite \
      EVALUATION.task_id=$tid \
      EVALUATION.num_trials=20 \
      EVALUATION.num_inference_steps=5 \
      EVALUATION.dataset_stats_path=... \
      EVALUATION.output_dir=output/$suite
  done
done
```

### 关键 Hydra overrides

| Override | Default | 含义 |
|----------|---------|------|
| `ckpt` | null | checkpoint 路径 (必填) |
| `EVALUATION.task_suite_name` | libero_spatial | suite 名 |
| `EVALUATION.task_id` | 0 | **只跑这一个 task** |
| `EVALUATION.num_trials` | 50 | 每 task episode 数 |
| `EVALUATION.num_inference_steps` | 从 config | denoise 步数 |
| `EVALUATION.dataset_stats_path` | null | 归一化 stats JSON |
| `EVALUATION.output_dir` | auto | 结果输出目录 |

### 结果文件
每个 task 生成 `gpu0_task{id}_results.json`，内含:
```json
{"task_suite": "...", "task_id": 0, "successes": 19, "total_episodes": 20, ...}
```

## Notes
- Date: 2026-04-29
- Environment: xdlab23, fastwam conda env (Python 3.10)
- `run_libero_manager.py` 的多 GPU 分发模式因缺少 shell 脚本不可用，直接循环更可靠
- 首次运行需要下载 Wan2.2 text encoder (~10GB from HF/ModelScope)，后续走 cache
- 800 episodes (4 suites × 10 tasks × 20 ep) 耗时约 6 小时 (RTX 5880 Ada, 5-step)
