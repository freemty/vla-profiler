# xdlab23 cuDNN Version Mismatch

> System cuDNN 9.1.1 vs torch 2.9 needs 9.10+ — DiT models crash with `cudnnGetLibConfig` undefined symbol

## Problem
Cosmos Policy 和 LingBot-VA (DiT 模型) 在 xdlab23 上启动时 core dump：
```
Could not load symbol cudnnGetLibConfig. Error: /lib/x86_64-linux-gnu/libcudnn_graph.so.9: undefined symbol: cudnnGetLibConfig
Aborted (core dumped)
```

## Cause
- System cuDNN: 9.1.1 (`/usr/lib/x86_64-linux-gnu/libcudnn*.so.9.1.1`)
- torch 2.9+cu128 需要 cuDNN ≥ 9.10 (报告 `torch.backends.cudnn.version() = 91002`)
- pip 包 `nvidia-cudnn-cu12==9.10.2.21` 已安装但 system lib 优先被 ld 加载

## Solution
在所有 eval 脚本中前置 pip cuDNN 的 `LD_LIBRARY_PATH`：
```bash
export LD_LIBRARY_PATH=/home/ybyang/miniconda3/envs/vit-probe/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
```

已应用到：
- `scripts/run_libero_all.sh` (全局)
- `scripts/run_exp04d_libero.sh`
- `scripts/run_exp04d_parallel.sh`

## Commands
```bash
# 诊断
python -c "import torch; print(torch.backends.cudnn.version())"  # 91002
ldconfig -p | grep cudnn  # 显示 system 9.1.1
find /home/ybyang/miniconda3/envs/vit-probe/lib/python3.12/site-packages/nvidia -name "libcudnn*"  # pip 9.10

# 修复 (在 bash 脚本或 shell 中)
export LD_LIBRARY_PATH=/home/ybyang/miniconda3/envs/vit-probe/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

# 验证
python -c "import torch; t=torch.randn(2,3,4,4,device='cuda'); torch.nn.functional.conv2d(t,torch.randn(3,3,3,3,device='cuda')); print('cuDNN OK')"
```

## Notes
- Date: 2026-05-13
- Environment: xdlab23, vit-probe conda env, torch 2.9+cu128
- 影响: 所有 DiT 类模型 (Cosmos Policy, LingBot-VA/WAN, NitroGen) — non-DiT 模型 (VLA, ACT) 不受影响
