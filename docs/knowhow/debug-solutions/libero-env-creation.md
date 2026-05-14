# LIBERO Environment Creation API

> LIBERO benchmark 对象没有 `get_env()` 方法，需用 `OffScreenRenderEnv` + `init_states`

## Problem
```python
env = bench.get_env(task_id)
# AttributeError: 'LIBERO_SPATIAL' object has no attribute 'get_env'
```

## Cause
LIBERO benchmark 类只提供 task metadata (`get_task`, `get_task_names`, `get_task_init_states`)，不直接创建 env。需要用 `OffScreenRenderEnv` 手动创建 + `set_init_state` 设置初始状态。

## Solution
```python
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
import libero.libero.benchmark as benchmark

bench = benchmark.get_benchmark("libero_spatial")()
task = bench.get_task(task_id)

# 创建 env (不是 bench.get_env)
bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=256, camera_widths=256)
env.seed(0)

# 设置初始状态
initial_states = bench.get_task_init_states(task_id)
env.reset()
obs = env.set_init_state(initial_states[ep % len(initial_states)])
```

参考: Cosmos Policy vendor 的 `get_libero_env()` 函数 (`cosmos_policy/experiments/robot/libero/libero_utils.py`)

## Notes
- Date: 2026-05-13
- 必须设 `MUJOCO_GL=egl` (headless rendering)
- EGL cleanup 警告 (`EGLError: EGL_NOT_INITIALIZED`) 是无害的
