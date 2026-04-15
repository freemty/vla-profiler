# ACT Action Queue Prevents Hook Triggering

> LeRobot ACT 的 `select_action()` 内部有 action queue 缓存，导致 PyTorch forward hooks 只在第一次调用时触发。

## Problem
ACT profiling 显示所有 phase timing 为 0ms — hooks 注册在 `model.backbone` 和 `model.decoder` 上，但 20 次 benchmark run 中 hooks 从未触发。

## Cause
ACT 的 `select_action()` 实现了 action chunking 缓存：
```python
# LeRobot ACTPolicy.select_action()
if len(self._action_queue) == 0:
    actions = self.predict_action_chunk(batch)  # 实际调 model.forward()
    self._action_queue.extend(actions.transpose(0, 1))
return self._action_queue.popleft()  # 后续 N-1 次直接 dequeue
```
chunk_size=100 意味着只有每 100 次调用才触发一次 `model.forward()`。如果 `timer.reset()` 在每次 benchmark run 前调用，第 2-100 次 run 的 hooks 永远不会触发。

## Solution
对 profiling 场景，直接调 `model.forward()` 而不是 `select_action()`：
```python
# WRONG: select_action 有缓存
action = pipeline.model.select_action(observation)

# CORRECT: 每次都触发完整 forward pass
actions, _ = pipeline.model.model(observation)
```

## Notes
- Date: 2026-04-15
- 适用于所有有 action chunking/caching 的 VLA policy
- LeRobot 的 Pi0、SmolVLA 等 policy 可能也有类似缓存机制
