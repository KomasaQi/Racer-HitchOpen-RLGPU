# 训练性能 — 2026-11-18

完整200轮32768环境训练耗时：PPO7 seed1 36.4分钟、PPO9 seed1 38.7分钟、PPO7 seed2 37.0分钟、PPO9 seed2 37.2分钟。用户观察到相同配置约2.8倍端到端加速，训练曲线及表现接近历史版本。该2.8倍是完整任务观察值，不能用微基准5.3倍物理加速替代。

物理保持5×10ms子步、12次二分、2次牛顿修正、动态括区、电机/LSD和全部状态。加速来源是子步不变量缓存、Fx残差专用计算及CUDA图。网络更新保留既有AMP和运算序列；FP32推理为默认。FP16仍为可选实验接口，中附8字对小扰动敏感，未作为无影响替代推广。

已有数值验证：48环境三附着30秒物理状态/Fx逐位一致；完整环境40步含16次reset的状态/观测/奖励逐位一致；图捕获后显式参数DR一致；3轮全规模训练120个checkpoint tensor叶子与参考路径一致。这些是有限范围验证，不是任意长训练逐位相同保证。

完整实验总wall time3小时34分47秒，包含串行CPU评测：372份CarSim、36份FastPlant（各8初态）。FastPlant评测造成队列拖尾，今后不再作为策略验收阶段。只保留必要的物理/数值回归。

诊断接口仍保留 `--fastplant-solver original|cached|cuda_graph|auto` 和 `--network-execution reference|cuda_graph`；不维持两个正式训练默认。详细原始性能证据在父目录 FastPlant_Optimization_20261117、Network_Optimization_20261117、DSH_Accelerated_200_2026-11-17。

### Windows 全量 PPO9 训练

在 RTX 5070 Laptop、PyTorch 2.7.1+cu128、cuDNN 9.7.1 上，可在启动训练的 PowerShell 中设置以下选项。CUDA Toolkit 12 的安装目录通过 `CUDA_PATH` 提供。续训时在同一命令后添加 `--resume` 和已有 checkpoint 路径。

```powershell
$env:RACER_NATIVE_WHEEL_SOLVER = '1'
$env:RACER_NATIVE_GRU = '1'
python run.py train continuous_drift_iwd --profile ppo9 --fastplant-solver plant_cached_graph
```

原生轮速核执行完整的隐式轮速求解。GRU 加速用于 32,768 环境、九点预瞄、128 维隐状态的 FP32 采样，沿用原有镜像分支和 checkpoint 格式。梯度更新继续走现有 PPO 训练路径。两个选项可独立启用。

全量配对基准从同一第 265 轮 checkpoint 训练至第 280 轮，每轮 1,310,720 帧。两组各自完成首轮预热后，后 14 轮 TensorBoard 墙钟耗时为 148.01 秒和 65.62 秒，端到端加速 2.26×。第 270、275、280 轮的 actor、独立 critic、两套 Adam 和固定输入动作逐元素相同。该数据对应上述设备及回合区间。
