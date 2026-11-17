# 训练性能 — 2026-11-18

完整200轮32768环境训练耗时：PPO7 seed1 36.4分钟、PPO9 seed1 38.7分钟、PPO7 seed2 37.0分钟、PPO9 seed2 37.2分钟。用户观察到相同配置约2.8倍端到端加速，训练曲线及表现接近历史版本。该2.8倍是完整任务观察值，不能用微基准5.3倍物理加速替代。

物理保持5×10ms子步、12次二分、2次牛顿修正、动态括区、电机/LSD和全部状态。加速来源是子步不变量缓存、Fx残差专用计算及CUDA图。网络更新保留既有AMP和运算序列；FP32推理为默认。FP16仍为可选实验接口，中附8字对小扰动敏感，未作为无影响替代推广。

已有数值验证：48环境三附着30秒物理状态/Fx逐位一致；完整环境40步含16次reset的状态/观测/奖励逐位一致；图捕获后显式参数DR一致；3轮全规模训练120个checkpoint tensor叶子与参考路径一致。这些是有限范围验证，不是任意长训练逐位相同保证。

完整实验总wall time3小时34分47秒，包含串行CPU评测：372份CarSim、36份FastPlant（各8初态）。FastPlant评测造成队列拖尾，今后不再作为策略验收阶段。只保留必要的物理/数值回归。

诊断接口仍保留 `--fastplant-solver original|cached|cuda_graph|auto` 和 `--network-execution reference|cuda_graph`；不维持两个正式训练默认。详细原始性能证据在父目录 FastPlant_Optimization_20261117、Network_Optimization_20261117、DSH_Accelerated_200_2026-11-17。
