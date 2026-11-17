![Overview](assets/fig/banner.png)
# Racer-HitchOpen-RLGPU

CarSim规模后驱电动车的持续漂移与路径跟踪。基于XCAR的GPU并行仿真和PPO范式，使用可观测actor、独立特权critic及域随机化。训练使用FastPlant，策略验收直接使用CarSim。

## 当前默认：PPO9

```powershell
python run.py train continuous_drift_iwd
# 显式覆盖示例；输出默认在仓库外 ../training_runs
python run.py train continuous_drift_iwd --seed 20261114 --epochs 200
```

默认配置集中在 [profiles/ppo9.json](profiles/ppo9.json)，命令行显式选项优先。仅continuous_drift_iwd训练自动应用PPO9；其他任务和test保持旧接口。`--profile legacy`恢复历史CLI默认，供对照实验；这不代表推荐历史训练方案。PPO7可覆盖 `--preview-time .25,.5,1,1.5,2,2.5,3`。

- 9点时间预瞄：0.25/0.5/0.75/1/1.25/1.5/2/2.5/3秒；距离由前轮平均速度确定，坐标基于车身航向。
- 真实观测旁路 + GRU128 + 256/124/64动作网络；结构镜像、低速渐隐β辅助任务、stop-gradient β反馈；独立特权critic。
- 完整动力学/传感器/执行器DR；不人为添加观测延时或丢帧，执行器动力学仍保留。
- 跟踪宽度1.5m、平滑权重0.1、uncapped速度奖励；原有随机路径分布，持续任务。
- 20Hz控制、100Hz物理；32768环境、horizon40、BPTT片段10、minibatch81920、4次mini-epochs、200 epochs。默认seed20261115，需多种子评估。
- CUDA图物理和网络更新；FP32 actor推理、既有AMP更新。物理子步/求解精度不削减。

## 安装与运行

Python环境依赖见 [environment.yml](environment.yml)。本机已验证环境为 `D:/Anaconda/envs/xcar-win/python.exe`；GPU规模需适配显存。CarSim为单独授权的外部依赖，场景位于本地项目的 `../CarSim_Configuration`，不随Git发布。

```powershell
# 明确checkpoint、匹配配置和参考路径，直接CarSim评测（无FastPlant评测队列）
python experiments/carsim_bridge/evaluate_checkpoint.py --checkpoint CHECKPOINT.pth --config ENV_CONFIG.json --reference-path PATH.npy --output EXTERNAL_OUTPUT
```

每次使用独立输出目录，保存原生CarSim结果，并备份后更新对应LastRun。不要并行启动多个CarSim写入进程。历史动画和TensorBoard入口见 [STATUS.md](STATUS.md)。

## 结果与工程说明

200轮PPO9两个训练种子均完成29/29项120秒CarSim测试；该结果不是任意轨迹、任意初态的100%保证。中附8字漂移连续性、高速直线摆动及跨种子速度差异仍需改善。

[当前状态与结果](STATUS.md) · [性能实测](PERFORMANCE.md) · [物理模型](PLANT.md) · [数据与备份](ARTIFACTS.md)

保留PPO7/其他预瞄数、DSAC-T、奖励/网络实验及原始IWD任务接口。原始/缓存物理解法仅作为数值诊断基准；正式默认使用已验证的加速方案。过期日期固定的调度器已归档到仓库外，下一轮计划由讨论后生成。

本仓库源自本地 `xcar-rlgpu` 项目，原有署名和授权见 [LICENSE](LICENSE)。
