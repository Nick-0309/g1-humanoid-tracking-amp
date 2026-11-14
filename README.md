# G1 人形机器人全身动作跟踪与 AMP 风格化控制

宇树 G1（29 DoF）在 Isaac Lab 3.0 上跟随一段 29 s 的参考舞蹈。仓库里是 AMP 判别器模块、着地奖励组、参考动作清理工具，以及评测与诊断脚本。

演示视频：

- `media/tracking.mp4` — 纯跟踪策略
- `media/amp.mp4` — 加入 AMP 之后

## 结果

### AMP

纯跟踪训 10k 轮，从这份权重热启动、加 AMP 再训 3k 轮。64 环境、29.14 s 整条、确定性评测：

| 指标 | 纯跟踪 | + AMP |
|---|---|---|
| 完整跟完成功率 | 100% | 100% |
| 关节位置误差 | 0.385 rad | 0.285 rad |
| 关键刚体位置误差 | 23.4 mm | 17.3 mm |
| 着地次数（左 / 右） | 4.1 / 7.8 | 11.2 / 19.0 |
| 接触不一致率 | 33.9% | 29.8% |
| 躯干位置误差 | 83.9 mm | 99.5 mm |

参考动作的着地次数是 16 / 23。躯干误差同时涨了 15.6 mm，这部分精度被风格项换走了。

### 着地

跟踪奖励只约束位置和速度，脚什么时候落地没有约束，所以单独加了一组着地奖励。4096 环境 × 3000 轮，同一份 checkpoint 热启动，一次只改一个变量：

| 参考动作 | 着地奖励 | 不一致率 | 摆动相误接触 | 关键刚体误差 | 动作 p95 |
|---|---|---|---|---|---|
| 原始 | 无 | 33.9% | 94.0% | 0.0234 m | 0.109 |
| 清理后 | 无 | 32.1% | 92.0% | 0.0245 m | 0.115 |
| 原始 | 有 | 8.6% | 20.4% | 0.0264 m | 0.139 |
| 清理后 | 有 | 7.4% | 17.9% | 0.0249 m | 0.141 |

只换清理后的动作、不加奖励，不一致率是 32.1%；加上奖励掉到 7.4%。

两脚各 26.4 次 touchdown，参考是 22 次，多出来的是碎步。降 `swing_clear` 权重或者加 `--foot_termination` 可以收掉。

## 方法

### 任务与环境

- G1 29 DoF，策略 50 Hz，物理仿真 200 Hz（decimation 4）
- 参考动作 29.14 s，hhtools 重定向到 G1，重采样到 50 Hz
- 4096 并行环境，rollout 24 步，5 epoch × 4 minibatch
- RSI 参考状态初始化、自适应采样、域随机化（地面摩擦、躯干质心、关节零位）、随机推力
- 终止条件：躯干高度、朝向或末端高度偏离参考超过阈值

### 观测与动作

观测约 154 维：参考关节位置和速度 58、参考与机器人的躯干朝向差 6、机器人角速度 3、关节位置 29、关节速度 29、上一步动作 29。

动作 29 维，是默认位姿上的偏移乘以一个按关节力矩和刚度算出的缩放系数，作为 PD 控制器的目标角。

Critic 额外拿到各刚体位姿和线速度，Actor 只用真机上能拿到的量。

### 奖励

跟踪 6 项，都是 `exp(−err²/std²)`：

| 项 | 权重 | std |
|---|---|---|
| 躯干位置 | 0.5 | 0.3 m |
| 躯干朝向 | 0.5 | 0.4 |
| 14 个刚体相对位置 | 1.0 | 0.3 m |
| 刚体朝向 | 1.0 | 0.4 |
| 刚体线速度 | 1.0 | 1.0 m/s |
| 刚体角速度 | 1.0 | 3.14 rad/s |

正则项：关节加速度与力矩、动作变化率 −0.1、关节限位 −2.0、自碰撞 −0.1。

着地 6 项：`stance_loaded` 0.4、`swing_clear` 0.6、`foot_height` 0.2（std 4 cm）、`anchor_height` 0.1（std 5 cm）、`feet_slide_world` −0.05、`action_l2` −0.002。权重是在基线上用 `scripts/calibrate_rewards.py` 量出来的缺口定的，都集中在 `src/g1dance/tasks/env_cfg.py` 的 `LANDING_REWARDS` / `LANDING_STDS` 里。

### AMP 判别器

输入 3 帧 × 79 维 = 237 维。每帧：骨盆角速度 3、29 个关节位置、29 个关节速度、6 个关键刚体（脚踝、手腕、肘）在 heading 系下相对骨盆的位置 18。位置放在只保留 yaw 的坐标系里，对世界朝向不敏感，但留着身体倾斜。

网络 237 → 1024 → 512 → 1，ELU，输入做滑动标准化。损失是 LSGAN，加专家样本上的梯度平方范数惩罚（λ=5）和输出层 L2（0.05），lr 3e-5，梯度裁剪 1.0。每轮 PPO 更新之后用同一批 rollout 更新一次判别器。

风格奖励 r = (D+1)/2，D 截断到 [−3, 1]，总奖励 = 0.9 × 任务 + 0.1 × 风格。

3 帧窗口不跨 reset：reset 后重新攒满 3 帧才进判别器，终止帧取自 `extras["final_obs"]`，窗口不完整的样本用同一步其它样本的均值代替。

判别器、优化器、归一化统计一起存进 checkpoint，中断后能原样续训。

风格奖励用的是线性映射。更常见的 `−log(1 − 0.25(D+1)²)` 在 D ≤ −1 时恒为 0，判别器把策略压到低分区之后这一项就没有区分度了。

### 参考动作清理

重定向出来的是纯运动学轨迹，脚底可以穿进地面，摆动可以蹭着支撑脚过去。`scripts/data/clean_motion.py` 改两件事，都只作用在参考上：

1. 接地对齐。在至少一只脚贴地的帧上，把最低脚底对到 z = 0。只改根节点高度、关节不动，clip 仍然自洽，等同于重定向里用支撑脚反推根节点高度那一步。两脚都悬空的帧用插值保留原高度。
2. 摆动抬脚。按跟踪任务用的姿态标签划出摆动区间，在中段用升余弦把脚抬到离地 3 cm，两端为零，所以 touchdown 和离地时刻（也就是接触标签）不会移动。抬脚走腿 IK（位置 + 朝向，6 自由度，`mj_jacBody` + 阻尼最小二乘，关节限位硬约束），824 帧最大残差 0.1 mm。

脚底高度取 MJCF 里脚上 4 个接触球的最低点，位置在 `ankle_roll_link` 下 35 mm，和 Isaac 里实测的站立踝高一致。

## 代码结构

```
src/g1dance/amp/      判别器、特征、滑动窗口、损失、AmpModule、AmpOnPolicyRunner
src/g1dance/tasks/    任务注册、环境配置（amp 观测组、着地奖励权重）、PPO 超参
src/g1dance/mdp/      参考动作命令、观测、残差动作、接触标签
src/g1dance/data/     hhtools 转换、参考动作清理（foot_model / motion_cleanup）、数据检查
scripts/              train / eval / play / calibrate_rewards / diagnose_amp / offline_disc
tests/                75 个单元测试，纯 torch / numpy，不需要 Isaac Sim
cloud/                tmux 训练与对照实验脚本
```

`eval.py` 每个环境从第 0 帧完整跑一遍，输出成功率、各项跟踪误差、着地次数、接触不一致率、动作抖动 P95。

## 运行

需要 Isaac Lab 3.0、`unitree_rl_lab` 和 G1 资产，以及 `rsl_rl`。

```bash
# 单元测试（不需要仿真器）
pytest -q

# 纯跟踪基线
python scripts/train.py --task G1D-Tracking-v0 --num_envs 4096 --max_iterations 10000 \
    --run_name track --headless

# 从基线热启动，加 AMP
python scripts/train.py --task G1D-AMP-v0 --num_envs 4096 --max_iterations 3000 \
    --run_name amp --init_from logs/g1d_tracking/<run>/model_9999.pt --disc_warmup 20 --headless

# 参考动作清理（离线，只需要 mujoco）
python scripts/data/clean_motion.py motions/dance_102.npz --fix

# 着地奖励的配对对照：同一初始化、同样轮数，只差 --foot_rewards
bash cloud/run_pairs.sh

# 评估与录像
python scripts/eval.py --task G1D-AMP-v0 --load_run amp --num_envs 64 --headless
python scripts/play.py --task G1D-AMP-v0 --load_run amp --video --headless
```

参考动作不在仓库里。`motions/dance_102.npz` 是 `unitree_rl_lab` 自带的已重定向舞蹈；换自己的动作走 `scripts/data/prepare_motion.py`（BVH/GLB → hhtools 重定向 → npz）。

## 依赖

训练框架是 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 和 `unitree_rl_lab` 的动作跟踪任务，PPO 用 `rsl_rl`，动作重定向用 [hhtools](https://github.com/Roboparty/human-humanoid-tools)。本仓库是在这之上的任务配置、AMP 模块、着地奖励、清理工具和评测脚本。

AMP 出自 *Learning Agile Robotic Locomotion Skills by Imitating Animals*（arXiv:2104.02180）。

## 说明

- 单随机种子。着地对照是 3000 轮热启动，还没有从零训练的同等轮数基线。
- 只有一段参考舞蹈，没验证换动作之后的泛化。
- 没上真机。Isaac → MuJoCo 的 sim2sim 在做。

## License

MIT
