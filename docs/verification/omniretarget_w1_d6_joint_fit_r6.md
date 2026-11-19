# W1 固定六窗：R6 只读准入

本记录与[冻结方案 v0.2](../plans/omniretarget_w1_d6_joint_fit_diagnostic_plan.md)和[I6 纯合成复阅](omniretarget_w1_d6_joint_fit_i6.md)分开。R6 不是拟合结果，也不授予 D6/F 权限。

## 准入和执行门

- [R6 入口](../../scripts/run_w1_d6_r6.py)只接受固定的一次尝试。启动前由独立调用方采样单调时钟；独占 audit 根、同步登记 running 台账，再启动锁定环境。R6 只核路径关系、JSON、字节哈希、NPY 头及环境/设备/磁盘元数据。105 s 后不准新增工作；120 s 内必须覆盖实际子进程返回、回收、定额日志、最后存档同步及回读。`consume` 需要调用方的实际返回见证，无见证或仅有本地候选均不可消费。失败保留记录，不重试。
- [SOURCE 与运行时绑定](../../src/ireset/eval/w1_d6_r6.py)重推 `pilot6 ∩ train24` 的六个 `[0,60)` 训练窗；对照原登记及指定迁移缓存的 55 对日程文件哈希与历史 55/55 逐位证明，不重比对数组值。真实运行只能用原登记 `train24` 的 `root_stats` 及这六份绑定迁移日程；不会按时间挑选缓存。资产从 A1 与 FK 注册对账（四个锁定文件及 37 个网格引用），运行时身份与 T6 注册对账。CPU/GPU 数值试运行不属于 R6。
- [唯一真实调用矩阵](../../src/ireset/eval/w1_d6_joint_fit_protocol.py)由 R6 封存并由 [D6/F 执行门](../../src/ireset/eval/w1_d6_joint_fit_real.py)消费。D6 六个派生场景和六份日程各读一次，从已验字节重建；F 不重建窗口，先核元数据，再校验两端 checkpoint，最后才构造 fresh 模块并共用一份已验输入。每阶段统计量快照 1 次、资产快照 41 次；场景编译和已绑定缓存命中是互斥分支。冷路径的关节范围 plain 编译、场景装置另一次 plain 编译、scene 编译、`MjData` 和 `mj_forward` 分列，不能以一次访问代替。调用计数是完整路径契约，不是 R6 实测生产计数。
- D6 固定 1000 次原生 AdamW、4000 micro、24,000 窗次、101 次整批 probe / 606 窗行；调用上限合计 conditioner/backbone 各 4103，生产 FK batch 为 D6 8612 / F 12。更新 501 首个 micro 前须经 500 行、51 整批/306 窗行及已绑定来源的落盘门才全组切换 LR。D6/F 硬上限/收口预算仍为 `3600/120`、`600/60` s；F 从 D6 实际关账接续，整链最后 60 s 仅外层可用。启动空闲至少 7 GiB，受管峰值至多 2 GiB（含临时 checkpoint、日志、并存写入和收口预留）。
- R6 与真实 CLI 不相互启动。仅覆盖通过的 R6 可导出 `authorized=false` 草案；真实执行另须 §8.3 新的明确一次性授权，范围严格 `includes=["D6","F"]`。不使用 D1 handoff、不追加恒定 LR 对照、无补跑或自动执行。

## 唯一一次 R6：2026-11-19

**`r6_admitted`，`covered=true`，无问题。** 从 CLI 启动前采样到实际退出、日志、外层观察存档与最终同步回读为 **3.958656 s ≤ 登记 8.307731 s ≤ 硬上限 120 s**；工作候选在 **3.307731 s < 105 s**。worker、collector、最外层捕获的 CLI 实际退出码均为 0，清理均已确认。独立只读消费者以这些实际返回观察而非退出码或本地报告单独放行。

- 依注册规范序冻结 `sub10_largebox_097`、`sub12_largebox_082`、`sub16_largebox_000`、`sub3_largebox_027`、`sub7_largebox_022`、`sub8_largebox_035` 的 `[0,60)` 训练窗。共绑定 421 个来源文件，核 20 个 NPZ 的 70 个成员头；迁移证据 55/55，原 `train24` 根统计 24 文件/4351 帧，原件与迁移副本同 SHA `b9d9cc049a6b58eea7ab048b886a5fd9f87755690d1f0a789cb4f113c5bd2eda`。一次真实运行时元数据对账匹配 T6（RTX 3090）；未做 GPU 数值探针。空闲 23.12 GiB，D6/F 冻结工件上界约 202 MiB（设计上界，非实测峰值）。
- 实现提交 `274fc5a`，绑定 84 文件，SHA `b2afcb4ce26f2c436dc42b9bbc123fb83aeb4c8f2f066cb6c1d90849b6041607`；真实矩阵 SHA `d4f79600f8167e45a198a00bc9406b5858f42bda93807dde8d889d8507868b42`。最终字节定向 34、I6 90、旧单窗 LR 73、旧拟合/外层 80、准入/N0/N1 91 项通过（各组独立执行，未跑全项目套件）。历史及冻结 449 项哈希前后都一致。
- [封口准入包](../../derived/audit/w1_d6_joint_fit_lr_half_v1/r6/attempt_001/r6_admission.json) SHA `9e54107c7a6bf812b0f87098f1925e1d2f09293f4c1affd825179c28988eb527`；[独立复核](../../derived/audit/w1_d6_joint_fit_lr_half_v1/r6/review/independent_verification.json) SHA `b309a0cb77f34f330fe1c2efb1487f7414bdc915185bd2787c1712b96de8e3a5`，含实际退出、末次采样及各级原始文件绑定。[未签草案](../../derived/audit/w1_d6_joint_fit_lr_half_v1/r6/review/authorization_draft.json) SHA `5a6db751bd9f1932dc4c4b5710b9c5d2f22af7fcb500e6fa327af2d5253ceb1d`，`authorized=false`、清单全 `null`，不写正式授权路径。

**仅消费一次 R6 尝试，不再启动 R6。** 未解码真实数组值、未加载 checkpoint、未构造真实模型/FK、未编译真实场景或重建缓存。D6/F 未授权、未运行，真实实验根未创建。R6 不证明真实拟合可达、显存充足、能在预算内完成训练或端点可逐位复现。
