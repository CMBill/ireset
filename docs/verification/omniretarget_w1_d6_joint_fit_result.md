# W1 固定六窗联合拟合：一次真实 D6 → F 结论

2026-11-20（本机 +08:00）。[冻结方案 v0.2](../plans/omniretarget_w1_d6_joint_fit_diagnostic_plan.md) **已执行并验收，裁决 `fit_insufficient`**。`complete_evidence=true`、`replay_passed=true`、`mean_fit_passed=false`。在冻结配置、seed `20261008` 和 1000 步下，**0/6** 个固定训练窗在末次 probe 同时达到四项平均门。这是证据完整的负结果，不是执行中断或重放失败。

## 授权与来源

- [I6 合成正负控](omniretarget_w1_d6_joint_fit_i6.md)、[唯一 R6 准入](omniretarget_w1_d6_joint_fit_r6.md)和[授权前复核](omniretarget_w1_d6_joint_fit_preauthorization.md)分别验收。R6 保留原身份和实测退出证据，没有重跑。当前 88 文件实现 SHA-256 为 `64b5b079436eb66728237bcb0b275801731e040ba53fa5546d29890a2b879daf`（执行时 HEAD `b138bb4`）；调用矩阵 SHA-256 仍为 `d4f79600f8167e45a198a00bc9406b5858f42bda93807dde8d889d8507868b42`。
- 经用户明确确认 §8.3 五项并绑定[审阅材料](../../derived/audit/w1_d6_joint_fit_lr_half_v1/preauthorization_v2/review_materials.json) SHA-256 `4067acde7b1620eed3ce12f7d2903bf88712810aca439c489abd0309d17e2d31`，签发[正式授权](../../derived/audit/w1_d6_joint_fit_lr_half_v1/authorization.json) SHA-256 `28c224564677dc393442473e39cc57ab5c0ad2ec8d96fdca49970a51ac53800a`。范围精确 `includes=["D6","F"]`、每阶段最多一次。固定根[独占消费记录](../../derived/experiments/omniretarget_first/w1_d6_joint_fit_lr_half_v1/authorization_consumed.json) SHA-256 `70c63171bc9cd05e5079fb97928999e2d71249a9d9fd883cc7a960a44dec3a63`；一次 D6 和一次条件 F 均已消费。未签草案、旧授权和冻结方案保持原字节。
- 正式启动前空闲 24,122,339,328 bytes（约 22.47 GiB，要求 ≥7 GiB）。421 个来源及 449 个保护条目执行前后哈希一致；只使用登记的 `pilot6 ∩ train24` 六首窗、55/55 已绑定迁移日程和原 `train24` 根统计。两阶段的构造、转移、加载与终点报告核对 `cuda:0`/float32、TF32 与 autocast 关闭、`highest` matmul 和既定运行时身份；CPU 初始/参考构造另验。

## 执行与实际返回

- [D6 报告](../../derived/experiments/omniretarget_first/w1_d6_joint_fit_lr_half_v1/d6/report.json)和[工件索引](../../derived/experiments/omniretarget_first/w1_d6_joint_fit_lr_half_v1/d6/artifacts_index.json)绑定[实际窗载荷](../../derived/experiments/omniretarget_first/w1_d6_joint_fit_lr_half_v1/d6/payload.json)、[逐步更新账](../../derived/experiments/omniretarget_first/w1_d6_joint_fit_lr_half_v1/d6/history.jsonl)和[逐 probe 日志](../../derived/experiments/omniretarget_first/w1_d6_joint_fit_lr_half_v1/d6/probe_index.jsonl)。一个共享的确定性模型从头完成 **1000 更新、4000 micro、24,000 窗次**，整批 **101** 次 probe、**606** 条窗行。D6 实测 conditioner/backbone 各 4101 次、生产 FK batch 8612 次。逐槽载荷和原生 AdamW 连续性均已核对：[前缀收据](../../derived/experiments/omniretarget_first/w1_d6_joint_fit_lr_half_v1/d6/prefix_receipt.json)覆盖 500 行、51 批、306 窗行；[切换收据](../../derived/experiments/omniretarget_first/w1_d6_joint_fit_lr_half_v1/d6/switch.json)确认唯一参数组在第 501 步首个 micro 前由 `1e-4` 改为 `5e-5`，第 500 步后态、切换两侧及第 501 步前态摘要相同。
- D6 本地完整但拟合不通过，实际关账后才准入 [F](../../derived/experiments/omniretarget_first/w1_d6_joint_fit_lr_half_v1/f/report.json)。F 在预算内核对全部 606 条保存指标，验证两份 checkpoint 后用同一份已验输入快照做 4 次 strict module load 和 2 次 `batch=6` 整批 probe。`initial`/`last` 的原始 FP32 `x0`、生产解码和预测 FK 点逐位一致。F 的 conditioner/backbone 各 2 次、预测 FK 12 次，零更新。D6/F 全部类别实测计数均匹配冻结矩阵。

| 账目 | 实际耗时（秒） | 登记（秒） | 硬上限（秒） |
| --- | ---: | ---: | ---: |
| R6 独立历史账（未重跑） | 3.958656 | 8.307731 | 120 |
| D6 实际关账 | 1625.942543 | 1629.820263 | 3600 |
| F 自 D6 实际关账至 F 关账 | 11.666049 | 15.582792 | 600 |
| D6 → F，含父进程核验和见证副本同步回读 | **1638.648367** | **1642.608866** | **4200** |

D6/F 候选采样在各自工作截止前（1624.820263/10.582792 秒，工作上限 3480/540 秒），阶段返回、清理和证据回读均被本阶段登记覆盖。最后 60 秒外层专用额度未借给阶段。R6 与整链实测相加 1642.607023 秒，不重复叠加 D6/F 子阶段。运行树最终 138 文件、79,105,685 bytes；写前按含临时文件、日志和收口预留的 212,361,216 bytes 设计上界治理，低于 2 GiB，不把结束值当峰值实测。八路定额日志绑定到返回证据；D6/F stderr 各 5014 bytes 的 URDF `collision` 属性提示按原样保留。

父进程在导入真实入口前取单调起点，主线程直接调用 `launch_authorized` 并持有真实 `ExternalReturn`。collector 实际 `wait` 为 exit 0、清理完成；API 末次回读为 1638.353064 秒。同一父进程把原见证交给只读 `consume`，在 1638.626520 秒返回，再同步回读见证副本（上表末行）。[运行根 `outer_observation.json`](../../derived/experiments/omniretarget_first/w1_d6_joint_fit_lr_half_v1/outer_observation.json)仍为 `pending_actual_readback`，不凭该文件或裸退出码发布结论。

## 第 1000 步逐窗裁决

粗体表示超过门限；表中为显示舍入，实际判门使用未舍入 SI 值。完整数值、超门倍数、p95、max 和最坏坐标见[独立机读裁决](../../derived/audit/w1_d6_joint_fit_lr_half_v1/execution_review/verified_outcome.json)。

| 规范窗键 | 根位置 RMSE（m，≤0.01） | 根旋转均值（rad，≤π/180） | 关节角 RMSE（rad，≤0.05） | 教师 FK MPJPE（m，≤0.01） |
| --- | ---: | ---: | ---: | ---: |
| `sub10_largebox_097:0:60` | 0.009562 | **0.022385** | **0.156749** | **0.044030** |
| `sub12_largebox_082:0:60` | **0.011767** | **0.029225** | **0.196529** | **0.052608** |
| `sub16_largebox_000:0:60` | **0.010439** | **0.030105** | **0.133492** | **0.043565** |
| `sub3_largebox_027:0:60` | 0.008482 | **0.023703** | **0.100144** | **0.041268** |
| `sub7_largebox_022:0:60` | 0.008532 | **0.026194** | **0.103980** | **0.039744** |
| `sub8_largebox_035:0:60` | **0.010521** | **0.027901** | **0.117922** | **0.047251** |
| 达标窗数 | 3/6 | 0/6 | 0/6 | 0/6 |
| 等权宏平均（不判门） | 0.009884 | 0.026586 | 0.134803 | 0.044744 |

最坏门为 `sub12_largebox_082` 的教师 FK（约 **5.26×** 门限，52.61 mm 对 10 mm）。复阅全部保存曲线：六窗各自四项指标均由 step500 到 step1000 降低，但 101 次记录中没有一个窗曾同时过四门；根旋转、关节角、教师 FK 三项各自在全部 606 条窗行中均未过门。根位置宏平均虽低于门限，仍有三窗末次不达标。各指标的最差窗不全相同（根旋转最大在 `sub16`，其余指标最大在 `sub12`）；这些是保存指标的描述，不归因于梯度冲突、容量或 LR 日程，也不把单窗 B 当成六窗对照。

## 独立只读验收与停止线

[调用方保存的见证](../../derived/audit/w1_d6_joint_fit_lr_half_v1/execution_review/supervisor_return.json) SHA-256 `fa4d5059568e4b8a5e83fc4902ab0264fc6964c80e3b68634378a6a852ce8cfb`。另起只读消费者从该副本恢复真实见证，复核授权消费、阶段关账、存档和日志字节，并核对全部更新/LR/槽位顺序、前缀和 101 × 6 指标日志。`verified_outcome.json` SHA-256 `1de06dae2eca5d554b948db24e33399647f868ef60283ace4775e887e57e56b0`；另含完整实测调用、预算、逐窗门和最后回读现场观察。复核未导入数值库、未读取真实数组值或再加载 checkpoint，只读了哈希与 JSON，没有更改运行树或补做数值单元。

按方案在此收口，不重试、不续训、不重跑 R6，不启动 D1、六窗恒定 LR 对照、DDIM、正式 val/test 或生产接入。`fk_replay_scope=initial_last_only`；中间 FK、教师 FK 的状态到点映射和尖峰门仍 `not_evaluated`。不外推泛化、碰撞或持续抓握，T6 的 `audit_incomplete`/539 缺口与旧拟合结论保持。
