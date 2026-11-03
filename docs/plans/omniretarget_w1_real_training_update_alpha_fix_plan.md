# W1 真实训练更新审计——alpha 有效常数修正补充计划

日期：2026-11-03。版本：v0.12，承接冻结执行计划 v0.11。本轮范围为实现修复、纯合成回归与 Gate B 重冻结；真实恢复未授权。

## 1. 固定依据与已消耗额度

- [预注册 v0.9](omniretarget_w1_real_training_update_audit_prereg_draft.md)，SHA `e2ba8ef9edb7381dbe410f7e0e290a6fbd702323a4416eeacfee4d2d14796242`。
- [执行计划 v0.11](omniretarget_w1_real_training_update_provider_execution_plan.md)，SHA `e66cd8a9ac1d94abe8ea3f08aeadd45bec8e5d29f0f2baccfec948a4eca7cb4d`。该文件、预注册及 attempt_012 工件保留原字节。
- [首次真实执行报告](../verification/omniretarget_w1_real_training_update_audit_report.md)：原根 `derived/experiments/omniretarget_first/20261102_2704709_w1_real_training_update_audit` 的 R0 完成，P 在 diffusion 首次 forward 前以 `production diffusion alpha schedule differs` 停止；已执行 deterministic 原生 AdamW **2 次**，登记成本 **67.10273322396097 秒**，I/F 未启动。失败 attempt、未封口部分数组、账本与 reservation 不改写。
- 用户原授权是一次真实执行、不自动重试；该次执行已结束。本轮不创建真实 cohort、授权、reservation 或注册，不运行 `--real` / `--resume`。不得改 run 名重置既有消耗。

## 2. 缺陷与修正契约

预注册 §3.2 要求独立 x_t 使用生产 binary32 alpha 提升后的有效常数，原始 float64 schedule 仅用于来源与数学对账。旧实现把完整 float64 schedule 只在末尾转为 float32，遗漏原生 Diffusers 在 linspace、平方、减法及 cumprod 各节点的 float32 求值。纯合成固定配置复现有 767/1000 个位置不同、最大 11 ULP、绝对差 `1.7881393432617188e-7`；首次差异为 t=2。

修正仅落实既有有效常数语义：

1. `alpha_cumprod_reference(config)` 保持 NumPy float64 公式和父来源严格逐位门；不改父 NPZ、配置、checkpoint、生产 scheduler 或生产加噪输入。
2. `effective_alpha_reference(config)` 从固定配置独立执行 CPU Torch float32 `linspace(beta_start**0.5,beta_end**0.5,N) → square → 1-betas → cumprod`，最后提升为 float64。该 helper 不读取生产 buffer，不构造 scheduler/网络/优化器，不消耗 RNG；协议名为 `torch_cpu_float32_scaled_linear_v1`。
3. fixture、P/I/F 的独立 x_t 和真实 P input admission 共用此参考。input admission 分别绑定 raw/effective alpha 摘要、求值协议和 x_t 摘要；I/F 逐项消费绑定。
4. 真实和合成 provider、fixture 在 diffusion 首次 forward 前共用严格门：原生 buffer 必须是 float32、shape 正确、逐位等于独立有效常数。1 ULP 改写、末尾才舍入的旧结果、错 schedule/dtype/shape/NaN 均拒绝。保留真实构造截止与 unit guard。
5. runtime identity 增加 Diffusers 版本以及实际导入 DDPM/DDIM scheduler 源文件 SHA；校准、接受、metadata admission、R0/P/I/F 仍只做版本/源码摘要对账，不在真实元数据阶段运行数值探针。

既有容差、规则、零底、micro/noise 矩阵、RNG 协议、参数 roster、调用 caps、预算与七项硬零不变。P/I 的原生网络输入、prediction、梯度和更新数组应与 attempt_012 逐位一致；独立 x_t 参考及相应比较事实允许因本修正改变，必须逐项保存差异。不存在以放宽阈值让原生路径通过的步骤。

## 3. Gate B 验证与冻结

本补充计划加入 `IMPLEMENTATION_FILES`（26 文件）；`EXECUTION_PLAN` 指向本文件，原 v0.11 仍在清单中，`PREREG_DRAFT` 保持 v0.9。冻结后不得再改这些文件；结果写入独立验证记录。

先用临时合成权重、分析输入和原生 scheduler 复现旧门错误，再验证修复与拒绝反例、fixture/输入准入/P/I 参考一致性、有效常数跨阶段绑定以及 runtime 漂移拒绝。执行 provider/audit 两文件与 v0.11 所列六个相关回归文件；不要求重跑全项目套件。

使用未占用的 `calibration/attempt_013`、`synthetic/attempt_013` 和 `admission_real/attempt_013`，命令沿用 v0.11 §2 Gate B；本次所有数值运行保留 `cpu_threads=20`，不使用该历史命令示例的单线程覆盖。

校准实际重算必须 `recompute_problems=[]` 且每门 `4×observed≤1`；阈值/规则/零底逐项对比旧 contract。完整合成必须 R0/P/I/F done/sealed、264 tensor/3864 行齐全、计数精确、F 新进程重放/重推通过、无 failure/problem；必测精度/相消 gap 仍如实计为 `audit_incomplete`。逐数组和逐行对比 attempt_012，保存原因明确的差异。接受报告必须通过引用闭包消费校验。

新的真实 admission 仅按 metadata-only 规则核验文件字节/头部、父锚、资产文本闭包及新材料身份；checkpoint payload、父数组数值、真实模型/优化器、MuJoCo 编译、forward/VJP/update、cohort/注册均不执行。原真实根和授权目录以及旧冻结材料按事前摘要清单复核，确认无改写。

## 4. 真实恢复的后续条件

新 Gate B 通过仅产生新的实现/证据绑定，不能使旧授权自动适用于新代码。现有 resume 入口要求 cohort 的实现及授权身份不变，不能直接用旧授权续跑新实现。本轮不更改该协议；后续应先形成可审阅的身份迁移/重新登记方案，保留原 cohort 历史、2 次更新及全部成本，并取得针对该方案和新材料的明确授权。没有这一步，不运行真实恢复，也不声称真实参数链已经验收。

`objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false` 与既有 U0/UR/原 F/r6/旋转融合结论保持。
