# 文档索引

最后更新：2026-11-20。**[固定六窗联合拟合](verification/omniretarget_w1_d6_joint_fit_result.md)已完成唯一一次真实 D6 → F 并通过证据验收：`fit_insufficient`（0/6 窗四门全过），F 两端逐位重放通过，一次性授权已消费。** [固定单窗学习率日程对照](verification/omniretarget_w1_d1_lr_schedule_diagnostic.md#8-真实-b-执行与最终比较2026-11-18)的 B 仍为四项平均门全过，仅限固定单窗。冻结原文和历史工件保持原路径、原字节。

## 当前状态

- **2026-11-20 六窗误差复盘完成。** 40 项 CPU 合成验收通过，唯一归档复盘独立核对 606 窗行、24 条曲线和 101 对共同窗。四指标最早持续差距均始于 step10，六窗的固定 FK 三点组均无持续集中，裁决 `attribution_insufficient`。[逐窗误差复盘报告](verification/omniretarget_w1_d6_window_error_review.md)
- **2026-11-20 六窗真实诊断收口**：D6 1000 更新、101 批 probe/606 窗行；F 端点逐位重放、阶段和外层实际返回覆盖均通过。整链至最后见证同步回读 1638.648367 s ≤ 登记 1642.608866 s ≤ 4200 s。详见[逐窗裁决及返回证据](verification/omniretarget_w1_d6_joint_fit_result.md)。
- **2026-11-19 六窗授权前复核**：补齐 §6.3 的逐构造/转移/加载和终点证据硬门；双身份绑定与未签草案已独立回读。定向回归共 235 项通过，见[授权前记录](verification/omniretarget_w1_d6_joint_fit_preauthorization.md)。
- **2026-11-19 六窗外层计时修复**：父进程覆盖启动前起点、实际退出和清理、定额日志、最后同步与回读；缺见证或迟返仅判 `incomplete`。新增 48 项、R6/适配回归 34 项、I6 回归 90 项通过，见[修复验收记录](verification/omniretarget_w1_d6_joint_fit_outer.md)。
- **2026-11-19 六窗 R6**：只读核对六个训练首窗、55/55 历史迁移证明、原 `train24` 根统计、资产和 T6 运行时。实际退出、清理及最终同步回读独立核验，3.958656 s ≤ 登记 8.307731 s ≤ 120 s。R6 尝试已消费；真实链仍须新的 §8.3 一次性明确授权，范围仅 `['D6','F']`。
- **2026-11-19 六窗 I6 修复及复阅**：初版六项问题及复阅发现的文件同步迟返问题均已修复，90 项 I6 测试通过。两条独立 CPU 链每条 4 次原生更新、5 次整批 probe、30 条窗行，F 两端逐位重放、实际返回及末次存档覆盖均经只读复核。正控 `fit_pass`（6/6），负控四项宏平均达标但一窗未过，裁决为 `fit_insufficient`（5/6）。详见[I6 验证记录](verification/omniretarget_w1_d6_joint_fit_i6.md)。
- **2026-11-19 单窗LR工程复阅修复**：优化器钩子导致LR取样不实、实际观察器退出/存档覆盖、前缀收据写后截止、工件写前空间门四项已修复。定向73＋旧拟合80＋准入/N0/N1 91项通过；真实 B 的 `single_window_pass` 保留，旧材料/授权不复用，见[修复记录](verification/omniretarget_w1_d1_lr_review_hardening.md)。未重跑真实阶段或追加D6。
- **2026-11-18 新授权真实链验收与保存曲线分析**：`attempt_002` worker/监督器均 exit0，外层 `candidate_complete`/problems=[]，F `fit_insufficient`；根旋转 0.023780910989rad、FK 0.016761962649m 未过门。新旧 101/101 probe NPZ 逐字节相同，无四门同时通过的 probe；278 保护文件零漂移。见验证记录 §34、方案 §44。后续先冻结单因素优化/目标诊断方案，学习率日程仅为候选，不自动启动新训练。
- **2026-11-17 真实拟合负结果与外层预算修复**：`derived/experiments/omniretarget_first/attempt_001` 的 D1 1000 步/101 probes 完整，旋转与 FK 平均门未过，F 逐位重放后为 `fit_insufficient`；监督器以 `budget_binding_mismatch` 裁决 incomplete。修复只补顶层完整预算，已消费阶段及旧工件保持原样，见验证记录 §32–§33。
- **2026-11-16 I 当前实现合成验收通过**：`attempt_019` 的微型 D1→D6→F 与 **115/115** 反例通过；监督器实际返回后的外层采样文件写盘 **1393.687296s ≤ 登记 1396.483622s ≤ 3300s**（以更早的观察器 t0 计）。
- **2026-11-16 撤回成本复审修复**：`77938bf` 对撤回和失败成本证明写盘做有界重算；原封口成本单列，尚未获外层返回证明时不冒称覆盖已证明，详见验证记录 §27。
- **2026-11-16 I 阶段审阅修复**：阶段名、阶段报告正文与哈希绑定、stdout flush 计费问题分别修复；`attempt_018` 的 D1→D6→F 数值链和 **115/115** 反例事实保留，但其 **1454.429s** 观察早于最终 stdout flush，严格完整验收不能由该次证明。
- 数据/资产/配对、运动学与独立验收基础已建立；W1有限动作基线已完成，但最终交互方法仍 `not_ready`。
- W1 D/E的两模型各完成4000步；Teacher raw 2/31、确定性/扩散raw各0/31，持续双手交互与有效加速尚未证实。
- 最新T6（2026-11-06）为 **0 direct failures、539 mandatory gaps**（350 precision/189 sensitivity/0 bound）；证据完整，但 `covered_contract_passed=false`、`audit_incomplete`。没有新生产checkpoint或质量评估。
- T6一次性授权已消费；cohort累计34次诊断更新、1347.195507/3240秒。剩余墙钟不构成新运行权限，不自动重试。
- N0/N1（2026-11-07，attempt_006 权威）：350 precision_limited + 189 cross_path_sensitive + 0 implementation_issue_candidate；同输入公式逐位验证、六类47合成用例全过。
- **I 阶段历史（2026-11-11）**：第十六轮 4 P1+1 P2 与教师目标转换截止回归已修复；`attempt_017` 的数值链通过，但该次缺少 uv 启动前起点与实际 wait 的成本覆盖证明。`attempt_016` 为旧代码 `199f3fd` 的历史结果。两者保留原状，新验收见 `attempt_018`。
- `objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`production_integration=false`保持。

## 当前入口

| 文档 | 用途与状态 |
| --- | --- |
| [固定六窗联合拟合诊断方案](plans/omniretarget_w1_d6_joint_fit_diagnostic_plan.md) | **v0.2 冻结执行前快照**；完成情况以真实结果报告为准。 |
| [固定六窗真实诊断结果](verification/omniretarget_w1_d6_joint_fit_result.md) | **已收口**：`fit_insufficient`，证据完整、端点重放通过；不自动续行。 |
| [固定六窗逐窗误差复盘方案](plans/omniretarget_w1_d6_window_error_review_plan.md) | **冻结的 v0.3**，含两侧历史准入与组级候选判据。 |
| [固定六窗逐窗误差复盘报告](verification/omniretarget_w1_d6_window_error_review.md) | **已验收**，共同窗 `comparable`，组级调权候选 `null`。 |
| [固定六窗 I6 验证记录](verification/omniretarget_w1_d6_joint_fit_i6.md) | **纯 CPU 合成复阅通过**：载荷、日程、逐窗门、端点重放与实际退出覆盖。 |
| [固定六窗 R6 与旧身份](verification/omniretarget_w1_d6_joint_fit_r6.md) | **唯一正式 R6 已消费**：历史封存和未签草案不变，不绑定修订后的代码。 |
| [固定六窗外层计时修复](verification/omniretarget_w1_d6_joint_fit_outer.md) | **纯合成验收通过**：启动、返回、最后回读及独立见证；不放行真实运行。 |
| [固定六窗 §8.3 授权前复核](verification/omniretarget_w1_d6_joint_fit_preauthorization.md) | **授权前历史记录**：最终材料、未签草案、预算和历史核验；正式授权见真实结果报告。 |
| [固定单窗学习率日程对照方案](plans/omniretarget_w1_d1_lr_schedule_diagnostic_plan.md) | **已完成**：以真实 `attempt_002` 为对照，第 501 步起学习率减半；一次 D1′/F′ 核验通过，B 为 `single_window_pass`，见[验证记录 §8](verification/omniretarget_w1_d1_lr_schedule_diagnostic.md#8-真实-b-执行与最终比较2026-11-18)。 |
| [I 阶段小集拟合诊断器验证记录](verification/omniretarget_w1_smallset_fit_diagnostic.md) | **合成验收、缓存迁移与 R 通过；预算修复后的新授权真实链见 §34**：外层 candidate_complete，D1/F 为 fit_insufficient，D6 合法跳过；保存曲线分析支持进入固定同窗单因素诊断方案阶段 |
| [N1 最小合成验证与有限数值收口说明](verification/omniretarget_w1_n1_synthetic_closure.md) | **N1已收口（2026-11-07，五轮复阅修订后 attempt_006）**：C1 audited vs 独立节点图逐位对照、C5 三例门失败全区间内、C6 八偏差全经N0检测器检出；六类47用例全过，go=no_go→go_for_conditional_fit_diagnostics_review。封口=登记成本同盘覆盖+写后核验；返回核验先读盘后采样、return_observation.json SHA绑定证明且所有退出路径核验；撤回走覆盖成本循环 |
| [N0 坐标级数值缺口分析记录](verification/omniretarget_w1_n0_coordinate_analysis.md) | **N0已收口（2026-11-07，五轮复阅修订后 attempt_006）**：uint32存在性/单侧可辨识位（4行）；same_input公式门+delta_measured一致性；分侧写界；roster精确集合；来源闭链（journal逐级SHA+registration三阶段严格校验+F重放双锚）；旧629全三类交付（381/223/25）；封口=登记成本覆盖封口与返回+终态时序；返回核验先读盘后采样+SHA绑定观察文件+撤回覆盖成本循环。结论=350 precision_limited+189 cross_path_sensitive+0 implementation_issue_candidate |
| [数值缺口有限收口与受控拟合诊断方案](plans/omniretarget_w1_numerics_closure_fit_diagnostic_plan.md) | **v0.4+§14–§44 修订与运行记录**：N0/N1、I、缓存迁移、R 和授权真实链证据验收已完成；§44 为真实 fit_insufficient 及后续 §9 一因素诊断方案分支，未授权新运行 |
| [最新T6执行报告](verification/omniretarget_w1_gradient_numerics_retest_report.md) | **已执行的基线，不是待执行任务**：329 failures清零，539 gaps保留；一次性授权已消费，完整目标与交互质量不通过 |
| [项目整体实现方案](plans/interaction_retarget_impl_plan.md) | 总体目标、坐标/资产约定和最终质量/端到端加速标准；其历史完成快照不能覆盖最新裁决 |
| [已完成阶段归档](archive/omniretarget_completed_20261106/README.md) | A2、W1训练/归因/融合、合成/真实目标与FK审计、更新审计/迁移、数值修复/T6的分组导航、实际结论和边界 |

## 下一步的范围

**2026-11-20 六窗范围已结束：**唯一 `["D6","F"]` 授权已消费。保留负结果与保存曲线，不重跑 R6 或真实链，不追加 D1、六窗恒定学习率对照或其他实验。

六窗归档复盘已按 v0.3 完成，组级调权判据未满足。任何新真实数值诊断须另立范围、预算并取得明确授权。[逐窗误差复盘报告](verification/omniretarget_w1_d6_window_error_review.md)

学习率对照已按[冻结方案](plans/omniretarget_w1_d1_lr_schedule_diagnostic_plan.md)完成，前 500 步与 A 的共同前缀核验通过，B 在 step1000 四门达标。结论只覆盖固定窗和固定 seed，不代表泛化或生产资格；不自动追加 D6、续训或下一实验。

先用已有保存量区分不可辨识、相消敏感和实现问题，不机械追求539项全部改判通过。拟合诊断先回答新实现能否拟合固定训练窗，不直接续训旧checkpoint、扩数据、改变损失或重启旧修复路线。

真实D1/D6只在新数据/代码/资源/预算/精确调用矩阵冻结并获明确授权后启动；每阶段固定1000步、101次probe，只用最终last判平均门。D阶段仅发布local_complete_evidence/local_mean_fit_passed；D1按本地落盘核验与handoff_ready交接，不等待或提前执行F。F再重算保存指标并逐位重放initial/last，独立发布最终验收字段。根/关节指标从原始x0规范CPU解码并核对关联，中间FK只核对保存点的度量自洽，状态关联为not_evaluated，不冒称完整中间物理链已验证。运行时沿用已审计CUDA/float32硬要求，probe固定整批D1=1/D6=6，每路径总forward≤8206；不能拆批或额外预测。峰值门保持 `spike_gate_status="not_evaluated"`；平均拟合通过不排除局部尖峰，也不代表泛化、严格碰撞、持续抓握或生产接入。长序列融合、质量干预、扩散对照和最终加速分别作为后续候选，不自动执行。

## 已完成阶段归档

2026-11-06对 **39份OmniRetarget/W1方案和验证文档** 采用“原位冻结、导航归档”：

- [阶段归档与实际裁决](archive/omniretarget_completed_20261106/README.md)：区分阶段结束、负结果、不可验收及覆盖范围内通过。
- [详细历史索引](archive/omniretarget_completed_20261106/completed_stage_index.md)：保留归档前完整进度导航；其中“当前”“待实施”按原记录时点理解。
- [逐文件哈希清单](archive/omniretarget_completed_20261106/manifest.json)：39份原文的路径、完整SHA-256和字节数。
- [更早的物理归档与路径迁移表](archive/README.md)：2026-10-11迁移的22份旧M2/M3、P0和教师修复文档。

**归档不等于通过，不删除失败证据，不重置预算或授权。** U0 incomplete、UR no_quality_gain、旧修复R矩阵未过门、融合候选不接入、预测r6不可验收及T6 audit_incomplete均保留。

## 冻结方案与进度说明

完成阶段原文因已被代码、测试、注册、授权或保护清单按路径/SHA消费，保留在原 `plans/`、`verification/` 位置；不能当作普通整理任务搬走或写入跳转页。冻结首页的“未实施”可能是执行前快照，应结合其最终报告阅读。

例如T6验证记录的执行前段落与§6执行结果属于不同时间点；最新状态以[独立T6报告](verification/omniretarget_w1_gradient_numerics_retest_report.md)及绑定的终局工件为准。本次不修订其历史正文或旧契约。

运行HEAD不必等于实际实现身份；真实执行以绑定的文件字节和快照为准。新文档不回填到旧实现闭包，不宣称旧契约被重新冻结。历史实验位于ignored的 `derived/`，迁移仓库时须恢复实际数据，链接存在不能证明产物已携带。

## 原路径保留的参考资料

以下是基础设计或历史机制参考，**不是当前启动指令**；旧数字不自动成为新基线或新准入门。

| 参考 | 保留用途 |
| --- | --- |
| [M0验证](verification/m0_verification.md)、[M1验证](verification/m1_verification.md) | 资产、布局、配对、运动学、接触和独立验收基础 |
| [M2验证](verification/m2_verification.md)、[M3验证](verification/m3_verification.md) | 修复/生成机制与历史局限，不将局部成功外推为最终完成 |
| [P0-fix验证](verification/p0fix_verification.md) | 接触/插值/终检语义修复及教师缺陷背景 |
| [标签分级/train24方案](plans/m2_m3_label_quality_train24_plan.md) | 现行标签协议来源；[已归档验证](archive/verification/m2_m3_label_quality_train24_verification.md) |
| [CPU并行方案](plans/cpu_parallel_optimization_plan.md)、[验证](verification/cpu_parallel_optimization_verification.md) | 线程/进程约定及已有工程性能证据 |
| [M1修复方案](plans/m1_fix_plan.md)、[P0分层方案](plans/p0_relaxed_validation_plan.md) | 现行代码或冻结父方案仍引用的基础资料，不授权重开阈值 |
| [train24fix3方案](plans/train24fix3_convergence_recovery_plan.md)、[验证](verification/train24fix3_convergence_recovery_verification.md) | 旧自研修复终局与停线边界；不与新弱动作拟合诊断混淆 |

## 文档维护

新方案与实际验证分开记录：方案存在不表示已实施，实验结束不表示质量通过。修改协议需新版本、事前门槛、明确预算/调用/停止条件及新授权；保护旧字节和引用关系后才建立新材料。

归档批次仅涉及文档导航、归档清单和新方案；v0.2/v0.3/v0.4仅修订方案与当前索引，没有修改实现/配置、旧冻结方案/验证正文、源资源或实验产物，也没有运行pytest或新数值链。
