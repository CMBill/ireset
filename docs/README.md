# 文档索引

最后更新：2026-11-06。**当前工作是数值缺口有限收口与受控拟合诊断的方案设计；新方案尚未实施、预注册、授权或运行。** 已结束阶段集中转入历史导航，冻结原文及实验产物保持原路径、原字节。

## 当前状态

- 数据/资产/配对、运动学与独立验收基础已建立；W1有限动作基线已完成，但最终交互方法仍 `not_ready`。
- W1 D/E的两模型各完成4000步；Teacher raw 2/31、确定性/扩散raw各0/31，持续双手交互与有效加速尚未证实。
- 最新T6（2026-11-06）为 **0 direct failures、539 mandatory gaps**（350 precision/189 sensitivity/0 bound）；证据完整，但 `covered_contract_passed=false`、`audit_incomplete`。没有新生产checkpoint或质量评估。
- T6一次性授权已消费；cohort累计34次诊断更新、1347.195507/3240秒。剩余墙钟不构成新运行权限，不自动重试。
- `objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`production_integration=false`保持。

## 当前入口

| 文档 | 用途与状态 |
| --- | --- |
| [N1 最小合成验证与有限数值收口说明](verification/omniretarget_w1_n1_synthetic_closure.md) | **N1已收口（2026-11-07，四轮复阅修订后 attempt_005）**：C1 audited vs 独立节点图逐位对照、C5 三例门失败全区间内、C6 八偏差全经N0检测器检出；六类47用例全过，go=no_go→go_for_conditional_fit_diagnostics_review。封口=登记成本同盘覆盖+写后核验；返回核验先读盘后采样、return_observation.json SHA绑定证明且所有退出路径核验；撤回走覆盖成本循环。I/R/D/F仍未授权 |
| [N0 坐标级数值缺口分析记录](verification/omniretarget_w1_n0_coordinate_analysis.md) | **N0已收口（2026-11-07，四轮复阅修订后 attempt_005）**：uint32存在性/单侧可辨识位（4行）；same_input公式门+delta_measured一致性；分侧写界；roster精确集合；来源闭链（journal逐级SHA+registration三阶段严格校验+F重放双锚）；旧629全三类交付（381/223/25）；封口=登记成本覆盖封口与返回+终态时序；返回核验先读盘后采样+SHA绑定观察文件+撤回覆盖成本循环。结论=350 precision_limited+189 cross_path_sensitive+0 implementation_issue_candidate |
| [数值缺口有限收口与受控拟合诊断方案](plans/omniretarget_w1_numerics_closure_fit_diagnostic_plan.md) | **v0.4+§14–§17修订对照，N0/N1已完成（2026-11-07 attempt_005权威）、I待复阅后实施**：N0保存证据坐标级分析✓→N1最小合成验证/有限收口✓→I共享Trainer诊断实现→R新冻结/授权→D1单窗/D6六窗确定性拟合→F收口。平均门、预算和8206次/路径forward上限不放宽；真实拟合须新明确授权 |
| [最新T6执行报告](verification/omniretarget_w1_gradient_numerics_retest_report.md) | **已执行的基线，不是待执行任务**：329 failures清零，539 gaps保留；一次性授权已消费，完整目标与交互质量不通过 |
| [项目整体实现方案](plans/interaction_retarget_impl_plan.md) | 总体目标、坐标/资产约定和最终质量/端到端加速标准；其历史完成快照不能覆盖最新裁决 |
| [已完成阶段归档](archive/omniretarget_completed_20261106/README.md) | A2、W1训练/归因/融合、合成/真实目标与FK审计、更新审计/迁移、数值修复/T6的分组导航、实际结论和边界 |

## 下一步的范围

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
