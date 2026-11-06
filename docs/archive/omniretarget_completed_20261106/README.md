# OmniRetarget / W1 已完成阶段归档（2026-11-06）

本次将 **39 份已结束执行、已有裁决或已被后续证据承接的方案与验证文档** 从当前工作导航归入历史参考。这里的“完成”指阶段工作结束，**不表示数值契约、动作质量或项目最终目标通过**。

## 归档方式与保护边界

- 采用**原位冻结、导航归档**：冻结正文仍在 `docs/plans/`、`docs/verification/` 原路径，不移动、不替换为跳转页、不补写历史结论。
- 原因：已有实现、测试、授权、注册、保护清单及冻结方案按路径和 SHA-256 消费这些文档。普通移动或修订会破坏证据引用。
- [归档清单](manifest.json)逐文件记录39份原文的路径、完整 SHA-256 和字节数；[历史索引](completed_stage_index.md)保留归档前的详细进度导航，并修正新位置的相对链接。
- 源数据、资产、checkpoint、`derived/` 工件、旧授权、成本/调用台账及运行根不在本次整理的修改范围内。
- 当前入口是[文档索引](../../README.md)；下一项是[数值缺口有限收口与受控拟合诊断方案](../../plans/omniretarget_w1_numerics_closure_fit_diagnostic_plan.md)，**仅方案，未实施、未授权、未运行**。

## 阶段归档与实际裁决

| 已结束阶段 | 方案 / 记录 | 实际裁决及后续边界 |
| --- | --- | --- |
| 教师来源比较、A2收口 | [总方案](../../plans/omniretarget_first_training_plan.md)、[A2方案](../../plans/omniretarget_a2_closure_plan.md)、[验证](../../verification/omniretarget_a2_closure_verification.md) | 原U0为 `incomplete`；独立UR为完整 `no_quality_gain`；不混合来源，不恢复旧额度 |
| W1 A3/B/C | [方案](../../plans/omniretarget_w1_a3bc_plan.md)、[验证](../../verification/omniretarget_w1_a3bc_verification.md) | reviewfix8完成55段准入、尺度正确的源箱条件、缓存/训练统计与S0，作为后续只读SOURCE |
| W1 D/E/F动作基线 | [方案](../../plans/omniretarget_w1_de_plan.md)、[验证](../../verification/omniretarget_w1_de_verification.md) | 两模型各4000步、155条raw评估完整；Teacher 2/31、两模型各0/31；`motion_baseline=complete`，交互 `not_ready` |
| 失败归因 | [方案](../../plans/omniretarget_w1_attribution_plan.md)、[验证](../../verification/omniretarget_w1_attribution_verification.md) | 两类旋转融合机制已确认，预测残差仍显著；不支持教师是主要瓶颈的独占归因 |
| 旋转融合v2 | [方案](../../plans/omniretarget_w1_rotation_fusion_plan.md)、[验证](../../verification/omniretarget_w1_rotation_fusion_verification.md) | r10的GS失败、GQ 3/26通过/94项退化；候选不接入，没有r11 |
| 预测/窗口一致性诊断 | [方案](../../plans/omniretarget_w1_prediction_consistency_plan.md)、[验证](../../verification/omniretarget_w1_prediction_consistency_verification.md) | r6虽已运行，但尝试超限及未计费预检使完成声明不可验收；旧cohort无合法正式复验 |
| 训练目标合成审计及刷新 | [初始方案](../../plans/omniretarget_w1_training_fit_objective_audit_plan.md)、[刷新方案](../../plans/omniretarget_w1_training_objective_audit_refresh_plan.md)、[验证](../../verification/omniretarget_w1_training_objective_audit_verification.md) | attempt_004为89行/0失败/0缺口，5项限制保留；不外推为真实训练或完整目标通过 |
| 真实批次目标/输出梯度审计 | [方案](../../plans/omniretarget_w1_real_objective_audit_prereg_draft.md)、[准入说明](../../plans/omniretarget_w1_real_objective_audit_admission_package.md)、[报告](../../verification/omniretarget_w1_real_objective_audit_report.md)、[验证](../../verification/omniretarget_w1_real_objective_audit_verification.md) | 固定四窗98目标/14梯度行通过，F重放通过；仅覆盖范围内有效 |
| 真实FK几何/输出导数审计 | [方案](../../plans/omniretarget_w1_real_fk_geometry_derivative_audit_prereg_draft.md)、[报告](../../verification/omniretarget_w1_real_fk_audit_report.md)、[验证](../../verification/omniretarget_w1_real_fk_audit_verification.md) | 122主表行、32状态/4672 FD行通过；不是网络参数或整段质量验收 |
| 真实参数/更新审计首次执行 | [预注册](../../plans/omniretarget_w1_real_training_update_audit_prereg_draft.md)、[provider方案](../../plans/omniretarget_w1_real_training_update_provider_execution_plan.md)、[报告](../../verification/omniretarget_w1_real_training_update_audit_report.md)、[验证](../../verification/omniretarget_w1_real_training_update_audit_verification.md) | 2026-11-02实际2次更新后alpha门拒绝，P incomplete、I/F未启动；失败历史保持 |
| alpha修复与绑定迁移 | [alpha方案](../../plans/omniretarget_w1_real_training_update_alpha_fix_plan.md)、[迁移方案](../../plans/omniretarget_w1_real_training_update_binding_migration_plan.md)、[报告](../../verification/omniretarget_w1_real_training_update_binding_migration_report.md)、[验证](../../verification/omniretarget_w1_real_training_update_binding_migration_verification.md) | 一次迁移真实链证据闭合，329 failures/629 gaps，`audit_incomplete`；该次授权已消费 |
| 保存输入归因与局部验证 | [只读归因](../../verification/omniretarget_w1_real_training_update_failure_attribution.md)、[局部分项验证](../../verification/omniretarget_w1_real_training_update_local_gradient_verification.md) | 为稳定四元数转换、结构零key bias和后续参数级复验提供范围限定的证据，不代替整链验收 |
| 梯度数值稳定性修复 | [方案](../../plans/omniretarget_w1_gradient_numerics_repair_plan.md)、[验证](../../verification/omniretarget_w1_gradient_numerics_repair_verification.md) | N1/N2/N3及复审修复已完成；旧30文件身份属于当时快照，真实复验由下行承接 |
| 数值修复后T6复验 | [方案](../../plans/omniretarget_w1_gradient_numerics_retest_plan.md)、[授权请求历史](../../verification/omniretarget_w1_gradient_numerics_retest_authorization_request.md)、[执行报告](../../verification/omniretarget_w1_gradient_numerics_retest_report.md)、[验证](../../verification/omniretarget_w1_gradient_numerics_retest_verification.md) | 2026-11-06一次执行：0 failures、539 mandatory gaps，证据完整但covered contract未过；累计34次更新/1347.195507秒；授权已消费 |

## 统一阅读规则

1. 最新更新链基线是T6：`complete_evidence=true`、`covered_contract_passed=false`、`audit_incomplete`；旧329/629仍是旧M6的事实，不改写旧报告。
2. `objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false`保持。
3. T6验证记录第5行、§2等仍含执行前快照，§6及独立执行报告记录真实执行后状态；此处说明时序，不修改原文。原页头的日期也不应取代实际执行日期。
4. 旧预测r6、融合r10及自研修复路线的失败/停止裁决不因归档而恢复资格。数值近π修复不等于长序列旋转路径修复。
5. 历史资料存在于导航不等于大体积工件始终可用；恢复实验前按其清单核验实物。`derived/`被git忽略，仓库本身不携带全部实验数据。

## 本次整理的检查范围

只检查文档链接、39份原文哈希、T6登记的32项实现字节和git改动范围。没有运行pytest、模型、FK、反向、AdamW、训练、重放或新的真实阶段，也没有生成新实验授权或注册。
