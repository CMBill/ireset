# 文档索引

整理日期：2026-10-11（最后更新 2026-10-12 第四轮审阅修复）。当前推进到 **W1 A3/B/C 四轮审阅修复完成：第四轮 3 项代码缺口（1 P1+2 P2）+1 项文档订正修复后权威链重建，`ready_for_w1_d=true`（新链 `20261012_05b3f6e_w1_a3bc_reviewfix5`）；W1-D/E 正式训练尚未运行，最终可行性结论不变**。本页区分当前执行入口、保留的参考资料和历史归档。

## 当前入口

| 文档 | 用途与状态 |
| --- | --- |
| [W1 弱动作监督 A3/B/C 方案](plans/omniretarget_w1_a3bc_plan.md) | 现行执行契约（v0.1）；A3/B/C 与 2026-10-12 审阅修复轮（四轮共 24 项缺口+1 项文档订正全修）已执行，W1-D/E 契约固定但 `not_run` |
| [W1 A3/B/C 验证记录](verification/omniretarget_w1_a3bc_verification.md) | 接口实现、68 项反例与正式运行的实际证据；§5R4 为第四轮审阅修复权威链（运行根 `20261012_05b3f6e_w1_a3bc_reviewfix5/w1_reviewfix5`，就绪报告 `decisions/attempt_001`），§5R3/§5R2/§5R 链与旧链保留为被取代证据 |
| [OmniRetarget 优先教师与初步训练总方案](plans/omniretarget_first_training_plan.md) | 父方案，约束来源裁决、W1 用途、训练和裸输出评估；原文字节是 A2 快照证据的一部分 |
| [A2 收口与有限恢复方案](plans/omniretarget_a2_closure_plan.md) | 已执行阶段的实验契约，W1 入口依据；原文字节保留供 A2 证据复核 |
| [A2 最新验证记录](verification/omniretarget_a2_closure_verification.md) | C0/C1/D0/G0/UR/C2 及六轮修复的实际证据；当前权威 C2 为 attempt_010 |
| [项目整体实现方案](plans/interaction_retarget_impl_plan.md) | 项目目标、资产/坐标约定、模型与最终验收设计；当前实施顺序由 OmniRetarget 总方案和 W1 方案具体化 |

当前事实：A1 冻结与配对已完成；原 U0 保持 `incomplete` 并停线；独立 UR 六段比较为完整 `no_quality_gain`；W1 按发布 original 数据、用途 `weak_motion_only` 完成注册/审计（55 段全准入、源箱几何应用登记尺度 0.68-0.80、质量按测量存在性契约（参考帧账本×intervals 复算）完整登记）、键控缓存（代码身份含距离实现、逐文件内容哈希+消费侧复验+complete 标志）与 train 统计、两模型 S0（各 1000 步、小集整批 24 样本/步、趋势 0.394→0.014/0.429→0.011 与前三轮逐位一致、checkpoint 重建+固定 probe 重放核验（重放与记录值容差内一致，逐位相等的是 summary/工件两种序列化的记录值）+注册六段首窗与扩散 t/noise 生成器复现逐位绑定、2744.0s≤1h），工程就绪门经深度重算式核验（loss 逐行全字段存在+有限/完整 probe 日程/checkpoint 重放与对账/注册集重建绑定）在新链上 `ready_for_w1_d=true`。UR 标签不混入该来源；W1-D/E 正式训练与最终比较未运行。

[权威 C2](../derived/experiments/omniretarget_first/20261010_a39d763_a2_closure/decisions/attempt_010/a2_closure.json) 的 SHA-256 为 `34a392f92a9cda5b8317a08d5b5b486f750c5fc83933174d36d361cb7b1d0fa2`。实际证据位于 ignored 的 derived/，迁移仓库时需按资源和验证记录恢复，不能把文档链接存在视作实验产物已携带。

## 冻结方案与进度说明

两份 OmniRetarget 方案的首页状态保留其编写时快照；“收口尚未实施”等表述不代表当前进度。当前执行状态由本页、最新验证记录和其权威产物共同说明：A2 已完成收口，W1 仅起草。

A2 的 `baseline/snapshots/attempt_006/evidence_files.json` 已按文件路径记录两份方案的完整哈希。本次不覆写其内容，也保留它们直接引用的文档路径：

| 冻结文档 | SHA-256 |
| --- | --- |
| `plans/omniretarget_first_training_plan.md` | `ca7cd4d8db553cac0057d1392c3238f0ca0fbfcbe4748f4c5946a3e650da2e84` |
| `plans/omniretarget_a2_closure_plan.md` | `28fd40e43af0f524e5c2c23dc776a5a9b1b00bd872d75ee90442b5f41a8b81b3` |

后续需要修改其协议时，应明确新版本、旧版本保存位置和证据引用关系，避免把历史核验锚点当作普通进度备注覆盖。

## 原路径保留的参考资料

以下材料保留，是因为冻结父方案或现行代码/配置仍直接引用其路径。它们提供带版本范围的参考，**不是当前执行计划**；旧实验数字不自动成为 W1 的新基线或准入门。

| 参考 | 保留作用 |
| --- | --- |
| [M0 验证](verification/m0_verification.md) | 资产、布局、分组和源/目标配对基础；父方案及资源/任务配置仍引用 |
| [M1 验证](verification/m1_verification.md) | 运动学、独立验收和已知发布教师缺陷；父方案仍引用 |
| [M2 验证](verification/m2_verification.md) | 现行修复配置的测量依据与历史局限 |
| [M3 验证](verification/m3_verification.md) | 模型与历史训练参考；旧口径不与 W1 直接作配对比较 |
| [P0-fix 验证](verification/p0fix_verification.md) | 语义修复及发布数据悬浮证据；父方案仍引用 |
| [分级标签/train24 方案](plans/m2_m3_label_quality_train24_plan.md) | 现行 graded_labels_v1 配置的来源；其[实施记录](archive/verification/m2_m3_label_quality_train24_verification.md)已归档 |
| [CPU 并行方案](plans/cpu_parallel_optimization_plan.md)与[验证](verification/cpu_parallel_optimization_verification.md) | 线程/进程约定及现有测试的依据 |
| [M1 修复方案](plans/m1_fix_plan.md) | 现有接触标定工具的文档依据 |
| [P0 分层验证方案](plans/p0_relaxed_validation_plan.md) | 现有诊断工具的文档依据；不授权重开旧阈值或教师路线 |
| [train24fix3 方案](plans/train24fix3_convergence_recovery_plan.md)与[验证](verification/train24fix3_convergence_recovery_verification.md) | 父方案冻结引用的旧 D+ 终局与自研路线边界；停线结论保留 |

## 历史归档

[archive 索引与路径迁移表](archive/README.md)收纳22份已结束执行、被替代或随旧路线停线的文档，按 `plans/`、`verification/`、`review/` 保留分类和原文件名。方案—审阅—验证的相对链接已随位置更新。

历史实验产物及其哈希不改写。旧产物或消息中出现的文档原路径，通过 archive 的迁移表查找。归档不表示旧阶段目标达成，不删除失败证据，也不恢复已停止的训练或教师生产。

## 文档维护

新增执行方案写入 plans/，明确日期、范围、真实前置证据、预期集合、预算、停止条件和当前状态。实施后在 verification/ 写实际结果，关联方案和产物；方案存在不等于实施完成。

归档前检查冻结哈希及当前引用，更新文档相对链接和项目导航，并核对本地文件链接与标题锚点。涉及已有证据锚点的文件先保留历史字节和引用关系，再建立新版本。本次没有修改代码、配置、资源或实验数据。
