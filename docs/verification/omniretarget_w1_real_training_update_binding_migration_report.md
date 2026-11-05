# W1 绑定迁移一次真实恢复执行报告

日期：2026-11-04。依据：[绑定迁移方案 v0.13 修订2](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_real_training_update_binding_migration_plan.md)、[最终 Gate B/M5 记录 §8](/home/c112/codes/bill/ireset/docs/verification/omniretarget_w1_real_training_update_binding_migration_verification.md)。

本轮 M6 已按新明确授权执行一次 R0_002→P_002→I_001→F_001 并返回，未自动重试。四阶段均 done/complete，提交、调用结算及外层返回证明闭合，`complete_evidence=true`。**数值契约未通过：329 项 failure、629 项 gap，`covered_contract_passed=false`；程序原始结论为 `audit_incomplete` / exit 3。** 本报告保留全部 failure，不将证据完整或 F 重放一致解释为数值通过。

## 1. 授权与冻结身份

用户明确指令：

> 同意本轮 attempt_005 授权草案的十项 checklist，绑定实现身份 `0cc0622c…`，授权按草案固定预算执行一次真实 R0_002→P_002→I_001→F_001，最多新增16次诊断更新、cohort上限18次，不自动重试。

[正式 v2 授权](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/binding_migration_authorization/attempt_005/authorization.json) SHA `673d750f84029d9fe5c0df80a4269bf7abd03f736c45ca2fb739038126ecd76b`，十项 checklist 全 true。原[草案](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/binding_migration_authorization/attempt_005/authorization_draft.json) SHA `d83d4064aead0fe97107e72b14ede49de04a060ba7077b44d81b31941125a4a4` 保留原字节，仍为未授权模板；用户原文另存于同目录 `user_authorization_evidence.json`。

| 绑定 | SHA-256 |
| --- | --- |
| 28 文件实现身份 | `0cc0622c86829b3e4bcbdefc1a3c1edb552ec02b31294dfd98bb70fd563205d8` |
| Gate B v2 attempt_005 | `c59e719ab53853d770e688f1dcd07109602dcd476c6e74417dbb437acfc5ef3c` |
| real admission attempt_019/attempt_001 | `62d1965f48b7b90645dad24f089e58eaec0aad1fa0b1285380a6445f9f4eead1` |
| snapshot attempt_006 | `42ffd63f1dca3b695f6ffed523e22fc70186138c4b177f37c5645993f36a08b6` |
| binding spec attempt_006 | `5b0feaa5f3151a5560f42179d219315017ea6d86cfb58c594b76f7e5b5528d78` |
| migration admission attempt_004/attempt_001 | `b900f8d21d20ee1cfd3941e1b2cea597d3dae932445de579627bee0cac213d18` |
| legacy head | `b499a6ce7e5ef8fac06821cc063f27bd109374d9673791551fe92ad56302a424` |

执行前元数据核对通过：实现文件一致、94 项保护清单 drift 0、原根仍 10 文件 / 466,350,106 bytes、启动日志 known/0 records、历史核验无问题；磁盘空闲约 91.27 GiB。记录见同授权目录的 `authorization_preflight.json`（SHA `514108c0…`）。未修改实现、方案、容差、校准、接受包或 M5 绑定文件。

## 2. 阶段与成本

运行根：[20261102_2704709_w1_real_training_update_audit](/home/c112/codes/bill/ireset/derived/experiments/omniretarget_first/20261102_2704709_w1_real_training_update_audit)。唯一 invocation 为 `w1-migration-20261104-002`，使用既有 cohort 的 `generation_002`。

| 阶段 | attempt / execution_seq | 状态 | 新增诊断更新 | 本轮登记成本（秒） | 启动前该阶段余额（秒） |
| --- | --- | --- | ---: | ---: | ---: |
| R0 | attempt_002 / 3 | done / complete | 0 | 124.442447620 | 181.400094766 |
| P | attempt_002 / 4 | done / complete | 4 | 28.024673619 | 891.497172010 |
| I | attempt_001 / 5 | done / complete | 4 | 102.326411103 | 900 |
| F | attempt_001 / 6 | done / complete | 8 | 245.949143446 | 1200 |
| 本轮合计 | 一次入口 | audit_incomplete | 16 | 500.742675788 | 3172.897266776 |

历史成本 `67.10273322396097s` 与历史 2 次更新保留；cohort 登记总成本为 **567.845409012 / 3240 秒**。外层入口实际 wait 墙钟为 **496.950582235 秒**（19:42:04.936593 至 19:50:21.887178，+08:00），另列于[退出观察](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/binding_migration_authorization/attempt_005/execution_exit.json)，不替代包含保守 seal-tail 的登记成本。F 的 `seal_tail_reserve_s=3.0`。余额仅为账面事实，不恢复本次一次使用权；本次授权新增更新额度已用完。

## 3. 提交与实际返回

| 阶段 | commit SHA-256 | 已封口报告 SHA-256 |
| --- | --- | --- |
| R0 | `0e8175186a97937f91cf0f4f60a3d41ae563bef180ca0705839b98227433c8a7` | admission `fc500e696d26e4a6c1f1cfc521fdbf9c2841b5caa23efe936d2ae9733fd51686` |
| P | `83412cf49f4041cc02584529d82b40557a67de31f529daf567b0ba748765cc32` | `0f2c16c8ca76b39eed37445135fe8e6221271bf944249f4fd37f21233ffa8531` |
| I | `371a01dd35336ca8ae0b8a7fe107f79ddd99b34bb18663f33f5231de8a2df3f5` | `4cc236e62f43950ac1516399fddea88981dfb07c25409970f067b5e60b13dc1c` |
| F | `c650c24bf3bb218055b3ed3bc54e33046c69896b642fd3de442e199ccbf2727d` | `ede2254f9212a34a50f1478e85bc98dc199db56dd719efff208d2886ad0d9799` |

新 R0 registration SHA 为 `38824f200887260fdffcf8b34cec774ac378ee8f0e90363c4b0254e8a2dfd588`；P input admission SHA 为 `0c5fb7f0c3f145eb1c859518d002fd467dcb3381b911540762248d189906a115`。

迁移 worker PID/PPID 为 `870249/868975`，实际退出码 3；F 独立 worker PID/PPID 为 **876939/870249**，exit 0 / `ok`。F `worker_snapshot_ok=true`、`verification_problems=[]`。固定启动日志内已生成 claim、registered、四份 stage_closed 及写后证明、terminal 候选及写后证明、实际外层 [return observation](/home/c112/codes/bill/ireset/derived/experiments/omniretarget_first/20261102_2704709_w1_real_training_update_audit/launch_journal/w1-migration-20261104-002/return_observation.json)（SHA `32599cf7f84112f031ff648866e839568d3b3605f97b0fb52bc70f099d8aff53`）。其 `coverage_ok=true`，外层扫描 `launch_journal.known=true/problems=[]`，返回观察的消费问题为空。

新链选中 R0_002/P_002/I_001/F_001，`selected_chain_training_updates=16`；`training_updates_scope=cohort`，`training_updates=cohort_training_updates=training_updates_known_lower_bound=18`，`call_accounting_known=true`。旧 R0_001/P_001 两行保留，旧 P_001 仍为 incomplete，旧部分数组未纳入新选中链。执行器未报告旧前缀或身份漂移问题。

## 4. 真实数值结果

[终检报告](/home/c112/codes/bill/ireset/derived/experiments/omniretarget_first/20261102_2704709_w1_real_training_update_audit/finalize/attempt_001/training_update_final.json)记录 264 tensor / 3864 行，`complete_evidence=true`、`covered_contract_passed=false`、`verification_problems=[]`。原始程序结论保持 `audit_incomplete`；以下 failure 与 gap 同时保留，不合并为“仅有精度 gap”。

| failure 前缀 | 数量 |
| --- | ---: |
| i_micro_param | 202 |
| i_accumulated | 101 |
| i_clipped | 25 |
| i_output_cotangent | 1 |
| 合计 | 329 |

| gap 类型 | 数量 |
| --- | ---: |
| parameter_delta_precision_gap | 381 |
| cross_path_sensitivity_gap | 223 |
| cross_path_bound_unverifiable | 25 |
| 合计 | 629 |

例如 failure 中包含 `deterministic:u1:m0:model.net.blocks.0.cross_attn.k.bias` 的微批参数梯度比较。F 重放保留相同 failure/gap 集合，证明结果可重放；这些失败的原因尚未归因，不据此断言是生产实现、独立参考或数值可辨识性中的哪一项。

## 5. 调用、资源与结论边界

| 计数 | 新选中链 | cohort（含旧前缀） |
| --- | ---: | ---: |
| checkpoint loads / model constructions / conditioner constructions | 各 8 | 各 10 |
| conditioner forwards / backbone forwards / micro units | 各 64 | 各 72 |
| native AdamW steps | 8 | 10 |
| explicit reference updates | 8 | 8 |
| diagnostic parameter update events | 16 | 18 |
| reference mj_forward | 15840 | 15840 |
| reference mj_jacSite | 115200 | 115200 |
| scene setup mj_forward | 2 | 3 |

完整 24 键 selected/stage/cohort 调用账见[执行结果摘录](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/binding_migration_authorization/attempt_005/execution_observation.json)（SHA `97b5a94b397f2b212d41099745d2330f195bf851af41ae1ffd53956344514aee`）。该摘录只复制已生成 JSON、台账及实际 wait 事实，不是新 stage、commit、返回收据或额外成功核验；本报告中的工件 SHA 来自执行器封口记录，未事后重新哈希大型工件或重算数值链。

运行根当前为 83 文件 / 7,835,408,541 bytes，约 7.2973 GiB（含原历史工件）。资源上限保持 GPU 22 GiB、host 32 GiB、工件 16 GiB、启动空闲盘至少 20 GiB；执行器未报告资源越界。

七项整数零声明全部为 0：parameter/state FD、DDIM、validation/selection、生产 checkpoint 写入和 contact 分支。保持 `objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false` 及旧质量结论。

本轮只新增正式授权、一次真实执行、结果存档和文档记录；冻结的 28 个实现文件及 Gate B/M5 文件未改写，全项目测试未重跑。后续如需处理 329 项 failure，应先另行审阅既有证据的归因范围，不自动新增更新、放宽阈值、扩额、重试或接入生产。

2026-11-04 后续用户授权的既有工件只读归因已完成，见[失败归因报告](/home/c112/codes/bill/ireset/docs/verification/omniretarget_w1_real_training_update_failure_attribution.md)：156 项 key bias 路径与 173 项 diffusion u2 局部偏差已逐项索引，具体 quaternion 分项因果验证尚待执行。该分析没有产生新模型/梯度/FK/更新，也不改写本报告的原始数值结论或恢复已用完的真实授权。

此后用户“继续”指令下完成[保存输入局部分项验证](/home/c112/codes/bill/ireset/docs/verification/omniretarget_w1_real_training_update_local_gradient_verification.md)：5个输出超差主要来自trace-only quaternion FK，归一化库候选已通过局部对照；key bias结构零路径和旧AdamW状态约束也已验证。该后续轮单列57次autograd/1444个运动学状态、10.3349s，训练网络/checkpoint/训练更新仍0；[最小修复方案](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_gradient_numerics_repair_plan.md)尚未实施，原真实授权、工件与329/629结论保持。
