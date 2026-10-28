# 文档索引

整理日期：2026-10-11（最后更新 2026-10-28）。训练目标合成审计当前权威为 `derived/audit/w1_training_objective_audit/attempt_004/`（2026-10-27 当前源码重新验收，89 行/0失败/0缺口、5 项限制不变；`attempt_003` 为历史版本证据）。**2026-10-27 用户授权后真实训练目标审计已完成 R0→P→F 一次：四窗、两固定 step-4000 checkpoint、98 目标行/14 梯度行全部 `within_tolerance`，F 两模型逐位重放通过，结论 `real_batch_covered_contract_passed`（`problems=[]`、`gaps=[]`），成本 7.5015s/900s，更新/FD/DDIM/checkpoint 写入 0；覆盖仍 `partial`、最终交互 `not_ready`。** 当前 **W1-D/E/F、失败归因、旋转融合 v2 与 W1 预测/窗口一致性诊断的已运行结论保持：Teacher raw 2/31、模型各 0/31，最终交互方法 `not_ready`**。旋转融合实验权威仍为 r10：GS 失败、GQ 3/26 配对通过/94 项退化/0 缺口，V 按条件未执行，候选不接入；第三轮审计后的只读复验含冻结重放65/65、Teacher标签13/13、V2重推39/39、重测116/116，度量和失败集合不变。S 累计1,513.866/1,800s、扣预留后可用226.134s低于r10实测252.156s，仍未生成r11。**W1 预测/窗口一致性诊断已完成 I→R→W→P→F（r6），但 2026-10-22 两轮独立审计确认同 cohort 尝试数 R 6/W 5/P 3 超过每阶段最多两次、且有一次未计费真实数据预检，r6 的 `implementation_contract_passed`/`diagnosis_complete` 不可验收；第二轮另修复 3 项残留核验缺口（撤回失败终态成本、布尔比较绕过、去噪矩阵分组宏），保守 `not_ready` 结论保留。** 统计已订正：跨窗超描述线按 demo×模型×帧为 **2,892/260**（合并模型 1,471/130），P 主统计改为 demo 等权宏平均（DDIM normalized **0.225310** vs t=999 **0.212026**）；总成本 1,801.345/7,200s 是下界。本 cohort 不再有合法正式复验，须另立新预注册方案。**2026-10-28 W1 真实训练解码、FK 几何与输出空间导数审计的 I/C 已完成并按两轮只读审阅修复 8 P1+3 P2 与 5 P1+3 P2：**新增审计器、CLI、68 项纯合成测试；固定 122 几何行、32 状态/4672 行双步长 FD；冻结校准 SHA `e70fa5b5…`（证据 `8f46fbe1…`，全部 4× observed ≤ proposal）；纯合成 R0→G→D→F（`synthetic/20261028_review11`）结论 `synthetic_contract_passed`，122+20+4672 行 0 失败/0 缺口，F 在派生子进程中重算并逐单元准入，G/D/F 逐阶段调用与提案矩阵精确一致（全链 4/8/6/4/21088/14400/18688），索引与台账同终态调用快照，三处预留 `consumed_complete=true`，全链零计数 0，成本约 11.8s；只读准入包 `attempt_012` SHA `c53e2ce4…`（运行时契约 applied、TF32 off）。第二轮修复覆盖主 provider 计数器绑定、索引/台账同快照、worker 中断结算与未结算预留、几何/FD 单元内截止、梯度明细逐字段绑定、三份 NPZ 独立 schema、缺失 NPZ 不抛异常、TF32 运行时设置与父子复核。真实数组/资产/checkpoint/cohort/预算/注册/R0→G→D→F 均未执行，§9.2 九项未勾选；通过仍 partial/not_ready，不接入生产。原SOURCE、ATTR、FUSION与旧方案/产物保持只读。

## 当前入口

| 文档 | 用途与状态 |
| --- | --- |
| [W1 真实训练解码、FK 几何与输出空间导数审计预注册草案](plans/omniretarget_w1_real_fk_geometry_derivative_audit_prereg_draft.md) | **I/C 已完成、真实阶段未授权（2026-10-28）**：新增审计器/CLI/68 项纯合成测试，固定 122 几何行、32 状态/4672 行双步长 FD、`D_decode`/`J35`/`J38` 独立公式、账本与封口协议。冻结校准 SHA `e70fa5b5…`（证据 `8f46fbe1…`，全部 4× observed ≤ proposal，近 π/clamp/退化 6D 负例保留）；纯合成 R0→G→D→F 结论 `synthetic_contract_passed`，122+20+4672 行 0 失败/0 缺口，F 派生子进程重算、逐单元准入，G/D/F 逐阶段调用与提案矩阵精确一致（全链 4/8/6/4/21088/14400/18688），索引/台账同快照，零计数为 0，合成成本约 11.8s。两轮只读审阅修复（设备、live 标量、调用点计数、阶段撤销门、F 截止/独立进程、NPZ 分离哈希与 schema、R0 闭包/消费者/schema、撤回顺序、负 clamp 边界；主 provider 计数绑定、索引/台账同快照、worker 中断结算、单元内截止、梯度明细绑定、缺失 NPZ、TF32 运行时）。真实数组/资产/checkpoint/cohort/预算/注册/R0→G→D→F 均未执行，§9.2 九项未勾选；通过仍 partial/not_ready，不接入生产。详见[验证记录](verification/omniretarget_w1_real_fk_audit_verification.md) |
| [W1 真实 FK 几何/导数审计 I/C 验证记录](verification/omniretarget_w1_real_fk_audit_verification.md) | **I/C 已完成（2026-10-28）**：实现、68 项测试、冻结校准与纯合成 R0→G→D→F 的实际证据；G 122/20 行、D 4672 行全部通过，F 在派生子进程中重读重放 `problems=[]`，G/D/F 调用精确匹配提案矩阵，索引/台账同快照，零计数 0；校准 SHA `e70fa5b5…`、证据 `8f46fbe1…`、准入包 `attempt_012` SHA `c53e2ce4…`（运行时契约 applied）。两轮审阅修复（8 P1+3 P2 与 5 P1+3 P2）与反例已登记。真实阶段未授权/未运行；覆盖 partial、最终交互 not_ready |
| [W1 真实批次目标与输出空间梯度审计报告](verification/omniretarget_w1_real_objective_audit_report.md) | **已执行（2026-10-27）**：用户授权记录 SHA `1071810f…`，准入包 `attempt_005` SHA `1aadb296…`，运行根 `20261027_b45cd7c_w1_real_objective_audit`。R0 身份/资产/四窗/checkpoint strict load 全过；P 两模型各一次真实前向+production/independent batched VJP，98/98 objective、14/14 gradient 全部 within_tolerance；F fresh load+RNG 恢复，两模型 prediction/x_t/FK/分项/production 梯度逐位重放；成本 7.5015s/900s，调用各 4/4，contact/更新/FD/DDIM/checkpoint 写入 0。覆盖范围内通过、未观察到目标偏离，不触发单因素诊断；真实 FK 几何/导数、参数梯度、累积/裁剪/AdamW/更新、选模、DDIM 与端到端质量未审计，`full_objective_contract_passed=null`/最终交互 `not_ready` |
| [W1 训练目标合成审计证据更新计划](plans/omniretarget_w1_training_objective_audit_refresh_plan.md) | **已执行（2026-10-27）**：`losses.py` observer 改动后按身份与保护清单→两模块合成测试（33/58 passed）→新 A0–A5 attempt→只读复核与新旧对比→文档指针更新完成当前版本验收。新权威 `derived/audit/w1_training_objective_audit/attempt_004/`：生产侧/独立复核/只读 A5 均 `synthetic_contract_passed`，89 行、0失败/0缺口、5 项限制不变；全部 JSON 数值与三个 NPZ 和 `attempt_003` 逐位相同（仅身份字段与 `losses.py` SHA-256 不同）；356 个旧证据文件未变（父方案仅追加 §8）。沿用原数值契约，不授权真实审计，不创建真实cohort/预算，不训练或接入生产 |
| [W1 真实训练目标审计准入包与冻结校准契约](plans/omniretarget_w1_real_objective_audit_admission_package.md) | **准入材料 + 真实执行（2026-10-27）**：契约 SHA `2904fbc4…`/证据 `49371d30…`；当前包 `attempt_005` SHA `1aadb296…`（索引 `bdedc33c…`），由执行前 `b45cd7c` 修复后重生成；`attempt_004` SHA `1cedbb15…` 保留为历史。用户授权记录 `1071810f…` 通过严格 schema/预算/次数/计数门；R0→P→F 全封口，结论 `real_batch_covered_contract_passed`，98/14 行、`problems=[]`、`gaps=[]`、成本 7.5015s/900s；更新/FD/DDIM/checkpoint 写入 0。覆盖仍 partial，最终交互 not_ready |
| [W1 真实训练目标审计预注册草案](plans/omniretarget_w1_real_objective_audit_prereg_draft.md) | **已授权并执行（2026-10-27）；§8 八项已由授权记录与真实产物勾选**：固定四窗/两 checkpoint/`t=[0,250,500,999]`/noise `20261008`、98/14 必测行、180/360/360s、每阶段 2 次、调用各 4。R0→P→F 一次通过：`problems=[]`、`gaps=[]`、`complete_evidence=true`，F 两模型逐位重放，成本 7.5015s/900s；覆盖范围内未观察到目标偏离，不触发单因素诊断。真实 FK 几何/导数、参数梯度、累积/裁剪/AdamW/更新、选模、DDIM 与端到端质量仍未审计；`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、最终交互 `not_ready`。详见[真实审计报告](verification/omniretarget_w1_real_objective_audit_report.md) |
| [W1训练拟合与目标一致性单因素诊断执行方案](plans/omniretarget_w1_training_fit_objective_audit_plan.md) | **合成审计已实现并运行（2026-10-22；2026-10-27 当前源码重新验收，权威改为 `attempt_004/`，§8）**：A0–A5 审计模块、CLI、33项测试与权威合成运行完成；A1–A4 67行+A0 22=89行、0失败/0缺口、5项已登记限制，生产侧与独立复核均 `synthetic_contract_passed`；A4 新增 `(demo,start,t)` 注册多重集与 12 类键级变体（删窗/重窗/换demo/缺t/重t/missing metric/bool与数值/NaN/Inf/缺字段/bool状态/额外t），A5 用独立实现从 raw 键级行重算多重集、宏微、逐键 `per_demo`/`per_window` 明细及关系，并以冻结定义和 raw 行独立分类反例语义；A5 另独立重放4组固定run更新与2组选模，绑定裁剪参数/形状与小量梯度容差、核验AdamW状态非负/二阶尺度/步数、重算FD与batch排列/重复关系、A2明细严格bool/数值类型、对账全部14条A2明细与完整配置，并拦截空梯度证据/清零梯度状/NaN/×2梯度/root平移/权重篡改/x_t错接/DDIM时间表或末步替换/坏结构。未加载真实 dataset/cache/checkpoint，未对 W1 训练 checkpoint/生产模型做前向、真实数据训练更新0（A3 仅随机初始化微型合成模块）；真实数据审计与单因素实验仍未注册，未分配新cohort/预算；`factor_selected=false`、`auto_execute=false`、最终交互`not_ready`。详见[合成审计验证记录](verification/omniretarget_w1_training_objective_audit_verification.md) |
| [W1 预测/窗口一致性验证记录](verification/omniretarget_w1_prediction_consistency_verification.md) | **最新已运行交付（审计后不可验收）**：r6 完成 I→R→W→P→F；130 模型窗/65 Teacher/3,512 跨窗对/520 探针；91 高层调用、403 网络前向；Det/DDIM 65/65 逐位重放；Teacher oracle 59/65（6 例旋转精度负门 max 0.02426°）。**2026-10-22 两轮审计：首轮 14 项 + 第二轮 3 项残留核验缺口已修复；同 cohort 尝试数 R 6/W 5/P 3 超限、一次未计费真实数据预检，r6 的 `diagnosis_complete=true` 不可验收；保留 `not_ready`。** 统计订正：跨窗超描述线 2,892/260（合并模型 1,471/130），P 宏平均 DDIM 0.225310 vs t=999 0.212026；完整去噪矩阵分组 FK 宏 train6/val6/known 0.061434/0.100596/0.172076（r6 原始 summary 只存 n 与窗口微平均、无宏字段；宏为第二轮修复后的只读再聚合）；总成本 1,801.345/7,200s（下界，含 r1–r5 失败尝试）。本 cohort 无合法正式复验 |
| [W1预测误差与窗口一致性诊断执行方案](plans/omniretarget_w1_prediction_consistency_plan.md) | 已执行并冻结的契约（v0.1，2026-10-21）：I→R→W→P→F；13段/26份保存窗，52常规+13尾窗，130窗误差/跨窗对账，520单元探针，F固定三段独立重放；训练0步、7,200s预算、不启动新融合/训练或生产接入。实际结果见验证记录，方案字节不改 |
| [W1 旋转融合验证记录](verification/omniretarget_w1_rotation_fusion_verification.md) | 上一轮已运行交付：`w1_rotation_path_nlerp_v2` 的完整 I/R/S/V/F，实验权威运行 r10（第一轮审计修复 11 项缺口后重跑）。S 117/117（116 measured + 1 拒收），V0/V1 各26/26逐位重放，V2位置/关节25/25逐位；GS 因 Det frame120 blend_norm=0.462 拒收与 Diffusion frame108 7.5926 rad/s 失败，V 条件未执行；GQ 修正为3/26通过、94项退化、0缺口；F 重推39/39、重测116/116、54项合成测试通过；候选 `integration_eligible=false`，最终交互 `not_ready`。第二轮审计修复 7 项（4 P1+3 P2）并以只读沙箱复验 r10：冻结重放65/65、重推39/39、重测116/116、GQ 3/26/94 与 GS 失败集合不变。第三轮审计再修复 1 P1+1 P2：S/F 要求 Teacher 行 measured、F 将 Teacher root/joints 与冻结 clip 逐位绑定（沙箱13/13），算法拒收豁免只限独立重推的 V2；路径数组形状/数值类型/有限性/四元数范数先校验，零范数异常登记为失败而不中断 F。r10 真实数据攻击（Teacher 替换/伪装拒收/零范数 path）均被拦截；因 S 阶段累计预算不足仍未生成 r11。rotation 测试68 passed、相关回归243 passed；全量891 passed+1 skipped |
| [W1 旋转融合与固定 checkpoint 验证方案](plans/omniretarget_w1_rotation_fusion_plan.md) | 已执行并冻结的契约（v0.1，2026-10-15）：唯一固定路径NLERP候选、117/155固定键集、7200s预算、训练0步与接入门；实际结果见验证记录，方案字节不改 |
| [W1 失败归因验证记录](verification/omniretarget_w1_attribution_verification.md) | 13段/78轨迹，523只读锚不变，26组保存窗对照重建、14条原产物逐位重放；确认两类局部旋转融合机制，预测侧残差仍在，标签主要瓶颈不获支持；相关159项测试通过 |
| [W1 失败归因执行方案](plans/omniretarget_w1_attribution_plan.md) | 已登记并执行的诊断契约；固定train6/val6/已知案例、两份best、seed7/DDIM25及3600s预算。全贡献SLERP仅作诊断，不自动采用或启动训练 |
| [W1 D/E 验证记录](verification/omniretarget_w1_de_verification.md) | I0–I4及七轮收口修复，全量801 passed+1 skipped；R0/R1事实复验；两模型各4k及完整val31选模；155条raw、18条val6及F独立重测/重放。动作基线证据complete，模型raw均0/31，最终交互not_ready |
| [W1 正式训练与裸输出评估方案](plans/omniretarget_w1_de_plan.md) | 已执行并冻结的契约（v0.1，2026-10-13）；绑定reviewfix8，固定I0–I4、R0/R1、每模型4k/累计12h、val31主/附表和val6/F。首页not_run是起草时快照，实际结果见验证记录 |
| [W1 弱动作监督 A3/B/C 方案](plans/omniretarget_w1_a3bc_plan.md) | 已执行的前置契约（v0.1）；A3/B/C与2026-10-12七轮审阅（30项缺口+1项文档订正全修）继续有效，reviewfix8作为D/E的只读SOURCE |
| [W1 A3/B/C 验证记录](verification/omniretarget_w1_a3bc_verification.md) | 接口实现、74 项相关测试与正式运行的实际证据；§5R7 为第七轮审阅修复权威链（运行根 `20261012_21e5784_w1_a3bc_reviewfix8/w1_reviewfix8`，就绪报告 `decisions/attempt_001`），§5R6/§5R5/§5R4/§5R3/§5R2/§5R 链与旧链保留为被取代证据 |
| [OmniRetarget 优先教师与初步训练总方案](plans/omniretarget_first_training_plan.md) | 父方案，约束来源裁决、W1 用途、训练和裸输出评估；原文字节是 A2 快照证据的一部分 |
| [A2 收口与有限恢复方案](plans/omniretarget_a2_closure_plan.md) | 已执行阶段的实验契约，W1 入口依据；原文字节保留供 A2 证据复核 |
| [A2 最新验证记录](verification/omniretarget_a2_closure_verification.md) | C0/C1/D0/G0/UR/C2 及六轮修复的实际证据；当前权威 C2 为 attempt_010 |
| [项目整体实现方案](plans/interaction_retarget_impl_plan.md) | 项目目标、资产/坐标约定、模型与最终验收设计；当前实施顺序由 OmniRetarget 总方案和 W1 方案具体化 |

当前事实：A1冻结与配对完成；原U0保持 `incomplete` 并停线，独立UR为完整 `no_quality_gain`。W1按published original/`weak_motion_only`完成A3/B/C及七轮修复，reviewfix8的55段准入、尺度正确的源箱条件、键控日程、train统计和两模型S0继续有效；D/E的R0又从当前文件重算四项前置事实并通过，未重写SOURCE。

W1-D/E正式链为 `derived/experiments/omniretarget_first/20261013_be00c50_w1_de/`。两模型各4,000真实更新、16次loss验证及4次完整val31生成，best均step4000；教师拟合宏为0.12204454/0.11134759m。E的155键、val6的18主产物和F独立核验完整，原始数值均有效；质量主比较Teacher2/31、模型各0/31，持续双手交互未证实。D累计8657.971s≤12h、E524.652s≤2h、detail+F570.840s≤1h。原F记录的归因不足保留为当时结论；新诊断详见下一段。UR标签未混入来源。原运行使用内容哈希固定的未提交实现，快照可还原，运行HEAD本身不等于实现提交。

2026-10-15归因诊断完整结束，成本416.249s≤3600s、训练步数0。已知案例frame108的旋转跳变在全贡献对照中由29.715°/99.652°降为1.471°/1.682°；但两窗相对π分支切换仍造成158.324°跳变，且26条模型轨迹的解码窗已存在穿透。train6教师拟合仍约5.8–5.9cm，模型对照仍train6/val6各0/6，部分穿透反而恶化。三个Teacher往返未过注册旋转精度门，负结果原样保留；其最坏手部FK误差约0.124mm。当时提出的下一候选是单独规定/验证旋转融合与拒收契约，该实验现已完成并得到下述负结果；剩余预测/条件/覆盖问题由新诊断方案承接。没有自动续训、扩数据、处理教师或替换融合器。

旋转融合 v2 已按[执行方案](plans/omniretarget_w1_rotation_fusion_plan.md)完成 I/R/S/V/F。2026-10-19 第一轮独立审计在 r7 上发现 8 P1+3 P2 缺口，修复后以 r10 重跑权威链（`20261019_worktree_w1_rotation_fusion_r10`）：116 条 measured product、全部 path 与 metrics 与 r7 逐位相同，唯一差异是 seam 元数据由每秒角度改为每帧角度；固定窗矩阵 117 行完整，V0/V1 各26/26逐位重放、V2位置/关节25/25逐位、Teacher往返3个既有负门保留且无新增；known case 中 Deterministic V2 在 frame120 因 blend_norm=0.462 被拒收、Diffusion V2 在 107→108 为 7.5926 rad/s（>7），GS 失败，V 按注册条件未执行；S 的 GQ 修正为 **3/26 配对通过、94项退化、0证据缺口**（89→93、90→93 与 `sub10_063/diffusion` 右脚漂移均登记，111→110 不再误报）。F 重推39/39、重测116/116、54项合成测试通过，`experiment_complete=true`、`integration_eligible=false`、最终交互 `not_ready`；r10 成本413.582s，同 cohort 累计1,963.290s≤7,200s。第二轮审计（4 P1+3 P2）修复 GS 速度重算、V0/V1 冻结产品、decoded 窗口来源、固定 cohort 根、成本数值对账、V 跳过条件与坏工件出口；新增 8 项反例，rotation 测试 62 passed。只读沙箱对 r10 复验：117 行、冻结重放65/65、重推39/39、重测116/116、GQ 3/26/94 与 GS 失败集合完全一致。由于 S 阶段累计 1,513.866/1,800s、扣除 60s 收口后可用 226.134s 小于 r10 实测 252.156s，且计划禁止阶段互借与顺序子进程改并行，本轮**未生成 r11**；没有换父目录/run 名/参数/新 cohort 重置预算。没有切换默认生产入口、没有训练或改教师。完整证据与成本见[验证记录](verification/omniretarget_w1_rotation_fusion_verification.md)。

W1 预测/窗口一致性诊断（r6）已按方案完成 130 个保存模型窗、65 Teacher 参考、3,512 个同帧正贡献跨窗对、520 个探针；Det/DDIM 65/65 逐位重放，Teacher oracle 59/65（6 例已知旋转精度负门，max 0.02426°）。**2026-10-22 独立审计发现并修复 6 项 P1+8 项 P2 核验/统计缺口，同时确认同 cohort 尝试数 R 6/W 5/P 3 超过每阶段两次、及一次未计费真实数据预检，故 r6 的 `implementation_contract_passed=true`、`complete_evidence=true`、`diagnosis_complete=true` 不可验收；保守 `not_ready` 结论保留。** 订正后的统计为：跨窗超描述线按 demo×模型×帧 **2,892** 帧/三窗 **260** 帧（合并模型 1,471/130）；P 主统计为 demo 等权宏平均，DDIM normalized **0.225310** vs t=999 **0.212026**（+0.013284），微平均 0.218241/0.206212 仅附表；F 三段重放实际 **17 高层调用/89 网络前向**（方案文本 21/93 为估计，实际未超上限）。修复后的入口按 cohort 拒绝新的 R/W/P 尝试（每阶段上限 2）；方案固定 cohort 身份，不能以 run 名/父目录/方案哈希/实现 SHA 重置，因此本 cohort 无合法正式复验，若需可验收诊断须另立正式修订的新预注册方案并取得授权。总成本 1,801.345/7,200s 是下界（含 r1–r5 的 1,083.650s 失败尝试）；S 旧预算未挪用、未生成 r11。下一候选为另立的训练拟合/目标或实现单因素诊断；不自动续训、换教师、扩数据或接入融合。

[权威 C2](../derived/experiments/omniretarget_first/20261010_a39d763_a2_closure/decisions/attempt_010/a2_closure.json) 的 SHA-256 为 `34a392f92a9cda5b8317a08d5b5b486f750c5fc83933174d36d361cb7b1d0fa2`。实际证据位于 ignored 的 derived/，迁移仓库时需按资源和验证记录恢复，不能把文档链接存在视作实验产物已携带。

## 冻结方案与进度说明

父方案、A2收口方案、W1 A3/B/C方案和W1 D/E方案首页保留编写时快照；“尚未实施”及 `not_run` 不代表当前进度。当前结果由本页、最新验证记录和权威产物共同说明：A2已收口，A3/B/C七轮修复完成，D/E正式链、F及随后归因诊断完整结束，最终交互方法未就绪。

A2的 `baseline/snapshots/attempt_006/evidence_files.json` 按路径冻结两份父方案；reviewfix8的 `freeze/preregistration.json` 冻结A3/B/C方案；D/E的 `freeze/execution_registration.json` 冻结D/E方案；失败归因的 `freeze/registration.json` 冻结归因方案和诊断入口。下列字节及直接引用路径保留：

| 冻结文档 | SHA-256 |
| --- | --- |
| `plans/omniretarget_first_training_plan.md` | `ca7cd4d8db553cac0057d1392c3238f0ca0fbfcbe4748f4c5946a3e650da2e84` |
| `plans/omniretarget_a2_closure_plan.md` | `28fd40e43af0f524e5c2c23dc776a5a9b1b00bd872d75ee90442b5f41a8b81b3` |
| `plans/omniretarget_w1_a3bc_plan.md` | `e7247e4b76d7cdfec04e288883dbdaccd868c7908108a8f2f5171839c766fc2f` |
| `plans/omniretarget_w1_de_plan.md` | `bed1e62d5c85579e96b94695cbbbb067cb79a820f18b9b692d21bda9c1ec1525` |
| `plans/omniretarget_w1_attribution_plan.md` | `4bf026c71311f6d169bcc9d8cb2c9383c601d2f0a93d22febf60a74482542be4` |
| `plans/omniretarget_w1_rotation_fusion_plan.md` | `c6aad7cc6f71ee0c7de70f331626e22647bd872c2f0200f4415cda7f0931a822` |
| `plans/omniretarget_w1_prediction_consistency_plan.md` | `b82401f8fb4ea35cc5ac986236fefa59693671042809484627e3b17cce5bfa41` |

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

归档前检查冻结哈希及当前引用，更新文档相对链接和项目导航，并核对本地文件链接与标题锚点。涉及已有证据锚点的文件先保留历史字节和引用关系，再建立新版本。当前结果登记在正式链结束后编辑文档；源资源、父方案、SOURCE与既有实验产物保持原字节。
