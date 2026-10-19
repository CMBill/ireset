# W1 旋转融合修复与固定 checkpoint 验证记录

日期：2026-10-19。权威运行根：`derived/experiments/omniretarget_first/20261019_worktree_w1_rotation_fusion_r10/`。

状态：**实验完整；实现契约（GS）未通过；V 按注册条件未执行；候选不具备接入资格；最终交互方法仍 `not_ready`。** 本轮训练步数 0，未改生产默认入口、未重训练、未扩数据、未处理或重生成教师、未使用 test31。

本轮同时完成了 2026-10-19 审计修复：11 项缺口（8 P1 + 3 P2）全部修复，**候选不接入生产的结论不变**，但 S 的 GQ 统计与 F 收口核验已按修正口径重算。r7/r9 保留为被取代证据，r10 为权威运行。

## 0. 结论摘要

| 裁决 | 结果 | 依据 |
| --- | --- | --- |
| `implementation_contract_passed` | **false** | 已知案例 Deterministic V2 在 frame 120 被 `quaternion_blend_ill_conditioned` 拒收；Diffusion V2 在 107→108 的角速度为 **7.5926 rad/s**，高于原 7 rad/s 门 |
| `experiment_complete` | **true** | S 117/117 键、116 measured + 1 算法拒收；V 条件性 `not_run_due_to_contract_gate`；F 重推 **39/39** 条 V2（38 measured + 1 拒收）、重测 **116/116** 条完整行，0 失败、0 证据缺口 |
| `integration_eligible` | **false** | GS 未通过；且即使 GS 通过，S 的 GQ 也为 false（**3/26 配对通过、94 项退化、0 证据缺口**） |
| `final_interaction_method` | **not_ready** | 与历史 `not_ready` 一致；本轮没有产生可接入的旋转融合协议 |
| 训练步数 | 0 | D/E 的 4,000 步 checkpoint 原样复用；本轮无学习模型调用 |

结论分列：**融合规则按契约实现且证据完整**，但这不等于真实案例修复成功；**known case 仍未过门**，且逐段质量代价显著，候选不进入生产。

## 0.1 审计修复轮：11 项缺口与修正

独立审计在 r7 上发现 8 项 P1、3 项 P2。全部修复后，候选算法与 S 的产品/路径字节未变（见 §0.3），只修正统计与核验口径。

| # | 审计发现 | 修复 |
| --- | --- | --- |
| 1 | 穿透违规计数使用“不得下降”比较器，89→93、90→93 漏报，111→110 反被记为退化 | 新增 `_compare_count_not_increase`；违规计数改为“不增加”判据 |
| 2 | 双手无参考时 `continue` 跳过同侧脚部检查；`sub10_063/diffusion` 右脚漂移 0.06911→0.077702 被漏报 | 手部与脚部检查独立执行；脚部参考变化也显式登记 |
| 3 | F 的 `experiment_complete` 只看 gaps，不看重推/重测 failures | `_experiment_complete` 要求 `failures` 与 `gaps` 均为空；验证失败阻断完整裁决与接入资格 |
| 4 | 重推只核对路径 NPZ，未把重建结果提升到世界系并与产品 root/joints 对账；替换 V2 产品为 V0 仍可通过 | `_recompute_samples` 用注册 clip 的 `world_from_clip` 提升重建结果，与产品 root/joints、world frame、target box、starts 逐位绑定 |
| 5 | 两个 F 核验器遍历现存的 scene 行，清空所有 rows 仍得 0 检查、0 缺口、complete | `_scene_section` 按注册键集重建并把 scene 结果、`rows.json`、summary rows 三方对账；缺行/重复/键序不符即失败 |
| 6 | GS 使用原 `scene_facts`，V 的 GQ 直接读 summary；篡改 known-case 派生记录即可翻门 | GS 的 known-case 事实从实际 path 工件重建；F 重跑合成测试；GQ 使用全行重测后的 verified rows；summary 仅作可复现性交叉检查 |
| 7 | F 重测只对账 4 个字段；手部 gap 增加 1 m 仍零失败 | `_metric_mismatches` 对账完整 metrics 及手/脚/认证计数/违规计数/任务最大误差/硬失败集合等 GQ 相关行字段 |
| 8 | 实质工作前没有索引/台账 running 登记；SIGKILL 后新根仍得已知历史、0 成本、完整预算 | 每个 stage 在实质工作前写 running 索引+台账+`cohort.json`（`accounting_schema=pre_registered_v1`）；历史执行盘×索引×台账三方对账；未封口/未登记尝试为 unknown=停止；已封口的未注册同 cohort 尝试计入成本 |
| 9 | V 跳过分支忽略 `_finish_stage` 返回 False | `val31_stage` 直接返回封口结果，封口失败即调用失败 |
| 10 | 注册异常时 `gaps.append` 先于 `gaps` 初始化，F 抛 `UnboundLocalError` | `failures, gaps` 在任何登记检查前初始化；新增反例驱动 `_finalize` 落盘 incomplete |
| 11 | seam 角度把 rad/s 直接转度；真实 44.8112° 实际单步 1.4937° | `_seam_angle_deg` 乘 `frame_dt`；字段现在是每帧旋转角 |

### 0.2 修正后的 GQ 与关键反例

r10 的 26 个 demo×model 配对：

- **3/26 通过**：`sub3_largebox_001:deterministic`、`sub3_largebox_001:diffusion`、`sub3_largebox_018:deterministic`。
- **94 项失败、0 证据缺口**：92 项数值退化 + 1 项 V2 未测量（Det known）+ 1 项新任务失败（Diff known）。
- 真实增加的违规计数：`sub10_063:deterministic` 89→93、`sub7_032:diffusion` 90→93；`sub7_032:deterministic` 111→110 不再误报。
- 被漏报的脚部退化：`sub10_063:diffusion` 右脚漂移 **0.06911→0.077702 m** 现被登记为 `gq_regression:feet_r_drift_max_m`。
- 失败字段计数：手部 FK 左 10、右脚漂移 8、穿透最大 7、右脚速度 p95 7、教师拟合 7、手部 FK 右 6、右手相对速度 max 6、右手 gap p95 5、右手漂移 5、右脚 gap p95 5、根角速度 4、右手相对速度 p95 4、任务 p95/max 各 3、左手相对速度 max 3、关键帧违规计数 2、左脚漂移 2、左手漂移 2，以及各 1 项的左脚速度 p95、左手相对速度 p95、左手 gap p95、V2 未测量、新任务失败。
- seam 修正样例：`model:sub10_063:diffusion:V0` **44.81116896089725° → 1.4937056320299082°**（恰为 30 倍）。

### 0.3 算法不变性

r10 与 r7 的 117 行逐键对比：

- 116 条 measured 行的 **product SHA-256 逐位相同**；全部 path evidence SHA-256 逐位相同；metrics 逐位相同。
- 唯一差异是 116 行的 `seam_rotation_max_deg` 元数据（r7 为每秒角度，r10 为每帧角度，比值恰 30）。
- 因此审计修复只改变统计/核验口径，没有改变候选融合输出或质量测量。

## 1. 注册、身份与只读前置

- 权威注册文件：`freeze/execution_registration.json`，SHA-256 `6ad7649e060eb078f6e048b946e447c5337a6a50c970db38bf33603be6df5400`。
- 协议：`w1_rotation_path_nlerp_v2`。固定参数：F=60、stride=30、路径歧义门 `abs(dot)<=1e-6`、blend 范数下限 0.5、角速度门 7 rad/s、四元数范数误差上限 1e-5。
- 注册公共身份：固定 checkpoint `deterministic best/step4000`（`7b1efd57…`）与 `diffusion best/step4000`（`1f9c16b1…`），seed 7、DDIM 25。
- 只读绑定：W1-D/E 注册/F 终局、归因注册/主报告/收口、9 项方案锚点、523 项归因锚（含 350 源文件、155 原 W1 产物、26 份保存窗、78 份归因产物）、13 段场景、31 段 val31。注册共 **635** 项锚；R 与 F 均逐项重哈希通过，SOURCE/归因旧字节未改写。
- 六 stage 与四预算域从当前文件重算通过：`r0r1=37.536342120030895s`、`d=8657.971344897s`、`e=524.651844115986s`、`detail_f=570.8396364780492s`，与归因注册逐位一致；SOURCE 六 stage 均无问题。
- 环境：Python 3.12.13、torch 2.14.0+cu126、MuJoCo 3.13.0、CUDA 12.6、RTX 3090；CUDA forward/backward 探针通过；`uv.lock` 哈希入册。
- 实现快照：`freeze/implementation_snapshot/` 保存 tracked diff 与未跟踪实现文件；权威运行 r10 注册的是最终审计修复后的脚本、模块和 54 项测试字节。
- F 封口后仅更新 `AGENTS.md`、`docs/README.md` 与本记录；五个注册实现文件哈希仍与注册快照逐位一致，见 `decisions/attempt_001/post_run_documentation.json`。由于注册的 `implementation_identity` 覆盖 tracked diff/untracked 文件，当前工作树对新调用 `verify_registration` 会报告文档性漂移；635 项锚与四个 `stage_state` 仍干净，已封口 stage/decision 证据不变。
- 固定集合：train6 `sub10_063/sub12_047/sub16_000/sub3_001/sub7_003/sub8_005`；注册 val6 `sub10_100/sub12_008/sub16_052/sub3_018/sub7_032/sub8_001`；已知案例 `sub16_largebox_048`；val31 原 W1 注册名单。

### 1.1 已花成本与运行历史

同一 cohort 的早期尝试均保留，未通过改 run 名洗掉：

| 运行 | 结果 | 已计成本 |
| --- | --- | ---: |
| `…20261019_worktree_w1_rotation_fusion` | 注册锚遗漏 358 项归因闭包，作废 | 6.396s |
| `…_r2` | 注册 index 缺 `report_sha256`，作废 | 7.471s |
| `…_r3` | 脚本组装丢失 `fixed_stage`，S 未启动，作废 | 7.667s |
| `…_r4` | S 在 13 单元后崩溃（键名与 known-case 记录缺陷），crashed 计费 | 7.602 + **261.022s** |
| `…_r5` | S 以 demo-major 顺序封口为 incomplete（键序与注册不符） | 7.406 + **250.027s** |
| `…_r6` | S 完整但 F 实现随后修正（V 未执行的误判、路径/GS 重推），被取代 | 7.789 + **250.224s** |
| `…_r7` | 首个完整 R+S+V+F；审计发现 11 项统计/收口缺口，被 r10 取代 | 334.817s |
| `…_r8` | 审计修复过程中 register 预算检查先于事前登记顺序缺陷产生的 pre-schema 前缀；未注册、无 schema 标记，保留为缺陷证据 | 0.000946s（按 pre-schema 规则不计入 cohort） |
| `…_r9` | 完整 R+S+V+F；修正 GQ/F 的预演，但实现早于解码窗哈希绑定与最终测试，被 r10 取代 | 409.285s |
| **`…_r10`（权威）** | R+S+V+F 全部封口，最终审计修复实现 | **413.582s** |

累计已花（含 r10）：**1,963.290s / 7,200s**。r10 各阶段：R 7.466s、S 252.156s、V 2.419s、F 151.541s；均在 R600/S1800/V3600/F1200 阶段上限内。r8 前缀的 0.000946s 已实测但按 pre-schema 兼容规则未计入；最终 schema 的未注册尝试会被 `cohort.json`/`accounting_schema` 识别为 unknown 并阻止新运行。

## 2. S：固定窗口配对对照

### 2.1 矩阵与身份

- 精确 **117** 键：13 Teacher 参考 + 13×2×3 模型保存窗（V0/V1/V2）+ 13×2 Teacher 往返（V0/V2）。
- 行状态：**116 measured + 1 algorithm_rejected**，无缺行、无重复、无错误分母；`rows.json` 按注册的 teacher/model/roundtrip 顺序落盘。
- 26 份模型保存窗与归因产物逐位重放：V0 **26/26**、V1 **26/26** 的 root/joints 逐位相等（V0 为原生产融合，V1 为归因全贡献顺序短弧 SLERP）。
- V2 位置/关节：25/25 个可计算模型 V2 与同次 V0 逐位相等；25/25 输入数组未被改写。Teacher 往返 V0 **13/13**、V2 **13/13** 的位置/关节逐位相等。
- 13 次 Teacher 编码/生产解码往返为 CPU oracle 调用，与学习模型推理分列；学习模型调用 **0**。
- 逐段配对报告：`decisions/attempt_001/paired_metrics.csv`（117 行，由已封口 `rows.json` 派生；不修改注册产物或 stage 哈希）。

### 2.2 Teacher 往返精度

- 原 3 个旋转负门保留：`sub16_largebox_000`、`sub16_largebox_052`、`sub7_largebox_003`；V2 未新增失败 demo。
- 全 13 段最大：位置 **1.1999e-7 m**、旋转 **0.0242604°**、关节 **5.0636e-7 rad**（均来自既有负门段）。
- V2 相对同次 V0 的增量：位置/关节 0，旋转 0；唯一非零为 `sub8_005` 的旋转增量 **-4.11e-9°**（数值底噪方向为改善）；手部 FK 增量绝对值最大约 **9.8e-18 m**。绝对门（位置/关节 1e-5、旋转 0.001°）与相对门（1e-6）均未被 V2 新增违反。

### 2.3 已知案例 `sub16_largebox_048`

速度序列索引 107 表示帧 107→108，索引 113 表示 113→114；单位 rad/s：

| 模型/方法 | 107→108 | 113→114 | 全局最大 | 结果 |
| --- | ---: | ---: | ---: | --- |
| Deterministic V0 | 15.5588 | 92.4382 | 92.4382 | 原生产 |
| Deterministic V1 | 0.7701 | 5.8955 | 59.0011 | 归因全贡献对照，不参与选优 |
| Deterministic V2 | — | — | — | **frame 120 拒收** |
| Diffusion V0 | 52.1775 | 7.9882 | 82.8984 | 原生产 |
| Diffusion V1 | 0.8806 | 3.1505 | 82.8984 | 归因全贡献对照 |
| Diffusion V2 | **7.5926** | 2.9912 | 9.0810 | 可计算，但 frame 108 超 7 rad/s 门 |

- Deterministic V2 拒收：`quaternion_blend_ill_conditioned`，frame 120，`blend_norm=0.46216499127980826`，active windows `[3,4]`，alpha `[0.7073170731707318, 0.2926829268292683]`，窗口符号 `[1,1,1,-1,1]`，已算前缀 120 帧完整保存。
- 模型窗口自身跳变：Deterministic 第 3 窗（start 90）在全局步 116→117 有 **126.518°** 的窗内跳变；该坏跳变保留，未用速度裁剪、平滑或投影掩盖。
- 结论：固定路径把 Diffusion 107→108 从 52.18 降到 **7.59 rad/s**（约 -85%），但未过 7 rad/s 门；Deterministic 在后续 120 帧因近反极点加权退化被诚实拒收。**已知案例机制未被完整修复。**

### 2.4 GS 裁决

GS **未通过**，失败事实：

1. `known_case_v2_not_computable:deterministic`（frame 120，blend norm 0.462）；
2. `known_case_frame_108_over_speed:diffusion`（7.5926 > 7 rad/s）。

GS 的其余事实均通过：117 键完整、V0/V1 逐位重放、V2 位置/关节逐位、输入未改写、Teacher 往返绝对/相对门、已知案例拒收有完整前缀与 parent 窗口。合成数学测试 **54 项**在 S 启动时复跑通过。

### 2.5 GQ：逐段质量不退化

GQ 为保守门，S 的 train6/val6/known 共 26 个 demo×model 配对：

- **3/26 通过**：`sub3_largebox_001:deterministic`、`sub3_largebox_001:diffusion`、`sub3_largebox_018:deterministic`。
- **94 项失败、0 证据缺口**：92 项数值退化 + 1 项 V2 未测量（Det known）+ 1 项新任务失败（Diff known）。
- 修正后的比较器把“违规计数增加”作为退化，同时允许违规计数下降；手部无参考不再跳过脚部检查。审计指出的 89→93、90→93 与 `sub10_063/diffusion` 右脚漂移均现被登记，111→110 不再误报。
- 退化字段与计数见 §0.2；结论是“把全部正权重纳入 + 固定路径”确实改变了质量，且多数变化为退化；即使 GS 通过，GQ 也会阻止接入。

## 3. V：固定 checkpoint 的完整 val31

S 的 GS 未通过，按注册条件 V **未执行**：

- `val31/attempt_001/summary.json` 状态 `not_run_due_to_contract_gate`，`runs=0`、`units=0`、`rows=0`；
- 学习模型调用 **0**，没有用“当前可加载/准入的子集”推导或缩分母；
- 没有声称已完成 val31；155 行五方法矩阵未生成。

## 4. F：独立重推与收口

- 从保存的 decoded 窗口重推全部 V2：**39/39** 条（25 模型 + 13 Teacher 往返 + 1 拒收）；38 条 measured、1 条拒收。位置/关节逐位、旋转按 SO(3)、lift/权重/blend/符号记录逐位对账；拒收的 120 帧前缀、code/frame/window/local_frame/prefix_length 全部重推一致。
- 重推前验证 decoded window / oracle window 工件哈希；重建结果提升到世界系后与产品 `root`/`joints` 逐位绑定，并核对 `world_from_clip`、`target_box`、`window_starts`。替换 V2 产品为 V0 会触发 `product_not_bound_to_recomputed_path`。
- 用原独立验收器重测全部 **116/116** 条 measured 行（Teacher + V0/V1/V2 + 往返）：完整 metrics 以及 root/joint 误差、手部 FK、角速度、seam 单步角、关键帧违规计数全部逐位对账；0 失败、0 证据缺口。
- scene 结果按注册键集重建：117 键精确、无重复；`rows.json` 与 summary rows 必须与 scene 结果逐位一致。清空 rows 或删除 scene 会阻断 complete。
- GS 的 known-case 事实从实际 path 工件重建，合成数学测试在 F 重新运行（54 passed）；V 的 GQ 从重测后的 verified rows 重建（V 未执行时为 `gq_missing`）。summary 的 `gs`/`gq` 只作为可复现性交叉检查。
- 635 项锚在 F 收口时再次重哈希不变；四个 stage 的盘×索引×台账一致；无 running/publishing 残行。每个 stage 事前写 running 登记，`accounting_schema=pre_registered_v1`。
- 最终文件：`decisions/attempt_001/fusion_final.json`，SHA-256 `055432bf9d201408175a1b560ec9a69cdc01f3879fc9be6f7ff25247f56b9abe`。`failures=[]`、`evidence_gaps=[]`；`blocking_findings` 97 项（2 项 GS + 94 项 GQ + 1 项 V 条件未执行标记）。

## 5. 成本与预算

r10 实测：R 7.466s、S 252.156s、V 2.419s、F 151.541s，合计 **413.582s**；加同一 cohort 已花 1,549.708s，累计 **1,963.290s ≤ 7,200s**。各阶段均未超过注册上限；S 的先前崩溃/作废尝试与 r7/r9 成本均已计入 cohort 历史，未通过改名或改提交洗掉。训练步数 0，无未入账 GPU 采样。

## 6. 测试

- `tests/test_rotation_fusion.py` 17 项 + `tests/test_w1_rotation_fusion.py` 37 项，合计 **54 passed**（8.44s）。新增反例覆盖：违规计数方向、手部无参考时脚部独立检查、完整 metrics/行字段对账、scene 键集/空行、产品世界系绑定、解码窗哈希、失败阻断 complete、GS 从 path 重建、事前登记、盘×索引×台账、未注册封口尝试计费、SIGKILL 中止新运行、V 跳过封口失败传播、gaps 初始化、seam 单步角。
- 相关回归（rotation + generation + attribution + W1-D/E closure/driver/numerics/raw）：**229 passed**（50.95s）。
- 全项目套件：`uv run --locked pytest tests/ -q` → **877 passed, 1 skipped**（1957.17s；32:37）。较上轮 861 增加 16 项，正好对应审计修复新增的反例。

## 7. 边界与下一步

- 候选 `w1_rotation_path_nlerp_v2` **不接入生产**；没有切换默认生成器、没有改缓存、没有改条件代码、没有重训练。
- 已知案例仍有 Deterministic frame 120 拒收与 Diffusion frame 108 的 7.59 rad/s；26 条保存窗的预测侧残差仍在（归因已确认），融合规则不能替代模型/条件侧改进。
- GQ 的 94 项退化表明全贡献固定路径不是质量中性修复；任何后续融合候选必须另立预注册，明确处理近反极点 NLERP 放大/拒收和接触质量代价，不能在本注册上就地调参或重跑。
- 审计修复只改变统计/核验口径：r10 与 r7 的 product/path/metrics 逐位相同（除 seam 元数据）；r7/r9 保留为被取代证据，r8 前缀保留为 pre-schema 缺陷证据。
- 原 U0 `incomplete`、UR `no_quality_gain`、原 F `not_ready` 继续有效；`next_action.auto_execute=false`。
- 旧 SOURCE、归因 523 锚、155 原产物和本方案字节保持不变；本记录是新增证据，不回写冻结方案。
