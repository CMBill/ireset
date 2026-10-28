# W1 真实训练解码、FK 几何与输出空间导数审计 —— I/C 实现与纯合成验证记录

日期：2026-10-28。状态：**`implemented_synthetic / calibrated / admission_ready / not_authorized / not_run`**。

本记录只覆盖方案 §4.1 的 I/C 工程工作、纯合成校准、只读准入包，以及 11 项只读审阅问题（8 P1 + 3 P2）的修复。真实 R0→G→D→F、真实父数组数值读取、锁定 G1/MuJoCo 编译、checkpoint、cohort、预算与注册均未执行。所有通过结论均为**纯合成契约通过**，不替代真实证据，也不改变 `objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false`。

## 1. 实现身份

| 文件 | SHA-256 |
| --- | --- |
| `src/ireset/eval/w1_real_fk_audit.py` | `c2237383b28185e3195177481284fa921486dfccd06ff039503549bb561e46dc` |
| `scripts/audit_w1_real_fk.py` | `80ededcb5edef05e798e8ac89b3d5c89343f0654996e4df63010069b8d451673` |
| `tests/test_w1_real_fk_audit.py` | `1f1c0469af7b2581a01ed0e768074904401e00e5bcaf1e34fa6e719c07d65ffe` |
| `src/ireset/train/losses.py`（仅新增可选注入） | `33f68d0bf1b9b5c31940b887bdf5413afebe192ee53fd0229782232abcf8f661` |
| 预注册草案（最终字节，准入包绑定） | `92a0ec574191ee00e4a3136dc46dc4eb9fc5e1d7347d743804ce36048fc8b895` |

`LossAssembler` 的 `fk=` / `static_frame=` 仍只是可选注入：默认值构造生产 `G1TorchFK` 并调用 `largebox_static_frame()`，生产数值路径不变；合成路径注入 29 关节合成 FK 与单位静态框，避免加载真实资产。`RealMujocoReference` 在真实 G 首次构造时校验 nq/nv、29 关节名称/顺序/qpos 地址/hinge 类型/限位、15 site 所属 body 与局部位姿；该路径未在 I/C 执行。

本轮新增/修改的实现要点：

- `_capture_run` 从 `assembler.device` 创建 leaf 与 `x0` 张量，避免真实 CUDA 装配器的设备冲突；同时读取 live `term_tensors["fk_m2"]`，要求其与回报标量和元素归约三方一致。
- `_CountingFk` 在真实 `site_world_positions` 调用点累计 `production_fk_batch`；G/D/F 在计算前写入 `call_reservations.json`（中断不返还）。
- `run_r0` 新增资产闭包（锁定 4 文件 + 全部 mesh/include 引用哈希）、父 R0/P/F 盘×索引×台账消费者状态、输入 dtype/形状/非空/有限/限位顺序、生效 float32 常数与 6D/clamp 域事实。
- `run_f_stage` 逐单元调用 `unit_admission` 与实时截止；G/D 工件分别核验记录哈希、schema、行结构与重叠数组后才合并；重算在派生 Python 子进程中执行并记录进程身份与调用增量。
- `run_registered_audit` 在 D/F 之前重新消费上游阶段；预算撤销/未知历史停止链条（纯数值 mismatch 仍可继续定位）。
- `withdraw_attempt_claim` 先写 `publishing`、核验覆盖后才写终态；任一未证明即保持 publishing 并返回失败。
- FD clamp 分段检查覆盖 `±1` 与边界落点；解码导数域缺口传播为梯度行 `domain_gap`。

## 2. 测试

```bash
uv run --locked pytest tests/test_w1_real_fk_audit.py -q
```

→ **58 passed（32.23s）**。

相关回归（父真实目标审计、真实准入、生成、标签质量、FK）：

```bash
uv run --locked pytest tests/test_w1_real_objective_audit.py \
  tests/test_w1_real_admission.py tests/test_generation.py \
  tests/test_label_quality.py tests/test_fk.py -q
```

→ **107 passed（35.57s）**。全项目套件本轮未重跑。

覆盖与方案 §9.1 的对应：

| 验证组 | 已覆盖 |
| --- | --- |
| 解码/几何 | 列/行 6D 混用、wxyz/xyzw、退化 6D、`|x_j|=1` 正负边界 gap、`|x_j|>1` 零导数、近 π 正域外负例、目标两副本逐位 |
| Jacobian/梯度 | 35/38 维 FD、漏 `R.T`、错关节枢轴、错关节尺度、分母 3600/900、窗口均值=batch、清零梯度、clamp 零梯度、分组 L2/cosine 门、解码域缺口传播 |
| 证据独立性 | 键缺失/额外/重复、行字段/分母/状态/bool 篡改、数组非逐位/超容差/NaN/缺键、父绑定逐位、G/D 两份 NPZ 分别哈希与重叠冲突、坏 JSON/NPZ 结构不崩溃 |
| 准入/调用 | 真实入口无授权零写入拒绝、实验根守卫、实现/草案哈希绑定与 R0 漂移门、调用点 FK 计数、单元预留、累计 cap、尝试上限、R0 零 FK/model 调用、F 单元准入/截止/独立进程 |
| 账本/封口 | 负/NaN/Inf 成本拒绝、running/publishing 不可消费、半损坏历史 unknown、外部 cohort/无标记 legacy 归属、可消费 done 封口、慢持久化不得发布成功、撤回 publishing 先行与不可验证保持 unknown |
| R0 事实 | 输入 schema/生效常数/6D 域、限位顺序反例、资产闭包（临时 XML/URDF 树，缺失引用即失败）、父消费者状态（未完成即停止） |

真实资产相关测试仍按方案 §9.1 保留为真实计算，不包装成 I/C 校准。

## 3. 冻结校准

命令：

```bash
uv run --locked python scripts/audit_w1_real_fk.py --calibrate-synthetic \
  --out derived/audit/w1_real_fk_audit/tolerance_contract.json
```

| 项 | 值 |
| --- | --- |
| 契约文件 | `derived/audit/w1_real_fk_audit/tolerance_contract.json` |
| 契约 SHA-256 | `4e51b7d7e9327b1693782c39286b2e8121200c14db0e3889a629025ed0490bb4` |
| 证据 SHA-256 | `8f46fbe1830d4576f61ff85c8384a78e057ff3f5a900299e5c8c3440e3df0ed9` |
| 模式 | `synthetic_calibrated` |
| seed / 样本 | `20261028` / 64 状态；4 个 FD 状态 × 35/38 维 × 2 个 h |
| 真实读取 | `real_asset_loaded=false`、`real_checkpoint_loaded=false`、`real_dataset_loaded=false`、`real_model_calls=0` |

证据哈希与上一轮相同（校准数学未改）；契约文件 SHA 变化仅来自 `created_wall` 时间戳。

观测最大比例与 4× 规则：

| 检查 | max ratio | 4× | 提案上界 |
| --- | ---: | ---: | ---: |
| decode_float32 | 0.090188 | 0.36075 | atol 2e-6 / rtol 2e-6 |
| decode_rotation_matrix | 0.008517 | 0.03407 | atol 2e-5 / rtol 0 |
| kernel64_fk | 4.29e-10 | 1.72e-9 | atol 1e-6 / rtol 1e-6 |
| production_fk | 0.056518 | 0.22607 | atol 5e-6 / rtol 2e-6 |
| fk_scalar | 0.002838 | 0.011352 | atol 1e-8 / rtol 2e-5 |
| fk_gradient | 0.200238 | 0.80095 | atol 1e-8 / rtol 1e-4 |
| j35_fd | 5.47e-5 | 2.19e-4 | atol 2e-6 / rtol 2e-5 |
| j38_fd | 2.28e-5 | 9.14e-5 | atol 1e-5 / rtol 1e-4 |

全部 `4× observed ≤ proposal`，阈值保持提案值，未抬高。校准器在加载契约时重算证据与门槛；替换证据、同步重哈希后放宽阈值、改动提案阈值、删除负例均被拒绝（测试覆盖）。已登记边界/负例：近 π 生产矩阵误差与 quat 标量作为事实保留（不参与容差）；`|x_j|=1` 无唯一双侧导数（gap）；`|x_j|>1` 导数精确为 0；退化 6D 为 gap；零残差损失/梯度为 0。

## 4. 纯合成 R0→G→D→F

命令：

```bash
uv run --locked python scripts/audit_w1_real_fk.py --synthetic \
  --out derived/audit/w1_real_fk_audit/synthetic/20261028_review10
```

运行根：`derived/audit/w1_real_fk_audit/synthetic/20261028_review10/`（直接位于 `derived/audit/`，不是研究实验根）。`synthetic/20261028/` 是修复前历史运行，`synthetic/20261028_review10_preduplicate/` 是移除重复 helper 前的同语义运行（两份 NPZ 与本次逐位相同），均保留对照；不覆盖、不改写。

| 阶段 | 结果 | 成本（合成） |
| --- | --- | ---: |
| R0 | 绑定/窗口/键数/schema/输入 schema/零调用检查通过；父消费者与资产闭包在合成模式显式 `checked=false` | 1.1195s |
| G | 122 几何行 + 20 梯度行，0 失败/0 缺口 | 1.4331s |
| D | 4672 FD 行（32 状态 × 两个 h），0 失败/0 缺口 | 2.9642s |
| F | 独立子进程重读磁盘、重算 G+D、逐位/容差对账、全 stencil 重放；`problems=[]` | 6.2167s |

合计 **11.7335s**（F 含子进程启动），仅证明管线可运行，不是真实运行时间估计。

最终产物：

| 产物 | SHA-256 |
| --- | --- |
| `finalize/attempt_001/fk_audit_final.json` | `f2622d4d58899b4a4a00fe98515483e286bc97b59e78fafd2afc0fcf7ef07d9b` |
| `finalize/attempt_001/independent_verification.json` | `eb269be4c1ae3a9aa5fb6ff02033dfd7b60e78a49691bc1facdbff0de575694f` |
| `geometry/attempt_001/fk_geometry_arrays.npz` | `be251f2dcd17826ac374ccaf1e39ca0fa216f02aa4f919d0d68b3172e910ae29` |
| `geometry/attempt_001/fk_gradient_arrays.npz` | `98f3b21cdf8ef3c4ff99ae9495b6bafd7c65a82c27ad331c05d41ce2dc100b8e` |
| `geometry/attempt_001/geometry_rows.json` | `2a6089f549670d496bcd0e2e57b7e46b9d761b2a4dddffeabd0fc2662b90b25f` |
| `derivative/attempt_001/fd_arrays.npz` | `3202046f95fd07913f4edf1d23c198edb1ef2ddd4673f8ea3e8f6b959bce216d` |
| `derivative/attempt_001/fd_rows.json` | `98c98ff2e423c69bd337eb82aa7a18242c0d896bbed1379085a4c58402f8fd0f` |
| R0 `registration.json` | `b427b0c56994a69b6543d961ecb9505ecd98e16da9d76b7a56e70b67984c8468` |
| R0 `admission.json` | `5c492f27bfe2bbe25c271fdd9a564ebee4733c3c6c824b177ada03a32c16f13c` |

最终 `conclusion=synthetic_contract_passed`、`complete_evidence=true`、`covered_contract_passed=true`、`failures=[]`、`gaps=[]`、`verification_problems=[]`；四个阶段均 `done/complete=true` 且 `stage_state` 可消费。

调用与提案矩阵一致：G `loss_assembler=2`、`production_fk_batch=4`、`kernel64_fk_batch=3`、`production_vjp=2`、`mj_forward=1200`、`mj_jac_site=7200`；D `mj_forward=9344`、`fd_pose_eval=9344`；F `mj_forward=10544`、`mj_jac_site=7200`、`fd_pose_eval=9344`。累计快照与全链名义值一致；合成模式 `scene_setup_forward=0`（真实模式应为 1）。全链零计数均为 JSON 整数 0。

F 过程与单元：

- `recompute_process.start_method=spawned_python_subprocess`，pid 与父进程不同、ppid 为父进程，`exit_reason=ok`。
- 单元 `artifact_read`（父）、`recompute_backend`（父）、`recompute_backend`/`geometry_replay`/`fd_replay`（子）全部 `admitted=true`。
- 父进程按子进程回报的 `call_delta` 记入同一调用台账；单元调用上界在计算前写入 `call_reservations.json` 并在完成后记录 `consumed`。

R0 记录：`inputs.problems=[]`（形状/dtype/有限/限位顺序/生效常数/域），`checks.input_schema=true`；`parent_consumer_state.checked=false` 与 `locked_assets.checked=false` 是合成模式的显式事实，不伪造真实核验。

## 5. 准入包

命令：

```bash
uv run --locked python scripts/audit_w1_real_fk.py --admission \
  --out derived/audit/w1_real_fk_audit/admission \
  --tolerance-contract derived/audit/w1_real_fk_audit/tolerance_contract.json
```

| 项 | 值 |
| --- | --- |
| 包路径 | `derived/audit/w1_real_fk_audit/admission/attempt_009/admission_package.json` |
| 包 SHA-256 | `21a7509f3b1206399e402bc07dd6bf9dc21e777d11f427706a54206b9f627985` |
| `passed` | true（`problems=[]`） |
| 状态 | `authorized=false / registered=false / run=not_run / section_9_2_checked=false` |
| 父运行绑定 | 7/7 文件哈希匹配（注册/准入/p_report/两份数组/终局/独立复核） |
| 固定窗口 | 4/4 与父注册 `window_keys` 一致 |
| 键数 | geometry 122 / fd 4672 / states 32 |
| 全链调用上限 | `loss_assembler=4`、`production_fk_batch=8`、`kernel64_fk_batch=6`、`production_vjp=4`、`mj_forward=21088`、`mj_jac_site=14400`、`fd_pose_eval=18688`、`scene_setup_forward=2` |
| 读取范围 | 仅元数据/文件哈希；未加载窗口数组、未编译资产、未 `torch.load` checkpoint、未创建 cohort 或正式运行根 |

包内 `implementation.files` 与草案/校准文件/证据哈希均与当前字节一致；改动任一被绑定文件会使准入包和真实入口拒绝。`attempt_001`–`attempt_008` 是逐步加入撤回成本、慢持久化、R0 实现漂移门、真实参考事实、审阅修复前实现与草案、审阅修复后实现、以及同步成本/耗时纪录的历史包；`attempt_009` 为当前权威。

包内 `r0_obligations` 明确列出真实 R0 必须落实的 6 项：资产闭包、父消费者状态、输入 schema/生效常数/域、调用点计数与单元预留、F 独立子进程、G/D 消费者门。

## 6. 只读审阅修复（8 P1 + 3 P2）

| # | 级别 | 问题 | 修复 | 反例/证据 |
| --- | --- | --- | --- | --- |
| 1 | P1 | CPU leaf 与真实 CUDA assembler 设备冲突 | `_capture_run` 使用 `assembler.device` 创建全部张量 | `test_capture_binds_tensors_to_the_assembler_device`：声明 `cuda:0` 的 assembler 实际收到 `device="cuda:0"` |
| 2 | P1 | G/D 预算撤销后仍继续并最终通过 | 阶段间重新 `stage_state`；不可消费即停止（数值 mismatch 仍可定位） | `test_budget_revoked_g_stops_the_chain`：撤销的 G 后无 D/F、零新调用 |
| 3 | P1 | F 无实时截止与逐单元准入 | F 在工件读取/后端/几何/FD 前逐单元 `unit_admission`（子进程内同样执行） | `test_f_refuses_an_unregistered_parent_attempt`、`test_f_deadline_blocks_before_backend` |
| 4 | P1 | FK 标量审计未使用真实 live 标量 | batch 行使用 `run.fk_m2` 并与 live tensor、元素归约三方对账 | `test_shifted_live_scalar_fails_the_batch_row`：+5e-7 被 1e-8 门拒绝 |
| 5 | P1 | 两份 NPZ 合并掩盖冲突 | 分别核验路径/记录哈希、schema、行结构、重叠数组逐位一致 | `test_f_verifies_gradient_npz_hash_separately`、`test_f_detects_overlap_conflict_between_geometry_and_gradient_npz` |
| 6 | P1 | R0 未落实资产闭包/父消费者/输入 schema | 新增资产闭包、父 R0/P/F 消费者状态、输入 schema/生效常数/域事实；合成模式显式不检查 | `test_asset_closure_hashes_every_reference`、`test_parent_consumer_state_requires_complete_rows`、`test_input_schema_facts_reject_bad_shapes` |
| 7 | P1 | 撤回先写可消费终态再查成本 | 先 publishing、核验覆盖后才写 incomplete；未证明即保持 publishing 并返回失败 | `test_withdraw_publishes_before_any_terminal_claim`、`test_withdraw_stays_publishing_when_terminal_write_outruns_budget` |
| 8 | P1 | 生产 FK 调用数预设 +2 | `_CountingFk` 在真实调用点计数；单元上界预留持久化 | `test_production_fk_calls_are_counted_at_the_call_site`：未执行 FK 为 0，成功捕获等于真实调用数 |
| 9 | P2 | F 未启动新进程 | F 重算在派生 Python 子进程执行并记录 pid/ppid/python/exit_reason/call_delta | 合成全链 `recompute_process` 与 `test_synthetic_self_test_end_to_end` |
| 10 | P2 | clamp 域只查 `+1` 且未传播到梯度 | FD 分段检查改为 `|u±h|` 双侧；解码域缺口使梯度行 `domain_gap` | `test_negative_clamp_boundary_is_a_domain_gap` |
| 11 | P2 | 坏工件在结构校验前 `.get()` 崩溃 | 先校验 JSON/NPZ 容器、行与键类型，异常转为 verification gap | `test_malformed_fd_rows_become_gaps_not_exceptions` |

审阅基线的 58 passed 重新成立；修复未改变校准证据哈希。

## 7. 仍未执行与未覆盖

- 未勾选方案 §9.2 九项；未创建真实 cohort/预算/注册；未执行真实 R0/G/D/F。
- 未读取父 `objective_real_arrays.npz` 的数值数组（准入包只哈希文件）、未编译锁定 MJCF/URDF、未加载两份 checkpoint。
- 未审计：真实 float32/CUDA 生产前向、真实 MuJoCo 编译后地址/site 位姿、真实父 FK 逐位绑定、真实 FD、网络参数梯度、累积/裁剪/AdamW/更新、选模、DDIM、端到端动作质量。
- 资产闭包/父消费者状态/输入 schema 的真实分支未执行，只在临时合成树上验证实现；真实资产内容仍以 R0 运行时哈希为准。
- `x0_independent` 只做 schema/dtype/有限性与 6D 域检查；`x0`↔`x0_independent` 的编码关系与 `raw_root`/`raw_q`/`raw_wfc` 未在本模块重算，仍由父注册与父审计绑定，本方案不把它当作新参考真值。
- F 的独立性来自新进程/新后端/磁盘重读；参考公式与 G 共用同一套 NumPy/MuJoCo 实现，不是第三套数学证明。
- I/C 的 `scene_setup_forward=0` 是合成事实；真实 G/F 每个 stage 应各为 1，由 R0/G 首个已登记单元核验。
- 真实入口在无授权、无校准、无准入包或运行根不是固定历史根直接子目录时拒绝，且在读取数组/加载资产/创建运行根之前退出（测试覆盖）。

## 8. 复现命令

```bash
uv run --locked pytest tests/test_w1_real_fk_audit.py -q
uv run --locked python scripts/audit_w1_real_fk.py --calibrate-synthetic \
  --out derived/audit/w1_real_fk_audit/tolerance_contract.json
uv run --locked python scripts/audit_w1_real_fk.py --synthetic \
  --out derived/audit/w1_real_fk_audit/synthetic/20261028_review10
uv run --locked python scripts/audit_w1_real_fk.py --admission \
  --out derived/audit/w1_real_fk_audit/admission \
  --tolerance-contract derived/audit/w1_real_fk_audit/tolerance_contract.json
```

真实执行入口保持不可达：

```bash
# 无授权/校准/准入包时拒绝；本仓库状态未勾选 §9.2
uv run --locked python scripts/audit_w1_real_fk.py --real \
  --out derived/experiments/omniretarget_first/<new_run> \
  --authorization <authorization.json> \
  --tolerance-contract derived/audit/w1_real_fk_audit/tolerance_contract.json \
  --admission-package derived/audit/w1_real_fk_audit/admission/attempt_009/admission_package.json
```

## 9. 边界

`objective_coverage=partial`、`full_objective_contract_passed=null`、`attribution=attribution_insufficient`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false`。旧 U0 `incomplete`、UR `no_quality_gain`、原 F `not_ready`、预测 r6 不可验收、旋转融合不接入均保持不变。真实通过/失败只能由取得授权后的 R0→G→D→F 产生，不能由本记录外推。
