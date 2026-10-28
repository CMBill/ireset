# W1 真实训练解码、FK 几何与输出空间导数审计 —— I/C 实现与纯合成验证记录

日期：2026-10-28。状态：**`implemented_synthetic / calibrated / admission_ready / not_authorized / not_run`**。

本记录覆盖方案 §4.1 的 I/C 工程工作、纯合成校准、只读准入包，以及三轮只读审阅问题（第一轮 8 P1 + 3 P2，第二轮 5 P1 + 3 P2，第三轮 1 P1 + 1 P2）的修复。真实 R0→G→D→F、真实父数组数值读取、锁定 G1/MuJoCo 编译、checkpoint、cohort、预算与注册均未执行。所有通过结论均为**纯合成契约通过**，不替代真实证据，也不改变 `objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false`。

## 1. 实现身份

| 文件 | SHA-256 |
| --- | --- |
| `src/ireset/eval/w1_real_fk_audit.py` | `1e5fc55974978bda678c95ada69a5868abfa8b9795b3db80e6b47dc7a02a4c2c` |
| `scripts/audit_w1_real_fk.py` | `ae1108a632b9bd097d8f23c82da79b16cd23d92675676e06fb93e73a03b749d9` |
| `tests/test_w1_real_fk_audit.py` | `dce5885de5c2054768c039afdb18a911b94df848126dced3b7d779e02056e3a9` |
| `src/ireset/train/losses.py`（仅新增可选注入） | `33f68d0bf1b9b5c31940b887bdf5413afebe192ee53fd0229782232abcf8f661` |
| 预注册草案（最终字节，准入包绑定） | `e6fb0407850bd308d371a53ca9857c987478a020921c6dc815a2a00362b57463` |

`LossAssembler` 的 `fk=` / `static_frame=` 仍只是可选注入：默认值构造生产 `G1TorchFK` 并调用 `largebox_static_frame()`，生产数值路径不变；合成路径注入 29 关节合成 FK 与单位静态框，避免加载真实资产。`RealMujocoReference` 在真实 G 首次构造时校验 nq/nv、29 关节名称/顺序/qpos 地址/hinge 类型/限位、15 site 所属 body 与局部位姿；该路径未在 I/C 执行。

本轮（第二轮审阅修复）新增/修改的实现要点：

- `run_registered_audit` 在建立 `CallCounter` 后调用 `provider.bind_counter(counter)`；G/D 各自记录真实 `call_delta`，合成全链与 `full_chain_calls()` 精确相等，任一阶段调用为零即记 `stage_call_delta_zero:*`。
- `finish_attempt` 把同一终态 `call_balance` 写入索引与台账；撤回路径同步写入。`stage_state`/`stage_history` 校验两侧快照一致（`call_balance_single_sided`/`call_balance_disagreement`）。
- worker 在 `finally` 结算实际调用差；父进程合并任何已报告 delta。worker 未报告 delta（超时/被杀）时预留保持 `consumed=null` 并记 `f_reservation_left_unsettled`；`stage_state`/`stage_history`/`prior_call_counts` 把未结算预留视为未知历史。
- `_ensure_deadline` 在模型批次、参考/核调用与每个 FD 状态前复核阶段截止与预留；G/D/F（含子进程）传入同一截止。
- F 把单独保存的 20 条 `gradient_rows` 同时绑定主表行与 fresh 重推行，并把 `rows` 文件纳入封口哈希校验。
- geometry/gradient/FD 三个 NPZ 分别按冻结 schema 校验必需键、dtype、shape 与非数值类型后才合并；gradient 文件另与 fresh 分区逐元素比较。
- 最终工件哈希记录改用 `_file_sha`（缺失返回 `None`），缺失 NPZ 只登记缺口，不再抛 `FileNotFoundError`。
- 新增 `RUNTIME_REQUIREMENTS`/`configure_production_runtime`：真实入口在构造 provider 前将 TF32 关闭、matmul precision 设为 `highest`、确认 autocast 关闭；R0 复核实际值，F 父/子进程分别记录并互比；准入包记录运行时契约、设置前/后观测与 `applied`。

第一轮修复（设备绑定、live 标量、调用点计数、阶段门、F 单元准入/子进程、工件逐文件哈希、R0 资产闭包/父消费者/输入 schema、撤回 publishing 先行、负 clamp 边界）继续有效。

## 2. 测试

```bash
uv run --locked pytest tests/test_w1_real_fk_audit.py -q
```

→ **71 passed（50.15s）**。

相关回归（父真实目标审计、真实准入、生成、标签质量、FK）：

```bash
uv run --locked pytest tests/test_w1_real_objective_audit.py \
  tests/test_w1_real_admission.py tests/test_generation.py \
  tests/test_label_quality.py tests/test_fk.py -q
```

→ **107 passed（35.40s）**。全项目套件本轮未重跑。

覆盖与方案 §9.1 的对应：

| 验证组 | 已覆盖 |
| --- | --- |
| 解码/几何 | 列/行 6D 混用、wxyz/xyzw、退化 6D、`|x_j|=1` 正负边界 gap、`|x_j|>1` 零导数、近 π 正域外负例、目标两副本逐位 |
| Jacobian/梯度 | 35/38 维 FD、漏 `R.T`、错关节枢轴、错关节尺度、分母 3600/900、窗口均值=batch、清零梯度、clamp 零梯度、分组 L2/cosine 门、解码域缺口传播 |
| 证据独立性 | 键缺失/额外/重复、行字段/分母/状态/bool 篡改、数组非逐位/超容差/NaN/缺键、父绑定逐位、三份 NPZ 各自 schema 与重叠一致、梯度明细与主表/fresh 逐字段、坏 JSON/NPZ 结构不崩溃、缺失 NPZ 不抛异常 |
| 准入/调用 | 真实入口无授权零写入拒绝、实验根守卫、实现/草案哈希绑定与 R0 漂移门、调用点 FK 计数、全链逐阶段真值、单元预留与未结算未知、累计 cap、尝试上限、R0 零 FK/model 调用、F 单元准入/截止/独立进程 |
| 账本/封口 | 负/NaN/Inf 成本拒绝、running/publishing 不可消费、半损坏历史 unknown、外部 cohort/无标记 legacy 归属、可消费 done 封口、索引/台账调用快照一致、慢持久化不得发布成功、撤回 publishing 先行与不可验证保持 unknown |
| R0 事实 | 输入 schema/生效常数/6D 域、限位顺序反例、资产闭包（临时 XML/URDF 树，缺失引用即失败）、父消费者状态（未完成即停止） |
| 运行时 | TF32/high 环境被拒、设置后五项要求全部满足、准入包要求 applied 与设置后观测一致 |

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
| 契约 SHA-256 | `e70fa5b51e0be94cd6c0f809c606648af787cd2fbe4f6ac52d39ce37557c2392` |
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
  --out derived/audit/w1_real_fk_audit/synthetic/20261028_review11
```

运行根：`derived/audit/w1_real_fk_audit/synthetic/20261028_review11/`（直接位于 `derived/audit/`，不是研究实验根）。历史/中间运行保留对照且不改写：

- `synthetic/20261028/`：修复前历史运行；
- `synthetic/20261028_pre_drift/`、`20261028_pre_reffacts/`、`20261028_pre_withdrawal/`：更早的中间运行；
- `synthetic/20261028_review10/`：第一轮审阅修复后的权威运行。其 G/D `call_delta` 因主 provider 未绑定计数器而全零，调用统计据此订正为“仅 F 计入”，其余工件保留；
- `synthetic/20261028_review10_preduplicate/`：移除重复 helper 前的同语义运行；
- `synthetic/20261028_review11_prefinal/`–`_prefinal5/`：第二/三轮修复过程中绑定早期模块/草案字节的中间运行（数值与最终运行一致，仅身份/时间戳不同）。

| 阶段 | 结果 | 成本（合成） |
| --- | --- | ---: |
| R0 | 绑定/窗口/键数/schema/输入 schema/零调用检查通过；父消费者与资产闭包在合成模式显式 `checked=false` | 1.1087s |
| G | 122 几何行 + 20 梯度行，0 失败/0 缺口 | 1.4430s |
| D | 4672 FD 行（32 状态 × 两个 h），0 失败/0 缺口 | 3.0044s |
| F | 独立子进程重读磁盘、重算 G+D、逐位/容差对账、全 stencil 重放；`problems=[]` | 6.2031s |

合计 **11.7593s**（F 含子进程启动；逐次运行有几十毫秒波动），仅证明管线可运行，不是真实运行时间估计。

最终产物：

| 产物 | SHA-256 |
| --- | --- |
| R0 `registration/attempt_001/registration.json` | `df1e17505d935b48db38e58477885338009b7f5957615ad8a2b7e13344b6e2b1` |
| R0 `registration/attempt_001/admission.json` | `5c492f27bfe2bbe25c271fdd9a564ebee4733c3c6c824b177ada03a32c16f13c` |
| `geometry/attempt_001/geometry_rows.json` | `a3658b69d5d0a625a8b9f0e29b4c02f077e9d73833aa01bc24b73299c0085e0d` |
| `geometry/attempt_001/fk_geometry_arrays.npz` | `be251f2dcd17826ac374ccaf1e39ca0fa216f02aa4f919d0d68b3172e910ae29` |
| `geometry/attempt_001/fk_gradient_arrays.npz` | `98f3b21cdf8ef3c4ff99ae9495b6bafd7c65a82c27ad331c05d41ce2dc100b8e` |
| `derivative/attempt_001/fd_rows.json` | `98bb4fe75d13e9f2d4a811be4afe57454dda1195772a21085a6fdc593f38dfca` |
| `derivative/attempt_001/fd_arrays.npz` | `3202046f95fd07913f4edf1d23c198edb1ef2ddd4673f8ea3e8f6b959bce216d` |
| `finalize/attempt_001/fk_audit_final.json` | `f5ddd094f4b395d6c2cc92a8c38f1f943953599b9831bc65defee6b5dcc69668` |
| `finalize/attempt_001/independent_verification.json` | `0924ca9fbd77a077060eb811407fab399588cc0fdbe557f0d6f1490f2a0d3d88` |

最终 `conclusion=synthetic_contract_passed`、`complete_evidence=true`、`covered_contract_passed=true`、`failures=[]`、`gaps=[]`、`verification_problems=[]`；四个阶段均 `done/complete=true` 且 `stage_state` 可消费。R0 不产生 FK/model 调用，无 `call_delta`；G/D/F 的 delta 逐项等于提案矩阵。

调用与提案矩阵一致且逐阶段实际计入：G `loss_assembler=2`、`production_fk_batch=4`、`kernel64_fk_batch=3`、`production_vjp=2`、`mj_forward=1200`、`mj_jac_site=7200`；D `mj_forward=9344`、`fd_pose_eval=9344`；F `loss_assembler=2`、`production_fk_batch=4`、`kernel64_fk_batch=3`、`production_vjp=2`、`mj_forward=10544`、`mj_jac_site=7200`、`fd_pose_eval=9344`。全链合计 `loss_assembler=4`、`production_fk_batch=8`、`kernel64_fk_batch=6`、`production_vjp=4`、`mj_forward=21088`、`mj_jac_site=14400`、`fd_pose_eval=18688`；合成模式 `scene_setup_forward=0`（真实模式应为 1）。G/D/F 的索引与台账携带同一终态调用快照，三处单元预留均 `consumed_complete=true`。全链零计数均为 JSON 整数 0。

F 过程与单元：

- `recompute_process.start_method=spawned_python_subprocess`，pid 与父进程不同、ppid 为父进程，`exit_reason=ok`。
- 单元 `artifact_read`（父）、`recompute_backend`（父）、`recompute_backend`/`geometry_replay`/`fd_replay`（子）全部 `admitted=true`。
- 父进程按子进程回报的 `call_delta` 记入同一调用台账；子进程异常中断时仍在 `finally` 结算已发生调用；完全无 delta 时预留保持未结算。
- 工件哈希绑定包含 `arrays`、`gradient_arrays`、`rows`（G）与 `arrays`、`rows`（D）；`rows` 文件内容中的 20 条梯度明细同时与主表和 fresh 重推逐字段对账。

R0 记录：`inputs.problems=[]`（形状/dtype/有限/限位顺序/生效常数/域），`checks.input_schema=true`；`parent_consumer_state.checked=false` 与 `locked_assets.checked=false` 是合成模式的显式事实，不伪造真实核验；`runtime` 在合成模式为 `null`（真实模式必须为 applied 事实）。

## 5. 准入包

命令：

```bash
uv run --locked python scripts/audit_w1_real_fk.py --admission \
  --out derived/audit/w1_real_fk_audit/admission \
  --tolerance-contract derived/audit/w1_real_fk_audit/tolerance_contract.json
```

| 项 | 值 |
| --- | --- |
| 包路径 | `derived/audit/w1_real_fk_audit/admission/attempt_013/admission_package.json` |
| 包 SHA-256 | `883dd09bea692b5403a487d1e203c071f5ebbc931f18da7eb011e47c609788bc` |
| `passed` | true（`problems=[]`） |
| 状态 | `authorized=false / registered=false / run=not_run / section_9_2_checked=false` |
| 父运行绑定 | 7/7 文件哈希匹配（注册/准入/p_report/两份数组/终局/独立复核） |
| 固定窗口 | 4/4 与父注册 `window_keys` 一致 |
| 键数 | geometry 122 / fd 4672 / states 32 |
| 全链调用上限 | `loss_assembler=4`、`production_fk_batch=8`、`kernel64_fk_batch=6`、`production_vjp=4`、`mj_forward=21088`、`mj_jac_site=14400`、`fd_pose_eval=18688`、`scene_setup_forward=2` |
| 运行时 | 契约 `TF32 off / highest / deterministic=false / autocast off`；构建进程内应用后 `applied=true`，`observed_after_setup` 五项全部满足 |
| 读取范围 | 仅元数据/文件哈希；未加载窗口数组、未编译资产、未 `torch.load` checkpoint、未创建 cohort 或正式运行根 |

包内 `implementation.files` 与草案/校准文件/证据哈希均与当前字节一致；改动任一被绑定文件会使准入包和真实入口拒绝。`attempt_001`–`attempt_012` 是逐步加入撤回成本、慢持久化、R0 实现漂移门、真实参考事实、审阅修复前实现与草案、两/三轮审阅修复实现、以及同步成本/耗时纪录的历史包；`attempt_013` 为当前权威。

包内 `r0_obligations` 明确列出真实 R0 必须落实的 7 项：资产闭包、父消费者状态、输入 schema/生效常数/域、调用点计数与单元预留、F 独立子进程、G/D 消费者门、生产运行时设置与父子复核。

## 6. 第一轮只读审阅修复（8 P1 + 3 P2）

| # | 级别 | 问题 | 修复 | 反例/证据 |
| --- | --- | --- | --- | --- |
| 1 | P1 | CPU leaf 与真实 CUDA assembler 设备冲突 | `_capture_run` 使用 `assembler.device` 创建全部张量 | `test_capture_binds_tensors_to_the_assembler_device` |
| 2 | P1 | G/D 预算撤销后仍继续并最终通过 | 阶段间重新 `stage_state`；不可消费即停止（数值 mismatch 仍可定位） | `test_budget_revoked_g_stops_the_chain` |
| 3 | P1 | F 无实时截止与逐单元准入 | F 在工件读取/后端/几何/FD 前逐单元 `unit_admission`（子进程内同样执行） | `test_f_refuses_an_unregistered_parent_attempt`、`test_f_deadline_blocks_before_backend` |
| 4 | P1 | FK 标量审计未使用真实 live 标量 | batch 行使用 `run.fk_m2` 并与 live tensor、元素归约三方对账 | `test_shifted_live_scalar_fails_the_batch_row` |
| 5 | P1 | 两份 NPZ 合并掩盖冲突 | 分别核验路径/记录哈希、schema、行结构、重叠数组逐位一致 | `test_f_verifies_gradient_npz_hash_separately`、`test_f_detects_overlap_conflict_between_geometry_and_gradient_npz` |
| 6 | P1 | R0 未落实资产闭包/父消费者/输入 schema | 新增资产闭包、父 R0/P/F 消费者状态、输入 schema/生效常数/域事实 | `test_asset_closure_hashes_every_reference`、`test_parent_consumer_state_requires_complete_rows`、`test_input_schema_facts_reject_bad_shapes` |
| 7 | P1 | 撤回先写可消费终态再查成本 | 先 publishing、核验覆盖后才写 incomplete；未证明即保持 publishing 并返回失败 | `test_withdraw_publishes_before_any_terminal_claim`、`test_withdraw_stays_publishing_when_terminal_write_outruns_budget` |
| 8 | P1 | 生产 FK 调用数预设 +2 | `_CountingFk` 在真实调用点计数；单元上界预留持久化 | `test_production_fk_calls_are_counted_at_the_call_site` |
| 9 | P2 | F 未启动新进程 | F 重算在派生 Python 子进程执行并记录 pid/ppid/python/exit_reason/call_delta | `test_synthetic_self_test_end_to_end` |
| 10 | P2 | clamp 域只查 `+1` 且未传播到梯度 | FD 分段检查改为 `|u±h|` 双侧；解码域缺口使梯度行 `domain_gap` | `test_negative_clamp_boundary_is_a_domain_gap` |
| 11 | P2 | 坏工件在结构校验前 `.get()` 崩溃 | 先校验 JSON/NPZ 容器、行与键类型，异常转为 verification gap | `test_malformed_fd_rows_become_gaps_not_exceptions` |

## 7. 第二轮只读审阅修复（5 P1 + 3 P2）

| # | 级别 | 问题 | 修复 | 反例/证据 |
| --- | --- | --- | --- | --- |
| 1 | P1 | 主 provider 未绑定计数器，G/D 调用漏计（`review10` G/D `call_delta` 全零） | `run_registered_audit` 绑定 provider；每阶段记录真实 delta，零值即记 gap；合成全链与名义值精确相等 | `test_synthetic_self_test_end_to_end`（逐阶段 delta 与全链 `loss_assembler=4`） |
| 2 | P1 | 封口只同步台账，索引保留事前快照，消费者仍通过 | `finish_attempt`/撤回把同一终态快照写入两侧；`stage_state`/`stage_history` 校验一致 | `test_seal_writes_the_same_call_snapshot_to_index_and_ledger`（篡改索引 → `call_balance_disagreement`、历史未知） |
| 3 | P1 | 子进程中途退出丢失已发生调用；未结算预留不被历史消费 | worker `finally` 结算 delta；父进程合并任何 delta；无 delta 时预留保持未结算并阻塞后续额度；`stage_state`/`stage_history`/`prior_call_counts` 视为未知 | `test_worker_settles_known_calls_when_fd_aborts`、`test_f_merges_an_aborted_worker_delta_and_keeps_unsettled_unknown`、`test_prior_calls_unknown_when_a_reservation_is_unsettled` |
| 4 | P1 | 截止只在阶段入口检查，几何/FD 循环内继续执行 | `_ensure_deadline` 在模型批次、参考/核调用与每个 FD 状态前复核 | `test_geometry_deadline_stops_the_second_model`、`test_fd_deadline_stops_before_the_next_state` |
| 5 | P1 | F 未逐字段核验单独保存的梯度明细 | 梯度明细与主表、fresh 重推逐字段绑定；`rows` 文件纳入封口哈希 | `test_f_binds_saved_gradient_rows_to_the_main_table`（`max_abs_error=123` 被拒） |
| 6 | P2 | 两份 NPZ 未分别执行 schema 检查 | 三份 NPZ 分别校验必需键/dtype/shape/数值类型；gradient 文件另与 fresh 分区比较 | `test_f_validates_each_npz_schema_before_merging` |
| 7 | P2 | 缺失 NPZ 在最终哈希阶段抛 `FileNotFoundError` | 使用 `_file_sha`（缺失返回 `None`），缺口在读取阶段登记 | `test_f_missing_npz_reports_a_gap_instead_of_raising` |
| 8 | P2 | 官方入口未落实冻结的 TF32 配置 | `configure_production_runtime` 在真实入口与 F worker 应用；R0 与 F 父子互比；准入包记录契约与应用事实 | `test_production_runtime_is_applied_and_verified`、`test_admission_package_problems`（`admission_runtime_not_applied`） |

两轮修复均未改变校准证据哈希（`8f46fbe1…`）。第二轮权威合成运行在最终模块/草案字节下重新生成。

## 8. 第三轮只读审阅修复（1 P1 + 1 P2）

| # | 级别 | 问题 | 修复 | 反例/证据 |
| --- | --- | --- | --- | --- |
| 1 | P1 | `delta_settled` 只看是否为 Mapping；字段无效或 `counter.add` 超限时仍把预留标为已结算，空/部分 delta 也被结算为零 | 新增 `_worker_call_delta`：要求 delta 覆盖全部 8 类调用、每项为非负 JSON 整数，并与 worker 自身终态计数器快照逐项一致；合并任一字段超限即撤销结算并保持 `consumed=null` | `test_worker_call_delta_requires_a_complete_consistent_snapshot`、`test_f_does_not_settle_an_unmergeable_worker_delta`（已有 31632 再报 10545 → 合并失败、计数不变、预留未结算、历史未知） |
| 2 | P2 | 最终调用矩阵否决只在 F 封口后修改返回值，磁盘摘要仍保留通过声明 | 矩阵判据前移到 `finish_attempt` 之前；不匹配时先把 F 摘要与独立验证重写为 `audit_incomplete`/`complete_evidence=false` 并重绑哈希，再以 incomplete 封口 | `test_call_matrix_mismatch_withdraws_the_disk_claim`（`kernel64_fk_batch` 期望 +3 → 磁盘摘要 `audit_incomplete`、验证 `complete_evidence=false`、`stage_state(F)` 报 `attempt_not_complete`） |

第三轮未改变校准证据哈希；权威合成运行在最终模块/草案字节下重新生成（`review11`），历史 `_prefinal*` 中间运行保留对照。

## 9. 仍未执行与未覆盖

- 未勾选方案 §9.2 九项；未创建真实 cohort/预算/注册；未执行真实 R0/G/D/F。
- 未读取父 `objective_real_arrays.npz` 的数值数组（准入包只哈希文件）、未编译锁定 MJCF/URDF、未加载两份 checkpoint。
- 未审计：真实 float32/CUDA 生产前向、真实 MuJoCo 编译后地址/site 位姿、真实父 FK 逐位绑定、真实 FD、网络参数梯度、累积/裁剪/AdamW/更新、选模、DDIM、端到端动作质量。
- 资产闭包/父消费者状态/输入 schema 的真实分支未执行，只在临时合成树上验证实现；真实资产内容仍以 R0 运行时哈希为准。
- 真实运行时契约（TF32 off 等）只在本机 CPU 环境验证了可应用性；真实 CUDA 前向的数值行为仍待 R0/G 执行。
- `x0_independent` 只做 schema/dtype/有限性与 6D 域检查；`x0`↔`x0_independent` 的编码关系与 `raw_root`/`raw_q`/`raw_wfc` 未在本模块重算，仍由父注册与父审计绑定，本方案不把它当作新参考真值。
- F 的独立性来自新进程/新后端/磁盘重读；参考公式与 G 共用同一套 NumPy/MuJoCo 实现，不是第三套数学证明。
- I/C 的 `scene_setup_forward=0` 是合成事实；真实 G/F 每个 stage 应各为 1，由 R0/G 首个已登记单元核验。
- 真实入口在无授权、无校准、无准入包或运行根不是固定历史根直接子目录时拒绝，且在读取数组/加载资产/创建运行根之前退出（测试覆盖）。

## 10. 复现命令

```bash
uv run --locked pytest tests/test_w1_real_fk_audit.py -q
uv run --locked python scripts/audit_w1_real_fk.py --calibrate-synthetic \
  --out derived/audit/w1_real_fk_audit/tolerance_contract.json
uv run --locked python scripts/audit_w1_real_fk.py --synthetic \
  --out derived/audit/w1_real_fk_audit/synthetic/20261028_review11
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
  --admission-package derived/audit/w1_real_fk_audit/admission/attempt_013/admission_package.json
```

## 11. 边界

`objective_coverage=partial`、`full_objective_contract_passed=null`、`attribution=attribution_insufficient`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false`。旧 U0 `incomplete`、UR `no_quality_gain`、原 F `not_ready`、预测 r6 不可验收、旋转融合不接入均保持不变。真实通过/失败只能由取得授权后的 R0→G→D→F 产生，不能由本记录外推。
