# W1 I 阶段：小集拟合诊断器实现与合成反例验证记录

日期：2026-11-07。方案：[数值缺口有限收口与受控拟合诊断方案 v0.4](../plans/omniretarget_w1_numerics_closure_fit_diagnostic_plan.md) §6/§7（I 阶段交付）。

**性质：纯合成实现与验证交付。本记录不含任何真实数据读取、真实 checkpoint、真实模型/优化器构造、真实 forward/backward/FK/AdamW 或真实训练更新；全部模块与测试在合成世界（合成 tokenizer/关节限位/教师窗/15-site FK 替身）上离线运行，不声称与生产 CUDA/foreach 等价。** 真实 R 冻结与真实 D1/D6/F 执行仍须方案 §8.3 的新冻结材料与新明确授权，本交付不创建 cohort/attempt/授权，不恢复旧 T6 一次性授权。

## 1. 交付物

| 文件 | 内容 |
| --- | --- |
| [src/ireset/eval/w1_smallset_fit_diagnostic.py](../../src/ireset/eval/w1_smallset_fit_diagnostic.py) | I 阶段独立诊断模块：FitPlan/FitRegistration、StageBudget、运行时硬门、`metric_decode_v1`、§7.1/§7.2 物理指标、probe 机制、FitDriver（复用已审计 Trainer 更新单元）、§7.4 第 1–3 项本地核验、D1→D6 交接、F 逐位重放、attempt 台账、R 只读元数据守卫、35 项合成反例电池、合成验收运行器 |
| [scripts/verify_w1_smallset_fit_diagnostic.py](../../scripts/verify_w1_smallset_fit_diagnostic.py) | 合成验收 CLI（唯一入口，synthetic-only）：微型 D1→本地核验→交接判定→（D1 过门时）微型 D6→F 逐位重放→反例电池；t0 先于重导入、独占 attempt 目录、封口发布-核验、返回覆盖证明、fail-closed 退出码 |
| [tests/test_w1_smallset_fit_diagnostic.py](../../tests/test_w1_smallset_fit_diagnostic.py) | 16 项测试：冻结常量、注册校验、解码/公式/尖峰、状态分层校验器、台账/R 守卫/参数契约、D6 数据分母、微型阶段+F 逐位重放、交接门、预算停止，及完整 35 项反例电池 |

## 2. 复用边界（方案 §6.1）

驱动**只通过已审计 Trainer 更新单元**执行训练：每个 micro 经 `Trainer.accumulate_micro`（内部 `_forward`→`predict`→损失→`(loss/N).backward`+有限性检查），裁剪经 `Trainer.clip_gradients`，更新经 `Trainer.step_optimizer`；AdamW/loss 装配均由 Trainer 构造，驱动不重写任何前向/反向/优化器算术。共享单元本身不查截止时间——驱动在每个单元边界（step/micro 开始、加载后、pre_backward 钩子、backward 后、clip、step、probe、checkpoint、写盘）插入 `StageBudget.check`，并把 `work_deadline` 传给 Trainer 构造器使 `save()` 的内部检查同口径。`scripts/w1_s0.py` 的独立手写更新循环**未被沿用**；probe 语义（专用 generator、固定 t/noise、eval+no_grad、RNG 保存恢复）参考 S0 并经由 `Trainer.predict` 执行。

## 3. 冻结契约（I 阶段确定，R 阶段绑定）

- **FitPlan 冻结默认值**（非 synthetic 计划偏离即拒绝）：steps=1000、probe_every=10（101 次 probe，step 0 与 10..1000）、grad_accum=4、D1 micro_batch=1 / D6 micro_batch=6、seed/val_seed=20261008、lr=1e-4、weight_decay=1e-5、grad_clip=1、d240/4层/6头/time_window31；`frozen_plan_defaults_problems()` 由 CLI 与测试断言为空。
- **FitRegistration**：D1 恰 1 窗、D6 恰 6 窗、规范字典序、start=0、跨度=T_WINDOW=60、split 全为 train；驱动另做**注册↔物化数据集窗键绑定**（`dataset_registration_window_mismatch`），并绑定 trainer 配置与计划（steps/micro/grad_accum/lr/wd/clip/seed/架构）。
- **metric_decode_v1**（域 `cpu_float64_from_saved_x0`）：对保存的 FP32 x0 先做合法性核验（dtype/形状 (60,38)/有限性/rot6d 非退化），再精确提升为 CPU float64；mean/std/lo/hi 取生产实际生效 FP32 常数的精确提升；6D 解码复用 dtype 泛型的 `rot6d_to_mat_torch`；关节走干净限位映射（clamp[-1,1]）。固定保存根位置 (60,3)、根旋转**矩阵** (60,3,3)、29 关节角 (60,29)，全部冻结 clip 系。
- **保存记录表**（冻结形状/dtype）：每窗每 probe 12 个数组（raw_x0_pred/prod_*/fk_pred/md_*/err_*），教师参考每阶段一份（raw_x0_tgt/fk_tgt 等 8 类）；npz 键前缀 `w{b}__`；`artifacts_index.json` 登记全部文件 SHA-256+字节。
- **§7.1 四门**：root_position_rmse_m≤0.01、root_rotation_mean_rad≤π/180、joint_rmse_rad≤0.05、teacher_fk_mpjpe_m≤0.01；度量归约 float64；根/旋转/关节只消费重解码量，FK MPJPE 只消费保存的生产 FP32 FK 点。**逐窗四门合取**才记 `window_mean_fit_passed`；宏平均仅展示且 `macro_is_not_a_gate=true`。
- **§7.2 尖峰可见性**：每窗保存逐帧根位置距离、逐帧根旋转角、逐帧×关节绝对误差、逐帧×site FK 距离；p95（linear 分位）+ max（并列取最小帧号、再按关节/site 固定序取首项）+ 最坏坐标；`spike_gate_status="not_evaluated"` 冻结——平均门通过不宣称窗内尖峰已排除。
- **§7.6 状态分层**：阶段报告只发布 `local_complete_evidence`/`local_mean_fit_passed`/`local_conclusion` 与 `final_verification_status="pending_F"`；`local_report_problems()` 校验器拒绝任何未加前缀的 `complete_evidence`/`mean_fit_passed`/`replay_passed` 字段。
- **调用矩阵**：驱动按计划推导期望矩阵（D1：4000 训练+101 probe=4101 次/路径 conditioner/backbone、4000 loss、FK 批调用=2×micro+probe窗+教师窗、metric_decode=窗×(probe+1) 等），run 结束时**实际计数（钩子/包装实测，非名义）与期望逐项相等**，不符即 `local_complete_evidence=false`。

## 4. 运行时硬门（§6.3）

- `ProductionRuntimeGuard` 导入并调用 `w1_real_fk_audit.configure_production_runtime()`（TF32 off、float32_matmul_precision=highest、autocast off、deterministic=false 的绑定字节），每单元 `check_unit` 重采样 `environment_facts` 并按 `RUNTIME_REQUIREMENTS` 逐项比对，漂移即 `runtime_drift` 停止。
- 构造/strict 加载后、下一构造或 forward 前执行 `parameter_contract_problems`：全部浮点参数**与缓冲**的 dtype/device 必须匹配冻结契约（真实路径 cuda/float32；合成路径 cpu/float32），错误即停、不静默 cast。
- 合成路径使用 `SyntheticRuntimeGuard`：只记录单元数，明确 `production_equivalence_claimed=false`，不冒充生产运行时。

## 5. 反例电池（§6.2/v0.3/v0.4 全清单；首轮 35 项 + 第二轮 6 项 + 第三轮 5 项 + 第四轮 4 项 + 第五轮 2 项 + 第六轮 6 项 + 第七轮 5 项 + 第八轮 5 项 + 第九轮 8 项 = 76 项）

| 方案要求 | 反例 ID（捕获断言） |
| --- | --- |
| 注册名单/窗键篡改 | `registration_tamper`（结构校验+注册↔数据集窗键绑定） |
| 少窗/重窗/val-test 混入 | `missing_window` / `duplicate_window` / `val_test_mixing` |
| target 泄漏进 conditioner | `conditioner_target_leak`（访问追踪守卫：直接单元+端到端，x0/监督掩码即拒） |
| drop_last 空 loader | `empty_loader_drop_last`（数据集窗数绑定+首单元 StopIteration 拒绝） |
| 空/缺项/非有限指标 | `nonfinite_prediction_stops`（NaN 预测即停）+ `saved_metric_missing_or_nonfinite` |
| 参数未进联合 clip | `param_outside_joint_clip`（clip 触发时逐参数 |post|=coef×|pre| 核验） |
| 跳过 micro/更新多计少计 | `call_matrix_miscount` |
| probe 改变训练 RNG | `probe_uses_training_generator`（训练 generator 身份拒绝+全局 RNG 前后相等记录） |
| checkpoint/root_stats 错绑定 | `checkpoint_root_stats_misbinding` |
| 工作截止后继续构造/forward/backward/step | `work_after_deadline_stops`（受控时钟） |
| 末次写盘越界仍声明 complete | `complete_claim_after_deadline_withdrawn`（封口撤回+成本只升不降） |
| R 偷读数组值 / 阶段间物化未计费 | `r_metadata_array_read_forbidden` / `unbilled_materialization_refused` |
| SIGKILL 遗留 running | `sigkill_running_row_stops`（未知历史=停止） |
| 失败后新 attempt 重置预算 | `new_attempt_budget_not_reset` |
| F 身份失败仍执行模型 | `f_identity_failure_no_forward`（跳过 fresh 路径且 build_modules 零调用） |
| 总 loss 下降但单窗未过门 | `window_gate_not_masked_by_macro` |
| RMSE/MPJPE、米/厘米、弧度/度混用 | `metric_formula_and_units`（解析用例逐公式核对） |
| 宏平均掩盖单窗失败、60° 单帧尖峰 | `rotation_spike_visibility`（方案原例：均值过 1° 门而 max=60°@frame17 保持可见） |
| 坐标标记缺行/offset 错/截断 | `metric_schema_row_offset_truncation`（冻结记录表逐记录形状/dtype/有限性） |
| v0.3：保存与 fresh 均过门但数组不同 | `fresh_gate_pass_but_bitwise_mismatch`（1-ULP 扰动+自洽重导出后 F 仍 `replay_mismatch:x0`） |
| 只篡改中间 probe 曲线/p95/max | `intermediate_curve_tamper` |
| 伪造 D1 通过布尔 | `d1_forge_pass_boolean` |
| initial/last 角色或 step 对调 | `initial_last_role_swap` |
| TF32/autocast/运行时漂移 | `tf32_runtime_drift`（真实 `runtime_facts_problems` 逐单元） |
| 构造后 dtype/device 错误仍 forward | `param_dtype_device_mismatch`（preflight 拒绝，零 forward） |
| D6 逐窗六次记一次 probe | `probe_forward_count_violation`（每 probe delta 必须 1+1） |
| 重放不一致后放宽/补跑/发布 mean_fit | `replay_mismatch_no_loosening`（`bitwise_no_tolerance`、mean_fit=null、无重试路径） |
| v0.4：D1 无 F 合法交接但不发最终验收 | `d1_handoff_cannot_publish_final`（pending_F+无最终字段+无授权拒交接） |
| v0.4：F 失败不回写/不洗本地 | `f_failure_preserves_local_facts`（F 失败后阶段报告 SHA 不变） |
| v0.4：step-N metric_decode 篡改同步仍拒 | `decode_tamper_synced_rejected`（原始 x0 重解码逐位绑定） |
| v0.4：中间 FK 自洽替换保留限制 | `fk_self_consistent_replacement_keeps_limitation`（核验通过但 `intermediate_fk_state_binding_status="not_evaluated"` 不升级） |
| 第九轮：checkpoint 有效配置/身份语义绑定 | `checkpoint_config_binding_rejected`（重签 `lr=123`/`grad_accum=99` 被 14 字段冻结配置逐项对账拒绝）+ `checkpoint_identity_binding_rejected`（外来 identity 对阶段声明身份拒绝） |
| 第九轮：D 内部多窗单元链越截止 | `teacher_window_units_deadline_stop` / `probe_window_units_deadline_stop`（首窗解码越线后逐窗生产解码/FK/归约零启动）+ `training_micro_internal_deadline_stop`（conditioner 越线→backbone/loss/FK 零启动） |
| 第九轮：F 身份 helper 的 checkpoint 读取 | `f_identity_checkpoint_read_deadline_stop`（入口与两次读取之间均守卫，越线零晚读取、零构造） |
| 第九轮：已完成 conditioner 计数保留 | `f_completed_conditioner_counter_kept`（backbone 被拒时 conditioner=1 保留在返回与盘上报告） |
| 第九轮：嵌套坏结构结构化收口 | `nested_probe_schema_containers_report_incomplete`（step 字符串/window_id 列表/forward_counts 列表/多一窗四形态均落 f_report+返回证明） |

## 6. 测试结果

- `tests/test_w1_smallset_fit_diagnostic.py`：**17 passed**（CPU，离线；含完整 76 项电池 `test_counterexample_battery_all_caught`）。
- 反例电池独立运行：**76/76 caught**，ID 集合与冻结清单精确相等（§16 第九轮审阅修订后）。
- N0/N1 回归：**33 passed**（N0 `verify_return_coverage` 的加法性 `final_doc` 返回不改变 N0/N1 行为）。

## 7. 权威合成验收运行

**`derived/audit/w1_smallset_fit_diagnostic/attempt_010/`**（2026-11-09，§16 第九轮审阅修订后代码，独占创建，CLI exit 0）：单一单调时钟 t0 先于重导入，硬截止 3300s/预留 300s；summary 经发布-核验封口 `sealed_within_budget`（round 1，登记成本 **1468.88s** 覆盖封口、候选写盘、必要回读与返回）+ `return_observation.json` SHA 绑定 covered=true（elapsed 1464.88 ≤ 1468.88）。`attempt_001–004`、`attempt_006–009`（修复前/二轮/三轮/四轮/五轮/六轮/七轮/八轮代码）保留为被取代证据；`attempt_005` 为第五轮修复运行中途实现字节再次变更而终止的残缺目录（无 acceptance_summary/return_observation，不消费）。

- **微型 D1**（300 步/probe_every 30/lr 0.03，d16 合成世界，登记成本 235.34s，返回覆盖 covered=true（elapsed 231.34））：`local_complete_evidence=true`、调用矩阵实际=期望零偏差、本地核验零问题（checkpoint 逐项配置/身份绑定、forward/backbone、strict 加载、逐次读取、解码/FK、checkpoint 读取均在逐单元截止守卫下通过）；教师与预测双侧 metric_decode **重解码逐位绑定**、生产 FP32 解码量**重推导逐位绑定**；step-300 四门全过（root RMSE **0.00115m**、root 旋转均值 **0.00288rad**、joint RMSE **0.00186rad**、teacher FK MPJPE **0.00121m**）→ `local_mean_fit_passed=true`、`complete_fit`；`handoff_candidate.json` 落盘且经 D6 入口消费 → `handoff_ready=true`（problems=[]，交接声明前复查通过）。
- **微型 D6**（300 步/lr 0.02，六注册窗，登记成本 1388.25s，返回覆盖 covered=true（elapsed 1384.25））：覆盖行多重集精确、**六窗四门全过**（逐窗 `window_mean_fit_passed` 全 true，macro RMSE 0.00633m，`macro_is_not_a_gate` 显式登记）、`local_mean_fit_passed=true`。
- **F**：返回证明前置核验通过；对两阶段各 fresh 构造+initial/last 各一次契约核验后 strict 加载 probe（每阶段恰 2 次 fresh 预测）；输入重建逐次读取、conditioner→backbone、strict 加载、教师重读、fresh 解码/FK、指标归约与 checkpoint 读取均在逐单元截止守卫下通过；原始 x0、生产 FP32 解码量与 FK 点对保存数组**逐位一致**，教师侧从原始 x0 重解码绑定后对账；完成声明门 `claim_deadline_exceeded` 本轮未触发；F 自身对 f_report.json 的封口后回读执行**独立返回覆盖核验**（covered=true）→ `replay_passed=true`、`complete_evidence=true`、`mean_fit_passed=true`、`verdict=fit_pass`、`tolerance_policy=bitwise_no_tolerance`。
- **反例电池**：76/76 caught（ID 集合与冻结清单精确相等，含 §16 新增 8 项）。

历史运行 `attempt_001`（修复前代码）、`attempt_002`（§8 修订后）、`attempt_003`（§10 修订后）、`attempt_004`（§11 修订后）、`attempt_006`（§12 修订后）、`attempt_007`（§13 修订后）、`attempt_008`（§14 修订后）、`attempt_009`（§15 修订后）保留为被取代证据：其证据分别不含后续轮次的修复项（教师解码绑定/必需工件集合 → 证明预算绑定/候选时序 → 回读入窗/F 自身覆盖/声明门 → 保存量重算截止守卫/阶段停止收口/撤回成本同步 → 驱动与交接预算守卫/F 端点边界/撤回最终文档快照 → helper 内部逐单元守卫/交接声明复查/驱动截断计数 → forward 与 strict 加载边界/输入重建与端点逐读/坏容器结构化收口 → checkpoint 配置/身份语义绑定/D 内部单元链/F 身份读取/计数边界/嵌套坏结构），不可消费为本轮结论。**合成四门通过只证明 I 阶段管线机制与门实现本身在合成世界上自洽；不外推到真实 W1 训练窗（真实可达性属 R/D1/D6 的授权后问题）。**

## 8. 第二轮独立审阅修订（2026-11-08：7 P1 + 2 P2 全部修复）

第一轮交付经独立审阅否决"完整收口"（7 P1 + 2 P2；审阅复现脚本留存于 /tmp/opencode/）。全部接受并修复，反例电池扩至 **41 项**（`tests/` 16 passed；电池独立运行 41/41 caught），审阅者的两份攻击脚本在修复后代码上全部呈现 fail-closed 结果。`derived/audit/w1_smallset_fit_diagnostic/attempt_001/` 自本轮起为**被取代证据**（其报告含 §8 修复前的 `macro.local_mean_fit_passed=false` 事实错误与零 `n_micro` 历史），权威验收重跑为 attempt_002。

| # | 审阅发现 | 修复 |
| --- | --- | --- |
| P1-1 | 教师解码未绑定原始 target：教师派生量替换为末次预测并同步曲线/哈希后可制造 `fit_pass` | 本地核验与 F 均从保存的**原始教师 x0** 重解码并**逐位绑定** `md_pos/md_R/md_q`（`teacher_decode_binding_mismatch`）；绑定失败的窗不得消费任何派生量（无 metrics、不入 final_windows）。生产 FP32 解码量（教师与每 probe 的 `prod_*`）经同一生产 `decode_clip` 从原始 x0 **重推导并逐位绑定**（`production_decode_binding_mismatch`；解码非模型/FK/优化器，属 §7.4 第 2 项重算允许范围）。FK 点（`fk_pred/fk_tgt`）仍为生产来源登记+度量自洽，新增显式 `teacher_fk_state_binding_status="not_evaluated"`——FK 单独替换无法翻转合取（三个解码侧门先拒） |
| P1-2 | D1→D6 与 F 未完整消费封口/返回证明 | `resolve_handoff` 现在核验：seal 终态、`return_observation.json` 存在/covered/**summary_sha256 与报告字节绑定**/elapsed≤registered≤cap 链、**交接候选存在且绑定当前报告 SHA、其本地事实为 complete+pass**（`finish_stage` 对 D1 落盘 `handoff_candidate.json`）；F 每阶段前置同一返回证明核验（缺失/零 SHA/覆盖链断裂即拒绝 fresh 路径） |
| P1-3 | 空工件索引被当作完整身份核验；报告登记的索引 SHA 未消费 | `verify_artifacts_index` 增**必需工件集合**（教师参考/probe 输入/probe 索引/全部 probe npz/两 checkpoint；缺一即 `artifact_required_missing`）；`verify_stage_local` 消费阶段报告登记的 `artifacts_index_sha256`（不符即 `artifacts_index_sha_mismatch`）；索引恒排除三个晚置文件（报告/返回证明/候选——它们由 SHA 字段互相绑定，避免循环） |
| P1-4 | checkpoint dtype 加载前未检查，strict load 静默转换 | 本地核验对 checkpoint 内全部浮点张量断言 **dtype==float32**；F 新增 `_load_state_dicts_checked`——**复制前**对完整键集/shape/dtype/有限性按 fresh 模块契约核验，不符即拒绝（零 load、零 forward、build_modules 可为零调用） |
| P1-5 | 运行时硬门未覆盖每个 micro/probe | 驱动在每个 **micro 开始、pre_backward 钩子、micro backward 后、clip 前、optimizer step 前、probe 开始、checkpoint 写盘**处调用 `runtime_guard.check_unit`；审阅者的真实守卫+mock facts 攻击现于 `step_1_micro_1_backward` 即停（`unguarded_train_micros=0`、backward_calls=1、adamw=0） |
| P1-6 | F 加载越过工作截止后仍启动 forward | F 在 **checkpoint 加载后**（`f_probe_{role}_loaded`）与 **probe 输入重建后**（`_inputs_ready`）及 fresh 完成后各插入 `budget.check`；审阅者的慢加载+受控时钟攻击现于加载后即停（fresh forwards=0） |
| P1-7 | CLI 在返回覆盖核验之后才输出 | CLI 重排为：**先 flush stdout**（盘上事实载荷）→ 再 `verify_return_coverage`（输出耗时纳入被检窗口）→ 失败分支 stderr 诊断后**二次核验**（覆盖诊断耗时）→ 按最终盘上状态决定退出码；审阅者的 1000s 输出阻塞攻击现 exit 3、盘上 `budget_exceeded_after_seal`/proof covered=false |
| P2-8 | 宏聚合合取字段恒为 false | `aggregate_local_verdicts` 改从 `w["metrics"]` 读取；补全过控制（单窗集合合取=true） |
| P2-9 | 逐步 n_micro 恒为 0 | 驱动每 micro 递增 `n_micro`（历史现记录每步恰 grad_accum 次） |

新增 6 项电池反例：`teacher_derived_replacement_rejected`（教师=末次预测的全链自洽重签替换）、`return_proof_binding_rejected`（零 SHA 证明拒交接+删除证明拒 F）、`empty_artifacts_index_rejected`（空索引+报告 SHA 失配）、`checkpoint_dtype_tamper_rejected`（float64 checkpoint 加载前拒绝）、`runtime_drift_mid_micro_stops`（micro 间漂移停机）、`f_forward_after_deadline_refused`（加载越界零 forward）。

## 9. 边界与不主张

- 本交付只覆盖 I 阶段退出条件（复用实际 Trainer 更新单元；身份/计费/数据分母测试通过）。**R（新冻结/授权材料）、真实 D1/D6/F 均未实施、未授权、未运行**；本模块不含真实数据入口，CLI 拒绝一切真实输入。
- 合成 D1/D6 的四门通过只证明管线机制与门实现本身，**不说明真实 W1 训练窗可达这些目标**（真实 S0 历史只支持训练/normalized 误差下降）。
- `spike_gate_status="not_evaluated"` 冻结；平均拟合通过不排除窗内尖峰，不代表泛化、严格碰撞、持续抓握、生产接入或端到端加速。
- 中间 FK 只保存生产来源登记/工件哈希与保存点距离的度量自洽核验；`fk_replay_scope="initial_last_only"`，不宣称全部 101 次物理/FK 链已独立验证。
- F 逐位重放是事前验收要求而非已证明的可重现性结果；真实 CUDA 路径首现合法漂移时按 §7.4 第 5 项 `replay_mismatch` 收口。
- T6 `audit_incomplete`/0 failures/539 gaps、旧 629 分类、cohort 34 次更新/1347.195507s、一次性授权已消费、`partial/null/false/not_ready/不接入` 全部保持。
- 本轮未提交 git、未运行全项目测试套件（仅运行本交付测试文件与方案相关既有文件的冒烟）。

## 10. 第三轮独立审阅修订（2026-11-08：5 P1 + 1 P2 全部修复）

第二轮修订经独立复审确认有效（attempt_002 的 SHA/宏合取/`n_micro=4` 已核实），但新增隔离反例发现 5 P1 + 1 P2（复现脚本 `/tmp/opencode/review_i_round2_20261108.py`）。全部接受并修复，电池扩至 **46 项**（测试 17 passed），审阅者脚本在修复后代码上全部呈现 fail-closed。attempt_002 自本轮起为被取代证据，权威验收重跑为 attempt_003。

| # | 复审发现 | 修复 |
| --- | --- | --- |
| P1-1 | 返回证明的预算数值未与权威报告对账（自洽的 700≤800≤900 证明在报告 cap=600 时仍通过 F 与交接） | `_return_proof_problems` 现在读取**封口报告**的 `seal.measured_wall_s_at_summary` 与 `seal.wall_cap_s` 并**逐项绑定**证明的 `registered_measured_s`/`wall_cap_s`（`return_observation_registered_mismatch`/`return_observation_wall_cap_mismatch`）；覆盖链用报告的权威值复核 |
| P1-2 | 交接候选写盘发生在最后一次返回覆盖核验之后（慢候选写盘不可见） | `finish_stage` 重排为 **seal → 候选写盘 → 最终 `verify_return_coverage`**：覆盖采样落在候选写盘（含其必要回读）之后；注入候选写盘 1000s 延迟的反例现以 `budget_exceeded_after_seal`+covered=false+handoff 拒绝收口 |
| P1-3 | CLI 成功分支的新增最终回读再次留下未计费尾部 | CLI 时序改为 **flush stdout → 回读 summary → 最终覆盖采样**，采样后零 I/O；注入回读 1000s 延迟的 CLI 用例（pytest，镜像审阅者注入点）现 exit 3、盘上撤回、登记成本 ≥1000s |
| P1-4 | F 前置核验越过工作截止后仍构造模型 | 构造移入守卫区：`budget.check(f_stage_*_pre_construct)` 在 `build_modules()` **之前**执行；审阅者的慢核验+受控时钟攻击现 `construct_events=[]`（零构造）、零 forward，于 `f_stage_D1_pre_construct` 即停 |
| P1-5 | F 输入重建后的 fresh 调用仍缺运行时硬门 | `_inputs_ready` 处**同时**执行 `budget.check` 与 `runtime_guard.check_unit`；fresh 返回后同样双检（`_fresh_complete`）；审阅者在 last 输入读取期间注入 TF32 漂移的攻击现 `bad_forwards=0`、`complete_evidence=false` |
| P2-6 | 可解析但非对象的返回证明使 F 崩溃（`[]` → AttributeError，无 f_report） | 新增 `_read_json_object`：解析成功但非 Mapping 一律转为结构化失败（`*_not_a_mapping`）；F 的报告/证明读取、`resolve_handoff`、`verify_stage_local` 的报告读取全部改用该守卫；list 形证明现以 `return_observation_not_a_mapping` 拒绝且 f_report.json 正常落盘（verdict=incomplete） |

新增 5 项电池反例：`proof_budget_binding_rejected`（自洽伪造 700/800/900 证明拒 F+拒交接）、`candidate_write_after_covered_refused`（候选写盘注入延迟→撤回+拒交接）、`f_construct_after_deadline_refused`（构造前零调用）、`f_runtime_drift_before_fresh_refused`（inputs_ready 漂移→零 fresh forward）、`proof_not_a_mapping_survives_f`（list 证明结构化拒绝）；新增 1 项 pytest 用例 `test_cli_output_and_reread_inside_return_coverage`（CLI 回读注入 1000s → exit 3 撤回）。

**权威合成验收 `derived/audit/w1_smallset_fit_diagnostic/attempt_003/`**（2026-11-08，第三轮修复后代码，独占创建，工件 7.0MB，CLI exit 0）：封口 `sealed_within_budget`、登记成本 **1504.31s** ≤ 3300s、返回覆盖 covered=true；微型 D1（300 步，wall 240.4s）四门全过（RMSE 0.00115m / 0.00288rad / 0.00186rad / FK 0.00121m）、`handoff_ready=true`（候选经 D6 入口消费）；微型 D6（wall 1472.5s）六窗四门全过（macro RMSE 0.00211m）；F 两阶段逐位重放通过（`fit_pass`）；46/46 反例捕获。`attempt_002` 保留为被取代证据（其证据不含第三轮的证明预算绑定/候选时序/构造前截止/输入期运行时双检修复项）。**attempt_003 自第四轮修订（§11）起保留为被取代证据**（其证据不含回读入窗/F 自身覆盖/声明门修复项）。

## 11. 第四轮独立审阅修订（2026-11-08：3 P1 + 2 P2 全部修复）

第三轮修订经独立复审确认有效（attempt_003 摘要 SHA/返回证明/登记成本 1504.31s≤3300s/46 项 caught/D1·D6·F 正向状态已核实），但隔离反例发现 3 P1 + 2 P2（复现结果 `/tmp/opencode/review_i_round3_20261108_results.json`）。全部接受并修复，电池扩至 **50 项**（测试 17 passed），审阅者的全部攻击形态在修复后代码上呈现 fail-closed。attempt_003 自本轮起为被取代证据，权威验收重跑为 attempt_004。

| # | 复审发现 | 修复 |
| --- | --- | --- |
| P1-1 | 阶段最终回读仍在返回覆盖采样之后：仅对最后一次报告回读注入延迟（实际耗时 1000s）时，登记仍 5s、证明仍 covered=true、D6 交接仍 handoff_ready=true | `finish_stage` 重排为 **seal → 候选写盘 → 必要回读 → 最终 `verify_return_coverage`**：全部普通 I/O（含封口后的报告回读）都落在最终采样之前，**通过采样后成功路径零 I/O**；覆盖失败时盘上由 `verify_return_coverage` 依确定性契约撤回（成本只升不降），返回视图镜像该撤回（`seal.status=budget_exceeded_after_seal` + 撤回字段）而非再花费采样后 I/O 重读；注入回读 1000s 延迟的新反例现撤回+covered=false+handoff 拒绝 |
| P1-2 | F 最终报告回读没有返回覆盖核验：封口后回读耗至 1000s 仍返回登记 5s、`sealed_within_budget`、`fit_pass`（只依赖外层总预算） | F 尾部重排为 **seal → 必要回读 → F 自身的 `verify_return_coverage`**（撤回字典 `F_WITHDRAW_ON_EXCEED`）；返回值携带 `seal`/`return_coverage`，覆盖失败时返回视图镜像撤回（verdict=`budget_exceeded_after_seal`、`complete_evidence=false`）——验收事实 `facts["f"]` 新增 `return_coverage` 字段；注入 F 报告回读 1000s 延迟的新反例现 F 自身 covered=false、verdict≠fit_pass |
| P1-3 | F 可在收口预留内补完数值核验：最后端点教师核验返回时把时钟推进到工作截止后（cap 内），fresh 指标解码/指标重算照常执行并发布 `fit_pass`、`budget_exceeded=false` | fresh 指标解码成为独立数值单元：教师核验返回后、`decoder.decode` 前**每窗插入 `budget.check(f_fresh_metrics_{role}_w{b})`**（BudgetExceeded 由既有 `except FitDiagnosticError` 捕获→`fresh_replay_error`→incomplete）；**完成声明前新增声明门**——`claim_ok = clock < work_deadline`，不满足则 `claim_deadline_exceeded=true`、`complete_evidence=false`、`mean_fit_passed=null`、`verdict=incomplete`（replay 事实保留），budget facts 记 `budget_exceeded=true/stopped_at=f_final_claim`；审阅者攻击（截止+91s、cap 内）现收口为 incomplete，非 fit_pass |
| P2-4 | 候选慢写盘电池用例可空洞通过：`StageBudget` 用从零起的假时钟而封口/覆盖函数仍用真实 `perf_counter`，延迟注入前就已超限；刻意恢复错误顺序仍记 caught | **统一时钟**：`finish_stage`/F 尾部/`write_acceptance_outputs`/电池封口用例全部显式传 `clock=budget.clock`（N0 封口/覆盖函数与预算同一时间线）；新增**正向控制例** `candidate_write_chain_control_passes`——先断言未注入延迟时同一 seal→候选→回读→覆盖链闭合干净（sealed+covered+候选落盘），注入例的 caught 以控制例成立为前提，空洞通过被封死 |
| P2-5 | CLI 回读测试未精确命中最终回读：钩子匹配所有 summary 读取，首次延迟发生在更早的 `write_acceptance_outputs`，旧错误 CLI 也通过全部新断言 | 测试钩子改为**精确帧判定**：仅延迟 CLI 自身（`main` 调用帧）对 acceptance_summary.json 的读取，`verify_return_coverage`/`_write_summary_sealed`/`write_acceptance_outputs` 的内部读取不延迟；并新增**回退变体对照**——把源码重排为"先覆盖采样、后回读"（round-2 缺陷形态）后同一钩子下 exit 0，证明当前实现的 exit 3 断言确实锁住该时序（回归即翻转失败） |

新增 4 项电池反例：`candidate_write_chain_control_passes`（无注入控制例：同一时钟下 seal/覆盖/候选链闭合干净）、`stage_report_reread_inside_coverage_window`（P1-1 攻击形态：慢报告回读→撤回+covered=false）、`f_claim_deadline_refused`（P1-3 攻击形态：教师核验返回后时钟越过工作截止（cap 内）→fresh 指标解码被 `f_fresh_metrics_last` 拒绝、`claim_deadline_exceeded=true`、verdict=incomplete、F 自身覆盖仍 covered=true）、`f_report_reread_covered_by_f`（P1-2 攻击形态：慢 F 报告回读→F 自身覆盖翻转为撤回）；重写 1 项 pytest 用例 `test_cli_output_and_reread_inside_return_coverage`（精确帧钩子+回退变体对照）。

**权威合成验收 `derived/audit/w1_smallset_fit_diagnostic/attempt_004/`**（2026-11-08，第四轮修复后代码，独占创建，CLI exit 0）：封口 `sealed_within_budget`（round 1）、登记成本 **1431.43s** ≤ 3300s、返回覆盖 covered=true（elapsed 1427.43 ≤ 1431.43）；微型 D1（300 步，登记成本 235.47s）四门全过（RMSE 0.00115m / 0.00288rad / 0.00186rad / FK 0.00121m）→ `handoff_ready=true`（候选经 D6 入口消费）；微型 D6（登记成本 1396.05s）六窗四门全过（macro RMSE 0.00633m）；F 两阶段逐位重放通过（`fit_pass`、`claim_deadline_exceeded` 未触发、F 自身返回覆盖 covered=true）；50/50 反例捕获。`attempt_003` 保留为被取代证据（其证据不含第四轮的回读入窗/F 自身覆盖/声明门/统一时钟修复项）。**attempt_004 自第五轮修订（§12）起保留为被取代证据。**

## 12. 第五轮独立审阅修订（2026-11-08：1 P1 + 2 P2 全部修复）

第四轮修订经独立复审确认有效（attempt_004 摘要/证明绑定、登记成本 1431.43s、D1/D6 保存量重算与交接成立，50 项均 caught），但新夹具复现 1 P1 + 2 P2（复现结果 `/tmp/opencode/review_i_round4_20261109_results.json`、`/tmp/opencode/review_i_round4_fresh_fixture_20261109_results.json`）。全部接受并修复，电池扩至 **52 项**（测试 17 passed），审阅者攻击形态在修复后代码上 fail-closed。attempt_004 自本轮起为被取代证据。

| # | 复审发现 | 修复 |
| --- | --- | --- |
| P1 | F 调用保存量核验（`verify_stage_local`）未向内部循环传递截止守卫：首个解码返回时把时钟推进到工作截止后（cap 内），随后仍执行 3 次规范解码、4 次生产解码、3 次指标归约（声明门只降级结论，未阻断工作） | `verify_stage_local` 新增 `budget: StageBudget | None` 参数：**教师解码（`saved_teacher_decode_*`）、教师侧生产解码（`saved_teacher_prod_decode_*`）、预测侧生产解码（`saved_probe_prod_decode_*`）、预测解码（`saved_probe_decode_*`）、指标归约（`saved_metrics_*`）**逐单元 `budget.check`；F 调用传入同一预算并捕获 `BudgetExceeded` → 该阶段收口为 `stage_deadline_stop`（`deadline_stop` 记录具体检查点）→ **停止本阶段与全部后继阶段**；`resolve_handoff` 等未传预算的调用点行为不变（默认无检查） |
| P2-2 | 双阶段 F 超时跳过最终报告：D1 fresh 返回越过工作截止后异常被捕获，继续进入 D6 时阶段入口检查在异常收口区之外抛出未捕获 `BudgetExceeded` → 没有 `f_report.json`、也没有返回证明 | 阶段入口检查改为**非抛出预检**（`clock >= work_deadline` 时记录 `stage_deadline_stop` 并 `break`，不启动后继阶段）；保存量重算的 `BudgetExceeded` 同样按阶段收口并 break；F 尾部封口、返回覆盖核验与 incomplete 报告（含已执行计数与 `budget` 成本）在所有路径上照常执行 |
| P2-3 | 撤回后的返回视图成本仍旧：`finish_stage` 与 F 尾部只镜像撤回标志、未同步最终成本——慢回读反例中盘上登记 1007s、返回 `summary_doc` 仍 5s、返回顶层 `seal.status` 仍 `sealed_within_budget`（返回契约自相矛盾） | 新增 `_coverage_return_view`：撤回时从覆盖函数返回的 `registered_measured_s`（= 盘上最终登记成本）与 `elapsed_s` **在内存中重建最终文档快照**（seal.status=`budget_exceeded_after_seal`、terminal、`measured_wall_s_at_summary`/`budget.measured_wall_s_at_summary`=最终成本、撤回字段与 note），返回顶层 `seal` 与 `summary_doc` **均与盘上一致**，且不新增采样后 I/O；`finish_stage`/F 尾部共用该视图（成功路径返回封口文档的 seal 块） |

新增 2 项电池反例：`f_saved_recompute_deadline_stop`（P1 攻击形态：首个解码返回后时钟越过工作截止（cap 内）→仅 1 次规范解码执行、生产解码在 `saved_teacher_prod_decode` 被拒、报告+覆盖仍成立）、`f_second_stage_deadline_stop_reports`（P2-2 攻击形态：D1 回放中越截止→D6 记 `stage_deadline_stop` 不启动、f_report 与 return_observation 均落盘、verdict=incomplete）；既有撤回用例补成本同步断言（返回 `summary_doc.budget`、盘上 `budget` 与 `return_coverage.registered_measured_s` 三方相等；返回顶层 `seal.status`=撤回终态）。审阅者另注：attempt_004 的 D1 initial 在新进程复放出现逐位差异且 F 正确拒收——未归因为代码缺陷，原同进程记录不改写；该现象与 F 逐位重放的设计语义（真实 CUDA 首现合法漂移按 `replay_mismatch` 收口）一致。

**权威合成验收 `derived/audit/w1_smallset_fit_diagnostic/attempt_006/`**（2026-11-08，第五轮修复后代码，独占创建，CLI exit 0）：封口 `sealed_within_budget`（round 1）、登记成本 **1433.63s** ≤ 3300s、返回覆盖 covered=true（elapsed 1429.63 ≤ 1433.63）；微型 D1（300 步，登记成本 236.74s，返回覆盖 covered）四门全过（RMSE 0.00115m / 0.00288rad / 0.00186rad / FK 0.00121m）→ `handoff_ready=true`（候选经 D6 入口消费）；微型 D6（登记成本 1396.38s，返回覆盖 covered）六窗四门全过（macro RMSE 0.00633m）；F 两阶段逐位重放通过（`fit_pass`、`claim_deadline_exceeded` 未触发、F 自身返回覆盖 covered=true）；52/52 反例捕获。`attempt_005` 为第五轮修复运行中途再次变更实现字节（撤回返回视图细化）而终止的残缺目录（无 acceptance_summary/return_observation，不消费）；该细化只影响覆盖失败路径的**返回视图**，盘上撤回行为不变，但仍以最终字节重跑 attempt_006 为准。`attempt_004` 保留为被取代证据（其证据不含第五轮的保存量重算截止守卫/阶段停止收口/撤回成本同步修复项）。**attempt_006 自第六轮修订（§13）起保留为被取代证据。**

## 13. 第六轮独立审阅修订（2026-11-08：2 P1 + 1 P2 全部修复）

第五轮修订经独立复审确认有效（attempt_006 四份证明的 SHA/预算绑定、D1/D6 保存量重算与交接成立，52 项均 caught，登记成本 1433.63s），但新隔离反例复现 2 P1 + 1 P2（复现结果 `/tmp/opencode/review_i_round5_20261109_results.json`、`/tmp/opencode/review_i_round5_authority_20261109_results.json`）。全部接受并修复，电池扩至 **58 项**（测试 17 passed、N0/N1 回归 33 passed），审阅者攻击形态在修复后代码上 fail-closed。attempt_006 自本轮起为被取代证据。

| # | 复审发现 | 修复 |
| --- | --- | --- |
| P1-1 | D 驱动与交接未传入预算守卫：`FitDriver.run` 与 `resolve_handoff` 调用无预算的本地核验——时钟推进到工作截止后，D1 仍执行 3 次解码/4 次生产解码/3 次指标归约才停止；交接仍执行 11/12/11 次对应调用并返回 `handoff_ready=true`、`budget_exceeded=false` | `FitDriver.run` 的本地产核验传入 `budget=self.budget`（并置 `_phase="local_verification"`；越线由驱动既有收口记为 `BudgetExceeded`/incomplete），`budget.check("local_verification_complete")` 保留；`resolve_handoff` 传入同一 D6 预算并捕获 `BudgetExceeded` → `handoff_deadline_exceeded:*`、`handoff_ready=false`（`budget_exceeded` 由检查置真） |
| P1-2 | F 端点仍存在截止后数值调用：①保存量读取返回越线后仍启动教师解码；②fresh 解码返回越线后仍启动指标归约 | 新增 **`f_teacher_recheck_{role}_w{b}`**（保存量读取与逐位比较之后、`_teacher_reference_checked` 之前）与 **`f_fresh_metrics_reduce_{role}_w{b}`**（fresh `decoder.decode` 返回之后、`compute_window_metrics` 之前）两处 `budget.check`；越线由既有 `except FitDiagnosticError` 收口为 `fresh_replay_error`→incomplete，晚解码/晚归约均为 0 |
| P2 | 多轮/耗尽撤回的返回快照仍靠推算：第二轮撤回盘上 `return_coverage_elapsed_s=1000` 而返回 1008；轮数耗尽盘上 `cost_coverage_unproven=true` 而返回缺失该标志、预留值不一致（单轮成本同步已修复） | **N0 `verify_return_coverage` 三个返回分支新增 `final_doc`**——覆盖成功返回入口文档、撤回返回最后一次写入的 withdrawn 文档（含多轮成本与耗尽标志），即**盘上最终文档的精确快照**；I 模块 `_coverage_return_view` 直接使用该快照（不再重建撤回语义，也无采样后 I/O）。N0 修改为加法性返回字段（N0/N1 CLI 与测试不使用/不受影响，N0/N1 回归 33 passed） |

新增 6 项电池反例：`d_driver_recompute_deadline_stop`（P1-1 驱动形态：首解码返回即越线→该阶段 `local_verification` 收口 incomplete、仅 1 次解码）、`handoff_recompute_deadline_refused`（P1-1 交接形态：越线→`handoff_deadline_exceeded` 拒绝、`budget_exceeded=true`）、`f_teacher_recheck_deadline_stop`（P1-2①：慢保存量读取→教师重解码零调用）、`f_fresh_metrics_reduce_deadline_stop`（P1-2②：慢 fresh 解码→指标归约被拒、`mean_fit_passed=null`）、`withdrawal_view_matches_disk_multi_round`（P2：慢证明写盘强制第二轮撤回→返回文档与盘上**逐字段相等**、盘上 `return_coverage_elapsed_s` 为入口值 1000、measured≥1027）、`withdrawal_view_matches_disk_exhausted`（P2：乘性慢写盘耗尽全部轮次→返回文档与盘上逐字段相等、`cost_coverage_unproven=true` 双方一致）。

**权威合成验收 `derived/audit/w1_smallset_fit_diagnostic/attempt_007/`**（2026-11-08，第六轮修复后代码，独占创建，CLI exit 0）：封口 `sealed_within_budget`（round 1）、登记成本 **1432.07s** ≤ 3300s、返回覆盖 covered=true（elapsed 1428.07 ≤ 1432.07）；微型 D1（300 步，登记成本 233.79s，返回覆盖 covered）四门全过（RMSE 0.00115m / 0.00288rad / 0.00186rad / FK 0.00121m）→ `handoff_ready=true`（候选经 D6 入口消费，交接重算在预算下通过）；微型 D6（登记成本 1387.21s，返回覆盖 covered）六窗四门全过（macro RMSE 0.00633m）；F 两阶段逐位重放通过（`fit_pass`、`claim_deadline_exceeded` 未触发、F 自身返回覆盖 covered=true）；58/58 反例捕获。`attempt_006` 保留为被取代证据（其证据不含第六轮的驱动/交接预算守卫、F 端点边界与撤回最终文档快照修复项）。**attempt_007 自第七轮修订（§14）起保留为被取代证据。**

## 14. 第七轮独立审阅修订（2026-11-08：2 P1 + 1 P2 全部修复）

第六轮修订经独立复审确认有效（attempt_007 四份证明、保存量重算、调用矩阵及 58/58 电池记录核验成立，登记成本 1432.07s≤3300s），但新隔离边界反例复现 2 P1 + 1 P2（复现脚本 `/tmp/opencode/review_i_round6_boundaries_20261109.py`、结果 `/tmp/opencode/review_i_round6_boundaries_20261109_results.json`）。全部接受并修复，电池扩至 **63 项**（测试 17 passed、N0/N1 回归 33 passed）。attempt_007 自本轮起为被取代证据。

| # | 复审发现 | 修复 |
| --- | --- | --- |
| P1-1 | 截止守卫未贯穿 helper 内部：①`_fresh_probe` 首个生产解码返回越线后仍启动 5 次生产解码、6 次 FK；②`_teacher_reference_checked` 内部教师读取返回越线后仍启动 1 次教师解码；③`verify_stage_local` 首窗读取返回越线后仍启动另外 5 次教师数组读取 | **内部调用链逐一传守卫**：`_fresh_probe` 新增 `budget` 参数——每窗生产解码前 `f_fresh_prod_decode_w{b}`、每窗 FK 前 `f_fresh_fk_w{b}`；`_teacher_reference_checked` 新增 `budget`——教师读取与逐位校验之后、解码之前 `f_teacher_decode_w{b}`；`verify_stage_local` 逐次磁盘读取守卫 `saved_teacher_read_w{b}`/`saved_inputs_read_w{b}`/`saved_probe_read_{idx}`；`_checkpoint_binding_problems` 新增 `budget`——每次 checkpoint 读取前 `saved_checkpoint_read_{name}`。越线一律经既有收口（F 阶段 `stage_deadline_stop`/`fresh_replay_error`）→incomplete，晚读取/晚解码/晚 FK 均为 0 |
| P1-2 | 交接核验尾部越线仍返回可交接：最后一次指标归约返回至 91s 后仍启动 2 次 checkpoint 读取；最后一次 checkpoint 读取返回至 91s 仍返回 `handoff_ready=true`、`budget_exceeded=false` | ①checkpoint 读取逐次守卫（见 P1-1）；②`resolve_handoff` 在本地核验返回后、交接声明前**非抛出复查** `clock >= work_deadline`：越线置 `budget_exceeded=true`/`stopped_at=handoff_final_claim`、追加 `handoff_deadline_exceeded:handoff_final_claim`、拒绝交接（不与 `handoff_ready=true` 并存） |
| P2 | D 预算停止路径丢失已执行核验计数：`FitDriver.run` 只在核验成功返回后保存 `verification_counters`，首解码完成后越线时返回与盘上报告均缺少已执行的 1 次解码 | `verification_counters` 提升到 `run()` 的 `try` 之外，异常收口分支保存**实际快照**（`summary["verification_counters"] = dict(verification_counters)`）；该计数始终与训练调用矩阵分列（矩阵仍只判阶段自身单元） |

新增 5 项电池反例：`f_fresh_probe_deadline_stop`（D6 六窗：首个 fresh 生产解码返回即越线→仅 1 次解码、0 次 FK，`production_fk_batch_calls=0`）、`f_teacher_decode_deadline_stop`（慢教师读取→教师解码零调用）、`saved_teacher_read_deadline_stop`（D6 六窗：首窗读取越线→第二窗读取被拒，仅 1 次读取）、`handoff_checkpoint_read_deadline_stop`（末次归约越线→checkpoint 读取零调用、交接拒绝）、`handoff_final_claim_deadline_refused`（末次 checkpoint 读取越线→交接声明被拒、`budget_exceeded=true`）；驱动截止用例补 `verification_counters.metric_decode_calls==1` 断言。

**权威合成验收 `derived/audit/w1_smallset_fit_diagnostic/attempt_008/`**（2026-11-08，第七轮修复后代码，独占创建，CLI exit 0）：封口 `sealed_within_budget`（round 1）、登记成本 **1476.95s** ≤ 3300s、返回覆盖 covered=true（elapsed 1472.95 ≤ 1476.95）；微型 D1（300 步，登记成本 237.30s，返回覆盖 covered）四门全过（RMSE 0.00115m / 0.00288rad / 0.00186rad / FK 0.00121m）→ `handoff_ready=true`（候选经 D6 入口消费，交接声明前复查通过）；微型 D6（登记成本 1417.38s，返回覆盖 covered）六窗四门全过（macro RMSE 0.00633m）；F 两阶段逐位重放通过（`fit_pass`、`claim_deadline_exceeded` 未触发、F 自身返回覆盖 covered=true）；63/63 反例捕获。`attempt_007` 保留为被取代证据（其证据不含第七轮的 helper 内部逐单元守卫、交接声明复查与驱动截断计数修复项）。**attempt_008 自第八轮修订（§15）起保留为被取代证据。**

## 15. 第八轮独立审阅修订（2026-11-08：2 P1 + 1 P2 全部修复）

第七轮修订经独立复审确认有效（attempt_008 四份证明、保存量、调用矩阵及 63/63 电池记录核验成立，登记成本 1476.95s≤3300s），但新隔离边界反例复现 2 P1 + 1 P2（复现脚本 `/tmp/opencode/review_i_round7_new_boundaries_20261109.py`、结果 `/tmp/opencode/review_i_round7_new_boundaries_20261109_results.json`）。全部接受并修复，电池扩至 **68 项**（测试 17 passed、N0/N1 回归 33 passed）。attempt_008 自本轮起为被取代证据。

| # | 复审发现 | 修复 |
| --- | --- | --- |
| P1-1 | F 的 forward/strict-load 内部仍缺截止守卫：conditioner 返回越线后仍启动 1 次 backbone forward；checkpoint 读取返回越线后仍执行 2 次 strict 加载；model strict 加载返回越线后仍启动 conditioner strict 加载 | `_fresh_probe` 在 conditioner 返回后、backbone 前插入 **`f_fresh_backbone`**；`_load_state_dicts_checked` 新增 `budget` 参数——读取后 **`f_strict_load_{initial,last}_pre`**、model 加载前 `_model`、model 加载返回后 conditioner 加载前 `_conditioner` 三处检查。越线由既有收口→incomplete，晚 backbone/晚加载为 0 |
| P1-2 | F 的另一条数组读取链仍未守卫：六窗输入重建首读返回越线后仍执行 5 次原始量读取＋30 次条件数组读取；首窗 fresh 指标归约返回越线后仍读取下一窗端点工件 | 输入重建**逐次读取**守卫：每窗原始量前 `f_probe_{role}_input_w{b}_raw`、每个条件键前 `f_probe_{role}_input_w{b}_{key}`；端点工件每窗读取前 **`f_saved_probe_read_{role}_w{b}`**（在 `_saved_probe_records` 之前）；越线经既有收口→incomplete |
| P2 | 部分坏容器仍使 F 无报告退出：`probe_index.json=[]` 与 `stage_report.seal=[1]` 各触发未捕获 `AttributeError`，无 `f_report.json`、无返回证明 | **先校验容器类型再使用**：`verify_stage_local` 的 artifacts_index/probe_index 改经 `_read_json_object`（解析失败/非映射→结构化问题并返回 incomplete），`probes` 列表与逐 entry/逐 window 的映射类型逐一校验；F 流/`_return_proof_problems`/`resolve_handoff` 的 `seal` 非映射→结构化拒绝/跳过（`stage_seal_not_consumable`/`report_seal_not_a_mapping`）；`verify_artifacts_index` 的 `files`/逐 `meta` 映射校验；`_saved_probe_entry` 非映射/缺窗→按 missing 跳过、`_saved_probe_records` 结构异常→`FitDiagnosticError`（被既有 replays 收口捕获）。两种坏容器现均正常落盘 incomplete f_report + return_observation |

新增 5 项电池反例：`f_fresh_backbone_deadline_stop`（conditioner 返回越线→backbone 零调用）、`f_strict_load_deadline_stop`（读取越线→0 次 strict 加载；model 加载越线→conditioner 加载被拒，仅 1 次加载）、`f_input_rebuild_deadline_stop`（D6 六窗：首原始量读取越线→条件读取 0 次、原始读取 1 次）、`f_saved_probe_read_deadline_stop`（D6 六窗：首窗归约越线→下一窗端点读取被拒，仅 1 次读取）、`bad_containers_report_incomplete_f`（probe_index=[] 与 seal=[1] 均正常落盘 incomplete 报告与证明，无崩溃）。

**权威合成验收 `derived/audit/w1_smallset_fit_diagnostic/attempt_009/`**（2026-11-08，第八轮修复后代码，独占创建，CLI exit 0）：封口 `sealed_within_budget`（round 1）、登记成本 **1383.51s** ≤ 3300s、返回覆盖 covered=true（elapsed 1379.51 ≤ 1383.51）；微型 D1（300 步，登记成本 225.96s，返回覆盖 covered）四门全过（RMSE 0.00115m / 0.00288rad / 0.00186rad / FK 0.00121m）→ `handoff_ready=true`（候选经 D6 入口消费，交接声明前复查通过）；微型 D6（登记成本 1313.61s，返回覆盖 covered）六窗四门全过（macro RMSE 0.00633m）；F 两阶段逐位重放通过（`fit_pass`、`claim_deadline_exceeded` 未触发、F 自身返回覆盖 covered=true）；68/68 反例捕获。`attempt_008` 保留为被取代证据（其证据不含第八轮的 forward/strict 加载内部边界、输入重建与端点逐读、坏容器结构化收口修复项）。

## 16. 第九轮独立审阅修订（2026-11-09：3 P1 + 2 P2 全部修复）

第八轮修订经独立复审确认有效（attempt_009 四份证明、保存量、调用矩阵、交接及 68/68 电池记录成立，登记成本 1383.51s≤3300s），但新隔离反例复现 3 P1 + 2 P2（复现脚本 `/tmp/opencode/review_i_round8_new_boundaries_20261109.py`、结果 `/tmp/opencode/review_i_round8_new_boundaries_20261109_results.json`）。全部接受并修复，电池扩至 **76 项**（测试 17 passed、N0/N1 回归 33 passed），审阅者四类攻击形态在修复后代码上全部 fail-closed（定点验证脚本 `/tmp/opencode/verify_round9_fixes.py`、结果 `/tmp/opencode/verify_round9_fixes_results.json`）。attempt_009 自本轮起为被取代证据。

| # | 复审发现 | 修复 |
| --- | --- | --- |
| P1-1 | checkpoint 配置/身份未完整绑定：重签文件绑定链后，checkpoint `lr=123`、`grad_accum=99` 或外来 `identity` 均使本地核验 `problems=[]`、F 发布 `complete_evidence=true/replay_passed=true` | 新增 `CHECKPOINT_CONFIG_FIELDS`/`_checkpoint_config_problems`：checkpoint 内嵌 `config` 对冻结 `FitPlan` **14 字段逐项对账**（steps/micro_batch/grad_accum/seed/lr/weight_decay/grad_clip/model_type/d_model/n_layers/n_heads/ffn_mult/time_window/val_seed，含 bool 型伪造拒绝）；新增 `_checkpoint_identity_problems`：checkpoint `identity` 必须等于阶段声明身份——`FitDriver.run` 向本地核验传 `expected_identity=self.identity`，F/交接以**阶段报告 `identity`** 为锚（报告身份非映射→`stage_report_identity_invalid`；driver 身份与报告身份并存时须相等）。`_checkpoint_binding_problems` 与 `_f_identity_problems` 均执行同一语义契约，F 在**构造/加载前**拒绝 |
| P1-2 | D 内部调用链仍可越截止继续执行：教师首窗规范解码越线后仍执行 5 次规范解码+6 次生产解码+6 次 FK；probe 首窗解码越线后另有 6 次指标归约；训练 conditioner 越线后仍启动 1 次 backbone+1 次 loss+2 次 FK（直到 pre_backward 才停） | `_ensure_teacher` 逐窗 `teacher_window_w{b}_decode/_prod_decode/_fk`；`run_probe` 逐窗 `{point}_w{b}_decode/_prod_decode/_fk/_metrics`；训练 micro 经驱动注册的 model **pre-forward/forward 钩子**守卫 `{micro_point}_pre_backbone`/`{micro_point}_pre_loss`（共享 Trainer 字节与数值路径不变；probe predict 同样受钩子保护）。越线后下一单元零启动 |
| P1-3 | F 身份 helper 的 checkpoint 读取未守卫：本地核验末次读取越线后仍 2 次身份读取；身份 helper 首读越线后仍启动第二次读取 | `_f_identity_problems` 新增 `budget`/`report_identity` 参数，每次 checkpoint 读取前 **`f_identity_checkpoint_read_{name}`**；调用处捕获 `BudgetExceeded` → `stage_deadline_stop` 并停止后续阶段（不构造、报告/证明照常落盘）；artifacts_index 读取改经 `_read_json_object` |
| P2-4 | 新截止出口丢失已完成 conditioner 计数：conditioner 实际完成 1 次、backbone 被正确阻断，但返回值与盘上报告记 `conditioner_forward_calls=0` | `_fresh_probe` 把 conditioner/backbone 计数移到**各自 forward 返回后的实际调用边界**，截止异常路径的计数快照保留已完成调用 |
| P2-5 | 嵌套坏结构仍使 F 无报告退出：`step` 字符串→`ValueError`、`window_id=[]`→`TypeError`、`forward_counts=[1]`→`AttributeError`、多一窗→`IndexError` | `verify_stage_local` 在任何格式化/哈希键/下标前校验类型与长度：entry `step`/`idx` 必须 JSON 整数（bool 拒绝）、`window_id` 必须字符串、每 entry 窗口数必须等于注册窗数、`forward_counts` 必须映射；F 端点重放另校验保存窗口数。四形态均落 `f_report.json`+`return_observation.json`（`local_reverify_failed`→incomplete） |

新增 8 项电池反例：`checkpoint_config_binding_rejected`（lr/grad_accum 重签篡改）、`checkpoint_identity_binding_rejected`（外来 identity）、`teacher_window_units_deadline_stop`（首窗解码越线→后续零单元）、`probe_window_units_deadline_stop`（含归约）、`training_micro_internal_deadline_stop`（conditioner→backbone/loss/FK 零启动、backward/adamw=0）、`f_identity_checkpoint_read_deadline_stop`（两个跳变点均零晚读取/零构造）、`f_completed_conditioner_counter_kept`（返回与盘上均 conditioner=1/backbone=0）、`nested_probe_schema_containers_report_incomplete`（四形态均落报告与证明）。定点复现（修复后）实测：教师/探针晚解码、晚生产解码、晚 FK、晚归约均为 0；训练晚 backbone/loss/FK 均为 0；身份读取晚读 0、构造 0。

**权威合成验收 `derived/audit/w1_smallset_fit_diagnostic/attempt_010/`**（2026-11-09，第九轮修复后代码，独占创建，CLI exit 0）：封口 `sealed_within_budget`（round 1）、登记成本 **1468.88s** ≤ 3300s、返回覆盖 covered=true（elapsed 1464.88 ≤ 1468.88）；微型 D1（300 步，登记成本 235.34s，返回覆盖 covered（231.34））四门全过（RMSE 0.00115m / 0.00288rad / 0.00186rad / FK 0.00121m）→ `handoff_ready=true`（候选经 D6 入口消费，交接声明前复查通过）；微型 D6（登记成本 1388.25s，返回覆盖 covered（1384.25））六窗四门全过（逐窗 true，macro RMSE 0.00633m）；F 两阶段逐位重放通过（`fit_pass`、`claim_deadline_exceeded` 未触发、F 自身返回覆盖 covered=true）；**76/76 反例捕获**。`attempt_009` 保留为被取代证据（其证据不含第九轮的 checkpoint 配置/身份语义绑定、D 内部单元链、F 身份读取、conditioner 计数边界与嵌套坏结构收口修复项）。

**边界（与前轮一致）：R/真实 D1/D6/F 未实施、未授权、未运行；真实拟合仍须 §8.3 新冻结材料与新明确授权；T6 旧结论与 partial/null/false/not_ready/不接入全部保持；本轮未运行全项目套件（仅本交付 17 passed + N0/N1 回归 33 passed）。**
