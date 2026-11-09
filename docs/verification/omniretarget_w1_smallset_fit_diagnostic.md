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

## 5. 反例电池（§6.2/v0.3/v0.4 全清单；首轮 35 项 + 第二轮 6 项 + 第三轮 5 项 + 第四轮 4 项 + 第五轮 2 项 = 52 项）

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

## 6. 测试结果

- `tests/test_w1_smallset_fit_diagnostic.py`：**17 passed**（CPU，离线；含完整 52 项电池 `test_counterexample_battery_all_caught`）。
- 反例电池独立运行：**52/52 caught**，ID 集合与冻结清单精确相等（§12 第五轮审阅修订后）。

## 7. 权威合成验收运行

**`derived/audit/w1_smallset_fit_diagnostic/attempt_006/`**（2026-11-08，§12 第五轮审阅修订后代码，独占创建，CLI exit 0）：单一单调时钟 t0 先于重导入，硬截止 3300s/预留 300s；summary 经发布-核验封口 `sealed_within_budget`（round 1，登记成本 **1433.63s** 覆盖封口、候选写盘、必要回读与返回）+ `return_observation.json` SHA 绑定 covered=true（elapsed 1429.63 ≤ 1433.63）。`attempt_001–004`（修复前/二轮/三轮/四轮代码）保留为被取代证据；`attempt_005` 为第五轮修复运行中途实现字节再次变更而终止的残缺目录（无 acceptance_summary/return_observation，不消费）。

- **微型 D1**（300 步/probe_every 30/lr 0.03，d16 合成世界，登记成本 236.74s，返回覆盖 covered）：`local_complete_evidence=true`、调用矩阵实际=期望零偏差、本地核验零问题（保存量重算在新截止守卫下逐单元通过）；教师与预测双侧 metric_decode **重解码逐位绑定**、生产 FP32 解码量**重推导逐位绑定**；step-300 四门全过（root RMSE **0.00115m**、root 旋转均值 **0.00288rad**、joint RMSE **0.00186rad**、teacher FK MPJPE **0.00121m**）→ `local_mean_fit_passed=true`、`complete_fit`；`handoff_candidate.json` 落盘且经 D6 入口消费 → `handoff_ready=true`（problems=[]）。
- **微型 D6**（300 步/lr 0.02，六注册窗，登记成本 1396.38s，返回覆盖 covered）：覆盖行多重集精确、**六窗四门全过**（macro RMSE 0.00633m，`macro_is_not_a_gate` 显式登记）、`local_mean_fit_passed=true`。
- **F**：返回证明前置核验通过；对两阶段各 fresh 构造+initial/last 各一次契约核验后 strict 加载 probe（每阶段恰 2 次 fresh 预测）；保存量重算在教师解码/生产解码/预测解码/指标归约边界均受同一预算守卫（本轮真实时钟下全部通过）；原始 x0、生产 FP32 解码量与 FK 点对保存数组**逐位一致**，教师侧从原始 x0 重解码绑定后对账；完成声明门 `claim_deadline_exceeded` 本轮未触发；F 自身对 f_report.json 的封口后回读执行**独立返回覆盖核验**（covered=true）→ `replay_passed=true`、`complete_evidence=true`、`mean_fit_passed=true`、`verdict=fit_pass`、`tolerance_policy=bitwise_no_tolerance`。
- **反例电池**：52/52 caught（ID 集合与冻结清单精确相等，含 §12 新增 2 项）。

历史运行 `attempt_001`（修复前代码）、`attempt_002`（§8 修订后）、`attempt_003`（§10 修订后）、`attempt_004`（§11 修订后）保留为被取代证据：其证据分别不含后续轮次的修复项（教师解码绑定/必需工件集合 → 证明预算绑定/候选时序 → 回读入窗/F 自身覆盖/声明门 → 保存量重算截止守卫/阶段停止收口/撤回成本同步），不可消费为本轮结论。**合成四门通过只证明 I 阶段管线机制与门实现本身在合成世界上自洽；不外推到真实 W1 训练窗（真实可达性属 R/D1/D6 的授权后问题）。**

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

**权威合成验收 `derived/audit/w1_smallset_fit_diagnostic/attempt_006/`**（2026-11-08，第五轮修复后代码，独占创建，CLI exit 0）：封口 `sealed_within_budget`（round 1）、登记成本 **1433.63s** ≤ 3300s、返回覆盖 covered=true（elapsed 1429.63 ≤ 1433.63）；微型 D1（300 步，登记成本 236.74s，返回覆盖 covered）四门全过（RMSE 0.00115m / 0.00288rad / 0.00186rad / FK 0.00121m）→ `handoff_ready=true`（候选经 D6 入口消费）；微型 D6（登记成本 1396.38s，返回覆盖 covered）六窗四门全过（macro RMSE 0.00633m）；F 两阶段逐位重放通过（`fit_pass`、`claim_deadline_exceeded` 未触发、F 自身返回覆盖 covered=true）；52/52 反例捕获。`attempt_005` 为第五轮修复运行中途再次变更实现字节（撤回返回视图细化）而终止的残缺目录（无 acceptance_summary/return_observation，不消费）；该细化只影响覆盖失败路径的**返回视图**，盘上撤回行为不变，但仍以最终字节重跑 attempt_006 为准。`attempt_004` 保留为被取代证据（其证据不含第五轮的保存量重算截止守卫/阶段停止收口/撤回成本同步修复项）。
