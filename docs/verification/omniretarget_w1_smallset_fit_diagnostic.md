# W1 I 阶段：小集拟合诊断器实现与合成反例验证记录

日期：2026-11-07。方案：[数值缺口有限收口与受控拟合诊断方案 v0.4](../plans/omniretarget_w1_numerics_closure_fit_diagnostic_plan.md) §6/§7（I 阶段交付）。

**性质：纯合成实现与验证交付。本记录不含任何真实数据读取、真实 checkpoint、真实模型/优化器构造、真实 forward/backward/FK/AdamW 或真实训练更新；全部模块与测试在合成世界（合成 tokenizer/关节限位/教师窗/15-site FK 替身）上离线运行，不声称与生产 CUDA/foreach 等价。** 真实 R 冻结与真实 D1/D6/F 执行仍须方案 §8.3 的新冻结材料与新明确授权，本交付不创建真实 cohort/attempt/授权，不恢复旧 T6 一次性授权。

**2026-11-16 当前状态：**审阅 `b4d5729` 后修复 2 P1+1 P2（§26），复审 `eba46de` 的撤回成本 1 P2 已修复（§27）；当前实现 `77938bf` 的完整纯合成验收 `attempt_019` 通过，见 §28。`attempt_018` 的数值链和 uv wait 事实保留，但其最终 stdout flush 缺成本覆盖证明；`attempt_017` 也缺外层返回证明。不把合成结果当作真实拟合证据。

## 1. 交付物

| 文件 | 内容 |
| --- | --- |
| [src/ireset/eval/w1_smallset_fit_diagnostic.py](../../src/ireset/eval/w1_smallset_fit_diagnostic.py) | I 阶段独立诊断模块：FitPlan/FitRegistration、StageBudget、运行时硬门、`metric_decode_v1`、§7.1/§7.2 物理指标、probe 机制、FitDriver（复用已审计 Trainer 更新单元）、§7.4 第 1–3 项本地核验、D1→D6 交接、F 逐位重放、attempt 台账、R 只读元数据守卫、35 项合成反例电池、合成验收运行器 |
| [scripts/verify_w1_smallset_fit_diagnostic.py](../../scripts/verify_w1_smallset_fit_diagnostic.py) | 仅合成的验收工作进程：微型 D1→本地核验→交接判定→D6→F 逐位重放→反例电池；摘要为等待外层返回核验的候选 |
| [scripts/run_w1_i_synthetic_acceptance.py](../../scripts/run_w1_i_synthetic_acceptance.py) | 标准库监督器：uv 启动前采样，实际 wait 后对账登记成本、来源 SHA 和阶段证明，完成观察绑定裁决文件 SHA |
| [tests/test_w1_i_outer_return.py](../../tests/test_w1_i_outer_return.py) | 外层真实子进程 wait、阶段返回、缺失证明、源漂移、越界和慢写盘反例 |
| [tests/test_w1_smallset_fit_diagnostic.py](../../tests/test_w1_smallset_fit_diagnostic.py) | 16 项测试：冻结常量、注册校验、解码/公式/尖峰、状态分层校验器、台账/R 守卫/参数契约、D6 数据分母、微型阶段+F 逐位重放、交接门、预算停止，及完整 35 项反例电池 |

## 2. 复用边界（方案 §6.1）

驱动**只通过已审计 Trainer 更新单元**执行训练：每个 micro 经 `Trainer.accumulate_micro`（内部 `_forward`→`predict`→损失→`(loss/N).backward`+有限性检查），裁剪经 `Trainer.clip_gradients`，更新经 `Trainer.step_optimizer`；AdamW/loss 装配均由 Trainer 构造，驱动不重写任何前向/反向/优化器算术。共享单元本身不查截止时间——驱动在每个单元边界（step/micro 开始、加载后、pre_backward 钩子、backward 后、clip、step、probe、checkpoint、写盘）插入 `StageBudget.check`，并把 `work_deadline` 传给 Trainer 构造器使 `save()` 的内部检查同口径。`scripts/w1_s0.py` 的独立手写更新循环**未被沿用**；probe 语义（专用 generator、固定 t/noise、eval+no_grad、RNG 保存恢复）参考 S0 并经由 `Trainer.predict` 执行。

## 3. 冻结契约（I 阶段确定，R 阶段绑定）

- **FitPlan 冻结默认值**（非 synthetic 计划偏离即拒绝）：steps=1000、probe_every=10（101 次 probe，step 0 与 10..1000）、grad_accum=4、D1 micro_batch=1 / D6 micro_batch=6、seed/val_seed=20261008、lr=1e-4、weight_decay=1e-5、grad_clip=1、d240/4层/6头/time_window31；`frozen_plan_defaults_problems()` 由 CLI 与测试断言为空。
- **FitRegistration**：D1 恰 1 窗、D6 恰 6 窗、规范字典序、start=0、跨度=T_WINDOW=60、split 全为 train；驱动另做**注册↔物化数据集窗键绑定**（`dataset_registration_window_mismatch`），并绑定 trainer 配置与计划（steps/micro/grad_accum/lr/wd/clip/seed/架构）与**七项冻结损失权重**（w_pos/w_rot6d/w_joint/w_geodesic/w_fk/w_contact=0/w_vel，容差目标与 checkpoint 内嵌配置同契约）；**阶段身份嵌入 plan/registration 的规范 SHA-256**，F 每个 stage spec 与交接另须携带调用方声明的**完整 expected identity**，报告与两 checkpoint 身份逐字段等于该外部声明（摘要仅是派生绑定）——协同改写 stage/module/implementation 等任一字段均被拒；实际 `loss_fn.w` 在 preflight/probe/micro/save/每次 loss 调用边界复核，`LossAssembler` 内部每次 decode/FK 前后均有工作截止检查；**实际 AdamW `param_groups` 的 lr/weight_decay/betas/eps/maximize/amsgrad 与 model+conditioner 参数名册在构造/preflight/micro/clip/step/save 逐组复核**（`optimizer_binding`），conditioner 入口、backbone/loss 入口与 clip 取证之后各有截止/精度守卫；checkpoint 实际张量有限性与 state key set 跨端点绑定（不信任自报 `parameters_finite`）；F 输入重建的 condition 数组按冻结键/形状/float32 契约读取，NPZ 坏存档（含 object 数组）不启用 pickle、结构化拒绝。
- **第十五轮补强（历史状态/构造准入/键类型）**：实际 AdamW 必须是**从头、step-0 空状态**——构造与每次运行期复核要求 `opt.state` 为空（`optimizer_state_not_empty`），之后每个已填充条目的 `step` 必须精确等于驱动已完成的 native step 数且 `exp_avg/exp_avg_sq` 形状对齐、有限、`exp_avg_sq` 非负（`optimizer_state_step/_shape/_nonfinite/_negative`），每步执行后计数恰加一；新增的 decode/FK/F fresh 精度门在 runtime 读取之后各加预算复查（`*_post_runtime`/`f_fresh_entry_post_runtime`/`f_fresh_fk_w*_post_runtime`）；plan 派生参考模块的 model/conditioner 构造各自成单元（`saved_state_contract_model/_conditioner`），且**身份已拒绝的阶段不再构造参考模块**；checkpoint 架构检查先验证键为字符串再排序（混合 int/str 键为结构化拒绝而非 TypeError）。
- **第十四轮补强（实际算术/架构/精度边界）**：实际 AdamW `param_groups` 另冻结 `decoupled_weight_decay=true`、`foreach=None`、`fused=None`、`capturable=false`、`differentiable=false`（与 lr/wd/betas/eps/maximize/amsgrad 及参数 id 名册同一 `optimizer_binding` 契约）；checkpoint state 的**独立架构契约**由冻结 plan 构造参考模块派生（键集/shape/dtype，不依赖 checkpoint 自身；参考构建为预算守卫、RNG 隔离的元数据单元），本地/F 加载前/交接共用——两端同时删键、错 shape、int32 均拒；conditioner 入口、loss 内部 FK 调用前、F fresh 入口与逐窗 FK 前均做 runtime+截止复查（慢精度读取与解码后精度漂移都不再买到一次晚调用）；NPZ 读取把 `zipfile.BadZipFile`/`EOFError`/`zlib.error` 一并结构化包装（CRC 损坏不致 F 无报告退出）。
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

## 5. 反例电池（§6.2/v0.3/v0.4 全清单；第十五轮 110 项 + 第十六轮 5 项 = 115 项）

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
| 第十轮：loss 权重未绑定冻结目标 | `loss_weight_binding_rejected`（checkpoint `w_fk=999`/`w_contact=1`/缺 `loss` 块、驱动实际目标 `w_fk=999`、冻结计划元数据省略均被七字段逐一拒绝；三 checkpoint 形态 F 均 incomplete 且报告/证明落盘） |
| 第十轮：F 身份锚可被协调改写 | `stage_identity_external_anchor`（同步改写 stage_report.identity 与两 checkpoint identity 并重签后，对调用方 plan/registration 摘要绑定拒绝；F incomplete+报告落盘） |
| 第十轮：backbone 输出合法性检查后越截止 | `post_backbone_validation_tail_stop`（`Trainer.predict` 内部输出检查越过截止→`*_post_predict` 拒绝 loss 与 FK，零晚 loss/晚 FK、backward/step=0） |
| 第十轮：被拒 backbone 仍计入调用数 | `refused_backbone_not_counted`（post-hook 只在 forward 实际返回后计数；被拒 backbone 在返回、失败快照与盘上报告均不再多计） |
| 第十轮：metrics/root_stats 坏容器 | `bad_metrics_and_root_stats_report_incomplete`（metrics 列表/嵌套列表与 root_stats 错误映射均落 incomplete F 报告+返回证明，无 TypeError 崩溃） |
| 第十一轮：loss 内部 decode→FK 链越截止 | `loss_internal_fk_units_deadline_stop`（训练预测解码越线后 FK 零启动；首 FK 返回越线后教师解码/下一 FK 零启动；零晚解码/晚 FK、backward/step=0） |
| 第十一轮：构造后实际 loss 权重漂移 | `live_loss_weight_drift_refused`（preflight/probe/micro/save/loss 调用边界复核实际 `loss_fn.w`；篡改后零 micro 消费、D/F 均 incomplete） |
| 第十一轮：完整外部身份锚 | `stage_identity_full_external_anchor`（保留 plan/registration 摘要但协同改写 stage/module/implementation 字段，对调用方完整 expected identity 拒绝；缺声明身份 F 结构化 skip） |
| 第十一轮：checkpoint 配置坏容器 | `bad_checkpoint_config_containers_report_incomplete`（`config.loss.w_fk` 多元素 Tensor / `config.lr` 多元素 ndarray 均落 incomplete F 报告+返回证明） |
| 第十一轮：已进入后抛错的 forward 计数 | `executed_failing_backbone_counted`（准入拒绝不计；forward 已进入但随后抛错照常计数：实际=返回=失败快照=盘上=2） |
| 第十一轮：整数/布尔严格性 | `typed_counts_and_coordinates_rejected`（checkpoint step 布尔/浮点、forward_counts 布尔、worst_coordinate 浮点均被严格类型契约拒绝并落 incomplete） |
| 第十二轮：实际 AdamW 参数组未绑定 | `effective_optimizer_binding_rejected`（构造后改 param_groups 的 lr/weight_decay：零次 `opt.step`，`optimizer_binding` 拒绝，D/F incomplete） |
| 第十二轮：conditioner/clip 前截止空档 | `conditioner_and_clip_pre_boundaries_stop`（predict 输入检查越线→conditioner 零启动；clip 取证越线→clip 零启动；BudgetExceeded、step=0） |
| 第十二轮：F 已进入后抛错 forward 计数 | `f_entered_failed_forwards_counted`（fresh conditioner/backbone 执行数值 body 后抛错：实际=返回=盘上=1，F incomplete） |
| 第十二轮：巨大配置标量 | `huge_config_scalars_report_incomplete`（`10**400` 的 lr/w_fk 为类型严格不匹配，无 OverflowError，落 incomplete 报告+证明） |
| 第十二轮：坏 NPZ dtype | `bad_npz_dtype_reports_incomplete`（raw/FK 数组改字符串 dtype：dtype 失败后不再调用 isfinite，落 incomplete 报告+证明） |
| 第十三轮：AdamW 扩展字段与参数名册 | `optimizer_extended_binding_rejected`（betas/eps/maximize 篡改与移除 conditioner 参数：零 step，`optimizer_binding` 拒绝） |
| 第十三轮：绑定读取后的 clip/step 空档 | `binding_read_boundaries_stop`（绑定读取越线后 clip/step 零启动；控制例 clip=1/step=1 且 complete） |
| 第十三轮：checkpoint 实际张量与键集 | `checkpoint_tensor_facts_bound`（NaN 参数/删除 state key：不信任自报 `parameters_finite`，本地与交接均拒绝） |
| 第十三轮：内部 forward 精度复查 | `runtime_drift_between_numeric_units_stop`（真实 ProductionRuntimeGuard + TF32 注入：conditioner 后 backbone 零启动，D/F 均 incomplete） |
| 第十三轮：object NPZ | `object_npz_reports_incomplete`（object 数组存档不启用 pickle、落 incomplete 报告+证明） |
| 第十三轮：条件数组契约 | `condition_contract_reports_incomplete`（contact 缺失/字符串 dtype 经冻结键/形状/dtype 契约结构化拒绝） |
| 第十四轮：AdamW 有效算术标志 | `optimizer_arithmetic_flags_rejected`（`decoupled_weight_decay=false`、`foreach=true`、`fused=true` 均被实际参数组冻结契约拒绝：零 `opt.step`、D incomplete） |
| 第十四轮：精度检查自身耗时 | `runtime_check_time_charged` / `f_runtime_check_time_charged`（D 在 `pre_backbone_post_runtime` 拒绝、晚 backbone=0；F 在 `f_fresh_entry` 拒绝、conditioner=0、报告+证明落盘） |
| 第十四轮：checkpoint 独立架构契约 | `checkpoint_architecture_contract_bound`（两端同删键→`key_set_not_frozen`；last 错 shape→`shape_not_frozen`；last int32→`dtype_not_frozen`；三者本地 mean=null、交接拒绝、F incomplete） |
| 第十四轮：conditioner/FK/F fresh 精度准入 | `conditioner_and_fk_precision_admission` / `f_fresh_fk_precision_admission`（输入检查后 TF32→conditioner 零执行；生产解码后 TF32→FK 零执行；F fresh FK 前检查后零 FK、报告+证明落盘） |
| 第十四轮：CRC 损坏 NPZ | `crc_corrupt_npz_reports_incomplete`（probe raw / condition contact 成员 CRC 翻转：结构化 `record/condition_archive_unreadable`，F incomplete 报告+返回证明照常落盘） |
| 第十五轮：AdamW 初始历史状态 | `optimizer_initial_state_bound`（预置某参数 `step=17`+非零动量：preflight `optimizer_state_not_empty`、零 step、D incomplete；控制例 1 step 完整） |
| 第十五轮：decode/FK 精度门自身耗时 | `decode_fk_runtime_read_time_charged`（在 `loss_decode_pre`/`loss_fk_pre` 的 runtime 读取后越线：晚 decode=0、晚 FK=0，`*_post_runtime` BudgetExceeded） |
| 第十五轮：F fresh 精度门自身耗时 | `f_fresh_runtime_read_time_charged`（`f_fresh_entry` 越线→conditioner=0；`f_fresh_fk_w0` 越线→fresh FK=0；F incomplete 报告+证明） |
| 第十五轮：参考构造的截止与身份门 | `state_contract_construction_guarded`（冷缓存 model 返回越线→conditioner 不再构造、RNG 不变；身份已拒→零构造；控制例 metadata+replay 两对构造） |
| 第十五轮：混合类型 checkpoint 键 | `mixed_checkpoint_keys_report_incomplete`（int+str 键：`checkpoint_*_key_type_not_string`，F incomplete 报告+证明，无 TypeError） |
| 第十六轮：参数冻结与状态名册 | `trainable_parameter_roster_bound`（模型/conditioner 冻结时零 forward、零 step；控制例正常更新） |
| 第十六轮：pre-backward 检查耗时 | `pre_backward_runtime_read_time_charged`（runtime 读取越线后零 backward、零 step；控制例完成四次 backward） |
| 第十六轮：教师参考精度准入 | `teacher_reference_precision_admission`（教师 decode/FK 前精度与截止守卫，TF32 漂移和检查耗时越线均阻止后继单元） |
| 第十六轮：报告身份格式 | `bad_report_identity_no_reference_construction`（身份缺失/list/null 时冷缓存零参考构造、F 报告与证明落盘） |
| 第十六轮：稀疏 checkpoint | `sparse_checkpoint_layout_report_incomplete`（非 strided 张量在 isfinite/加载前拒绝，F incomplete 报告与证明落盘） |

## 6. 第十五轮测试结果（历史快照；当前见 §23）

- `tests/test_w1_smallset_fit_diagnostic.py`：**17 passed**（CPU，离线；含完整 110 项电池 `test_counterexample_battery_all_caught`）。
- 反例电池独立运行：**110/110 caught**，ID 集合与冻结清单精确相等（§22 第十五轮审阅修订后）。
- N0/N1 回归：**33 passed**（N0 `verify_return_coverage` 的加法性 `final_doc` 返回不改变 N0/N1 行为）；本轮 I 测试 + N0/N1 合并运行 **50 passed（219.63s）**。

## 7. 第十五轮历史合成验收运行（非当前代码验收）

**`derived/audit/w1_smallset_fit_diagnostic/attempt_016/`**（2026-11-10，§22 第十五轮审阅修订后代码，独占创建，CLI exit 0）：单一单调时钟 t0 先于重导入，硬截止 3300s/预留 300s；summary 经发布-核验封口 `sealed_within_budget`（round 1，登记成本 **1495.72s** 覆盖封口、候选写盘、必要回读与返回）+ `return_observation.json` SHA 绑定 covered=true（elapsed 1491.73 ≤ 1495.72，summary_sha256 `7f1ed608…`）。`attempt_001–004`、`attempt_006–015`（修复前/二轮…/十三轮/十四轮代码）保留为被取代证据；`attempt_005` 为第五轮修复运行中途实现字节再次变更而终止的残缺目录（无 acceptance_summary/return_observation，不消费）。

- **微型 D1**（300 步/probe_every 30/lr 0.03，d16 合成世界，登记成本 240.07s，返回覆盖 covered=true（elapsed 236.07））：`local_complete_evidence=true`、调用矩阵实际=期望零偏差、本地核验零问题（完整外部 expected identity、checkpoint 逐项配置/身份（含七项损失权重）与实际张量/键集、实际 AdamW 参数组字段与参数名册、forward/backbone、strict 加载、逐次读取、解码/FK、checkpoint 读取均通过；运行期实际 `loss_fn.w`、loss 内部 decode/FK 截止守卫、conditioner/backbone/loss 入口精度守卫与 clip 取证后守卫均无触发）；教师与预测双侧 metric_decode **重解码逐位绑定**、生产 FP32 解码量**重推导逐位绑定**；step-300 四门全过（root RMSE **0.00115m**、root 旋转均值 **0.00288rad**、joint RMSE **0.00186rad**、teacher FK MPJPE **0.00121m**）→ `local_mean_fit_passed=true`、`complete_fit`；`handoff_candidate.json` 落盘且经 D6 入口消费（外部身份锚与张量事实一致）→ `handoff_ready=true`（problems=[]，交接声明前复查通过）。
- **微型 D6**（300 步/lr 0.02，六注册窗，登记成本 1400.49s，返回覆盖 covered=true（elapsed 1396.49））：覆盖行多重集精确、**六窗四门全过**（逐窗 `window_mean_fit_passed` 全 true，macro RMSE 0.00633m，`macro_is_not_a_gate` 显式登记）、`local_mean_fit_passed=true`。
- **F**：每阶段外部 expected identity 前置核验通过；返回证明前置核验通过；对两阶段各 fresh 构造+initial/last 各一次契约核验后 strict 加载 probe（每阶段恰 2 次 fresh 预测）；输入重建逐次读取与 condition 键/形状/float32 契约、conditioner→backbone、strict 加载、教师重读、fresh 解码/FK、指标归约与 checkpoint 读取均在逐单元截止/精度守卫下通过；原始 x0、生产 FP32 解码量与 FK 点对保存数组**逐位一致**，教师侧从原始 x0 重解码绑定后对账；完成声明门 `claim_deadline_exceeded` 本轮未触发；F 自身对 f_report.json 的封口后回读执行**独立返回覆盖核验**（covered=true）→ `replay_passed=true`、`complete_evidence=true`、`mean_fit_passed=true`、`verdict=fit_pass`、`tolerance_policy=bitwise_no_tolerance`。
- **反例电池**：110/110 caught（ID 集合与冻结清单精确相等，含 §21–§22 新增 12 项）。

历史运行 `attempt_001`（修复前代码）、`attempt_002`（§8 修订后）、`attempt_003`（§10 修订后）、`attempt_004`（§11 修订后）、`attempt_006`（§12 修订后）、`attempt_007`（§13 修订后）、`attempt_008`（§14 修订后）、`attempt_009`（§15 修订后）、`attempt_010`（§16 修订后）、`attempt_011`（§17 修订后）、`attempt_012`（§18 修订后）、`attempt_013`（§19 修订后）、`attempt_014`（§20 修订后）、`attempt_015`（§21 修订后）保留为被取代证据：其证据分别不含后续轮次的修复项（教师解码绑定/必需工件集合 → 证明预算绑定/候选时序 → 回读入窗/F 自身覆盖/声明门 → 保存量重算截止守卫/阶段停止收口/撤回成本同步 → 驱动与交接预算守卫/F 端点边界/撤回最终文档快照 → helper 内部逐单元守卫/交接声明复查/驱动截断计数 → forward 与 strict 加载边界/输入重建与端点逐读/坏容器结构化收口 → checkpoint 配置/身份语义绑定/D 内部单元链/F 身份读取/计数边界/嵌套坏结构 → 七项 loss 权重绑定/身份摘要锚/predict 后守卫/被拒 backbone 计数/metrics 与 root_stats 坏容器 → 完整外部身份锚/运行期损失权重复核/loss 内部 decode-FK 守卫/进入即计数/严格整数与坏配置容器 → 实际 AdamW 参数组/conditioner 与 clip 前守卫/F 入口计数/溢出安全/坏 NPZ dtype → AdamW 扩展字段与参数名册/绑定读取守卫/checkpoint 实际张量与键集/内部 forward 精度复查/object NPZ 与条件契约 → AdamW 有效算术标志/精度读取耗时/runtime 检查自身耗时/checkpoint 独立架构契约/conditioner 与 FK 准入精度/CRC NPZ 收口 → AdamW 初始历史状态/新精度门耗时/参考构造截止与身份门/混合键类型），不可消费为本轮结论。**合成四门通过只证明 I 阶段管线机制与门实现本身在合成世界上自洽；不外推到真实 W1 训练窗（真实可达性属 R/D1/D6 的授权后问题）。**

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

## 17. 第十轮独立审阅修订（2026-11-10：3 P1 + 2 P2 全部修复）

第九轮修订经独立复审确认有效（attempt_010 四份证明、保存量、调用矩阵、交接及 76/76 电池记录成立，登记成本 1468.88s≤3300s；第九轮原攻击重放全部拦截），但新隔离反例复现 3 P1 + 2 P2（复现脚本 `/tmp/opencode/review_i_round9_new_boundaries_20261109.py`、结果 `/tmp/opencode/review_i_round9_new_boundaries_20261109_results.json`；前轮攻击与权威核验分别在 `/tmp/opencode/review_i_round9_previous_attacks_20261109_results.json`、`/tmp/opencode/review_i_round9_authority_20261109_results.json`）。全部接受并修复，电池扩至 **81 项**，定点验证脚本 `/tmp/opencode/verify_round10_fixes.py`、结果 `/tmp/opencode/verify_round10_fixes_results.json`。attempt_010 自本轮起为被取代证据。

| # | 复审发现 | 修复 |
| --- | --- | --- |
| P1-1 | 有效配置绑定不完整：checkpoint 的 loss 权重缺失或改为 `w_fk=999`、重签文件链后本地核验仍 `problems=[]`、F 仍 `complete_evidence/replay_passed=true`；驱动也接受实际 `w_fk=999` 的目标 | 新增 `LOSS_WEIGHT_FIELDS`/`LOSS_WEIGHT_EXPECTED`（w_pos/w_rot6d/w_joint=1、w_geodesic=0.5、w_fk=2、w_contact=0、w_vel=0.5，bool 伪造与额外字段拒绝）与 `_loss_weight_problems`：驱动构造同时核对 `trainer.cfg.loss`（`config_loss`）与**实际 loss assembler** `loss_fn.w`（`loss_weight_binding`）；checkpoint 内嵌 `config.loss` 缺失即 `checkpoint_config_loss_missing`、逐字段不符即 `checkpoint_config_mismatch`（本地核验与 F 的身份 helper 同契约）；`frozen_plan_defaults_problems` 增 `loss_weight_defaults` |
| P1-2 | F 身份锚可被协调改写：同步改 `stage_report.identity` 与两 checkpoint identity 并重签后，本地核验无问题、F 仍 `complete_evidence/replay_passed=true`；F 只把报告自身 identity 当锚 | 驱动把 **plan/registration 的规范 SHA-256**（`plan_sha256`/`registration_sha256`）嵌入写入的阶段身份（调用方冲突字段直接拒绝 `identity_binding_conflict`）；F/交接/本地核验按**调用方给出的 plan/registration** 重算摘要并逐项对账（`stage_identity_binding_mismatch`）。协调改写报告+checkpoint 身份不再自洽；同一绑定也拒绝“报告 plan、checkpoint config 与 identity 协同改 lr”的形态 |
| P1-3 | backbone 后的截止守卫过早：`Trainer.predict` 内部输出合法性检查期间越过工作截止，之后 loss 与 2 次 FK 仍启动，最终才在 pre_backward 停止 | 驱动对 `Trainer.predict` 做**实例级包装**（共享 Trainer 源不变）：在其返回后、loss 之前执行 `{micro/probe point}_post_predict`，越线不再启动 loss 及其 FK 链；模型 forward 返回后的 `_pre_loss_gate` 保留为更早的边界 |
| P2-4 | 被拒的 backbone 仍计入调用数：前向守卫拒绝模型执行，但 pre-hook 已增加 `backbone_forward_calls`；返回、失败快照与盘上报告均多计 1 次 | `_forward_hook_counter` 改为 **post-forward hook**：只在 `forward` 实际返回后计数；被 admission 守卫拒绝的 backbone 不再出现在返回、失败快照或盘上报告 |
| P2-5 | 嵌套坏结构仍可让 F 崩溃：`metrics` 为列表/嵌套列表或 checkpoint `root_stats` 为错误映射时抛 TypeError，未生成 f_report 或返回证明 | `verify_stage_local` 对 saved `metrics` 先校验映射（非映射→`probe_metrics_not_a_mapping`）；新增 `_checkpoint_root_stats_problems` 在 float32 转换前校验容器/数值类型（坏形态→`checkpoint_root_stats_unreadable`），本地核验与 F 身份 helper 共用；两类坏容器均落 incomplete f_report + return_observation |

新增 5 项电池反例：`loss_weight_binding_rejected`（checkpoint `w_fk=999`/`w_contact=1`/缺 `loss` 块、驱动实际目标 `w_fk=999`、冻结计划元数据省略均被拒；三 checkpoint 形态 F 均 incomplete 且报告/证明落盘）、`stage_identity_external_anchor`（协同改报告+两 checkpoint 身份并被 plan/registration 摘要拒绝；F incomplete+报告落盘）、`post_backbone_validation_tail_stop`（`Trainer.predict` 输出检查越线→loss 与 FK 零启动、backward/step=0）、`refused_backbone_not_counted`（post-hook 实际返回后计数；被拒 backbone 在返回、失败快照与盘上报告均不多计）、`bad_metrics_and_root_stats_report_incomplete`（metrics 列表/嵌套列表与 root_stats 错误映射均落 incomplete F 报告+返回证明）。

定点复现（修复后，`verify_round10_fixes.py`）实测：控制 D/F 全过；loss 权重四种省略/篡改形态与驱动实际目标均被拒（三 checkpoint 形态 F `complete_evidence=false`、`verdict=incomplete`、报告与证明存在）；协同身份改写得到 `stage_identity_binding_mismatch:plan_sha256`/`registration_sha256` 且 F incomplete；predict 输出检查越线时 `validity_in_predict=1`、训练内层 loss 调用 **0**、晚 loss/晚 FK **0**、失败点 `step_1_micro_1_post_predict`、backward/step=0；被拒 backbone 形态实际 conditioner=2/backbone=1，返回、失败快照与盘上报告均为 conditioner=2/backbone=1（修复前 backbone 多计）；三类坏容器均无异常、f_report 与返回证明落盘、`verdict=incomplete`。

**权威合成验收 `derived/audit/w1_smallset_fit_diagnostic/attempt_011/`**（2026-11-10，第十轮修复后代码，独占创建，CLI exit 0）：封口 `sealed_within_budget`（round 1）、登记成本 **1466.78s** ≤ 3300s、返回覆盖 covered=true（elapsed 1462.78 ≤ 1466.78）；微型 D1（300 步，登记成本 236.11s，返回覆盖 covered（232.09））四门全过（RMSE 0.00115m / 0.00288rad / 0.00186rad / FK 0.00121m）→ `handoff_ready=true`（候选经 D6 入口消费，交接声明前复查通过）；微型 D6（登记成本 1385.76s，返回覆盖 covered（1381.76））六窗四门全过（逐窗 true，macro RMSE 0.00633m）；F 两阶段逐位重放通过（`fit_pass`、`claim_deadline_exceeded` 未触发、F 自身返回覆盖 covered=true）；**81/81 反例捕获**。`attempt_010` 保留为被取代证据（其证据不含第十轮的七项 loss 权重绑定、身份外部锚、predict 后守卫、被拒 backbone 计数与 metrics/root_stats 坏容器修复项）。

**边界（与前轮一致）：R/真实 D1/D6/F 未实施、未授权、未运行；真实拟合仍须 §8.3 新冻结材料与新明确授权；T6 旧结论与 partial/null/false/not_ready/不接入全部保持；本轮未运行全项目套件（仅本交付 17 passed + N0/N1 回归 33 passed）。**

## 18. 第十一轮独立审阅修订（2026-11-10：3 P1 + 2 P2 全部修复）

第十轮修订经独立复审确认有效（attempt_011 四份证明、保存量、调用矩阵、交接及 81/81 电池记录成立，登记成本 1466.78s≤3300s；第十轮原攻击重放全部拦截），但新隔离反例复现 3 P1 + 2 P2（复现脚本 `/tmp/opencode/review_i_round10_new_boundaries_20261110.py`、结果 `/tmp/opencode/review_i_round10_new_boundaries_20261110_results.json`）。全部接受并修复，电池扩至 **87 项**，定点验证脚本 `/tmp/opencode/verify_round11_fixes.py`、结果 `/tmp/opencode/verify_round11_fixes_results.json`。attempt_011 自本轮起为被取代证据。

| # | 复审发现 | 修复 |
| --- | --- | --- |
| P1-1 | 身份仍可协调改写：保留两个摘要字段，协调修改报告与 checkpoint 的 `stage/module/implementation_sha256` 后，本地仍 `problems=[]`、F 仍完整通过；F 未传完整外部 `expected_identity` | `run_f_verification` 的每个 stage spec 与 `resolve_handoff` 现在**必须**携带调用方声明的完整 `expected_identity`（缺失→结构化 `stage_identity_not_declared`/`handoff_identity_not_declared`）；`verify_stage_local`/`_checkpoint_binding_problems`/`_f_identity_problems` 逐字段比较报告与两 checkpoint 身份与外部声明（`stage_report_identity_mismatch`/`checkpoint_identity_mismatch`），plan/registration 摘要仅作派生绑定；合成验收与全部电池调用点显式传入驱动身份 |
| P1-2 | 实际 loss 权重仅在构造时检查：构造后、运行前把实际 `loss_fn.w_w_fk` 改成 999，四个 micro 均实际使用 999，checkpoint 配置仍记 2，D/F 却声明完整 | 新增 `_check_loss_weights`：在 `_preflight`、每次 probe、每个 micro 开始、每次 checkpoint 保存与 `CountingLoss` 实际调用前复核活的 `loss_fn.w`（`loss_weight_drift`）；构造后漂移在 preflight 即失败，零 micro 消费，D 封口 incomplete、F skip |
| P1-3 | loss 内部 FK 链缺截止守卫：首个预测解码返回越线后仍启动两次 FK；首个 FK 返回越线时仍启动教师解码及下一 FK | `CountingFK` 增加 pre/post 回调、`decode_clip` 以实例级包装替代：每次 decode/FK 前后各查工作截止；首解码返回越线→零 FK，首 FK 返回越线→零教师解码/下一 FK（失败为 BudgetExceeded、backward/step=0） |
| P2-4 | checkpoint 配置坏容器仍使 F 无报告退出：`config.loss.w_fk` 为多元素 Tensor、`config.lr` 为多元素 ndarray 分别触发 RuntimeError/ValueError | 新增 `_json_value_equal`：类型严格 JSON 值比较（Tensor/ndarray/畸形容器一律不匹配，绝不把多元素相等放进 `if`）；`_checkpoint_config_problems`/`_loss_weight_problems`/身份比较统一使用，坏形态正常落 incomplete f_report+返回证明 |
| P2-5 | post-hook 漏计已执行后抛错的 forward：实际调用 2 次，返回/失败快照/盘上报告只记 1 | 计数改回 **forward pre-hook** 并注册在准入守卫之后：准入拒绝不计、已进入（即使 forward 内部随后抛错）照常计数；实际=返回=失败快照=盘上 |
| P2-6 | 整数/布尔严格性遗漏：checkpoint step 改为布尔/浮点、forward 次数改为 `true`、最坏坐标改为浮点，三个变体均被本地与 F 接受 | checkpoint step 必须 `_is_json_int`；`forward_counts` 的 conditioner/backbone 必须是整数 1（bool 拒绝）；`_metrics_equal` 对整数/列表逐元素类型严格（`metric_int_type_or_value`/`metric_list_type_or_value`），三变体均落 incomplete |

新增 6 项电池反例：`loss_internal_fk_units_deadline_stop`（loss 内部 decode→FK 四单元前后截止守卫）、`live_loss_weight_drift_refused`（构造后实际权重漂移零 micro 消费）、`stage_identity_full_external_anchor`（完整外部身份锚拒绝协同改写、缺声明身份结构化 skip）、`bad_checkpoint_config_containers_report_incomplete`（Tensor/ndarray 配置值）、`executed_failing_backbone_counted`（已进入后抛错的 forward 计数）、`typed_counts_and_coordinates_rejected`（step/forward_counts/worst_coordinate 严格类型）。

定点复现（修复后，`verify_round11_fixes.py`）实测：控制 D/F 全过；完整身份改写得到 `stage_report_identity_mismatch`+两 `checkpoint_identity_mismatch`，F incomplete 且报告/证明存在，未声明身份的 F skip 为 `stage_identity_not_declared`；构造后 `w_fk=999` 漂移在 preflight 失败（`loss_weight_drift`），训练内层 loss 调用 0 次、D/F incomplete；loss 内部链在 `pred_decode` 跳变时仅 1 次解码、0 次 FK，在 `pred_fk` 跳变时 1 次解码+1 次 FK，两例零晚解码/晚 FK、BudgetExceeded、backward/step=0（无注入控制 8 解码/8 FK、complete）；执行后抛错的 forward 实际 2 次、返回/失败快照/盘上均 2 次；Tensor/ndarray 配置值与三类严格类型变体均无异常、f_report 与返回证明落盘、`verdict=incomplete`。

**权威合成验收 `derived/audit/w1_smallset_fit_diagnostic/attempt_012/`**（2026-11-10，第十一轮修复后代码，独占创建，CLI exit 0）：封口 `sealed_within_budget`（round 1）、登记成本 **1465.12s** ≤ 3300s、返回覆盖 covered=true（elapsed 1461.13 ≤ 1465.12）；微型 D1（300 步，登记成本 237.01s，返回覆盖 covered（233.00））四门全过（RMSE 0.00115m / 0.00288rad / 0.00186rad / FK 0.00121m）→ `handoff_ready=true`（候选经 D6 入口消费，交接声明前复查通过）；微型 D6（登记成本 1383.58s，返回覆盖 covered（1379.57））六窗四门全过（逐窗 true，macro RMSE 0.00633m）；F 两阶段逐位重放通过（`fit_pass`、`claim_deadline_exceeded` 未触发、F 自身返回覆盖 covered=true）；**87/87 反例捕获**。`attempt_011` 保留为被取代证据（其证据不含第十一轮的完整外部身份锚、运行期损失权重复核、loss 内部 decode/FK 守卫、进入即计数与严格整数/坏配置容器修复项）。

**边界（与前轮一致）：R/真实 D1/D6/F 未实施、未授权、未运行；真实拟合仍须 §8.3 新冻结材料与新明确授权；T6 旧结论与 partial/null/false/not_ready/不接入全部保持；本轮未运行全项目套件（仅本交付 17 passed + N0/N1 回归 33 passed）。**

## 19. 第十二轮独立审阅修订（2026-11-10：2 P1 + 3 P2 全部修复）

第十一轮修订经独立复审确认有效（attempt_012 四份证明、保存量、调用矩阵、交接及 87/87 电池记录成立，登记成本 1465.12s≤3300s；第十一轮原攻击重放全部拦截），但新隔离反例复现 2 P1 + 3 P2（复现脚本 `/tmp/opencode/review_i_round11_new_boundaries_20261110.py`、结果 `/tmp/opencode/review_i_round11_new_boundaries_20261110_results.json`）。全部接受并修复，电池扩至 **92 项**，定点验证脚本 `/tmp/opencode/verify_round12_fixes.py`、结果 `/tmp/opencode/verify_round12_fixes_results.json`。attempt_012 自本轮起为被取代证据。

| # | 复审发现 | 修复 |
| --- | --- | --- |
| P1-1 | 实际 AdamW 参数组未绑定冻结配置：分别只改实际 `optimizer.param_groups` 的 `lr=0.125` 或 `weight_decay=0.25`，更新确实使用修改值；checkpoint 仍记录原配置，D/F 均声明证据完整 | 新增 `_optimizer_binding_problems`：逐组核对**实际 `trainer.opt.param_groups`** 的 lr/weight_decay 与冻结 plan（`optimizer_group_mismatch`）；构造时、preflight、每次 micro、clip 取证后、step 前与 checkpoint 保存前均经 `_check_runtime_bindings`（同时复核 cfg 与活的 loss 权重）。篡改后零 `opt.step`、D 封口 incomplete、F skip |
| P1-2 | conditioner/clip 前仍有截止空档：predict 输入检查返回到 91s 后仍启动一次 conditioner；clip 取证返回到 91s 后仍启动一次 clip | conditioner 增加独立 `forward_pre_hook` 截止门（`*_pre_conditioner`，注册在计数钩子之前：拒绝不计、进入即计）；`_check_clip_coverage_pre` 返回后新增 `step_{n}_clip_coverage_pre` 预算检查，越线不再启动 `clip_gradients` |
| P2-3 | F 仍漏计已进入后抛错的 forward：conditioner/backbone 执行数值 body 后抛错，实际各 1 次，返回与盘上计数均为 0 | `_fresh_probe` 的 conditioner/backbone 计数移到**实际调用入口**（conditioner 调用前、`f_fresh_backbone` 检查通过后且 backbone 调用前）：已进入的调用即使随后抛错仍计数；准入拒绝不计数 |
| P2-4 | 配置比较可抛溢出异常：checkpoint 的 `lr` 或 `w_fk` 改为整数 `10**400`，`float(got)` 抛未捕获 OverflowError，无 F 报告及返回证明 | `_json_value_equal` 的 float 分支捕获 `OverflowError`/`ValueError` 并返回不匹配；巨大整数配置值经类型严格比较进入 incomplete 收口 |
| P2-5 | 坏 NPZ dtype 使 F 无报告退出：raw 预测或 FK 数组改成字符串 dtype 后，虽已记录 dtype 不符仍调用 `np.isfinite`，触发 TypeError | `_check_record` 仅对浮点 dtype 执行 isfinite（字符串 dtype 只记 `record_dtype`）；`MetricDecoder.verify_x0` 在 dtype 不符时立即返回；坏 NPZ 经既有 `probe_records_invalid` 收口为 incomplete 报告+证明 |

新增 5 项电池反例：`effective_optimizer_binding_rejected`、`conditioner_and_clip_pre_boundaries_stop`、`f_entered_failed_forwards_counted`、`huge_config_scalars_report_incomplete`、`bad_npz_dtype_reports_incomplete`。

定点复现（修复后，`verify_round12_fixes.py`）实测：控制 D/F 全过；实际 param_groups 改 lr/weight_decay 后 `opt.step` 调用 0 次、失败为 `optimizer_binding:optimizer_group_mismatch:0:...`、D/F incomplete；predict 输入检查越线后训练 conditioner 调用 0 次、`BudgetExceeded`、step=0；clip 取证越线后 clip 调用 0 次、`BudgetExceeded`、step=0（无注入控制 conditioner=4/clip=1、D complete）；F 的 conditioner/backbone 执行后抛错均为实际=返回=盘上=1、F incomplete；`10**400` 的 lr/w_fk 与 raw/FK 字符串 dtype 四形态均无异常、f_report 与返回证明落盘、`verdict=incomplete`。

**权威合成验收 `derived/audit/w1_smallset_fit_diagnostic/attempt_013/`**（2026-11-10，第十二轮修复后代码，独占创建，CLI exit 0）：封口 `sealed_within_budget`（round 1）、登记成本 **1462.59s** ≤ 3300s、返回覆盖 covered=true（elapsed 1458.59 ≤ 1462.59）；微型 D1（300 步，登记成本 232.07s，返回覆盖 covered（228.06））四门全过（RMSE 0.00115m / 0.00288rad / 0.00186rad / FK 0.00121m）→ `handoff_ready=true`（候选经 D6 入口消费，交接声明前复查通过）；微型 D6（登记成本 1377.96s，返回覆盖 covered（1373.94））六窗四门全过（逐窗 true，macro RMSE 0.00633m）；F 两阶段逐位重放通过（`fit_pass`、`claim_deadline_exceeded` 未触发、F 自身返回覆盖 covered=true）；**92/92 反例捕获**。`attempt_012` 保留为被取代证据（其证据不含第十二轮的实际 AdamW 参数组绑定、conditioner/clip 前截止守卫、F 入口计数、溢出安全比较与坏 NPZ dtype 修复项）。

**边界（与前轮一致）：R/真实 D1/D6/F 未实施、未授权、未运行；真实拟合仍须 §8.3 新冻结材料与新明确授权；T6 旧结论与 partial/null/false/not_ready/不接入全部保持；本轮未运行全项目套件（仅本交付 17 passed + N0/N1 回归 33 passed）。**

## 20. 第十三轮独立审阅修订（2026-11-10：4 P1 + 2 P2 全部修复）

第十二轮修订经独立复审确认有效（attempt_013 四份证明、保存量、调用矩阵、交接及 92/92 电池记录成立，登记成本 1462.59s≤3300s；第十二轮原攻击重放全部拦截），但新隔离反例复现 4 P1 + 2 P2（复现脚本 `/tmp/opencode/review_i_round12_new_boundaries_20261110.py`、结果 `/tmp/opencode/review_i_round12_new_boundaries_20261110_results.json`）。全部接受并修复，电池扩至 **98 项**；attempt_013 自本轮起为被取代证据。

| # | 复审发现 | 修复 |
| --- | --- | --- |
| P1-1 | 实际优化器绑定仍不完整：分别修改 `betas`、`eps`、`maximize=True`，D/F 仍完整；移除 optimizer 中全部 12 个 conditioner 参数后它们有非零梯度却完全未更新，D/F 也通过 | `_optimizer_binding_problems` 扩展为逐组核对 lr/weight_decay/**betas/eps/maximize/amsgrad**，并核对 **model+conditioner 完整参数名册**（id 集合/数量，`optimizer_parameter_roster_mismatch`）；所有运行期边界同契约复核；篡改后零 `opt.step` |
| P1-2 | 新增绑定检查之后仍有截止空档：`_check_runtime_bindings` 返回推进至 91s 后仍分别启动一次 clip、一次实际 AdamW step | 在 pre-clip 与 pre-step 的绑定读取之后各加预算检查（`step_{n}_clip_binding_read`/`step_{n}_step_binding_read`，save 同理），绑定读取耗时计入被检窗口后再准入 clip/step |
| P1-3 | 损坏 checkpoint 仍可交接：隔离副本中将参数改为 NaN 或删除 `net.root_in.weight` 并重签工件链后，本地 `problems=[]`、`handoff_ready=true`；只信任自报 `parameters_finite=true` | `_checkpoint_state_problems` 扫描 model/conditioner **实际张量**有限性（`checkpoint_*_nonfinite`）与 state key set，并在 initial/last 两端点做键集相等绑定（`checkpoint_state_key_set_mismatch`）；本地核验、F 身份 helper 与交接共用；自报标志仅作附加检查 |
| P1-4 | 内部 forward 边界未复查精度：conditioner 返回时开启 TF32，D/F 仍继续一次 backbone 后才拒绝 | conditioner→backbone 的 `_pre_backbone_gate`、backbone→loss 的 `_pre_loss_gate` 与 F `_fresh_probe` 的 conditioner 后/backbone 后边界均调用 `runtime_guard.check_unit`；真实 `ProductionRuntimeGuard` 下 TF32 注入后 backbone 零启动，D/F incomplete |
| P2-5 | object dtype NPZ 使 F 无报告退出：raw/FK 改 object 数组，读取阶段抛未捕获 ValueError；零构造但无报告/证明 | `read_npz_records` 捕获 NpzFile 的 ValueError/OSError/TypeError 并转 `FitDiagnosticError(record_archive_unreadable:...)`；**不启用 pickle**，坏存档进入 incomplete 报告+证明 |
| P2-6 | 条件数组缺少读取前契约：删除 `contact` 或改为字符串 dtype，分别抛 KeyError/TypeError；已构造 fresh 模块仍无报告/证明 | F 输入重建对每个 condition 先查存在性、再校验 **float32/冻结形状/有限性**（`condition_missing`/`condition_dtype`/`condition_shape`/`condition_nonfinite`，并包装 NpzFile 异常），失败经既有 `fresh_replay_error` 收口为 incomplete 报告+证明 |

新增 6 项电池反例：`optimizer_extended_binding_rejected`、`binding_read_boundaries_stop`、`checkpoint_tensor_facts_bound`、`runtime_drift_between_numeric_units_stop`、`object_npz_reports_incomplete`、`condition_contract_reports_incomplete`。

定点复现（电池内）实测：betas/eps/maximize/省略 conditioner 参数四变体均零 `opt.step`、`optimizer_binding` 拒绝、D/F incomplete；clip/step 绑定读取越线后对应调用零启动、`BudgetExceeded`、step=0（控制例 complete）；checkpoint NaN 参数得到 `nonfinite`、删除 state key 得到 `state_key_set_mismatch`，本地与交接均拒绝（未篡改控制例本地零问题）；真实 ProductionRuntimeGuard 下 D/F 控制例 complete 且 TF32 下 backbone=0，注入后 `injected=1`、`backbone_under_tf32=0`、D/F incomplete；object NPZ 与 contact 缺失/字符串 dtype 四形态均无异常、f_report 与返回证明落盘、`verdict=incomplete`。

**权威合成验收 `derived/audit/w1_smallset_fit_diagnostic/attempt_014/`**（2026-11-10，第十三轮修复后代码，独占创建，CLI exit 0）：封口 `sealed_within_budget`（round 1）、登记成本 **1443.87s** ≤ 3300s、返回覆盖 covered=true（elapsed 1439.87 ≤ 1443.87）；微型 D1（300 步，登记成本 231.40s，返回覆盖 covered（227.39））四门全过（RMSE 0.00115m / 0.00288rad / 0.00186rad / FK 0.00121m）→ `handoff_ready=true`（候选经 D6 入口消费，交接声明前复查通过）；微型 D6（登记成本 1356.14s，返回覆盖 covered（1352.12））六窗四门全过（逐窗 true，macro RMSE 0.00633m）；F 两阶段逐位重放通过（`fit_pass`、`claim_deadline_exceeded` 未触发、F 自身返回覆盖 covered=true）；**98/98 反例捕获**。`attempt_013` 保留为被取代证据（其证据不含第十三轮的 AdamW 扩展字段与参数名册、绑定读取守卫、checkpoint 实际张量与键集、内部 forward 精度复查、object NPZ 与条件契约修复项）。

**边界（与前轮一致）：R/真实 D1/D6/F 未实施、未授权、未运行；真实拟合仍须 §8.3 新冻结材料与新明确授权；T6 旧结论与 partial/null/false/not_ready/不接入全部保持；本轮未运行全项目套件（仅本交付 17 passed + N0/N1 回归 33 passed）。**

## 21. 第十四轮独立审阅修订（2026-11-10：4 P1 + 1 P2 全部修复）

第十三轮修订经独立复审确认有效（attempt_014 四份证明、保存量、调用矩阵、交接及 98/98 电池记录成立，登记成本 1443.87s≤3300s；第十三轮原攻击重放全部拦截），但新隔离反例复现 4 P1 + 1 P2（复现脚本 `/tmp/opencode/review_i_round13_new_boundaries_20261110.py`、结果 `/tmp/opencode/review_i_round13_new_boundaries_20261110_results.json`）。全部接受并修复，电池扩至 **105 项**；attempt_014 自本轮起为被取代证据。

| # | 复审发现 | 修复 |
| --- | --- | --- |
| P1-1 | AdamW 有效算术字段仍未绑定：仅把实际参数组的 `decoupled_weight_decay` 置 false（实际更新改变 47 个参数张量）或把 `foreach`/`fused` 置 true，D/F 仍完整、端点重放仍通过 | `ADAMW_ARITHMETIC_FLAGS` 冻结 `decoupled_weight_decay=true`、`foreach=None`、`fused=None`、`capturable=false`、`differentiable=false`，与实际 `param_groups` 的 lr/wd/betas/eps/maximize/amsgrad 和参数 id 名册同契约逐组复核（`optimizer_group_mismatch`）；构造/preflight/每 micro/clip 取证后/step 前/save 前同契约复核，篡改后零 `opt.step` |
| P1-2 | 精度检查自身耗时未覆盖准入：工作截止 90s 时 runtime 检查返回才把时钟推进至 91s，D 仍启动一次 backbone；F 在 `f_probe_initial_inputs_ready` 的 runtime 检查后仍启动一次 conditioner 才停止 | 每个 runtime 检查后紧接预算复查：`_pre_conditioner/_pre_backbone/_pre_loss` 分别新增 `*_post_runtime`，F `_fresh_probe` 入口新增 `f_fresh_entry`（预算+runtime）、backbone 完成后新增 `f_fresh_backbone_complete` 预算复查；慢精度读取不再买到一次晚调用 |
| P1-3 | checkpoint 两端自洽不等于架构合法：两端同时删除同一参数键、或把 last 参数改为错误形状/int32，重签后均 `problems=[]`、`handoff_ready=true` | 新增 `_plan_state_contract`：由**冻结 plan**（model_type/d_model/n_layers/n_heads/ffn_mult/time_window）用生产构造器构建参考模块，派生每侧**完整键集+shape+dtype** 契约（预算守卫、`torch.random` 状态隔离、无 forward/优化器、按架构缓存）；`_checkpoint_architecture_problems` 对两端逐项比较（`*_key_set_not_frozen`/`*_shape_not_frozen`/`*_dtype_not_frozen`），本地核验与交接共用；F 的 strict 加载前契约（fresh 模块）继续拒绝同类篡改 |
| P1-4 | conditioner/FK 准入仍缺精度复查：输入检查后开启 TF32 仍执行一次 conditioner；生产解码后开启 TF32，D/F 分别继续执行 2 次/1 次 FK 才拒绝 | `_pre_conditioner_gate` 增 runtime 复查；loss 内部每次 FK 的 pre-call 增 runtime 复查（`*_loss_fk_pre`），损失解码后也复查（`*_loss_decode_pre/_post`）；F `_fresh_probe` 入口与逐窗 FK 前各增 runtime 复查（`f_fresh_entry`/`f_fresh_fk_w*`）；真实 `ProductionRuntimeGuard` 下 TF32 注入后 conditioner=0、FK=0，D/F incomplete |
| P2-5 | CRC 损坏的 NPZ 仍使 F 无报告退出：翻转 raw 或 contact 成员的一个字节（ZIP 目录完好）触发未捕获 `BadZipFile`，分别零构造/一次构造但均无 f_report 与返回证明 | `read_npz_records` 与 F condition 读取的异常集合加入 `zipfile.BadZipFile`/`EOFError`/`zlib.error`，CRC 损坏转为结构化 `record_archive_unreadable`/`condition_archive_unreadable`，经既有收口落 incomplete 报告+返回证明；不启用 pickle |

新增 7 项电池反例：`optimizer_arithmetic_flags_rejected`、`runtime_check_time_charged`、`f_runtime_check_time_charged`、`checkpoint_architecture_contract_bound`、`conditioner_and_fk_precision_admission`、`f_fresh_fk_precision_admission`、`crc_corrupt_npz_reports_incomplete`。

定点复现（隔离脚本 `/tmp/opencode/verify_round14_fixes.py`，结果 `verify_round14_fixes_results.json`）实测：三个算术标志篡改均为 `optimizer_binding` 拒绝、零 step、D incomplete；D 慢 runtime 读取得到 `pre_backbone_post_runtime`、晚 backbone=0；F 慢 runtime 读取得到 `f_fresh_entry`、conditioner=0、报告+证明落盘；checkpoint 同删键/错 shape/int32 分别得到 `key_set_not_frozen`/`shape_not_frozen`/`dtype_not_frozen`，本地 mean=null、交接拒绝、F incomplete；TF32 在输入检查后注入使 conditioner 零执行（`pre_conditioner`），在生产解码后注入使 FK 零执行（`loss_decode_post`），F fresh 解码后注入使 fresh FK 零执行；CRC 两形态均 `error=None`、报告+证明落盘、`verdict=incomplete`。电池内同一组反例作为 `test_counterexample_battery_all_caught` 的一部分运行；I 测试 + N0/N1 回归 **50 passed（204.96s）**。

**权威合成验收 `derived/audit/w1_smallset_fit_diagnostic/attempt_015/`**（2026-11-10，第十四轮修复后代码，独占创建，CLI exit 0）：封口 `sealed_within_budget`（round 1）、登记成本 **1441.70s** ≤ 3300s、返回覆盖 covered=true（elapsed 1437.70 ≤ 1441.70，summary_sha256 `b9513c3c…`）；微型 D1（300 步，登记成本 231.12s，返回覆盖 covered（227.12））四门全过（RMSE 0.00115m / 0.00288rad / 0.00186rad / FK 0.00121m）→ `handoff_ready=true`（候选经 D6 入口消费，交接声明前复查通过）；微型 D6（登记成本 1350.85s，返回覆盖 covered（1346.85））六窗四门全过（逐窗 true，macro RMSE 0.00633m）；F 两阶段逐位重放通过（`fit_pass`、`claim_deadline_exceeded` 未触发、F 自身返回覆盖 covered=true）；**105/105 反例捕获**。`attempt_014` 保留为被取代证据（其证据不含第十四轮的 AdamW 有效算术标志、runtime 检查耗时复查、checkpoint 独立架构契约、conditioner/FK 准入精度、CRC NPZ 收口修复项）。

**边界（与前轮一致）：R/真实 D1/D6/F 未实施、未授权、未运行；真实拟合仍须 §8.3 新冻结材料与新明确授权；T6 旧结论与 partial/null/false/not_ready/不接入全部保持；本轮未运行全项目套件（仅本交付 17 passed + N0/N1 回归 33 passed，合并 50 passed）。**

## 22. 第十五轮独立审阅修订（2026-11-10：3 P1 + 1 P2 全部修复）

第十四轮修订经独立复审确认有效（attempt_015 四份证明、保存量、调用矩阵、交接及 105/105 电池记录成立，登记成本 1441.70s≤3300s；第十四轮原攻击重放全部拦截），但新隔离反例复现 3 P1 + 1 P2（复现脚本 `/tmp/opencode/review_i_round14_new_boundaries_20261110.py`、结果 `/tmp/opencode/review_i_round14_new_boundaries_20261110_results.json`）。全部接受并修复，电池扩至 **110 项**；attempt_015 自本轮起为被取代证据。

| # | 复审发现 | 修复 |
| --- | --- | --- |
| P1-1 | AdamW 初始历史状态未绑定：保持初始参数、超参数与名册不变，仅预置一个参数的 `step=17`+非零 `exp_avg/exp_avg_sq`，实际更新 17→18、参数较控制差 0.00283，D/F 仍完整、端点重放通过 | 新增 `_optimizer_state_problems(opt, expected_step)`：构造与每次运行期复核要求 **step-0 空状态**（`optimizer_state_not_empty`），之后每个已填充条目的 `step` 精确等于驱动已完成的 native step 数（`self._completed_steps`，每执行一步恰加一）且 `exp_avg/exp_avg_sq` 形状对齐、有限、`exp_avg_sq` 非负；`_check_runtime_bindings`（preflight/每 micro/clip 取证后/step 前/save 前）与每次 `step_optimizer` 返回后同契约复核（`optimizer_state_binding`/`optimizer_state_drift`），预置状态在 preflight 即零 step 拒绝 |
| P1-2 | 新增 runtime 门仍遗漏自身耗时：在 `loss_decode_pre`/`loss_fk_pre` 读取后越线，D 仍启动一次 decode/FK；F 在 `f_fresh_entry`/`f_fresh_fk_w0` 读取后越线，仍启动一次 conditioner/FK | 每个新增 runtime 读取之后紧接预算复查：`*_loss_decode_pre_post_runtime`、`*_loss_decode_post_post_runtime`、`*_loss_fk_pre_post_runtime`、F `f_fresh_entry_post_runtime` 与 `f_fresh_fk_w*_post_runtime`；晚 decode/FK/conditioner 均为 0 |
| P1-3 | 架构契约构造内部缺截止守卫：冷缓存下参考 model 构造返回至 91s 仍继续构造 conditioner；且身份核验失败时仍实际构造参考模块（`build_modules=0` 不等于实际零构造） | `_plan_fresh_modules` 的 model 与 conditioner 构造各自成单元（`saved_state_contract_model`/`_conditioner`，model 越线即拒绝 conditioner）；`verify_stage_local` 先把报告身份与 plan/registration 绑定核验结果记为 `identity_failed`，`_checkpoint_binding_problems` 对已拒绝身份**跳过参考构造**（checkpoint 级 identity 失败同样跳过）；参考构建保持 `torch.random` 状态隔离 |
| P2-4 | 混合类型 checkpoint 键使 F 无报告退出：临时副本增加整数键与字符串键并重签，`sorted()` 抛未捕获 `TypeError`，无 F 报告/返回证明 | `_checkpoint_architecture_problems` 在集合差与排序**之前**验证 key 均为字符串（`checkpoint_{side}_key_type_not_string`，与非字符串键的 repr 排序安全），混合键经既有 incomplete 收口落 F 报告+返回证明 |

新增 5 项电池反例：`optimizer_initial_state_bound`、`decode_fk_runtime_read_time_charged`、`f_fresh_runtime_read_time_charged`、`state_contract_construction_guarded`、`mixed_checkpoint_keys_report_incomplete`。

定点复现（隔离脚本 `/tmp/opencode/verify_round15_fixes.py`，结果 `verify_round15_fixes_results.json`）实测：预置 step-17/动量后 preflight 即 `optimizer_state_drift:['optimizer_state_not_empty:1']`、零 step、D incomplete（控制例 1 step 完整）；D 两个越线点分别得到 `loss_decode_pre_post_runtime`/`loss_fk_pre_post_runtime` 的 BudgetExceeded、晚 decode/晚 FK=0；F 两个越线点 conditioner/FK 零启动、报告+证明落盘、`verdict=incomplete`；冷缓存 model 越线只构造 model（conditioner 不构造、RNG 不变），身份被拒零构造事件，控制例 metadata+replay 两对构造且 F 完整；混合键得到 `checkpoint_model_key_type_not_string` 且 F incomplete 报告+证明、无 TypeError traceback。I 测试 + N0/N1 回归 **50 passed（219.63s）**。

**权威合成验收 `derived/audit/w1_smallset_fit_diagnostic/attempt_016/`**（2026-11-10，第十五轮修复后代码，独占创建，CLI exit 0）：封口 `sealed_within_budget`（round 1）、登记成本 **1495.72s** ≤ 3300s、返回覆盖 covered=true（elapsed 1491.73 ≤ 1495.72，summary_sha256 `7f1ed608…`）；微型 D1（300 步，登记成本 240.07s，返回覆盖 covered（236.07））四门全过（RMSE 0.00115m / 0.00288rad / 0.00186rad / FK 0.00121m）→ `handoff_ready=true`（候选经 D6 入口消费，交接声明前复查通过）；微型 D6（登记成本 1400.49s，返回覆盖 covered（1396.49））六窗四门全过（逐窗 true，macro RMSE 0.00633m）；F 两阶段逐位重放通过（`fit_pass`、`claim_deadline_exceeded` 未触发、F 自身返回覆盖 covered=true）；**110/110 反例捕获**。`attempt_015` 保留为被取代证据（其证据不含第十五轮的 AdamW 初始历史状态、新精度门耗时复查、参考构造截止与身份门、混合键类型修复项）。

**边界（与前轮一致）：R/真实 D1/D6/F 未实施、未授权、未运行；真实拟合仍须 §8.3 新冻结材料与新明确授权；T6 旧结论与 partial/null/false/not_ready/不接入全部保持；本轮未运行全项目套件（仅本交付 17 passed + N0/N1 回归 33 passed，合并 50 passed）。**

## 23. 第十六轮独立审阅修订（2026-11-11：4 P1 + 1 P2）及教师转换截止回归

在第十五轮代码 `199f3fd` 上，隔离反例实证了下列 4 P1 + 1 P2；原脚本 `/tmp/opencode/review_i_round15_new_boundaries_20261111.py` 及其结果 `review_i_round15_new_boundaries_20261111_results.json` 保留。五项修复分别提交，不改 Trainer 的原生更新算术：

| 项 | 已复现问题与修复 | 本地提交 |
| --- | --- | --- |
| P1 参数冻结/缺梯度 | 冻结 model 或 conditioner 参数后，参数仍在优化器名册但无梯度、无 AdamW 状态且不更新，D/F 仍声称完整。现运行时检查 `requires_grad`，联合 clip 前拒绝缺梯度并校验裁剪前后梯度存在性，step 后要求优化器状态名册覆盖全部参数；合法的**存在但为零**的梯度仍允许 | `8c0919c` |
| P1 pre-backward 截止 | pre-backward 的 runtime 检查返回时越过工作截止，仍实际执行一次 backward。现检查返回后再次检查预算；反例零晚 backward、零 step，控制例四次 backward、一次 step | `8c18210` |
| P1 教师参考精度 | 教师生产 decode 后注入 TF32，旧路径在下一个守卫前仍执行一次 FK。现逐窗在教师生产 decode 与 FK 前复核运行时精度，并在检查返回后复核截止；注入后 FK=0 | `c6e9a68` |
| P1 身份格式准入 | 报告身份缺失、list 或 null 已导致 incomplete，但冷缓存参考模型和 conditioner 仍被构造。现身份坏形态也设置失败标志，跳过参考构造；不把 `build_modules=0` 误当作所有构造均为零 | `acfdc5a` |
| P2 checkpoint 布局 | 等 shape/dtype 的稀疏 checkpoint 进入 `isfinite` 抛 `NotImplementedError`，F 无报告与返回证明。现本地核验及 strict 加载前拒绝非 `torch.strided`；COO/CSR 各侧反例均结构化 incomplete | `77dc221` |

**本轮引入并修复的额外回归：**`c6e9a68` 把教师目标 `.to(device)` 移到了先前的预算检查之前；受控时钟中 metric decode 返回至 91s（工作截止 90s）后仍启动一次目标转换。新增 `teacher_window_w*_target_to` 转换前检查，原 `teacher_window_w*_prod_decode` 检查继续位于转换返回之后、runtime 与生产 decode 之前；runtime 检查后的预算复查和 FK 精度门保留。修复单独提交 `aa30368`。回归测试先在修复前实测 **1 failed / 2 passed**（慢 metric 的晚转换=1），修复后慢 metric 晚转换=0；慢 `.to` 返回越线后 runtime/decode/FK=0；metric decode 已执行计数=1，正常控制完整。三个转换场景以及教师精度、D6/F 微型对照合计 **10 passed（23.73s）**。

**最终实现验证：**在 `aa30368` 上运行 `TMPDIR=/tmp/opencode uv run --locked pytest -q tests/test_w1_smallset_fit_review16.py tests/test_w1_smallset_fit_diagnostic.py tests/test_w1_n0_coordinate_flags.py tests/test_w1_n1_synthetic_numerics.py`，**75 passed（262.33s）**：第十六轮 25 个参数化测试、原 I 阶段 17 项（含 **115/115** 电池，ID 集与 `COUNTEREXAMPLE_IDS` 精确相等）、N0/N1 33 项。随后只把原测试文件文档字符串的电池数量从 110 更正为 115，未改测试逻辑或实现；额外轻量复查曾因测试名误写而 exit 4、未运行测试，更正名称后 **4 passed（6.38s）**。未运行全项目套件。

**隔离复验与证据保护：**`/tmp/opencode/verify_i_round16_final_20261111.py` 的结果 `verify_i_round16_final_20261111_results.json` 绑定 `aa30368`：五类原攻击均 fail-closed；正常微型 D/F 控制证据完整、F 重放通过但裁决为 **`fit_insufficient`**，并非拟合通过。独立转换边界脚本 `verify_i_round16_teacher_conversion_20261111.py` 及结果 `verify_i_round16_teacher_conversion_20261111_results.json` 绑定同一实现，验证两个越线场景、控制例、盘上报告及返回覆盖；原攻击与转换两组隔离运行均记录历史权威目录和受检源码哈希未变。转换脚本首次运行因测试脚本把 `budget()` helper 名覆盖为实例而报 `TypeError`，仅控制例在临时目录执行、未写结果；更名后重跑成功，非产品路径错误。旧四类攻击的独立重放 `verify_i_round16_previous_attacks_20261111_results.json` 绑定较早修复提交 `77dc221`，不是最终实现 `aa30368` 的重跑。

**历史临时结果更正：**复跑旧四类攻击时，旧脚本曾覆盖 `/tmp/opencode/review_i_round15_previous_attacks_20261111_results.json`；已从原完整日志 `review_i_round15_previous_attacks_20261111.log` 恢复原内容（SHA-256 `e5541f2880c89144806f556b49bc0610a80d31fb62dd98ca6f61ea430db55b51`，文件时间戳改变），新结果另存于上述 `verify_i_round16_previous_attacks_20261111_results.json`。仓库历史产物和 `attempt_016` 未改写。

**验收与授权边界（本节修复轮的历史快照）：**当时没有运行完整微型 D1→D6→F 合成验收或权威工件的 fresh 预测；`attempt_016` 的 `fit_pass`、110/110 和 1495.72s 仅对应第十五轮 `199f3fd`，不作为新实现的验收结果。后续 `attempt_017` 见 §24。R/真实 D1/D6/F 未实施、未授权、未运行；真实拟合仍须 §8.3 新冻结材料与新明确授权。T6 `audit_incomplete`/539 gaps、cohort 34 次更新/1347.195507s、一次性授权已消费，以及 `partial/null/false/not_ready/不接入` 保持；合成检查不证明真实拟合、泛化、碰撞、抓握或生产接入。

## 24. 当前实现合成验收运行（2026-11-11：数值链通过，外层计费证据未闭合）

按用户要求执行 `TMPDIR=/tmp/opencode uv run --locked python scripts/verify_w1_smallset_fit_diagnostic.py --wall-cap-s 3300 --reserve-s 300`，独占创建 `derived/audit/w1_smallset_fit_diagnostic/attempt_017/`；运行时 HEAD `799702b96986f8ce518381182faea09d4fe2e69d`、实现修复至 `aa30368`，工作区干净，相关实现及环境文件的 SHA 在运行前后未变。后台进程已结束，stdout 给出 `label_verdict=complete`，stderr 为空；**没有单独保存进程实际 wait 时间或可独立核验的退出码**。`attempt_016` 的 43 个文件在本次运行前后逐一 SHA-256 相等，文件数仍为 43；没有覆盖历史工件。

- **微型 D1**：CPU 合成世界、300 步、每 30 步 probe 共 11 次；覆盖 11/11，实际更新 300、backward/micro 1200、窗口消费 1200，conditioner/backbone 各 1211 次，调用矩阵无问题。最终四项为 **0.0011507113m / 0.0028825341rad / 0.0018577272rad / 0.0012062979m**，四门均过，`local_complete_evidence=true`；D1 交接 `handoff_ready=true`、`problems=[]`。
- **微型 D6**：同为 300 步/11 次 probe，覆盖 **66/66**，六窗四门逐窗均过，根位置 RMSE 六窗宏平均 **0.0063342440m**（宏平均不是门）。实际更新 300、backward/micro 1200、窗口消费 7200，conditioner/backbone 各 1211 次，调用矩阵无问题。D1/D6 的报告成本来自同一合成总时钟，D6 **1384.724s 是累计值，不应与 D1 237.038s 相加**。
- **F 与反例**：F 对两阶段 initial/last 各执行一次 fresh probe，每阶段 conditioner/backbone 各 2 次；报告 `replay_passed=true`、`complete_evidence=true`、`mean_fit_passed=true`、`verdict=fit_pass`，精度策略 `bitwise_no_tolerance`。115 个反例全部 `caught=true`，与冻结 ID 集合精确相等。教师/预测规范解码均报告重解码逐位绑定；**中间 FK 状态绑定仍为 `not_evaluated`，重放范围仅 initial/last，尖峰门仍 `not_evaluated`**。
- **四份进程内证明只读对账**：D1、D6、F、总报告均 `sealed_within_budget`，对应 `return_observation.json` 均 `covered=true`、summary SHA 精确匹配且满足 `elapsed ≤ registered ≤ cap`。分别为 **233.039≤237.038≤3300s**、**1380.726≤1384.724≤3300s**、**1.691≤5.690≤600s**、**1480.303≤1484.301≤3300s**；总 summary SHA `ac866168c3bf743109919c851b9c50dbc1b07542288c90064bbab200c97ccb3b`。这四项是**进程内覆盖检查的记录**，不是实测外层 `wait()` 证明。
- **独立只读检查**：解析报告及全部登记工件文件大小/SHA、探针与窗口键、300 条每步 micro/样本计数、实际/期望调用矩阵、四份证明和交接候选绑定；无额外模型/FK/训练调用。脚本与结果、运行前 `attempt_016` 清单及检查输出、stdout/stderr 副本位于独立的 `derived/audit/w1_smallset_fit_diagnostic_postcheck/attempt_017/`，**这些事后检查不计入合成阶段预算，也不证明阶段内做过额外工作**。

**尚未闭合的 §8.2 边界：**现有 CLI 的 `_T0` 在 Python 脚本中采样（重模块 import 之前，但在 `uv` 启动之后），且 `verify_return_coverage` 在该进程返回之前采样；本次没有父进程在启动 `uv` 前采样并在实际 `wait()` 返回时记录可与登记成本对账的证明。进程结束通知后的额外单调钟采样为 CLI t0 后 **1518.517s**，虽低于 3300s，但超过登记 **1484.301s**，因此不能用该迟采样证明真实返回被登记成本覆盖；它也**不证明真实返回超限**。该采样的工具输出转录保存在独立 postcheck 目录，未改写 `attempt_017` 的封口或声明。按证据缺失处理：数值链和进程内证明事实保留，**不宣称满足 §8.2 的外层实际返回覆盖，也不宣称严格完整合成验收已闭合**；不自动重跑或补记成本。后续须先审阅并补齐外层起点与实际 wait 覆盖机制及反例，再决定是否另开合成 attempt。

本次未改代码或运行全项目测试；上轮 `75 passed` 是 `aa30368` 上的既有回归，本次新运行的是上述合成验收及其 115 项电池。R/真实 D1/D6/F 未实施、未授权、未运行；真实拟合仍须 §8.3 新冻结及明确授权，T6 `audit_incomplete`/539 gaps、已消费授权与 `partial/null/false/not_ready/不接入` 不变。

## 25. 外层实际返回修复与新的合成验收（2026-11-12）

以下保留当时的实现和采样事实；该版“完成观察”发生在最终 stdout flush 之前，其完整计费声明已按 §26 更正。

本地提交 `b4d5729` 增加独立纯合成监督器，在启动 `uv` 前登记单调时钟，在实际 `wait()` 返回后保存退出码、返回时刻及封口/代码 SHA 绑定，并对登记成本和 3300s 硬截止做 fail-closed 对账。D1、D6、F 函数实际返回后再覆盖采样；D1 返回未覆盖不能交接。摘要与监督器磁盘裁决均是候选，最终完成观察在裁决文件写入并采样通过之后发出，并绑定其 SHA；观察缺失不构成完成证明。

相关组合回归 **89 passed（257.19s）**；该组合回归在外层完成观察细化前启动，最终外层测试另 **15 passed（4.63s）**，包含真实子进程延迟退出、实际 wait 超过登记成本、非零退出、缺失/错误证明、返回晚于登记成本、源漂移和慢写盘。新增测试首次单独运行时有 2 项因测试文件依赖其他文件加入 `src` 路径而导入失败，补齐测试自身路径后独立重跑 15 项全过；不涉及数值实现失败。启动 `attempt_018` 之前，为 `attempt_016` 与 `attempt_017` 各 43 文件建立 SHA 清单。

`TMPDIR=/tmp/opencode python3 scripts/run_w1_i_synthetic_acceptance.py --wall-cap-s 3300 --reserve-s 300` 完成纯 CPU 合成 `derived/audit/w1_smallset_fit_diagnostic/attempt_018/`，运行时 HEAD `b4d572992a8a4016605263d10520977261e8bacc`，工作进程和监督器 exit 0，stderr 均为空；源文件、入口脚本和依赖清单的运行前后 SHA 一致。

- **D1 与交接**：300 步、11/11 probe，四项平均门分别为 **0.0011507113m / 0.0028825341rad / 0.0018577272rad / 0.0012062979m**，均通过；`handoff_ready=true`。更新 300、backward/micro 1200、窗消费 1200，conditioner/backbone 各 1211 次，调用矩阵无问题。
- **D6 与 F**：300 步、**66/66** 窗行覆盖，六窗四门逐窗通过；根位置 RMSE 宏平均 **0.0063342440m**（宏平均不是门）。更新 300、backward/micro 1200、窗消费 7200，conditioner/backbone 各 1211 次。F 两阶段 initial/last 各一次 fresh probe，逐位重放与四门通过，`fit_pass`；**中间 FK 状态绑定和尖峰门仍为 `not_evaluated`，重放范围仅 initial/last**。
- **反例与时钟**：115 项反例全部捕获，ID 与冻结集合精确一致。D1 返回/观察/登记 **230.665908≤230.666156≤234.663422s**，D6 为 **1354.765862≤1354.766062≤1358.763733s**（D1 与 D6 使用同一总时钟，不能相加），F 自身为 **1.729788≤1.729915≤5.727993≤600s**。父进程 t0 早于 CLI 入口 **0.259984s**；总进程内观察 **1453.397201s**，uv 实际 wait 返回 **1454.380406s**，返回证明写后 **1454.428601s**，裁决文件写后观察 **1454.429314s**，登记 **1457.394136≤3300s**，全部落在覆盖链中。
- **证据与只读对账**：总摘要 SHA `b7894257236dab5cc3a0586eb151c82f8fe4815ce071b5b55d7795a917bed6ce`，外层返回证明 SHA `36755bbe3833178bcb843f35df71cb0dc7eea46f4df3f0a00bfc88ba87f6d7e2`，磁盘候选裁决 SHA `e37847ef40ec332ff68501d022528b8b6ef4836f6c7e582d82e48405da1ea5ce` 与完成观察一致。独立检查逐项核对摘要/证明 SHA、各阶段保存工件大小与 SHA、窗口键及 probe、300 条 micro/窗计数、调用矩阵、交接候选、阶段返回和实际 wait 成本；`attempt_016`、`attempt_017` 各 **43/43 文件 SHA 不变**。脚本、运行日志、事前清单及结果位于 `derived/audit/w1_smallset_fit_diagnostic_postcheck/outer_20261112/`；事后检查不计入合成预算，也不执行模型或 FK。

**后续审阅更正：**以上 1454.429314s 采样早于最终 stdout flush；因此不再宣称 `attempt_018` 的严格完整合成验收已闭合，见 §26。数值链、实际 wait 和已记录的只读对账事实保留；不包含真实 R/D1/D6/F 或全项目测试，也不能证明真实拟合、泛化、碰撞、抓握、无尖峰或生产资格。真实拟合仍须 §8.3 新冻结和明确授权；T6 与既有结论保持。

## 26. 外层监督器审阅修复（2026-11-16）

独立审阅 `b4d5729` 复现 2 P1+1 P2：受控时钟下最终 stdout 阻塞至硬截止后仍报告完成；在历史工件副本中将 F 报告改为 incomplete 并重签证明后仍报告完成；F 阶段名为列表或映射时抛未捕获 TypeError 且无外层裁决。复现仅使用合成临时夹具和历史工件副本。

| 审阅问题 | 修复及本地提交 |
| --- | --- |
| 异常阶段名 | 集合比较前检查字符串类型，错误结构落 `fresh_replay_incomplete` 裁决（`94ccb45`）。 |
| 阶段报告与摘要矛盾 | 对 proof 绑定的 D1/D6/F 报告正文和摘要状态逐项对账；收据绑定实际核验字节的报告/返回证明 SHA；F 报告 incomplete、阶段状态篡改及合法 `fit_insufficient` 控制均覆盖（`a51db20`）。 |
| 最终 stdout 未计费 | 输出改为不可消费的待覆盖事件；flush 返回后检查登记成本和硬截止并写出绑定观察，消费者还须确认监督器成功退出。慢 stdout、慢观察写盘和断管均拒绝完成（`eba46de`）。 |

定点测试分别为 **17、22、26 passed**。最终执行 `TMPDIR=/tmp/opencode uv run --locked pytest -q tests/test_w1_i_outer_return.py tests/test_w1_smallset_fit_diagnostic.py tests/test_w1_smallset_fit_review16.py`，**68 passed（237.66s）**。审阅者复现脚本的隔离副本（结果 `/tmp/opencode/w1_i_reviewfix_20261116_yp43ot/probe_results.json`）对慢最终输出、F 报告 incomplete、列表阶段名均返回 3 且裁决 incomplete；历史 `attempt_018` 副本中的 F 矛盾也被拒。新测试同时覆盖映射阶段名、D1/D6 报告不一致、合法 `fit_insufficient`、stdout 在登记成本内/硬上限内外越线、慢观察写盘和断管。`attempt_016/017` 各 43 文件与原清单一致，`attempt_018` 的保存量、调用矩阵和 115 项反例作只读对账通过。

原 `attempt_018` 的数值链与 uv wait 事实保留，但没有最终 stdout flush 采样，不能通过事后补写证据宣称严格完整合成验收；本轮未改写历史工件，也未自动启动新的完整合成 attempt。未运行全项目套件；真实 R/D1/D6/F 未实施、未授权、未运行。

## 27. 撤回写盘成本复审修复（2026-11-16）

复审 `eba46de` 提出 1 P2：最终 stdout/观察写盘越过登记成本后，旧实现撤回为 incomplete，但直接写裁决后返回，未复查撤回写盘。审阅者受控例中原封口 **20s**、首次越线观察 **21s**，撤回写盘返回时已到 **101s**，磁盘仍只记录 20s/21s；该情况只证明失败成本记录不足，不证明历史 `attempt_018` 确曾越限。

本地提交 `77938bf` 增加 `_write_incomplete_costed`：启动前截止、启动器或最终 stdout 写入失败、最终 stdout/观察/候选裁决/返回证明写入越界、worker 超时及其余已识别 incomplete 共用有界成本登记。每次撤回裁决与成本观察写后采样；记录成本预留只升不降，原封口成本存于 `sealed_registered_measured_s`，不能改写 worker 封口或把预留当实测。失败成本观察绑定被核验的裁决 SHA；监督器实际返回仍需外部观察，所以磁盘 `cost_coverage_proven=false`、状态 `pending_external_return`。持续慢写耗尽轮数时保留最高已采样下界并标 `unproven_rounds_exhausted`，不冒称最终写盘成本已覆盖；无法写盘时也不生成覆盖通过声明。

外层测试 **38 passed（4.77s）**，包括 9 类失败入口的慢撤回、慢失败观察写入、慢裁决哈希读取及有界轮数耗尽。相关合并回归 `TMPDIR=/tmp/opencode uv run --locked pytest -q tests/test_w1_i_outer_return.py tests/test_w1_smallset_fit_diagnostic.py tests/test_w1_smallset_fit_review16.py` 为 **80 passed（238.81s）**。审阅者原复现脚本的隔离副本结果 `/tmp/opencode/w1_i_withdrawal_fix_20261116_lBU6NO/withdrawal_results.json`：控制例 rc=0；两种攻击均 rc=3/incomplete，原封口 **20s** 不变、撤回后已观测 **101s**、提高的预留 **264s**，且覆盖仍待外层实际返回证明。未重写历史工件，未自动重跑完整 D1→D6→F 合成验收，也未运行全项目套件；真实 R/D1/D6/F 未实施、未授权、未运行。

## 28. 当前实现完整合成验收（2026-11-16）

应用户明确要求，在 HEAD `77938bf14f8cb9c55385da97148a5ff1e37570ef` 上执行一次独占的纯 CPU 合成 `derived/audit/w1_smallset_fit_diagnostic/attempt_019/`。实际监督器调用为 `TMPDIR=/tmp/opencode python3 scripts/run_w1_i_synthetic_acceptance.py --wall-cap-s 3300 --reserve-s 300`；独立外层观察器先采样单调钟，再启动监督器，持续等待其实际退出并在收到成功退出后立即采样。工作进程、监督器、观察器和返回采样脚本均 exit 0；worker 和监督器 stderr 为空。

- **D1 与交接**：300 步、11/11 probe，最终四项平均指标为 **0.0011507113m / 0.0028825341rad / 0.0018577272rad / 0.0012062979m**，四门通过，`handoff_ready=true`。实际更新 300、backward/micro 1200、窗消费 1200，conditioner/backbone 各 1211 次，调用矩阵无问题。
- **D6 与 F**：300 步、**66/66** 窗行覆盖，六窗四门逐窗通过，根位置 RMSE 宏平均 **0.0063342440m**（宏平均不是门）。实际更新 300、backward/micro 1200、窗消费 7200，conditioner/backbone 各 1211 次；F 两阶段 initial/last 各一次 fresh probe 逐位一致，`complete_evidence=true`、`replay_passed=true`、`verdict=fit_pass`。**中间 FK 状态绑定及尖峰门仍为 `not_evaluated`，重放范围仅 initial/last。**
- **反例和阶段返回**：115 项反例全部捕获，ID 与冻结集合精确一致。D1 函数返回/观察/登记 **224.548685≤224.548820≤228.546531s**，D6 为 **1298.104051≤1298.104217≤1302.101710s**（同一合成总时钟、不可相加），F 自身为 **1.639494≤1.639626≤5.637653≤600s**；报告正文及证明的实际 SHA 与外层收据逐项一致。
- **总成本链**：以更早的观察器起点为零点，监督器 t0 晚 **0.054710s**；换算后进程内观察 **1392.540328s**、uv 实际 wait **1393.488367s**、监督器 stdout flush **1393.541100s**、监督器实际 wait **1393.552560s**、外层 wait 证明写后 **1393.589414s**、观察器退出后即时采样 **1393.667601s**、采样文件 fsync 后 **1393.687296s ≤ 登记 1396.483622s ≤ 3300s**。此处把较早起点的耗时与按监督器起点登记的成本比较，包含约 0.055s 外层启动余量。外层返回采样在工具收到观察器 exit 0 后立即执行；采样脚本对采样文件写盘自身再次采样并以 exit 0 确认覆盖。
- **只读对账与保护**：摘要 SHA `626392e31fc21f7f7bbed0c6faa580f4b86a3969f3ecf04b093b70952b26855f`，外层监督器返回证明 SHA `35520d7fa813e3e40f62e18e22332224e569b21b40b2146a307fd14dcb129451`，观察器返回采样 SHA `03c31644e21cb5e3b0b81433911c7570e5d289076fe4d4f3f02b338024bfc8be`。独立核对全部登记工件大小/SHA、probe 与窗键、300 条 micro/样本计数、调用矩阵、交接、各阶段/F 和外层证明；66 个实现/环境文件 SHA 与启动前一致，历史 `attempt_016/017/018` 分别 **43/43/48 文件**均不变。脚本、运行日志、事前清单和独立检查结果位于 `derived/audit/w1_smallset_fit_diagnostic_postcheck/outer_77938bf_20261116/`；事后只读对账及其结果报告独立于已完成的计费窗口，不执行模型或 FK。

本次只执行纯合成验收及其 115 项电池，未运行全项目套件；此前 `77938bf` 的相关合并回归为 80 passed（§27）。真实 R/D1/D6/F 仍未实施、未授权、未运行；真实拟合须 §8.3 新冻结及明确授权。合成通过不证明真实拟合、泛化、碰撞、抓握、无尖峰或生产资格；T6 `audit_incomplete`/539 gaps 与 `partial/null/false/not_ready/不接入` 保持。

## 29. R 元数据准入与真实诊断入口实施（2026-11-17）

按方案 §3.2/§6.3/§7/§8 完成下一交付：R 元数据准入、真实数据适配与执行入口、以及准入拒绝/来源漂移/截止/计费反例。新增 [src/ireset/eval/w1_smallset_fit_r_admission.py](../../src/ireset/eval/w1_smallset_fit_r_admission.py)、worker CLI [scripts/run_w1_smallset_fit_real.py](../../scripts/run_w1_smallset_fit_real.py)（`--mode r-admit` / `--mode chain`）与外层监督器 [scripts/run_w1_smallset_fit_real_supervisor.py](../../scripts/run_w1_smallset_fit_real_supervisor.py)，以及 38 项反例测试 [tests/test_w1_smallset_fit_r_admission.py](../../tests/test_w1_smallset_fit_r_admission.py)。实现细节、命令、真实 R 运行结果与未决阻塞见本节；真实 D1/D6/F 仍未实施、未授权、未运行。

### 实现范围

- **R 元数据准入**（§3.2/§6.3/§8）：`run_r_admission` 在 R 预算（120s/15s）内以 `MetadataOnlyReader` 只读元数据——缓存身份→预注册/manifest 绑定链复验（复用已审计 `verify_preregistration` 全输入+资产闭包重哈希与 `verify_cache_contents` 全调度/root_stats 重哈希）、六窗选择（`pilot6 ∩ train24`，逐窗 `(demo,0,60)`，D1=规范字典序首项，只读注册键）、逐窗绑定（教师 npz/pt、derived NPZ、调度 NPZ 字节哈希+NPZ 头成员）、§6.3 运行时元数据对照 T6 注册（复用 T6 `runtime_identity_facts/runtime_identity_problems`）、`RUNTIME_REQUIREMENTS` 绑定字节、实现身份（26 文件字节 SHA+git）、配置 yaml 哈希及与冻结计划逐值交叉核对、`BUDGET_CAPS`+§8.1 总额、由与驱动共享同一实现的纯函数 `expected_call_matrix` 冻结的精确调用矩阵（D1/D6 训练类精确、F 重放类上限）、磁盘余量（≥2GiB+5GiB）与授权模板（六项 checklist 全 null、`authorized=false`）。输出为封口 `r_admission.json`（含 package、zero declarations、metadata reads、seal 与返回覆盖）。方案小改：`FitDriver._expected_call_matrix` 提取为模块级纯函数以共享单一实现。
- **计费区间物化**（§3.2）：`materialize_fit_apparatus`（root_stats 读取、关节域、tokenizer、`metric_decode_v1` 解码器、生产 `LossAssembler`——真实路径显式构造 `G1TorchFK` 与 `largebox_static_frame` 并拒绝注入，编译/缓存命中如实计数）与 `materialize_real_world`（`materialize_for_fit` 硬门：仅已启动未越限的 D1/D6 计费区间；准入包全绑定先复验；逐窗 clip 物化+`TeacherWindowDataset.__getitem__` 同语义 start=0 切窗+逐窗 x0 SHA；D6 物化后对 D1 保存 `probe_inputs.npz` 的共享窗位逐位绑定）。fixture 模式（仅测试）注入 SyntheticFK/固定 static frame 并显式不作生产等价声明。
- **真实链编排**（§7/§8.1/§8.3）：`run_real_chain` 按 §8.3 授权门（六项 checklist 全 True、绑定封口准入包 SHA/cohort/包含阶段/预算/调用矩阵 SHA/当前实现字节）与准入复验全部通过后才在 D1 计费窗口（t0=监督器 uv 前）物化并经共享 `FitDriver` 执行；D6 以自己的 t0 起算，先在 D6 计费区间消费 D1 收据（`resolve_handoff`）再物化训练窗；F 以自己的 t0 物化验证装置并经 `run_f_verification` 逐位重放；每阶段 `finish_stage`+返回覆盖，链摘要以 6120s 链预算封口。
- **外层监督器**：与合成监督器同一纪律——uv 前 t0、实际 `wait()` 采样、逐阶段证明（D1 1800s/D6 3600s/F 600s 上限）、收据/裁决/成本覆盖循环（复用撤回成本登记）、实现字节前后一致；chain 模式另核验链摘要与监督器启动所用准入/授权文件 SHA 一致。

### 测试

`uv run --locked pytest -q tests/test_w1_smallset_fit_r_admission.py` **38 passed（5.06s）**：夹具 SOURCE 树（同结构假数据+真实代码符号链接视图）上的准入全过与零声明、R 只读守卫、窗位选择六类拒绝+预注册 train24 摘除、pilot6 形状/A1 冻结名单绑定拒绝、manifest/缓存身份绑定破坏、四类来源漂移、条件代码漂移、运行时身份漂移、磁盘余量、配置漂移、受控时钟预算耗尽撤回、准入装载拒绝、准入复验漂移、授权门九类拒绝+全过控制、无计费区间/R/F 阶段物化拒绝、fixture D1 物化与逐位重物化一致、D6 共享窗位逐位绑定与篡改拒绝、物化来源漂移拒绝、真实路径注入拒绝、装置窗口守卫、调用矩阵与方案 §8.2 常数及驱动共享实现一致性；外层监督器 R 模式以假 worker 覆盖正例、准入未过/零声明破坏/窗不全/迟 wait/非零退出/实现漂移六类 incomplete。相关合并回归 `pytest -q tests/test_w1_smallset_fit_r_admission.py tests/test_w1_smallset_fit_diagnostic.py tests/test_w1_n0_coordinate_flags.py tests/test_w1_n1_synthetic_numerics.py tests/test_w1_i_outer_return.py` 为 **126 passed（207.50s）**。未运行全项目套件。

### 真实 R 运行与 fail-closed 阻塞（待复阅裁决）

`python3 scripts/run_w1_smallset_fit_real_supervisor.py --mode r-admit` 执行两次：`r_admission/attempt_001` 为输出目录独占创建与监督器预留冲突的 worker 崩溃，按既有撤回成本纪律 fail-closed 保留为证据（修复为检查自身输出文件不存在后未重写该目录）；权威运行为 **`r_admission/attempt_002`**（监督器 exit 3，worker exit 1）。该次准入在 120s/15s 预算内完整执行：封口 `sealed_within_budget`、登记 9.637s、返回覆盖 covered（5.642s）、六窗选择正确（`sub10_largebox_097/sub12_largebox_082/sub16_largebox_000/sub3_largebox_027/sub7_largebox_022/sub8_largebox_035` 各 `(0,60)`，D1=`sub10_largebox_097:0:60`）、运行时元数据与 T6 注册零偏差（python/torch/numpy/scipy/CUDA/设备/cpu_threads=20/MuJoCo/pytorch_kinematics 及绑定源码 SHA）、磁盘 25.18GiB≥7GiB、实现身份 26 文件无缺失、zero declarations 全零。但 **`admission_passed=false`**，唯一拒绝原因：

> `r_cache_reverify_failed:['cache_condition_code_drifted', 'schedule_key_not_reproducible:sub10_largebox_097', 'schedule_file_missing:sub10_largebox_097', …]`

这是 **真实来源阻塞，不是实现缺陷**：登记缓存 `cache_identity.condition_code_sha256=d73f411d…` 绑定 SOURCE 运行（提交 `21e5784`）时的条件代码字节；当前树仅 `src/ireset/geometry/distance_adapter.py` 在其后被修改（`f4724f7`，2026-10-31 为真实 FK 审计给 `largebox_static_frame` 增加可选 `observer`/`check` 场景编译守卫参数），五个 `CONDITION_CODE_FILES` 字节哈希随之改变。已审计消费侧核验 `verify_cache_contents` 因此判定键不可复现、登记缓存对当前代码不可消费；§3.2 明确不得默认以最新 cache 替换登记版本，故 R 按登记规则 fail-closed。diff 核验确认该改动为纯插桩（默认参数 None、编译值不变），但不构成绕过已审计门的理由。处理选项（须用户裁决，不在本轮自动执行）：(a) 将五个条件文件恢复为登记字节——会回退已审计 FK 审计工具的编译守卫，影响其它已审计工件；(b) 以当前代码在独立运行根重建并登记新的键控缓存（真实计费运行+自有预算/台账授权），再重跑 R 绑定该缓存——符合 W1 缓存机制但超出本轮交付范围；(c) 修订 §3.2 允许"内容逐字节核验+按登记文件名消费"并在方案中记录代码漂移理由——需方案修订与复阅。监督器对拒绝准入按 `r_admission_incomplete` 记 incomplete：外层裁决治理链完整性，准入结论以封口 `r_admission.json`（含精确诊断）为权威。

**边界：本轮交付 R 实现与真实 R 准入运行；真实 D1/D6/F 未实施、未授权、未运行。T6 `audit_incomplete`/539 gaps、cohort 34 次更新/1347.195507s、一次性授权已消费、`partial/null/false/not_ready/不接入` 全部保持；合成权威验收仍为 `attempt_019`（§28）；`w1_smallset_fit_diagnostic` 合成 attempt 清单新增 `r_admission/attempt_001/002` 两目录。**
