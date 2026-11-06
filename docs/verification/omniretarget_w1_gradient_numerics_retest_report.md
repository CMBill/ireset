# W1 数值修复后真实复验执行报告（T6）

日期：2026-11-06。结论：**`audit_incomplete` / exit 3；0 failures、539 项 mandatory gaps；`complete_evidence=true`、`covered_contract_passed=false`。** 原 329 failures 全部 resolved；key bias 结构零、完整输出 cotangent 与 micro2 `root_out.weight` 梯度门通过。未自动重试、未生产接入。

## 1. 授权与身份

- 用户明确授权：**“上述十项全部同意，并同意按绑定值执行一次 T6”**。
- 正式 v3 授权：`derived/audit/w1_real_training_update_audit/gradient_numerics_retest/authorization/attempt_001/authorization.json`，SHA `09ee83debed0c1bf5aae1606455947aa82c44fbaa06109ac82c4a898b0d5b7f1`。
- 消费记录：同目录 `consumed.json`，SHA `5e92c17922c10cae96e771b891319b5233cb5adff628748715f8fa74a8ba3396`；`one_shot_used=true`、`automatic_retry=false`。
- 实现身份：**32 文件 `c4d4a88926b985ee365732da0d940e28985f86c64d4f828e0bb6c5dca4f01b3b`**。
- invocation：`w1-retest-20261106-003`；generation_003：R0_003 / P_003 / I_002 / F_002，seq 7–10。

## 2. 实际执行与返回证明

- 外层实际返回：`launch_journal/w1-retest-20261106-003/return_observation.json`，SHA `99e63f90d4addfffe1ccdd9140f8d20cfb0d097d63f8dd0b2be41173ae112cef`；`coverage_ok=true`、`source=outer_wait`、`total_wall_s=774.6779822630015`、`exit_code=3`。
- 外层 pid/ppid：`365796 / 363397`；F worker pid/ppid：`373450 / 365796`、exit 0、wall `260.441657527` 秒；F 报告的 `verification_problems=[]`。
- F replay：P/I 均 `complete_evidence=true`、`numeric_passed=true`；`npz_problems=[]`、`row_problems=[]`、`report_problems=[]`、`call_matrix_problems=[]`；I 主行 3,846（expected 3,846），P 行 18。
- 新提交：seq 7 `1d6f25e0…`、seq 8 `773cb48c…`、seq 9 `459ef764…`、seq 10 `7ee1965b…`；`stage_state` 四个新 attempt 均可消费，`verify_history=[]`、`scan_launch_journals.known=true`、`debts=[]`。
- 旧六行 canonical 内容全部未变；原 83 文件起点未删除。执行后原根为 **158 文件 / 15,204,894,348 bytes**（14.16 GiB，≤16 GiB cohort 上限）；磁盘空闲约 **26.46 GiB**。

## 3. 数值结果

最终报告：`finalize/attempt_002/training_update_final.json`，SHA `3d5e9c111191971c4d53985b0f0075180e1390682738fc4b7ea4b301523aee11`。

| 项目 | 旧 M6 | 新 T6 |
| --- | ---: | ---: |
| failures | 329 | **0** |
| gaps | 629 | **539** |
| precision gap | 381 | 350 |
| sensitivity gap | 223 | 189 |
| bound gap | 25 | **0** |
| selected chain updates | 16 | 16 |
| cohort updates | 18 | **34** |

- 旧 329 failures 按类别：**156 key bias + 173 diffusion u2 组**（其中 172 参数行 + 1 `output_cotangent:diffusion:u2:m0`）。逐键归一化后 **329/329 resolved、0 unresolved**。
- 旧 629 gaps：**95 resolved、534 persisted、5 new**。resolved = 32 precision + 38 sensitivity + 25 bound；new = 1 precision + 4 sensitivity。新 539 = 534 persisted + 5 new；bound gap 清零。
- 新 failures 为空；新 gaps 全部为 `parameter_delta_precision_gap`（350）或 `cross_path_sensitivity_gap`（189），没有 `cross_path_bound_unverifiable`。
- 完整输出 cotangent：16/16 `synthetic_verified`，max ratio **0.0171412459**，group max ratio **0.0060535099**。
- 参数 micro 梯度：全部 `synthetic_verified`，max ratio **0.5166358718**（<1）。
- Key bias：224 条 key bias 行全部 `synthetic_verified`；32 条 key bias AdamW 行全部 `classification=passed`、无 gap；focus 中 128 个 key bias 数组逐位为零，16 个 `root_out.weight` 数组非零。
- micro2 `root_out.weight`：diffusion u2/m2 P/I 梯度 `synthetic_verified`、max ratio **0.3078734901**；对应 AdamW 行不是 direct failure，但仍有 `cross_path_sensitivity_gap`（`max_state_ratio=1.0861218843`）与 `parameter_delta_precision_gap`，因此仍计入 539 gaps。
- Focus sidecar：`independent/attempt_002/i_focus_gradients.npz`，SHA `62abc77a19d6af919e29f49211f086f66173f4cac3188d6aa530196cb3a37e66`，144 keys / 261,120 bytes；P focus manifest 72 keys / 261,120 bytes。F replay 对 focus sidecar 做了逐位比较。

## 4. 成本与调用

| stage | attempt | measured_wall_s | 状态 |
| --- | --- | ---: | --- |
| R0 | attempt_003 | 316.523288673 | done/complete |
| P | attempt_003 | 37.470196019 | done/complete |
| I | attempt_002 | 119.413655903 | done/complete |
| F | attempt_002 | 305.942957069 | done/complete |
| 本轮合计 | | **779.350098** | |
| 累计 | | **1347.195507 / 3240** | |

- selected chain calls = 16 updates；cohort calls = 34 updates。
- 24 类 cohort 调用与 §4.1 cap 逐项相符：checkpoint_loads 18、model_constructions 18、conditioner_constructions 18、native_optimizer_constructions 10、conditioner/backbone forwards 136、native_add_noise 64、cpu_draw_units 104、production_loss_assembler 72、production_float32_fk_batch 144、production_parameter_backward 72、independent_output_vjp 64、independent_total_vjp 64、native_global_clip 18、independent_clip_formula 16、native_adamw_steps 18、explicit_reference_updates 16、diagnostic_parameter_update_events 34、anchor_state_restores 8、optimizer_formula_checks 16、reference_mj_forward 31,680、reference_mj_jac_site 230,400、scene_setup_mj_forward 5、micro_execution_units 136。
- 七项零声明全部为 0：`parameter_fd`、`state_fd`、`ddim_calls`、`validation_calls`、`selection_calls`、`production_checkpoint_writes`、`contact_branch_executions`。

## 5. 逐键对照材料

- `derived/audit/w1_real_training_update_audit/gradient_numerics_retest/real_analysis/attempt_001/analysis.json`，SHA `d03f2d78c6933b8ada37766674367de903024b88f7019464e6ff8472f4d6cd3a`。
- `failures.csv`（329 行逐键 resolved/status），SHA `029ebff74992eaf6da0ac44d8a16aa2c781bfa6fd8cfb15d8e69e5763bbea7ac`。
- `gaps.csv`（629 旧 gap + 5 新 gap，逐键 state），SHA `100c2f6d5360bac7d05beb371032e51a1e04f48887e8353269b82aaa5c112ef4`。
- 旧/新失败与 gap 的分类、逐键明细、focus、micro2、成本、调用和实际返回均保存在上述 `analysis.json`。

## 6. 裁决与边界

- **329 项 direct failures 已全部 resolved**；原 key bias、diffusion u2、output cotangent 失败不再出现。
- **仍不能宣称 covered contract 通过**：539 项 precision/sensitivity gap 保留，`covered_contract_passed=false`，结论 `audit_incomplete`。按冻结协议，这些是必测但不可辨识/相消敏感的项，不能合并成通过。
- micro2 `root_out.weight` 的梯度级独立问题已消失，但它的 AdamW 跨路径状态/更新比较仍是 sensitivity gap + precision gap；不得用 micro0 或梯度门通过推断更新门通过。
- `objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false` 保持；旧质量负结果不变。
- 本次授权为一次性，已消费；不自动重试、不启动第二 invocation、不放宽阈值、不接入生产。
