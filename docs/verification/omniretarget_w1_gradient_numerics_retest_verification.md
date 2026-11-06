# W1 数值修复后真实复验与证据重冻结验证记录

日期：2026-11-06。状态：**T0–T4 已完成；T5 十项授权已确认；T6 已执行一次，结论 `audit_incomplete`（0 failures / 539 gaps）；授权已消费，不自动重试**。执行报告见[omniretarget_w1_gradient_numerics_retest_report.md](omniretarget_w1_gradient_numerics_retest_report.md)。

本记录对应[复验方案](omniretarget_w1_gradient_numerics_retest_plan.md)。真实数值链（新 R0_003→P_003→I_002→F_002）没有运行；原 cohort 的 18 次更新、567.8454090119048 秒成本、329 项 failure / 629 项 gap 与已耗尽的 M6 授权全部保持。`objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false` 不变。

## 0. 复审修复

### P1（本轮）：授权草案的 calibration/evidence/fp32 绑定为空

- 根因：草案生成器从 Gate B v2 顶层读取三个不存在的字段，写出 `calibration_verification_sha256=null`、`tolerance_evidence_sha256=null`、`fp32_contract_sha256=null`；preflight 未检查这些绑定。
- 修复：草案改为从**已核验的 real admission** 取值：
  - `calibration_verification_sha256` ← `real_admission.calibration_verification_sha256`；
  - `tolerance_evidence_sha256` ← `real_admission.tolerance.evidence_sha256`；
  - `fp32_contract_sha256` ← `real_admission.fp32_contract_sha256`。
- `build_retest_authorization_draft` 也支持从 spec 的 `bindings.gate_b_v2` 默认补齐；preflight 现检查：
  - 顶层 7 个绑定（calibration verification、tolerance evidence、fp32、acceptance、real admission、prereg、retest plan）；
  - `retest` 的 10 个绑定（snapshot/spec/budget/protected/retest admission/Gate B v2/retest acceptance/real admission/closed head/invocation）；
  - 当前 `implementation_files`。
- 当前 canonical 说明：attempt_017 的 `40d6b63b…` 是旧实现身份 `a87a769d…` 下的 canonical；实现字节变更后 real admission 重新计算为 `admission_real/attempt_018` 的 **`a9e7b9228189ecb2f1d06ed400e3f4f2aa68b095877cfa5256759f3713e40446`**。新草案逐位使用当前 real admission 绑定的 canonical，而不是硬编码旧值。

### P1（上一轮）：invocation 豁免绑定运行根

- `scan_launch_journals` 的 `allow_root` 与 `allow_invocation` 必须同时匹配 sibling 根的 `resolve()`；未传 `allow_root` 时不豁免。
- 跨根同名反例：另一根的同名未结清 claim 仍产生 debt，当前根预检返回 `retest_root_journal_unknown`。

### P1/P2（更早）：当前 invocation 与目标目录

- 当前 invocation 的 failure/revoke 不豁免；`claim.json.tmp.123.456` 被拒；非 claim-only 目录被拒。
- `R0/attempt_003`、`P/attempt_003`、`I/attempt_002`、`F/attempt_002` 任一存在即启动前拒绝。

### 反例与回归

- 闭包内测试新增：草案从 Gate B v2/spec 默认补齐三个绑定；跨根同名、failure、tmp、非 claim-only、孤立目标目录等反例。
- 聚焦集合 **181 passed（387.43s）**；闭包内 retest 文件 **12 passed（54.68s）**。
- 闭包外补充文件 `tests/test_w1_gradient_numerics_retest_admission.py`：冻结 M5 admission 的摘要漂移/保护项变化/strict 缺 real admission **3 项**，加草案与 real admission 绑定一致性 **1 项**，合计 **4 passed（404.44s）**；该文件不进入 `IMPLEMENTATION_FILES`。
- 未冒充全项目套件通过。

## 1. T1 最终实现身份

- 实现身份：**32 文件，`c4d4a88926b985ee365732da0d940e28985f86c64d4f828e0bb6c5dca4f01b3b`**（files-only canonical SHA）。
- 新增 `tests/test_w1_gradient_numerics_retest.py` 与本方案到 `IMPLEMENTATION_FILES`；未修改 N1/N2 数值算法。

## 2. 原根只读起点

固定原根 `derived/experiments/omniretarget_first/20261102_2704709_w1_real_training_update_audit` 保持 **83 文件 / 7,835,408,541 bytes**，关键值与方案 §2 一致：

| 材料 | SHA-256 |
| --- | --- |
| `cohort.json` | `79f42f5fc2bb205a4f8327b7897644e4759ff60bf189f4dd6054cb980e9f3f6b` |
| `attempts_index.json` | `07195f557ab5cb746c889167192ecdf0681479ea86acd6e0c3ec92f8cefc210d` |
| `costs.json` | `4bbeabcec52368fc6c02aa1134616528346fc5bc7628c7fa96b67df1e1d1e7b3` |
| seq 6 / F_001 commit | `c650c24bf3bb218055b3ed3bc54e33046c69896b642fd3de442e199ccbf2727d` |
| M6 实际 return observation | `32599cf7f84112f031ff648866e839568d3b3605f97b0fb52bc70f099d8aff53` |

`fresh_retest_root_problems(root)` 与 `verify_history(root)` 均返回 `[]`；未创建 generation_003 binding、execution_events 000007–000010、v3 行、真实 claim、正式授权或真实 attempt。

## 3. T2 合成校准

`calibration/attempt_025`：

| 工件 | SHA-256 |
| --- | --- |
| `tolerance_contract.json` | `2a86e06d70e12e44159bef4e4a22e5032b5e2da92fb9dbbe94c547f2c36662df` |
| evidence | `962b625c18080d6c1e08b8769d2a6284161cd86c3d08c0de057e232a6364ccda` |
| fp32 canonical | `94290e009c5238416a4f24404fefd2b18c1d5efa9f1ecdeef004935de0bd90ce` |
| `calibration_verification.json` | `4c42dff2502c1086ffb45a290ea14d596147325c06a01159232697a6d4b67711` |

`passed=true`、`recompute_problems=0`、`four_x_ok=true`；阈值、规则、零底与冻结契约相同，未放宽。

## 4. T3 合成接受

| 分支 | 产物 | SHA-256 | 结果 |
| --- | --- | --- | --- |
| 普通链（16 更新） | `synthetic/attempt_025/synthetic_acceptance.json` | `4da28a3fcbdb16355c02f3881c1bdfbd4e9c5f0eecba79ed9a934a3deb20e8fd` | passed，`audit_incomplete`，0 problems |
| 旧 v1→v2 迁移链（18 更新） | `binding_migration_synthetic/attempt_025/migration_synthetic_acceptance.json` | `8c92d164f26fab26717670fa35864649375985c4eb3f14aac7d173a8c2b38acb` | passed，0 problems |
| Gate B v2 | `binding_migration_gate_b/attempt_025/synthetic_acceptance.json` | `8157ae0be4bbfed2d06490f4f2e3a119908c49823863105c331c276f65345635` | passed，0 problems |
| 新 v2→v3 复验链（34 更新） | `gradient_numerics_retest/retest_synthetic/attempt_007/retest_synthetic_acceptance.json` | `af453f7adf4179e77131782a932ccb04bb2ee006d53b373686dbcc8b21d00b36` | passed，0 problems |

v3 合成链选中 `R0 attempt_003 / P attempt_003 / I attempt_002 / F attempt_002`，`selected_chain_training_updates=16`、`cohort_training_updates=34`。I 阶段 focus sidecar 有 48 个数组；两模型全部线性 self/cross `k.bias` 梯度 presence、shape、dtype 正确且逐位为零；`root_out.weight` 保留。v3 F 在独立子进程中重放 P/I，并逐位比较 `i_focus_gradients.npz`。

**资源处理说明**：为满足 20 GiB 启动门并给新冻结让出空间，5 个已被取代的合成 Gate B 中间 attempt（`synthetic/attempt_020`、`binding_migration_synthetic/attempt_020`、`retest_synthetic/attempt_001–003`，合计 59,144,114,474 bytes）移至 `/dev/shm/w1_retest_superseded/`；另对 `synthetic/attempt_021–024`、`binding_migration_synthetic/attempt_021–024` 与 `retest_synthetic/attempt_004–006` 的 96 个 >8 MiB bulk 文件（73,745,318,096 bytes）在逐文件 SHA 后删除，保留小型报告/验收 JSON。路径/哈希记录在 `gradient_numerics_retest/superseded_moved.json`（当前 SHA `84a1dc176a009dfe3f230a449cdb643448143d0cacc0f591a6305f6578ed0785`）；其中 538 个小文件（190,363,524 bytes）已复制到磁盘上的 `gradient_numerics_retest/superseded_metadata/`。原真实根未移动、未删除。

## 5. T4 只读 M5 材料

`derived/audit/w1_real_training_update_audit/gradient_numerics_retest/materials/attempt_009`：

| 工件 | SHA-256 | bytes |
| --- | --- | ---: |
| `history_snapshot.json` | `710a85e484411004992795d6bfbd95846ef336b67f8125f971ac1ef1671003b6` | 33,677 |
| `budget_revision.json` | `c394091f2203339944f6ca35271524a7ccdd3f0c68e334feec0ac0660e8010a9` | 3,926 |
| `protected_manifest.json` | `c6c07c7c747ef4e43de9c79abe9a10fd4b9347b47cc4b0f7bf5d220e617ba22f` | 751,227 |
| `binding_spec.json` | `357daf3d796276117397d1125e4b89e318cee28ccc9bfcb538dbb608c23e5131` | 19,478 |
| `attempt_001/retest_admission.json` | `49ae9011a63bc1efd923945c999a96bd1473370073866efeb3093d2e397cad21` | 8,827 |
| `authorization_draft.json` | `64fef827f353a29e9d7f61e88e445122d3c6664a904224385be7ac6235e6521e` | 11,900 |
| `preflight_verification.json` | `6c27634e0d71f5e02564db4c910540906bcb721515641b348b49dfc5d9a93e43` | 1,753 |

- 受保护清单 **2,923 entries**，覆盖原根 83 文件、旧授权/校准/准入/迁移/合成接受、父锚、两份 checkpoint、train config、G1/largebox 资产、修复证据、淘汰材料元数据与最终实现文件。
- `budget_revision.sha256 = 2ceff41a7c72558a1b1c5b1b1914f1108de1fb8ee8ce7c4c2d1dbc767f8246c8`；`prior_total_cost_s=567.8454090119048`、`remaining_s=2672.154590988095`；prior call counts 与 caps 逐项匹配方案 §4.1。
- `history_snapshot` 的 `closed_history_head_sha256 = af5c797126de82eb1d79b8c319e0f7a5a5cfcbd24b604fb6f4eb35ef36324896`；seq 7 的 `previous_head_sha256` 仍为 seq 6 commit `c650c24b…`。
- `binding_spec.bindings.protected_manifest.sha256 = c6c07c7c…`；该摘要同时绑定在 retest admission 与授权草案的 `protected_manifest_sha256` 中。
- `retest_admission` 为 `passed=true`、metadata-only，`authorized=false / registered=false / run=not_run / section_9_checked=false`；实际深度消费 `retest_admission_problems(..., strict=True)` 返回 `[]`。
- **授权草案三个字段已非空**：`calibration_verification_sha256 = a9e7b922…`、`tolerance_evidence_sha256 = 962b625c…`、`fp32_contract_sha256 = 94290e00…`，逐位等于 `admission_real/attempt_018` 绑定值；`acceptance_sha256 = af453f7a…`，`admission_package_sha256 = 6d986cb3…`。
- 授权草案为 `authorized=false / template_only=true`，十项 checklist 全为 `null`；`preflight_verification` 为 `passed=true`、`problems=[]`，并检查上述全部顶层/retest 绑定。
- `real_admission` 使用 `admission_real/attempt_018`（SHA `6d986cb392b7bcf06398f6e4762046d6b268938dd27f038edcb8a31c04b0058f`，passed），Gate B v2 使用 attempt_025，retest acceptance 使用 attempt_007。
- 实际反例复核：snapshot/spec/budget 摘要漂移、保护项 SHA 改变并同步重算清单摘要、strict 缺 real admission 均被拒绝。
- `materials/attempt_001–008` 为中间/失败材料并保留；attempt_009 为最终通过版本。

## 6. T6 执行结果

T6 已按用户明确十项授权执行一次。正式授权 `authorization/attempt_001/authorization.json`（SHA `09ee83de…`），消费记录 `consumed.json`（SHA `5e92c179…`），invocation `w1-retest-20261106-003`。

- **结论 `audit_incomplete` / exit 3**：`complete_evidence=true`、`covered_contract_passed=false`。
- **failures 329 → 0**：156 key bias + 173 diffusion u2 组（172 参数行 + 1 output cotangent）全部 resolved。
- **gaps 629 → 539**：350 precision + 189 sensitivity + 0 bound；95 resolved、534 persisted、5 new。
- 完整输出 cotangent 16/16 verified，max ratio `0.0171412459`；参数 micro max ratio `0.5166358718`。
- Key bias：224 行 verified、32 条 AdamW 全 passed、focus 128 个 key bias 数组全零；micro2 `root_out.weight` 梯度 verified（max ratio `0.3078734901`），但其 AdamW 行仍有 sensitivity/precision gap。
- F 独立进程 pid `373450` / ppid `365796`、exit 0；P/I replay complete/numeric，npz/row/report/call problems 全空。
- 实际返回 `99e63f90…`，`coverage_ok=true`、total wall `774.6779822630015` 秒、exit 3；新提交 seq 7–10 为 `1d6f25e0…` / `773cb48c…` / `459ef764…` / `7ee1965b…`。
- 本轮成本 R0 `316.523288673` + P `37.470196019` + I `119.413655903` + F `305.942957069` = **779.350098** 秒；累计 **1347.195507 / 3240** 秒；selected 16、cohort 34 次更新。
- 原根执行后 158 文件 / 15,204,894,348 bytes（14.16 GiB，≤16 GiB cohort 上限）；旧六行 canonical 未变；剩余空闲盘约 26.46 GiB。
- 详细逐键失败/gap 对照见[执行报告](omniretarget_w1_gradient_numerics_retest_report.md)与 `gradient_numerics_retest/real_analysis/attempt_001/`。

## 7. 下一步

T6 一次性授权已消费；不自动重试、不放宽阈值、不启动第二 invocation。539 项 precision/sensitivity gap 仍使 covered contract 不通过；`objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、最终交互 `not_ready`、生产接入 false 保持。后续若需继续，只能另立新预注册、预算与授权，不能复用本次授权或把 gap 合并为通过。
