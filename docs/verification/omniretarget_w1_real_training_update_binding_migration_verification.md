# W1 训练更新绑定迁移实现与验证记录

日期：2026-11-04。状态：**M0–M5 完成；第三轮复审 2 P1 已修复，最终字节 Gate B 为 `calibration/attempt_018` + `synthetic/attempt_018` + `binding_migration_synthetic/attempt_005` + `binding_migration_gate_b/attempt_005` + `admission_real/attempt_019`；M5 为 snapshot 006/spec 006/migration admission 004/授权草案 005；M6 未授权、未执行**。本记录只描述实际写入的代码、合成测试和只读迁移材料，不构成真实恢复授权。§6 的 015/002 系列与 §7 的 017/004 系列均为历史权威，已被 §8 最终系列取代。

## 1. 范围与权威边界

本轮按[绑定迁移与有限恢复方案 v0.13 修订2](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_real_training_update_binding_migration_plan.md)开始实现。方案文件 SHA-256 仍为 `1b27a41462c4d40fb3e0f54b240c3ac6669e4871062191d470dc734721142b17`。

未执行：

- 未读取真实 checkpoint payload、真实父 NPZ 数值、数据 cache 或真实资产编译；
- 未在原真实根创建 launch journal、binding、execution event、R0/P/I/F attempt 或 reservation；
- 未创建真实授权、真实 R0_002、真实 P_002、真实 I_001、真实 F_001；
- 未生成新的 Gate B 校准/普通合成接受/迁移合成接受/metadata real admission；
- 未取得 M6 授权，未自动重试。

因此真实历史仍为原 R0 attempt_001 done、P attempt_001 incomplete、I/F 未创建，`training_updates=2`、`cost=67.10273322396097s`。当前实现证据权威仍是 v0.12 / Gate B attempt_013，直到 M3/M4/M5 用最终字节重新冻结。

## 2. M0：冻结输入与只读迁移材料

CLI 新增 `--migration-snapshot`，在原根之外生成：

| 材料 | 位置 | SHA-256 / 结果 |
| --- | --- | --- |
| history snapshot | `derived/audit/w1_real_training_update_audit/binding_migration/attempt_001/history_snapshot.json` | `28a542e66dfe121822c5c8d7635ec76d7f123ee1752b1905ba8dc7b5a4088500` |
| source index copy | 同目录 `source/index.json` | `78a2f97dc42545c21f2200a9353b4e1ce9f4dea3fde9cf88f27c29548932b950` |
| source ledger copy | 同目录 `source/ledger.json` | `92773314541bec3b3d751c1627fb8cf78b331ea21303b8c753b8d4a447aa0f34` |
| source marker copy | 同目录 `source/cohort.json` | 原根 marker 字节，snapshot 内登记 |
| protected manifest (initial) | 同目录 `protected_manifest.json` | `a033e856a191c053a7b3ce5713b673d74aae561dd37f3672eccd55a7d581c94f`，52 entries |
| protected manifest v2 | 同目录 `protected_manifest_v2.json` | `0d0ffe6f63e0bfe73e663a9ad9fb649177da2bed2c5c5c890ebbc2c67c556e80`，93 entries；覆盖既有 62 文件清单 `03eba72eb291cfb6b3c2b8108752ad8fb3cb68b99633a6218734c3a705a84d53` 的全部 62 项（drift 0） |

snapshot 核对结果：

- `legacy_head_sha256 = b499a6ce7e5ef8fac06821cc063f27bd109374d9673791551fe92ad56302a424`，与方案 §4.1 固定值一致；
- 两条旧执行顺序为 R0/attempt_001 → P/attempt_001；
- 阶段成本 R0 `58.59990523397573`、P `8.502827989985235`，合计 `67.10273322396097`；
- 24 类 `call_balance_after` 为原 P 终态累计，`native_adamw_steps=2`、`diagnostic_parameter_update_events=2`；
- 未封口文件 1 个：`production/attempt_001/p_deterministic.npz`；
- 原根 10 文件清单、SHA 和 bytes 均进入 snapshot；启动日志清单为空（原根没有 `launch_journal`）；
- protected manifest 另存了实现基线 `HEAD=f808aa975ba2ccbabdddba348f9cbeb2b85b9dfb` 的 git blob 哈希和预期改动文件清单。

本轮没有向原真实根写入任何文件。

## 3. M1：实现的核心接口

### 3.1 规范编码、legacy head 与快照

新增：

- `canonical_json_bytes` / `canonical_sha256`：UTF-8、sort keys、无空格、`allow_nan=False`；
- `strict_read_json`：拒绝重复 JSON key；
- `build_legacy_head_payload`：只从两条旧行、两份 reservation 和原文件摘要独立重算；
- `build_history_snapshot` / `write_history_snapshot`；
- `build_protected_manifest` / `write_migration_baseline`。

旧 index/ledger 是追加视图，因此读取器同时核对 source copy 整文件 SHA、旧行 canonical 哈希和旧 reservation 字节，不要求当前视图仍有旧整文件 SHA。

### 3.2 执行序号与代次选链

新增 v2 常量与记录字段：

- `AUDIT_SCHEMA_V2 = w1_real_training_update_audit_v2`；
- `AUTHORIZATION_SCHEMA_V2 = w1_real_training_update_authorization_v2`；
- `REGISTRATION_SCHEMA_V2 = w1_real_training_update_registration_v2`；
- `ACCOUNTING_SCHEMA_V2 = pre_registered_binding_v2`；
- `generation / binding_spec_sha256 / execution_seq / predecessor_kind / previous_head_sha256 / invocation_id`。

`_execution_records` 按真实执行序号重放：seq 1/2 来自冻结 legacy head，seq 3 必须 `predecessor_kind=legacy_head` 且等于固定头，seq 4–6 必须接前一 commit 文件字节 SHA。seq 重复、跳号、回退、旧行 canonical 改变、reservation 改变或 start event 不一致均返回 unknown。

`stage_state` 增加显式 `generation`：

- 存在迁移且未指定代次 → `generation_ambiguous`；
- generation_001 只能只读查询旧行，不能作为第二代 P/I/F 上游；
- generation_002 要求 R0_002/P_002/I_001/F_001 各唯一可消费；
- 公共消费还要求 commit、`stage_closed` 候选、写后结果和外层 `return_observation`；
- 同一执行上下文只能通过显式 `internal_context` 处理当前 in-flight 尝试。

### 3.3 launch journal、commit 与返回观察

新增：

- `claim_launch`：不可覆盖 claim，绑定授权/spec/legacy head/marker/目标 seq/t0/pid；
- `_verify_prepared_attempt`：内部准备态核验，不要求当前 commit，不返回公共可消费行；
- `commit_attempt`：准备态核验和计费稳定后写一次 finish/commit；commit 后禁止 `finish_attempt` 再封口/撤回；
- `write_stage_closed_candidate` / `write_terminal_candidate`：先写候选，再在 write/fsync/回读返回后按实际 `perf_counter` 检查已计费 floor 和阶段 remaining；
- `write_launch_stage_closed_candidate` / `write_launch_terminal_candidate`：在固定原根 `launch_journal/<invocation>/` 写同类候选并再次执行写后门；
- `record_return_observation`：由外层实际 wait/exec 返回后写入，绑定实际 exit/signal、进程、commit/candidate、写后结果和覆盖判定；
- `scan_launch_journals`：所有历史/预算/准入读取器扫描固定 history root 的 launch journal；未结清 claim/failure/revoke、缺 registered/terminal 候选、terminal 候选/commit/event 引用漂移、任一份 `terminal_write_after` 证明缺失或不过门、或坏返回观察均使历史 unknown。

写后门失败的候选会追加 `revoke_*.json`，记录
`uncovered_wall_s_lower_bound = max(0, observed_elapsed - committed_measured_wall_s)`。
commit 已写时不再改写行、finish 或 commit。

### 3.4 新版授权、迁移准入与 CLI

新增：

- `migration_authorization_template` / `migration_authorization_problems`：
  - 10 个 checklist 名称不变；
  - `prior_updates=2`、`incremental_update_limit=16`、`cohort_update_limit_for_this_authorization=18`，严格要求 JSON 整数且关系成立；
  - 绑定 snapshot/spec/legacy head/launch journal/plan/implementation/runtime/Gate B/real admission/剩余预算/唯一 invocation；
  - 拒绝未知 migration scope、bool 数字、旧预算重置或模板直接套用；
- `build_migration_authorization_draft`：只生成 `authorized=false/template_only=true` 草案；
- `build_migration_admission` / `write_migration_admission` / `migration_admission_problems`：只读复核快照、spec、source copies、legacy head、预算/attempt、固定启动日志、实现身份和 Gate B/admission 绑定；
- CLI 新增 `--migration-snapshot`、`--migration-admission`、`--migration-authorization-draft`；普通 `--resume` 遇到 v2 授权会拒绝，不会自动升级身份。

### 3.5 M1 复审后的修复（6 P1 + 3 P2）

复审在 M1 边界发现并已修复以下问题，均加入迁移合成反例：

1. **快照文件清单消费**：`_snapshot_problems` 现在逐项复核 `snapshot.files` 的 SHA/bytes；仅 `attempts_index.json`、`costs.json` 允许合法追加，其旧行仍按 canonical 行哈希和 reservation 摘要核验。原 marker、旧 NPZ/JSON/reservation 漂移均拒绝。
2. **提交前只做一次全量准备态核验**：`_migration_stage_seal` 的一次全量核验在成本稳定前完成并计费；`commit_attempt(..., prepared=...)` 只复核冻结的小型 `index_view/ledger_view/status/complete/reservation/identity` 视图，不再重复大工件哈希。成本重封口允许 `measured_wall_s/sampled_wall_s/seal_tail_reserve_s` 及其派生字段 `wall_s` 变化（2026-11-04 复审补充，见 §3.6）。
3. **写后证明自身计费**：`stage_closed`/`terminal` 及两支 launch 变体在写完 `*_write_after.json` 返回后再次采样；超过已计费上界或阶段预算时追加 `*_write_after_proof_gate` revoke，不再出现“最后一次检查之后又慢写 5 秒仍成功”。
4. **F 不完整证据硬否决**：迁移 F 封口前检查 `complete_evidence is True` 且 `verification_problems` 为空；否则改写报告、以 `incomplete` 撤回 F，保留已结算调用账并令 `selected_attempts=null`，不写 F commit。
5. **日志结构错误累积**：`scan_launch_journals` 不再用返回观察覆盖 `registered`/`terminal`/`stage_closed` 的结构错误；有效观察与缺失记录会同时保留在 `return_problems` 和 `known=false` 中。
6. **返回覆盖由实际时间/成本重算**：返回观察新增 `stage_timings`；公开读取器从各 commit 的 `stage_t0_perf_counter/committed_measured_wall_s` 与 `finished_perf_counter` 重算覆盖，`coverage_ok` 不再是唯一判据。合成把结束时间推到 1,000,000 秒时得到 `return_coverage:F`。
7. **登记前 claim 进入公共历史**：`verify_history` 在存在固定 `launch_journal` 时即扫描，即使尚无 binding/v2 行；仅当前 `allow_invocation` 获得运行内豁免。只有 claim、缺 registered/terminal/返回观察的根不再返回空 problems。
8. **真实准入适配 acceptance v2**：新增 `acceptance_leaf_record`，从 `ordinary.record` 读取 roster/tolerance/implementation/runtime，并同时检查 ordinary 与 migration 两个子记录；real admission 不再因 v2 包装得到空 roster 或伪实现漂移。
9. **授权草案不依赖原根 binding**：新增 `load_external_binding_material`，从外部 snapshot/spec 只读构造上下文；`--migration-authorization-draft` 不再要求原根已发布 intent/spec，授权后再由迁移流程发布原根绑定。

### 3.6 2026-11-04 复审后的 3 项 P1 修复

1. **成本补记不再被派生字段误拒**：`_frozen_row_view` 排除 `wall_s`（`upsert_cost` 每次重封口同步写入的 `round(measured_wall_s,1)`）；新增 `_billed_cost_problems` 在准备态核验和提交前刷新中要求 ledger 的 `wall_s` 与 `measured_wall_s` 一致（index 通常不携带该字段，若存在则同样核对）；刷新视图还要求成本不得低于准备态（`commit_prepared_cost_decreased`）。隔离迁移根上把成本提高 5 秒后刷新通过；篡改 `wall_s` 或降低成本分别被拒。
2. **阶段交接纳入计时**：迁移执行器改用单一 `stage_cursor`，下一阶段 `stage_t0` 等于上一阶段记录的 `end_perf_counter`，receipt 与下一阶段预算历史扫描全部落入下游阶段区间并随其成本计费（binding 发布已改计入 R0，见 §3.7）；消费端 `_return_timing_coverage_problems` 新增 `return_stage_gap`（相邻阶段不得留空档，重叠仍拒绝）、`return_total_wall`（`total_wall_s` 必须等于外层返回减起点）和 `return_started`（起点必须等于 claim t0）。
3. **terminal 闭包解析而非存在性**：扫描器解析 launch `terminal_candidate.json` 及其 commit/event 引用（commit 必须为 seq 6/F、spec 与 claim 一致），并解析 event 目录 terminal 候选与两份 `terminal_write_after` 证明（launch 与 event 各一份），要求 `passed=true`、`candidate_sha256` 对应、`observed<=committed` 且 `<remaining`。launch 候选改为 `{}`、删除 event 候选或任意/两份写后证明均使 journal unknown，并经公共 `verify_history`/`stage_state` 报出。

### 3.7 2026-11-04 第二轮复审后的 2 项 P1 + 1 项 P2 修复

1. **绑定发布计入 R0 预算**：`activate_binding` 不再引用尚不存在的 commit（避免自摘要环），改为在 R0 封口前发布；`commit_attempt` 在 seq=3 的 finish/commit 中绑定 activation 路径与 SHA，`_commit_problems` 复核该引用、activation 内容与 R0 reservation SHA，`_activation_problems` 要求 activation 只有被 R0 commit 引用后才生效。受控时钟注入 200 秒 activation 延迟（R0 剩余 181.4001 秒、P 预算充足）后，R0 在封口阶段即 `budget_exceeded_after_registration`，P 不创建 attempt，返回 `r0_seal:budget_exceeded_after_registration`，R0 登记成本 201 秒；旧实现会把 activation 记入 P 并完整执行。
2. **terminal 写后门绑定权威上界**：新增 `_terminal_commit_bounds`，从 F commit 的 `committed_measured_wall_s` 与 `finish.ledger_row_snapshot.budget.remaining_s`（以 commit 的 `ledger_row_sha256` 绑定）取得权威成本与剩余预算；两份 proof 的 `committed_measured_wall_s/remaining_s` 必须与该权威值逐位相等，放行检查改用权威上界。把两份证明同时改为 observed=19.8921、committed=24.8921（自洽但超过 F commit 冻结成本的篡改）后 journal unknown（`terminal[_event]_write_after_committed`），remaining 改为 1234 秒得到 `..._write_after_remaining`。
3. **非对象 terminal JSON 不再崩溃**：`terminal_candidate`/`terminal_write_after`/commit/event/registered/stage_closed 及其嵌套 `commit`/`start`/`event_candidate` 引用在调用 `.get` 前检查 `Mapping`；数组/null 使 `scan_launch_journals`/`verify_history`/`stage_state` 返回结构错误（`terminal_candidate_not_object`、`terminal_write_after_not_object`、`terminal_event_candidate_not_object`、`commit_not_object` 等）而不是 AttributeError；claim `target`、binding intent/spec 与 activation 同样加固。

## 4. M2：本轮已执行的反例

新增 `tests/test_w1_real_training_update_migration.py`，共 **24 passed（51.90s）**：

1. 真实原根独立重算的 legacy head 等于 `b499a6ce…`，两条记录为 R0→P，旧累计更新为 2；
2. 合成迁移根按 seq 3→6 完整执行第二代 R0_002/P_002/I_001/F_001，`complete_evidence=true`，selected chain 更新 16，cohort 更新 18；
3. 旧 R0_001/P_001 无 seq 字段且 canonical 内容保持不变；新行全部为 v2 schema、`generation_002`、正确 predecessor；
4. 外层返回观察前，公共 `stage_state` 因缺 `return_observation` 拒绝；写入外层返回事实后四阶段均可消费；
5. 迁移根未指定代次返回 `generation_ambiguous`；旧 generation_001 行仍存在且只读保留；
6. 篡改执行序号后不能重推为已知链；
7. v2 授权拒绝更新额度重置（prior=0、incremental=18、remaining=3240）；
8. `stage_closed` 和 terminal 候选各有一个受控时钟慢写反例：尾部只预留 1 秒而实际 5 秒时，写后门否决并写入至少 4 秒的未覆盖下界 revoke；另有 return observation 缺来源/覆盖/过程返回事实的拒绝测试；
9. 快照消费复验 marker/旧 reservation 摘要，同时允许 index/ledger 合法追加；
10. commit 复用冻结的小型准备态视图，不再重复全量大工件哈希；
11. `*_write_after.json` 自身慢写时由最终放行检查否决并 revoke；
12. F `complete_evidence=false` 或 `verification_problems` 非空时不写 commit，F 行撤回为 incomplete 且 selected 为空；
13. 有效返回观察不能覆盖缺失 registered/terminal 的结构错误；
14. 外层结束时间扩到 1,000,000 秒时公开读取器报 `return_coverage:F`；
15. 只有 claim、尚无 binding 的根被公共历史判为负债，当前 invocation 可有显式运行内豁免；
16. acceptance v2 的 ordinary leaf 与外部 snapshot/spec 授权草案入口可用，且不要求原根已有 binding；
17. 迁移结果中相邻阶段的 `t0` 与上一阶段 `end` 逐位相等（交接无空档）；
18. 终端闭包：launch 候选改 `{}`、删除 event 候选或两份写后证明的不同组合均使 journal unknown，公共 `verify_history` 报 `history_journal`；
19. 返回覆盖：20 秒交接空档报 `return_stage_gap:R0:P`，`total_wall_s` 漂移报 `return_total_wall`，起点偏离 claim t0 报 `return_started`；
20. 成本重封口：提高 5 秒后冻结小视图通过；`wall_s` 不投影或成本下降被拒（旧实现报 `commit_prepared_ledger_view`）；
21. R0 commit 绑定 activation 路径/SHA，activation 的 R0 reservation SHA 与 commit 一致；
22. 受控时钟注入 200 秒 activation 延迟（R0 剩余 181.4001 秒、P 预算充足）后 R0 封口即超预算，P 不启动，R0 成本覆盖 activation；
23. terminal 两份证明同时改为自洽的 observed=19.8921/committed=24.8921 后 journal unknown（绑定 F commit 冻结成本），remaining 改 1234 秒同样被拒；
24. terminal candidate/proof、event candidate、commit 改为数组或 null 时 `scan_launch_journals`/`verify_history`/`stage_state` 报结构错误而不抛 AttributeError。

另运行既有相关回归：

- `tests/test_w1_real_training_update_audit.py` + `tests/test_w1_real_training_update_provider.py` + 新迁移测试：**141 passed（293.20s，最终字节）**。

全项目套件未重跑；这些数字不是 380 旧基线的替代。

## 5. M3–M5 完成状态

- M3 已完成：`IMPLEMENTATION_FILES` 为 **28 项**（原 26 项 + 本方案 + 迁移测试），`EXECUTION_PLAN` 指向本方案，`PREREG_DRAFT` 保持 v0.9；最终字节回归见 §6.1。
- M4 已完成：新校准、普通合成链和迁移合成链均通过；Gate B v2 同时绑定两者；P/I/F 数组与 rows 相对 attempt_013 **12/12 逐字节相同**（两条链各自）。
- M5 已完成：新 real metadata admission、history snapshot、binding spec、migration admission 和 v2 授权草案均生成并通过只读复核；原真实根仍 10 文件 / 466,350,106 bytes，无 launch journal、无新注册/授权/数值更新。
- M6：未取得绑定最终字节的新明确授权，未执行真实 R0_002→P_002→I_001→F_001；旧一次授权与本轮材料均不得解释为 M6 授权。

### 5.1 被取代的本轮中间尝试

为避免混淆：`calibration/attempt_014`、`synthetic/attempt_014`、`binding_migration_synthetic/attempt_001`、`binding_migration_gate_b/attempt_001`、`admission_real/attempt_014` 是在发现 `--migration-spec` 返回值解包错误之前按当时字节生成的中间尝试，全部保留但**不是最终 Gate B 权威**。§6 的 attempt_015/attempt_002 系列也已被 §7 的复核修复最终系列（attempt_017/attempt_004）取代；M5 的 attempt_015→016→018、spec 002→003→005、admission 001→002→003、draft 001→002→004 均为历史尝试。旧 attempt_013 及更早历史均未删除、未改写。

`objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false` 及旧 U0/UR/原 F/r6/旋转融合结论全部保持。

## 6. M3–M5 最终执行事实（2026-11-04）

### 6.1 M3：实现清单与最终字节回归

- `IMPLEMENTATION_FILES`：原 26 项保留，新增 `docs/plans/omniretarget_w1_real_training_update_binding_migration_plan.md` 与 `tests/test_w1_real_training_update_migration.py`，共 28 项。
- `EXECUTION_PLAN` 改为本方案；`PREREG_DRAFT` 仍为预注册 v0.9（SHA `e2ba8ef9…`）。
- 最终字节定向测试：`test_w1_real_training_update_audit.py` + provider + migration **141 passed（359.87s）**。
- 最终字节 M3 回归集（provider + audit + 六个 v0.12 相关文件）**380 passed（469.08s）**；全项目套件未重跑。
- 中间尝试（attempt_014/001 系列）测试为 **141 passed（357.48s）**、**380 passed（469.83s）**，保留但不再作为权威。

### 6.2 M4：校准与双合成链

校准 `derived/audit/w1_real_training_update_audit/calibration/attempt_015/`：

| 材料 | SHA-256 |
| --- | --- |
| tolerance contract | `634f2d0ff56c6c1c979639d5ec83d4a156c421e6c49909f3c3683e3b474d4cbc` |
| calibration evidence | `b848cbb4e54553dd19daf69a471076ddfc3bd2c6030c55608413d746d0c096b2` |
| fp32 contract canonical | `574d9a0a2115e3619aa260f3eef4ef83c8ff85e377e2f23eebc078434a0d441e` |
| `calibration_verification.json` | `4cef73aa755a0597d907b76bb451d320ec8023f1b3ec993b96181a50d8d7be73`（canonical `ba4044ad…`） |
| `adamw_fp32_contract.json` | `2c4ddba84d43a2264eca94b603b400af8be764de97352e3cf933e569822c44b2` |

`recompute_problems=0`、`four_x_ok=true`；`tolerances`、`rules`、`tiny_zero_floors` 与 attempt_013 逐项相同，差异仅 `created_wall` 与实现身份相关字段。

普通合成链 `synthetic/attempt_015/`：

| 项目 | 结果 |
| --- | --- |
| 结论 | `audit_incomplete` / exit 3；`complete_evidence=true`、`failures=0`、**582 gap**（392 precision + 190 sensitivity + 0 bound） |
| 选中链 | R0/P/I/F 均 `attempt_001`；selected chain updates 16；training_updates 16 |
| 阶段文件数 | R0 3 / P 5 / I 5 / F 12 |
| 成本（秒） | R0 1.181511805 / P 18.241404025 / I 72.379080199 / F 166.204949871 = **258.006945900** |
| `audit_result.json` | `31dee169eb14bab9c8c201f47e9c0ed6a6f68e24ec17b848c250777a10e67925` |
| final | `96bc097dce224f84ac7d9ce68bbed43f4a63146bdfbf169631d20cf863487d84` |
| registration | `60edc8b109b6b00648ac9a0f6b1ebd3da816a64125a73050a0ac92ccbf357403` |
| acceptance | `9eb4ce4ccf69007f4fc2a470a80ae44ff1f7eed3e0bd063256829eaeafaa9857`（passed/problems 空，`run_sha256=ca2f5bd2…`） |
| 外层返回 | `synthetic_return.json` `24c814c50c1aee8a1114f07fd76f576eb22893902d2239d12565fdfc189ad4f5` |
| 工件 | 7,404,802,655 bytes |

迁移合成链 `binding_migration_synthetic/attempt_002/`：

| 项目 | 结果 |
| --- | --- |
| 结论 | `audit_incomplete` / exit 3；`complete_evidence=true`、`failures=0`、**582 gap**（与普通链相同分类） |
| 选中链 | R0/P `attempt_002`、I/F `attempt_001`；selected chain updates 16；cohort updates **18**（旧 2 + 新 16） |
| 阶段文件数 | R0 4（含合成 noise）/ P 5 / I 5 / F 12 |
| 新阶段成本（秒） | R0 1.411957657 / P 18.419276195 / I 71.977563807 / F 105.407851266 = **197.216648925**；含旧 67.102733224 的 cohort 累计 264.319382149 |
| migration acceptance | `65ced169c8abe46637a395ce5988a827934bc316a0f3060a6108cf193acd06da`（passed/problems 空，`run_sha256=ad32fd51…`） |
| migration result | `ecee70205288fb725610af8b199b69ce28b3c62cafcf01070389824dec2c901f` |
| synthetic spec / snapshot | `e1ac7d7c…` / `43888f4f…`；synthetic legacy head `97a75062…`（与真实头 `b499a6ce…` 不同，属预期） |
| return observation | `177fca06441a4f6a0c3c55aa01f06ac451776c04b17ad36f47ed402a1303a62e`；journal `known=true`、0 problems |
| 工件 | 7,404,765,832 bytes |

Gate B v2 `binding_migration_gate_b/attempt_002/synthetic_acceptance.json` SHA **`e937c27adf1cf6cddba69f123b781ef19290a25e9bb83d4bbcc07414daf38c66`**，`passed=true/problems=[]`。消费端 `synthetic_acceptance_v2_problems` 重算通过，校验两侧 child acceptance、阶段报告/数组摘要、候选收据、commit、外层 return observation（含 claim/覆盖/exit）、launch journal 与独立重算的 legacy head。

P/I/F 对比（`binding_migration_gate_b/attempt_002/gate_b_comparison.json` SHA `0ff9f07d134652e0f2c14ab49cce894078c3bb82dffcd352ba3a0c2f8919eb27`，`passed=true`）：

- 普通 attempt_015 与 attempt_013：**12/12 文件字节完全相同**（8 个 NPZ + 4 份 rows）。
- 迁移 attempt_002 与 attempt_013：**12/12 文件字节完全相同**（P 位于 `production/attempt_002`）。
- 两链均为 582 gap、0 failure、`audit_incomplete`；未因迁移放宽任何数值门。

### 6.3 M5：只读迁移准入与 v2 授权草案

M5 材料在**最终文档树**上重新生成；精确文件 SHA 与逐项绑定值统一记录在 `binding_migration_authorization/attempt_002/m5_verification.json`，避免把易受文档树 git 身份影响的摘要写死后再次漂移。下表的路径与结果在生成时核验：

| 材料 | 结果 |
| --- | --- |
| real metadata admission `admission_real/attempt_016/attempt_001/admission_package.json` | passed/problems 空，`metadata_only`，`authorized=false/registered=false/run=not_run`；SHA 见复核 JSON |
| history snapshot `binding_migration/attempt_003/history_snapshot.json` | legacy head `b499a6ce…`；cost 67.10273322396097s；1 个未封口旧文件；SHA 见复核 JSON |
| protected manifest v2 `binding_migration/attempt_003/protected_manifest.json` | 94 entries；drift 0；SHA 见复核 JSON |
| binding spec `binding_migration_spec/attempt_003/binding_spec.json` | 绑定 snapshot/Gate B/real admission/28 文件实现/runtime；SHA 见复核 JSON |
| migration admission `binding_migration_admission/attempt_002/migration_admission.json` | passed/problems 空；SHA 见复核 JSON |
| v2 授权草案 `binding_migration_authorization/attempt_002/authorization_draft.json` | `authorized=false/template_only=true`，10 项 checklist 均 null，`authorization_problems` 为预期 11 项（`not_authorized` + 10 项 checklist）；SHA 见复核 JSON |
| M5 独立复核 `binding_migration_authorization/attempt_002/m5_verification.json` | passed/problems 空；`implementation_sha256` 与当前最终文档树逐位一致 |

M5 复核事实：

- 草案绑定 snapshot/spec/migration admission/legacy head/Gate B/real admission/plan/prereg/implementation/runtime，`prior_updates=2`、`incremental_update_limit=16`、`cohort_update_limit_for_this_authorization=18`、`prior_cost_s=67.10273322396097`、`remaining_s=3172.897266776039`、`invocation_id=w1-migration-20261104-001`。
- 原真实根仍 **10 文件 / 466,350,106 bytes**，文件集合与 snapshot 完全一致；无 `launch_journal`、无新 attempt/registration/authorization；`scan_launch_journals(HISTORY_ROOT)` 为 known、0 records；`verify_history` 无问题。
- 草案不含未来 R0/commit/return observation 的 SHA；真实 admission / migration admission / snapshot / spec 均为只读，未加载 checkpoint payload、NPZ 数值、模型、MuJoCo 或执行 forward/VJP/optimizer/cohort 注册。
- 当前实现树为**未提交工作区**（文件字节已绑定）；M6 授权前若提交或改写任何实现/方案文件，必须重新生成 spec/authorization draft 并重新核验，不能沿用当前 `implementation_sha256`。

### 6.4 边界

本轮未读取真实 checkpoint payload、父 NPZ 数值、数据 cache 或真实资产编译；未构造真实模型/优化器；未执行真实 forward/backward/FK/AdamW；未创建真实授权、真实 R0_002/P_002/I_001/F_001 或真实 reservation。M4 的 16 次普通合成更新与 18 次迁移合成累计更新均为合成证据，不增加真实额度。M6 仍需绑定最终字节的新明确授权；旧一次授权、M3–M5 实现或本轮草案均不得解释为 M6 授权。`partial/null/false/not_ready/不接入` 与旧质量结论保持。

## 7. 复核 2 P1 + 1 P2 修复与最终冻结（2026-11-04）

独立复核指出三项阻断：真实迁移入口缺少启动器；合成返回证明不是实际进程返回；提交后 M5 完整身份失配。三项均已修复，并按最终代码重新完成回归、Gate B 与 M5。

### 7.1 真实迁移启动器

`--real --migration` 不再直接调用 `run_binding_migration_audit`。新流程：

1. 校验 v2 授权、spec/snapshot、real admission、Gate B/实现/预算绑定；`--synthetic-config-json` 与真实模式互斥；
2. 原根必须仍是迁移前状态（无 `bindings/generation_002`、`launch_journal`、`execution_events`、v2 行）；
3. 由外层写入 `bindings/generation_002` 与 `launch_journal/<invocation>/claim.json`（claim t0 取入口时钟）；
4. 写入 worker job，spawn 独立 `--migration-worker` 子进程；worker 重新校验授权/准入/claim 后执行真实 R0→P→I→F；
5. 外层 `wait()` 后读取 compact worker 结果，按实际 pid/ppid/exit/signal/时钟写 `return_observation.json`；缺结果时写 `failure.json` 并保持未结清；
6. 公开读取器只有在实际返回证明与 journal 闭合后才可消费。

### 7.2 实际进程返回证明

- 普通与迁移合成 CLI 均改为 spawn `--synthetic-worker` / `--migration-worker` 子进程；外层在真实 `wait()` 后写返回事实，不再由同进程从 conclusion 推导退出码。
- 迁移 worker 只写 compact 结果（commit/candidate/stage timing/报告引用），不写返回观察或 acceptance；外层在返回后构建 acceptance。
- `record_return_observation` 的起点取 claim t0；`--migration-synthetic` 的 worker exit=3（`audit_incomplete`）被如实记录，外层 acceptance 成功可返回 0，两者不再矛盾。
- F 最终阶段新增 `FINAL_PROCESS_RETURN_TAIL_S=3.0` 的计费地板，覆盖真实进程退出/清理；若实际返回仍超过该上界，返回观察仍会拒绝。

### 7.3 实现身份稳定化

新增 `implementation_files_sha256(identity)`，授权草案与执行端只绑定 **28 个实现文件的字节**；spec 仍完整记录 git identity 供审计。文档/AGENTS 提交不再改变 `implementation_sha256`，解决“文件字节一致但 git 状态不同”导致的 drift。

### 7.4 最终回归与 Gate B

| 项目 | 结果 |
| --- | --- |
| 定向三文件 | **144 passed（427.19s）** |
| M3 回归集 | **380 passed（467.05s）** |
| 迁移测试 | 27 项（含普通/迁移 CLI 实际子进程返回、files-only 身份、binding/claim/acceptance 闭合） |

最终 Gate B：

| 材料 | SHA-256 / 结果 |
| --- | --- |
| calibration contract | `1192da63ca780077728432c3902116f92df9d02489dcd9d8ec48545f47363848` |
| calibration verification | `2a1e7d1b4613749416f5278fc1a8e74ed5f7086c50a1c2ff04b156b46e5674c2`（canonical `66d47470…`；`recompute_problems=0/four_x_ok=true`） |
| ordinary acceptance | `ade5e22cd7b6d8c33efa3fab0c85ee9f0131d88bdb0787707fb029db0828cef2`（passed；worker exit 3，actual pid/ppid 记录于 return） |
| ordinary return | `e1082f0ee1b4cd31c7a53de1e7cd3bb50d4cd884fbccb4b496b891070a3c19f1` |
| ordinary audit_result / final / registration | `d35964b7…` / `51765f52…` / `9abdd477…` |
| migration acceptance | `92f3203a4b93f98904d611c53f4d2e708ebf5c18b76f688a24df5403dfff7c3c`（passed；cohort=18，selected=16） |
| migration result / spec / snapshot | `45080f3d…` / `d3a8f467…` / `97c2c83a…` |
| migration return | `680a59a992871cb39bf93f30d76caee0b2d207a81992f5b1e1f0ed28b73238c3`（actual worker pid 511967/ppid 511749，exit 3） |
| Gate B v2 | `046f30a1d63e2bf87505b26e4e7cf4624a77873a2b38c3041e8ad1f75b2c89cb`（passed/problems 空） |
| 对比 | `24602e0f…`；普通/迁移各 **12/12 文件与 attempt_013 逐字节相同** |

校准 contract file 随实现身份变化；evidence `b848cbb4…`、fp32 `574d9a0a…` 与 attempt_013 相同；阈值/规则/零底逐项不变。

### 7.5 最终 M5

| 材料 | 结果 |
| --- | --- |
| real metadata admission `admission_real/attempt_018/attempt_001/admission_package.json` | `239033409400b2159215cdd0a6a42149b7a90d8c798269828962c8c75c6c379b`，passed/metadata_only |
| history snapshot `binding_migration/attempt_005/history_snapshot.json` | `4ab581a2b8db81abd0287612b569e519dce4f78dc15f8dcd3e095229ec74320e`；legacy head `b499a6ce…`；cost 67.10273322396097s |
| protected manifest v2 | `2c37d681…`；94 entries；覆盖 calibration 017 / synthetic 017 / admission 018；drift 0 |
| binding spec `binding_migration_spec/attempt_005/binding_spec.json` | `b2cbfae4752615ff239ed2cc1183521abede458246f149921eb61fa40e027db4` |
| migration admission `binding_migration_admission/attempt_003/migration_admission.json` | `ba12a74c…`，passed/problems 空 |
| v2 授权草案 `binding_migration_authorization/attempt_004/authorization_draft.json` | `7a027deb…`；`authorized=false/template_only=true`；10 项 checklist null；`implementation_sha256=22c53707…`（files-only） |
| 独立复核 `binding_migration_authorization/attempt_004/m5_verification.json` | passed/problems 空；`draft_implementation_sha_match=true`、`spec_files_match=true`、manifest drift 0 |

原真实根仍 **10 文件 / 466,350,106 bytes**，无 launch journal、无新注册/授权/数值更新；`scan_launch_journals` known/0 records；`verify_history` 无问题。`authorization_problems` 仍为预期 11 项（`not_authorized` + 10 项 checklist），草案不可直接执行。

### 7.6 边界

未读取真实 checkpoint payload/父 NPZ 数值/资产编译，未构造真实模型/优化器，未执行真实 forward/backward/FK/AdamW，未创建真实授权/cohort/注册或真实 R0_002→P_002→I_001→F_001。M6 仍需绑定最终字节的新明确授权；旧一次授权、M3–M5 实现或本轮草案均不得解释为 M6 授权。`partial/null/false/not_ready/不接入` 与旧质量结论保持。

## 8. 第三轮复审 2 P1 修复与最终字节重冻结（2026-11-04）

### 8.1 修复

1. **P1 条件导入导致真实入口 `UnboundLocalError`**：`main()` 的 `--migration-authorization-draft` 分支曾在函数内重复导入 `load_external_binding_material`，使该名字在整个函数内成为局部变量；真实迁移分支未执行该导入时直接未绑定。已删除重复条件导入，统一使用模块级导入。
2. **P1 新鲜根检查拒绝本次 claim**：bootstrap 先写入 `launch_journal/<invocation_id>/claim.json`，原 `_fresh_migration_root_problems` 却无条件拒绝 `launch_journal` 存在。现先以 `launch_claim_record` 核验当前 claim（授权/spec/legacy/marker/目标/当前进程），再以 `allow_invocation` 仅豁免“该 invocation 目录且只含 `claim.json`”的当前记录；其他 invocation、`failure/revoke`、`.tmp` 或额外文件仍拒绝。

### 8.2 回归

| 项目 | 结果 |
| --- | --- |
| 迁移测试 | **30 passed（114.70s）**，新增当前 claim 豁免反例与“全部预检通过后到达 worker 分派”的隔离测试 |
| 迁移 + audit + provider | **148 passed（346.39s）** |
| 尾部预留 | `_charge_final_verification` 重封口保留 `FINAL_PROCESS_RETURN_TAIL_S=3.0`；受控时钟反例通过 |

### 8.3 最终 Gate B（重冻结）

| 材料 | SHA-256 / 结果 |
| --- | --- |
| calibration `attempt_018` contract | `7f8bd191d9254cd6a5c491ffb6af71e3c4a1cd951813dff82da9ba41843b6f32` |
| calibration verification | `31992515b44fab40fb52f11ef12dbe51f555fdc757c5df4809eb41e2e1e9dfc1`（canonical `4622f9bd…`；`recompute_problems=0/four_x_ok=true`） |
| calibration evidence / fp32 | `b848cbb4…` / `574d9a0a…`（与 attempt_017 相同） |
| ordinary `synthetic/attempt_018` acceptance | `142123d2a1a2543f4bca7f851c7d4019faa697371ed4e785567e6beb84344bfc`（passed；exit 3） |
| ordinary return / audit_result / final / registration | `f8a856af…` / `a86800b5…` / `2a2d3a83…` / `23a51525…` |
| migration `binding_migration_synthetic/attempt_005` acceptance | `9b3e54e1a507d463fce660f848be6f674a6e2049a828e5e832926c0fd12f65e7`（passed；cohort=18，selected=16） |
| migration result / spec / snapshot | `3421ea5d…` / `6a8ce5fc…` / `6dc14fd3…` |
| Gate B v2 `binding_migration_gate_b/attempt_005` | `c59e719ab53853d770e688f1dcd07109602dcd476c6e74417dbb437acfc5ef3c`（passed/problems 空） |
| 对比 | `266fc016…`；普通/迁移各 **12/12 文件与 attempt_013 逐字节相同** |

### 8.4 最终 M5（重冻结）

| 材料 | 结果 |
| --- | --- |
| real metadata admission `admission_real/attempt_019/attempt_001/admission_package.json` | `62d1965f48b7b90645dad24f089e58eaec0aad1fa0b1285380a6445f9f4eead1`，passed/metadata_only |
| history snapshot `binding_migration/attempt_006/history_snapshot.json` | `42ffd63f1dca3b695f6ffed523e22fc70186138c4b177f37c5645993f36a08b6`；legacy head `b499a6ce…`；cost 67.10273322396097s |
| protected manifest `binding_migration/attempt_006/protected_manifest.json` | `7c5c87e321dc30fcea7ce5e75a2607f28cc5eec6e19dfda993dcb1add953cdae`；94 entries；drift 0；覆盖 calibration 018 / synthetic 018 / admission 019 |
| binding spec `binding_migration_spec/attempt_006/binding_spec.json` | `5b0feaa5f3151a5560f42179d219315017ea6d86cfb58c594b76f7e5b5528d78` |
| migration admission `binding_migration_admission/attempt_004/attempt_001/migration_admission.json` | `b900f8d21d20ee1cfd3941e1b2cea597d3dae932445de579627bee0cac213d18`，passed/problems 空 |
| v2 授权草案 `binding_migration_authorization/attempt_005/authorization_draft.json` | `d83d4064aead0fe97107e72b14ede49de04a060ba7077b44d81b31941125a4a4`；`authorized=false/template_only=true`；10 项 checklist null；files-only `implementation_sha256=0cc0622c…` |
| 独立复核 `binding_migration_authorization/attempt_005/m5_verification.json` | `60f1a7a07f334675179a72d58245fd87f6f87e9949cffaa20b044be445b6cf68`；passed/problems 空；draft/spec files 与当前 28 文件一致、manifest drift 0 |

实现身份为 **`0cc0622c86829b3e4bcbdefc1a3c1edb552ec02b31294dfd98bb70fd563205d8`（28 文件）**。原真实根仍 **10 文件 / 466,350,106 bytes**，无 launch journal、bindings、新注册/授权/数值更新；`scan_launch_journals` known/0 records；`verify_history` 无问题。M6 未执行；旧 §7 的 017/004 系列保留为历史，不能作为本轮最终字节的授权依据。
