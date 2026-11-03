# W1 训练更新绑定迁移实现与验证记录

日期：2026-11-03。状态：**M0 完成；M1 核心接口完成；M2 部分反例完成；M3–M6 未执行**。本记录只描述本轮实际写入的代码、合成测试和只读迁移材料，不构成真实恢复授权，也不替代 Gate B attempt_013 的既有证据。

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
- `scan_launch_journals`：所有历史/预算/准入读取器扫描固定 history root 的 launch journal；未结清 claim/failure/revoke、缺 registered/terminal 候选或坏返回观察均使历史 unknown。

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

## 4. M2：本轮已执行的反例

新增 `tests/test_w1_real_training_update_migration.py`，共 **9 passed（27.98s）**：

1. 真实原根独立重算的 legacy head 等于 `b499a6ce…`，两条记录为 R0→P，旧累计更新为 2；
2. 合成迁移根按 seq 3→6 完整执行第二代 R0_002/P_002/I_001/F_001，`complete_evidence=true`，selected chain 更新 16，cohort 更新 18；
3. 旧 R0_001/P_001 无 seq 字段且 canonical 内容保持不变；新行全部为 v2 schema、`generation_002`、正确 predecessor；
4. 外层返回观察前，公共 `stage_state` 因缺 `return_observation` 拒绝；写入外层返回事实后四阶段均可消费；
5. 迁移根未指定代次返回 `generation_ambiguous`；旧 generation_001 行仍存在且只读保留；
6. 篡改执行序号后不能重推为已知链；
7. v2 授权拒绝更新额度重置（prior=0、incremental=18、remaining=3240）；
8. `stage_closed` 和 terminal 候选各有一个受控时钟慢写反例：尾部只预留 1 秒而实际 5 秒时，写后门否决并写入至少 4 秒的未覆盖下界 revoke；另有 return observation 缺来源/覆盖/过程返回事实的拒绝测试。

另运行既有相关回归：

- `tests/test_w1_real_training_update_audit.py` + 新迁移测试：**81 passed（116.10s）**；
- `tests/test_w1_real_training_update_provider.py`：**45 passed（149.10s）**。

全项目套件未重跑；这些数字不是 380 旧基线的替代。

## 5. 仍未完成 / 下一步

- M3：把本方案、新测试和最终源码纳入完整实现清单，重新冻结 Gate B；
- M4：重算校准、普通合成接受和迁移合成接受，确认普通链 16 次与迁移链累计 18 次，并逐项解释 P/I/F 数组/rows 与 attempt_013 的差异；
- M5：用最终字节生成新的 real metadata admission、迁移 snapshot/spec/migration admission 和严格 v2 授权草案；任何未结清 claim/failure/revoke 都必须使准入失败；
- M6：只能在 M0–M5 完成、复审无阻断并取得绑定最终字节的新明确授权后，执行一次真实 R0_002→P_002→I_001→F_001；不得把本轮实现或旧授权解释为 M6 授权。

`objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false` 及旧 U0/UR/原 F/r6/旋转融合结论全部保持。
