# W1 真实训练更新审计——既有 cohort 绑定迁移与有限恢复方案

日期：2026-11-03。版本：v0.13 草案，修订2（补齐成功收据写入后的实际耗时核验；修订1的2项P1与2项P2保留）。状态：**plan_only / not_implemented / real_resume_not_authorized**。

本方案采用原运行根追加第二代绑定、重新登记 R0、再完整重做 P→I→F 的路径。旧 R0/P 的终态、封口证据、2 次更新和全部成本继续保留；恢复链最多新增16次诊断更新，累计最多18次。此次仅制定方案与更新文档，迁移实现、Gate B 重冻结、新授权及真实恢复均未执行。

## 1. 事实锚点与冻结范围

源码基线为 `f808aa975ba2ccbabdddba348f9cbeb2b85b9dfb`。原 cohort/execution 身份均为 `w1_real_training_update_audit_v1`，原根为 [20261102_2704709_w1_real_training_update_audit](/home/c112/codes/bill/ireset/derived/experiments/omniretarget_first/20261102_2704709_w1_real_training_update_audit)。继续使用这个根及固定 history root，绑定代次不产生新的实验额度。

| 既有依据 | SHA-256 |
| --- | --- |
| [预注册 v0.9](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_real_training_update_audit_prereg_draft.md) | `e2ba8ef9edb7381dbe410f7e0e290a6fbd702323a4416eeacfee4d2d14796242` |
| [执行计划 v0.11](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_real_training_update_provider_execution_plan.md) | `e66cd8a9ac1d94abe8ea3f08aeadd45bec8e5d29f0f2baccfec948a4eca7cb4d` |
| [alpha 修正补充计划 v0.12](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_real_training_update_alpha_fix_plan.md) | `9b49e673d975fdd1fd61ea80e7926618d328750af6054b312827ca97fac646a0` |
| [原授权](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/authorization/attempt_012/authorization.json) | `199bfb30f11502aaf57acf38f1c49525ac58e511f870f230f223cb12b8525d78` |
| 原根 `cohort.json` | `79f42f5fc2bb205a4f8327b7897644e4759ff60bf189f4dd6054cb980e9f3f6b` |
| 原根 `attempts_index.json` | `78a2f97dc42545c21f2200a9353b4e1ce9f4dea3fde9cf88f27c29548932b950` |
| 原根 `costs.json` | `92773314541bec3b3d751c1627fb8cf78b331ea21303b8c753b8d4a447aa0f34` |
| R0 `registration/attempt_001/registration.json` | `6abab029788f0e87859a560e3e9866c9958758f5a8765b7092f34ca5d53848aa` |
| [首次执行观察记录](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/authorization/attempt_012/execution_observation.json) | `54ca9d2ad155fdac77c1d4d60acd3037cfb8ec289af3d31f492ebd72ab1394e4` |

[首次执行报告](/home/c112/codes/bill/ireset/docs/verification/omniretarget_w1_real_training_update_audit_report.md)记录 R0 attempt_001 done/complete=true，P attempt_001 incomplete/complete=false，I/F 未创建 attempt。登记成本分别为58.59990523397573和8.502827989985235秒，合计67.10273322396097秒；两份 reservation 已结算。原根10文件共466,350,106 bytes，含未封口的 `production/attempt_001/p_deterministic.npz`。该部分数组不可消费，不补造旧 P report/rows/diffusion 数组或旧最终报告。

当前实现依据仍为 [验证记录 §12](/home/c112/codes/bill/ireset/docs/verification/omniretarget_w1_real_training_update_audit_verification.md)的 Gate B attempt_013：

| 材料 | SHA-256 |
| --- | --- |
| `calibration/attempt_013/tolerance_contract.json` | `d8515d2f51f966da7324e8c0f385d9b12d034fcbe79fc160d82b6939bd65b649` |
| 校准 evidence / fp32 canonical | `b848cbb4e54553dd19daf69a471076ddfc3bd2c6030c55608413d746d0c096b2` / `574d9a0a2115e3619aa260f3eef4ef83c8ff85e377e2f23eebc078434a0d441e` |
| `calibration/attempt_013/calibration_verification.json` | `ac66dd8b15fa60885b8f6488316f3aed469584b78318197d5eb833d06426bd63` |
| `synthetic/attempt_013/synthetic_acceptance.json` | `8f655f41184234a61536f5bfb058609538d7d5381fff05b7b3775576838686d5` |
| `admission_real/attempt_013/attempt_001/admission_package.json` | `53de1b1e03d2928803a6140b10342d642e87f95d85fd52dd0d863f174e4073b2` |

上述材料均位于 `derived/audit/w1_real_training_update_audit/`。attempt_013 的380项测试、完整合成证据和 metadata 准入只证明 v0.12 实现；实施迁移后必须用最终字节生成新的 Gate B，不直接沿用其 passed。v0.9/v0.11/v0.12 方案、旧 Gate B、原授权目录和原终态 attempt 文件保持原字节。

## 2. 必须同时解决的三个接口约束

现有 [run_registered_audit](/home/c112/codes/bill/ireset/src/ireset/eval/w1_real_training_update_audit.py:7829)要求 resume 授权 SHA 与原 marker 一致，新实现/新准入又必须绑定新授权。新增专门的迁移入口与版本化验证，普通 resume 的同身份门继续有效。

现有 [_cohort_chain](/home/c112/codes/bill/ireset/src/ireset/eval/w1_real_training_update_audit.py:4768)按阶段 R0/P/I/F、再按 attempt 编号重推。真实顺序已有 R0_001→P_001；追加 R0_002 后，阶段排序会把它插到 P_001 前面，破坏累计前态。本方案为新记录引入执行序号，并用冻结的旧记录映射建立历史前缀。

现有 [stage_state](/home/c112/codes/bill/ireset/src/ireset/eval/w1_real_training_update_audit.py:5631)拒绝多个可消费候选。旧 R0_001 不应被撤销；新增 R0_002 后，选链必须明确绑定代次，并在该代次内保持唯一可消费。全历史的成本、次数、调用和完整性检查仍覆盖所有代次。

重新登记 R0 能在剩余的第二次 R0 attempt 内核验新实现/校准/准入，并生成正式的新 registration。旧 R0 只作为第一代历史证据保留，不被改签为新版本已通过的登记；旧 P 部分参数也不成为新 P 的起点。

## 3. 记录、身份与摘要依赖

以下文件与 schema 是拟实现接口，本轮不创建这些执行记录。

| 位置 | 内容及作用 |
| --- | --- |
| 原根之外的 `binding_migration/<attempt>/history_snapshot.json` | `w1_real_training_update_history_snapshot_v1`：原根规范路径、marker/原授权、10文件 SHA/bytes、旧 index/ledger 原始副本摘要、逐行 canonical SHA、2条执行顺序、24类余额、成本、attempt 使用量、未封口文件列表、§4.1的legacy_head_payload/legacy_head_sha256，以及原根启动日志清单摘要 |
| 同目录 `binding_spec.json` | `w1_real_training_update_binding_spec_v1`：旧 snapshot SHA、legacy_head、第一代身份、目标第二代身份、原根、新计划/实现/运行时/新 Gate B/新准入绑定、预定 stage/attempt/sequence、固定启动日志路径及其事前清单摘要、预算及增量上限、写后门/外层返回证明协议和执行窗口边界；不含未来授权或新 R0 SHA |
| 同目录 `migration_admission.json` | `w1_real_training_update_migration_admission_v1`：独立复核 snapshot/spec 与新 metadata 准入，声明 `authorized=false/registered=false/run=not_run`；不写原根，不加载 checkpoint/数组值或构造模型/资产 |
| 独立的新授权目录 | 新版授权草案及最终授权；草案 `authorized=false/template_only=true`，正式记录另行产生并绑定 spec、migration admission 及新 Gate B |
| 原根 `launch_journal/<invocation_id>/` | 授权后、uv/import及任何实质预检前写入不可覆盖的claim.json；后续追加registered.json、各stage_closed候选及terminal候选/failure/revoke记录，执行返回后由外层调用方存档return_observation.json。候选收据自身不是成功证明。该目录是所有历史/预算/准入读取器必须扫描的启动与成本负债来源，外部授权目录中的副本不替代它 |
| 原根 `bindings/generation_002/` | 授权后写入 spec 的逐字节副本、`intent.json`、旧 index/ledger 原始快照及`activation.json`；绑定生效引用`execution_events/000003/commit.json`，不再另设一份binding commit；原`cohort.json`不改写 |
| 原根 `execution_events/<sequence>/` | 新 attempt 的不可覆盖start事件、带编号的prepare_revision、最终finish/commit事件；存储对应行快照及SHA、前驱头、同次invocation、绑定与reservation状态。commit仅在准备态核验和计费稳定后写一次 |

新授权、登记和最终报告分别使用 `w1_real_training_update_authorization_v2`、`w1_real_training_update_registration_v2` 和 `w1_real_training_update_audit_v2`，新行标记 `accounting_schema=pre_registered_binding_v2`。新的 synthetic acceptance v2 同时绑定普通合成链与迁移合成链，两者均为准入硬门。旧 v1 记录按冻结契约只读解析，旧消费者对不支持的新 schema 明确拒绝。实现集中修改现有 audit/provider/CLI 及测试，不引入通用迁移框架。

第二代身份由 spec SHA 唯一确定，并显式保留`cohort_id/execution_id`、原marker SHA、原授权SHA和原根realpath。新授权绑定spec/admission；启动claim绑定该授权、spec、legacy_head及invocation；新R0 registration再绑定claim、新授权和新Gate B。activation绑定这些文件及新R0 registration/admission/noise/reservation的SHA；最终finish记录稳定后的index/ledger行SHA，commit绑定finish、activation和前驱头；stage_closed候选再绑定commit。P/I/F绑定已提交且通过写后门的新R0 activation/commit、新R0 SHA和spec。外层return_observation绑定候选收据、commit及实际返回事实，不被这些前序文件反向引用。spec不反向引用授权，registration/activation不引用包含自身摘要的commit；当前行及其artifacts不保存自身finish/commit、stage_closed或return_observation的SHA，避免自摘要环。

原 marker 的授权字段仍解释为第一代授权。存在迁移记录时，第二代身份必须通过完整的 spec→新授权→R0→activation/commit 链求得；不得覆盖 marker、默认选择最新目录或因 SHA 不符降级为普通 resume。无迁移记录的旧根保持原验证行为。

## 4. 历史前缀与执行顺序

目标执行映射固定如下；仅当各阶段成功且预算仍足够时才登记下一行。

| execution_seq | 代次 | 阶段 / attempt | 处理 |
| ---: | --- | --- | --- |
| 1 | generation_001 | R0 / attempt_001 | 只读映射，原 done/complete=true 保留 |
| 2 | generation_001 | P / attempt_001 | 只读映射，原 incomplete/complete=false 保留 |
| 3 | generation_002 | R0 / attempt_002 | 新登记与绑定迁移，使用剩余 R0 预算 |
| 4 | generation_002 | P / attempt_002 | 从固定 checkpoint 与 fresh AdamW 完整重做 |
| 5 | generation_002 | I / attempt_001 | 独立参考更新 |
| 6 | generation_002 | F / attempt_001 | 一个新 worker 完整重放与终检 |

旧行不补写seq、代次或新授权字段。snapshot通过旧行SHA映射seq=1/2；新行、ledger、reservation和事件均显式保存generation/spec SHA/seq、`predecessor_kind`及`previous_head_sha256`。统一定义`canonical_json(value)`为UTF-8 `json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False)`字节，`H(value)=SHA256(canonical_json(value))`；不转换原数值类型，解析时拒绝重复JSON键。

### 4.1 旧v1前缀的唯一承接头

旧seq=1/2没有commit。本方案不为它们补造提交文件，而是在snapshot内嵌以下固定字段组成的`legacy_head_payload`，并令`legacy_head_sha256=H(legacy_head_payload)`：

| 字段 | 精确定义 |
| --- | --- |
| schema | `w1_real_training_update_legacy_head_v1` |
| cohort_id / execution_id / run_root | 原marker中的两项身份及原根规范绝对路径 |
| marker_sha256 / authorization_sha256 | §1原marker字节摘要及其原授权SHA |
| source_index_sha256 / source_ledger_sha256 | §1原index/ledger的完整文件字节摘要 |
| last_execution_seq | JSON整数2 |
| records | 按seq升序的两个对象，每个对象且仅含execution_seq、stage、attempt、index_row_sha256、ledger_row_sha256、reservation_sha256；前两行分别为R0/attempt_001、P/attempt_001，行SHA使用H，reservation SHA使用原文件字节 |
| call_balance_after | 原P_001的call_balance.counts完整24键整数映射，须与两个已结算reservation按序重推的结果相等 |

payload不含snapshot自身SHA、legacy_head_sha256、spec、新授权或任何新记录，因此无摘要循环。本轮只读原JSON重算出的固定头为 **`b499a6ce7e5ef8fac06821cc063f27bd109374d9673791551fe92ad56302a424`**；这只是方案中的派生值，本轮没有创建snapshot或迁移事件。

seq=3的start/index/ledger/reservation必须使用`predecessor_kind=legacy_head`和`previous_head_sha256=legacy_head_sha256`。读取器须从仍受保护的两个旧行及reservation独立重算payload，核对上述固定值、snapshot、spec和新授权的绑定，不能只信任嵌入摘要。source_index/source_ledger的原字节SHA从已冻结的原始副本校验，现用视图的两个旧行canonical内容须与副本相等；不能要求已追加新行的视图仍具有旧整文件SHA。seq≥4必须使用`predecessor_kind=attempt_commit`及前一个执行序号commit文件的完整字节SHA；空前驱、旧行SHA、snapshot文件SHA或任意其他hash均不等价。legacy_head分支只允许此既有两条v1记录后的seq=3。

### 4.2 按前驱头重推与选链

重推先核对冻结前缀，再按seq验证每条`before=previous_after`、`after=before+reservation.consumed`，同时核验index/ledger/disk集合、状态、成本和前驱头链。seq重复/跳号/回退、零增量导致的顺序歧义、负计数、bool冒充整数、未结算reservation或缺任一副本均为unknown。仅§6限定的持锁控制进程/已绑定worker可显式处理其当前一个in-flight记录；这不是供新准入或普通选链使用的全局known结果。不得改用时间戳或寻找任意可拼接顺序修复不一致。

`attempts_index.json` 与 `costs.json` 是会追加新行、结算当前行的视图，未来整体字节允许改变；**两个旧终态行的 canonical 内容必须不变**，旧副本原始字节保存在迁移快照中。原终态 attempt 下的 registration/admission/noise/reservation/partial NPZ 均不改写。当前新行的 running→terminal 变更须由相应事件记录支持，不允许借结算重写历史前缀。

选链入口显式接收 generation/spec SHA。先验证整个 cohort 历史，再在指定代次中要求 R0/P/I/F 各唯一可消费、父 registration 一致。旧 R0_001 仍可被第一代只读查询，但不能成为第二代 P/I/F 的上游。显式 attempt 也不能绕过同代次多个可消费或跨代次混用。存在迁移的根在未指定代次时返回身份歧义，不默选第一代或第二代。

固定history root的同cohort扫描继续纳入所有可归属根、迁移记录和§6.1固定位置的启动日志。复制、改名、软链接、改变generation或更换外部授权目录不产生新余额；额外归属根、分叉身份链或无法识别/未结清的启动记录使预算unknown。

## 5. 成本、attempt、调用与更新上限

以原账本 `measured_wall_s` 原值计算，不以一位小数 `wall_s` 或外层启动耗时覆盖。表中余量为制定方案时快照，真正准入必须重算且与快照一致。

| 阶段 | 原累计秒 | 原阶段上限秒 | 剩余秒 | 工作前保留秒 | 已用 / 原最多 attempt | 本次最多新增 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | 58.59990523397573 | 240 | 181.40009476602427 | 30 | 1 / 2 | 1 |
| P | 8.502827989985235 | 900 | 891.4971720100148 | 120 | 1 / 2 | 1 |
| I | 0 | 900 | 900 | 120 | 0 / 2 | 1 |
| F | 0 | 1200 | 1200 | 180 | 0 / 2 | 1 |
| 合计 | 67.10273322396097 | 3240 | **3172.897266776039** | 按原阶段规则保留 | — | 一次恢复入口 |

阶段不互借预算。迁移检查、启动/uv/import、旧文件摘要、快照、绑定发布、封口与消费核验全部计入新 R0，扣除这些耗时后才是该阶段的工作余量；不会新增独立免费的迁移时间窗。后续 P/I/F 各自包含启动、读取/解码、构造、数值工作、封口和消费/终检成本，含stage_closed/terminal候选的写入、文件及目录fsync、回读与写后检查。阶段交接时下一阶段t0在本阶段写后门通过的交接点取得，准入及交接开销不得留在两个时钟之外。原reserve、最小1秒seal-tail和有界封口轮数保持，尾部预留只是计费上界候选，不能替代写入返回后的实际越界检查。执行窗口及外层返回观察的明确边界见§6.2，并须绑定到新spec/授权。

下表为完整恢复成功时的24类矩阵；列顺序为原历史、恢复新增 R0/P/I/F、预计 cohort 累计、原 cohort cap。每个新阶段仍受原单阶段 nominal 与 reservation 约束，累计不重新从0开始。

| kind | 原历史 | 本次新增 | 累计 | 原 cap |
| --- | ---: | ---: | ---: | ---: |
| checkpoint_loads | 2 | 8 | 10 | 16 |
| model_constructions | 2 | 8 | 10 | 16 |
| conditioner_constructions | 2 | 8 | 10 | 16 |
| native_optimizer_constructions | 2 | 4 | 6 | 8 |
| conditioner_forwards | 8 | 64 | 72 | 128 |
| backbone_forwards | 8 | 64 | 72 | 128 |
| native_add_noise_batches | 0 | 32 | 32 | 64 |
| cpu_draw_units | 8 | 48 | 56 | 96 |
| production_loss_assembler | 8 | 32 | 40 | 64 |
| production_float32_fk_batch | 16 | 64 | 80 | 128 |
| production_parameter_backward | 8 | 32 | 40 | 64 |
| independent_output_vjp | 0 | 32 | 32 | 64 |
| independent_total_vjp | 0 | 32 | 32 | 64 |
| native_global_clip | 2 | 8 | 10 | 16 |
| independent_clip_formula | 0 | 8 | 8 | 16 |
| native_adamw_steps | 2 | 8 | 10 | 16 |
| explicit_reference_updates | 0 | 8 | 8 | 16 |
| diagnostic_parameter_update_events | 2 | 16 | **18** | 32 |
| anchor_state_restores | 0 | 4 | 4 | 8 |
| optimizer_formula_checks | 0 | 8 | 8 | 16 |
| reference_mj_forward | 0 | 15840 | 15840 | 31680 |
| reference_mj_jac_site | 0 | 115200 | 115200 | 230400 |
| scene_setup_mj_forward | 1 | 2 | 3 | 4 |
| micro_execution_units | 8 | 64 | 72 | 128 |

原 `updates.diagnostic_parameter_update_events=16` 继续表示选中完整链的期望，`cohort_hard_cap=32` 不变。新版授权额外严格绑定 `prior_updates=2`、`incremental_update_limit=16`、`cohort_update_limit_for_this_authorization=18`，并校验三者关系；不得把18次写成16次或把原32次cap视为本次可继续使用的额度。新链 P=4、I=4、F=8；其中原生 AdamW 合计8次、显式参考更新8次，历史另有原生2次。

资源沿用 GPU peak≤22 GiB、host peak≤32 GiB、同 cohort 工件≤16 GiB、启动 free≥20 GiB、单 GPU/单数值 worker。新增绑定/快照/事件文件总上界32 MiB，计入同 cohort 工件；保留原466,350,106 bytes及所有失败历史。实施后按实际 schema、行数和数组清单重新计算“旧工件+新全链+迁移元数据”上界，不直接复制旧7.145 GiB估计，不通过删除历史满足门槛。当前磁盘检查只证明本轮文档工作的快照，不替代未来启动检查。

## 6. 发布状态与中断处理

未授权的准备阶段只允许在原根之外生成 metadata snapshot/spec/admission/授权草案。它不声明真实阶段完成，不产生 activation、真实 attempt 或可用 reservation。再次实际启动时仍须在计费截止内复核摘要和全部绑定，外部预检不会使真实执行免检。

### 6.1 固定启动日志与登记前负债

启动日志的唯一权威位置固定为**原根下的`launch_journal/<invocation_id>/`**，不能由授权目录或CLI另行指定。最小启动器在uv和项目/数值库import前记录t0、持有原根独占锁，并用不可覆盖写入持久化`claim.json`。写claim所需的最小授权/路径解析也计入该t0；claim完成前不做父文件摘要、完整准入、数组/模型读取或任何数值工作。本轮不创建这个目录。

claim使用`w1_real_training_update_launch_claim_v1`，至少包含cohort/execution、原根realpath、原marker SHA、新授权/spec/legacy_head SHA、唯一invocation_id、目标R0/attempt_002/seq=3、t0及启动身份。写入临时文件后完成文件/目录fsync和不可覆盖发布，再允许实质预检。若创建目录或claim期间中断，留下的孤立目录/临时记录同样是必须发现的unknown，不能被扫描器忽略。

所有`cohort_history/cohort_attempts/prior_call_counts/budget_for/verify_history/stage_state`、真实及迁移准入、执行入口和F预检均通过同一日志读取器扫描固定history root内每个可归属cohort根的这个位置；**即使该根仍只有v1 marker、两条旧行且尚无bindings/intent，也必须扫描**。读取器不以本次授权目录或本次invocation作为历史过滤条件。只有已验证的原v1根、且没有任何v2启动/迁移线索时，目录不存在才可作为空清单；无法枚举、未知schema、缺claim、错身份、分叉或未结清记录均使新准入/预算unknown。

M0/M5在原根之外保存该目录的完整相对路径/SHA/bytes清单，并将`launch_journal_manifest_sha256`和固定路径绑定到snapshot/spec/迁移准入/授权。实际领取只允许在事前清单上增加该授权声明的一次invocation；更换或新建外部授权目录不会改变扫描集合。外部授权目录可保存日志副本，不能成为唯一负债证据或替代原根日志。

R0登记后追加`registered.json`，绑定同一invocation与seq=3及start事件；每阶段结算后追加`stage_closed_<seq>.json`候选，绑定其最终commit和成本行。该候选的写后门通过后才对当前执行上下文闭合，公开历史还须有§6.2的有效外层返回观察。负债归属键为`(invocation_id,stage,execution_seq)`：已证明覆盖且闭合的启动耗时只由对应`measured_wall_s`计一次，不能把claim耗时再加一次；无完整闭合证明的claim/failure/revoke不得视为零耗时或释放额度。登记前退出时追加`failure.json`保存已知墙钟和原因，仍标记未对账；若连failure无法写入，未结清的claim本身就足以使历史unknown。

公开只读准入及新启动不能豁免待结清日志。当前持锁控制进程及其绑定worker可通过显式执行上下文处理同一invocation当前阶段的in-flight claim/row/reservation，保留已验证历史前缀与当前实际计数/计时；该上下文必须核验授权、spec、claim、seq、前驱头、进程归属及当前写后门，不能把in-flight状态伪装成可重新领取额度的全局known。其他invocation、其他阶段或旧记录没有该例外。整个入口结束还需terminal候选与有效return_observation；缺失返回证明、结算链或存在未对账failure/revoke时，不因成功文件存在、进程消失或锁释放而恢复准入。已正确闭合的失败记录只结清已证明成本，不恢复其一次使用权或可消费声明。

### 6.2 准备态核验、计费与一次性提交

每个新阶段都使用下列发布协议。R0的activation只是待生效数据，只有成功R0的最终提交及日志闭合满足后才激活绑定。

| 状态 | 允许的操作与进入下一状态的硬门 |
| --- | --- |
| launch_claimed | §6.1的claim已持久化，一次使用权已领取；启动/迁移耗时计入新R0 |
| registered / running | 使用旧累计余额登记当前stage/attempt/seq、start事件及reservation；R0使用§4.1的legacy_head，后续阶段使用前一commit；副本相符后才工作 |
| prepared | 数值工作停止、实际reservation已结算，报告/选择字段/activation等工件写完；写带编号的prepare_revision候选及对应index/ledger视图，此时没有当前最终commit，不能供下游消费 |
| verified / cost_stable | 仅当前控制进程通过内部准备态核验检查全部证据，随后把核验耗时计入cost_floor，完成有界重封口和最终小型元数据回读；验证/计费稳定后冻结候选行与工件摘要 |
| committed / closing | 写一次最终finish和commit，其写入/fsync/回读返回后检查实际耗时及覆盖；尚不供下游消费 |
| stage_close_candidate | 写匹配的stage_closed候选；候选本身的写入、文件及目录fsync、回读全部返回后，再执行同一阶段实际时间/计费覆盖/身份检查；通过前不得激活R0或进入下游 |
| stage_closed / next_stage | 写后门通过且done/complete=true，持锁上下文才生成绑定候选SHA、commit、seq及实测时点的内存凭据；凭据未撤销、同代次上游唯一且预算仍够，才激活成功R0或登记下一阶段。该凭据不是公开历史的成功文件 |
| terminal_candidate / returning | 最后阶段停止所有工作，写terminal候选并完成其fsync/回读后再次检查实际耗时；随后只做受计时的清理/返回，不再启动数值或验收工作。内层成功文件不等于成功返回 |
| observed_return | 外层调用方实际取得执行进程的返回结果，验证最后阶段覆盖到进程实际结束，再存档return_observation；公开消费者至此才可判断闭合成功。一次入口永久终结，不自动重试 |

拟新增内部`_verify_prepared_attempt`，只允许当前持锁控制进程对授权上下文指定的一个候选执行，返回准备态事实及错误，不返回公开可消费行、不生成selected声明、不启动下游。它核验完整历史前缀、父阶段已提交状态、当前候选index/ledger快照与实际consumed、所有必需工件及引用闭包、身份/runtime/预算。**它不要求当前阶段已有commit**；公开`stage_state`则始终要求最终commit、stage_closed/terminal候选、有效外层return_observation和未撤销状态。执行中的下游仅可使用§6.1限定且已通过写后门的显式上下文，两条路径不混用。

准备态先完成全量工件/引用核验和F最终选择，再调用v2计费/重封口路径；现有`_charge_stage_consumption/_charge_final_verification`的v2分支消费准备态核验结果，不能先调用要求当前commit的`stage_state`。`cost_floor`只能在当前commit写入前增加。每次修订写新的prepare_revision，最终小型元数据回读需核验成本不减、预算/complete状态、index/ledger/reservation及工件摘要一致；若又变动报告/selection/引用工件，必须重新准备态核验，不能沿用过期证据。

最终finish内嵌或绑定冻结候选的index/ledger行快照及canonical SHA、reservation SHA、工件清单与准备态核验结果；commit绑定finish文件SHA、当前seq/spec/claim和前驱头，R0另绑定activation SHA。当前行及artifacts不反向引用自身finish/commit/stage_closed/return_observation；这些文件由独立提交验证器读取。公开消费者要求当前两个视图的行SHA与finish完全一致，并核验commit及日志链。新R0的activation不引用自身artifacts、成本行、finish或commit。

所有大工件读取及核验在cost_stable前结束。最终成本含finish/commit、stage_closed/terminal候选及其小型元数据回读的保守seal-tail。**每次发布及fsync/回读返回后必须再次采样，而不是只在写前预测覆盖**：令`observed_elapsed=post_publish_perf_counter-stage_t0`，同时要求计时字段均为有限、非bool数值，耗时非负、`observed_elapsed < remaining_s`且`observed_elapsed <= committed_measured_wall_s`，身份/摘要也一致，才可通过当前写后门。stage_closed和terminal各自执行，不用前一文件的检查代替后一文件。提交前可按原有界轮数提高cost_floor；提交后任一写后门失败则停止并按§6.3记负债，不通过改写commit或重新发成功收据恢复执行。旧v1封口行为不追溯改写，外层返回证明只约束新增v2 invocation，不为旧两条v1终态行补造记录。

候选收据中的`status`只能表示候选的意图，不能自证其后续write/fsync或清理的时间已覆盖。写后通过事实先留在持锁执行上下文中；R0阶段即使已写出done候选，也须在上述实际检查通过后才允许P启动。terminal候选写后通过仍须等执行进程真正返回；最终输出/flush/清理期间的延迟归最后阶段，外层观察结果若显示已超已计费上界或阶段预算，仍为未闭合，不接受内层exit意图替代实际返回。

拟增加`w1_real_training_update_return_observation_v1`，固定存于同一原根日志下的`return_observation.json`。它由**外层调用方在wait/exec实际返回之后**记录，至少绑定invocation/授权/spec/claim、执行进程及启动身份、各commit和候选收据SHA、各阶段写后结果、外层实际退出码/信号、执行总墙钟、最后阶段结束时点、成本覆盖判定及返回来源摘要。来源必须是外层实际返回结果，不能由内层预写、从terminal推测，或把内层的声明当外层观察。数值结论audit_incomplete的预期退出码须与报告匹配，不能简单要求exit0，也不能把任意正常退出视为闭合成功。

执行窗口从§6.1的t0开始，至外层取得实际进程返回为止，包含最小启动器、worker、所有执行记录发布和清理；按相同计时基准核对总墙钟，各阶段实测区间连续且不重叠，原保守seal-tail按既有规则计费、不因交接退费。return_observation是与既有execution_observation同类的执行后观察存档，仅复制已产生的外层返回事实，并单列存档墙钟与`scope=post_execution_observation`；其自身持久化不被伪装成另一个已在执行内完成的成功自证。该时间边界必须写入spec及新授权：不得借观察存档启动项目核验、大工件哈希、重算、模型/资产读取或更新，也不产生额外执行窗口、attempt或可用预算。存档缺失、坏结构、来源不可证明或返回覆盖不成立时，所有公开读取器都将原claim视为未结清、cost_accounting_known=false。不能通过再添一份未被实际观察的“最终成功证明”消除该unknown。

### 6.3 提交后撤销与不完整状态

提交使用同文件系统临时文件、flush/fsync、不可覆盖发布与目录fsync。最终commit只能写一次；**提交后禁止调用finish_attempt或withdraw_attempt_claim改写已被commit绑定的行/报告/成本**。写入、fsync、回读或最终返回的实际耗时超出已计费上界/阶段预算，或发现身份漂移时，先使当前执行上下文失效、撤销全部下游放行凭据，再尝试追加不可覆盖revoke/failure；停止数值及验收工作，finally只结算已发生调用、清理和返回。即使write/fsync未抛异常，也必须执行这一检查；已经存在的done候选不能覆盖否决。

revoke/failure绑定commit、候选收据SHA、失败单元、写后实际采样和阶段t0，记录`uncovered_wall_s_lower_bound=max(0,observed_elapsed-committed_measured_wall_s)`及未对账原因；追加撤销/清理自身的时间也进入最终外层返回观测，不只记录写前的估计。按`(invocation_id,stage,execution_seq)`合并多个观测时，对同一覆盖区间取最大已知耗时/缺口，不能重复累加，也不能把整个阶段耗时再加到已计费floor上。提交后出现缺口即cost_accounting_known=false，禁止将剩余预算视为可用，后续只可另行审阅只读对账。

若revoke/failure无法持久化，内存否决仍禁止本次下游；执行必须以带明确闭合失败原因的结果返回。公开读取器要求有效外层return_observation，缺失或显示任何写后/实际返回覆盖失败均拒绝，即使terminal候选、done行已存在且锁已释放。不能用“已写成功文件且没有revoke”推断成功；来源丢失的外层返回也按unknown处理。后续独立读取若发现工件漂移，其必需字节/引用核验直接拒绝消费，不以能否追加revoke为前提。提交前的已知失败可以封成complete=false的终态成本记录，仍不可进入下游。

已知实际调用必须在finally结算；无法证明consumed、时间或三方视图一致时为unknown，更新总数按§11处理，不归零、不删除目录、不释放额度。登记前失败的claim/failure和提交后revoke均由固定启动日志传播给所有历史读取器；后续仅可另行审阅对账方案，不由当前入口自动补写成功或恢复数值工作。

旧前缀在登记前按事前SHA核验；登记后index/ledger已合法新增，改用旧行canonical不变与新事件约束复核。中断留下未结清claim/intent、running、publishing、缺commit/候选收据/有效返回观察、撤销或孤立目录时全部拒绝新准入；执行中只有通过§6.2写后门的持锁上下文可进入下游。已正确闭合的intent/启动日志不因文件仍存在而被误判为未结清。R0/P已用最后一次阶段attempt，失败后即使总秒数有余也不再自动执行。

## 7. R0、P/I 与 F 消费契约

新 R0 是正式 metadata 登记：重新验证父锚、两 checkpoint 文件身份、资产文本闭包、新实现/运行时、新校准/接受/准入及迁移历史；checkpoint payload、父数组值、模型/优化器、MuJoCo 编译仍为0。按同一 noise seed/protocol 生成新的8个 CPU draw unit，并绑定新 sealed noise 文件。旧 noise 仅作为历史证据保留，不在阶段间解码、重放模型或补算旧 P。

P 使用新 R0 的封口路径/SHA，在已登记且已预留的阶段内调用 `_decode_r0_noise`，解码前后执行同一 deadline+unit guard；随后 materialize、父数组/资产闭包复验、input admission、新 checkpoint strict load 与 fresh AdamW。确定性和扩散两模型均从原固定 step-4000 checkpoint 重做各两次连续诊断更新；不读取旧部分数组恢复参数或优化器状态。

input admission 同时绑定新 generation/spec/activation、新 R0 registration/noise、父数组/config 实测摘要，以及 raw/effective alpha 摘要和 `torch_cpu_float32_scaled_linear_v1`。真实/合成 alpha 严格 dtype/shape/逐位门保持。I 在自己的登记阶段解码同一新 R0 noise，核对新 P admission 后独立参考，不构造原生优化器；materialize、构造分项、场景编译/forward、参考 FK/Jacobian 和更新的守卫及完成后计数保持。

F worker spec额外绑定generation/spec/activation、新R0的commit、旧前缀snapshot SHA及legacy_head、固定启动日志/claim、执行序号与前驱头、新R0与P/I的已提交候选及其写后凭据、当前F reservation和调用基线。worker在from_spec、fixture、任何真实读取/构造/重放之前核对身份、runtime（含cpu_threads及Diffusers/AdamW源码）、全部历史、日志和候选。allow_current只允许同根、同代次、同seq、同invocation及控制进程归属的当前F处于in-flight状态，并由持锁父进程提供前序已通过写后门的事实；不能把尚未取得整个入口外层返回观察的运行内例外推广为公开known，也不能掩盖其他未结清日志或旧reservation。

F的provisional selected chain按第二代候选增量计算16次，cohort provisional则包含第一代2次。父进程对完整worker快照结算，在准备态完成盘上终检、最终选择及报告改写，再计费、冻结摘要、一次性提交；当前F未提交时使用内部准备态核验，不能调用要求其当前commit的公开消费者。完整快照缺失时不得把父进程计数器或已结算小计视为cohort总数，按§11输出unknown及已知下界。拒绝分支保留已证明的实际计数，超界撤回complete/selected；已提交后的异常通过§6.3撤销记录处理，不再改写提交绑定的行。

## 8. 实施前后必须覆盖的反例

全部新增运行测试使用明确标识的独立合成根，不复制或修改真实根来制造测试终态。真实 metadata 测试只读 JSON、文件字节/头部及资产文本；不能解码真实 checkpoint/父 NPZ，不能执行真实 forward/update。

| 主题 | 必须验证的行为 |
| --- | --- |
| 旧协议兼容 | 无迁移时仍拒绝新授权直接 resume；旧 schema/封口消费与同代次多可消费拒绝不退化 |
| 实际执行次序 | 合成 R0_001→失败 P_001→R0_002→P_002→I_001→F_001 可逐增量重推；按阶段排序、重复/跳过seq、前驱错误必须拒绝 |
| legacy_head承接 | 从旧v1副本和当前旧行独立重算固定头，seq=3只接受该头；空值、旧行SHA、snapshot文件SHA、错误payload或seq≥4再用legacy_head全部拒绝，且不新增seq=1/2事件 |
| 旧前缀保护 | 改 marker、授权、旧 canonical 行、旧 reservation 或 sealed 工件任一项即拒绝；新增 index/ledger 行不造成合法旧前缀误拒 |
| 零增量与类型 | 零调用 attempt 不产生排序歧义；bool计数、负值、非有限成本、重复JSON键不能通过 |
| 代次选择 | 两代各有 done R0 时第二代唯一选择R0_002；混用旧R0/旧P admission、未指定代次、同代次两个可消费均拒绝 |
| 新绑定闭包 | 改 spec/新授权/新 Gate B/runtime/原根路径或存在分叉即拒绝；存在activation但无R0提交不能启动P |
| 剩余额度 | 原67.10273322396097秒、2次更新与各stage attempt全部扣除；不得因切代次多领R0/P第三次或新3240秒 |
| 迁移计费 | 受控时钟在hash/快照/写盘/消费核验越界时R0 incomplete，P构造/读取/更新均不启动；登记前失败保留一次使用与成本负债 |
| 准备态与提交 | 准备态核验超过旧seal-tail但仍在预算内时，提高cost_floor后生成匹配新行SHA的唯一commit并允许下游；越预算时不生成可消费提交；公开stage_state拒绝无commit候选，内部核验不依赖当前commit |
| 提交后撤销 | 分别在stage_closed/terminal写入返回后注入覆盖失败，内存上下文先失效，再追加revoke/负债；原行/finish/commit SHA保持不变，不得重封口改写。revoke写入失败、候选仍存在时，当前下游仍不启动，公开读取器因缺有效外层返回观察或其失败结果而拒绝 |
| 成功收据慢写 | 尾部只预留1秒而stage_closed写入/fsync/回读耗时5秒，即使仍在阶段总预算内，也因实际超过已计费floor而停止，已知未覆盖缺口至少4秒，P的读取/构造/更新全0；另测floor足够但实际超过阶段remaining_s的独立否决 |
| terminal与返回慢写 | terminal候选写入/fsync耗时5秒而只预留1秒时，不发布有效闭合；清理/输出在内层最后采样后延迟越界时，外层实际wait结果也必须否决。候选done、无revoke、进程已退出/锁已释放均不能使其可消费 |
| 外层返回证明 | 只有各写后门及真实返回覆盖均成立、退出码与数值结论匹配时才公开闭合；覆盖匹配的audit_incomplete/exit3正常路径也能闭合。缺观察、错误来源、内层预写自证、错候选SHA或坏计时全部unknown；后置存档只能复制返回事实，不能夹带新的执行工作 |
| 启动负债发现 | 在R0/intent之前退出且旧index/ledger仍原字节时，仅留下原根claim/failure也必须使所有准入/预算unknown；换授权目录、换invocation、新建M5或释放锁均不能绕过；孤立日志目录也拒绝 |
| 日志成本去重 | 同一invocation/R0的claim和已closed成本行只计一次启动耗时；未绑定成本行的failure/revoke无已知余额，不得按0或只按旧账本发放余量 |
| 原子性和并发 | 在领取、index/ledger写入、reservation、intent、seal、activation和commit间逐点中断；半发布/重复进程/孤立目录均不进入下游、不重用授权 |
| 读取边界 | 缺迁移授权或未完成登记时，checkpoint/数组解码、tokenizer、模型、MuJoCo、forward/VJP/update spy 全为0 |
| F身份传递 | 错代次/前驱/候选/余额或runtime在worker预检即停；父结算、最终选择同样拒绝，实际局部计数不丢失 |
| 未知更新数 | 旧2+新P4+新I4已结算而F未返回完整快照时，总更新/完整cohort调用为null，已知下界10、selected为空、预算unknown；F部分执行但完整结算时才报告真实整数总数，不能填0/16/18默认值 |
| 最终计数与撤回 | 完整合成恢复 selected=16/cohort=18，24类矩阵逐项精确；异常与预算越界清空selected声明，complete=false |

完成 provider/audit 两文件及 v0.12 所列六个相关文件回归；数值契约相关的 geometry/review 定向回归按改动范围补充。新测试须验证以上独立行为，不以“模板字段写入后读回”替代。380 passed 是旧基线，不预先填写未来测试数量或宣称已通过。

## 9. 实施顺序与可审阅产物

1. **M0 保护与冻结输入**：在原根之外扩展不可改工件清单，覆盖原62文件、attempt_013和本计划；单独保存待修改实现的事前基线及预期改动清单。保存旧10文件清单、index/ledger原始副本、legacy_head及固定启动日志清单，核对事实与预算。此时不向原根写文件。
2. **M1 最小实现**：修改audit的legacy_head/序号重推/代次选择/准备态核验与一次性提交/撤销/授权v2，补齐最小启动器、逐次发布后的实际时间门、外层实际返回观察适配、固定日志扫描与成本去重、unknown总数及下界的报告/CLI输出，贯穿provider/spec/guard/接受闭包消费者及两份测试。拟新增只读--migration-admission和显式迁移执行参数；名称及互斥门随实现冻结，当前CLI尚不支持。普通--resume不自动升级身份，迁移执行必须显式绑定本计划和新材料。
3. **M2 反例与回归**：在隔离合成根完成 §8，尤其是旧失败前缀+新R0的次序、保留2次更新、两代R0共存、stage_closed/terminal实际慢写及清理延迟、撤销写入失败与缺外层返回证明、发布中断与F；同时验证完整正常路径可闭合。不得对真实根进行试跑验证迁移功能。
4. **M3 最终字节重冻结 Gate B**：将本计划及任何新增源码/测试加入完整实现清单，保留原26项，`EXECUTION_PLAN` 指向本计划，`PREREG_DRAFT` 保持原v0.9。确认后冻结，不改旧方案/旧工件；选择未占用的新校准/合成/准入目录，不预先假定某个attempt号可用。
5. **M4 新校准与接受证据**：实际重算校准、4×各门通过；容差/规则/零底/alpha协议/roster/RNG/micro矩阵与attempt_013一致。完整普通合成链仍为16次、264tensor/3864行、F新进程；另有明确标识的迁移合成链验证旧失败前缀加新链累计18次。P/I/F数值数组与rows对比attempt_013；差异必须逐项解释，不能因迁移放宽数值门。接受闭包同时绑定两个合成运行及其原始工件，含各自候选收据、return_observation及实际返回来源，消费端重算验证。
6. **M5 只读迁移准入与授权草案**：用最终实现生成新的real metadata admission、history snapshot/spec/migration admission和严格v2授权草案，核对不可改工件、实现变更及新冻结身份，独立重算legacy_head并强制扫描固定启动日志、完整余额、资源和全路径SHA。有任何未结清claim/failure/revoke即生成拒绝结果，不能输出passed或新的可执行授权草案。产物均位于原根之外，真实注册/新授权/数值更新0；对照§10提供可审阅结果。
7. **M6 复审与一次恢复**：在M0–M5全部完成且复审无阻断后，取得绑定最终字节的新明确授权，才允许一次原根 R0_002→P_002→I_001→F_001。不得把本次“制定方案”、原一次授权或M1实施授权解释为M6授权；不自动重试。

M1的身份参数、启动日志/写后门、外层返回证明和known/unknown状态必须贯穿_cohort_chain、cohort_history/cohort_attempts/prior_call_counts、budget_for、verify_history、登记/结算/reservation、公开stage_state与内部_verify_prepared_attempt、unit_admission、worker spec/候选预检、父结算/最终选择、报告与CLI，以及provider的acceptance/metadata/materialize消费端。v2在commit前后使用不同的计费/撤销协议，不能只修改CLI授权比较、消费者过滤或沿用v1提交后改行流程。

接受/准入 schema 及实现文件只要再改变，就重新完成受影响回归、校准/合成接受和metadata准入；不能用旧签名材料覆盖新字节。实现完成后另写验证记录和授权草案，本方案冻结后保持原字节。

## 10. 新授权应明确的十项与绑定

沿用原 `AUTHORIZATION_REQUIRED` 十个键的名称和严格布尔门，新版记录增加显式 `migration` 对象，不将第一代授权视为续期。复审材料逐项说明：

1. `new_question_cohort_and_parents`：同一cohort、原根与父锚；承认首次incomplete和原消耗，并同意此特定第二代迁移，不创建重置额度的新cohort。
2. `fresh_adamw_and_diagnostic_steps`：固定checkpoint、fresh AdamW，从头重做新P；最多新增16次、累计最多18次诊断更新，不续正式训练。
3. `two_continuous_updates_and_common_prestate`：两模型各两步、既有micro/noise及P/I共同前态契约保持。
4. `real_network_objective_parameter_backward_fk_clip_adamw`：在登记阶段内允许计划所列真实读取、网络/目标/FK/参数反向、裁剪、原生及参考更新。
5. `call_matrix_and_unsettled_unknown`：原24类余额加新矩阵逐项扣减；未结算、分叉或不明历史即停，不自动补写或重试。
6. `budgets_attempts_and_resources`：沿用累计3240秒上限和各阶段上限、按当前实账扣67.10273322396097秒；本次每阶段最多一次，R0/P是原第二次；同意迁移耗时归R0及原资源门。明确执行窗口覆盖到外层实际wait返回，含候选收据发布/清理；执行后仅复制返回事实的观察存档单列墙钟，不得用其开展额外工作或释放预算。
7. `synthetic_acceptance_fp32_zeros_propagation`：绑定新最终实现、新计划、新校准/验证/fp32、新普通及迁移合成接受、新真实及迁移准入与runtime。
8. `zero_fd_ddim_selection_checkpoint_contact`：parameter/state FD、DDIM、validation/selection、生产checkpoint写入和contact分支七项整数零声明保持。
9. `publishing_three_way_and_withdrawal`：旧终态行/工件保留；接受新代次/序号/一次提交、实际写后门、外层返回证明及超时撤回协议，知悉候选成功文件不能单独放行，残缺闭合会阻止下游。
10. `partial_coverage_not_ready_no_auto`：一次领取、首个准入/预算/执行阻断即停；不自动重试、扩额、换根、继续训练、选模或接入生产，gap仍按incomplete报告。

`migration`至少严格绑定原根realpath、原marker/原授权/snapshot/spec/migration admission SHA、legacy_head_sha256、固定launch_journal相对路径及事前清单SHA、目标generation、新计划/实现/运行时/新Gate B/新准入SHA、预定seq/attempt/前驱类型、原成本/24类调用/2次更新、剩余秒数/次数、本次新增上限/累计18上限及唯一invocation_id；另绑定写后检查协议、外层返回证明必需门、统一计时基准、执行窗口截止于实际wait返回及仅复制返回事实的后置观察范围。授权不含未来R0、commit或返回观察自身SHA，后续记录在授权后绑定它；授权schema验证拒绝缺字段、额外不明scope、bool数字、NaN、旧模板套用或余额关系不成立。事前日志清单不等于领取后的实时日志，执行上下文仅允许本次声明的追加，不能忽略其他负债。

本轮不填写十项true、不生成 `authorized=true` 记录。新的明确授权应在M5产物可完整审阅后取得。

## 11. 恢复验收与结论边界

只有新R0/P/I/F全部done/complete、封口闭包/迁移提交/全历史重推/各写后门/外层实际返回与计费核验齐全时，才允许公开第二代 `selected_attempts` 为 R0_002/P_002/I_001/F_001，`selected_chain_training_updates=16`、`training_updates_scope=cohort`、cohort实际更新为18，且selected/stage/cohort三套调用账与 §5一致。原P_001仍是不完整的历史attempt，旧部分数组不纳入新链。

执行中断、预算越界或证据/发布不完整时，selected_attempts/selected_chain_counts/selected_chain_training_updates均为null，complete_evidence=false；更新总数须按下表区分，不能统一使用父计数器或已结算小计冒充实际总数。若否决发生在commit之后，保留其已绑定报告原字节，公开汇总/CLI从commit、候选收据、revoke及外层返回观察计算上述有效状态，不把准备态报告的complete/selected当作独立成功声明；返回观察缺失时也不默认成功。

| 调用核算状态 | 必须报告的字段 |
| --- | --- |
| 所有已启动attempt的完整consumed与累计余额均已证明并结算 | call_accounting_known=true；training_updates与cohort_training_updates为原2次加本次实际增量的精确整数N；training_updates_known_lower_bound=N，cohort_call_counts为完整实际24键映射。即使阶段只部分执行或数值证据不完整，也不能改填名义18 |
| 任一已启动attempt缺完整worker快照、consumed或累计余额证明 | call_accounting_known=false；training_updates、cohort_training_updates、cohort_call_counts及选中链字段均为null；training_updates_known_lower_bound与cohort_call_counts_known_lower_bound仅累加已证明的结算记录，另列unsettled_attempts及原因，预算保持unknown |

例如旧2次、新P4次与新I4次已结算，F执行后未返回完整快照：实际总数为unknown/null，已知下界为10，不能显示实际10、0或预计18。即使F可能已执行更新，也不推测增量。CLI明确显示总数未知及已结算下界，JSON和最终报告使用同一字段；禁止None转0、用BOUNDARIES中的名义16或授权上限18回填。若F中断但完整结算了6次，则精确总数为16，选中链仍为空，不能因数字恰等于名义16宣称链完成。

调用核算与时间核算分开：有未对账启动/提交尾部成本负债时cost_accounting_known=false，秒数余额unknown，不能仅因调用已知就再次执行。若调用证明也缺失，按上表报告null及下界；不将未知变成0。证据完整时的数值failure与gap按原预注册分类优先级判定；存在精度/相消必测gap仍为audit_incomplete。582条是既有合成基线，不预判真实gap数量。

最终验证记录应列出新授权/绑定/执行事件、完整旧前缀保全、新选中链与cohort账、预算/资源、F pid及退出状态、失败/gap分类和未覆盖项。`objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false` 及旧 U0/UR/原F/r6/旋转融合结论保持。选模、DDIM、端到端质量与生产接入不因本次诊断恢复变为已审计。

## 12. 修订1的审阅问题对照

| 审阅问题 | 方案修复与未来反例 |
| --- | --- |
| P1 消费核验依赖commit且核验后又重封口 | §6.2区分内部准备态核验与公开消费；核验/选择/计费稳定后再写唯一commit，提交后仅可追加撤销/负债。§8加入超旧tail仍可成功提交及预算越界拒绝两类反例 |
| P1 登记前外部负债不可发现 | §6.1固定原根launch_journal，所有历史/准入/预算读取器扫描，未结清claim即unknown；§8要求在旧账本未变且更换授权目录后仍拒绝 |
| P2 旧v1无前驱提交锚点 | §4.1给出完整legacy_head_payload、canonical算法及本机固定头，seq=3专用分支承接，seq≥4使用前一commit；§8拒绝空/错误种类/错误摘要 |
| P2 未结算更新小计冒充cohort总数 | §11规定null总数、已知下界与未结算列表，区分调用/时间known状态；§8覆盖F无完整快照时null总数/下界10及已完整结算部分执行的精确整数 |

修订1仅修改方案与文档入口，上表反例是M2实施要求；其尾部发布次序已由下述修订2补充，不能沿用“成功文件写前检查即闭合”的解释。当前Gate B仍为v0.12/attempt_013。

## 13. 修订2：成功收据的写后耗时与失败闭合

本轮修复复审的1项P1：§6.2不再把stage_closed/terminal的写前预留当实际证明；写入、fsync和回读返回后逐一核对实际耗时、已计费上界与阶段预算，最后由外层实际返回事实覆盖清理/输出。stage_closed通过写后门才可在持锁上下文放行，公开闭合还要求有效return_observation；终态候选、锁释放或缺revoke都不能单独证明成功。

§6.3规定先在内存否决并阻断下游，再追加撤销/未覆盖成本下界，不改写已提交行/finish/commit；撤销或返回证据丢失仍为成本unknown。§8加入1秒尾部/5秒慢写、阶段越界、最后清理延迟、撤销写入失败、外层来源及正常exit3闭合反例，M1/M2、F传递和新授权同步。执行窗口与后置观察存档的区别必须在新spec/授权中明示，不由实现临时扩大免计费范围。

本轮仅修订方案与文档入口；上述接口、观察记录和反例均为未来要求，未创建迁移/授权/注册/返回观察记录，未实施迁移、运行测试或真实恢复。当前实现证据仍为v0.12/attempt_013，原2次更新/67.10273322396097秒及全部预算/调用上限保持。
