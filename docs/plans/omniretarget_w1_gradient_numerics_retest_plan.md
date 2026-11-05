# W1 数值修复后真实复验与证据重冻结方案

日期：2026-11-05。版本：v0.1，待审阅。状态：**plan_only / retest_not_implemented / gate_b_m5_not_refrozen / real_retest_not_authorized**。

本方案验证 N1–N3 修复在真实参数 VJP、连续两步更新及独立重放中的效果。拟在既有 cohort 原根追加 `generation_003`，完整执行一次 R0_003→P_003→I_002→F_002，最多新增 16 次诊断更新。原 18 次更新、567.845409012 秒成本、329 项 failure / 629 项 gap 和已耗尽的 M6 授权均保留。本轮只编写方案与更新文档入口，不实施新协议、不生成授权或真实 attempt，也不运行合成冻结或真实复验。

## 1. 已有证据与本次问题

依据：[M6 执行报告](/home/c112/codes/bill/ireset/docs/verification/omniretarget_w1_real_training_update_binding_migration_report.md)、[失败只读归因](/home/c112/codes/bill/ireset/docs/verification/omniretarget_w1_real_training_update_failure_attribution.md)、[最小修复方案](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_gradient_numerics_repair_plan.md)和[修复验证记录 §7](/home/c112/codes/bill/ireset/docs/verification/omniretarget_w1_gradient_numerics_repair_verification.md)。

N1 已替换 trace-only quaternion 路径；N2 已以 `bias * 0` 保留线性 key bias 的参数与存在零梯度；N3 已修正失败 micro 行状态。保存输入 `failure_attribution_local/attempt_006` 的 18 项检查全过，完整输出 cotangent 为 0 超差、max ratio `0.0154715335`，但该结果没有验证参数 VJP、micro 2 的 `root_out.weight` 或全部 gap。

本次完整链回答以下固定问题，不承诺 gap 清零：

| 问题 | 必须取得的证据 |
| --- | --- |
| 近 π 输出偏差是否仍向参数梯度传播 | 完整输出 cotangent、所有 micro 参数梯度、累积及裁剪行；单列 diffusion u2/m0 原失败坐标和完整输出门 |
| key bias 结构零路径是否在真实网络中成立 | 两模型全部线性 self/cross key bias 的梯度 presence、shape/dtype、精确零值，以及两步 AdamW 的 m/v/step/参数行为 |
| micro 2 的独立问题是否消失 | 单列 diffusion u2/m2 `model.net.root_out.weight` 的 P/I 梯度与原门比；不得仅由 micro 0 修复推断 |
| 更新比较还缺少哪些证据 | 对 381 precision、223 sensitivity、25 bound gap 逐键比较消失、保留及新增项，保留可辨识性和区间否决原因 |
| 新链能否独立重放 | F 新进程重放本次 P/I，核对 RNG、行多重集、数组、全部调用及实际返回证明 |

真实数值工作全部位于下述一次 P/I/F 内，不另加保存输入参数重放、分项 backward、有限差分或额外 optimizer step。旧 M6 数值结果只作历史对照；新旧预测可能因 N1/N2 改变，不要求新数组与旧数组逐字节相等。

## 2. 只读起点与保护清单

固定原根为 [20261102_2704709_w1_real_training_update_audit](/home/c112/codes/bill/ireset/derived/experiments/omniretarget_first/20261102_2704709_w1_real_training_update_audit)，固定 history root 仍为 `derived/experiments/omniretarget_first`。cohort/execution 身份仍为 `w1_real_training_update_audit_v1`；改目录、代次或实现身份不重置累计账。

| 起点材料 | 2026-11-05 检查值 |
| --- | --- |
| 当前实现 | 30 文件，`00581d19b5e4e1f5b75b52cbc3c7dfdda8fce314ddd231828352eaa41e4d1245` |
| 原根文件数 / bytes | 83 / 7,835,408,541 |
| `cohort.json` SHA | `79f42f5fc2bb205a4f8327b7897644e4759ff60bf189f4dd6054cb980e9f3f6b` |
| 当前 `attempts_index.json` SHA | `07195f557ab5cb746c889167192ecdf0681479ea86acd6e0c3ec92f8cefc210d` |
| 当前 `costs.json` SHA | `4bbeabcec52368fc6c02aa1134616528346fc5bc7628c7fa96b67df1e1d1e7b3` |
| seq 6 / F_001 commit SHA | `c650c24bf3bb218055b3ed3bc54e33046c69896b642fd3de442e199ccbf2727d` |
| M6 实际 return observation SHA | `32599cf7f84112f031ff648866e839568d3b3605f97b0fb52bc70f099d8aff53` |
| M6 正式 v2 授权 SHA | `673d750f84029d9fe5c0df80a4269bf7abd03f736c45ca2fb739038126ecd76b` |
| M6 final SHA | `ede2254f9212a34a50f1478e85bc98dc199db56dd719efff208d2886ad0d9799` |
| 局部 attempt_006 result SHA | `0357b118198db8694451b9d516e8bf52fc8f0fca1b9452364ebef49cd73d35a9` |

当前校准 `calibration/attempt_019` 的 contract 为 `ca6aad96…`、evidence 为 `962b625c…`、fp32 canonical 为 `94290e00…`；按当前 30 文件身份重生成的 verification 为 `fc4e2f3438d6642f98566b41940ec07b239292eb7a09b373d40bcacf24686200`，旧 `84df5132…` 记录已另存。这些是起点证据，不是未来复验协议最终字节的 Gate B。

准备阶段在原根之外建立新的保护清单：覆盖原根全部 83 个文件、旧授权/Gate B/M5 及其既有保护清单、父锚、两份 checkpoint/config、资产与本轮修复证据。每项记录规范路径、SHA-256 和 bytes；只复制小型 JSON 源视图，不复制或硬链接大型 NPZ/checkpoint 作为新的真实根。

旧 `cohort.json`、bindings/generation_002、seq 3–6 事件、已存在的启动日志/返回观察、全部旧 attempt 目录与旧外部材料保持原字节。将来执行时，只有 `attempts_index.json` / `costs.json` 允许追加新行并结算本次当前行，六条旧行的 canonical 内容须不变。所有其他新写入只能落在本方案列明的新路径；不删除失败历史以满足资源门。

## 3. 路径选择与必须新增的协议

采用**同一原根、第三代绑定、显式预算修订**。旧 v1→v2 迁移入口只适用于 seq 1/2 的旧前缀；当前原根已有完整 generation_002，不能直接再次执行旧 `--real --migration`，也不能用普通 `--resume` 更换实现身份。

当前代码只实现 generation_001/002、seq 3–6、v2 一次迁移与旧额度。实施前须增加专用复验上下文及 v3 授权/登记/报告分派；拟使用显式 `--real --retest`，该参数**当前不存在**，本文件不提供可直接运行的真实命令。新实现集中在既有 audit/provider/CLI 与测试，不建设通用工作流框架，不再次修改 N1/N2 数值算法。

| execution_seq | generation | stage / attempt | 前驱 |
| ---: | --- | --- | --- |
| 1–6 | 原 generation_001/002 | 全部历史终态 | 按各自冻结协议只读核验 |
| 7 | generation_003 | R0 / attempt_003 | seq 6 commit 的完整文件 SHA |
| 8 | generation_003 | P / attempt_003 | seq 7 commit 的完整文件 SHA |
| 9 | generation_003 | I / attempt_002 | seq 8 commit 的完整文件 SHA |
| 10 | generation_003 | F / attempt_002 | seq 9 commit 的完整文件 SHA |

新 `bindings/generation_003`、`execution_events/000007` 至 `000010`、一次新 invocation 的 `launch_journal` 和四个 attempt 目录只在正式新授权后创建。目标路径必须全部不存在；已有孤立目录、tmp、claim/failure/revoke 或未知事件导致拒绝，不清理后自动再试。

授权、spec、每条新行和 reservation 必须绑定同一 `budget_revision_sha256`、generation、invocation 和 stage plan。旧 marker 的 32 次更新硬上限及旧 stage budget 不改写；v3 仅在下述明确新授权作用域内按修订额度验证 seq 7–10。旧版本对新协议应明确拒绝，不能把 34 次更新解释成旧 32 次上限内合法。

同一复验上下文须贯穿 `_cohort_chain`、cohort history/attempt/call 汇总、`budget_for`、登记/结算/reservation、`verify_history`、公开 `stage_state` 与内部准备态核验、`unit_admission`、provider metadata/materialize、F spec/worker 预检、父结算及最终选择、报告和 CLI。全历史扫描始终覆盖所有代次及固定启动日志；只在 CLI 放行授权或在选链时过滤旧行不构成实现完成。

### 3.1 承接完整旧头

新 snapshot 保存六条旧 index/ledger 行、reservation、seq 3–6 commit 与 stage_closed/terminal/write_after/实际 return observation 的闭包、完整旧启动日志清单，以及成本、调用与 attempt 总账。必须验证已结清后才能产生 passed 的复验准入。

另定义 `closed_history_head`，绑定原 marker SHA、当前 index/ledger 原始副本 SHA、last_seq=6、seq 6 commit SHA、旧 return observation SHA、六行 canonical SHA 与累计账。其摘要仅用于 snapshot/spec/授权身份绑定；**seq 7 的 `previous_head_sha256` 仍直接指向 seq 6 commit**，不得用该 snapshot/head 摘要或旧 seq 2 的 legacy_head 代替。

历史材料按其原冻结 schema、旧绑定和封口证据核验，不把当前源码/runtime 与历史值不同视为可改签理由；旧校准缺少新四元数字段也不允许成为新准入材料。历史真实性与当前数值路径的准入分别核验，任一无法证明均停止。

## 4. 固定预算修订与一次性额度

本节全部是**待新授权批准的提案**，本轮不使任何余额可执行。现有 M6 授权已用完，账面剩余秒数不恢复其一次使用权。

总累计墙钟上限仍为 **3240 秒**，包含旧登记成本 **567.845409012 秒**；最多可用于本轮的新登记成本为 **2672.154590988 秒**（两者均为展示值）。为覆盖原根 83 文件保护复验及启动，提出将 R0 累计上限从 240 调为 600 秒，同时将 P 从 900 调为 540 秒；I/F 保持 900/1200 秒。该重新分配必须由 v3 授权逐项同意，运行中不再互借或调整。

| stage | 旧累计秒（展示值） | 原上限 | 拟修订累计上限 | 本轮启动前余额（展示值） | 工作前保留秒 | 原已用 attempt | 拟累计最多 / 本轮新增最多 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R0 | 183.042352854 | 240 | 600 | 416.957647146 | 30 | 2 | 3 / 1 |
| P | 36.527501609 | 900 | 540 | 503.472498391 | 120 | 2 | 3 / 1 |
| I | 102.326411103 | 900 | 900 | 797.673588897 | 120 | 1 | 2 / 1 |
| F | 245.949143446 | 1200 | 1200 | 954.050856554 | 180 | 1 | 2 / 1 |
| 合计 | 567.845409012 | 3240 | 3240 | 2672.154590988 | 按阶段保留 | 6 | 一次入口、四阶段各一次 |

表中九位小数仅作展示；准入、spec 与授权从已结算 `measured_wall_s` JSON 原数值按 execution_seq 顺序使用既有求和方式重算并绑定余额，不使用四舍五入后的 `wall_s`，不以展示值或十进制文本重新求和替代冻结结果。启动前若原账发生变化，须生成拒绝结果并返回审阅，不据新的余量自动扩大或重新签发草案。

本轮新增更新上限 **16**（P=4、I=4、F=8），cohort 授权上限须显式从已耗尽的 18 扩至 **34**，通用更新硬上限从旧 32 修订为本轮 34。原生 AdamW 新增 8、显式参考更新新增 8；F 重放已计入 16，不另有“免费验证更新”。旧授权的 18 和旧 marker 的 32 均保留为历史规则，不允许直接全局替换常量使旧记录获得新额度。

### 4.1 24 类调用矩阵

R0/P/I/F 各单次 nominal 沿用原协议；本轮累计硬门为“旧实账 + 本轮单链 nominal”，没有额外 retry 余量。以下矩阵与额度修订一起进入新授权，不能只扩更新次数而遗漏其他已接近旧 cap 的调用。

| kind | 旧累计 | 本轮最多新增 | 拟本轮累计硬上限 |
| --- | ---: | ---: | ---: |
| checkpoint_loads | 10 | 8 | 18 |
| model_constructions | 10 | 8 | 18 |
| conditioner_constructions | 10 | 8 | 18 |
| native_optimizer_constructions | 6 | 4 | 10 |
| conditioner_forwards | 72 | 64 | 136 |
| backbone_forwards | 72 | 64 | 136 |
| native_add_noise_batches | 32 | 32 | 64 |
| cpu_draw_units | 56 | 48 | 104 |
| production_loss_assembler | 40 | 32 | 72 |
| production_float32_fk_batch | 80 | 64 | 144 |
| production_parameter_backward | 40 | 32 | 72 |
| independent_output_vjp | 32 | 32 | 64 |
| independent_total_vjp | 32 | 32 | 64 |
| native_global_clip | 10 | 8 | 18 |
| independent_clip_formula | 8 | 8 | 16 |
| native_adamw_steps | 10 | 8 | 18 |
| explicit_reference_updates | 8 | 8 | 16 |
| diagnostic_parameter_update_events | 18 | 16 | 34 |
| anchor_state_restores | 4 | 4 | 8 |
| optimizer_formula_checks | 8 | 8 | 16 |
| reference_mj_forward | 15840 | 15840 | 31680 |
| reference_mj_jac_site | 115200 | 115200 | 230400 |
| scene_setup_mj_forward | 3 | 2 | 5 |
| micro_execution_units | 72 | 64 | 136 |

表中上限不作为实际消耗默认值。只计实际完成的调用；完整快照缺失时总数为 null，已证明下界单列。例如旧 18 + 新 P4 + 新 I4 已结算而 F 未结清时，总更新未知、下界 26、选中链为空，不能填实际 26 或名义 34。

## 5. 资源、启动、计费与中断

资源门沿用 GPU peak≤22 GiB、host peak≤32 GiB、同 cohort 工件≤16 GiB、启动空闲盘≥20 GiB、单 GPU/单数值 worker。原根 7,835,408,541 bytes 先计入工件预算；新链、focus sidecar 及控制记录也全部计入。未来准入须按最终数组 shape/文件格式重算上界；若总量不能落在 16 GiB 内，本方案拒绝执行，不删除历史或临时移走同 cohort 工件来制造余额。

本轮 focus sidecar 总量≤32 MiB，新绑定/事件/观察等控制文件总量≤32 MiB；两者均计入上述 16 GiB。外部准备材料也单列存储成本与生成墙钟，不借准备工作执行真实数值探测。M5 仅读 JSON、文件字节/头部、资产文本及依赖源码，允许流式摘要，checkpoint payload/父数组解码/模型/编译/forward/VJP/update 均为 0。

实际复验沿用最小标准库启动器：在 uv、项目及数值库导入、完整授权校验和大文件读取之前，在固定原根写新 claim 并持有独占锁。claim 绑定新授权/spec/closed head、目标 generation/seq=7 和唯一 invocation。当前 claim 的运行内例外仅允许已核验的当前 invocation；旧 generation_002 的已闭合日志属于历史，其他未结清记录不获豁免。

启动/import、旧闭包复核、绑定发布及 R0 封口计入本轮 R0；连续 stage_cursor 覆盖各阶段交接。P/I 的噪声解码、源摘要、构造及数值工作在各自登记后执行并计费，所有耗时保留原 deadline + unit admission 守卫和成功后计数。三套 selected/stage/cohort 调用账保持分离。

activation 在本轮 R0 封口前发布，绑定其 reservation；seq 7 commit 再绑定 activation。不能沿用“仅 seq=3 才是代次首 R0”的硬编码。F 候选/terminal 门以本次 stage plan 的 seq=10 为权威，不能沿用 seq=6。finish/commit 一次性发布，阶段消费核验、终检重封口和写后实际时间门继续计费；F 重封口始终保留至少 3 秒退出尾部，仍以实际返回是否被登记成本覆盖为最终门。

外层须真实 spawn/wait，再记录退出码、pid/ppid、信号、连续阶段时间和实际返回观察。已提交后的异常追加 revoke 与未覆盖成本下界，不改旧 commit/行。缺完整快照、缺返回证明、写后越界或未知负债时保留不完整/unknown 状态，停止下游；不自动 retry/resume，不启动第二个 invocation 补证明。后置观察只存档实际返回事实，存档墙钟单列，不借此进行新的项目核验或数值工作。

## 6. 真实数值范围与附加保存证据

固定两模型、原两份 step-4000 checkpoint/config、W0–W3 四窗、micro_batch=4、每步四 micro、连续 u1/u2、原 `MICRO_SCHEDULE` 与 `NOISE_SEED=20261008`。模型配置、loss 权重、raw/effective alpha、float32 生产 / 独立参考精度、clip、AdamW 公式、容差、零底、4× 校准规则及传播界沿用冻结契约，不因结果放宽。

新 P 从固定 checkpoint 和 fresh AdamW 开始，I 取匹配共同前态；不从旧 M6 更新后的模型/优化器继续训练。u2 必须消费新 u1 实际产生的 m/v/step，N2 不能清空旧于该步的动量、删参数、改 None 或跳过 AdamW。两步内存在零梯度仍按原公式执行 weight decay 与状态衰减。

为使残留 micro 失败可归因，在 I 已计算的 `grad_map` 上保存额外 focus 数组：两模型、两步、四 micro 的 `model.net.root_out.weight` 与四层 self/cross attention 的全部线性 `k.bias`。P 对应值取既有 `contrib` 数组；F 重放的 I 也保存同一固定名册。只做 detach/copy，不新增 autograd、FK 或更新；拷贝与写入计入所在 I/F 阶段。

focus 的参数键、dtype、shape、presence、总 bytes 及文件清单在最终 spec 冻结，缺失/漂移即证据不完整。新增 `i_focus_gradients.npz` 等 sidecar 必须进入 required artifacts、sealed SHA/bytes、接受报告和 F 消费闭包，不能仅在报告中写“已保存”。沿用 264 tensor / 3864 主行名册；focus 作为补充证据，不删主行、不把补充行计作额外通过覆盖。最终实现若需要改变主名册，须先修订本方案再冻结。

不重放旧 trace-only 网络作为另一个真实对照，不加参数/状态 FD，不运行 DDIM、验证集、选模、contact 或生产 checkpoint 写入。七项整数零声明逐项保持 0。

## 7. 合成反例、回归与实现身份

先实施复验身份/额度协议和 focus 保存，再在明确标识的隔离合成根测试。禁止复制真实根后改写记录制造反例；只读真实 metadata 可用于核对起点。

| 必测类别 | 验收要求 |
| --- | --- |
| 完整既有前缀 | 合成 v1 失败前缀 + 完整 v2 闭合链 + v3 新链，seq 1–10 重推，旧 canonical 行和工件不变 |
| 旧头与授权边界 | seq 7 指向 seq 6 commit；错旧返回证明/closed head、旧 v2 授权、普通 resume、缺额度修订均拒绝 |
| 额度修订 | 原 cap 下 34 更新、第三次 R0/P、scene=5、loads=18 等均拒绝；明确 v3 授权下仅放行固定本次增量，额外调用/attempt/入口仍拒绝 |
| 历史负债 | v2 已结清日志正常消费；其他未结清 claim、孤立目录、tmp、revoke、分叉和未知 schema 使新准入 unknown |
| 计费与实际返回 | 注入启动/activation/交接/封口/写后/worker 清理延迟，成本覆盖不足即停止；F terminal 针对 seq 10，不误用旧 seq 6 |
| 选链与部分结算 | generation_003 唯一选中；跨代混链/同代多可消费拒绝；F 无完整快照时总数 null、下界 26 |
| 数值与 focus | 近 π 值/梯度、key bias 存在零梯度及非零动量、成功/失败 micro 状态；focus 缺文件/键/dtype/shape/摘要漂移被拒 |
| 版本与冻结 | 四元数库版本/源码缺失或漂移拒绝；新实现/新计划/新 Gate B 绑定不一致拒绝；旧只读协议兼容性保留 |

回归至少覆盖聚焦 gradient numerics、audit/provider/migration、拟新增 retest 测试，以及既有 FK/objective/training-objective 相关集合；根据实际代码变更补充 geometry/generation 等受影响检查。不为纯文档变更重跑真实数值，也不把相关集合通过冒充全项目套件通过。

最终 `IMPLEMENTATION_FILES` 至少纳入本方案与新增复验测试，所有实际修改/导入的实现与依赖都要闭合。当前 30 文件 `00581d19…` 仅作起点，未来最终文件数/聚合 SHA 重新实测；文档本身不填写包含自身的聚合摘要。所有方案、测试、fixture 和实现修改先完成，再冻结文件字节。测试夹具通过显式输入选择新校准，避免生成校准后再改测试源码切换 attempt。

## 8. Gate B 与新一轮 M5 的重冻结顺序

本节是后续实施顺序，本轮不运行。准备材料全部位于原根之外；每次使用不存在的新 attempt 目录，失败材料保留。

拟以 `derived/audit/w1_real_training_update_audit/gradient_numerics_retest/` 为准备材料基目录；各次准备使用独立 `attempt_*`，不充当新的真实根或启动日志来源。M5 至少保存 `history_snapshot.json`、`protected_manifest.json`、`budget_revision.json`、`retest_spec.json`、`retest_admission.json`、`authorization_draft.json` 和 `preflight_verification.json`，以及原始 index/ledger 等小型 source copies。校准与三支合成接受可使用现有各自的新 attempt 目录，M5 用完整路径/SHA/bytes 引用；实际目录编号在生成时选择不存在的值，并写入清单，不预占本轮真实 attempt。

1. **T0 起点快照与协议实施**：只读保存 §2 起点及保护清单；实现 §3–7 的 v3 复验、预算修订、历史兼容与 focus 闭包，完成反例和相关回归。此时仍没有真实 claim/binding/attempt。
2. **T1 最终实现冻结**：定稿本方案、实现清单、测试/fixture，记录逐文件 SHA、files-only 聚合身份、git 参考身份和完整 runtime。Git 提交本身不改变 files-only 身份；清单中文件字节变更会使后续冻结材料失效。
3. **T2 合成校准**：在最终树上新建 calibration attempt，生成 tolerance contract、fp32 contract 与 calibration verification，独立 recompute 与 4× 门通过，容差/规则/零底不变。保留 attempt_019 的当前/历史 verification 原字节，不再回填。
4. **T3 合成接受**：运行普通链、旧 v1→v2 迁移链和新增“已完成 v2→v3 复验”链。原 Gate B v2 继续绑定普通/旧迁移两侧；另产生固定 schema 的 `retest_acceptance`，绑定第三支、两侧 Gate B、最终实现/runtime/校准及闭合收据。新授权/准入必须同时消费 Gate B v2 与该补充接受，不能只看任一 passed。三支更新分别为 16、18、34（第三支含已明示的合成历史），均不增加真实额度。
5. **T4 只读 M5**：以当前完整六行历史生成 real metadata admission、新 full-history snapshot、budget revision、generation_003 spec、retest admission 和 v3 授权草案，独立验证保护闭包、旧提交/实际返回、累计账、资源及最终实现/运行时/校准/三支合成接受。任何未知负债只输出拒绝，不生成可执行材料。
6. **T5 审阅与新授权**：提交完整材料供用户审阅；草案保持 `authorized=false/template_only=true`、十项 checklist 为 null。取得 §9 的新明确授权后才可写正式授权及真实执行记录。
7. **T6 一次真实复验**：按固定映射执行新链、真实 wait/观察和最终封口。结论与剩余问题按 §10 出具，失败不自动重试。

spec 的摘要方向为“snapshot/预算修订/实现/runtime/校准/Gate B v2/retest_acceptance/real metadata admission→spec→retest admission→新授权→claim→R0/activation→commit→候选收据→实际返回观察”。这里 real metadata admission 不依赖新 spec；retest admission 在 spec 之后发布，不能被 spec 反向引用。spec 不包含未来授权或 commit SHA；registration/activation 不引用包含自身的 commit；旧 seq 6 commit/return 是可直接绑定的历史锚。M5 独立核验须检查缺文件、路径解析、逐文件 SHA/bytes 与消费结果，不只复制 builder 的 passed。

新增 `retest_acceptance`、v3 授权及上下文 schema 的确切字段须在 T0 实现审阅时定稿并写入测试，然后再进入 T1；当前 CLI、schema 或普通 M5 生成器不能被假定已支持本方案。

## 9. 新授权须明确的十项

沿用既有十个 checklist 键名，v3 草案为每项提供以下具体解释；授权记录另绑定本方案最终 SHA，不以“继续”“同意旧草案”或此前 M6 授权推定本轮授权。

| checklist 键 | 本轮必须明确同意的范围 |
| --- | --- |
| `new_question_cohort_and_parents` | 同一原根/父锚，承认完整 M6 与六行历史；追加 generation_003 / seq 7–10，不重置 cohort |
| `fresh_adamw_and_diagnostic_steps` | 固定 checkpoint 起点、本轮仅新增 16 次诊断更新，累计上限明确为 34，旧 18/32 规则作为历史保留 |
| `two_continuous_updates_and_common_prestate` | 固定两步/共同前态；u2 保留 u1 的真实 AdamW 状态，N2 不删参数或清空动量 |
| `real_network_objective_parameter_backward_fk_clip_adamw` | 本轮 P/I/F 的真实网络、目标、参数 VJP、FK、clip、更新和固定 focus 保存范围 |
| `call_matrix_and_unsettled_unknown` | §4.1 全 24 键及各阶段 nominal；完整结算前不填名义总数，F 未结清时 null/下界 26 |
| `budgets_attempts_and_resources` | 累计仍为 3240 秒；R0/P 重分配为 600/540 秒，I/F 900/1200 秒；累计 attempt 3/3/2/2，本轮各一次及资源门 |
| `synthetic_acceptance_fp32_zeros_propagation` | 最终校准/运行时/四元数源码、新 Gate B v2 与 retest_acceptance；原零规则、可辨识性与传播界 |
| `zero_fd_ddim_selection_checkpoint_contact` | 七项整数零声明，额外真实探测/分项 VJP/FD/选模/写 checkpoint 不在授权内 |
| `publishing_three_way_and_withdrawal` | failure/gap/证据完整性分别发布；提交后 revoke、实际返回覆盖、unknown 与不完整结果如实保留 |
| `partial_coverage_not_ready_no_auto` | partial/null/false/not_ready/不接入保持，一次入口、不自动重试，不自动生产接入 |

v3 授权还须逐项绑定原根 realpath/marker、旧正式授权及 seq 6 commit/return、full-history snapshot、六行与启动日志事前清单、新预算修订和全部增量/累计 cap、新 spec/admission、最终实现/本方案/runtime、校准/evidence/fp32/verification、Gate B v2/retest_acceptance、新 generation/stage plan/唯一 invocation、写后协议和实际返回边界。只允许本次声明的新增记录，缺字段、超范围、bool 冒充数字及余额关系不成立均拒绝。

## 10. 复验验收与交付

协议完整性与数值结果分别裁决。四阶段成功封口、同代唯一选链、全部快照结算、写后门与外层实际返回闭合时，才发布新选链 R0_003/P_003/I_002/F_002；完整成功执行时 selected 更新为 16、cohort 实际更新为 34。任何部分执行按实际完整结算数报告，unknown 按 §4.1 报告，不能以“链已运行”替代可消费证明。

数值报告沿用原分类器和全部冻结门，必须同时展示 complete_evidence、covered_contract_passed、failure 与 gap：

- 逐键列出原 329 failure 的 resolved/persisted 及新 failure，附原 156 项 key bias、173 项 diffusion u2 分组；micro 2 `root_out.weight` 单列。
- 原 629 gap 与新 gap 按参数/步/类型对齐，不把 precision、sensitivity 或 bound gap 合并成通过；差异需能从本次数组、写回界、传播区间和主行复算。
- 机制修复目标是完整输出、真实 key bias 结构零与相关梯度比较过原门；仅比较 loss、原五坐标、总 failure 数下降或 F 重放一致均不足以宣称数值通过。
- 若仍有 direct failure，原样保留失败和分类结论；若只有不可辨识/敏感性 gap，仍不能宣称 covered contract 通过。阶段 done/complete 和接受材料 passed 仅表示相应证据协议成立。

交付物为新实现验证记录、最终 Gate B/M5 清单、正式新授权（仅在用户明确授权后）、一次执行报告、三套调用/更新账与累计成本、实际返回证明、focus sidecar 清单、旧/新失败及 gap 对照和保护清单复核。执行后的静态对照必须在已登记阶段或明确单列的只读存档范围内完成，不追加未计费真实重算。

无论本次参数链结果如何，`objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false` 与旧质量结论保持。选模、DDIM、loader、端到端动作质量及生产接入均不在本次复验覆盖内。
