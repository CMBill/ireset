# W1 固定单窗学习率日程诊断：验证记录

日期：2026-11-18。对应[方案 v0.1＋§7–§8](../plans/omniretarget_w1_d1_lr_schedule_diagnostic_plan.md)。

**当前交付：I′ 合成验收与复审（§6）、一次 R′（§7）、一次获授权的真实 D1′→F′ 与只读验收（§8）均已完成。真实 B 在第 1000 步四项平均指标全部达标，最终为 `fit_pass`／`single_window_pass`。** §1–§7 保留各阶段的历史快照；合成夹具不是学习率效果证据。

2026-11-19 工程复阅发现并修复2 P1＋2 P2，见[单独修复记录](omniretarget_w1_d1_lr_review_hardening.md)。真实结果保持；修复后代码身份已经变化，旧绑定材料不授权新运行，原核验命令须在其冻结源码下执行。

## 1. 范围、身份与不变项

- 本轮起点 HEAD：`d533565`，分支 `codex/omniretarget-first-training-plan`。验收发生于未提交实现树，**不能仅用该 HEAD 指代实现字节**。
- 实现闭包为 **31 文件**，规范摘要：`f427905d01c02a45b1c1c7d6315a05eeff60ffc5ec0512a8ae743eb5e59f4f5d`。
- 方案文件 SHA-256：`6d063812101bb9ad40811ceda71c7dd6161d8188785ab01bf87cdbebf07347aa`。§1–§6 实验规则保留，新增 §7 工程记录及首页状态。
- 完整逐文件摘要保存在合成准入的 `package.code_identity` 与 `/tmp/opencode/w1_lr_implementation_20261118/acceptance_readonly_verification.json`；后者为只读复核记录，不是额外真实运行。
- 使用项目 `.venv`、`uv run --locked`、CPython 3.12.13；未新增依赖或修改 `pyproject.toml`/`uv.lock`。

与真实基线 A 已登记的旧实现清单逐文件重哈希，变化恰为：

1. `src/ireset/eval/w1_smallset_fit_diagnostic.py`：默认无操作的日程扩展点、保存/加载计数及 F 封口前证据钩子。
2. `src/ireset/eval/w1_smallset_fit_r_admission.py`：可选版本化包扩展/cohort、物化时准入验证器接线。

两项均在方案 §7 差异白名单中。`Trainer` 原生更新、模型、损失、tokenizer、数据/条件、FK、配置及锁文件字节未改变。旧常量 LR 的计划及入口仍独立；没有把旧授权或含 D6 的 6120s 执行范围投射为新授权。

## 2. 实现交付

| 文件 | 职责 |
| --- | --- |
| `src/ireset/eval/w1_d1_lr_schedule.py` | D1-only `SchedulePlan`/`ScheduleDriver`、实际 LR 与动量连续性检查、落盘前缀准入、独立日程复核、四类比较及消费门 |
| `src/ireset/eval/w1_d1_lr_workflow.py` | R′ 元数据包、独立授权 schema、精确调用矩阵、一次性范围检查、事前 running 成本行、D1′/F′ 编排及 CPU 比较计费 |
| `scripts/run_w1_d1_lr_schedule.py` | 版本化 worker，分离 R′、授权真实链和显式合成夹具 |
| `scripts/run_w1_d1_lr_schedule_supervisor.py` | 私有适配旧监督器、D1/F 预算与比较绑定、固定根独占 claim、独立实际 wait/写后覆盖观察 |
| `tests/test_w1_d1_lr_schedule.py` | 45 项纯合成测试，含临时 SOURCE 物化至 F 和真实子进程 CLI 贯通 |

关键契约：

- `u=1..500` 使用 `1e-4`；仅在 `u=501` 的第一个 micro 前把所有参数组切到 `5e-5`。step500 的更新和 probe 属于前缀；不重建优化器、不清除动量、不改变 weight decay。
- 前缀文件每份写后即核。第 501 步准入前再次读取 **51 probes＋输入＋教师共 53 文件**及 fsync 后的 500 行历史；缺失或差异在首个改变后的 micro 之前拒绝。独立 F 再从盘上重算期望日程和共同前缀。
- 除继承的 AdamW 配置、参数名册、状态 step、有限性及梯度覆盖检查外，增加实际状态摘要链，拒绝实例替换、原位清零及 `.data` 清零；记录切换前后状态摘要相等。这些是只读证据，非新的优化器更新。
- 调用矩阵分域记录 loss 内解码、保存量解码、fresh 解码、checkpoint 检读/strict 加载、参考模块构造/缓存命中、实际 LR/状态读取、前缀哈希与配对归约。真实上限由同一定义给出：conditioner/backbone 各 4103、生产 FK batch 8104、D6 全零。
- `comparison.json` 在 F 工作截止内生成，SHA/字节数进入 F 封口及链摘要，再受监督器和独立返回观察绑定。文件自带 `pending_outer_return`，**单独文件不可消费**；消费端还要求外部实际观察的 observer exit0。
- 完整但四门未过时仍允许 F 核验并发布负结果；前缀、日程、工件、重放或返回覆盖失败时为 `not_comparable`，比较值为 null，不伪装成科学负结果。

## 3. 测试与回归

最终源文件及方案字节冻结后的命令和结果如下；各命令独立进程运行，**没有运行全项目套件**。

```bash
uv run --locked pytest tests/test_w1_d1_lr_schedule.py -q
```

**45 passed，135.03s。**

```bash
uv run --locked pytest tests/test_w1_smallset_fit_diagnostic.py tests/test_w1_smallset_fit_review16.py tests/test_w1_i_outer_return.py -q
```

**80 passed，246.82s。** 包含旧常量路径、旧合成反例电池、身份/截止/外层返回回归；不把旧真实验收重新解释为新实现的真实验收。

```bash
uv run --locked pytest tests/test_w1_smallset_fit_r_admission.py -q
```

**58 passed，8.24s。**

```bash
uv run --locked pytest tests/test_w1_n0_coordinate_flags.py tests/test_w1_n1_synthetic_numerics.py -q
```

**33 passed，15.28s。** 本交付相关测试分进程合计 **216 passed**；不是新增了 216 项测试，也不是全项目结果。

### 3.1 新测试覆盖

- 真实编号 `500/501/502` 期望序列与短等价切换驱动；旧计划拒日程参数、D6/扩散/非法切换点拒绝。
- 提前切换、滞后切换、多参数组漏切、按 micro 改 LR、换优化器、原位动量清零、`.data` 清零均在后续实际 step 前拒绝。
- 前缀差一字节、已保存收据后再篡改前缀、缺 probe、缺历史、伪造 LR/状态 step、端点篡改及 F fresh 不一致。
- 四类互斥比较、根位置退化、布尔/NaN/Inf/超大整数/字符串/null 测量拒绝。
- 旧授权 schema、多含 D6/重复 includes、旧 SHA、未签草案、布尔预算及重复启动/未知历史拒绝。
- 截止后零晚更新、比较耗时越工作截止撤回、日程失败不构造 F 模型、6120s 旧预算错绑拒绝。
- R′ 扩展在合成 SOURCE 上仅元数据，禁止 `torch.load`/Trainer 构造；准入后临时 SOURCE→物化→原生 Trainer→F 全链通过。
- 子进程监督器正常贯通、返回后比较消费，以及无实际 exit0 或比较文件被篡改时 null 消费。

### 3.2 开发期间失败与处置

早期合并运行旧诊断、review16、准入和外层测试曾得到 **134 passed / 4 failed**，四项均为 `runtime_identity:cpu_threads`：旧诊断文件的 session 级自动 fixture 把 Torch 线程改为 2，而准入夹具引用 T6 已登记的线程元数据。单独运行准入即通过；本轮未为此放宽生产 runtime 门，也未修改旧测试或 T6 元数据。最终采用上面的独立进程回归，58 项准入全部通过。

新测试开发中还暴露了坏历史的结构化异常、F 检读计数快照缺项，以及直接调用编排器尚未经过 worker stdout 覆盖而没有顶层返回证明的夹具假设；均已修正后再运行最终 45 项。最终命令没有遗留失败；此前通过数不代替最终字节验证。

## 4. 独立合成验收

实际执行命令：

```bash
/usr/bin/python3 scripts/run_w1_d1_lr_schedule_supervisor.py --synthetic --output-root derived/audit/w1_d1_lr_half_after_500_v1/synthetic/acceptance_001
```

外层命令 **exit0**。该目录为本次独立合成验收；pytest 中的子进程试验另在临时目录，不是新的真实候选。

夹具 A/B 均为随机初始化微型确定性模型和替身 FK：单窗 60 帧、d16、4 次更新、每步 probe，等价切换点为第 3 次更新前。**没有执行真实 A/B 的 1000 步训练。**

| 事实 | 保存结果 |
| --- | --- |
| B 更新/probe | 4 / 5 |
| 实际 LR | `[0.002, 0.002, 0.001, 0.001]` |
| 前缀 | 3 份 probe＋输入＋教师，共 5 文件全部 SHA/字节数相等；前 2 行历史相等 |
| conditioner/backbone | D1 各 21 forward；F 各 2 |
| D1 原生更新 | loss/backward 各 16；clip/AdamW step 各 4 |
| F | initial/last fresh 逐位重放通过；checkpoint 读取 2、strict 模块加载 4；优化器更新 0 |
| 完整性与日程 | D1 complete、调用矩阵 problems=[]；F complete/replay=true；5 个配对 probe |
| 合成拟合裁决/比较 | `fit_insufficient` / `no_prespecified_improvement` |
| D6 | 未创建、未运行 |
| 工件大小（独立返回时） | 1,279,710 bytes，低于 2 GiB |

该负结果是验收正常分支，不要求合成夹具拟合达标才能证明机制；四类裁决另有固定值单测。

### 4.1 返回与成本证据

外层观察器的起点先于 uv/import/运行登记。监督器子进程实际 wait 返回后：

- 实际 wait 耗时 **29.797162030s**。
- `completion_verification.json` 写盘后的样本 **29.806407760s ≤ 登记 32.674414701s ≤ 链上限 2400s**。
- 再写 `observer_write_observation.json` 后复核覆盖，成功分支采样后没有输出/额外 I/O，以外层实际 exit0 验收。最后一次采样的数值未另写回文件，不把 29.806407760s 误称为该观察文件写后的数值。
- worker stdout 覆盖、阶段证明、链摘要、F 与比较文件由监督器及独立观察的 SHA 闭链核验。

关键哈希：

| 工件（相对合成验收根） | SHA-256 |
| --- | --- |
| `completion_verification.json` | `7bfca03535078b5c3d4cce9859311080af5d806142b0f114814a308bb727750d` |
| `attempt_001/acceptance_summary.json` | `0bf54afedf1f9feff1e221806ac358e5d7faa89a562008c0a39ca006f9482717` |
| `attempt_001/acceptance_verdict.json` | `a66a4d9b7da41805efce00f082369c46ad6b7c1cf653991c56a99a0168e6c059` |
| `attempt_001/f/f_report.json` | `353ce0ead988e587e8502dace6cfc1a0b7db688bb86e50ed0d493ac46c53d9c6` |
| `attempt_001/f/comparison.json` | `44742e58373e83e2c09240a6241236457771c6091b6bae8f2b589be34978d48d` |

只读复核命令 `uv run --locked python /tmp/opencode/verify_w1_lr_acceptance_20261118.py` 通过：重哈希实现与全部 D1 索引工件、核对日程序列/前缀/比较/F/外层绑定；没有额外模型构造、forward、optimizer 或 FK 调用。其后只新增本验证文档及项目进度摘要，未改变已绑定实现/方案字节。

## 5. 历史保护与下一停止点

`python3 /tmp/opencode/w1_lr_integrity_20261118.py` 最终重哈希 **393 文件、changed=[]**，覆盖旧保护清单的派生证据、真实 A 及其严格验收根。该检查只读文件字节/JSON，真实 checkpoint payload 加载、真实数组解码、模型调用均为 0。真实 R′ 根与真实 B 根均不存在；本轮新运行仅在 `synthetic/` 下。

已有的 `docs/README.md` 修改在本轮开始前就存在，未覆盖、未夹带入本轮提交。旧冻结主方案、运行、缓存及授权不改写；旧 `attempt_001` 的 incomplete、A 的 `fit_insufficient`、T6 `audit_incomplete`/539 gaps 和一次性授权消耗均保持。`partial/null/false/not_ready/production_integration=false` 保持，不宣称碰撞安全、持续抓握、泛化或 RL/真机资格。

**后续必须依次：复阅本 I′ 交付→明确安排一次只读元数据 R′→复阅新冻结准入包与未签草案→绑定最终 SHA 的新 §8.3 一次性明确授权→一次真实 B 的 D1′/F′。** 新授权范围必须恰为 `[D1,F]`；未获授权不运行，任一失败/未知不自动重试、不改倍率、不补跑 A、不追加 D6。

## 6. 复阅修复与复审（2026-11-18）

首轮复阅对 `d1cca09` 的裁决为不通过；六项反例记录保留在 `/tmp/opencode/w1_lr_review_20261118_002/review.md`。本节是修订后证据，不把原 `acceptance_001` 追认成修订版验收。

1. **硬截止与进程回收（P1）**：统一受管进程组，最外层负责终止与回收，worker 在工作前核对实际组。子进程夹具覆盖不响应 TERM 的 worker 和内层先退出留下的 worker；两条路径均无残留进程或僵尸。旧监督器文件未改。
2. **R′ 实际返回（P1）**：单独的 R 父进程等待观察器实际返回，并绑定 launch ID、PID、封口 R 摘要和两份观察文件。只有该次实际退出及成本覆盖成立才发布草案；正式 v2 授权还需绑定返回收据的 SHA，并记录该次 R 最外层由调用方实际观察的 exit 0。慢写观察文件的协议夹具保留完整候选文件但实际退出 3，不能发布实际返回收据，消费端拒绝。没有运行真实 R′；成功支对照仅使用合成协议文件。
3. **比较实际工件消费（P1）**：检查 A/B 文件字节、准入与授权、阶段报告/台账/返回证明。仅删除 B 的 probe、LR 历史、F 返回证明或 A 教师文件均返回空比较值；单字节翻转在只读复审另用隔离副本验证。
4. **合成根隔离（P2）**：默认输出为 `derived/audit/w1_d1_lr_half_after_500_v1/synthetic/acceptance_default`，不能指向真实根及其子目录、别名或其他项目内历史目录；仍允许独立临时目录。每个根只预留 `attempt_001`。
5. **阶段成本同步撤回（P2）**：报告、台账、证明由同一覆盖循环对账，计入最后一次证明写盘；慢台账/慢证明导致 D1′/F′ 同步撤回、成本只增，持续慢写则标记 `cost_unproven`。F′ 的起点绑定 D1′ 完成覆盖的采样。
6. **坏索引收口（P2）**：`files=[]` 现在落 F `incomplete` 报告与返回证明，零 fresh 构造/预测。前缀另补持久化边界：NPZ 写后同步，收口不重写；盘上 LR 历史与内存逐字段不一致时，首个变更后的 micro 前停止。

### 6.1 测试与新合成验收

```bash
uv run --locked pytest tests/test_w1_d1_lr_schedule.py tests/test_w1_d1_lr_review_regressions.py -q --basetemp /tmp/opencode/w1_lr_retest_suite_001
uv run --locked pytest tests/test_w1_smallset_fit_diagnostic.py tests/test_w1_smallset_fit_review16.py tests/test_w1_i_outer_return.py -q --basetemp /tmp/opencode/w1_lr_retest_legacy_001
uv run --locked pytest tests/test_w1_smallset_fit_r_admission.py -q --basetemp /tmp/opencode/w1_lr_retest_admission_001
uv run --locked pytest tests/test_w1_n0_coordinate_flags.py tests/test_w1_n1_synthetic_numerics.py -q --basetemp /tmp/opencode/w1_lr_retest_n0n1_001
```

LR 与复阅反例 **59 passed / 162.39s**（初版 45 项加 14 项）；旧诊断/review16/外层 **80 passed / 262.47s**；准入 **58 passed / 8.68s**，N0/N1 **33 passed / 16.77s**。四组分进程执行，合计 230 项通过，未跑全项目套件。

新权威合成根：`derived/audit/w1_d1_lr_half_after_500_v1/synthetic/acceptance_002/`。执行 `/usr/bin/python3 /tmp/opencode/run_w1_lr_review_acceptance_20261118.py`，父进程启动 `scripts/run_w1_d1_lr_schedule_supervisor.py --synthetic --output-root <该根>` 并观察实际 `wait()` 返回 **0**：

- 实际外层返回耗时 **31.759001s ≤ 登记 34.567427s ≤ 2400s**；内层等候结束 31.729380s，completion 文件写后样本 31.737369s。实际外层返回覆盖了后续观察文件的写入；外部存档本身是合成验收的单列只读复审证据。
- B 4 更新/5 probes、有效 LR `[.002,.002,.001,.001]`，5 份前缀数值工件逐字节相等，F initial/last 逐位重放通过；`fit_insufficient` / `no_prespecified_improvement` 为夹具结果。未执行 D6。
- 33 文件实现身份 `7bddd5ee1f1281c605fa91b86f8afa65e00c6c22f2437f2a6632cc5944cb8d95`；方案 SHA `2b5a0a491ea076703612c3d5e334fe2fadbd8bca6b448d09c6c5a787eede0c09`。相对 A 旧实现清单仅两份原白名单模块变化；原生 Trainer/模型/目标/数据/配置不变。
- 新 completion SHA `bacfcd327c5a4ef002810dc10cffa8704e495820643695640238c069cfd72c4f`；comparison SHA `7ff6d59833698c605a6ccf582c540840655fa9fe077995f8e9bd3bdc1d91b29c`。

### 6.2 只读复审结论

上一轮六项反例均有相应回归，修订后用独立读盘脚本 `uv run --locked python /tmp/opencode/verify_w1_lr_postfix_20261118.py` 对账两阶段的成本、交接、实际返回、实现与比较绑定，并在隔离副本上以 probe 单字节翻转确认不可消费。本次范围内未发现新的阻塞项。只读核对 **393 个历史保护文件、52 个旧合成文件均零漂移**；脚本结果与实际等待记录在 `/tmp/opencode/w1_lr_postfix_20261118/`。两真实根均未创建；没有读取真实数组数值或加载真实 checkpoint。**复审通过仅限工程与合成证据。下一步仍须明确安排仅元数据的 R′；真实 B 须另获绑定最终材料的严格 `[D1,F]` 一次性授权，不含 D6。**

## 7. 一次真实 R′ 准入通过，B 待新授权（2026-11-18）

按整份方案推进的任务安排，在 `e96e0ad` 上只执行一次 `/usr/bin/python3 scripts/run_w1_d1_lr_schedule_supervisor.py --mode r-admit`，不覆盖缓存或输出根。固定根为 [`r_admission/`](../../derived/audit/w1_d1_lr_half_after_500_v1/r_admission/)，唯一尝试 `attempt_001`。`admission_passed=true`、`problems=[]`、`seal.status=sealed_within_budget` 且 `terminal=true`。本节仅追加验证记录，不改已绑定的方案、实现或先前验收字节。

- `/tmp/opencode/run_w1_lr_r_admission_once_20261118.py` 的独立父进程 `wait()` 取得 R′ 最外层采集器实际 **exit 0**。观察器自身亦实际 exit 0，其返回耗时 **6.519662s**；采集器返回 **6.552050s**（相对监督器 `t0`）。从更早的启动前采样计至独立返回存档 `fsync` 后 **6.596295s ≤ 登记 9.580495s ≤ 120s**，内含 15s 收口预留。启动、封口、stdout、各级返回及 SHA 绑定一致。R′ 根最终 16 文件、103,998 bytes；准入时空闲 26,558,083,072 bytes，满足 7 GiB 下限。
- 只读 JSON、字节哈希与数组头。保存报告的数组值读取、checkpoint 反序列化、模型/FK 构造、forward/backward、optimizer step 都为 **0**；没有重建缓存。从原 SOURCE 复核预注册/manifest、资产和窗位，复用 A 绑定的 `20261117_84a1b06_w1_smallset_fit_cache_rebuild/cache/attempt_001`（identity SHA `da4bb29235dcdc2401df3ad58ebc51c7502a8187ccc3d8dacdcb0a01e5f2f810`），复核 **55/55** 已登记逐位迁移证明与 train24 root_stats。唯一执行窗为 `sub10_largebox_097:0:60`；另外五窗是来源登记，不授权执行。运行时元数据与 T6 登记一致（Python 3.12.13、Torch 2.14.0+cu126、RTX 3090），数值 runtime 硬门将在 D1′/F′ 各自计费窗口重新应用和核验。
- `PYTHONPATH="$PWD/src" uv run --locked python /tmp/opencode/verify_w1_lr_r_admission_20261118.py 191218.260867639` 从盘复核实际等待记录、终态封口、SOURCE/cache、固定窗、预约预算、冻结矩阵与 33 文件身份，重核 A 所绑定的 **125** 个文件（含 101 个 probe、旧 R005 及已消费授权），未解码数组。与 A 的旧实现清单仅两份白名单诊断/准入模块不同；393 个历史保护文件、52 个旧合成文件零漂移。未签草案经授权门实测拒绝，未调用 B 入口，B 根与正式授权目录均未创建。

### 7.1 冻结材料与拟执行命令

以下路径相对于 `derived/audit/w1_d1_lr_half_after_500_v1/`：

| 文件 | SHA-256 |
| --- | --- |
| `r_admission/attempt_001/r_admission.json` | `56e0386f8f2784e124864f845dfa8d27eba817acaee203e55d90f8c2bb9aef02` |
| `r_admission/authorization_draft.json`（未签） | `f101be40940f45428c5e48d8e796ac970699c1b1b63ca268465e0aa4d61613a1` |
| `r_admission/r_observer_actual_return.json` | `26be090e223cb7ae933e8bc62df1d2244f0180ee574efaa2dc959bf377b7abab` |
| `r_review/collector_actual_wait.json`（独立等待记录） | `cef2f282bcd13461ab889f8923e46f62e14cc0377707cfaa9a46ad79e775b8c4` |
| `r_review/b_execution_proposal.json`（本表及命令的绑定入口） | `8ed68153c68bfe1c0b471b2121c2eaf88b7135f70100137dd96d09bde41e2fe5` |

新矩阵 SHA `d89bcf4c5bbec9a4f5851a60e7029d0ed5875965070b9d7dfda111e34b4ce6fb`。最终包内还绑定方案、实现、基线及比较规则的各自摘要。根级草案为 `w1_d1_lr_authorization_v2`、`authorized=false`、`template_only=true`、七项 checklist 均为 `null`，绑定观察器收据 SHA，但 `capture_actual_return_code` 仍为 `null`；须由用户明确确认最外层实际返回码 0。不得用 attempt 内的预备草案或旧授权。以下命令已冻结但未执行，只能在取得新明确授权并复核材料、磁盘、固定根无历史之后启动：

```text
/usr/bin/python3 scripts/run_w1_d1_lr_schedule_supervisor.py --mode chain \
  --admission derived/audit/w1_d1_lr_half_after_500_v1/r_admission/attempt_001/r_admission.json \
  --authorization derived/audit/w1_d1_lr_half_after_500_v1/authorization/attempt_001/authorization.json
```

### 7.2 §8.3 一次性授权的复阅清单

1. 新研究问题是同窗同起点下后 500 次更新减半 LR 的有限拟合对照；历史 A 只读，不重跑、续训或重用其已消费的授权。
2. 旧 `attempt_001` incomplete、历史 A `fit_insufficient`、T6 `audit_incomplete`/539 gaps 及未覆盖范围均保留；拟合不能证明完整数值、碰撞、持续抓握、泛化或真机资格。
3. 批准 §7.1 冻结材料绑定的实现、配置、资产、SOURCE/cache、窗位和 §6.3 runtime，及方案 §3/§5 的物理指标、唯一终点、probe 布局、解码核验与分域调用矩阵。D1′ 最多 1000 更新、4000 micro、101 probes；F′ 复核全部保存量，只 fresh 重放两端；D6 为零。
4. 只允许固定 `w1_d1_lr_half_after_500_v1` 根一次，`includes` 恰为 `["D1","F"]`。预算 D1′ **1800s/60s**、F′ **600s/60s**，链 **2400s**，含已经执行的 R′ 合计上限 **2520s**；不互借。要求余盘 ≥7 GiB、新工件含临时文件 ≤2 GiB；失败或未知即停止，不更名、不重试。
5. D1′ 保存 initial/last 各一份诊断 checkpoint；F′ 仅对 B 的这两份做 strict 加载与逐位重放。中间 FK 仅核保存点距离归约；尖峰只描述，不做门限验收。
6. 不执行 DDIM、正式 val/test、D6 或生产接入；任何结果都不自动触发后续实验。
7. 1–500 次更新用 `1e-4`；step500 probe 后、501 首个 micro 前所有组切 `5e-5`，不重置优化器或动量。先对账 51 个前缀 probe、输入/教师与 500 行历史；不符即 `baseline_prefix_mismatch`。按四原门和两项各 ≥10% 改善规则判 step1000；证据不足为 `not_comparable`、比较值 `null`。

授权须明确引用 §7.1 提案 SHA，确认以上七项与本次 R′ 最外层采集器实际返回 0，并明确仅执行一次 B `["D1","F"]`。**截至本节：只有准入和未签材料，无真实 B 指标。** 获授权后仍需重核入口，由独立父进程记录 B 的实际退出；只在 F、两侧工件、各阶段成本/封口及外层返回全核验后才发布终点与配对曲线结论。

## 8. 真实 B 执行与最终比较（2026-11-18）

在固定窗 `sub10_largebox_097:0:60`、同一初始化与 1000 次更新下，第 501 步起将学习率减半的 B **在 step1000 通过四项既定平均门**。相对历史 A，根旋转平均误差下降 **82.6%**，教师 15-site FK MPJPE 下降 **77.7%**，根位置与关节也改善并保持达标。按预先冻结的优先顺序，最终是 **`fit_pass`／`single_window_pass`**，不是“改善但仍不足”。

### 8.1 授权与实测执行

- 用户明确复阅 §7.2、确认 R′ 采集器实际 exit 0，并引用 §7.1 提案 SHA `8ed68153c68bfe1c0b471b2121c2eaf88b7135f70100137dd96d09bde41e2fe5` 授权一次 `["D1","F"]`。复核冻结材料、运行时、绑定缓存和空运行根后，另写[正式 v2 授权](../../derived/audit/w1_d1_lr_half_after_500_v1/authorization/attempt_001/authorization.json)（SHA `1e60e045784dcd96fd4811a0946864bcdab4a7754a7b9b35efa2992bd8324dda`）；R′ 草案未改。监督器按 §7.1 的命令只执行一次，固定[运行根](../../derived/experiments/omniretarget_first/w1_d1_lr_half_after_500_v1/)仅有 `attempt_001`，`authorization_consumed.json` 状态 `consumed`、SHA `caa70d56f7c398bdce6cd8f570c96bb957affb4e31181c6fb7c4419f55a6614c`。不含 D6，没有重跑 A、R′ 或缓存，没有重试。
- D1′：1000 次 AdamW 更新、4000 micro／窗消费、101 probes。前 500 步逐字段历史与 A 精确相等；51 份前缀 probe 加输入和教师，共 53 份 NPZ 与 A 逐字节相等。读取落盘 `prefix_admission.json` 和日程证据，确认在 step500 probe 之后、501 首个 micro 前准入，只切换一次：所有组（本次 1 组）实测更新 1–500 为 `1e-4`，501–1000 为 `5e-5`。全部 1000 行步序与状态摘要相接，切换两侧状态摘要相等，保留原优化器及动量。原生训练算术、窗、教师和已冻结实现不变。
- F′：复核 B 全部保存量及实际日程，只 strict 加载并逐位重放 **B initial/last**。两次 fresh probe、两次 checkpoint 加载（各两个模块 strict load）；无训练更新。D1′ conditioner/backbone 各 4101 次 forward、生产 FK batch 8102；F′ fresh 的这三类调用各 2 次，全链分别合计 4103/4103/8104。两阶段调用矩阵与保存证据一致。
- `wait4()` 实测外层观察器 exit **0**。从更早的父进程启动前采样，到独立等待收据及观察文件 `fsync` 后为 **962.240579s ≤ 登记 965.114975s ≤ 2400s**；相对监督器 `t0` 的实际退出为 962.195717s。D1′ 收口采样 **958.105482s ≤ 962.095717s ≤ 1800s**；F′ **3.006640s ≤ 7.002063s ≤ 600s**；各内含 60s 工作尾部预留，交接无空档。各阶段登记预留单独核验，链成本以链封口为准；加 R′ 登记合计 **974.695470s ≤ 2520s**。两阶段、链和返回证明的 SHA 均重核。
- 入口空闲 26,542,993,408 bytes ≥7 GiB；B 终态 132 个文件、39,310,237 bytes。Linux `wait4` 的受管进程树文件写入计量（含临时和重复写入）为 44,109,824 bytes，远低于 2 GiB；无残留 `.tmp`。保护的 393 个历史文件与 52 个旧合成文件零漂移，33 个冻结实现/方案文件与 R′ 绑定一致。

### 8.2 Step1000 主终点

展示值舍入，裁决按绑定工件未舍入值。A 仍是历史 `fit_insufficient`，未重跑，不因 B 达标而改判。

| 指标 | A | B | B−A | B/A | 原平均门（B） |
| --- | ---: | ---: | ---: | ---: | --- |
| 根位置 RMSE（m） | 0.002787434597 | 0.001876769518 | −0.000910665079 | 0.673296 | ≤0.01，通过 |
| 根旋转平均测地角（rad） | 0.023780910989 | 0.004131500103 | −0.019649410886 | 0.173732 | ≤π/180，通过 |
| 关节角 RMSE（rad） | 0.027638625284 | 0.018747667508 | −0.008890957776 | 0.678314 | ≤0.05，通过 |
| 教师 FK MPJPE（m） | 0.016761962649 | 0.003741488069 | −0.013020474580 | 0.223213 | ≤0.01，通过 |

### 8.3 配对曲线与末次尖峰

[比较工件](../../derived/experiments/omniretarget_first/w1_d1_lr_half_after_500_v1/attempt_001/f/comparison.json)保存同一步号配对的全部 **101 点**。step0–500 四项一致；切换后的 50 个配对点（step510–1000），B 四项分别在 **50/50** 点低于 A。B 从 step520 起剩余 **49** 个采样点四门同时通过（全曲线 49/101，A 为 0/101）；不等于对每个未采样更新也作了评估。step800–1000 的 21 点表明改善不只出现在末探：

| 末段指标 | A 中位数 | B 中位数 | A 达标点 | B 达标点 |
| --- | ---: | ---: | ---: | ---: |
| 根旋转（rad） | 0.028872527084 | 0.004349221973 | 0/21 | 21/21 |
| 教师 FK MPJPE（m） | 0.020529593721 | 0.004543669394 | 0/21 | 21/21 |

Step1000 尖峰描述（坐标为窗内从 0 起算的帧、以及适用时的关节或 site 索引）：

| 局部误差 | A p95 | A max（坐标） | B p95 | B max（坐标） |
| --- | ---: | --- | ---: | --- |
| 根位置（m） | 0.004335956275 | 0.007367336934 `[0]` | 0.003430901229 | 0.004762770905 `[0]` |
| 根旋转（rad） | 0.028084010518 | 0.037709345273 `[0]` | 0.007857574776 | 0.018492568054 `[0]` |
| 关节（rad） | 0.049284709928 | 0.295979567290 `[0,9]` | 0.033014093651 | 0.261446583986 `[0,9]` |
| FK site（m） | 0.042028897785 | 0.060052104180 `[1,11]` | 0.008655371145 | 0.044133250931 `[0,4]` |

`spike_gate_status=not_evaluated`；平均门不适用于局部最大值，也不证明没有尖峰。`fk_replay_scope=initial_last_only`；中间 FK 只核对保存点距离归约。本次结果只回答固定窗、固定 seed、固定预算的拟合问题，不能用于推断跨窗泛化、碰撞、持续抓握或生产/真机资格；也不单独分离 AdamW 随学习率改变的衰减量。旧 `attempt_001` 的 incomplete、历史 A 裁决和 T6 `audit_incomplete`／539 gaps 不变。**本方案收口，不自动追加 D6 或后续实验。**

### 8.4 可复查的证据

独立[核验摘要](../../derived/audit/w1_d1_lr_half_after_500_v1/b_review/verified_outcome.json)（SHA `b40117137c02f1f65278749c12861262d9ca774350afb2ba995851909b255ac5`）绑定外部实际等待、D1′/F′ 报告、台账、比较与完成验证。底层 `comparison.json` 按协议保留 `pending_outer_return`，不改原字节；独立核验用实际退出码消费比较，重哈希 A 的 125 份绑定文件和 B 的 110 份 D1 索引工件，独立复算 101 点及四项终点规则。省略外部实际退出码的对照返回 `not_comparable`／`metrics=null`。

从项目根执行以下只读核验命令，不训练、不加载 checkpoint、不解码数组、不追加模型或 FK 调用。参数是这一次外部等待存档 `fsync` 后已观测的单调时钟值。执行和归档后核验均通过，本轮未重跑全项目套件。

```bash
uv run --locked python derived/audit/w1_d1_lr_half_after_500_v1/b_review/verify_saved_b.py 197765.26271136
```

| 关键文件 | SHA-256 |
| --- | --- |
| `b_review/observer_actual_wait.json` | `adee82812805ffbbd678cde4f3ed2d7cc8bc3f7a4358616a018c76bf0f3c2063` |
| B `completion_verification.json` | `6b629bcf350fd3f176445f2bf64129c3d3f15dd208465b9cad9449d180c313b4` |
| B `attempt_001/d1/stage_report.json` | `1744ae4061264705779e662fbf4559334753f48227859195991ce89855992949` |
| B `attempt_001/f/f_report.json` | `9beb3c7643c3162a95ea761f82a13e19b6970dcb669946a198a2ab8a16c958bf` |
| B `attempt_001/f/comparison.json` | `1c50c5974079d18a4141acbb8c608b5026839e54cf3e53523ed9430593d80fe7` |
