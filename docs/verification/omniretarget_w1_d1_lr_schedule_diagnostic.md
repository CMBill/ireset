# W1 固定单窗学习率日程诊断：I′ 验证记录

日期：2026-11-18。对应[方案 v0.1＋§7](../plans/omniretarget_w1_d1_lr_schedule_diagnostic_plan.md)。

**当前交付：首轮复阅发现的 3 项 P1、3 项 P2 已修复，新合成验收和只读复审见 §6。真实 R′ 未执行，B 未授权、未运行。** §1–§5 为初版 `d1cca09` 的历史验收记录；合成夹具不构成真实学习率效果证据。

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
