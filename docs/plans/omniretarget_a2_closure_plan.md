# OmniRetarget A2 收口与有限恢复实现方案

日期：2026-10-10。版本：v1。状态：**仅完成方案编写；下述接口、诊断、恢复实验和收口报告尚未实施。**

本方案承接 [OmniRetarget 优先教师与初步训练执行方案](omniretarget_first_training_plan.md) §3.3–§3.4、§8。当前默认安排是登记停止扩大 U0，并用一次有限诊断判断 sub12 是否存在值得尝试的具体恢复路径。只有诊断满足本文预设条件，才预注册并执行一个独立恢复候选。没有合适候选也是有效出口，不为取得训练入口反复调参。

本阶段交付的是可核验的来源收口和后续路线依据。原 A2 的 `incomplete` 裁决保留；诊断完成、恢复成功、教师质量改善、训练就绪分别登记。

## 1. 当前基线与事实边界

### 1.1 唯一基线

| 对象 | 基线位置或身份 |
| --- | --- |
| 主项目实现 | `cec0287`；审阅登记为 `d5f31bd`，后续实现使用其实际新提交身份 |
| A1 冻结 | `derived/experiments/omniretarget_first/20261009_13906c1_a1/freeze/` |
| 原 U0 生成证据 | `derived/experiments/omniretarget_first/20261009_123bc6f_a2_reviewfix/teacher_pilot/u0/`，含 `attempt_001/002` |
| 权威比较 | `derived/experiments/omniretarget_first/20261010_cec0287_a2_reviewfix7/teacher_pilot/cmp/attempt_001/teacher_selection.json` |
| 发布 P | revision `135259130158f1cb1799e4ab9b6566c938285fc9` 的 original 标签 |
| 上游代码 | Holosoma `bccd4d7451640a2800ddc77e469d911a84f91994`，保持原 checkout 只读、干净 |
| 上游环境 | `derived/upstream/holosoma_env/`，Python 3.12、MuJoCo 3.10.0；实际身份在执行时复验 |
| 已登记测试 | 第七轮全量 `438 passed + 1 skipped`；不作为本文未来新增接口的通过证据 |

编写本文时读取的完整 SHA-256 如下。执行时先对照这些锚点，再复验它们指向的输入和产物；清单文件未变不等于清单内文件未变。

| 文件 | SHA-256 |
| --- | --- |
| 权威 `teacher_selection.json` | `bc03d085535c462c70864fabf265d399b3a18838d36587138d3f2c67d2f3c683` |
| A1 `frozen_lists.json` | `395094a30d69158ad08bf5b576f07c541c052187d5bf242f1cc7fde05e353d9d` |
| A1 `assets.json` | `2f72c9ed0f1832b0c1d80e7d1f2e82382785c56ae83556a63a46ec6789c7ad79` |
| A1 `pairing_source_boxes.npz` | `39fce035ab63569bd69a4ddf61e1e6cd4fb70d4d1b2a12abf7051bd55ce3d09d` |
| 上游环境 `uv.lock` | `0fe6c3679e32925371a95b87a1953c72baeaab7571e91a19ef2f53887c4073b0` |

### 1.2 冻结六段与现有结果

| 演示 | T | P 最大穿透 / mm | U0 最大穿透 / mm | 当前证据 |
| --- | ---: | ---: | ---: | --- |
| `sub10_largebox_097` | 126 | 15.968 | 35.967 | g3/g5 失败；右手漂移 0.344→0.457 m |
| `sub12_largebox_082` | 347 | 25.546 | 缺产物 | 生成失败；保持 missing 行 |
| `sub16_largebox_000` | 89 | 1.886 | 19.229 | g3/g5 失败；新增 penetration 硬失败 |
| `sub3_largebox_027` | 165 | 9.768 | 19.839 | g3/g5 失败；右脚速度回退 |
| `sub7_largebox_022` | 151 | 1.783 | 37.187 | g3/g5 失败；双手 gap 改善但覆盖下降 |
| `sub8_largebox_035` | 157 | 1.484 | 21.108 | g3/g5 失败；新增 penetration 硬失败 |

以上数值来自权威报告的序列化字段，保留其登记精度。任务跟踪 g2 在五段可测 U0 上均在界内；g4 只有一个演示的双手间隙达到改善幅度，不满足至少三个演示的要求。完整比较仍缺 sub12 的有效 U0 标签及绑定证据。

sub12 在 MuJoCo 3.10.0 下两次默认运行均报 `CVXPY solve failed: infeasible`，最后可见进度为 171/347，墙钟分别为 143.95/144.65 s。**171 是历史进度条记录，不直接当作新诊断的零基失败帧。** 新诊断必须捕获实际 `frame_idx` 和帧内迭代序号。

只补齐 sub12、保持现有五段结果时，`local_preferred` 已不可达。该事实不能把当前 `incomplete` 提前改成 `no_quality_gain`。改变优化配置后重新生成的其他五段属于新候选，需重新测量，不能预先断言其结果。

## 2. 本阶段范围与路线

### 2.1 默认执行顺序

1. **C0：复验并冻结现有证据。** 确认历史报告与现行产物一致，只有已知 sub12 缺口。
2. **C1：登记原 U0 停线。** 保留正式裁决 `incomplete`，暂停同配置扩量和教师生产。
3. **D0：有限故障诊断。** 检查失败是否与上游步长半径存在直接关联，获得可核验的失败上下文。
4. **G0：登记下一路线。** 没有符合条件的恢复候选时结束本阶段；有候选时先冻结独立实验记录。
5. **UR：条件分支。** 仅执行一个预注册恢复配置，对冻结六段完整生成和比较，形成新的来源裁决。
6. **C2：交付收口报告。** 分别说明原 A2、新诊断、新候选和训练准备的状态。

C0/C1 是第一批实现与验收对象。完成后再实现 D0；不先开发通用求解器、训练数据管线或参数搜索框架。UR 是条件分支，不能在 D0 得到证据前直接运行。

### 2.2 固定约束

- 保持 A1 六段、完整长度、30 Hz、源人体参考、目标箱体、地面、普通手 G1 和数据分组不变。
- 源场景箱来自 A1 的 `pairing_source_boxes.npz`，目标箱来自教师绑定轨迹；不重新混为同一场景。
- 源 OMOMO、发布教师、模型、归档、上游 checkout 和旧实验结果只读。新产物写入新的 `derived/` run。
- 主项目使用 `uv run --locked`；上游求解在独立 uv 环境运行，不将其 CVXPY/NumPy/MuJoCo 依赖混入主环境。
- 不使用发布机器人标签作为初始化，不替换困难演示，不截短或降速，不放松独立验收阈值。
- 自研修复、T4 教师处理、train24/val31 生产、训练、增强和 test 指标分析均不属于本阶段的自动执行项。
- 原 U1 继续 `not_applicable`。失败后设计的恢复实验是新实验，不能补登记为原轮 U1。

## 3. C0：现有证据复验与冻结

### 3.1 复用现有实现

优先复用 `src/ireset/data/a2_integrity.py`、`omniretarget_sources.py` 以及 `src/ireset/eval/teacher_comparison.py`。调用现有比较入口生成新的比较 attempt，不复制质量指标、g1–g5 或配对算法。

复验内容：

1. 权威报告、A1 三个锚点文件和上游 lock 与 §1.1 一致。
2. 现行六段 PT/发布 NPZ 与冻结哈希一致；120 项资产逐文件重哈希。
3. 现行上游提交/干净状态、生成时记录和环境三元一致；实际导入版本为 3.10.0。
4. 五段 raw 哈希、raw→converted 逐位转换、人体/目标箱/fps/T 绑定均成立。
5. 现行消费映射仍为五段取 `attempt_001`，sub12 无 OK 产物；两个历史失败 attempt 都保留。
6. 新鲜比较的 `metrics`、质量门、证据缺口及 admission 与权威报告相符；时间戳、比较 attempt、新报告输出路径及耗时单独比较，不作为数值差异。实际消费路径及其输入哈希仍完整对账，不排除在核验之外。

比较中的 known missing 仅限 sub12。输入、资产、环境、产物链或其他演示新增失败时，登记 `unexpected_evidence_gap`，先修具体问题，不继续诊断或登记恢复候选。

### 3.2 消费范围与历史证据

保持第六/七轮的两条不变量：每个目录、每个 attempt 只核验其当前消费 demo；合并结果必须覆盖冻结六段，missing demo 的输入仍被核验。manifest 腿与冻结腿必须使用同一消费映射和实际记录目录。

旧 manifest 无 `data_path` 时，继续默认目录回退并标记 `recorded_in_manifest=false`。其哈希证明当前默认目录字节与生成记录一致；不能据此补写或宣称历史消费路径已被直接记录。新诊断与 UR 必须记录实际消费路径。

### 3.3 输出

拟生成 `baseline_evidence.json` 和 `evidence_files.json`：记录原路径、文件大小、完整哈希、引用关系、六段分母、现行消费映射及每一核验结果。报告、manifest、配置等小文件可保存哈希一致的快照；大资产和原数据保存路径/哈希引用，避免重复拷贝。

记录本次实际主项目 revision、dirty diff、全部未跟踪文件的内容哈希，以及实际运行的代码范围。开始/结束复核行为身份，运行期间不编辑仓库文件；不能把未来实现仍标为旧 `cec0287`，或把新代码的结果拼成旧运行。上游 checkout 另按 A1 的固定提交和干净树核验。

`evidence_files.json` 覆盖实际消费的 raw/converted、两份历史失败 manifest、attempts index、A1 锚点、配置、环境 lock、行为代码及资产闭包。缺失的 sub12 教师以显式 missing 记录，不制造空 NPZ 或零哈希。

## 4. C1：原 U0 的停线登记

拟新增 `scripts/omniretarget_a2_closure.py`，仅组织证据与决策。其核心判定放入小型 `src/ireset/eval/a2_closure.py`，方便用真实反例验证状态转换。

输出 `a2_closure.json` 与 `a2_closure.md`，至少包含：

| 字段组 | 必须表达的含义 |
| --- | --- |
| `baseline` | 原报告路径/哈希、原 `verdict=incomplete`、六段名单、五段可测、sub12 missing |
| `verification` | C0 逐项结果、新鲜比较路径/哈希、已知与新增缺口分列 |
| `disposition` | `stop_u0_expansion`；停止同配置扩量的质量与成本依据 |
| `quality_evidence` | g2–g5 逐演示证据；五段结果与六段正式裁决分开 |
| `diagnosis` | 初始 `not_run`，执行后写实际状态、证据路径/哈希 |
| `recovery` | 初始 `not_registered`，执行后写独立候选身份与结果 |
| `training` | `not_ready`；T1/W1、生产均不自动启动，记录具体未满足门 |
| `registration_status` | 登记过程是否完成；不得用它替代教师来源裁决 |

正常登记停线可返回 exit 0，含义为登记程序完成。终端与文字报告须同时显示 `baseline_verdict=incomplete`、`training=not_ready`。发现新增完整性问题返回非零并落盘原因；不能只写“收口完成”而掩盖证据损坏。

原 `teacher_selection.json` 永不改写。新诊断或 UR 结果只能作为关联的新记录，不把原 U0 的失败行替换为另一配置的标签。

## 5. D0：sub12 的有限故障诊断

### 5.1 已核实接口与待验证假设

当前上游 `config_types/retargeter.py` 提供 `step_size=0.2`；`examples/robot_retarget.py` 将该值传入 `InteractionMeshRetargeter`，配置字段为 `retargeter.step_size`。

`solve_single_iteration` 组装脚保持、非穿透、限位等约束，最后加入 `cp.SOC(self.step_size, dqa)`，用 Clarabel 求解。非最优且 `init_t=True` 时会尝试移除显式 SOC；后续帧失败直接抛异常。因此可以验证一个有限假设：**失败帧的其余线性约束存在解，但其所需最小增量范数超过 0.2。**

上述代码事实不证明 sub12 的真实原因就是半径不足，也不证明放大半径会改善质量。上游 `dqa` 使用其实际配置变量与尺度，诊断不得替换成项目修复器的 35 维变量或据半径推断物理位移。

### 5.2 诊断薄入口

拟新增 `scripts/diagnose_omniretarget_sub12.py`，父进程用主环境核验输入与组织输出，子进程使用未 `resolve()` 的上游 `.venv/bin/python` 路径运行。

子进程在原包 cwd 下，通过 `runpy` 或真实配置 API 调用现有生成入口；仅在该子进程中包装 `iterate`、`solve_single_iteration` 和 `cvxpy.Problem.solve`，捕获参数与求解状态。包装需校验实际导入文件的哈希和接口签名，退出时恢复；不修改或复制上游求解实现。

包装成功路径必须调用原函数并原样返回，不更新配置、初始化、约束、代价、求解器选项或随机状态。Phase I 只在原求解已失败、原上下文已保存后执行；得到的变量值不回流原生成过程。

若该接口不能可靠捕获原问题与 `dqa`，登记 `instrumentation_not_ready` 并结束。不能用另一套近似约束组装替代失败问题后声称已定位原因。

### 5.3 执行顺序

1. 使用成功对照 `sub16_largebox_000` 完整跑一次带包装的默认生成，比较 qpos/human/fps 与现有同配置 raw 数组逐位一致。时间不要求一致；数值不一致时先解释包装或身份问题，停止 sub12 诊断。
2. sub12 从第 0 帧、原默认初始化完整推进到失败，执行两次带包装运行。不能从发布机器人轨迹或手工 checkpoint 开始。
3. 每次记录零基 `frame_idx`、帧内求解调用序号、完成帧数、原状态、异常、输入/配置身份和失败上下文。两次失败帧、迭代序号、关键数组及原问题数据不一致时，登记 nondeterministic，不选恢复候选。
4. 对每次相同失败上下文各执行一次 §5.5 的 Phase I。两次结果用于重现核验，不再添加第三个参数或约束消融。

### 5.4 失败快照

保存完整 stdout/stderr，不能只留末尾 2,000 字符。数组保存 NPZ、稀疏问题数据用 SciPy NPZ、结构信息用 JSON，均用完整 SHA-256 入册。至少包含：

- demo、T/fps、输入路径/哈希、上游提交和实际导入文件、环境三元、完整生效配置、随机状态或明确的随机设置。
- 实际失败帧与迭代序号，`init_t`、`q_locked`、`q_a_n_last`、`q_t_last`、`q_a_indices`、脚保持状态、可用的 nominal 输入。
- 原求解状态、solver stats、约束数量/形状、显式 SOC 半径和涉及变量；CVXPY 原问题 canonical 数据及变量映射。
- 原失败前上游已返回帧的前缀及最后两个有效上下文；上游 qpos 布局、前缀长度和用途显式登记。该前缀未经整段独立验收，不称为可行轨迹。

失败前缀命名为 `prefix_debug.npz`，标记 `diagnostic_only=true`，保存于 diagnosis 目录，不加入教师 attempts index，不当成完整标签。成功对照的诊断输出也不进入教师生产。

### 5.5 唯一 Phase I 检查

在失败子进程中复用**原问题的约束表达式与原 `dqa` 变量**，确认只有一个与 `step_size` 对应的显式 SOC，再求解：

```text
minimize ||dqa||₂
subject to 原失败问题中除该步长 SOC 外的全部约束
```

使用现有 CVXPY + Clarabel；保留脚保持、非穿透、限位和所有其余约束。通过原 solver 方法调用诊断问题，防止包装递归。保存原求解状态后再执行，不用 Phase I 状态覆盖原失败状态。

诊断仅接受 `OPTIMAL` 且全部保留约束的违反量有限、最大值 ≤1e-6 的解；`OPTIMAL_INACCURATE`、缺值、NaN/Inf、超时或残差越界均为诊断证据不足。1e-6 是此诊断的数值核验界，不修改机器人独立验收阈值。

令 `rho` 为该解的 `||dqa||₂`。为本方案预设一个候选半径 0.3，诊断结果按下表处置：

| 结果 | 登记与出口 |
| --- | --- |
| 保留约束问题仍报 infeasible | 记录该线性化问题无可核验解，原 A2 保持 incomplete；不宣称原任务物理无解 |
| 状态或残差不足以判断 | `diagnostic_incomplete`，停止；不按 solver 字符串推导结构性结论 |
| `rho ≤ 0.2 + 1e-6` | 原失败可能涉及数值或其他问题，不支持本方案的半径恢复；停止 |
| `0.2 + 1e-6 < rho ≤ 0.3 - 1e-6`，两次一致 | 支持登记单一 0.3 半径候选；仍需完整重新生成验证 |
| `rho > 0.3 - 1e-6` | 超出本方案有限恢复范围；停止，不试 0.4/0.6 或无界半径 |

两次一致要求原问题关键数组逐位相等、分类一致，rho 的绝对差 ≤1e-6。Phase I 只回答该失败线性化问题的局部可行性，不证明非线性约束满足、整段可恢复或质量改善。

## 6. G0：恢复候选预注册与停线出口

### 6.1 不满足恢复条件

登记 `recovery=not_applicable` 或 `not_ready`，给出具体原因、两次诊断状态、信息缺口和成本；原 U0 保持停线。C2 仍可以完成证据登记与路线交付，不能将它写成教师比较通过。

后续建议最多列两个有证据的问题，例如输入条件/源目标语义排查、特定教师失败族处理。每项写具体证据、待验证假设、所需能力与有限预算。它们是后续方案提案，不直接调用 T4、自研修复或 W1。

### 6.2 满足恢复条件

在读取任何新恢复标签质量前，创建不可覆盖的 `preregistration.json`，完整身份为：

| 字段 | 固定规则 |
| --- | --- |
| `experiment_id` | `a2_recovery_step03_v1`，属于失败后的独立实验 |
| `source_id` | `upstream_recovery_step03_v1`，简称 UR，与原 U0/U1 分开 |
| `parent` | 原权威报告与 C0/D0 证据路径、完整哈希 |
| `expected_demos` | A1 冻结六段、完整 T，全部重新生成 |
| 唯一优化配置差异 | `retargeter.step_size: 0.2 → 0.3` |
| 保持项 | 代码、环境、资产、初始化、脚保持/非穿透/限位、物体/人体/fps/T 和验收协议不变 |
| 主质量指标 | 沿用原 g4 的主指标及已登记 fallback；g1–g5 全部沿用 |
| 候选数量 | 1；每段正常一次，工程异常或需核验新失败时至多一次相同配置重试；理由与两次成本入册 |
| 停止条件 | 缺证据/预算耗尽即 incomplete；无改善即停止，不追加候选 |

0.3 是本文的预设实验上限，不是已经验证的修复参数。若需要另一半径、 solver 参数或多字段更改，必须形成新方案，不能在该候选身份下临场修改。

注册时从上游配置类型与实际 builder 导出生效配置，保存完整配置和规范化哈希。与默认配置比较时仅排除 task 名称、读写路径等逐 demo 运行元数据；优化、初始化、资产和场景设置不得排除。优化差异必须恰好为上述一项。

预注册记录以独占创建方式写入并计算哈希，后续每个生成 attempt 和比较报告引用该哈希。UR 产物已存在而没有事前注册时，拒绝追认；配置更改写新实验，不覆写注册文件。

## 7. UR：独立恢复生成与质量比较

### 7.1 生成入口的最小扩展

在 `scripts/generate_omniretarget_teachers.py` 增加拟议参数 `--source-record`。省略时维持既有 U0 语义；指定时仅支持本文已验证的 UR 注册身份和单字段差异，不开放任意参数透传。

实施时上游 CLI 使用 `--retargeter.step-size 0.3`，并核验实际实例的 `step_size` 与配置哈希。新增 CLI 支持须经真实配置解析烟测，不能只验证字符串出现在 argv 中。

新的 generation manifest 记录 `experiment_id/source_id`、注册路径/哈希、完整生效配置/哈希、实际消费目录、开始/结束身份和全部原有证据。所有 consumed attempts 必须属于同一注册身份；比较前从 argv/冻结环境重新核验生效配置，不仅相信 manifest 的配置字符串。

### 7.2 六段与重试

UR 对六段**全部重新生成**，在独立根目录建立新 attempts index。不能把五段 U0 与一个恢复 sub12 拼成“统一 UR 配置”，也不能把 UR 子目录传给旧 U0 比较入口冒充原来源。

一次上游求解失败不会静默取消其余 demo。每段记录 `ok/generation_failed/binding_failed/postcheck_failed/timeout/not_run`；期望分母恒为六段。只有相同配置、同一注册身份的重试可以补充该来源；失败与旧 attempt 保留。重试不挑选质量最好的产物，消费规则沿用最新 OK 产物并保存映射。

### 7.3 比较入口的最小扩展

在 `scripts/omniretarget_teacher_comparison.py` 增加成组参数 `--generated-dir` 与 `--source-record`，与旧 `--u0-dir` 互斥。默认模式维持现有 P/U0 输出；新模式输出 P/UR、独立实验身份与裁决，不能把 JSON 中的 UR 仍标为 U0。

继续复用 `TeacherQualityReplayer` 和 `evaluate_local_preferred`。原函数内部的参数名称不决定来源身份，报告必须使用注册的实际来源；必要时为裁决函数增加默认 U0 的来源名称参数，仅调整字段/原因的来源标识，不改质量计算。入口为 UR 加配置/注册对账腿；其他核验沿用已有实现，包括真实消费范围、缺段输入、冻结资产、环境三元、产物链、绑定和证据有限性。

配置、注册或来源混入不一致时强制 incomplete。质量比较保持原 g1–g5：逐帧任务 MPJPE；验收器自适应细分/SLERP 最大穿透；逐 sole site 最坏脚指标；双手公共分母与有效覆盖；漂移、新增硬失败、关键帧穿透比例退化。

所有预期演示保留行，零双手分母与缺测量分开，计数必须是非负整数、双方一致。UR 报告写新的 `cmp/attempt_NNN/teacher_selection.json`；原权威报告保持原路径、原哈希、原裁决。

### 7.4 裁决与后续资格

| 独立 UR 结果 | 本阶段出口 | 后续工作的条件 |
| --- | --- | --- |
| 仍缺任一有效产物/绑定/可判定证据 | 新实验 incomplete，登记缺口后停止 | 不能直接启动 W1 或生产 |
| 六段证据完整，质量门不满足 | 新实验 no_quality_gain，停止当前候选 | 可另登记 W1 方案及其受限动作用途；仍需 A3/B/C |
| 六段完整且所有优先门满足 | 新实验 local_preferred | 另登记将 UR 作为实际来源的 A3 生产方案；不自动扩量 |
| 身份/环境/接口不成立 | not_ready 或 incomplete，分项登记 | 先修具体工程问题，不作质量否定 |

原 A2 的 U0 始终保留历史 `incomplete`。若未来用 UR 的完整裁决作为训练入口，须在对应执行方案中明确引用新实验、注册身份和替代入口依据；不能通过修改原报告“完成 A2”。

W1 仍为 `weak_motion_only`、接触损失为零、每模型最多 4k，不自动续至 8k/12k。本文不实现 W1 缓存、数据集、损失或训练入口，也不把一个训练提案当作训练就绪。

## 8. 文件与接口清单

下表为**拟实施**的改动，当前仅有本方案和文档引用实际写入。

| 文件 | 改动与职责 | 实施顺序 |
| --- | --- | --- |
| `src/ireset/eval/a2_closure.py` | 证据摘要、known/新增缺口区分、原裁决与后续状态分离 | C0/C1 |
| `scripts/omniretarget_a2_closure.py` | 调用既有完整核验与比较，写停线/收口记录 | C0/C1 |
| `scripts/diagnose_omniretarget_sub12.py` | 上游子进程包装、失败快照、唯一 Phase I、诊断结果登记 | D0 |
| `scripts/generate_omniretarget_teachers.py` | 条件实现 `--source-record`、配置身份、超时与来源隔离 | 仅 UR 启用后 |
| `scripts/omniretarget_teacher_comparison.py` | 条件实现 P/UR 比较入口与注册/配置对账 | 仅 UR 启用后 |
| `src/ireset/eval/teacher_comparison.py` | 条件补报告来源名称，默认 U0；指标与门计算复用原实现 | 仅 UR 启用后 |
| `src/ireset/data/a2_integrity.py` | 仅补所需的来源/注册核验；保持已有七轮不变量 | 随 UR 接口 |
| `tests/test_omniretarget_a2_closure.py` | 收口状态与真实完整性反例 | C0/C1 |
| `tests/test_omniretarget_sub12_diagnosis.py` | 包装不改变结果、原失败状态、Phase I 分类与预算 | D0 |
| `tests/test_omniretarget_recovery_source.py` | 注册、单差异、来源混入、消费范围和旧模式回归 | 条件 UR |
| `docs/verification/omniretarget_a2_closure_verification.md` | 实施后逐阶段证据、实际成本、裁决与未运行项 | C2 |

不添加依赖。诊断中的 CVXPY 使用既有上游环境；稀疏序列化使用既有 SciPy。配置读写、子进程和身份登记复用成熟库与现有函数。

## 9. 运行目录与命令契约

每次使用独立 run，建议结构：

```text
derived/experiments/omniretarget_first/<date>_<revision>_a2_closure/
  baseline/                 基线快照、文件哈希、新鲜比较 attempt
  diagnosis/
    attempt_001/            sub16 对照、sub12 诊断一
    attempt_002/            sub12 诊断二
    attempts_index.json     用途、进度、失败、预算；不与教师 index 混用
  recovery/                 仅登记候选后创建
    preregistration.json
    generation/attempt_NNN/
    generation/attempts_index.json
    cmp/attempt_NNN/teacher_selection.json
  decisions/
    attempt_NNN/a2_closure.json
    attempt_NNN/a2_closure.md
    attempts_index.json
  costs.json
```

目录守卫复用 `guard_output_dir`，resolve 后必须在 `derived/` 内；输出路径绝对化，禁止借 OMOMO 软链接写源数据。记录使用 `path_for_record`，解释器路径保持 venv 上下文。每次调用分配新 attempt，崩溃半写目录保留，旧裁决/快照不覆盖。

拟议入口命令形态如下，**接口实现并验证后才能使用**：

```bash
uv run --locked python scripts/omniretarget_a2_closure.py \
  --freeze derived/experiments/omniretarget_first/20261009_13906c1_a1/freeze \
  --selection derived/experiments/omniretarget_first/20261010_cec0287_a2_reviewfix7/teacher_pilot/cmp/attempt_001/teacher_selection.json \
  --u0-dir derived/experiments/omniretarget_first/20261009_123bc6f_a2_reviewfix/teacher_pilot/u0 \
  --out derived/experiments/omniretarget_first/<run_id>

uv run --locked python scripts/diagnose_omniretarget_sub12.py \
  --baseline-evidence derived/experiments/omniretarget_first/<run_id>/baseline/baseline_evidence.json \
  --out derived/experiments/omniretarget_first/<run_id>/diagnosis
```

收口入口后续可带 `--diagnosis` 或 `--recovery-report` 汇入已存在、哈希核验通过的新证据；它不替调用者启动诊断、恢复或训练。UR 的注册工具可放在诊断脚本的独立注册子命令中，不另建通用实验平台。

子进程使用参数列表、不用 shell 拼接。超时终止对应进程组并回收，保存部分日志和明确的 timeout 行；不能留下后台求解后又重新分配同一尝试。

## 10. 必须验证的反例与真实检查

### 10.1 C0/C1

- 权威报告/冻结文件/产物被改写，或现行输入与冻结不符：新增缺口不能变成正常停线登记。
- 只有预期 sub12 缺产物：可正常登记，但原裁决保持 incomplete、训练保持 not_ready。
- 把停线登记的 exit 0 或 `registration_status` 当作来源比较通过：报告接口必须显式保留不同状态。
- 删除某段行、失败 attempt 或令分母变五：拒绝完整收口声明。
- 二次调用和半写目录：旧证据字节不变，新 attempt 可定位。

### 10.2 D0

- 包装成功调用返回不同数组或改了输入：对照失败，禁止归因。
- 历史进度条数字与实际零基帧不同：报告使用捕获帧，不解析日志尾部猜帧号。
- Phase I 修改了原异常状态、其值回流生成、诊断前缀进入教师 index：拒绝。
- 已知小型锥问题覆盖 rho 在 0.2 内、0.2–0.3 间、0.3 外以及无可行解；非有限值/残差超界/OPTIMAL_INACCURATE 不能登记候选。
- 解释器缺失、探针超时、子进程超时、接口签名漂移：落盘明确状态，不崩溃丢报告。

### 10.3 条件 UR

- UR 生成前未注册、注册后改配置、两个优化字段变化、混入 U0 attempt：比较强制 incomplete。
- 六段同配置生成、相同配置重试和分目录消费：保持全部既有输入腿与全分母核验。
- 部分恢复成功、缺产物或缺质量字段：不改判 no_quality_gain，不直接打开 W1。
- 完整新候选 no_quality_gain/local_preferred：新实验可正确登记，原报告哈希与裁决不变。
- 旧 CLI 无新参数时，真实 P/U0 比较的度量、门和 admission 与基线一致。

先运行相关单元/CLI 反例，再执行 C0、成功包装对照、两次 sub12 和条件 UR 的真实检查。影响共享 A2 入口时运行现有 A2 七轮测试与全量测试；检查通过后不为同一文档或结果反复扩大测试。未来验证记录分别注明实际测试结果与历史登记值。

## 11. 预算与停止规则

预算是执行上限，不是耗时预测；接口开发工时与测试耗时另记。

| 阶段 | 运行上限 | 停止与交付 |
| --- | --- | --- |
| C0/C1/C2 的复验与登记 | 合计 30 min | 新增完整性问题先修；预算不足登记剩余项 |
| D0 成功对照、两次 sub12、Phase I | 合计 60 min；每次生成 ≤15 min；每个 Phase I ≤60 s | 无证据或超时停止诊断，不继续搜参数 |
| 条件 UR 六段、同配置重试与比较 | 合计 3 h；单次生成 ≤30 min，预留终检/比较时间 | 未完成段保留 not_run/timeout，裁决 incomplete |
| GPU 训练 | 0 | 本阶段不创建训练作业 |
| 新产物存储 | 起步上限 2 GiB | 先查空间，优先保存失败必要上下文；不删除旧证据释放预算 |

CPU 默认串行生成、每进程数值库线程受限；先沿用已验收设置，不做并发或后端优化实验。UR 若重试，不能突破总 3 h 或改变配置；原失败 attempt、诊断与重试的全部耗时入册。

发生身份漂移、包装不等价、环境故障或候选条件不成立即停止对应分支。没有候选、恢复仍失败、质量无改善均可结束本阶段；结束必须保留真实缺口及训练未就绪状态。

## 12. C2 交付与进入下一阶段的条件

交付内容：

1. 原 U0 的权威证据索引、新鲜完整性复验、六段分母和停线登记。
2. D0 的实际失败帧/迭代、原问题快照、两次重现、Phase I 分类；未执行或失败项明确标记。
3. 若启用 UR，事前注册、六段全结果/成本、独立 teacher_selection；否则记录未启用理由。
4. 一份收口报告，将“原 A2 状态”“新实验状态”“来源用途”“下一方案”“训练就绪”分别列出。
5. 实施验证记录与 AGENTS 摘要，引用具体产物和完整哈希；不将计划状态登记为已完成。

阶段完成判据是上述适用证据可核验、状态与分母无矛盾，并给出一个明确后续方向。它不要求得到成功恢复或可训练教师。

若 UR 完整比较得到 no_quality_gain，可编写 W1 的 A3/B/C 执行方案，明确发布数据仍有缺陷、弱监督的信息价值、train24/val31 基础审计、独立缓存/统计、关闭接触损失和数值烟测。通过这些门后才进入最多 4k 的两模型诊断。

若 UR 得到 local_preferred，先编写并冻结该来源的 A3 生产配置与准入清单，再按父方案执行。若仍 incomplete/not_ready 或没有恢复候选，保持停线，针对已有证据另立教师质量或条件侧诊断方案；不通过放宽原裁决、删除 sub12 或默认启用自研修复取得训练入口。

## 13. 实施检查表

- [ ] C0：报告锚点、冻结输入/资产、环境、五段产物链和绑定复验；仅已知 sub12 缺口。
- [ ] C1：原 U0 停线登记落盘；原 verdict、全分母和生产/训练限制保留。
- [ ] D0：真实成功对照证明包装等价；两次 sub12 捕获真实失败帧和原问题。
- [ ] D0：唯一 Phase I 的状态、rho、残差与局限入册。
- [ ] G0：明确结束，或在新质量读取前冻结唯一 UR 注册。
- [ ] UR（条件）：同配置六段重生成，注册/配置/全部完整性腿核验，独立质量裁决。
- [ ] C2：成本、缺口、后续方向、验证文档及项目进度登记一致。

编写本文时以上实施项均未开始；第一步实际实施应是 C0/C1 的最小收口入口与反例验证。
