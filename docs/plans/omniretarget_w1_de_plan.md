# OmniRetarget W1 正式训练与裸输出评估执行方案

日期：2026-10-13。版本：v0.1。状态：**方案已起草；本方案的执行接口加固、运行注册、D 正式训练、E 裸输出比较和 F 路线登记均未实施，运行状态为 not_run。** 已完成的前置阶段是 W1 A3/B/C，第七轮审阅后的权威链 ready_for_w1_d=true；这份文档不作为 D/E 已完成的证据。

父契约：[OmniRetarget 优先教师与初步训练总方案](omniretarget_first_training_plan.md) §6–§9、§12–§13；直接前置：[W1 A3/B/C 方案](omniretarget_w1_a3bc_plan.md) §5、§10 和[验证记录](../verification/omniretarget_w1_a3bc_verification.md) §5R7。父方案和 A3/B/C 方案的历史字节保持原状；本文件具体化正式执行及交付要求。

## 1. 当前事实、本轮问题与范围

### 1.1 已验收的入口

权威来源根（下文记为 SOURCE）：

```text
derived/experiments/omniretarget_first/
  20261012_21e5784_w1_a3bc_reviewfix8/w1_reviewfix8/
```

实现 revision 为 21e5784c429892115b9cc05cadac667d3de31583，登记提交为 4dfc13ae5810c0ca37870cdf6d81e56e27172fc3。SOURCE 的五门已独立复验通过，unmet 为空：

| 前置事实 | 已有结果及用途 |
| --- | --- |
| 来源裁决 | 原 U0 保持 incomplete；独立 UR 为完整 no_quality_gain。W1 使用发布 original 标签，用途 weak_motion_only |
| A3 审计 | train24+val31=55/55 基础准入，质量证据完整；已知 strict 缺陷保留登记 |
| 条件语义 | 源箱与目标箱分离；源箱几何应用登记相似尺度；55 份日程均经消费侧复验 |
| train 统计 | 仅 train24 原始帧，共 4,351 帧；完整训练窗口 110，val 窗口 145 |
| S0 | 两模型各 1,000 步；注册六段首窗的小集合拟合通过，checkpoint 重建和固定 probe 重放通过 |
| 工程就绪 | registration_complete、a3_complete、b_complete、c_complete、ready_for_w1_d 均为 true |

S0 的六窗口结果只提供数值和有限拟合证据。D 使用完整 train24 准入窗口；正式两模型从头初始化，S0 checkpoint 作为前置证据保存。

### 1.2 本轮回答的问题

1. 在有限 train24 和相同条件下，两模型能否在 val31 上生成完整动作，教师拟合如何变化？
2. 相比 Teacher raw，模型新增或复现哪些穿透、滑移、手部脱离、累计漂移及拼接问题？
3. 初步瓶颈更符合学习不足、条件/实现问题，还是已知动作标签缺陷？现有证据是否足以支持下一项实验？

本轮交付动作学习基线及其质量边界。整段机器人检查、人体任务工作门和持续双手交互证据分别报告；不以拟合误差下降替代交互验收。

### 1.3 执行范围

实施正式运行所缺的预算、台账、checkpoint 选择、完整产物和报告接口；执行 D、E 及 F 收口。训练来源保持 published original、训练用途保持 weak_motion_only、接触监督权重保持 0。评估三种方法：Teacher raw、Deterministic raw、Diffusion raw。

本轮不包含新的教师生成、扩大训练名单、rot/trans 增强、接触损失调参、额外平滑/投影、修复器调用或 test31 最终测试。这些事项需要依据本轮证据另立实验。

## 2. 前置锚点与新运行注册

### 2.1 已实测的固定锚点

以下路径相对 SOURCE，SHA-256 为起草时对实际文件重新计算的完整 64 位摘要：

| 文件 | SHA-256 |
| --- | --- |
| freeze/preregistration.json | 071c49d32bf822e00ee0854a86e5abf25a0df9c22f033b93732a26b8753301f5 |
| audit/attempt_001/training_manifest.json | a350a8b8922ca94308de3d97787e60bf8f6a51acff6f20af9f513fd42b10e525 |
| audit/attempts_index.json | 55ce32790ed32bddb13aa9bd0a09f628763ce5fec737b4a2de15c369d01bb5f6 |
| cache/attempt_001/cache_identity.json | 90af39f123c4dc295e8f16b07b0002602c173e8d68d33ae1abe23ed611cd2e71 |
| cache/attempt_001/root_stats.npz | b9d9cc049a6b58eea7ab048b886a5fd9f87755690d1f0a789cb4f113c5bd2eda |
| cache/attempts_index.json | 88e40985f12aca63ce4ac0ecb22884b23bd42fb106b812d687d23484c6639b3b |
| decisions/attempt_001/w1_readiness.json | 7245ced433bba50a66c6dcd175774866eb5c2062ef6970b508bd61d453c3b558 |
| costs.json | 43573f1bbe47a3199d0a238a0422cd0e4581b0840f47b4b59e01b00daa341aee |
| s0/deterministic/attempt_001/s0_summary.json | 50a23e3c29dd614a7e03fe2abf590930c13b2daf5429ae7dbbb9f4dd5a86e06b |
| s0/deterministic/attempt_001/last.pt | 716bc138f5f712c329c6c10674929f4cfa3e27c8c7f12d187e23a38c8c96a3aa |
| s0/deterministic/attempt_001/probe_replay.pt | 62602cc73c2a5c0c002453ee9846815d89c1e9709429dc0ab4b1a68ec239deed |
| s0/diffusion/attempt_001/s0_summary.json | 93e2298eb0f7f2e10e00d7e6cce404b1d924e5c472e8a781cbcb066a11b5e107 |
| s0/diffusion/attempt_001/last.pt | c583dbcb4cfbc5f7b9d19cfa4f33424f25aca37476c23bc86e263928616efc26 |
| s0/diffusion/attempt_001/probe_replay.pt | 142f68772b090d9900445d8d15ff9ca20b945a8af3b450cb754352b5d9b5c842 |

项目根下的协议和配置锚点：

| 文件 | SHA-256 |
| --- | --- |
| docs/plans/omniretarget_first_training_plan.md | ca7cd4d8db553cac0057d1392c3238f0ca0fbfcbe4748f4c5946a3e650da2e84 |
| docs/plans/omniretarget_a2_closure_plan.md | 28fd40e43af0f524e5c2c23dc776a5a9b1b00bd872d75ee90442b5f41a8b81b3 |
| docs/plans/omniretarget_w1_a3bc_plan.md | e7247e4b76d7cdfec04e288883dbdaccd868c7908108a8f2f5171839c766fc2f |
| configs/train/omniretarget_motion.yaml | 0e2cfa96849ee2de14ac84b7343289090c606f78116bb79114ff81b046b521e0 |
| configs/labels/omniretarget_motion_v1.yaml | 4672b3b97dd3cf0895d205171c1ba3a677a54edfefde1db21bfd3722eb086ce9 |
| uv.lock | f809284e21406f7f3b51c33bf200f0e6aa567f32cce43bc15ab00797b6d69a44 |

正式注册读取这些具体文件，不按“最新 attempt”自动迁移来源。进一步展开 SOURCE 注册中的九个 A1/A2/分组锚点、55 段输入、派生表示、120 项资产闭包和全部日程逐项复验。源文件、软链接目标、资产及历史产物只读。

### 2.2 R0：入口复验

复用 [w1_readiness.py](../../scripts/w1_readiness.py) 的 check_registration、check_a3、check_b、check_c，及 [w1_training.py](../../src/ireset/data/w1_training.py) 的统一准入和缓存核验。旧报告的 true 标记与本次事实重算都入册；只有事实重算通过才满足执行前置。

必须检查：

1. C2/UR 产物链和裁决重推成立，U0/UR 历史状态不变。
2. 名单精确一致、无重复、train24 与 val31 无交集；每个 manifest demo 恰一行且 split 正确。
3. 实际消费路径上的 NPZ/PT、derived 表示和资产仍与冻结一致。
4. 日程和 root_stats 内容、代码身份及盘上 identity×索引×台账状态一致；running/crashed/显式 incomplete 不可消费。
5. 两数据集真实构建、实际 micro-batch 可取；两模型 S0 原始损失、完整 probe 日程、checkpoint 深度重放和六段输入绑定均通过。
6. 当前 Python 3.12、uv.lock、实际依赖版本和 CUDA 前向/反向环境被记录。

SOURCE 的 condition_code_sha256 为 d73f411d874614d4b6b23f4686b4cd0eaa1d42b6aa33110dfa1facc9c2aebb8d。拟修改的运行编排、Trainer 和评估文件应保持既有条件语义。若实际修改了 CONDITION_CODE_FILES，须在新目录按新身份重建缓存并重新验证 B；不得改写 SOURCE。若改动影响模型、损失、初始化或训练数值行为，须先重新取得 C 证据，再修订入口登记。纯运行编排变更在现行核验下通过即可复用已验收的 A3/B/C 产物，不把历史训练重新标为新运行。

### 2.3 R1：正式运行注册

拟用执行身份 w1_de_v1；标签协议和原数据实验身份仍为 omniretarget_motion_v1 / w1_published_motion_v1。新增执行身份与训练用途分别存储。

实现与必要检查通过、执行代码固定后，在第一次正式模型初始化或 val 生成前，以独占创建方式写 freeze/execution_registration.json。记录：

- 本方案最终字节摘要、实现 revision、dirty/status/diff 与未跟踪文件内容身份、uv.lock 及实际环境；
- SOURCE 的具体锚点、复验报告、完整名单和输入/派生/资产/日程哈希；
- 两模型实际生效配置、验证日程、采样种子、数值重采样规则、checkpoint 选择和加载规则；
- 独立检查实际配置、排除对、任务工作门、指标必需字段和聚合口径；
- §8 预算、调用历史、停止条件、输出结构、预期矩阵和阶段完成规则。

已有注册只允许逐字段一致时复用；不覆盖、不借更换运行根重置成本。文档草案可在注册前修订；注册后修改规则、数据、配置或代码须保留旧证据并建立新版本。登记方案不会启动训练。

## 3. 固定数据、模型与配置

### 3.1 数据集合

train24 复用 teacher_production_train24_v1，其规范顺序数组 SHA-256 为 8fbfd008b98f56196d31f81352582006ba95c16bdcf2a4da2f5da56eb8267948。选择规则、24 个 demo 及顺序直接复用 SOURCE 注册，正式运行仍重推核验。

val31 复用 val_originals_v1，全部 31 个 original、完整长度。val6 复用 val6_pilot_v1：

```text
sub10_largebox_100  sub12_largebox_008  sub16_largebox_052
sub3_largebox_018   sub7_largebox_032   sub8_largebox_001
```

SOURCE 当前包含 train 24 段/110 窗口、val 31 段/145 窗口，完整 55 段共 10,032 个独立帧。窗口数、被使用的样本数和原始演示数分列。train 使用 shuffle 和 drop_last=True 的既有完整窗口加载；每周期的真实消费/丢弃数登记，尾批不会被当成一个完整 micro-batch。val 不截断、不 drop_last，全部 145 窗口参与损失验证。集合或计数变化时解释并停止入口，而非自动接受更小数据。

推理条件只来自人体参考、源箱、目标任务与冻结参考日程。源箱几何使用已登记 similarity_scale；教师根/关节/FK、机器人真值脚落点及标签质量字段不进入条件。源箱和目标箱的 +z 分离保持原证据。统计只消费已验收 root_stats，不从 val 重拟合。

### 3.2 训练与采样参数

| 参数 | 固定值 |
| --- | --- |
| 执行顺序 | Deterministic 从头初始化 → Diffusion 从头初始化；单 GPU 串行 |
| 每模型上限 | 4,000 个真实 optimizer steps；不进入 8k/12k 续行 |
| 主干 | d_model=240、4 层、6 头、FFN×4、时间窗口 31 |
| 状态 | 根位置 3 + 根旋转 6D + 29 关节；解码为根位姿+29 关节 |
| 动作窗/步长 | 60/30 帧；整段生成补 T-60 尾窗，覆盖最后一帧 |
| batch | micro_batch=4、grad_accum=4、有效 batch=16；FP32 |
| 优化器 | AdamW，lr=1e-4、weight_decay=1e-5、grad_clip=1.0 |
| 损失 | pos/rot6d/joint=1，geodesic=0.5，FK=2，vel=0.5，contact=0 |
| 随机/加载 | seed=20261008、val_seed=20261008；workers=4，每 worker 原生线程 1 |
| 扩散训练 | 1,000 diffusion timesteps，sample/x0-prediction，clip_sample=False |
| beta 参数 | scaled_linear、beta_start=1e-4、beta_end=0.02；实际 scheduler 参数入册 |
| 主生成 | DDIM25、eta=0、无 guidance；两模型完整生成主 seed=7 |
| 日程 | loss 每 250 optimizer steps；完整 val31 生成每 1,000 步 |

根/旋转不裁剪，既有干净关节映射与旋转/拼接解码按冻结实现执行。vel 项为标准化状态差分拟合；它不提供物理速度保证。contact=0 必须使接触监督分支调用和梯度贡献均为零。

## 4. 实施项与依赖顺序

以下是起草时实读代码得到的接口状态和计划要求，不能解释为本轮已完成的修复：

| 实施项 | 当前状态 | 必须补齐的行为 |
| --- | --- | --- |
| I0 执行注册和驱动 | A3/B/C 注册与准入可复用；尚无 D/E 统一阶段驱动 | 新增薄入口，绑定 SOURCE 和 execution_registration，固定阶段顺序、预算域和实际 argv |
| I1 正式 Trainer | 已接完整 val31 宏平均选模；训练循环无累计截止/事前成本登记 | 剩余预算约束、全退出口登记、数值失败立即停止、4k 计数门、日程和 best 重推 |
| I2 checkpoint 消费 | raw 入口固定读 models_dir/model/best.pt；只核对部分身份 | 显式逐模型 checkpoint 路径和选择来源；完整配置/权重/统计/资产/代码身份及可重建性核验 |
| I3 E 完整产物 | raw 入口可全长检查；只产模型行，未保存完整 raw 轨迹与 Teacher raw 主行 | 三方法×固定名单逐行产物、独立质量、任务/接触/漂移/成本、逐方法分母及多采样附表 |
| I4 F 收口和详细分析 | 指标及回放工具已有部分能力；尚无 D/E 证据收口 | val6 复用、产物重放、完整性/负结果/路线判断分别登记 |

### 4.1 修改边界

优先扩展现有文件，复用成熟依赖：

- [train_generation.py](../../scripts/train_generation.py)：正式执行注册入口、生效配置对账、禁止 debug 子集进入正式模式、实际路径入册、输出隔离；
- [trainer.py](../../src/ireset/train/trainer.py)：deadline/finite 检查、阶段状态、验证日程、完整记录和保存；已有训练/验证 RNG 隔离及宏平均选模继续成立；
- [eval_generation.py](../../scripts/eval_generation.py)：扩展 W1 raw 专用路径，不走历史修复矩阵；保存 Teacher raw 和模型完整轨迹、统一 checkpoint 加载、固定矩阵和逐行失败；
- [generator.py](../../src/ireset/model/generator.py)：限定数值重试、逐窗有效性和真实调用数；正常首尝试成功路径应维持数值行为；
- [teacher_comparison.py](../../src/ireset/eval/teacher_comparison.py) 的 TeacherQualityReplayer、[IndependentValidator](../../src/ireset/repair/validation.py)：复用度量和检查口径，通过薄适配测量模型轨迹；
- 拟新增 scripts/omniretarget_w1_de.py 和最小 D/E 记录辅助模块，承接注册、成本、阶段核验和汇总；不重建通用实验框架。

新增预算参数优先属于执行注册/驱动，模型训练 YAML 维持 §3.2 固定配置。需要扩展 TrainConfig 时明确执行字段与模型参数，避免参数只入册不生效。旧无清单入口的历史契约保持；只有绑定本轮执行注册的调用可登记为 W1-D/E。

### 4.2 正式运行前的真实接口验收

未知键、缺参、错误名单、debug max_files/max_windows、步数越界、来源/缓存身份漂移须在正式调用前拒绝。config.model_type 的唯一逐模型覆盖和其余实际生效字段分别对账，CLI 字符串存在不能作为生效证据。

用可控小 fixture 实际执行 TrainConfig→模型→Trainer→完整生成回调→checkpoint→raw 评估路径，覆盖真实参数解析、末帧、缺行和无 best。fixture 的 shortened cadence/名单不得进入正式注册。正式 first100 steps 属于对应模型的 4k 上限和同一预算，不额外安排重复 GPU 烟测。首次正式完整 val 回调发生在第 1,000 步。

## 5. D：正式训练、验证与选模

### 5.1 D1 确定性模型

从 §3 的固定配置初始化，在完整 train24 窗口上训练，完成最多 4,000 optimizer steps。记录每个 optimizer step 的 micro-batch 数、有效样本数、全部损失项、梯度范数、学习率和同步后耗时。

在 backward/optimizer 更新前后检查必需张量、损失、梯度及参数有限性；非有限、旋转解码退化、OOM、身份变化或预算停止均保存已有证据并退出。数值问题不得通过删除 FK/时序损失或改变配置隐式绕过。

每 250 步对全 val 窗口做 loss 验证，按有效样本/帧权重登记，完整运行有 16 次记录。每 1,000 步对完整 val31 生成，完整运行有 step=1000/2000/3000/4000 四次记录，每次恰 31 行。窗口损失、完整生成指标和独立交互检查用途分开。

### 5.2 D2 扩散模型

D1 终局已登记、成本可知且剩余预算足够启动后，以同名单、同动作损失、同主干规模和有效 batch 从头训练。D1 的负质量结果不会自动取消 D2；D1 的公共输入/实现完整性失败或未知历史成本会阻塞 D2。

loss 验证的 t/noise 来自专用固定生成器；完整生成 seed=7 的采样流也独立。验证前后训练 RNG、DataLoader 顺序及模型 train/eval 状态恢复有反例核验。时钟/验证改动不许悄悄重置训练流。

### 5.3 best 与 last

每次 formal 回调保存 demo 身份、T、numerical status、teacher-fit FK MPJPE 和失败原因。expected 永远精确等于注册 val31。只有 31 行全部存在、身份正确且指标有限时，计算演示宏平均：

```text
macro = sum(demo_mean_teacher_fit_FK_MPJPE) / 31
best = 宏平均严格下降的完整验证 checkpoint；同分保留较早一步
```

父方案要求的 best 由教师拟合选取，不能用 contact/penetration 或质量通过率选模。末次 incomplete 不撤销之前已取得的合法 best，但须保留末次缺口；不得用缩分母均值覆盖它。

last 保存实际已完成步数。若没有任何完整选模证据，selection_status=no_qualified_best，不创建或冒称 best；E 按预注册规则选择 last 做诊断，report 明确 checkpoint_role=last_diagnostic。last 也不可加载时，该模型所有预期行保留 checkpoint_unavailable，默认比较记 incomplete。

保存 best/last、loss_history、val_history、formal_history 和 checkpoint_index，按哈希绑定 step/config/selection。正式从头训练不加载 S0 或历史权重。本轮默认不实现精确 resume；进程中止后保留结果并停止。若后续需要恢复，先另验 optimizer、CPU/CUDA/NumPy RNG、loader generator/数据顺序和步数；仅模型权重不能拼接成一次完整运行。重试不重置累计预算，warm_start 另立身份。

## 6. E：完整裸输出矩阵与独立检查

### 6.1 checkpoint 冻结与加载

D 阶段完成或受控终止后，从 formal_history 和盘上 checkpoint 重新计算 §5.3 选择，独占写 evaluation/checkpoint_selection.json。逐模型显式记录路径、SHA-256、step、best/last 角色和原因；评估器消费记录，不从文件夹名称或 glob 猜测权重。

加载时 strict 重建实际模型/conditioner，完整使用 num_train_timesteps、beta_schedule/start/end 和解码配置。逐项核对执行注册、training_manifest、缓存 identity/root_stats、源协议、资产、条件代码、模型类型及所有权重有限性。checkpoint 内嵌 root_stats 与本轮真实统计逐位相等；加载/形状/类型错误登记原因，不能退回其他模型。

### 6.2 固定矩阵与逐行产物

主比较在完整 val31 上执行：

| 方法 | 主 seed | 预期行数 |
| --- | --- | --- |
| Teacher raw | 不适用 | 31 |
| Deterministic raw | 7 | 31 |
| Diffusion raw | 7 | 31 |

主矩阵为 93 个 method×demo 键。标签/条件加载失败时也为每个方法保留一行，不能用一条“demo 失败”代替三条方法行。模型生成 failure、timeout、未运行或 checkpoint 缺失保持原分母。

为具体落实父方案 §13 的采样变化报告，预注册扩散附表 seed=17、27，对同一完整 val31 分别执行；追加 62 行。含主 seed 在内的三次扩散结果全部保留，共 155 个不同 method×demo×seed 主/附键。附表不参与 best 选择或代替主 seed；按 seed 报全 val 宏平均、通过数和变化范围，不挑最优样本。所有附表调用计入 E 的 2 小时预算，未完成保留缺行并将附表/本轮 E 完整性分别说明。

每个成功生成的完整 raw 轨迹在新 attempt 内保存，至少包括根 [T,7]、关节 [T,29]、目标箱 [T,7]、frame_dt/fps、窗口起点、覆盖最后一帧的事实及实际重试信息。组合 scene [T,43] 使用项目 [pos,quat] 约定。产物列表记完整内容摘要和对应条件/模型/输入身份。

Teacher raw 读取冻结派生表示或发布原始标签的显式转换，保存相同布局的回放产物。A3 质量可作前置参照；最终比较按同一检查配置实测/复核，不能用旧配置的质量摘要顶替本轮逐行证据。

### 6.3 数值有效性与有限重采样

标准解码、既有干净关节映射和窗口 SLERP 拼接属于冻结生成接口。融合后检查 T、全部张量有限性、四元数范数/旋转有效性、目标箱逐帧绑定、窗口覆盖及末帧。

扩散每个无效窗口最多 3 次数值尝试，种子为主/附 seed+attempt_index；只重试非有限或退化旋转，已有效窗口保持原值。有效性更新按当前未恢复窗口进行，不累计“曾经坏过”的永久失败标记；实际模型调用数、每次无效窗口、最终有效窗口和 seed 全部入册。确定性无效输出直接数值拒绝，重复相同前向不视为恢复。

重采样不得读取碰撞、任务、接触或拟合分数作选择。超出次数仍无效时保留失败事实，numerical_valid=false；不能默认修补后宣称 raw。既有 metadata 中配置上限不能替代实际调用数。

### 6.4 独立检查与任务门

注册时实际实例化 load_mode_configs("strict") 并保存完整 ValidationConfig、参考接触推断实参、几何资产/允许对/排除对；表名或 YAML 路径不能替代实际生效值。复用自适应插值、SLERP、逐 site 最坏脚速/漂移及逐区间手部认证口径。

按父方案固定诊断任务工作门 task_mpjpe_tol_m=0.40，使用逐帧人体参考计算世界/目标箱系 MPJPE，单列 mean/p95/max。沿用现行逐帧任务门语义：每帧对应关键点平均误差均≤0.40m，等价于整段 max≤0.40m；A2 来源比较的 p95 指标仍作诊断量，不用于豁免超界帧。该门从 D/E 首次运行前冻结；它仅提供诊断任务检查，不替代主项目最终搬运门。

每条可测轨迹登记 numerical_valid、robot_strict_pass、task_gate_pass、raw_standard_pass，后者为前三者的合取。完整 val 的 Teacher raw 与模型 raw 使用相同规则。

机器人检查中沿用已登记 box_ground 排除；同时实测固定输入箱体与地面冲突的深度、比例和位置，声明排除范围。不移动箱体，不移除其他碰撞族，不改变插值/限位/速度/接触容差。模型加载、拼接或测量失败分别记录，不能把未知检查值变成通过。

双手同时抓取的参考分母从冻结源场景日程左右 grasp 区间交集复算，必须是有限非负整数。三方法共享同一参考；左右参考计数、区间、交集和实际测量逐项对账。有参考帧而测量缺失为 gap；确无参考帧时相关条件指标可为 null，保留明确零分母语义。

### 6.5 指标、完整性与聚合

| 指标族 | 每个演示必须保存的证据 |
| --- | --- |
| 拟合 | root RMSE、根旋转误差、关节 MAE、teacher-fit FK MPJPE；对 Teacher raw 的自身拟合可标 self-reference |
| 人体任务 | 逐帧对齐的世界/目标箱系 MPJPE，mean/p95、工作门及失败位置 |
| 几何/运动 | validator failed_checks/位置、插值最大穿透、关键帧违规比例、限位、速度、脚间隙/漂移、固定 box_ground 冲突 |
| 手部 | 左右参考/近接/认证/认证∩参考帧数、最长连续交集、相对速度与累计锚漂；近接不冒充认证 |
| 双手 | 共享参考分母、同时认证交集、最长连续交集、有效覆盖；无区间时保留证据不足事实 |
| 拼接 | 起点/末帧覆盖、边界位置/旋转/关节差分、段内速度参照、窗口和整段累计漂移 |
| 接触 F1 | 各侧值及 defined 演示数，未定义时 null/原因 |
| 成本 | 条件读取/构造、全部生成调用、解码/拼接、独立检查、失败/重试、保存登记、整段总耗时 |

宏平均以原始演示为单位。完整总体指标要求全部 31 个相应测量存在且有效；失败造成不可测值时总体宏平均为 null，并列 n_defined、expected=31 和仅已定义子集的明确诊断值。零参考的条件指标按预注册 defined 口径汇总，仍保存全部演示行。不能把“成功样本均值”写成完整总体均值。

各方法、各 seed 分别汇总 expected、attempted、processed、numerical_rejected、strict_rejected、task_rejected、timeout、error、not_run、raw_standard_pass。实际处理集合精确对账，无重复键、无缺行、无分母缩水。

raw_standard_pass 不宣称持续双手搬运或最终任务成功。另报告双手持续证据和主方案未满足项；检查零通过时仍交付 0/31。生成数值拒绝和质量拒绝是可交付负结果；输入/产物不可验证、缺默认方法、预算中止或未处理行导致比较 incomplete。标签缺失使 teacher-fit 不完整，即使保留条件任务也不能得到完整 best 或完整拟合宏平均。

在线成本包含条件构建、全部去噪、拼接、数值重试和检查；GPU 计时在相应边界同步，冷/热缓存和预热另列。Teacher raw 回放/检查成本与教师原生成成本区分；没有同条件上游完整生成成本时不宣称相对 OmniRetarget 的端到端加速。

## 7. val6 详细分析与 F 收口

### 7.1 固定详细集合

复用主 seed 的 val6×三方法=18 条完整产物，按哈希核验后输出逐帧曲线：任务/拟合、手部间隙/认证交集、脚部支撑/漂移、穿透、速度、拼接边界、首次失败位置。复用主矩阵输出，详细阶段不重新生成或按质量选 seed。

回放保存人体参考、源/目标箱关系、Teacher raw 和两个模型 raw。为控制存储，汇总视频从固定 val6 按 demo 字典序选择一个 raw_standard_pass 和一个未通过案例；若某类不存在则只展示已有类，明确原因。逐帧表覆盖全部 val6，不以视频子集替代详细分析。

### 7.2 三种完成状态分别登记

F 的机读终局至少包含：

- training：两模型实际步数、到达 4k/提前停止原因、selection_status、best/last 身份和训练证据完整性；
- evaluation：主矩阵/多采样/val6 各自 expected、processed、完整性、未满足原因与产物哈希；
- motion_baseline：complete / failed / incomplete 及依据；
- final_interaction_method：ready / not_ready / evidence_insufficient，逐项指向主方案的接触/任务/整段证据；证据不足不得记 ready；
- next_action：登记建议及支持证据，自动执行标 false；
- 原 U0=incomplete、UR=no_quality_gain 的历史锚点与不变状态。

训练完成和比较证据完整不要求 raw_standard_pass>0。全拒可以结束这一轮诊断并登记最终可行输出方法 not_ready；工程错误、未知成本、输入漂移、缺矩阵或预算未完成记 incomplete，不改写为运动学不可行。

### 7.3 路线判断的证据要求

| 观察 | 本轮优先归因与下一项候选 |
| --- | --- |
| train 拟合尚差或 val 生成数值失败 | 学习/归一化/条件/推理实现问题；保留曲线和错误，W1 到上限收口 |
| train 拟合好而 val 明显差 | 有限数据覆盖或过拟合；登记扩大同来源的候选实验及成本 |
| Teacher raw 有缺陷、模型在同处复现且预测误差较小 | 已知标签缺陷可能成为瓶颈；建议针对性教师处理 |
| Teacher raw 较好、模型新增脱离/穿透/边界突跳 | 预测或拼接/条件问题；先定位实际实现 |
| 机器人/任务工作门通过而持续双手证据不足 | 交互能力未证实；列出缺口，不升级为搬运成功 |
| 固定箱体与地面冲突或不可达 | 场景输入/范围问题，机器人标签或修复不能替代任务设置诊断 |

发布标签已有缺陷的事实继续有效。若进一步主张它是模型残余交互错误的主要瓶颈，按父方案 §9 提供至少三个不同 train 原始演示的同族 Teacher raw/模型 raw 证据，并比较教师缺陷和预测误差量级。候选诊断例从冻结 train24 按身份顺序筛选并保留选择依据；所需追加生成与检查计入 §8 的 F 预算，不能以追加训练替代归因。三例不构成总体统计证明；证据不足时只登记候选解释。

## 8. 预算、计时、历史和异常退出

### 8.1 累计上限

| 预算域 | 上限与组成 |
| --- | --- |
| R0/R1 | 2 小时 CPU 墙钟；来源复验和注册；已有 A3/B/C 全量重跑不包含在默认任务中 |
| D | 两模型累计 GPU 占用墙钟≤43,200s；包含本轮 GPU 入口验收、加载/准备、训练、训练期间全部验证、数值失败/重试及保存登记 |
| E | 2 小时墙钟；93 主行+62 扩散附行，完整生成/检查/保存；CPU/GPU 分项并列 |
| val6+F | 1 小时墙钟；复用/曲线/回放/必要三例诊断和收口 |

历史 S0 的 2,561.3s 属于 C 的既有独立预算，保留来源台账，不计作 D 新训练步数或新费用。所有本轮失败尝试在对应累计域扣除；改变 run-root/attempt/模型名称不能洗掉同一执行注册的历史调用。

D 的 12 小时并非完成保证。先记录正式前 100 步的同步耗时/峰值显存，第一次完整 val31 后更新完成成本估计；预算不足时保留已完成状态，不能删验证、降 DDIM 步数或延长 W1。仍有预算时按原配置执行另一模型；公共完整性失败先停线。

### 8.2 预算必须参与调度

复用单调钟 perf_counter/monotonic。驱动在分配 attempt 后、启动子进程前写 running；实际父进程从启动准备到退出及登记的 elapsed 是成本锚。禁止用 epoch 时间与 monotonic 相减，逐行/汇总均为有限非负数，bool 不作成本。

每个域剩余=注册上限−已核实历史实际费用。历史不可读、结构残缺、scope/索引/台账/盘上 attempt 不一致或 running 残留即 unknown，停止启动；有证据的零可以合法保留。成本汇总与逐行测量按记录精度推导舍入容差，不能用任意宽容差兜底。

D 预留 300s 给保存/结束登记，E 和 val6/F 各预留 60s；预留仍在该域上限内。剩余≤预留时不启动新 optimizer step、验证演示或回放单元。实际子调用 timeout 受 remaining−reserve 限制，父进程有独立全域 deadline；无法在剩余时间内退出则中止子进程并登记已测耗时/未完成项。

训练、验证每个 demo、解码/检查块、checkpoint 保存、内容哈希、identity/索引/台账最后登记之后均检查截止。完成最后一步或最后一行不豁免截止。任何超限都将同一 attempt 已声称 complete 的登记原位撤销为 incomplete，并记录 measured elapsed；索引不追加第二条同键行。

### 8.3 终止和记录一致性

盘上 attempt、索引和成本按 (stage, model, attempt) 一一对应。running/crashed/显式 incomplete 与完成声明矛盾时消费侧拒绝；SIGKILL 不可捕获，不能依赖生产侧 except 完成撤销。父进程仍活着时可以登记真实退出与 elapsed；父进程也失去记录能力时保留 unknown，后续不可视为零成本。

原子写临时文件并替换 identity/索引/台账，注册采用独占创建。可捕获异常保存已完成步数/行、最小摘要和 crashed 状态；撤销自身失败时保持可追溯 running/crashed 证人。消费侧核验必需覆盖生产侧的显式非完成集合。

计划默认每模型一次正式尝试。发生异常即登记并收口，修复原因和新代码版本明确后方可另起 attempt；新尝试仍受本轮预算或明确的新注册约束。自动重试只限 §6.3 的数值生成窗口。

## 9. 输出布局、入口与执行顺序

拟用新的运行根 RUN：

```text
derived/experiments/omniretarget_first/<date>_<revision>_w1_de/
  freeze/execution_registration.json  最终方案/来源/名单/配置/预算
  preflight/attempt_NNN/              当前来源事实复验及解析验收
  formal/deterministic/attempt_NNN/   best、last、损失/验证/逐行正式选择证据
  formal/diffusion/attempt_NNN/       同上
  evaluation/checkpoint_selection.json
  evaluation/raw/attempt_NNN/         固定矩阵/逐条轨迹/指标/哈希/成本
  evaluation/detail_val6/attempt_NNN/ 18 条曲线、回放与失败定位
  diagnosis/train/attempt_NNN/        可选的冻结三例归因证据
  decisions/attempt_NNN/             终局、证据缺口和下一项建议
  costs.json                         全阶段事前/事后调用成本
```

所有输出 resolve 后必须在项目 derived/ 内，并拒绝 SOURCE、其他历史实验根及只读资源。每阶段 attempt 隔离，不覆盖旧输出；实际路径使用 path_for_record 登记。索引是消费对象来源，目录存在或 glob 找到文件不构成完成证据。

以下是**拟新增驱动的 CLI 实现契约，当前入口尚不存在，不能直接作为可执行命令**。完成实现后，以真实解析测试生成准确运行命令并写入注册：

```text
uv run --locked python scripts/omniretarget_w1_de.py
  --stage register --source-run-root <SOURCE> --run-root <RUN>
  --plan docs/plans/omniretarget_w1_de_plan.md

uv run --locked python scripts/omniretarget_w1_de.py
  --stage train-deterministic --run-root <RUN>
uv run --locked python scripts/omniretarget_w1_de.py
  --stage train-diffusion --run-root <RUN>
uv run --locked python scripts/omniretarget_w1_de.py
  --stage raw-eval --run-root <RUN>
uv run --locked python scripts/omniretarget_w1_de.py
  --stage detail --run-root <RUN>
uv run --locked python scripts/omniretarget_w1_de.py
  --stage finalize --run-root <RUN>
```

阶段 register 必须先完成 R0/R1；后续阶段逐项读取注册和具体前置产物，不允许以 CLI 调用顺序代替证据前置。实际内部 train_generation/eval_generation argv、解析后的有效配置和消费路径同步记录；本方案不发布缺少预算驱动的裸训练命令。

提交顺序建议：I0 注册/成本/守卫 → I1 正式训练与选模 → I2/I3 checkpoint/raw 产物/检查 → I4 详细分析与收口 → 必要检查通过 → 固定执行提交后正式 R/D/E/F → 验证记录和导航登记。运行期间保持代码/配置/文档身份固定；登记实际结果放在执行结束后的文档提交。

## 10. 必要反例与实际验收

测试集中验证决策边界和证据链，复用当前 fixture，不为文档变更或实现的逐行镜像增加测试：

| 边界 | 有意义的反例/正向验证 |
| --- | --- |
| 入口 | true 就绪报告但输入/资产/cache 漂移；来源路径指向其他 attempt；当前 C 深度重算失败；零成本正向控制 |
| 配置 | max_files/max_windows 正式模式拒绝、steps>4000/负数、未知 formal 字段、argv 截断、YAML 值未实际生效 |
| 预算 | 前模型/失败尝试扣减；恰到上限停止；剩余数秒子调用 timeout 封顶；末次保存/台账延迟撤销完成 |
| 历史 | 缺/坏/NaN/负/bool 成本、scope 删除/重复行、目录×索引×台账不一致、真实 SIGKILL running 残行不可消费 |
| 训练 | NaN loss/gradient/weight 立即停止；step/micro-batch 不混计；验证 RNG/loader 顺序不变、模型状态恢复 |
| 选模 | val31 缺一行/重复行/缺指标/NaN 不选 best；窗口 loss 改善而完整宏变差不覆盖；同分保留较早模型；合法 best 逐位重推 |
| 加载 | beta_start/end 完整重建；外国 manifest/cache/root_stats/资产/代码、损坏/空/NaN 权重拒绝；last 明确诊断角色 |
| 数值重试 | 首次坏窗随后恢复正确接纳、有效窗不变；真实尝试数入册；次数耗尽拒；质量分数不能触发重采样 |
| 矩阵 | 共享条件加载失败仍三方法各有行；完整 93+62 键，无分母缩小；短段/缺 checkpoint/未处理保持行 |
| 指标 | 源箱尺度贯穿；人体逐帧对齐；自适应 SLERP 穿透；脚逐 site 最坏/首帧零排除；手逐区间认证 |
| 参考分母 | 左右缺键/None/分数/NaN、both 交集不符强制 gap；真实零参考合法 null；三方法共享参考一致 |
| raw 产物 | 最后非整窗覆盖；改写 NPZ 而保留哈希拒；录制错误目标箱拒；完整数值检查及零修复调用 |
| 完成语义 | 全 0/31 质量拒绝仍可完成诊断；缺方法/输入不可证/预算缺行是 incomplete；未知状态不空真通过 |
| 详细/收口 | val6 必须复用哈希绑定主产物；多采样不挑最好；修改通过标记但原始事实不支持时终局核验拒绝 |

实现后运行受影响的 W1/Trainer/Generator/评估测试及项目现有必要检查；已有约 638 项规模是起草时快照，实际结果写入验证记录。正式 GPU 运行前重查磁盘、GPU/CPU 占用、主环境和线程限制；不根据过时硬件占用登记启动。

收口至少独立重放两模型选中 checkpoint 在一个身份规则固定的 val6 演示上的完整输出，核验固定 seed 可复现及末帧；从完整逐演示指标重新计算宏平均/计数/失败族，从盘上 NPZ/哈希/配置重推通过状态。容差比较先核有限性；浮点重放使用已登记数值容差，序列化同源事实则精确对账。前置 true、summary passed、目录存在都只是待核验主张。

## 11. 交付与完成清单

实施后新增验证记录 docs/verification/omniretarget_w1_de_verification.md，链接本方案、SOURCE 与 RUN 的具体产物；更新 docs/README.md 和 AGENTS.md 的真实阶段状态。当前不创建伪验证记录。

- [ ] I0–I4 最小接口加固完成，实际参数和必要检查通过。
- [ ] R0 当前事实复验通过，R1 执行注册独占落盘。
- [ ] D 两模型实际训练/验证/成本完整登记，best/last 的来源重推成立。
- [ ] E 主矩阵 93 行和扩散附表 62 行逐键对账，完整轨迹/检查/失败产物可复核。
- [ ] val6 的 18 条主输出详细证据与回放完成，必要归因例和其范围如实登记。
- [ ] F 训练完成、比较完整性、动作基线和最终交互状态分别裁定；下一项建议指向具体证据。
- [ ] 原 U0/UR、发布标签风险和自研路线历史边界保留；所有未执行/失败/预算中止阶段使用实际状态。

本轮完成标准是有限训练及固定集合的可信诊断交付。预算内负结果可以收口；缺证据需要明确 incomplete。后续训练扩展、教师处理或可行性方法投入，以这份交付支持的新实验为依据。
