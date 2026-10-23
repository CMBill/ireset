# W1训练拟合与目标一致性单因素诊断执行方案

日期：2026-10-22。版本：**v0.1（审计设计草案）+ 2026-10-22 合成实施补充**。原设计覆盖“现有诊断文档收口 + 训练目标与实现一致性审计设计”；2026-10-22 已实现并运行 A0–A5 合成审计（§7）。**仍不是新的真实数据实验预注册，也不授权执行训练或真实 checkpoint 探针**。

| 状态字段 | 当前状态 |
| --- | --- |
| `static_code_review` | complete：完成下文列明的源代码与配置核对；不代表运行行为通过 |
| `audit_implementation` / `synthetic_audit` | **complete / complete**（2026-10-22）：审计模块、CLI、26 项测试与权威合成运行完成；结论仅 `synthetic_contract_passed`（合成适用域），不升级真实结论 |
| `execution_registration` / `cohort` | not_created / not_assigned |
| `real_data_audit` / `single_factor_experiment` | not_run / not_registered |
| 本次新增真实数据训练更新 / 生产模型调用 | 0 / 0（A3 仅合成替身/微型模块） |
| `factor_selected` / `auto_execute` | false / false |
| 最终交互方法 / 生产接入 | not_ready / false |

> 2026-10-22 实施轮补充：A0–A5 的合成实现与运行已完成，实际结果见[合成审计验证记录](../verification/omniretarget_w1_training_objective_audit_verification.md)与 `derived/audit/w1_training_objective_audit/`。真实数据审计、单因素预注册、cohort、预算与训练仍未开始；`synthetic_contract_passed` 不构成真实数据、G1 FK 或训练主因证据。

## 1. 依据、问题与证据地位

### 1.1 父契约与历史边界

依赖[OmniRetarget 总方案](omniretarget_first_training_plan.md) §6.3/§6.4、[W1 A3/B/C 方案](omniretarget_w1_a3bc_plan.md)、[W1 D/E 方案](omniretarget_w1_de_plan.md) §7.3，以及[预测误差与窗口一致性方案](omniretarget_w1_prediction_consistency_plan.md) §7；实际状态以各自验证记录为准，冻结方案不改字节。

- [W1 D/E/F 验证记录](../verification/omniretarget_w1_de_verification.md)：两模型各 4,000 步，best 均为 step4000；val31 teacher-fit FK 宏 0.12204454/0.11134759m，Teacher raw 2/31、两模型各 0/31 通过裸输出质量门。动作基线证据完整，交互方法未就绪。
- [A3/B/C 验证记录](../verification/omniretarget_w1_a3bc_verification.md)：S0 已证实固定六段首窗可以降低状态重建误差；这不直接证明正式训练全集拟合、整段生成或物理质量达标。
- [预测/窗口一致性验证记录](../verification/omniretarget_w1_prediction_consistency_verification.md) §0：r6 存在阶段尝试数超限（R6/W5/P3/F1）及未计费真实预检，**不能作为可验收的完整诊断**。其训练窗误差、跨窗分歧与去噪结果只用于提出问题，不作为本方案已证实的因果结论。
- [旋转融合验证记录](../verification/omniretarget_w1_rotation_fusion_verification.md)：r10 GS 失败，GQ 3/26 通过、94 项退化；候选不接入。旧融合 S 预算不足与预测诊断 cohort 不可复验的约束分别保留。

本方案不接续旧预测诊断的 R/W/P/F，不生成融合 r11，不通过新标题、目录、方案哈希或实现 SHA 重置任何旧预算。原 U0 `incomplete`、UR `no_quality_gain` 与原 W1 F `not_ready` 均保留。

### 1.2 本轮先回答的三个问题

1. 实际训练目标的编码、物理单位、归约和梯度是否与登记定义一致？
2. S0 与正式训练有哪些可由代码证明的差异，哪些仍需合成对拍或未来注册的真实数据测量？
3. 若工程契约通过，现有证据还缺什么，才能只选择一个训练拟合因素作下一次有限实验？

本轮不预判“训练步数不足”“FK 权重过小”“扩散采样是主因”或“教师是主瓶颈”。训练拟合、物理质量、跨窗一致性与验证泛化分开报告；改善其中一项不升级其他结论。

## 2. 静态核对结果与目标定义

核对基于代码提交 `ae6f6a8324c40ad7456493df411209e709d472b2`；本次读取源码与文本配置，没有实例化真实数据集、加载 checkpoint 或调用模型。后续若实现身份变化，应重新核对受影响项，不能把本表当作新代码已运行的证据。

### 2.1 已核对的调用链

| 链路 | 当前实现事实 | 源码 |
| --- | --- | --- |
| W1 数据入口 | `training_manifest` 与旧 graded 入口互斥；W1 使用键控缓存；按注册清单核验行和 split | [windows.py](../../src/ireset/data/windows.py) `TeacherWindowDataset._load_w1` |
| 训练窗口 | 60 帧、stride30；`range(0,T-60+1,30)`，不补推理尾窗、不做 padding；数据项 `valid` 为全 1 | [windows.py](../../src/ireset/data/windows.py) `__init__`/`__getitem__` |
| 状态目标 | 3 维标准化 clip 根位置 + 两列旋转 6D + 29 维线性限位归一化关节，共 38 维 | [tokens.py](../../src/ireset/model/tokens.py) `StateTokenizer.encode` |
| 根统计 | 仅准入 train 文件的原始帧，每帧一次；不按重复窗口拟合；std<1e-6 时置 1.0 | [w1_training.py](../../src/ireset/data/w1_training.py) `fit_w1_root_stats`；[tokens.py](../../src/ireset/model/tokens.py) `RootStats.fit` |
| 输出头 | 根位置/旋转不做全状态裁剪；关节 `u=2*sigmoid(logits)-1`；物理关节解码再 clamp 到 [-1,1] | [backbone.py](../../src/ireset/model/backbone.py) `assemble_x0`；[losses.py](../../src/ireset/train/losses.py) `decode_clip` |
| 确定性 | 由条件 memory 直接预测 x0，不读取目标根/关节作为生成输入 | [deterministic.py](../../src/ireset/model/deterministic.py) `forward` |
| 扩散训练 | 随机 t 与 epsilon 加噪；网络预测 x0；DDPM/DDIM 为 `prediction_type="sample"`、`clip_sample=False` | [generator.py](../../src/ireset/diffusion/generator.py)；[trainer.py](../../src/ireset/train/trainer.py) `_forward` |
| 条件通路 | 人体源物体系/clip 系关键点、目标箱体、参考接触日程、静态箱体/地面；接触条件仍开启 | [conditioner.py](../../src/ireset/model/conditioner.py) `ConditionEncoder.forward` |
| 损失与更新 | S0/正式训练共用 `LossAssembler`；每 micro loss 除 grad_accum 后反传，累积后 clip，再 AdamW 更新 | [w1_s0.py](../../scripts/w1_s0.py) `run_s0_model`；[trainer.py](../../src/ireset/train/trainer.py) `run` |
| 正式选模 | 不是窗口 loss：全部注册 val31 整段生成，seed7/DDIM25，完整且有效时才用 demo 等权 FK MPJPE 选 best | [train_generation.py](../../scripts/train_generation.py) `make_formal_eval` |

`LossAssembler` 的说明使用了 masked reconstruction 措辞，但实际位置/旋转/关节/差分归约不读取 `valid`。现有 W1 数据项是完整窗口且 `valid=1`，静态核对不能据此断言当前损失受污染；它明确限制了当前实现的适用域。未来 padding/变长窗口必须另立 mask 契约与反例，不能沿用该措辞当作已支持。

### 2.2 当前损失的精确定义

令 `D = x0_pred-x0`，`mean` 对该项全部 batch、帧与通道归约。配置来自[omniretarget_motion.yaml](../../configs/train/omniretarget_motion.yaml)：

| 项 | 实际计算 | 单位/分母 | 登记权重 |
| --- | --- | --- | ---: |
| pos | `mean(D[...,0:3]^2)` | 标准化位置平方，B×F×3 | 1.0 |
| rot6d | `mean(D[...,3:9]^2)` | 6D 表示平方，B×F×6 | 1.0 |
| joint | `mean(D[...,9:38]^2)` | 限位归一化关节平方，B×F×29 | 1.0 |
| geodesic_rad2 | `mean(norm(log(R_pred R_target^T))^2)` | rad²，B×F | 0.5 |
| vel | `mean(((pred[:,1:]-pred[:,:-1])-(target[:,1:]-target[:,:-1]))^2)` | 标准化每帧差分平方，B×(F-1)×38 | 0.5 |
| fk_m2 | `mean(sum_xyz((FK_pred-FK_target)^2))`，15 个解剖 site | m²，B×F×15；target FK 在 no_grad 内 | 2.0 |
| contact | W1 权重为 0，专用手部 FK/SDF 与监督 mask 分支不执行 | 不产生该损失字段；条件中的接触日程另属条件输入 | 0.0 |

`total = pos + rot6d + joint + 0.5*geodesic_rad2 + 0.5*vel + 2*fk_m2`。重建三组分别求均值，**不是**38 通道整体 MSE；vel 则对38通道共同求均值。当前实现无显式 t/SNR 权重、无碰撞罚项、无跨窗配对损失、无物理速度除 dt、无额外 Laplacian。它们是当前目标定义，缺少某项本身不构成 bug 或本轮新增许可。

FK m² 与评估的 FK MPJPE m 在开方、归约上不同；不能把 `fk_m2` 直接开方就称为 MPJPE。归一化误差、损失权重和梯度贡献也不是同一个量。后续测量应逐项保留原值、加权值与梯度，不能仅凭日志数值较小就提高权重。

### 2.3 S0 与正式训练的可证差异

| 项 | S0 | 正式 W1-D |
| --- | --- | --- |
| 数据覆盖 | 注册 pilot6，各取 start0 首个完整窗 | train24 全部常规完整窗；不加入尾窗 |
| 单 micro-batch | 六窗整批、每批等权；同一六窗反复消费 | micro_batch4、shuffle、drop_last=True |
| 每次更新 | grad_accum4，即24次样本消费但仅6个独特窗口 | grad_accum4，即16次样本消费；跨 epoch 继续迭代 |
| 历史数据规模 | 六个固定首窗 | 既有证据为110 train窗；实际新运行仍须重新登记核验，不能硬编码代替数据检查 |
| epoch 尾部 | 六窗整批无碎批 | 若110窗则每epoch丢2窗；shuffle 后不是登记固定删除两窗，需区分每epoch遗漏与累计覆盖 |
| 更新上限 | 每模型1000 | 每模型4000，原 W1 不自动续行 |
| 训练噪声 | 扩散每个 micro 重新采样 t/noise | 同样每个 micro 重新采样 t/noise |
| 验证对象 | 每10步固定同一六窗、固定 t/noise；状态三组 MSE 之和 | 每250步完整val窗口loss；每1000步val31整段生成 |
| 通过/选模 | S0 数值、趋势、重载与身份门；保存 last | val31完整生成的demo等权 teacher-fit FK MPJPE选best |
| 共享内容 | 状态参数化、条件编码、骨干参数、损失权重、AdamW及梯度裁剪取同配置 | 同左；训练入口不是同一循环实现，仍须合成对拍 |

因此“同一损失”不能推出“两阶段实验等价”；S0 小集趋势也不能排除正式训练的优化、目标尺度、覆盖或泛化问题。上表只描述代码，未对历史权重、梯度或数据覆盖作新增实测。

## 3. 审计顺序与验收工件设计

以下为**后续实现设计**，均未执行。先完成不读真实数据的合成核验，再根据结果另立真实数据审计/单因素实验预注册。不得直接把本节当作可运行的实验矩阵。

### A0：身份与语义清单

实现一个独立、默认纯合成的审计入口。静态清单至少登记：代码提交与逐文件 SHA、配置文件原字节与有效配置、每个参数到消费函数的映射、状态各维单位/坐标/范围、损失项及归约分母、S0/正式循环差异。清单生成不得因 import 自动访问真实资源或历史实验目录。

计划产物 `static_contract.json`/`s0_formal_comparison.json`，状态须区分 `static_confirmed`、`synthetic_verified`、`runtime_not_checked`、`contract_mismatch`。缺乏证据不能填写 passed；自报的通过字段不能成为独立验收依据。

### A1：编码、解码与标量目标对拍

固定解析合成状态（非平凡平移/旋转、29关节限位内值）与合成 mean/std/world_from_clip；不从真实 train/val 提取样本。分别核验：

1. NumPy编码与Torch编码的列顺序、坐标变换、根标准化和关节映射；用解析期望值独立对拍，不能只做同一函数往返。
2. `LossAssembler.decode_clip` 与 tokenizer 世界系解码经已知刚体变换后的物理一致性；保留四元数符号等价与旋转误差，不把 quaternion 分量差当 SO(3) 角。
3. 仅扰动位置、仅扰动旋转、仅扰动一个关节、仅扰动相邻帧等输入的逐项损失；按 §2.2 用独立公式重算标量、分母与 total。
4. 重复整个 batch、重排样本不改变均值；不同通道数量不会暗中把三组均值变成整体MSE。明确 vel 的38维归约与其他三项的分组归约差别。
5. 对近零6D列、平行列、近π旋转、关节边界和饱和logits单列数值域检查。已知近π精度负门保留；异常不得用零旋转/零误差回退成通过。
6. `w_contact=0` 时手部/SDF/接触监督分支真实零调用，同时参考接触仍可作为条件。未提供真实资产的合成FK替身只验证装配代数，不能冒充 G1 FK 正确性。

计划产物 `objective_rows.json`、小型 `synthetic_inputs.npz` 与按字段记录的 failures/gaps。mock FK 的行显式标记 `fk_backend=synthetic`；真实G1 FK对拍需后续注册与计费，另记 `not_run`。

### A2：梯度路径与尺度

在可微非奇异合成点，对 pos/rot6d/joint/geodesic/vel/FK 分别求输出梯度，保留未加权与加权梯度范数、方向、有限性和零值原因。验证 `grad(total)` 等于逐项加权梯度和；目标分支不需要梯度，预测分支的 detach、错误no_grad或权重未消费必须可被反例逮住。

有限差分只用于远离 clamp、sigmoid极端饱和与SO(3)分支切点的点；在分支切点报告适用域和不可判定项，不用放宽全局容差换通过。先用float64解析/有限差分对拍，再检查生产float32；不能强称现有硬编码float32的装配器已支持float64。

梯度余弦只在双侧非零且有限时定义，零梯度为 null+原因；不得以 bool 充数，不得通过NaN比较误过。输出空间梯度与模型参数梯度分开；合成梯度通过不能证明真实checkpoint的梯度平衡，更不能给出责任百分比。

计划产物 `gradient_rows.json`，键至少为 `(fixture,term,space,dtype)`，明确实际可计算的空间；缺一必测项不发布“全部梯度通过”。真实checkpoint上的梯度分布不在本草案授权内。

### A3：更新循环、噪声与RNG一致性

使用小型合成数据集和最小可微替身模块，两个入口取同一份初始参数、同一有序micro-batch、同一显式t/epsilon，比较前向、分项loss、裁剪前梯度、裁剪后梯度、AdamW参数与状态的一次更新。替身对拍用于隔离循环；真实骨干、真实缓存与历史checkpoint均不在此阶段调用。

- 先测无裁剪，再测真实触发裁剪；梯度累计与一次等权大batch的对拍须固定无批间随机差异的替身，避免将dropout/随机噪声差异误判为累计错误。
- 校验micro loss确实除以grad_accum，model与conditioner均入optimizer且无重复参数，AdamW lr/weight_decay有效；当前实现未发现lr scheduler调用，记录固定lr而非臆造warmup。
- 扩散的x_t需按独立alpha公式与注入的t/epsilon重建；t=0仍为第一档加噪，不称无噪声；DDIM采样与单步去噪测试分开。
- 对合成样本ID追踪每次实际消费、epoch尾部、跨epoch累计和真实optimizer更新；分别核验S0六窗整批与正式4×4，不强迫二者原生采样顺序相同。
- 在验证/异常路径前后比较Python/NumPy/Torch CPU/CUDA（若合成测试实际使用）的RNG及module模式。DataLoader独立generator另列，验证创建迭代器不得污染训练全局流。

计划产物 `update_parity.json`/`sampling_trace.json`/`rng_state_facts.json`。任何一条非有限梯度、更新未返回或状态不对齐，停止后续因果解释；不自动修正参数后继续。

### A4：归约、评价口径与模型选择

用演示长度不同、每段窗口数不同的合成完整集合，独立重建窗口loss、demo宏、窗口微平均、物理FK MPJPE。故意令宏/微排名不同，确认正式best只消费注册val全集完整宏；坏行必须留在固定分母内，不能删去后选出更优best。

必测键集用注册多重集核验，不能只查数量或 `all([])`；删窗、重窗、换demo、缺t、重复t、missing metrics、bool/数值混用、NaN/Inf、缺注册字段均保留具体原因。loss history已经round到6位，不能从舍入日志伪造精确逐位total对账；新合成工件保留未舍入值与表示精度。

拟登记相同输入/同代码的记录复制要求dtype/shape/逐位一致；独立数值路径先校验类型、定义域与有限性，再使用预先冻结的逐字段绝对/相对容差。容差须由解析合成误差与dtype得到并在真实数据打开前冻结；未确定容差的检查保持 `not_ready`，禁止运行后按结果调线。

### A5：独立复核与阶段结论

独立验收器从合成输入与有效配置重算键集、损失、梯度和更新事实，读取原始工件而非信任顶层通过字符串。必须同时核对失败/缺口、summary与明细，损坏工件登记失败而不中断报告。

结论仅允许：`implementation_mismatch`（有可复现反例）、`synthetic_contract_passed`（仅合成适用域通过）或 `audit_incomplete`（证据/定义域不足）。**不发布训练主因，不升级 `not_ready`，不自动启动单因素实验**。

## 4. 拟实现边界与测试要求

建议新增独立审计模块、脚本与测试，名称待审阅后固定：

- `src/ireset/eval/training_objective_audit.py`：键集、独立标量/梯度对账、聚合与结论纯函数；不改默认loss、模型或数据入口。
- `scripts/audit_w1_training_objective.py`：默认仅合成；显式拒绝真实manifest/cache/checkpoint路径及研究执行参数；不把旧诊断脚本当新入口调用。
- `tests/test_w1_training_objective_audit.py`：临时目录、解析fixture/替身模型；不得扫描生产历史或继承真实PROJECT_ROOT。必要时复用现有纯函数，不复制成熟旋转/优化器实现。

实现前核对现有loss/token/trainer/噪声测试并补其缺口，测试规模按实际反例记录，不预报“新增N项已通过”。至少覆盖A1–A4的正向完整链与每类篡改；特别钉死：单位/分母错、错误detach、累计少除一次、conditioner漏入optimizer、退化旋转、接触分支误执行、宏微替换和空集合空真。

若审计发现实现缺陷，先单独登记反例与最小修复，再决定哪些历史结论受影响；不覆盖冻结配置、缓存、教师、checkpoint或旧final，不把修复后运行伪装为原实现证据。

## 5. 真实数据与单因素实验的后续门

本文件尚不具备可运行预注册的要素。以下内容完成并取得明确的真实执行授权前，真实数据审计与训练均为 `not_run`：

| 后续必备项 | 冻结要求 |
| --- | --- |
| 可区分的研究问题 | 基于已完成审计选择一个假设；说明与旧预测诊断、融合问题的区别；不能只是重跑旧R/W/P/F |
| cohort与历史 | 正式修订说明父cohort及其拒收原因；显式定义新身份与账本，保留旧尝试数及成本下界；不能以标题/哈希变化自动取得新预算 |
| 实际单因素 | 唯一改动、对照、初始化/是否从头、固定seed、数据/缓存/条件、步数和评价冻结；未选定时不列候选网格并行搜索 |
| 真实审计矩阵 | train名单与完整窗口键、t/noise及调用/前向上限、需要梯度的空间、输出字段全部事前列出；禁止先读真实误差决定范围 |
| 评价边界 | train用于拟合诊断；已暴露val6/known只作描述性回归，不用于挑权重/阈值/seed；是否消费val31、用途与选模必须另写，不能宣称已暴露集合是新盲测 |
| 总预算与阶段预算 | CPU/GPU墙钟、模型调用、更新步数、attempt上限与收口预留均明确；全部失败/崩溃/预检累计，阶段不互借；不足即不启动 |
| 事前登记及消费者 | 实质工作前running索引/台账；父unit准入；盘×索引×台账状态和数值对账；unknown或publishing拒消费，封口失败不能exit0 |
| 验收与停止规则 | 工程不变量失败先停；真实实验可合法交付负结果/归因不足；不得因为改善teacher-fit就声称物理可行或生产可接入 |

代码/配置静态阅读、文档维护和纯合成测试不创建真实研究attempt。任何真实数据/资产的数值重算、真实模型前向/反向，即使名为只读预检或测试，也必须事前进入未来获授权的账本；不重演r6的未计费预检。

## 6. 当前交付验收与下一交付

本次完成条件为：历史统计与测试范围订正一致；本草案可追溯到真实代码/父方案；S0与正式训练差异、目标单位/归约、计划反例、未执行项及后续授权边界明确；文档链接有效，旧证据和冻结方案哈希不变。

本次文档检查：AGENTS、文档索引、预测诊断验证记录与本草案共 **117 处本地链接均有效**；七份冻结方案与r6注册/final/pairs共 **10 项SHA-256全部与原登记相等**；`git diff --check`通过。未运行新增合成审计或模型测试，也未重跑全项目回归；旧测试计数仍按原运行范围登记。

**2026-10-22 实施轮已完成 A0–A5 合成审计实现与报告（§7）；真实数据审计预注册、单因素选择及训练实验仍未交付。**下一项真实交付须按 §5 另立正式预注册并取得用户授权；若仍无可区分假设，交付 `attribution_insufficient`，不以运行更多旧探针代替问题定义。§6 的“本次文档检查”是起草时的文档交付快照，不代表当前实施范围。

## 7. 合成审计实施结果（2026-10-22）

本节记录 A0–A5 的实现与权威合成运行；详细证据见[验证记录](../verification/omniretarget_w1_training_objective_audit_verification.md)。§6 的“本次只完成文档与静态核对”是实施前快照，已被本节取代。

- **实现**：`src/ireset/eval/training_objective_audit.py`、`scripts/audit_w1_training_objective.py`、`tests/test_w1_training_objective_audit.py`。CLI 默认纯合成，显式拒绝 training manifest/cache/checkpoint/预注册/预算/demo/step/seed 等真实或研究输入；不调用旧诊断脚本。
- **权威命令**：`uv run --locked python scripts/audit_w1_training_objective.py --out derived/audit/w1_training_objective_audit`，退出码 0。
- **结论**：生产侧与独立复核均为 `synthetic_contract_passed`；`problems=[]`。66 行 = A0 静态 22 + A1 23 + A2 14 + A3 19 + A4 10；失败/缺口 0/0；已登记限制 5 项。
- **代表性事实**：A1 编码/解码误差为 0/2.801e-8，`total=5.5997726e-4`，分组均值与整体 MSE 差 9.907e-5，`vel` 38 通道归约与分组和不同；A2 加权梯度和与总梯度差 **4.34e-19**，有限差分最大相对误差 **3.77e-9**，float32 对 float64 最大相对差 **1.71e-5**；A3 四个 parity run 的前向输出/分项 loss/参数/裁剪前与裁剪后梯度/AdamW 状态/t·epsilon 全部逐位相等，等大小 2+2 微批与一次 4 窗大 batch 梯度差 1.86e-9，S0 24 次/6 唯一窗、正式 32 次/8 micro/每 epoch 丢 2，RNG 恢复与独立 generator 均通过；A4 宏/微排名故意不同且正式 best 只消费完整宏，10 类坏 roster 变体全部 `macro=null`、`complete=false`，`best_formal_step` 完整时 2、缺行时保持 1。
- **限制**：近 π 精度负门（实现 0.061667 vs 独立 1.233622）、平行 6D 的 tokenizer `ok` 标志漏检（正式 Trainer/生成器仍拒绝，未使用的 NumPy decode 会抛错）、零 6D 解码占位仅由正式调用方守卫、float32 审计使用合成构造器、`loss_history` 6 位舍入不能做逐位对账。真实 G1 FK 对拍 `not_run`。
- **测试**：`uv run --locked pytest tests/test_w1_training_objective_audit.py -q` -> **26 passed**（10.95s）；全项目套件未重跑，旧 947 passed + 1 skipped 仅作历史范围记录。
- **边界**：未加载真实 dataset/cache/checkpoint，未对任何 W1 训练 checkpoint/生产模型做前向，真实数据训练更新为 0；A3 的采样/循环事实只使用随机初始化微型 `DiffusionG1` 与替身模块；未创建执行注册、cohort 或预算；`factor_selected=false`、`auto_execute=false`、最终交互 `not_ready`、生产接入 false。通过合成审计只表示实现契约在合成适用域成立；下一交付须按 §5 另立真实审计/单因素预注册并取得用户授权；无可区分假设时交付 `attribution_insufficient`。
