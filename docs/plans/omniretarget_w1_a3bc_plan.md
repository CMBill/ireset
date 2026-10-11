# OmniRetarget W1 弱动作监督 A3/B/C 实现方案

日期：2026-10-11。版本：v0.1 草案。状态：**仅完成方案起草；运行预注册、数据审计、新训练接口和 S0 均未执行，正式训练仍为 `not_ready`。** 本文件明确下一轮实现范围，不能作为任何阶段已通过的证据。

父方案：[OmniRetarget 优先教师与初步训练执行方案](omniretarget_first_training_plan.md) §3.4–§8、§10–§12；入口依据：[A2 收口方案](omniretarget_a2_closure_plan.md) §7.4/§12，以及[最新验证记录](../verification/omniretarget_a2_closure_verification.md) §15。当前文档导航与历史归档见[文档索引](../README.md)。

## 1. 当前事实、问题与范围

### 1.1 已完成的前置工作

- A1 已冻结 pilot6、val31、val6、输入与资产身份，并验证 37 段源/目标场景配对。
- 原 U0 保持 `incomplete`，唯一缺口为 sub12；停线登记保留，不以新实验替代原裁决。
- 独立恢复实验 `a2_recovery_step03_v1` 已完成六段生成，唯一注册差异为 step_size 0.2→0.3。UR 的完整裁决为 `no_quality_gain`，g1/g2 通过、g3/g4/g5 失败。
- 最新 C2 为 `20261010_a39d763_a2_closure/decisions/attempt_010/a2_closure.json`；登记完成与训练就绪是两个状态，当前训练仍未就绪。
- 最近全量测试记录为 564 passed + 1 skipped；2026-10-11 独立复核的 188 项相关测试通过。它们是前序代码验证，不能替代本方案新增行为的验证。

### 1.2 本轮要回答的问题

在明确保留发布教师质量缺陷的条件下，先建立可追溯的动作学习基线：数据转换与条件是否正确，两类模型是否能够拟合有限动作样本，以及后续裸输出中的失败能否区分为学习不足、条件错误或标签本身的问题。

W1 使用**已经发布的 `*_original.npz`** 作为根/关节动作标签。UR 报告提供弱监督路线的入口依据，不提供本轮训练标签；其生成配置和 MuJoCo 环境不填写为发布标签的生产身份。发布时未提供的生产环境字段明确记 `not_published`，本轮实际训练环境必须实测登记。

发布教师中的悬浮、穿透、脚滑和漂移不会因关闭接触损失而消失。因此本轮用途固定为 `weak_motion_only`，不承诺获得严格可行搬运轨迹。其信息价值来自检验数据和学习链路；后续教师处理或可行性方法应依据裸输出证据另立方案。

### 1.3 实现范围

本轮实现并执行 A3 发布数据审计、B 数据/缓存/训练及评估接口、C 数值烟测，交付训练就绪判断。D/E 的来源、预算和完整评估契约在本文固定，但本轮不运行正式 4k 训练或最终 val31 比较。

沿用现有模型、Diffusers、运动学与独立验收器。源数据、归档、模型、A1/A2/UR 产物只读；不实现教师重生成、修复器调参、增强、标签平滑、速度投影或新的损失算法。

## 2. 入口证据与运行预注册

### 2.1 必须复验的锚点

以下路径相对项目根；A2 根目录为 `derived/experiments/omniretarget_first/20261010_a39d763_a2_closure/`，A1 根目录为 `derived/experiments/omniretarget_first/20261009_13906c1_a1/freeze/`。

| 锚点 | 文件 | 固定 SHA-256 |
| --- | --- | --- |
| C2 终局 | A2 根 `decisions/attempt_010/a2_closure.json` | `34a392f92a9cda5b8317a08d5b5b486f750c5fc83933174d36d361cb7b1d0fa2` |
| UR 完整比较 | A2 根 `recovery/cmp/attempt_002/teacher_selection.json` | `faa2df0cb9c1df37b325f0f73a7d2989c4ffc425270439679b4530d32724b00e` |
| UR 注册 | A2 根 `recovery/preregistration.json` | `f4676813b20c2784385237fc2c101636e9c77aaf280e2376f029b8252f9c42d9` |
| A1 冻结名单 | A1 根 `frozen_lists.json` | `395094a30d69158ad08bf5b576f07c541c052187d5bf242f1cc7fde05e353d9d` |
| A1 资产 | A1 根 `assets.json` | `2f72c9ed0f1832b0c1d80e7d1f2e82382785c56ae83556a63a46ec6789c7ad79` |
| A1 源箱配对 | A1 根 `pairing_source_boxes.npz` | `39fce035ab63569bd69a4ddf61e1e6cd4fb70d4d1b2a12abf7051bd55ce3d09d` |
| 原 train 分组 | `derived/data/split/train.json` | `93ff2ba4dd956e1ee8d12177cb48f5a0514343eb126b864696d94578aa196f1c` |
| 原 val 分组 | `derived/data/split/val.json` | `7387d68cc56e5f87059ce4be2a56386510a49d96732fa77450ee53bb4433f15e` |
| 分组登记 | `derived/data/split/manifest.json` | `fcfd13bf108766c05e92aa45fc751ffbf620bbc86de630a4a9e17faa1e052090` |

入口复验重新读取事实：C2 登记完整、原 U0 裁决不变；UR admission 通过、六段证据完整、无缺行，且从度量行/绑定重推得 `no_quality_gain`。复用已实现的 `verify_merge_chain` 和 `reverify_verdict_record`，不只读取顶层通过标记。资产清单中的实际文件同步重哈希；名单文件未变不等于输入和资产未变。

### 2.2 新注册的内容和顺序

拟用实验身份 `w1_published_motion_v1`，训练协议 `omniretarget_motion_v1`，用途 `weak_motion_only`，来源 `published_original`，发布 revision 为 `135259130158f1cb1799e4ab9b6566c938285fc9`。

实现验收后，在读取本轮新增 train18/val31 质量结果及运行 S0 前，独占创建运行注册。注册包含上述锚点、方案文件哈希、具体 train24/val31/val6 名单、实际 NPZ/PT 路径与完整哈希、资产闭包、审查规则、S0 规则、固定配置、预算和停止条件。文件存在时只允许核验一致后复用，不覆盖；实际注册和哈希尚未产生，不能追认旧缓存或旧模型为本轮产物。

文档草案允许正常修订；正式运行前记录最终方案字节及实现提交。修改审查规则、名单、模型配置或数据协议后分配新实验身份，保留旧尝试。

## 3. A3：名单与数据审计

### 3.1 固定集合

在原 train250 中按受试者分别计算 `SHA256("20261008|teacher_pilot|" + demo)`，按完整摘要升序取前四个；同摘要时按 demo 排序。每受试者四段，包含原 pilot6，集合标识为 `teacher_production_train24_v1`，实际标签来源另记为发布 original。选择只读取身份，不读取质量或模型结果。

按当前固定 train.json 复算的候选如下；这只是名单复算，不表示这 24 段已经完成动作准入或质量审计。

| 受试者 | 按哈希次序的四个 demo |
| --- | --- |
| sub10 | sub10_largebox_097、sub10_largebox_085、sub10_largebox_092、sub10_largebox_063 |
| sub12 | sub12_largebox_082、sub12_largebox_047、sub12_largebox_053、sub12_largebox_058 |
| sub16 | sub16_largebox_000、sub16_largebox_009、sub16_largebox_004、sub16_largebox_051 |
| sub3 | sub3_largebox_027、sub3_largebox_012、sub3_largebox_001、sub3_largebox_034 |
| sub7 | sub7_largebox_022、sub7_largebox_024、sub7_largebox_003、sub7_largebox_012 |
| sub8 | sub8_largebox_035、sub8_largebox_041、sub8_largebox_025、sub8_largebox_005 |

名单数组的规范序为受试者名字符串升序、组内哈希次序；UTF-8 的 `json.dumps(list, ensure_ascii=False, separators=(",", ":"))`，无末尾换行，其 SHA-256 为 `8fbfd008b98f56196d31f81352582006ba95c16bdcf2a4da2f5da56eb8267948`。它只校验 demo 数组，不代替运行清单及逐文件输入哈希。

val 使用 A1 已冻结的全部 `val_originals_v1`（31 段）。详细分析子集复用 `val6_pilot_v1`：sub10_100、sub12_008、sub16_052、sub3_018、sub7_032、sub8_001，均指对应完整 largebox demo。test31 和 reserve17 保持原身份。本轮不使用 rot/trans，不根据审计失败替换候选，不借用历史 calibration24 或 diagnosis6 作为留出集。

预期审计分母为 train24+val31=55 个独立演示；每个候选恰一条最终结果。复用原 pilot6 和 val31 配对证据时仍核验输入哈希；新增 train18 必须实际完成配对。窗口数量不作为独立演示数。

### 3.2 基础准入与损坏处置

拟新增 `scripts/audit_omniretarget_training.py`，复用 [pairing.py](../../src/ireset/data/pairing.py) 的相似变换和已有完整 SHA-256 工具，补齐训练协议要求：

1. NPZ 必须有 qpos、human_joints、fps；精确形状 `[T,43]`、`[T,52,3]`，fps=30，T 与 PT 一致。所有根/箱/关节/人体值有限。
2. 根和箱四元数非退化，单位范数误差 ≤1e-4 时仅在派生表示中作登记归一化；超过范围拒绝。bool、NaN、缺值不能洗白成合法计数或零误差。
3. 发布根/箱 `[qw,qx,qy,qz,x,y,z]` 显式转换为内部 `[x,y,z,qw,qx,qy,qz]`；PT 箱序列为 `[x,y,z,qx,qy,qz,qw]`。检查往返和四元数符号等价。
4. 29 关节与锁定普通手 G1 资产逐名对应；超关节范围拒绝文件，本轮不 clamp 教师。源与项目资产的坐标、尺度、mesh 原点和碰撞代理分别登记。
5. 身高缩放、源人体场景、目标任务场景及帧对应可重建。沿用配对界：人体最大残差 ≤1e-5 m，箱姿态映射误差 ≤1e-3 度，常数位置偏移残差 ≤1e-5 m。
6. demo、subject、variant、split、父关系唯一且相容；重复、跨 split、未知命名或缺输入显式记原因。

数值/单位/身份/时间/配对损坏使 `admission.motion=false`。T<60 的完整窗口数为零，单列短片段与时长。strict 穿透、接触或速度检查失败单独登记，不自动取消弱动作监督资格；不以提高质量均值为理由逐帧屏蔽困难动作。

突跳审查先检查时间断裂、单位和映射。没有已证实的数据损坏时记 `review_required`，不临场发明幅度阈值来筛选；未解决的审查项不能标为准入。需要新规则时保留本次证据并另登记修订。

### 3.3 源/目标场景与独立质量登记

参考人体对应源箱；目标箱来自给定任务序列。已知 original 源/目标常有纯 z 常数偏移，不能把两个字段都赋为 qpos 箱轨迹。复用 `source_obj_pose_in_target_scene` 和 A1 映射记录，在共同世界坐标中分别保存二者；源箱的几何尺寸同时应用已登记尺度。

`kps_obj` 和参考接触从源场景计算，模型目标箱条件使用目标轨迹。尺度、clip frame 和参考接触只依赖人体参考与已知场景，不能从教师根、关节、脚落点或 FK 恢复推理条件。教师 qpos 可作监督和质量诊断，不作条件。

55 段发布标签按冻结验收器记录穿透最大值/违规比例、限位/速度、脚滑/离地、手箱间隙/有效覆盖、认证区间/双手交集、漂移和固定箱体地面冲突。参考区间按源场景固定，统计采用现有验收器的自适应插值、SLERP、逐 site 最坏脚指标和逐帧任务误差。

质量检查运行失败或超时记 `incomplete`/`not_checked` 及原因，不填 0 或通过；质量检查成功但 strict 不通过是可登记的弱监督缺陷。本轮实际训练标签必须有完成的质量登记；条件有效但标签坏的 val 任务继续留在完整生成评估分母中，teacher-fit 标缺失。

### 3.4 A3 产物与出口

交付 `omniretarget_training_manifest_v1` 清单、逐演示配对/质量记录与按 split/subject 汇总。每行包含实际 NPZ/PT 路径及哈希、发布 revision、场景/资产身份、T/fps/时长/窗口数、动作准入及原因、完整质量状态、派生表示路径/哈希、协议与运行身份。

顶层同时保存候选数、处理数、准入/拒绝/待审查数、独立帧/演示/窗口统计。expected=processed+未完成；不通过删除行来使处理率成为 100%。源文件在审计前后重哈希，只读不变。

A3 出口要求 55 段均有结果、实际准入 train 标签质量登记完整、六受试者各有可用完整窗口。数量不足或某受试者全不可用时保留清单并解释；更小集合只能作调试，不自动补选下一段。

## 4. B：数据协议与缓存

### 4.1 协议与加载入口

拟新增 `configs/labels/omniretarget_motion_v1.yaml`，明确 `training_purpose=weak_motion_only`、`source_type=published`、`variant=original`、`w_contact=0`、禁止教师改写。旧 graded 协议和历史无清单入口保持原契约。

在 [TeacherWindowDataset](../../src/ireset/data/windows.py) 新增显式 `training_manifest`，与 `label_manifest` 互斥。W1 只消费所请求 split 的 `admission.motion=true` 行；启动时验证协议、名单、选择依据、输入、派生表示、资产和缓存哈希。缺文件、漂移、缺字段、重复行或不兼容协议直接报错，不回退旧缓存或其他来源。

在 [teacher.py](../../src/ireset/data/teacher.py) 通过薄适配消费明确的 source_obj_pose/target_obj_pose。旧入口的 source=target 近似只保留为历史兼容，不进入 W1。训练和生成使用同一条件构造函数；改教师根/关节但保持人体与任务不变，模型条件应逐位不变。

训练沿用 F=60、stride=30，只创建完整窗口：`max(0, floor((T-60)/30)+1)`。无 padding/任意帧 mask 新体系，不因重叠窗口重复计独立帧或时长。

### 4.2 统计量与缓存

扩展 [precompute_m3_cache.py](../../scripts/precompute_m3_cache.py) 的显式清单、输出目录和线程参数。只缓存原始 clip 的参考条件，不预保存重复滑窗。

- root_stats 只用当前 W1 train24 准入文件的原始帧，每帧一次；val/test 和被拒文件不参与。登记帧数、文件名单摘要、统计定义和原始/最终 std（沿用已验证的下界处理）。
- schedule 缓存键包含实际 NPZ/PT 哈希、源/目标映射、源几何、资产、协议、推断配置和条件代码身份。文件名相同不代表缓存可复用。
- 派生条件、root_stats 和 checkpoint 共同绑定训练清单及统计身份；历史 `m3_cache`、其他来源和其他 cohort 的统计不能借用。
- 源数据漂移导致运行错误；可证明仅派生缓存过期时分配新缓存并重新计算，保留旧产物，不静默改写证据。

## 5. B：训练、验证与完整生成接口

### 5.1 当前代码缺口与最小修改

| 位置 | 当前行为 | 本轮实现要求 |
| --- | --- | --- |
| [train_generation.py](../../scripts/train_generation.py) | 固定 pilot 配置/旧缓存，val 有文件截断 | 显式 config、training-manifest、cache/root-stats、out；未知键拒绝；debug 子集显式命名 |
| [losses.py](../../src/ireset/train/losses.py) | FK 分支内仍会计算手部接触，零权重不足以关闭 | 接触计算以 w_contact>0 单独门控，W1 不读取认证 mask 或回退参考 contact 作监督 |
| [trainer.py](../../src/ireset/train/trainer.py) | validate 默认最多 40 batches，去噪验证共用随机链 | 全部准入 val 窗口、按有效样本/帧加权、固定 t/noise、训练/验证 RNG 隔离 |
| [eval_generation.py](../../scripts/eval_generation.py) | 默认 val8/前120帧，构建修复流程 | 固定 expected-set、完整长度、显式模型/清单身份；W1 raw 路径不构建或调用 RepairPipeline |

引用列描述草案起点，不表示改动已经实现。正式训练 CLI 要显式接受清单与配置；缓存 CLI 要显式接受清单和目录；评估 CLI 要接受冻结 expected-set 和 full-length。先实现、检查 `--help` 和真实参数生效，再发布运行命令。

`w_contact=0` 时保留动作/FK/时序损失，手部 SDF 与接触监督分支不得执行。参考接触标记可以按既有条件协议作为可从源人体恢复的条件；它与机器人接触监督、质量标签是不同字段。

### 5.2 固定配置

拟新增 `configs/train/omniretarget_motion.yaml`，继承父方案 §6.2–§6.3：

| 参数 | 固定起点 |
| --- | --- |
| 模型 | 确定性与扩散；主干 240维、4层、6头、FFN×4，时间窗口31 |
| 状态 | 根平移3、旋转6D、关节29；解码到根位姿+29关节 |
| batch/精度 | micro-batch=4，grad_accum=4，有效batch=16，FP32 |
| 优化器 | AdamW，lr=1e-4，weight_decay=1e-5，grad_clip=1.0 |
| 扩散 | 1000步、prediction_type=sample、clip_sample=False；DDIM25、eta=0、无guidance |
| 损失 | pos=1、rot6d=1、joint=1、geodesic=0.5、FK=2、vel=0.5、contact=0 |
| 随机/加载 | seed=20261008，workers=4，每worker原生线程1 |
| 正式验证契约 | loss每250 optimizer steps、完整生成每1000步，完整生成seed=7 |

beta_schedule/start/end、限位映射和所有调度器实参按实际生效值入册。只对干净关节输出作既有映射，不裁剪根/旋转或前向噪声。时序项仍是标准化状态差分拟合，不称为机器人速度约束。两模型从头初始化，数据/主干规模/动作损失/有效batch一致；单卡顺序执行。

### 5.3 完整验证、长序列和模型身份

loss 验证覆盖所有准入 val 窗口，保存预期/处理样本和帧数。扩散验证 t/noise 单独固定、登记摘要，验证前后不改变训练随机序列或 DataLoader 顺序。

完整生成 expected-set 恒为 val31，而不是“有标签的子集”。拼接必须覆盖最后一帧；末端补一个以 T-60 起始的完整窗口，合并重叠处的状态与旋转遵循既有实现。T<60 明确记 unsupported，不能删除该任务。窗口训练规则与完整轨迹拼接规则分别测试。

后续 best 使用全部预期 val 的演示宏平均 teacher-fit FK MPJPE；缺标签或坏输出使该次完整选模证据不足，不能以遗漏演示获得 best。last 仍可交付诊断。另报人体任务误差、root/关节/旋转误差和数值失败，teacher-fit 不替代交互质量。

checkpoint 记录模型/条件器/配置、数据清单、统计量、协议/资产、来源和种子身份。当前只保存模型的历史 checkpoint 不构成精确 resume；本轮 S0 从头开始，预算内失败保留产物后停止。正式训练若实现 resume，须另验 optimizer/步数/RNG/数据顺序；否则 warm_start 作为新实验登记。

## 6. C：S0 数值烟测与小样本拟合

### 6.1 集合和执行顺序

S0 只使用 train，不用 val/test 调参。默认复用注册 train24 中的 pilot6，每段取首个完整60帧窗口作固定训练小集合与 probe；若某段未准入，登记缺口，不依模型结果换样本。先用 train24 的准入窗口检查正式 DataLoader 可形成实际 micro-batch，随后两模型在同一六窗口小集合上进行 S0 训练与拟合。S0 使用 train probe 验证，仅保存烟测曲线和last；完整val31验证与best选模在后续正式D阶段执行。

先检查一批编码/解码、条件与 FK，随后确定性模型至少100个 optimizer steps，再扩散模型至少100步。记录 optimizer steps、micro-batches、有效batch及实际调用数，梯度累积不能把4个micro-batches记为4个optimizer steps。

每模型最多1,000步，100步包含在上限内。前向/反向出现非有限、梯度异常、解码退化或身份变化立即停止；不靠关掉FK、改损失或换数据取得通过。数值问题定位修复后另分配 S0 attempt。

### 6.2 可核验的通过条件

1. 全部输入、标准化状态、条件、损失各项、有效梯度、权重和解码输出有限；四元数/旋转解码不退化。
2. 两模型完成至少100个真实 optimizer steps；记录梯度范数、学习率、采样t/noise身份、步时和峰值显存。
3. w_contact=0 的接触分支调用数为零，接触分支不存在梯度贡献；修改仅用于诊断的质量字段不能改变模型条件或动作损失。
4. 固定 train probe 上检查小样本重建趋势。从step0开始，每10个optimizer steps运行一次独立固定probe；扩散使用固定t/noise，probe不更新训练RNG。至少取得40次probe，使首20次与末20次不重叠；到1,000步上限前，末20次的平均动作重建误差须小于首20次。动作重建误差固定为状态空间 pos+rot6d+joint 三项的未加权和（权重本来均为1），不将接触或验证质量项加入该门。报告全部曲线和初/末值。该门只判断拟合趋势，不是泛化、搬运质量或显著性证据。
5. 保存 checkpoint 后重新加载，同一模型/条件器/统计下固定probe输出可复现；统计和清单身份一致，旧模型身份不被接受。

趋势不成立记 `fit_not_demonstrated`；先定位条件、归一化、目标、优化器或接口问题，保留负结果。达到预算或出现缺证据记 `incomplete`，不自动进入4k训练。

## 7. 必要反例与真实验收

不为文档常量增加测试；验证实际行为，保留足以识别修复前缺陷的反例。

| 范围 | 必须覆盖的反例/对照 |
| --- | --- |
| 入口 | UR报告改裁决/门/完整性、注册哈希漂移、输入与资产漂移均拒；当前完整 no_quality_gain 合法 |
| 数据 | human末维错误、fps非法、非有限、退化四元数、关节超限、缺PT、重复/跨split/缺候选行；合法轻微归一化不改源数据 |
| 场景 | original常数z偏移保留，source/target分离；仅改教师机器人状态时条件不变；源几何尺度和解析坐标往返对拍 |
| 消费 | training/label_manifest互斥；实际读取路径的哈希漂移拒绝；来源/协议错、缺文件不回退；旧graded行为保持 |
| 统计 | 改val/test不改变train统计；重叠窗口不重复计帧；改源/映射/配置导致缓存失效；合法同身份缓存可复用 |
| 损失 | w_contact=0不调用接触专用手部FK/SDF或读接触监督；动作FK梯度保留；未知配置键和未生效实参被识别 |
| 验证 | 超过40 batches/60文件的解析用例不截断；加权正确；固定去噪验证不污染训练RNG；坏输出/空结果不能改善best |
| 生成 | 非窗口整倍数覆盖末帧；失败/超时不缩分母；raw路径零RepairPipeline调用；模型/统计错身份拒绝 |
| S0 | 真100 optimizer steps/梯度累积计数；固定probe可复现、趋势规则可重算；运行失败与数值失败分别落盘 |

实现阶段先运行相关测试，再运行项目所需全量检查。正式A3完成后抽核已冻结pilot6与现有源箱配对数值一致、55行完整、新增18段配对实证、实际缓存/统计身份。最终真实S0结果与机读就绪判断交叉核验；测试通过不能代替真实运行。

## 8. 提交单位与依赖顺序

| 单位 | 主要范围 | 出口 |
| --- | --- | --- |
| W0 | 运行注册、锚点重推、具体名单与输入/资产哈希 | 可追溯入口，未改变U0/UR裁决 |
| I1 / A3 | 新协议、审计薄入口、场景适配、55段结果清单/质量记录 | motion准入与质量状态分离，完整分母 |
| I2 / B | teacher/windows、清单消费、独立缓存与train统计 | 来源身份闭合、条件无泄漏 |
| I3 / B | 显式训练配置/CLI、损失门控、完整验证和模型身份 | 真实参数生效、完整统计、无接触监督 |
| I4 / B | 完整长度raw评估入口、expected-set和独立验收适配 | 末帧覆盖、失败保留、不调用repair |
| I5 / C | 注册S0、两模型烟测/拟合、汇总与就绪判断 | A3/B/C完成或有证据的停止 |

独立接口/反例可以先实现；运行次序必须 W0→A3→缓存/统计→S0。实际调用与登记顺序保持一致，不先跑质量/模型后追补注册。开发和正式实验分开目录；长运行期间不写影响身份的仓库文件。

## 9. 运行目录、成本与恢复

拟使用 `derived/experiments/omniretarget_first/<date>_<revision>_w1_a3bc/`：

```text
freeze/                    preregistration、锚点、名单、输入与资产闭包
audit/attempt_NNN/         配对、质量、training_manifest及汇总
cache/attempt_NNN/         clip条件、train root_stats、身份清单
s0/deterministic/attempt_NNN/  数值/拟合曲线、checkpoint、调用成本
s0/diffusion/attempt_NNN/      同上
decisions/attempt_NNN/     A3/B/C事实、未满足门、w1_readiness
costs.json                每次实际阶段调用的时间与状态
```

新验证记录在实施后写入 `docs/verification/omniretarget_w1_a3bc_verification.md`；本次不创建伪实施记录。目录和注册采用独占分配，不覆盖旧attempt。复用任务必须核验实际消费子集的输入、协议、配置、资产和产物链；完整候选集合仍独立对账，不能借消费裁剪缩小分母。

输出resolve后须位于项目derived内，并检查软链接目标；拒绝写入OMOMO、发布NPZ、原始模型、上游checkout或历史实验根。路径记录采用path_for_record的项目内相对/项目外绝对规则。

| 阶段 | 初始运行上限 | 预算出口 |
| --- | --- | --- |
| W0+A3，55段基础/质量审计 | 2小时CPU墙钟 | 未处理/超时逐行登记，不能声称A3完成 |
| B缓存/统计实测 | 2小时CPU墙钟 | 保留实际帧数、缓存/统计进度，身份不闭合即停止 |
| C，两模型S0合计 | 1小时GPU墙钟；每模型最多1,000步 | 保存last/曲线，未达门为incomplete或fit_not_demonstrated |

接口开发与正确性修复工时另记。运行前检查实际磁盘/显存/进程占用；限制BLAS/OpenMP及worker线程，不切换主项目Python/依赖锁。首个100步记录实际步时与显存，按剩余预算估计可完成范围。

成本使用实际非负有限耗时，逐行与汇总对账；缺历史、坏历史、未知成本不能作零。重试前登记具体修复理由，预算含失败尝试；当前调用超时受剩余预算约束。预算用尽保留原因并停止，不临时延长小样本拟合或替换更容易数据。

## 10. 就绪判断与后续边界

拟交付机读 `w1_readiness` 和验证记录，分列 `registration_complete`、`a3_complete`、`b_complete`、`c_complete`、`ready_for_w1_d` 及每项原因。程序正常完成登记可以exit0；任何就绪门缺失时不能以exit0宣称训练可启动。

`ready_for_w1_d=true` 必须同时满足：入口锚点和来源用途明确；55个候选均有完整结果；六受试者各有准入训练窗口；实际窗口量满足micro-batch=4且数据加载无空循环；训练标签质量登记完整；协议/条件/统计/模型身份可追溯；B接口验证通过；两模型S0全部通过。每个门由事实重算，缺值与有证据的零分开处理。

val的标签准入与完整生成任务集合分别登记。标签不可用时继续保留条件有效任务和teacher-fit缺失行；若因此无法完整选模，必须在就绪交付中说明该限制，后续不得称取得合格best或完整宏平均。基础准入失败不引入“所有标签strict通过”或旧18/24 keep门。

A3/B/C通过只交付正式W1的工程入口，原U0 `incomplete`、UR `no_quality_gain`及最终可行性目标不变。本轮停止于就绪报告；正式D/E运行依后续明确任务安排执行。

后续W1-D/W1-F每模型最多4,000 optimizer steps、两模型累计GPU墙钟最多12小时（含训练期间验证），不续到8k/12k。两模型使用同一发布original清单，完整val31宏平均选模；最终比较Teacher raw、Deterministic raw、Diffusion raw并复用val6做详细分析。诊断全拒也须交付0/N和失败分类，不以取得非零通过率为理由启用修复或改教师。

## 11. 本轮交付清单

- [ ] W0：运行预注册实际落盘，当前锚点/身份重推通过。
- [ ] A3：train24/val31逐行审计、场景配对、动作准入和质量证据完成。
- [ ] B：协议、显式清单、独立缓存/train统计与训练/推理条件适配完成。
- [ ] B：接触分支关闭、完整验证/完整长度raw入口与身份校验完成。
- [ ] C：两模型至少100步及至多1,000步S0完成，固定probe趋势可复核。
- [ ] 必要反例、实际资产/输入/产物哈希和所需项目检查通过。
- [ ] 就绪/未就绪报告、未满足原因、成本和实施验证记录完整。
- [ ] D/E：本方案中仅固定契约，运行状态保持not_run。
