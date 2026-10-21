# W1预测误差与窗口一致性诊断执行方案

日期：2026-10-21。版本：v0.1。状态：方案已起草；本文规定的实现、执行注册与正式诊断尚未运行。登记后冻结本文字节，实际结果另写验证记录。

本轮承接[父方案 §9](omniretarget_first_training_plan.md#9-何时处理或更换教师)、[W1-D/E §7.3](omniretarget_w1_de_plan.md#73-路线判断的证据要求)、[失败归因记录](../verification/omniretarget_w1_attribution_verification.md)和[旋转融合验证记录](../verification/omniretarget_w1_rotation_fusion_verification.md)。交付目标是把“融合前预测残差显著”分解为可复核的单窗误差、同帧跨窗分歧、推理尾窗差异及扩散去噪/采样差异，为下一项单因素实验确定依据。本轮训练更新数固定为 **0**。

## 1. 当前事实与要回答的问题

### 1.1 已有结果

- W1-D/E/F 的动作基线证据完整；Teacher raw 通过 2/31，两模型各 0/31，最终交互方法为 `not_ready`。两模型各训练 4,000 步，best 均为 step4000，裸输出最坏穿透约 222/247mm，双手认证覆盖为 0。
- 既有归因发现两类旋转融合机制，也发现全部 26 条模型轨迹在解码窗中已经穿透；train6 教师拟合约 5.8/5.9cm，val6 约 11.5/10.0cm。融合不是全部失败的充分解释。
- 固定路径 NLERP v2 的权威实验仍为 r10：GS 失败，GQ **3/26 配对通过、94 项退化、0 缺口**；V 按注册条件未执行，候选不接入。3/26 是不退化配对门，不是 raw 接受率。
- r10 经新代码只读复验：冻结重放 65/65、Teacher 标签 13/13、V2 重推 39/39（38 measured + 1 拒收）、重测 116/116；GS 失败集合不变。旧 `fusion_final.json` 未用新核验字段重新封口。
- 原 U0 `incomplete`、UR `no_quality_gain`、原 W1 F `not_ready` 保留。教师缺陷已知，但“教师是当前模型质量的主要瓶颈”仍无充分证据；学习、条件信息与数据覆盖的责任尚未分离。

### 1.2 固定研究问题

| 问题 | 本轮可取得的证据 | 解释边界 |
| --- | --- | --- |
| 单窗是否已偏离教师，偏差落在哪里？ | 归一化损失、根平移/旋转、29 关节、FK/手部误差与关键帧几何 | 拟合教师不等于满足交互质量 |
| 同一帧在不同窗口中是否预测一致？ | 所有正权重贡献的成对差异、窗内速度、融合前后同帧比较 | 跨窗分歧是可测现象，不自动证明某个训练损失缺失是原因 |
| 误差是否集中在窗边或推理额外尾窗？ | 常规/尾窗、局部位置分箱、相同全局帧配对 | 尾窗与常规窗内容和上下文不同，相关性不等于因果 |
| Diffusion 对教师加噪状态的重建与实际 DDIM 采样有何差异？ | 五个固定噪声时刻的 x0 重建，与同 checkpoint 实际采样对照 | 加噪输入含教师状态，不能冒充从条件生成的效果 |
| 下一项干预应优先检验什么？ | 分组量级、逐例证据、可复现反例和解释的未决项 | 本轮只提出另立实验的优先级，不直接更换教师、续训或接入方法 |

保持模型、checkpoint、输入、资产、条件、源箱尺度、目标箱、缓存、root_stats、seed7、DDIM25 及原验收阈值。范围为下述 13 段；不新增融合候选，不执行 val31 全集生成，不使用 test31，不扩大数据、不改接触权重、不训练或启动修复器。

## 2. 只读前置、来源与身份

以下三个运行根均位于 `derived/experiments/omniretarget_first/`，分别简称 SOURCE、ATTR、FUSION：

| 简称 | 运行目录 |
| --- | --- |
| SOURCE | `20261013_be00c50_w1_de` |
| ATTR | `20261015_9000ac8_w1_attribution` |
| FUSION | `20261019_worktree_w1_rotation_fusion_r10` |

起草时已从盘上重哈希的入口如下。正式 R 仍须重新核验文件及其传递消费闭包，不能把本文的哈希表当作已完成准入。

| 文件（相对对应运行根） | SHA-256 |
| --- | --- |
| SOURCE `freeze/execution_registration.json` | `5561f68562189b7ba45f7193c280739e6dde5325b4df2ada82e8a9cf69ba5f7e` |
| SOURCE `decisions/attempt_001/w1_de_final.json` | `7f1a9a595cf7964e99a929c83d3bcaac4aa3482a96e515228921b7b67a576737` |
| SOURCE `evaluation/checkpoint_selection.json` | `a49f0ffea1d51dd103f0aa728794e7c891ae3c42fe59075519d68c452617b9af` |
| ATTR `freeze/registration.json` | `499257b0a983b8af1c94a3d819d573d9be5470cb97dcdf4c02a84ac03d4bab52` |
| ATTR `diagnosis/attempt_001/attribution.json` | `72186392cce075ce5893c68f82acaa41fd3efd45b534970696f698f5563c07ac` |
| ATTR `decisions/diagnostic_closure.json` | `93cac087a881e1a05c3f38d3542208e7690aa69afaaa30a626999960136c9064` |
| FUSION `freeze/execution_registration.json` | `6ad7649e060eb078f6e048b946e447c5337a6a50c970db38bf33603be6df5400` |
| FUSION `decisions/attempt_001/fusion_final.json` | `055432bf9d201408175a1b560ec9a69cdc01f3879fc9be6f7ff25247f56b9abe` |

checkpoint 只消费 SOURCE 选模记录指定的文件，角色与 step 也须对账：

| 模型 | 固定文件（相对 SOURCE） | SHA-256 |
| --- | --- | --- |
| deterministic | `formal/deterministic/attempt_001/best.pt` | `7b1efd57ceff6510d95c5fd682649c7900297030923904c3a7542f59af7b2eff` |
| diffusion | `formal/diffusion/attempt_001/best.pt` | `1f9c16b124ab2d77877e93cfa39af124aa30a4f5aeb998f05350de26dcd437b1` |

消费侧绑定包括：SOURCE 的 manifest/preregistration、derived clip、实际 PT/NPZ、资产闭包、键控日程、similarity_scale、root_stats、checkpoint 内完整配置/身份/权重；ATTR 的注册来源、26 份保存窗、78 份轨迹及参考；FUSION 的冻结注册、635 锚及历史实现快照。文件按其记录的实际路径核验，不能用规范目录替代消费路径。

**历史实现与当前实现分别登记。** r10 后审计修复已合法改变驱动和测试，历史核验应把注册的实现哈希对到 `freeze/implementation_snapshot/` 内的历史字节；本轮诊断代码另存当前 HEAD、tracked diff、实现文件哈希与快照。不得要求当前所有 r10 实现文件仍等于历史版本，也不得关闭来源核验、重写旧报告或把新代码只读复核宣称为新的 r10 F 封口。条件代码、模型与解码等实际被本轮执行的依赖必须与当前注册一致；发现不在预期修复范围内的漂移即停。

## 3. 固定集合、窗口规则与分母

沿用 ATTR 按身份选定的 train6、注册 val6 和已知案例。每个 demo 均为 `largebox`，表中采用完整名称。已知案例属于已暴露的验证数据，单列，不并入 val6 宏平均。所有 val6 也已参与既有诊断，不称作未接触的留出集。

窗口长 F=60、常规 stride=30、fps=30。常规起点为 `range(0, T-60+1, 30)`；推理起点再追加尚不存在的 `T-60`。每窗沿用整段的同一个 `world_from_clip`，不按窗口重新归零位置或 yaw。

| 组 | demo | T | 常规窗 | 推理窗 | 额外尾窗起点 |
| --- | --- | ---: | ---: | ---: | ---: |
| train6 | sub10_largebox_063 | 133 | 3 | 4 | 73 |
| train6 | sub12_largebox_047 | 358 | 10 | 11 | 298 |
| train6 | sub16_largebox_000 | 89 | 1 | 2 | 29 |
| train6 | sub3_largebox_001 | 178 | 4 | 5 | 118 |
| train6 | sub7_largebox_003 | 147 | 3 | 4 | 87 |
| train6 | sub8_largebox_005 | 190 | 5 | 6 | 130 |
| val6 | sub10_largebox_100 | 73 | 1 | 2 | 13 |
| val6 | sub12_largebox_008 | 161 | 4 | 5 | 101 |
| val6 | sub16_largebox_052 | 147 | 3 | 4 | 87 |
| val6 | sub3_largebox_018 | 191 | 5 | 6 | 131 |
| val6 | sub7_largebox_032 | 184 | 5 | 6 | 124 |
| val6 | sub8_largebox_001 | 151 | 4 | 5 | 91 |
| known_case | sub16_largebox_048 | 168 | 4 | 5 | 108 |
| 合计 | 13 段 | — | **52** | **65** | **13 个尾窗** |

常规窗分组数为 26/22/4，推理窗为 32/28/5；只有 **train6 的 26 个常规窗**属于训练窗键，val/known 的常规窗不能称为“训练已见”。26 份模型保存窗合计 **130 个模型窗**，对应 65 个唯一 `(demo,start)`。R 从原注册、真实 T、起点与文件 shape 交叉重建上述集合；不是从成功加载的行反推预期集合。

W 的主键为 `(demo,model,start)`，分母 130；共享 Teacher 参考为 65 个键。P 的主键为 `(demo,start,probe,t)`，完整矩阵见 §6，分母 520；只有 `diffusion_denoise` 的 t 为注册整数，其余三类固定为 null。跨窗比较的预期键为同帧所有正权重窗口的无序对 `(demo,model,global_frame,i,j)`；R 从固定起点和原 `window_weight` 枚举、冻结键集与数量。零权重窗不参与融合贡献，但其整窗预测误差仍保留。

## 4. 实现与阶段顺序

顺序固定为 **I → R → W → P → F**。W/P 是新研究，不以旧融合 GS 通过为前提；只有本轮的身份、输入与证据门通过才进入下一阶段。质量负结果不阻止诊断收口，缺证据会阻止 `diagnosis_complete`。

| 交付文件（拟新增） | 职责 |
| --- | --- |
| `src/ireset/eval/prediction_consistency.py` | 单窗误差、同帧跨窗配对、时间与分母验证、聚合及纯事实判定；不包含网络或融合候选 |
| `scripts/diagnose_w1_prediction_consistency.py` | `register/saved-windows/probes/finalize` 四入口；注册、阶段调度、薄适配、实际推理捕获与证据写入 |
| `tests/test_w1_prediction_consistency.py` | 数学/坐标、真实接口、固定键集、来源绑定、坏证据、RNG 和预算反例 |
| `docs/verification/omniretarget_w1_prediction_consistency_verification.md` | 实施后才新增，登记真实结果、失败、成本和下一候选 |

复用 `TeacherWindowDataset`、W1 准入/缓存消费核验、`load_w1_de_generator`、`StateTokenizer`、`LossAssembler`、现有采样观察器和几何度量。旋转比较用 SciPy `Rotation`；扩散加噪/采样用已有 `DiffusionG1` 与 Diffusers 调度器。训练损失、网络、坐标变换、解码器及台账封口不再复制一套；需要适配时用最小包装，生产默认行为不变。新文件不加入 `CONDITION_CODE_FILES`，因为它不参与生产条件构造或缓存内容；若实施发现必须改条件代码，应停止本契约并明确协议修订。

I 使用合成模型、解析动作、临时文件与已公开机制反例验证接口。正式 checkpoint/真实数据的新测量或模型调用在 R 后进行并计费；R 前仅允许核查文件身份、既有报告和集合元数据，不提前跑探针选时刻或选样本。

### 4.1 R：独占注册与执行前准入

运行根为固定 `derived/experiments/omniretarget_first/` 的直接子目录，建议 `<date>_<impl>_w1_prediction_consistency/`；resolve 后拒绝软链接逃逸、嵌套根及 SOURCE/ATTR/FUSION 的祖先或后代。已有非空根不覆盖，attempt 只追加。

R 在实质核验前事前登记 running attempt，再独占写 `freeze/execution_registration.json`：本方案/实现快照、§2 的来源闭包、§3 的精确集合、P/F 矩阵与顺序、噪声种子算法、数值容差、指标 schema、环境、预算与停止条件。注册中分开记录历史事实和本轮待测假设。

重新核验 SOURCE 六个 stage、四预算域及 cache 的盘×索引×台账；按实际消费文件重哈希和深度加载两 checkpoint，核对 strict state、有限参数、内嵌统计与当前 root_stats 逐位一致。ATTR 保存窗的路径和哈希必须绑定到其原注册来源，不能接受一份自报哈希相同逻辑的替代窗。运行前后锚包括旧方案、155 原 W1 产物、ATTR 及 FUSION 已登记证据；新增诊断产物只写新根。

每单元开始核验其输入锚，阶段结束与 F 再核对完整闭包。注册后实现漂移不能继续旧尝试；修正重注册保留所有历史费用与未完成记录。

## 5. W：保存窗误差与一致性分解

W 学习模型调用数为 **0**。每段读取两个保存窗文件，重建 Teacher 目标、条件、窗口与已有 V0/V1 轨迹的绑定；输出 130 个模型窗误差行、跨窗配对事实和每段帧级诊断。V0 为原生产融合，V1 为 ATTR 已冻结的全贡献顺序 SLERP，仅解释既有结果；不运行新 V2 候选或重建旧 117 行 S 实验。

### 5.1 输入、目标与表示核验

1. 校验归一化预测 `[N,60,38]`、decoded pos/quat/joints、starts、五类条件的 shape、dtype、有限性、合法旋转与窗口顺序；四元数先验单位范数，旋转比较前禁止用单位姿态替坏值。坏数组登记原因，不能中断整个报告路径或变成零误差。
2. 65 个 Teacher 目标由冻结 clip `root/q` 与原 `StateTokenizer.encode` 构造 float32 x0。52 个常规窗与真实 `TeacherWindowDataset` 对账，13 个尾窗用同一编码和条件切片构造。Teacher 身份绑定冻结标签字节，不通过自参考误差为零来证明来源。
3. 五类条件 `kps_obj/kps_clip/box_t_clip/box_quat/contact` 与原保存值逐键 dtype/shape/逐位对账。常规窗还与数据集项对账，尾窗明确 `inference_only_start=true`。接触条件只来自人体参考；质量标签和教师根/脚落点不进入 conditioner。
4. 从保存 normalized_prediction 经原生产 Torch 解码公式重建 decoded，与记录核验；本机同设备重建要求逐位相等，另报告物理旋转等价，不能只比较保存的 `decoded_rotation_error_deg`。跨设备验证单列 §6.3 的容差和是否逐位相等；分歧不得静默归因于融合。

### 5.2 每窗必测字段

| 层 | 记录量与单位 |
| --- | --- |
| 训练目标 | 原 `pos/rot6d/joint/geodesic_rad2/vel/fk_m2/total`，按 checkpoint 完整权重重算；`w_contact=0` 保持接触分支静默 |
| 物理状态 | 根位置误差 m；SO(3)相对角 rad/deg；29 关节误差 rad，逐关节及聚合；原始归一化分量另列 |
| FK | 15 解剖 site 教师拟合误差 m；左右手 site 误差分列；world/clip 变换互相核验 |
| 时间 | 窗内 59 条差分的根线速、SO(3)角速、各关节速度及教师差值；定位最大值/首败全局帧；首帧差分记未定义 |
| 关键帧几何 | 原世界系/目标箱/人体下的穿透深度及违规、手部间隙、脚部几何和相对 Teacher 的新增/共享失败 |
| 位置与参考 | 起点、是否尾窗、局部帧号、人体参考抓握/支撑帧数、首次重叠前/后、正权重贡献数 |

归一化各组 MSE 与物理误差不可互换，38 通道平均不能替代各组结果；`vel` 是原 x0 一阶差分误差，不是物理速度硬约束。逐窗误差保留 60 帧曲线，并按固定局部区间 `[0,10)、[10,50)、[50,60)` 汇总；区间不随结果调整。

物理指标固定定义：逐帧根误差 `e_p=||p_pred-p_teacher||_2`，`root_rmse=sqrt(mean(e_p**2))`；旋转误差为 SciPy 相对旋转 `R_teacher^-1 R_pred` 的角度；关节 MAE/RMSE 在全部 60×29 个标量上计算，不对有限位关节做周期折返；`fk_mpjpe` 为全部帧和 15 个 site 的欧氏距离均值，左右手另按固定 site 集合计算。帧曲线给 mean/p95/max（p95 使用 NumPy linear 分位法），逐例报告最大值位置；训练损失仍以原 `LossAssembler` 定义为准。

几何复用 `w1_de_frame_series` 的锁定模型与原测量口径。这里是逐关键帧诊断，不为 130 个窗执行/宣称完整自适应整段验收；支撑/抓握认证也不能用离散近接代替。左右手和脚独立记录，不因无手参考帧而跳过脚。参考帧数从冻结区间复算，必须是非负整数，缺键/NaN/分数不当零；0 参考帧才允许相应均值为 None。

### 5.3 同帧跨窗与融合放大

在原全局帧 t，对全部正权重贡献窗两两计算位置差、SO(3)角差、关节差、FK/手部差。保存窗口 i/j、局部帧、原始/归一化权重、成对值及 Teacher 误差。三窗时必须保留全部三对，零权重不能挤掉中间正贡献。

逐帧比较窗口与既有 V0/V1 的关键帧穿透，固定为三类：所有正贡献 `pen_i>0.002m`、所有正贡献 `pen_i<=0.002m`、混合；各类再分融合通过/失败。只有“所有贡献均过门、融合失败”可称该口径下的 `fusion_only`。计数总和须等于原 T；不把它解释为非线性误差的可加因果百分比。报告 Teacher 同帧是否失败及新增深度，区分继承缺陷与模型新增误差。

时间比较另列窗内坏跳变和融合跳变：对 `(t-1,t)` 只用同时覆盖两帧且两帧均为正权重的窗比较速度；没有共同正贡献时登记 `activation_transition`，不删分母。融合角速度由实际四元数和 frame_dt 重算，单位 rad/s；seam 单步角为度，不能把 rad/s 直接转度。原 7rad/s 门保持，记录全局/seam/非seam最大值。位置/关节不变也可能因根旋转改变 world FK 与接触。

已知案例必须保留固定帧邻域 `106..121`，特别是 Det 的窗起点 90、116→117 窗内跳变，Diff 的 107→108，以及既有 v2 frame120 拒收事实。该邻域是已暴露回归例，不计作新独立验证案例；旧 v2 拒收不以补零路径或缺产品豁免本轮输入完整性。

### 5.4 W 阶段门

`w_evidence_complete` 要求：精确 130 窗键、65 参考键、完整跨窗键集、无来源/解码/条件绑定失败、必测值与定义掩码齐全、所有 T 帧覆盖、计数闭合、前后锚无漂移。误差大、穿透多或一致性差本身不使 W incomplete；它们是诊断结果。W 工程缺证据时 P 不执行，F 仍登记已完成部分与缺口。

## 6. P：固定 checkpoint 单窗探针

### 6.1 唯一矩阵与噪声

对 §3 全部 65 个 `(demo,start)`，按以下固定矩阵执行。每行覆盖完整 60 帧，不选最好时刻/seed。教师状态只出现在 oracle 或明确标记的加噪诊断输入，不进入实际生成条件。

| probe | 每窗单元 | 总行数 | 输入/输出 |
| --- | ---: | ---: | --- |
| `teacher_roundtrip` | 1 | 65 | 冻结教师编码，经原生产解码；无学习模型 |
| `deterministic_generation` | 1 | 65 | 原五类条件 → 原确定性模型 |
| `diffusion_denoise` | 5 | 325 | `t={0,250,500,750,999}`，教师 x0 的同一 epsilon 加噪 → 预测 x0 |
| `diffusion_ddim25` | 1 | 65 | 原条件、seed7、完整窗批布局、原 DDIM25 → 实际采样 x0 |
| 合计 | **8** | **520** | 其中学习模型输出 455 行，oracle 65 行 |

加噪调用 `DiffusionG1.add_noise`，预测调用原 `forward(x_t,t,memory)`；调度器必须为 checkpoint 注册的 1000 步、scaled_linear、x0-prediction、clip_sample=false。记录每个 t 的真实 `alpha_cumprod`/SNR；**t=0 仍是第一档加噪输入，不是无噪声训练证明**。

固定 `noise_seed=20261021`。每唯一窗口的 epsilon 种子由 UTF-8 字符串 `w1_prediction_consistency_v1|20261021|<demo>|<start>` 的 SHA-256 前 8 字节按 big-endian 转整数再模 `2**63` 得到；CPU 专用 `torch.Generator` 产生 float32 `[60,38]` 的 epsilon，再复制到设备。五个 t 复用该 epsilon。种子公式、实际 epsilon/x_t/t 和哈希全部保存，F 从注册重新生成；不能只信保存 seed 或自报输入哈希。

### 6.2 真实调用、接口与随机流

按 §3 表序逐段执行，每段固定 oracle → Det → Diff 五 t（升序）→ DDIM25。每次模型批保持该 demo 的完整推理窗顺序，最大 11 窗；不得把 seed7 拆成不同 micro-batch 后声称重放旧采样。eval/no_grad、FP32、原注册 CUDA 设备；专用随机流和保存/恢复模式/RNG，确保探针不消耗全局训练/验证 RNG。所有 conditioner 实际输入严格限五类键，教师 x0/x_t 不混入条件。

每种实际生成会话最多一次尝试，本诊断不触发生产的数值补采样：数值失败留失败行与原始张量，不换 seed，不以质量失败重试。生产入口原来的最多三次数值重试配置不改；诊断适配器显式用 `resample_attempts=1`，并核对旧保存窗是否首尝试成功。

完整 P 正常调用为：13 次 Det 批前向、65 次加噪批前向、13 次 DDIM 会话；DDIM 每会话 25 次去噪网络前向。合计 91 次高层模型调用、**403 次学习网络前向**。conditioner 调用、采样会话、网络前向、窗输出与重试分别实记，不能把 DDIM25 算成一次网络前向。解码与测量亦计入 P 墙钟。

### 6.3 重放、输出与 P 阶段门

Det 与 DDIM25 的 normalized 和 decoded 窗须对账 ATTR 保存窗。报告逐位相等和容差结果；固定最大绝对差为 normalized 1e-5、位置 1e-5m、关节 1e-5rad、相对旋转 0.001°。同环境预期逐位重放，但不把记录值与 CPU 重放的容差一致写成逐位相等。超容差记 `baseline_window_replay_mismatch`；即使容差内，若跨越速度/几何门导致分类改变，也须单列 `replay_gate_disagreement`，不得当作同一质量事实。

每行保存条件身份、目标/噪声/输入、预测 normalized、原生产 decoded、world_from_clip、checkpoint、配置、调度器、原始损失与 §5.2 的物理/关键帧度量。325 个加噪重建与 130 个实际生成行分表、分图，不合并计算“生成通过率”。

Teacher oracle 的原绝对参考门仍为位置/关节 1e-5、旋转 0.001°，另测 FK 误差；既有整段 3/13 旋转负门不覆盖或放宽。新 65 个单窗与旧 13 条整段是不同分母，应分别报告。oracle 质量/精度负结果是完整诊断证据；解码、身份或参考不明则是工程缺口。

`p_evidence_complete` 要求精确 520 键、全输入/输出绑定、所有规定字段/定义掩码有效、实际调用可审计且预算可证。数值异常/缺输出使对应行 `execution_failed`、P incomplete，不能冒用旧 v2 的 `algorithm_rejected` 豁免。物理质量失败仍可 `measured`。

## 7. 聚合、解释与下一项实验的选择

所有分组先算每 demo 内统计，再对 train6/val6 等权宏平均，known 单列；按常规/尾窗及局部位置分箱同时给数量。原始窗/帧微平均可以附表，但不能把重叠窗当成独立演示增加样本量。跨窗配对只在其定义域统计，非重叠帧保留不适用计数。

零参考帧或无重叠的合法 None 与缺测量分别编码。NaN 不作有效数值；JSON 不写裸 NaN/Inf，曲线中允许的 NaN 仅限登记的未定义掩码，并拒绝 ±Inf。任何必需行失败时，不发布缩小分母后的“全集宏”；可计算子集统计必须同时列 `n_expected/n_measured/n_missing`。计数不得 int() 截断，违规数量上升单独按“不得增加”识别。

| 事实组合 | 允许的判断与后续候选 | 仍不能得出的判断 |
| --- | --- | --- |
| 条件/编码/解码/重放绑定失败 | 先修可复现的实现问题，另列受影响证据 | 不能用坏证据评价学习或教师 |
| train 常规窗已有显著物理/FK 误差，远大于 oracle 精度量级 | 优先另立训练拟合/目标或实现诊断；逐例列根/关节/手部量级 | 固定 checkpoint 的差拟合不能区分容量、优化与条件信息不足 |
| 加噪重建误差随 SNR 很不相同，实际 DDIM 又明显不同 | 登记去噪/生成差距，下一方案检验单一采样或训练因素 | 教师加噪重建好不证明实际生成好，也不单独证明采样器有错 |
| 同帧窗间分歧大，且窗内/融合时间错误均有证据 | 分开提出窗口上下文/一致性监督与融合路径候选 | 本轮不把二者混在一次改动，也不接入被拒 NLERP |
| train 与 val、常规与尾窗存在差异 | 提出覆盖/上下文的受控对照候选，保留逐例量级 | 不是扩数据必有效、尾窗导致错误或条件足够的因果证明 |
| 模型在相同处继承教师缺陷且预测误差较小 | 按父方案要求至少 3 个不同 train 原始演示比较缺陷与残差量级，再提教师候选 | 三例不是总体证明；不能用 val 调教师参数 |
| 多种机制共存或证据不足 | `mixed_evidence` / `attribution_insufficient`，列出最小可区分实验 | 不强行指定唯一主因或编造责任百分比 |

误差“较小/显著”不新增一个任意 teacher-fit 准入数值：必须给绝对量级、相对同例 oracle/Teacher 缺陷的比较及逐例曲线；表中是后续研究优先级，不是自动执行分支。train6 负责提出下一干预假设；已暴露 val6/known 只提供固定描述与回归，不能选择噪声时刻、权重、阈值或 seed。正式因果分离、训练干预、教师比较及新融合/修复各须另立可区分的事前实验。

## 8. F：独立重推与诊断收口

F 先重核 R/W/P 的盘×索引×台账、预算与前后来源锚，再从注册重建全部预期键，不信 summary 的通过标记。对所有 W/P 行从输入和预测工件重算字段、定义掩码、速度、跨窗对账、关键帧度量和全部聚合；scene、行清单、阶段 summary 三方精确对账。损坏 NPZ/非映射 JSON/缺字段/零范数四元数等转为结构化失败，继续保存可得证据，不让异常逃出收口。

固定独立学习重放子集为 **sub10_largebox_063、sub10_largebox_100、sub16_largebox_048**，按身份事前选定，覆盖三组而不按结果挑选。F 从当前绑定的 checkpoint/条件/噪声重新执行该三段完整 P 矩阵：4+2+5=11 个窗口、**88 个重复核验单元**；其中 77 个学习输出、11 个 oracle。正常为 21 次高层模型调用、93 次网络前向；列入 F 预算，不增加 P 的 520 分母。按 §6.3 核对实际输出与记录，其余行也必须完成全部度量和绑定重推。

F 的 CPU/float64 度量重推与生产 FP32 输出分开比较：同一序列化事实/计数/键集要求精确相等；模型重放按 §6.3；几何/FK 标量容差 1e-6m、关节 1e-6rad、旋转 1e-6rad。数值有限性先于差值比较。阈值两侧翻转必须披露为不一致，不能只因绝对差小而豁免。

独立物理测量基于已验证的生产 decoded 状态；float64 相对旋转仅用于准确量角，不能用另一种更精确的解码器替换生产四元数。原训练损失按原 FP32 `LossAssembler` 重算，另列其设备/精度；不能把 CPU/float64 重编码的误差当成同一训练损失或抹去既有近 π 往返误差。

终局分列以下状态，不把诊断完成写成方法就绪：

| 字段 | 判据 |
| --- | --- |
| `implementation_contract_passed` | 规定的合成/接口反例与独立重推通过，失败列表为空 |
| `complete_evidence` | R/W/P/F 固定集合齐全、无缺测量/绑定问题、预算/历史可证、封口可消费 |
| `diagnosis_complete` | 上两项均 true，`failures=[]` 且 `evidence_gaps=[]`；大误差和物理负结果不使其变 false |
| `attribution_status` | `mechanism_supported`、`mixed_evidence` 或 `attribution_insufficient`，各自绑定逐例事实；可与诊断完成并存 |
| `final_interaction_method` | 本轮恒为 `not_ready`，不能通过诊断升级 |
| `training_steps` / `production_integration` | `0` / `false` |
| `next_action` | 候选问题、所需证据与预算建议；`auto_execute=false` |

已完成注册而 W/P 未执行、超时或证据坏时，F 仅收集可核验部分并完成诚实的 incomplete 登记，明确未执行键与原因，不补做越界工作。R 无法完成注册时只留下 R 的失败报告/成本，不启动 W/P/F。缺证据不能改称完整算法负结果。旧 r10、原 W1 F 及原 U0/UR 的状态分别引用，不互相替代。

## 9. 预算、历史与停止条件

### 9.1 新问题预算及旧 cohort 边界

本轮是固定 checkpoint 的**预测/窗口诊断**，独立登记 `experiment_id=w1_prediction_consistency_v1`，累计墙钟上限 **7,200s**，单进程/子进程顺序执行。阶段上限不可互借，预留已包含在上限内：

| 阶段 | 累计上限 s | 收口预留 s | 覆盖 |
| --- | ---: | ---: | --- |
| R | 600 | 60 | 来源重哈希、注册、准入与登记 |
| W | 1,800 | 120 | 保存窗/目标/重叠/帧级几何与报告 |
| P | 3,000 | 180 | 模型加载、全部探针、解码、测量、保存 |
| F | 1,800 | 180 | 全量重推、固定三段模型重放、终局和登记 |
| 合计 | **7,200** | **540** | 所有失败、崩溃、重试与封口成本均计入 |

正常完整运行 P+F 为 112 次高层模型调用、496 次网络前向。每阶段最多两次尝试，仅允许一次有原因的工程修正重跑；成功或质量负结果不触发择优重跑。P/F 各两次的绝对模型调用上限为 224/992，实际未执行不补足。合成单测/全项目回归单列工程验证耗时；使用真实 checkpoint/数据的研究计算不得藏在测试、烟测或“只读复验”名下。

FUSION 原 cohort 保留：R 66.705/600、S 1513.866/1800、V 7.198/3600、F 375.520/1200，累计 1963.290/7200s。S 剩余扣 60s 后仅 226.134s，低于 r10 实测 252.156s，因此不生成 r11。本预算不扩大该 S、不给旧 v2 新一次完整验证，也不承接其生产接入；新报告必须列该父历史及本轮新增成本。若实质工作变成再验证/调参同一个融合候选，立即停止并按旧 cohort 处理，不能借新问题名称洗掉费用。

### 9.2 调度与封口

cohort 固定按研究问题族、SOURCE/ATTR 父注册与两 checkpoint 身份识别，不以运行名、父目录、方案哈希或新实现 SHA 重置。历史扫描固定 `derived/experiments/omniretarget_first/` 的直接子目录，所有同问题/同父身份尝试（含修复版）累计；模糊或不可读的匹配历史为 unknown。每 attempt 必须在实质工作前写 running 索引、台账与 cohort 标记。

盘上 attempt 集合、索引、台账须一一对应；状态和完整字段对账，成本非负有限且非 bool，索引 `measured_wall_s` = 台账实测值、台账 `wall_s=round(measured,1)`。已登记 scope 与实际行键集对账，不可删失败行省调用/成本；running、publishing、坏历史或无登记目录均为 unknown=停止。

使用 `perf_counter`/单调时钟；GPU 在计时边界同步。调度前可工作预算为 `stage_limit - verified_history - reserve`，≤0 不分配新工作。子进程 timeout≤剩余可工作预算，块返回、哈希/保存、报告/索引/台账最终登记之后均查截止；当前调用不能仅靠历史门约束。

封口复用已验证的 W1-D/E `finish_attempt` 语义：pending → publishing → 检查 → seal，预留计费与 seal 后覆盖验证覆盖 done/incomplete/crashed 各终态。尾部成本无法证明则 `publishing/complete=false`，撤回行及报告的完成声明，历史 unknown 且返回失败/非零；最终报告只有在索引/台账封口后才可消费。SIGKILL 留下的事前 running/publishing 必须被消费侧拒绝；不能把“异常未被捕获”当零成本。未知历史不填零，也不启动后续 stage。不要为本诊断重新实现一个较弱的完成发布协议。

超预算、输入漂移、来源不明、实际调用超界、磁盘不可写或固定键不齐时停止相应实质工作，F 用预留登记 incomplete。启动前检查 GPU、内存与磁盘，产物上限 5GiB；不预存重复训练滑窗或新增 checkpoint，不下载依赖。资源不足亦记真实原因，不删旧证据腾出预算。

## 10. 输出、命令与实施验收

建议布局如下，所有报告引用完整 path+SHA-256、schema/version、注册身份与实际调用，而不是只存一个 passed：

```text
<run>/
  freeze/execution_registration.json
  freeze/implementation_snapshot/
  cohort.json
  register/{index.json,attempt_NNN/}
  saved_windows/{index.json,attempt_NNN/{rows.json,curves/,summary.json}}
  probes/{index.json,attempt_NNN/{inputs/,outputs/,rows.json,summary.json}}
  decisions/{index.json,attempt_NNN/{prediction_consistency_final.json,tables/,plots/}}
  costs.json
```

报告至少含：130 窗误差表、常规/尾窗与位置分箱表、全部跨窗配对、同帧穿透分类、52 常规条件和 13 尾窗的逐模型绑定、520 探针表、固定三段重放结果、Teacher/模型逐例量级、已知帧回归、预算/缺口/下一候选。图采用可导出的标准绘图库，保留逐帧原值与单位；不得用图隐藏拒收、缺行或负门。

拟定 CLI（实现后才可执行；示意运行名不是已创建的实验）：

```bash
PREDICTION_RUN=derived/experiments/omniretarget_first/20261021_IMPL_w1_prediction_consistency
uv run --locked python scripts/diagnose_w1_prediction_consistency.py register --run "$PREDICTION_RUN"
uv run --locked python scripts/diagnose_w1_prediction_consistency.py saved-windows --run "$PREDICTION_RUN"
uv run --locked python scripts/diagnose_w1_prediction_consistency.py probes --run "$PREDICTION_RUN"
uv run --locked python scripts/diagnose_w1_prediction_consistency.py finalize --run "$PREDICTION_RUN"
```

I 的必要反例与正式验收要求：

| 类别 | 必须能区分的反例/正向不变量 |
| --- | --- |
| 集合/分母 | 缺一窗、重复窗、尾窗伪装训练窗、删失败行、缩为单 demo；52/65/130/520 的键和组别不缩水 |
| 来源/参考 | 自洽替代 decoded 窗、Teacher 换模型、条件错 demo、checkpoint/统计/尺度漂移；历史实现快照与当前实现独立核验 |
| 坐标/表示 | 非平凡平移+yaw 下 world/clip 对拍、四元数变号等价、零范数/非数值数组拒绝、原解码保持、教师近 π 精度负例不掩盖 |
| 重叠/时间 | 三窗中间贡献、零权重入场、全部三对、混合穿透、无共同活跃窗、原窗内坏跳变；rad/s 与单步度相差 frame_dt |
| 探针/RNG | 保存并重建真实 epsilon/x_t、五 t 同噪声、t=0 语义、完整批 seed7 对拍、全局 RNG 不变、禁止教师进生成条件 |
| 数值/缺测量 | NaN/Inf/缺键/坏 shape/嵌套 Tensor 真值、分数计数、零参考 None、无手参考仍测脚；报告落盘不崩 |
| F 重推 | 篡改曲线/速度/metrics/summary 自报通过、清空行、错误 next_action；从工件重推并让矛盾阻止 complete |
| 历史/预算 | 换目录不归零、成本数值三方矛盾、边界截止、最后写盘超时、真实子进程 SIGKILL 留 running/publishing、unknown 返回失败 |

先跑上述有意义的接口/反例及相关 attribution/rotation/generation/closure 回归，再按项目约定跑全项目套件；登记实际计数与耗时，不能把旧 891 加新增数当作新的全量结果。正式运行在已固定实现下按 I/R/W/P/F 顺序一次执行；所有缺陷修正保留 superseded 证据和累计成本。

本次文档交付仅完成本文与导航/进度登记。实施收口时应另交付代码与反例、只读前后锚、全部固定矩阵、独立 F、成本和验证记录；若无法区分责任，明确写 `attribution_insufficient`。后续训练/数据/教师/融合/修复候选的执行不由本诊断自动启动。
