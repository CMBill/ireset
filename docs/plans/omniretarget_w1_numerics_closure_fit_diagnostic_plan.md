# W1 数值缺口有限收口与受控拟合诊断方案

日期：2026-11-06。版本：v0.4（本地/最终状态分层与原始预测解码关联修订）。**状态：仅方案，待复阅；未实施、未预注册、未授权、未运行。** 本文不授予任何新真实数值计算、训练或重放权限，不恢复T6的一次性授权。v0.1由git提交 `9fb90e3` 保留；§11/§12保留v0.2/v0.3修订说明，§13记录本次修订。本方案尚未作为实验契约冻结，本次只修订文档，不重冻结旧证据。

## 1. 目标与当前基线

下一阶段从“重复全链审计”转向两个可区分的问题：

1. 剩余数值缺口中，哪些是当前精度下不可辨识或相消敏感，哪些仍需要实现修复？
2. 在已修正实现和固定弱动作目标下，确定性模型能否在固定预算内达到一窗及六窗的事前平均拟合目标？

先完成有限的保存证据分析，再决定真实小集诊断是否具备单独启动条件。不以“539项全部变成passed”为预设目标，也不以审计/训练运行完成代替质量改善。

### 1.1 已有事实（不是本方案的新运行结果）

- [T6执行报告](../verification/omniretarget_w1_gradient_numerics_retest_report.md)：0 direct failures、539 mandatory gaps（350 precision/189 sensitivity/0 bound）；`audit_incomplete`，`complete_evidence=true`、`covered_contract_passed=false`。
- 已有 `i_rows.json` 的只读汇总：3846行中528条AdamW参数/更新行；392条行带gap，其中203条仅precision、147条两类兼有、42条仅sensitivity。因此539不是539个独立程序错误；528条同输入公式检查为true。
- 实现身份32文件 `c4d4a88926b985ee365732da0d940e28985f86c64d4f828e0bb6c5dca4f01b3b`；本方案不是该旧闭包的成员，不改变旧绑定。
- 旧更新cohort已消费34次诊断更新和1347.195507/3240秒；T6一次性授权已消费。剩余墙钟不是新调用/尝试/更新权限。
- [W1 D/E](../verification/omniretarget_w1_de_verification.md)：两模型各4000步；Teacher raw 2/31、两模型各0/31。T6没有新生产checkpoint或新质量评估，不能宣称这些负结果已改善。
- [旧修复终局](../verification/train24fix3_convergence_recovery_verification.md)：§0.10的R1/R2/R3均未通过六段门；不恢复旧train24/生产/重训路线。

### 1.2 明确不做

不启动第二次T6或完整R0/P/I/F；不放宽旧数值阈值；不修改旧报告、授权、成本、源数据、冻结计划或历史工件；不扩数据、改教师、开启接触损失、换主干、改优化器、切换融合器；不运行test31、正式val31选模、DDIM或RL；不宣称生产接入、完整目标通过或端到端加速。

## 2. 阶段顺序与停止点

| 阶段 | 工作 | 退出条件 | 后续是否自动启动 |
| --- | --- | --- | --- |
| N0 | 保存数组的坐标级数值缺口分析 | 原始行/参数坐标/舍入与区间证据可追溯；无新增真实模型调用 | 否，先审阅数值收口结论 |
| N1 | 必要的最小合成验证与有限收口建议 | 能区分可证明公式正确、严格跨路径未过及未覆盖内容；反例仍拒绝 | 否，不自动修改旧契约 |
| I | 小集诊断实现与纯合成反例 | 复用实际Trainer更新单元，身份/计费/数据分母测试通过 | 否，真实阶段需新冻结与授权 |
| R | 新诊断元数据准入与绑定 | 固定训练窗、配置、代码、资源、预算与调用矩阵；授权匹配，不物化数组 | 授权明确包含时才进入D1 |
| D1 | 确定性单窗拟合 | 完成1000步及全部probe，`local_complete_evidence=true`，重推本地平均门及交接收据 | 仅 `handoff_ready=true`、合法封口且授权包含D6时继续；不等待F |
| D6 | 确定性六窗拟合 | 完成1000步及全部probe，发布本地覆盖/平均门结果，最终核验状态仍pending_F | 不自动扩展到扩散或全量训练；不排除窗内尖峰 |
| F | 只读核对及有限checkpoint重放 | 全部保存probe指标/门重推、initial/last逐位重放及成本对账通过后发布最终状态 | 收口；核验失败否决最终裁决，保留本地事实与成本，不追加重试 |

**首个实施交付限定为N0/N1工具与分析记录。** 真实拟合是后续有条件的候选，不因本文存在或数值公式门通过而直接获准。

## 3. 输入、来源与保护范围

### 3.1 N0只读输入

固定更新根：`derived/experiments/omniretarget_first/20261102_2704709_w1_real_training_update_audit/`。只消费以下已保存内容：

- `registration/attempt_003/registration.json`及绑定容差/运行时事实；
- `production/attempt_003/`的报告与两模型NPZ；
- `independent/attempt_002/i_rows.json`、`i_report.json`、两模型NPZ、focus sidecar；
- `finalize/attempt_002/training_update_final.json`、实际return observation和逐键对照；
- `derived/audit/w1_real_training_update_audit/gradient_numerics_retest/real_analysis/attempt_001/`。

先校验保存路径、SHA、数组键/shape/dtype/有限性与行键多重集；分模型、分数组读取，不将全部大NPZ重复解压保存。缺失或矛盾记 `evidence_incomplete`，不得从报告摘要补造原始量。

禁止加载真实checkpoint payload、实例化训练网络/优化器、调用生产forward/backward/FK、执行AdamW或重放更新。对保存数组的CPU算术重算要单列调用、墙钟和范围，不冒充“计算0”。

### 3.2 后续拟合来源（R仅元数据，获准D1/D6内物化）

- 只读SOURCE：`derived/experiments/omniretarget_first/20261012_21e5784_w1_a3bc_reviewfix8/w1_reviewfix8/`。
- 复用该SOURCE的注册train24、训练manifest、准入缓存、源/目标箱体分离、root_stats及固定资产；不得默认当前最新cache可以替换登记版本。
- R根据SOURCE的注册、manifest和数组头元数据确定 `pilot6 ∩ train24` 的六个窗键，每demo为start=0的60帧；须恰为六个不同已准入训练演示，否则R失败，不临时补样本。R不通过读取数组值构建训练batch。
- 单窗取这六个demo的规范字典序首项；选择只依赖注册键，不看拟合、碰撞或接触结果。
- R冻结精确demo/start/end/T、源/标签/缓存SHA与选择规则；允许原始文件字节哈希和数组头检查，不解码NPZ数值、不 `torch.load` checkpoint、不构造tokenizer/训练模块/FK。实际源数组、root_stats和训练窗的读取、解码、重建及逐位绑定，必须在获准D1/D6已事前登记、启动计时且通过单元守卫后执行；物化失败留在该阶段记账，不退回R或阶段间免费处理。
- D6即使复用D1的不可变内存缓存，也在D6计费区间内核验来源与绑定并记录命中，不能让缓存准备或核验发生在阶段外。所有推理条件继续禁止G1真值根、关节或教师脚落点；目标只用于监督与度量。
- 不使用val/test作为训练、挑窗、调阈值或选seed依据；已知验证负结果只作历史背景，不伪装为新盲测。

## 4. N0：数值缺口有限分析

输出以 `(model, update, parameter, coordinate, gap_kind)` 为键，不能只按张量最大值解释整行。

1. 重推350/189的事件集合、392个带gap行及147个重叠行；对照旧629项的resolved/persisted/new，不改写旧分类。
2. 对precision坐标列出 `theta_before/after`、未舍入参考增量、保存实测增量、最终写入界、ULP、可辨识掩码及门分母；区分“未写入变化”与“变化了但不足以通过相对增量门”。
3. 对sensitivity坐标列出P/I梯度、共同前态、m/v及更新差异、零值/符号、原始相对门、保存传播区间与相消位置；不得只因落在区间内就改判原始跨路径门通过。
4. 优先覆盖 `diffusion u2/m2 root_out.weight` 的对应更新行，再按规范键序覆盖全部缺口。focus只有指定参数，不能据此虚构未保存的micro梯度；全量分析限实际保存层级，缺micro证据时明确缺口。
5. 报告绝对参数差、可辨坐标上的更新差，以及无法由保存量推出的函数/动作影响。没有新forward时，不声称“对动作无影响”或“训练质量已证明不受影响”。

拟交付完整紧凑坐标标记 `coordinate_flags.npy`、行/数组索引 `coordinate_index.json`、`gap_rows.csv`、极值明细 `gap_coordinates_extrema.csv`、`summary.json`和人读分析。**不默认输出全坐标多列CSV**：逐坐标全量检查在内存/流式过程中完成，完整标记加原始量来源索引用于复算，CSV只承担有界极值展示，不以抽样代替覆盖。

拟定 `n0_coordinate_flags_v1`：按固定 `(model, update, parameter)` 行序和C-order坐标展开，用一维 `uint16` 保存每坐标的P/I不可辨识、原始m/v/theta/delta门失败、零/符号不一致、区间外、公式/来源异常等标记；位定义、行offset/shape、原始NPZ键、dtype及SHA在N0执行前冻结，不等待后继I阶段。缺证据必须有显式标记，不得以默认零伪装通过。至少逐一覆盖528条AdamW行，并由标记重推350/189事件及392行/147重叠，不能只检查极值CSV。

容量边界：在N0入口事前登记并启动本阶段计费时钟后，先执行保存文件头预验，确认标记总坐标数 `N≤16,000,000`（包含两模型、两步和所有528行；不是独立参数量）；该预验计入N0的900s上限，不安排阶段外免费检查，也不延期至后继R阶段。故标记数据≤32,000,000 bytes，NPY头预留1024 bytes。逐行索引≤16MiB、行CSV≤8MiB；每行每gap类别最多16个极值坐标、每条CSV记录≤2048 bytes，故最多 `528×2×16=16,896` 条、≤34,603,008 bytes，另设64MiB文件上限（含CSV表头与转义后的实际字节）。极值按门比/绝对误差的预定排序及坐标顺序决胜，不按结果挑选有利记录。

N0报告/日志/manifest合计≤32MiB，N1合成夹具和结果合计≤64MiB，全部临时文件峰值合计≤256MiB；连同上述各项，按 `32,000,000+1024+(16+8+64+32+64+256)×1,048,576` 计算，保守磁盘占用上界为493,374,464 bytes（小于共用1GiB上限）。这里对极值CSV使用64MiB文件上限，而不是依赖实际压缩或较小的明细估计。临时文件与最终文件同时存在时均计入。禁止把大NPZ全量解压到输出目录或复制原始浮点矩阵；原始量按保存来源+flat index回查。头部/规模不符、预计超限或写盘计数接近上界时先停止新明细，保留已完成覆盖范围并以 `evidence_incomplete/resource_limit` 收口；不得截断明细后声称全量完成。如需另存全坐标原始量表，须事前修订格式和容量，不在本预算内临时扩写。

N0的结论限定为：`implementation_issue_candidate`、`precision_limited`、`cross_path_sensitive`或`evidence_incomplete`；这些是分析标签，不是对旧mandatory gap的通过豁免。

## 5. N1：最小合成验证与有限收口

- 仅复用保存标量/小数组构造解析或合成反例；保留同输入有效float32公式逐位门、非零擦除/符号/状态局部清零反例、单侧可辨增量反例。
- 限定CPU纯算术，最多六类、每类八个用例（合计≤48）；每例向量≤16坐标、最多两步和两条比较路径，每条路径每步至多一次有效float32公式和一次float64数学参考求值，合计≤384次逐向量公式求值。实际用例、求值及合成算术状态推进非零次数分别登记，不冒称“计算0”或借用真实更新额度；不构造真实网络/原生优化器，不校准或宣称CPU等价于生产CUDA/foreach。
- 不把参数/梯度全部改成float64当作生产修复，不调学习率、AdamW epsilon或阈值来追求旧gap清零。
- 如发现区间外差异、公式失配或来源矛盾，先提出范围限定的最小修复及合成反例，停止真实拟合准入；实现变更后重新绑定受影响材料。
- 如剩余问题可由保存的舍入/敏感区间说明，形成**有限数值收口说明**：列清已验证的同输入公式、未通过的严格跨路径门和未覆盖的长期训练/动作质量。
- 若确需另设有限精度诊断契约，必须单独版本、单独标签、事前审阅；不能将新标签等同于旧 `covered_contract_passed`，旧结果仍 `audit_incomplete`。
- 允许随后在**明确承认partial及旧gap未消除**的条件下申请独立拟合诊断；并非声称全目标通过，也不把旧cohort改名重获复验额度。

停止扩展边界：N0/N1结束后只交付分类、最小反例和go/no-go建议；不继续堆叠新的整链校准、迁移generation或无限次重冻结。

## 6. I：复用实际训练路径，不再复制训练循环

### 6.1 可复用的现有接口

- `src/ireset/train/trainer.py`：`Trainer.predict/begin_update/accumulate_micro/clip_gradients/step_optimizer`。
- `src/ireset/data/w1_training.py`：注册、缓存准入和训练统计规则。
- `src/ireset/data/windows.py`：`TeacherWindowDataset`及确定性切窗。
- `scripts/w1_readiness.py::_rebuild_small_probe_batch`：参考注册六首窗重建/逐位绑定的做法。
- `scripts/w1_s0.py`：只参考probe和整批小集语义；**不直接沿用其独立手写更新循环**，避免绕开已审计的Trainer更新单元。

拟新增独立诊断模块/入口/测试，具体文件名和CLI在I阶段确定并写入新注册；本方案不提供当前不存在的可执行命令。不改旧S0门、旧CLI默认行为或32文件旧闭包以追认历史。

### 6.2 真实准入前必须通过的合成反例

注册名单/窗键/缓存内容篡改；少窗、重窗、val/test混入；target泄漏进conditioner；`drop_last`导致空loader；空/缺项/非有限指标；任一模型参数未进联合clip；跳过micro或更新多计/少计；probe改变训练RNG；checkpoint/root_stats错绑定；工作截止后继续构造/forward/backward/step；末次probe/checkpoint/封口写盘越界仍声明complete；R偷读数组值或阶段间物化未计费；SIGKILL遗留running；失败后新attempt重置预算；F身份失败仍执行模型；只剩总loss下降却单窗未过门；RMSE/MPJPE混用、米/厘米和弧度/度混用、宏平均掩盖单窗失败、平均旋转过门却单帧60°尖峰被误称已排除；坐标标记缺行/offset错/截断后声称全覆盖及临时文件漏计容量。

v0.3另要求以下合成反例：保存与fresh预测均过平均门但数组不同；只篡改中间probe曲线/p95/max或最坏坐标；删一窗/重复一窗而总行数不变；伪造D1通过布尔值但原始量未过门；initial/last角色或step对调；TF32/autocast开启、精度或运行时身份漂移；构造后发现参数dtype/device错误仍继续forward；D6逐窗调用六次却记作一次probe；重放不一致后放宽容差/补跑或仍发布最终mean_fit_passed。上述反例属后续I交付要求，本次没有实现或运行。

v0.4另要求：D1在F未运行时能按本地收据合法交接，但不能发布最终验收通过；F失败不能回写或洗掉本地成本/事实；篡改step500的metric_decode并同步指标/哈希仍须被原始x0重解码拒绝；只替换中间FK点并同步其指标/哈希时，必须保留“中间FK状态关联未独立评估”的限制，不能因度量自洽就把该限制升级为passed。后一个用例用于验证覆盖边界，不冒称无额外FK求值也能检出所有自洽替换。

### 6.3 继承已审计数值运行时的硬门

- R冻结已有 `src/ireset/eval/w1_real_fk_audit.py` 的 `RUNTIME_REQUIREMENTS` 和 `configure_production_runtime()` 绑定字节；不是仅记录任意一次环境快照。固定要求：CUDA/float32生产路径、`cuda_matmul_allow_tf32=false`、`cudnn_allow_tf32=false`、`float32_matmul_precision="highest"`、`autocast_enabled=false`、`deterministic_algorithms=false`。不为获得逐位重放临时改为deterministic=true，也不回退CPU或启用混合精度。
- R仅读元数据对照T6已登记的Python/Torch/NumPy/SciPy、CUDA/设备、CPU线程、MuJoCo、pytorch_kinematics及已绑定依赖源码身份；新诊断实现身份另行冻结。版本、依赖源码或运行时要求不符时停止准入，不能直接沿用旧数值结论；如确需迁移，先另行修订/复阅，不在本方案内追加真实校准。
- D1、D6、F各自在已登记计费区间内，物化真实数组、构造训练/FK模块或加载checkpoint之前应用上述配置并验证实际值；若F使用子进程，子进程同样应用、核验并与父注册对账。每次micro/probe/重放单元开始前再次检查精度开关；不符即停止新数值调用。复用Trainer接口本身不视为配置已生效。
- 每个模型/conditioner/FK构造或strict加载返回后、下一构造或forward之前，核验实际浮点参数/缓冲和生产浮点输入的dtype/device为已冻结的CUDA/float32；整数索引/布尔掩码及来源元数据保留其注册类型。发现错误就停止，不静默cast修好后继续；如构造已发生，保留实际调用和成本，不谎记为零。CPU上对保存量作§7.5规范解码及float64度量归约是单列的度量例外，不将模型、训练目标或生产FK改为CPU/float64，也不把该例外用于训练forward。
- 保存配置前/后、每阶段终点和F执行端的实际运行时事实及来源SHA；发生漂移即使平均门仍过也不能发布合法完成声明。缺事实、缺字段或核验异常均fail-closed，不以默认值补成符合。

## 7. D1 / D6：确定性小集拟合提案

以下是**待审阅的诊断设计和上限提案，不是已批准执行参数**；R前须冻结最终值，执行后不得按结果放宽。

| 项目 | D1 | D6 |
| --- | --- | --- |
| 网络 | 当前修正实现，确定性模型+conditioner | 同架构、同初始seed，独立从头构造 |
| 唯一训练窗 | 1 | 6，每个demo一个首窗 |
| micro-batch / grad_accum | 1 / 4 | 6 / 4 |
| 每步实际窗口消费 | 4次，同一窗重复 | 24次，六窗重复四次 |
| 计划完整optimizer更新数 / 上限 | 1000 / 1000 | 1000 / 1000 |
| 固定probe | step0及完成更新10,20,…,1000后，共101次 | 同左，逐窗保存 |
| probe批布局 / 每次预测调用 | 固定batch=1、规范窗序，恰一次conditioner及一次backbone forward | 固定batch=6、六个注册窗规范序，恰一次conditioner及一次backbone forward |
| 初始化与优化器 | 从头初始化、step0 AdamW | 同左，不从D1继续 |

两阶段有效batch分别为4/24，与正式W1的4×4=16不同；这是控制数据覆盖的诊断，不是对正式训练吞吐或泛化的等价复现。D1窗是D6子集，不能当七个独立演示；不得将重复消费计成新独立数据。

其余有效配置沿用 `configs/train/omniretarget_motion.yaml`：seed/val_seed=20261008，lr=1e-4、weight_decay=1e-5、grad_clip=1，d240/4层/6头/time_window31，损失权重不变且 `w_contact=0`。拟合使用真实Trainer目标，不加额外接触、碰撞或正则项。

训练目标不保证可行交互。每次probe分别记录每窗的normalized分组误差和下述四项物理指标，同时保留逐项loss与联合梯度范数；不得以六位舍入的历史loss记录做逐位公式验证。probe使用model/conditioner的eval模式和no_grad，保存并恢复原模式与Python/NumPy/Torch CPU/CUDA全局RNG，独立generator不推进训练/loader流；F使用相同求值模式。每次probe整批预测后再逐窗切片度量，不拆成逐窗预测、不为不同指标重复forward；若batch=6无法执行，按实际范围incomplete停止，不临时拆批或改变调用上限。已测历史S0只支持训练时间及normalized误差下降，不能证明下面的物理目标在1000步可达。

### 7.1 物理指标定义与平均拟合门（待复阅提案）

设每窗 `T=60`、关节数 `J=29`、site数 `S=15`，上标hat为模型预测，另一侧为同窗冻结教师，`norm2(v)`表示向量v的欧氏范数。两侧均在**同一个冻结clip坐标系**解码：根平移使用同一root_stats去标准化，根旋转使用同一合法6D解码，关节使用同一物理限位映射及固定关节顺序；FK的site集合/顺序显式保存并与 `KPT_TO_SMPLH` 的15项绑定。禁止额外平移/yaw/Procrustes对齐、动态删帧/site或改用人体关键点作拟合真值。

| 机读指标 | 每窗定义（不跨窗合并） | 判门值 |
| --- | --- | --- |
| `root_position_rmse_m` | `sqrt(sum_t norm2(p_hat[t]-p[t])^2 / T)`；先对xyz求和，再对帧均值和开方 | ≤0.01m |
| `root_rotation_mean_rad` | `sum_t angle(R_hat[t] R[t]^T) / T`，`angle`为SO(3)最短测地角，范围[0,π] | ≤π/180 rad（展示为1°） |
| `joint_rmse_rad` | `sqrt(sum_(t,j) (q_hat[t,j]-q[t,j])^2 / (T×J))`；按物理关节角直接相减，不作跨限位的角度取模 | ≤0.05rad |
| `teacher_fk_mpjpe_m` | `sum_(t,s) norm2(x_hat[t,s]-x[t,s]) / (T×S)`；先取每site欧氏距离，再求均值 | ≤0.01m |

物理指标以float64做度量归约，根位置/旋转/关节的权威度量源为§7.5从保存原始x0得到的规范CPU解码；FK MPJPE的输入仍为已执行生产FP32链保存的两侧FK点。根旋转角使用现有NumPy/SciPy成熟实现；不是重新以float64执行模型、训练目标或FK。先验证形状、完整键集、有限性、旋转合法性及来源，再计算归约；坏输入不得先归一化、裁剪或删除来洗成合法测量。不得直接把标准化 `parts.pos` 当米制RMSE，也不得用 `sqrt(parts.fk_m2)` 替代MPJPE，或把 `geodesic_rad2` 当平均测地角。比较使用未舍入SI值，不按报告展示值或额外容差放行。

每窗同时满足四门时记录 `window_mean_fit_passed=true`；D阶段以最终step1000的全部注册窗结果形成 `local_mean_fit_passed`，D6要求六窗各自过门，并另报六窗等权宏平均，不以宏平均替代逐窗门。未前缀的 `mean_fit_passed`仅用于F最终验收命名空间，不能在D阶段冒充已获F确认。这里不设置新的峰值通过阈值，统一记录 `spike_gate_status="not_evaluated"`；因此**平均拟合通过不表示窗内尖峰已排除**，不得以此跳过后继时序诊断。

### 7.2 尖峰可见性、终点与结论边界

每次probe保存逐帧根位置距离、根旋转角、逐帧逐关节绝对误差和逐帧逐site FK距离；每窗分别报告这些集合的p95/max和最坏坐标。p95固定为对相应集合使用linear分位数法；max并列按最小帧号、再按关节/site固定顺序取首项。逐帧/逐关节/site表保持完整，不能只保存均值或p95（p95也可能漏掉单帧尖峰）。例如60帧中单帧60°、其他帧0°仍可达到平均1°，此例必须被完整曲线和max显示，且不能被描述为时序误差已通过。

平均门数值不变，但名称收窄为**固定预算的平均拟合研究目标**，不是“充分拟合/无局部异常”证明，不是既有验收器阈值，也不是已测得可达值。须在新结果前复阅并锁定。初值/最终值和完整曲线用于展示学习进展，不替代最终绝对门，不按历史趋势反推新门通过。

完整D阶段必须恰好完成1000步并保存step0、10,…,1000的101次probe，本地判门只消费step1000的 `last`：不得因较早probe达标提前作为完成退出，也不得挑最好probe、追加步数/seed或按结果改挑窗规则。1000步完成且本地来源/数值/保存/计费证据均完整，但step1000任一平均门失败时记录 `local_mean_fit_passed=false`、`local_conclusion=fit_insufficient`；它只意味着“本配置/数据/预算下未达到事前平均目标”，**不证明实现有错、模型不能拟合、优化已充分收敛或教师是主要瓶颈**。该本地事实仍须经F核验才能形成最终裁决。

不足1000步、缺probe/坏数值/来源或封口未闭合时，记 `local_complete_evidence=false`、`local_mean_fit_passed=null`、`local_conclusion=incomplete`并保存具体原因/实测范围。D1只有本地覆盖完整、从落盘原始量重推平均门通过、§7.4第1–3项的本地适用检查通过、合法封口及授权明确包含D6时，才形成 `handoff_ready=true`；D6入口复核收据和封口后才启动。此门不依赖F第4–5项或F最终字段，不能只读D1报告里的通过布尔值。D1本地未过不追加D6作为“再试一次”，跳过原因显式登记。D6即使本地平均过门，仍不支持泛化、严格碰撞、持续抓握、无尖峰或正式模型优于旧W1。

### 7.3 保存与重放

每个D阶段保存initial/last诊断checkpoint、精确配置/身份、固定probe输入及全部101次probe的未舍入原始x0预测、§7.5的metric_decode、实际生产FP32解码量及预测15-site FK点、逐窗指标、更新/micro/样本消费计数、RNG和成本。规范度量解码与生产FP32解码使用不同数组键和domain标记，不互相替代；后者用于F端点逐位绑定及明确受限的生产链观测。冻结教师输入/两类解码量/FK点可每阶段只存一次，按SHA和固定窗/site顺序引用。逐帧误差及p95/max可由这些保存量完整重算，不能只存loss或误差汇总。每份工件登记stage/step、demo/start/end、batch位置、dtype/shape及SHA，不把probe编号当optimizer步号。

initial对应step0 probe的同一实际参数状态，在第一次optimizer更新前持久化；last对应完成第1000次更新并执行step1000 probe后的同一状态，保存step=1000及实际状态摘要，不重新初始化、不借用更早probe的权重。正常本地完整阶段的两份checkpoint及必需probe数组须在D阶段工作截止前写入，并完成§7.4第1–3项的本地适用检查（含§7.5解码关联）；**不要求在D阶段完成F第4–5项**。若工作时间不足，只能在收口预留内保存部分诊断状态并记本地incomplete，不能补成完整门。checkpoint写入和必要校验均计费，不能沿用T6的“checkpoint writes=0”声明。

不加载旧step4000权重续训；其没有历史AdamW状态，不能冒称历史续训。F仅在来源、身份和预算门满足且D阶段 `local_complete_evidence=true`、合法封口时，构造一套fresh模型+conditioner，分别strict加载initial/last各一次，并各执行一次固定probe；本地平均门为false的完整负结果也可核验，不以平均通过作为F准入条件。完整D1+D6时模型/conditioner构造各≤2、checkpoint加载≤4、conditioner/backbone probe forward各≤4，optimizer构造/step/backward均为0。物理指标从该次probe输出复用计算，不得为不同指标重复预测；FK/目标重建等数值调用仍另计并在I/R最终矩阵冻结。条件不满足时F只整理已有证据，不执行forward；不补足被跳过阶段或从checkpoint续跑。不得为了保存全参数梯度而复制每个micro的模型状态。

### 7.4 保存量消费核验、F逐位重放与否决

D1/D6只执行下列第1–3项在本阶段适用的本地检查；第3项的D1收据生成和D6入口消费分别计入各自阶段。F在其一次执行窗口内复核第1–3项并执行第4–5项，fresh预测只属于F，不能提前到D1或D6。§7.6定义本地与最终字段的命名空间，避免以F验收作为D1交接的循环前置。

1. **固定覆盖及重推对象。** I冻结规范度量schema、C-order/float64归约与完整键集；R绑定其实现。完整D1期望 `(step,demo,start)` 为101次注册probe×1窗，共101条；完整D6为101次×6窗，共606条。消费端按注册多重集核验，无缺/重/额外键，并校验原始输入/目标的SOURCE绑定、每数组shape/dtype/有限性/旋转合法性、实际SHA和initial/last的角色/step/状态绑定；不能把两侧同缺、改step或自报通过当作完整。
2. **全部保存probe的CPU重算。** 每个D阶段及F都从磁盘重新读取绑定的原始x0，按§7.5重新计算并精确对账metric_decode，再用重解码量计算根位置/旋转/关节误差；FK距离只从保存的两侧生产FK点重算。随后按§7.1/§7.2重算全部逐帧误差、四项指标、p95/max/最坏坐标、逐窗及宏平均结果和step1000四门，不只核对端点汇总布尔值。采用已冻结的规范CPU解码和float64度量器，与未舍入保存结果作精确数值和类型对账，整数/布尔必须同型，计数/键集/最坏坐标精确相等。报告不得以舍入展示值参与验收；另设展示字段时，机读原值必须保留。该重算不重新执行模型、训练更新或FK，不需要新增forward；规范解码/CPU算术/读取/核验次数和成本单列记录。中间FK点重算距离通过仅证明其度量自洽，不证明这些点等于相应状态的独立FK结果。
3. **D1→D6事前硬门。** D1在自己的工作时间窗内完成上述本地落盘消费核验，产出绑定必需工件SHA、重推四门及覆盖事实的 `handoff_verification`候选；候选本身不可消费，也不以尚未完成的封口/实际返回自证 `local_complete_evidence`或 `handoff_ready`。收口只登记已完成事实并绑定合法封口和外层实际返回证明，不补数值核验；三者共同支持本地完整且平均门通过、授权包含D6时，交接才可成立。D6入口在D6已计费区间内先复查D1注册、候选收据/工件SHA与封口/实际返回覆盖，从这些事实重推 `handoff_ready=true`后才物化或构造D6；不读取或等待尚不存在的F最终字段。核验失败、工件漂移、缺收据或超时均不启动D6数值单元。F后续复核交接事实，不追溯抹除已执行D6的调用/成本。
4. **fresh输出逐位绑定。** F只对已本地完整且合法封口的D阶段执行§7.3的initial/last各一次probe，batch分别固定为原D1的1窗或原D6的6窗，输入键/顺序/dtype/shape、权重角色/step、运行时和求值模式均与原probe一致。加载前校验checkpoint文件SHA、参数键/shape/dtype/有限性及内嵌配置/root_stats/身份；不得依靠strict load的隐式dtype转换。fresh原始x0预测、生产FP32解码量及本次实际FK点必须与保存的对应step0/1000数组逐位一致（dtype/shape及数值字节一致）；教师输入/目标及其绑定也须一致，不能只比范数、loss或都低于拟合门。FP32生产结果核验后再按§7.5从fresh x0计算metric_decode，并按第2项精确对账派生指标，不借平均门作重放容差。此项只确认端点，不能将中间FK覆盖标签升级为passed。
5. **不一致的裁决。** 逐位重放是事前验收要求，不是已证明的可重现性结果；即使deterministic_algorithms=false下首次出现合法数值漂移，也必须记录 `replay_mismatch`并incomplete收口，不改精度开关、不放宽容差、不新增probe或自动重试。保存量重算不符、解码关联失败、覆盖/身份缺失或预算/封口不成立时同样否决。只在F最终命名空间记录 `complete_evidence=false`、`replay_passed=false`（未能执行则为null并记原因）、`mean_fit_passed=null`，不发布有效的最终平均拟合通过/不足裁决，也不把证据问题改写成 `fit_insufficient`。F以追加的否决记录指明受影响阶段；原D阶段的local字段、收据、文件SHA、实际调用/成本和负结果原样保留，不回写它们冒称从未完成本地计算，不回退或重获额度。

上述核验只消费已执行probe和限额内的四次fresh预测，不逐步重训、不加载每次micro参数、不新增checkpoint角色。F对全部101次保存probe重解码/重算指标，不表示逐次重放中间模型权重或中间生产FK；fresh执行覆盖仍只到initial/last。范围限定的最终拟合结论需保存量/解码关联核验、fresh端点重放、身份、调用和预算封口同时通过；它不证明完整训练更新链或中间FK状态关联，`spike_gate_status="not_evaluated"`仍不改变。

### 7.5 原始x0解码关联与FK覆盖边界

- I/R在新结果前冻结 `metric_decode_v1`，明确域名 `cpu_float64_from_saved_x0`：对磁盘保存的原始FP32 x0预测/目标先核验合法性，再提升为CPU float64，按同一冻结clip系、同一有效mean/std/lo/hi执行根位置去标准化、既有合法6D解码及干净关节限位映射。常数采用生产实际生效FP32值的精确提升，而非另取精度不同的原始统计量；数组形状、关节顺序、坐标约定和解码实现字节均绑定。复用已有解码算术/旋转接口，不构造网络或FK；无新增真实forward、backward或优化器更新。
- metric_decode固定保存根位置、根旋转矩阵和29关节角；规范CPU解码的运算顺序/精度/有效域在I固定。D阶段生成指标时与消费端都使用这个域；消费端必须从原始x0重新计算，而不是读取保存物理量后只重算误差。重解码数组与保存metric_decode逐位精确对账；根/关节曲线和三个对应平均门只消费重解码结果。单独替换metric_decode、同步曲线和哈希仍须以 `decode_binding_mismatch`拒绝。
- 生产CUDA/FP32解码量和FK点另存为 `production_fp32`域，用于生产观察及F端点逐位重放；不把它们与规范CPU float64解码作逐位比较，也不声称FK点是由CPU解码状态重算得到。新方案此前未执行，本次统一其待冻结度量源，不改四个物理公式/阈值、实际训练目标、旧数值契约或旧测量，不回算旧结果。
- 对所有保存probe，报告 `decode_binding_status`和 `fk_metric_reduction_status`；对step10,…,990固定报告 `intermediate_fk_state_binding_status="not_evaluated"`，F的实际FK重放范围固定为 `fk_replay_scope="initial_last_only"`。中间FK点仅有生产来源登记/工件哈希及保存点距离的度量自洽核验，没有独立重算其状态→FK映射；自洽替换中间FK点不能被宣称已被本流程普遍排除，曲线中的该部分是覆盖受限的生产观测。
- 本方案不为补齐中间FK证据新增FK批次、模型调用或完整链审计；任何输出均不得写“全部101次物理/FK链已独立验证”。如后续需要该结论，必须另立范围/调用矩阵/授权，不能在现有F中临时扩算。CPU规范解码和一致性检查的实际次数/成本计入各D/F原工作时间窗和最终调用矩阵，工件仍受原2GiB上限约束。

### 7.6 本地交接与最终验收的状态分层

| 所属报告 | 字段 | 语义与消费方 |
| --- | --- | --- |
| D1/D6本地报告 | `local_complete_evidence` | 仅说明1000步/101probe、§7.4第1–3项本地适用检查及本阶段合法封口完整；不含F重放 |
| D1/D6本地报告 | `local_mean_fit_passed` | 本地完整时为step1000逐窗四门的布尔合取；本地不完整为null，不是最终验收 |
| D1交接收据 | `handoff_ready` | 仅当本地完整且平均门过、来源/预算封口及授权条件满足才为true；D6按§7.4第3项消费 |
| D1/D6本地报告 | `final_verification_status="pending_F"` | 明确F未作裁决；D阶段不写未前缀的complete_evidence/mean_fit_passed/replay_passed |
| F最终报告 | `complete_evidence`、`replay_passed`、`mean_fit_passed` | 仅F完成其适用的§7.4第1–5项后发布；失败语义按第5项，不能反向作为D1交接前置 |

F最终报告逐阶段列出验收结果和实际覆盖范围；本地完整但平均门未过且F核验通过时，最终证据/重放可为true、mean_fit_passed为false，结论为fit_insufficient。D6因D1门未过或未获授权而合法跳过时，D6最终结果为null/not_run并注明原因，不以空集合当成六窗通过；本地不完整或F失败时最终不发布有效拟合裁决。新报告一律按此分层，旧T6等历史文件的同名字段仍按其原契约解释，不修改旧字节。

本地数值核验与完成发布分时：D工作时间窗内完成重推并持久化核验候选；收口及外层实际返回证明仅登记已完成的事实和成本。本地报告中的完成/交接声明只有与合法封口及返回证明一起被核验后才可消费，不能要求候选在生成时已包含未来返回事实，也不能把候选通过布尔值当作最终 `handoff_ready`。外层返回证明只记录实际返回，不夹带额外数值工作；这一局部发布过程不等待F。

## 8. 预算、调用与授权门

### 8.1 待审阅资源提案

| 范围 | 墙钟上限 | 上限内部收口预留 | 新工作时间窗上限 | 真实训练更新 | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| N0保存证据分析 | 900s CPU | 60s | 840s | 0 | 校验、解压、CPU算术与输出均在本阶段计费 |
| N1合成验证 | 900s | 60s | 840s | 0 | 限CPU纯算术及§5的用例/求值上限，不混入真实网络 |
| R新准入 | 120s | 15s | 105s | 0 | 仅元数据、文件字节哈希/数组头；不物化数组值 |
| D1单窗 | 1800s | 60s | 1740s | ≤1000 | 含入口、物化、构造、全部训练/probe、checkpoint及封口 |
| D6六窗 | 3600s | 120s | 3480s | ≤1000 | 仅前置与授权同时满足才启动；不挪用D1余额 |
| F核对/有限重放 | 600s | 60s | 540s | 0 | 仅对本地完整且合法封口阶段有限重放；前置核验失败则不forward |

N0/N1合计≤1800s；拟合R/D1/D6/F合计≤6120s；两部分全执行总上限7920s（132分钟）。预留已包含在各上限内部，不另外加预算，不计入新增训练/probe可用时间。各阶段最多一次，不互借；失败不自动重试，未完成工作如实收口。新实现若在上述上限内不能完成，必须停止并先修订/复阅后续方案，不先执行再补预算。

历史成本仅作合理性参考：SOURCE六窗确定性S0的1000步训练区间 `wall_s=1284.5`，但其计时起点在构造/首probe之后、终点在保存/reload之前，不能直接当新驱动的完整阶段成本；原probe也没有本方案全部物理指标。D6的3600s保留余量，但不宣称新路径已实测可完成。D1没有等价历史完整墙钟，不按batch线性外推后偷减或增加预算；如在获准运行中观察吞吐，观测也计费，不做免费真实预跑。

### 8.2 工作截止、硬截止与封口

- 每阶段记录同一单调时钟的 `t0`、`work_deadline=t0+cap-reserve` 和 `hard_deadline=t0+cap`。首阶段t0由外层在uv/import/入口检查前采样；后继阶段的交接、来源检查、物化/缓存准备不得悬空在两个计时区间之间。最终矩阵/授权须绑定这一起止口径。
- 新构造、数组读取/解压单元、forward、backward、clip、step、probe、checkpoint重放均在实际调用前检查工作截止；CUDA调用返回后同步并复查。在 `now>=work_deadline` 后不启动任何新训练、probe或数值重放；不能依赖“复用Trainer”自动获得这些检查，I必须在更新单元边界及pre_backward等接点落实守卫并用受控时钟反例验证。
- 预留只用于保存已完成范围的部分诊断checkpoint及报告/成本/调用/实际返回证明；不得利用收口时间补训练、补probe、补本阶段的§7.4/§7.5数值核验或补F数值重放。D阶段所需step1000 probe、正常initial/last持久化、规范解码关联及§7.4第1–3项本地适用检查须在各自D工作时间窗内完成；F第4–5项及F重复的保存量核验只计入F工作时间窗。D工作截止时缺必需本地工作即按实际消费本地incomplete，不要求或提前执行F。
- 所有checkpoint/报告/台账写盘、同步、清理及子进程实际wait返回均计入硬截止，不能因“工作已做完”免计。单个已启动调用无法及时抢占而返回越界时，如实登记 `budget_exceeded`/incomplete和实际成本，阻断下游，不声称硬上限已被满足。
- 发布完成声明前必须证明封口/返回耗时被登记成本覆盖且未越硬截止；成本只升不降，不通过后移t0、漏尾部或回写较小成本制造余额。封口失败、返回覆盖未知、遗留running/publishing或成本不明时，不发布可消费完成声明、不启动下游；已测负结果保留，但完整性声明撤回。无法安全写完时保持未知/非可消费，而不是补零。

全量执行时D1/D6合计≤2000更新、8000训练micro、28000次窗口消费、唯一演示最多6个；两阶段固定probe各101次、合计202次，F额外probe forward≤4次，不混入训练样本数。每次probe必须恰好整批调用一次conditioner及一次backbone：D1每路径4000训练+101 probe=4101次，D6同为4101次，F每路径最多4次，所以完整D1+D6+F的conditioner/backbone forward各≤8206次。这是硬上限，不是条件估计；不允许D6拆成六次逐窗预测、为指标或核验额外预测，不能将实际多次调用合记一次。预检/物化/CPU保存量重算不执行预测；F不新增训练更新，失败分支按实际调用记账而不是强填名义上限。

真实启动前必须从最终实现冻结构造、加载、conditioner/backbone forward、loss/FK、backward、clip、step、probe、checkpoint及F重放的**逐类精确调用矩阵**和上限；不能拿更新数代替调用矩阵。矩阵未冻结即不可启动。

拟合工件增量上限提案2GiB；N0/N1共用CPU分析工件增量上限1GiB，§4给出包含临时文件的布局及最坏规模。每次写入前检查预计字节，写后按实际字节核对，未封口临时文件也计入；规模变化不得靠默认压缩比或未登记清理兜底。启动前重查磁盘，拟合空闲空间至少为增量上限加5GiB收口余量。`/dev/shm`不能作为唯一持久证据位置；不静默删除旧工件，也不复制约14GiB旧真实根。

### 8.3 新授权必须明确

1. 研究问题是新实现下的有限拟合能力，不是旧更新审计或旧预测cohort的重新命名复验；保留所有父历史消耗及原结论。
2. 数值收口结论、旧partial/539 gap及未覆盖范围已复阅；承认拟合不能证明完整数值或交互契约通过。
3. 绑定最终代码、配置、训练窗、源缓存、资产、§6.3运行时硬要求及实际事实、报告及精确调用矩阵；标注最终批准值、§7物理指标公式/终点/尖峰边界、probe批布局、§7.4核验及§7.6状态分层、§7.5规范解码域/中间FK覆盖限制、容量格式与§8.2内部预留/计费口径。
4. 新cohort定义和固定扫描根、阶段一次权限、预算及更新/调用上限、D1→D6条件、失败/未知历史停止规则。
5. initial/last诊断checkpoint与有限F重放的真实非零调用，以及禁止DDIM/正式val/test/生产接入的边界。

新cohort可采用明确区分研究问题的 `w1_patched_smallset_fit_v1`，但只有冻结材料和新明确授权完成后才创建；本轮不创建cohort、attempt、claim或授权草案。N0/N1的实现/运行范围也须按后续任务明确安排，本文不是自动执行命令。

## 9. F收口、归因与后续分支

下表中的拟合通过/不足仅引用F已核验的逐阶段最终裁决，不把D本地结果或交接候选直接升级为最终结论；F核验失败时按§7.4第5项保留局部事实并最终incomplete收口。

| 观察 | 可支持判断 | 后续候选（另立方案，不自动执行） |
| --- | --- | --- |
| 同输入公式、来源或区间仍失配 | 实现/证据问题尚未收口 | 最小修复及固定合成反例，不启动训练 |
| 数值边界说明成立但完整D1未达平均目标 | 本配置/数据/1000步预算下未达标；不证明实现有错或模型无法拟合 | 固定数据的一因素目标或优化诊断；不是直接扩数据 |
| D1平均过门、完整D6不过 | 当前条件下六窗平均目标未全部达到 | 逐窗误差与共享参数冲突分析，不直接归因为泛化或容量 |
| D1/D6平均过门、长序列仍差 | 不能排除窗内尖峰，平均拟合不能解释全部序列误差 | 先核对保存的逐帧/max，再分解三窗贡献/相对π路径；新候选独立验收 |
| 教师平均拟合过门但严格质量仍差 | 平均模仿目标不等于可行交互，不支持直接归咎教师 | 先定位局部误差/继承缺陷，再立可信监督或约束质量干预；须证任务可达性 |

阶段性保存窗拼接研究仅列为后继候选：生产 `stitch.py`仍是旧首尾SLERP，v2未获接入资格；数值矩阵转四元数修复不能替代旋转路径契约。既有 `box_ground`是固定输入问题，按现行协议单列，不可通过移动箱体增加机器人通过率。

只有独立质量改善成立后，再申请同数据/预算下的扩散比较、固定验证集质量评估和最终端到端计时；不能用小集训练速度、FK拟合或修复早拒绝证明加速。严格最终门、独立测试与生产接入仍属于后续交付。

## 10. 交付、状态与定义完成

后续每阶段各有新目录、机读结果、人读报告与来源SHA；旧材料只读。失败、mandatory gap、缺证据、预算停止分别记录，不能统一洗成passed或算法拒收。最终报告列明实际运行范围、调用、成本、唯一演示/窗数、更新数、未运行分支和限制。

本方案阶段清单（均未实施）：

- [ ] N0：原始保存量逐键/逐坐标分析、350/189及重叠集合核对。
- [ ] N1：最小合成反例与有限数值收口建议，保留原始不通过结论。
- [ ] I：独立小集诊断器、共享Trainer更新路径与合成反例。
- [ ] R：最终身份/数据/资源/预算/调用矩阵冻结及新明确授权。
- [ ] D1/D6：获准且前置满足时各一次受控拟合。
- [ ] F：有限重放、完整成本对账及go/no-go报告。

v0.1只创建方案并整理文档导航；v0.2修订预算/拟合/容量边界；v0.3补齐重放、运行时与probe调用硬门；v0.4拆分本地/最终状态并约束解码关联与中间FK证据范围。上述六项均未实施，没有运行新模型/FK/梯度/更新/训练或pytest。`objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false`与全部旧质量负结果保持。

## 11. v0.2审阅修订对照

| 审阅问题 | 本次文档修复 | 不变边界 |
| --- | --- | --- |
| 物理指标不能独立复算 | §7.1固定坐标、T/J/S、四项公式、SI值及无额外对齐；区分标准化loss、RMS和MPJPE | 四个平均目标数值不变，仍待复阅 |
| 平均门掩盖单帧尖峰 | §7.2完整逐帧误差、p95/max/最坏坐标；标签改为mean_fit_passed、spike_gate_status=not_evaluated | 不新增未经审阅的峰值通过门，不外推时序或交互可行 |
| 预算封口和终点不明确 | §8固定内部预留、工作/硬截止、单元守卫和尾部计费；§7固定1000步/101probe及最终last判门 | 各阶段和总上限不增加，未满步数为incomplete |
| N0全坐标CSV容量未证明 | §4完整紧凑标记+来源索引、有限极值表和493,374,464-byte保守上界；格式在N0前冻结、头预验在N0入口计费 | 不抽样代替覆盖，N0/N1工件共用1GiB |
| R与物化计费口径冲突 | §3.2、§8明确R仅元数据；D1/D6事前登记计时后再物化或核验缓存 | 不新增免费预检/模型调用，不重开旧授权 |

以上为v0.2文档修订记录，不是实现或实验验收。

## 12. v0.3复审修订对照

| 复审问题 | 本次文档修复 | 不变边界 |
| --- | --- | --- |
| P1：F重放及保存指标缺少消费端硬判据 | §7.3/§7.4固定原始数组、101×窗键覆盖、全部probe的CPU指标重推、D1交接前核验、initial/last逐位重放及失败撤回 | F仍仅四次fresh预测，不逐步重训；数值不一致即incomplete，不用平均门替代重放判据 |
| P1：未明确继承已审计数值运行时 | §6.3要求实际应用并核验既有RUNTIME_REQUIREMENTS、CUDA/float32和依赖身份，D/F端及每数值单元守卫；补反例 | 不新增真实校准，不改旧T6契约或授权，CPU指标归约不冒充生产路径 |
| P2：8206次forward仅为条件式 | §7批布局固定D1=1、D6=6；每次probe整批各一次conditioner/backbone，F匹配原布局；§8变为硬上限 | 8206/路径、2000更新/8000 micro/28000窗口消费及原预算不增加 |

以上为v0.3文档修订记录。四个平均目标、各阶段预算及内部预留、1GiB/2GiB工件上限和旧结论保持；精确调用矩阵仍须在I/R按最终实现冻结。

## 13. v0.4审阅修订对照

| 审阅问题 | 本次文档修复 | 不变边界 |
| --- | --- | --- |
| P1：本地完成与F验收交叉依赖 | §2/§7.2/§7.3/§8.2明确D只完成§7.4第1–3项本地适用检查；§7.6分离local字段、handoff_ready和F最终字段 | 不提前执行F、不增加F尝试；F否决只追加最终结果，保留本地事实和成本 |
| P2：原始x0与保存物理量关联不明、中间FK覆盖过述 | §7.5固定raw x0→规范CPU metric_decode的重算/逐位关联，三个根/关节指标只消费重解码；中间FK仅保存点度量自洽且显式not_evaluated | 四项公式/目标不放宽；不新增模型或FK批次，不宣称完整中间物理链验证 |

本次仅修订本方案和当前索引，未实施或运行任何阶段。预算/内部预留、2000更新/8000 micro/28000训练窗消费、8206次/路径forward、F最多4次fresh预测及原工件上限均不增加；新增CPU度量解码/核验在原D/F工作时间窗内计费。方案仍待复阅，精确调用矩阵/冻结材料和新明确授权仍是后续真实执行的必要前提，不恢复旧cohort或授权。

## 14. v0.5 独立审阅修订对照（N0/N1 工具修正与重跑）

2026-11-07 独立审阅未通过 N0/N1 首轮"已完整收口"验收（3 P1+5 P2），全部修复并以修正协议重跑：`n0/attempt_001` 与 `n1/attempt_001` 保留为被取代证据，权威为 `attempt_002`（`n0_coordinate_flags_v2` 格式）。本轮只改 N0/N1 工具、测试与人读记录；旧 T6 材料、契约与授权不改写。

| 审阅问题 | 修复 | 不变边界 |
| --- | --- | --- |
| P1：N0 漏跨路径分支（单侧可辨识/存在性不一致） | 格式升为 `n0_coordinate_flags_v2`（uint32）：新增 b16 `DELTA_PRESENCE_MISMATCH`、b17 `DELTA_ONE_SIDED_OBSERVABLE`；重跑标出审阅核实的 4 行单侧可辨识坐标 | uint16 v1 的 attempt_001 保留；容量上界按 4 bytes/坐标 重算（64MB+1024），总保守上界 525,374,464 bytes 仍 <1GiB |
| P1：同输入公式逐位证据不足 | 消费保存的 `same_input_p/i` 标量公式门（False 即 SOURCE_ANOMALY）；新增 `delta_measured == theta_after − theta_before`（float64 精确）一致性核对 | obs/ok 相对门仍重放，但不冒称公式逐位门 |
| P1：预算未覆盖启动与最终写盘 | 两 CLI 的 t0 采样前移至重模块导入之前；summary 写盘后采样最终墙钟，越硬截止即重写为 `budget_exceeded_after_seal`/不可消费完成声明（成本只升不降）；目录 fsync | 预算数值 900/60/840 不变 |
| P2：旧 629 项逐键对照未完成 | 解析 `i_adamw_gap:{model}:u{u}:{param}:{kind}` 键并填充 `old_state_*` 列，读取 `new_only_items`（424 行获得附注） | 只读附注，不改写旧分类 |
| P2：N1 判据不足 | C1 改为 audited 公式 vs 独立手写 float32 节点图逐位对照（中点 fallback 节点显式排除并登记）；C6 偏差输出全部喂给 N0 检测器（`compute_row_flags`）断言 implementation_issue_candidate 语义；C5 要求 ≥2 例跨路径门失败且全部区间内 | 检测能力主张限定于"显著偏差可检出"，不外推 |
| P2：键集只核计数 | `_canonical_rows` 按 `registration.roster.models[*].parameters` 精确集合核验（参数名×updates{1,2}，同数量替换被拒） | 行序仍为规范字典序（index 记录） |
| P2：P 侧写界错用 I 侧 theta | `bound_i == half_ulp32(theta_i)`；`bound_p` 经保存 `delta_ref_p` 反解 ratio（精确 f32）重构 P 侧公式参考 theta 并以 delta_ref 精确复核后逐位核对 | 反解不一致 fail-closed 为 SOURCE_ANOMALY |
| P2：attempt 覆盖风险 | 两 CLI 改为独占创建（mkdir exist_ok=False 递增），已有 001/003 时新写 004 | 旧 attempt 永不覆盖 |

N0 attempt_002（v2）结果：coverage_complete=true、0 异常、54.25s（预算 900s/工作截止 840s 内）；位分布新增 ONE_SIDED=4（恰为审阅核实的 4 行）、PRESENCE=0，其余位与 v1 一致；350/189/147/392 事件重放不变。N1 attempt_002：六类 47 用例全过（C1 独立对照 8/8、零 fallback 豁免；C5 三例门失败全区间内；C6 八偏差全经 N0 检测器检出），go=no_go 维持 `go_for_conditional_fit_diagnostics_review`。I/R/D1/D6/F 仍未实施；真实拟合仍须 §8.3 新冻结材料与新明确授权。

## 15. v0.5 第二轮复阅修订对照（封口口径、来源闭链与 bound 事件交付）

2026-11-07 第二轮独立复阅确认主要数值修复成立，但驳回"全部问题已闭合"（2 P1+3 P2），全部修复并以修正协议重跑：`n0/attempt_002` 与 `n1/attempt_002` 保留为被取代证据，权威为 **attempt_003**。本轮只改 N0/N1 工具、测试与人读记录；旧 T6 材料、契约与授权不改写；不进入 I 实施等待复阅。

| 复阅问题 | 修复 | 不变边界 |
| --- | --- | --- |
| P1：封口比较了不同时间口径（已用秒数 vs 绝对 `hard_deadline=t0+cap`，t0≈14467 时 901 秒不触发 900 秒超限） | `_write_summary_sealed` 重写为发布-核验循环：每次比较用**已用秒数**（`clock()-t0`）对 `wall_cap_s`；先写 `sealing`（非权威），写盘后采样，cap 内再写 `sealed_within_budget` 并再次采样核验 | 预算数值 900/60/840 不变；受控时钟反例钉死大 t0 下 901s 必须触发 |
| P1：最终写盘在最后采样之后（第二次写 summary 不再核验，慢写盘越界仍保留完成声明） | 每次写盘后都有采样；完成声明只能在"通过采样之后无任何写盘"时存续；采样越界即撤回为 `budget_exceeded_after_seal`（保守方向，撤回写盘自身无需再核验） | 新增反例：慢 sealed 写盘被撤回；通过采样后写次数断言为 2 |
| P2：N1 预算撤销未同步撤销 go 建议；CLI 用写盘前缓存决定退出码 | `N1_WITHDRAW_ON_EXCEED` 把 `go_no_go` 一并撤为 `no_go_budget_seal_exceeded`；两个 CLI 一律以**盘上最终 summary**（run/write 返回的落盘文档 + `seal.status`）决定打印与退出码，fail-closed | 非撤回路径 go/no_go 值不变 |
| P2：封口 SHA 锚未闭到旧封口链（协调修改 NPZ+报告可绕过） | 新增 `_verify_source_closure`：以 `launch_journal/w1-retest-20261106-003/return_observation.json` 为锚根（其 SHA 入册）→ `commits{P,I,F}` 逐级核验 commit.json/finish.json 文件 SHA → `index_row_snapshot.artifacts` 对每个消费输入（p_report/p_rows 级 NPZ×4/i_report/i_rows/NPZ×2）逐文件 SHA 对照；registration 对照三个 stage 快照一致的 `execution_registration_sha256`；P NPZ 另对照 F 终局报告 `worker_result.p_replay.arrays_sha256` 双锚；不匹配即硬失败。post-chain 的 analysis.json 非封口件，SHA 显式记录并标注 `analysis_journal_bound=false` | 闭链根为旧 journal 工件本身（只读消费）；不改写旧链 |
| P2：旧 629 对照遗漏 25 项 bound 事件 | `gap_rows.csv` 新增 `old_state_bound` 列；summary 新增 `old_chain_comparison`（items_total/items_by_kind/new_only_by_kind/rows_annotated），三类事件全部交付 | 只读附注，不改写旧分类 |

N0 attempt_003 结果：coverage_complete=true、0 异常、55.47s（seal=sealed_within_budget，round 1）；闭链全过——P/I/F commit SHA 与 journal 记录一致（`773cb48c…`/`459ef764…`/`7ee1965b…`）、registration SHA `87961838…`、F 重放锚 `44bb2b19…`/`8d02383d…`；事件重放 350/189/147/392 与标签 350/189/136 不变；旧对照 items_total=629（381 precision+223 sensitivity+25 bound）、new_only 5（1+4）、rows_annotated=424，`old_state_bound` 恰好 25 行 resolved。N1 attempt_003：六类 47 用例全过（计数 80/16/16/9），go=no_go 维持 `go_for_conditional_fit_diagnostics_review`，seal 3.95s（round 1）。测试 **29 passed**（N0 18+N1 11，新增 6 项：协调篡改/registration 篡改/封口时间口径/通过后零写盘/慢 sealed 写盘撤回/bound 列交付 + N1 go 撤回与两阶段正例）。I/R/D1/D6/F 仍未实施；真实拟合仍须 §8.3 新冻结材料与新明确授权。

## 16. v0.5 第三轮复阅修订对照（登记成本覆盖、终态时序与 registration 严格性）

2026-11-07 第三轮独立复阅确认前两轮修复与数值解释成立，另提出 2 P1+1 P2，全部修复并以修正协议重跑：`attempt_003` 保留为被取代证据，权威为 **attempt_004**。本轮只改 N0/N1 工具、测试与人读记录；旧 T6 材料、契约与授权不改写；暂不进入 I。

| 复阅问题 | 修复 | 不变边界 |
| --- | --- | --- |
| P1：登记成本未覆盖封口与实际返回（盘上 `measured_wall_s_at_summary` 只存写前 `pre`，写后样本仅在返回字典，N0 丢弃/N1 仅打印，无绑定覆盖证明） | 登记成本改为 **`pre + seal_tail_reserve_s`** 并与声明**同一写盘原子落盘**（预留 = 固定返回尾部 `SEAL_PROCESS_RETURN_TAIL_S=3.0` + 2×已观测写盘延迟，下限 1.0s）；写后核验 `post ≤ measured`，不覆盖即增大预留重写（成本只升不降）；新增 `verify_return_coverage`——CLI 在打印后独立采样实际返回时间并对盘上登记成本核验，未覆盖即盘上撤回为 `budget_exceeded_after_seal` 并 fail-closed 退出 3 | 预算数值 900/60/840 不变；正常路径反例断言盘上 54s ≥ 写后 40s |
| P1：完成声明先于最终核验落盘（`sealed/terminal=true` 在写后采样之前发布，写盘后/采样前被终止会残留未核验完成声明） | 三态时序：非终态 `sealing`（不可消费）→ 写后双核验（cap 与计费覆盖）→ 才写终态 `sealed_within_budget`，终态写盘自身的登记成本（含尾部预留）已先核验 ≤ cap 且与声明同盘；终态写盘后再采样核验，越界即撤回。残余暴露=写盘延迟超过登记预留的击杀窗口，以入册预留值定界（与 D/E 教义一致的审计口径） | 反例：慢 sealed 写盘仍被撤回（3 次写、终态 exceeded）；sealing 残留不可消费 |
| P2：registration 绑定字段缺失/类型错误被跳过（只剩一个有效 SHA 也能通过） | 逐阶段硬性要求 `execution_registration_sha256` 存在、为 64 位小写十六进制，且 P/I/F 三值相等；任一缺失/畸形/不一致即硬失败并带阶段名与三值明细 | 真实三阶段字段均匹配（`87961838…`） |

N0 attempt_004：coverage_complete=true、0 异常、54.29s 写盘前采样，登记成本 58.29s（=54.29+4.0 预留，覆盖封口与返回）且 CLI 返回覆盖核验通过（exit 0）；闭链、事件重放 350/189/147/392、标签 350/189/136、旧 629 对照（381/223/25）与 25 行 bound 全部不变。N1 attempt_004：六类 47 用例全过、go=no_go 维持 `go_for_conditional_fit_diagnostics_review`、seal 登记成本 7.87s 覆盖 3.87s 写前采样与返回、exit 0。测试 **31 passed**（N0 20+N1 11；新增返回覆盖覆盖/未覆盖两分支、registration 缺失/畸形/不一致三反例，并升级"通过后零写盘"反例为同时断言盘上计费覆盖写后样本）。I/R/D1/D6/F 仍未实施；真实拟合仍须 §8.3 新冻结材料与新明确授权。

## 17. v0.5 第四轮复阅修订对照（返回核验时序与撤回成本覆盖）

2026-11-07 第四轮独立复阅确认封口预留与 registration 严格校验修复成立，另提出 2 项 P1，全部修复并以修正协议重跑：`attempt_004` 保留为被取代证据，权威为 **attempt_005**。本轮只改 N0/N1 工具、测试与人读记录；旧 T6 材料、契约与授权不改写；暂不进入 I。

| 复阅问题 | 修复 | 不变边界 |
| --- | --- | --- |
| P1：返回核验采样过早（读盘/解析之前采样后不再查时；静态路径：采样 99s、登记 100s、读盘耗至 901s 仍判 covered；CLI 输出未 flush，采样不能称为实际返回证明） | `verify_return_coverage` 重写：**先读盘并解析，再采样判定**（必要读盘纳入被检时长）；两个 CLI 的 JSON 输出显式 `flush=True` 后才调用核验（打印耗时纳入被检时长）；**新增 `return_observation.json` 绑定证明**——覆盖成功时写入 `{covered, elapsed_s, registered_measured_s, seal_tail_reserve_s, wall_cap_s, checked_after_output_flush, summary_sha256}`（与 summary 字节 SHA 绑定），且证明写盘后**再采样核验**：证明写盘自身耗尽登记成本即翻转为撤回；两个 CLI 在所有退出路径上执行该核验，未覆盖即 fail-closed 退出 3 | 覆盖分支不重写 summary；非覆盖分支维持撤回 |
| P1：撤回写盘自身漏计（登记成本直接更新为写前 elapsed 后写撤回文件，无尾部预留、无写后覆盖核验；封口函数 `status=exceeded` 且 `post>cap` 直接返回也不查该次写盘计费） | **封口循环计费检查前移**：任何状态（含 exceeded）写盘后先核验 `post ≤ measured`，不覆盖即增大预留重写（成本只升不降），超限状态也须自身写盘计费覆盖后才返回；**撤回路径改覆盖成本循环**——每次重写登记 `measured = elapsed + reserve`（reserve = 返回尾部 + max(下限, 前次预留)，成本只升不降），写盘后核验覆盖，最多 SEAL_VERIFY_ROUNDS 轮；**不能证明时**保持最高观测成本并显式记 `cost_coverage_unproven=true`，状态维持 exceeded 不可消费（fail-closed） | 撤回后完成声明维持撤销；预算数值不变 |

N0 attempt_005：coverage_complete=true、0 异常、写盘前采样 54.13s、登记成本 **58.13s**、`return_observation.json` covered=true（elapsed 54.14 ≤ 登记 58.13，`summary_sha256` 与盘上 summary 一致、`checked_after_output_flush=true`，exit 0）；闭链、事件重放 350/189/147/392、旧 629 对照（381/223/25+25 行 bound）全部不变。N1 attempt_005：六类 47 用例全过、go=no_go 维持、登记成本 7.88s、返回观察 covered=true（elapsed 3.886 ≤ 7.885，SHA 绑定一致，exit 0）。测试 **31 passed**（新增/重写：返回覆盖覆盖分支证明绑定与 summary 零重写、复阅静态路径（早采样 99s→proof 写盘耗至 901s→写后再核验翻转为撤回→撤回成本覆盖证明）、迟采样（901s>cap 直接撤回且登记成本覆盖）、cap 内计费不足（150>100→撤回且登记≥写后样本）、封口超限写盘计费覆盖断言（盘上 2687s ≥ 905s）、两个 CLI 观察文件断言）。方案新增 §17、N0 §10/N1 §9 修订节、docs/README.md 已更新。**边界：T6 audit_incomplete/covered_contract_passed=false、539 gaps、cohort 34 次更新/1347.195507s、一次性授权已消费、partial/null/false/not_ready/不接入全部保持；I/R/D1/D6/F 未实施，真实拟合仍须 §8.3 新冻结材料与新明确授权；本轮未提交 git、未运行全项目套件。**

## 18. v0.5 第五轮复阅修订对照（诊断输出纳入核验与耗尽分支成本下界）

2026-11-07 第五轮独立复阅确认两项主修复成立，另提出 1 P1+1 P2，全部修复并以修正协议重跑：`attempt_005` 保留为被取代证据，权威为 **attempt_006**。本轮只改 N0/N1 工具、测试与人读记录；旧 T6 材料、契约与授权不改写；暂不进入 I。

| 复阅问题 | 修复 | 不变边界 |
| --- | --- | --- |
| P1：撤回后的诊断输出仍在成本核验之外（CLI 在 `verify_return_coverage` 返回后又执行 stderr 输出并 flush，此后不再采样；静态路径：撤回登记 163s、核验样本 160s 通过、诊断输出阻塞至 170s，仍留下"成本覆盖已证明"的低估记录） | 两个 CLI 的失败分支在 stderr 诊断输出（flush）之后**追加第二次 `verify_return_coverage`**：第二次核验重新读盘并在诊断输出之后采样，撤回覆盖成本循环以更高登记成本重写（成本只升不降）并写后核验——诊断输出耗时纳入被检窗口，低估记录被覆盖记录取代 | 成功路径无核验后输出；外层 wait 收口模式仍留待 I 阶段真实链采用 |
| P2：封口循环耗尽分支丢失成本下界（兜底分支重新复制原始 summary 并以不含登记成本的 seal 覆盖，可能把已增长成本降回最初值，也未标记 cost_coverage_unproven；单调时钟下每轮写盘都超预留即可耗尽轮数） | 兜底分支改为**保留最高已登记/已观测成本**（`max(highest_registered, highest_observed, now()-t0)`）写入 budget 与 seal 的 `measured_wall_s_at_summary`，并显式标记 `cost_coverage_unproven=true`、`rounds_exhausted=true`，状态维持 exceeded 不可消费（fail-closed）；返回事实带 `cost_coverage_proven=false` | 循环计费检查前移（第四轮）不变 |

N0 attempt_006：coverage_complete=true、0 异常、写盘前采样 53.91s、登记成本 **57.91s**、返回观察 covered=true（elapsed 53.91 ≤ 57.91、SHA 绑定一致、exit 0）；闭链、事件重放 350/189/147/392、旧 629 对照（381/223/25+25 行 bound）全部不变。N1 attempt_006：六类 47 用例全过、go=no_go 维持、登记成本 7.88s、返回观察 covered=true（elapsed 3.884 ≤ 7.883+、SHA 一致、exit 0）。测试 **33 passed**（新增：对抗时钟钉死轮数耗尽路径——兜底分支保留最高成本、`cost_coverage_unproven` 标记、不回写较小成本；第二次核验覆盖诊断输出——复阅静态路径（登记 163s/核验 160s/诊断至 170s→第二次核验重登记 ≥170s 且只升不降））。方案新增 §18、N0 §11/N1 §10 修订节、docs/README.md 已更新。**边界：T6 audit_incomplete/covered_contract_passed=false、539 gaps、cohort 34 次更新/1347.195507s、一次性授权已消费、partial/null/false/not_ready/不接入全部保持；I/R/D1/D6/F 未实施，真实拟合仍须 §8.3 新冻结材料与新明确授权；本轮未提交 git、未运行全项目套件。**

## 19. I 阶段实施记录（小集拟合诊断器与合成反例）

2026-11-08 按 §6/§7 完成 I 阶段首个实施交付：独立诊断模块 [src/ireset/eval/w1_smallset_fit_diagnostic.py](../../src/ireset/eval/w1_smallset_fit_diagnostic.py)、合成验收 CLI [scripts/verify_w1_smallset_fit_diagnostic.py](../../scripts/verify_w1_smallset_fit_diagnostic.py)（synthetic-only，拒绝一切真实输入）与 16 项测试 [tests/test_w1_smallset_fit_diagnostic.py](../../tests/test_w1_smallset_fit_diagnostic.py)。**退出条件达成：D1/D6 驱动只通过已审计 Trainer 更新单元（accumulate_micro/clip_gradients/step_optimizer/predict）执行训练，身份/计费/数据分母测试通过**；详见 [I 阶段验证记录](../verification/omniretarget_w1_smallset_fit_diagnostic.md)。

- 冻结契约：FitPlan 非合成计划偏离冻结默认（1000 步/101 probe/grad_accum 4/D1 micro 1/D6 micro 6/seed 20261008/lr 1e-4/wd 1e-5/clip 1/d240 架构）即拒绝；FitRegistration 窗数/字典序/跨度/split 校验+注册↔数据集窗键绑定；`metric_decode_v1`（`cpu_float64_from_saved_x0`，生产 FP32 常数精确提升+复用 dtype 泛型 6D 解码+干净限位映射）；12+8 类保存记录表（形状/dtype 冻结）；§7.1 四门 float64 归约+逐窗合取（宏平均 `macro_is_not_a_gate`）；§7.2 尖峰可见性（p95 linear/max 首序决胜/最坏坐标，`spike_gate_status="not_evaluated"` 冻结）；§7.6 状态分层（校验器拒绝未加前缀最终字段）。
- 运行时硬门（§6.3）：`ProductionRuntimeGuard` 绑定 `w1_real_fk_audit` 的 `configure_production_runtime()`/`RUNTIME_REQUIREMENTS` 字节并逐单元重检；构造/加载后参数与浮点缓冲 dtype/device 契约核验；合成路径 `SyntheticRuntimeGuard` 显式不主张生产等价。
- 本地核验（§7.4 第 1–3 项）：覆盖多重集（缺/重/额外窗键逐类拒绝）、全部 probe 从磁盘原始 x0 重解码并逐位绑定（`decode_binding_mismatch`）、误差曲线/四门/p95/max/最坏坐标从重解码量重算并与保存值精确对账、initial/last checkpoint 角色/step/root_stats/配置绑定；D1 交接收据由封口+返回覆盖+授权在 D6 计费区间内重推（候选不自治）。
- F（第 4–5 项）：身份门先于任何模型工作（失败零 forward、build_modules 零调用）；每阶段 fresh 构造 1 次+initial/last 各 strict 加载一次、恰 2 次 fresh 预测，x0/生产解码/FK 点对保存数组逐位绑定，fresh metric_decode 派生指标精确对账；不一致即 `replay_mismatch`、`bitwise_no_tolerance`、无重试/无放宽路径，最终字段 `complete_evidence/replay_passed/mean_fit_passed` 按第 5 项收口；F 失败不写阶段目录（电池断言阶段报告 SHA 不变）。
- 计费/身份：attempt 台账一行一 (stage,attempt)、事前 running、SIGKILL 残行=未知历史=停止、磁盘×台账集合相等、失败不重置预算（`budget_exhausted`）；R 只读元数据守卫（数组/`torch_load` 读取即拒）与无计费区间物化拒绝；受控时钟反例钉死截止后拒绝新单元与封口撤回（成本只升不降，复用 N0 的发布-核验封口与返回覆盖工具）。
- 权威合成验收 `derived/audit/w1_smallset_fit_diagnostic/attempt_001/`（exit 0，工件 7.0MB，封口 `sealed_within_budget`、登记成本 1512.79s ≤ 3300s、返回覆盖 covered）：微型 D1（300 步）四门全过（RMSE 0.00115m）→ `handoff_ready=true` → 微型 D6 六窗四门全过（macro RMSE 0.00211m）→ F 两阶段逐位重放通过（`fit_pass`）；35 项 §6.2/v0.3/v0.4 反例全捕获（含 target 泄漏、联合 clip 覆盖、D6 逐窗六次记一次、1-ULP fresh 数组差异、中间 FK 自洽替换保留 not_evaluated 限制等）。
- **边界：R/真实 D1/D6/F 未实施、未授权、未运行；本模块无真实数据入口；合成四门通过不说明真实 W1 训练窗可达；T6 audit_incomplete/covered_contract_passed=false、539 gaps、cohort 34 次更新/1347.195507s、一次性授权已消费、partial/null/false/not_ready/不接入全部保持；真实拟合仍须 §8.3 新冻结材料与新明确授权；本轮未提交 git、未运行全项目套件（仅本交付测试文件 16 passed 与 N0/N1 相关回归 33 passed）。**

## 20. I 阶段第二轮独立审阅修订（7 P1 + 2 P2）

2026-11-08 独立审阅否决 I 阶段首轮"完整收口"（7 P1 + 2 P2，复现脚本留存于 /tmp/opencode/），全部接受并修复；`derived/audit/w1_smallset_fit_diagnostic/attempt_001/` 保留为被取代证据，权威验收重跑为 attempt_002。修复明细见 [I 阶段验证记录 §8](../verification/omniretarget_w1_smallset_fit_diagnostic.md)；要点：①教师侧与每 probe 的 metric_decode 均从保存原始 x0 重解码并逐位绑定（`teacher_decode_binding_mismatch`），生产 FP32 解码量经同一 `decode_clip` 重推导绑定，FK 点显式 `teacher_fk_state_binding_status="not_evaluated"`；②`resolve_handoff`/F 完整消费封口+返回证明（SHA 绑定与覆盖链）+D1 交接候选（`finish_stage` 落盘、D6 入口消费）；③必需工件集合+报告→索引 SHA 闭链，索引恒排除三个晚置绑定文件；④checkpoint 加载前完整键集/shape/dtype/有限性契约（本地核验另断言全部浮点张量为 float32）；⑤运行时守卫覆盖每个 micro/pre-backward/clip/step/probe/checkpoint；⑥F 加载与输入读取后插入工作截止检查；⑦CLI 先 flush 输出再核验返回覆盖并按盘上状态退出；⑧聚合合取改读 `metrics.window_mean_fit_passed` 并补全过控制；⑨逐步 `n_micro` 修复。反例电池扩至 **41 项全捕获**（新增教师替换/证明绑定/空索引/dtype 篡改/漂移停机/加载越界六类），审阅者攻击脚本在修复后代码上全部 fail-closed；测试 **16 passed**。**边界：R/真实 D1/D6/F 未实施、未授权、未运行；真实拟合仍须 §8.3 新冻结材料与新明确授权；本轮未提交 git、未运行全项目套件。**

## 21. I 阶段第三轮独立审阅修订（5 P1 + 1 P2）

2026-11-08 独立复审确认第二轮修复有效（attempt_002 SHA/宏合取/n_micro 已核实）但提出 5 P1 + 1 P2，全部接受并修复；attempt_002 保留为被取代证据，权威验收重跑为 attempt_003。修复明细见 [I 阶段验证记录 §10](../verification/omniretarget_w1_smallset_fit_diagnostic.md)：①返回证明的 registered/cap 逐项绑定封口报告的 seal 数值（自洽伪造证明被 `return_observation_registered_mismatch`/`wall_cap_mismatch` 拒绝，F 与 D6 交接同拒）；②`finish_stage` 时序改为 seal→候选写盘→最终返回覆盖核验（候选写盘纳入被检窗口）；③CLI 的必要回读移至最终覆盖采样之前（采样后零 I/O）；④F 构造移入守卫区（`f_stage_*_pre_construct` 在 `build_modules` 之前）；⑤F `_inputs_ready`/`_fresh_complete` 处预算+运行时双检；⑥`_read_json_object` 守卫使非对象报告/证明结构化失败（F 不崩溃、f_report 照常落盘）。电池扩至 **46 项全捕获**（新增 5 项）+1 项 pytest CLI 时序用例，测试 **17 passed**；审阅者 round-2 脚本全部 fail-closed。**边界：R/真实 D1/D6/F 未实施、未授权、未运行；真实拟合仍须 §8.3 新冻结材料与新明确授权；本轮未提交 git、未运行全项目套件。**

