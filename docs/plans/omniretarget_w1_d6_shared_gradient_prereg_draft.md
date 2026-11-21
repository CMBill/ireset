# W1 固定六窗共享梯度诊断预注册草案

**v0.3 · 2026-11-21 · 复阅修订，非授权材料。** 本稿拟在 D6 的 step0 权重上测量首个六窗训练批的总损失梯度，判断其余五窗是否至少抵消共同窗等权贡献的四分之一。数值门和资源额度均为提案。真实阶段须在实现验收、逐项 call/I/O 矩阵冻结并取得新的明确授权后才可执行，训练更新额度为 **0**。

## 1. 研究问题

D6 与单窗 B 的共同窗为 `sub10_largebox_097:0:60`。四项指标的首个持续差距都始于 step10，学习率在第 501 次更新才切换。六窗复盘的组级调权候选为 `null`。本稿检验一个可直接测量的量，即在同一参数状态和同一六窗计算图中，其他窗的梯度沿共同窗梯度方向的投影。判据只覆盖这一个起始状态的原始训练总损失。 [六窗复盘 §2–3](../verification/omniretarget_w1_d6_window_error_review.md) · [主方案 §9](omniretarget_w1_numerics_closure_fit_diagnostic_plan.md#9-f收口归因与后续分支)

## 2. 固定来源与批次

1. 以 D6 独立裁决 SHA-256 `1de06dae2eca5d554b948db24e33399647f868ef60283ace4775e887e57e56b0` 为外部锚。核对 `complete_evidence=true`、`replay_passed=true`、`verdict=fit_insufficient`，以及 `launch`、D6/F 报告和 `d6/artifacts_index.json` 的字节绑定。两份阶段报告须具有同一身份并指向同一 D6 索引。以该身份的 `r6_admission` 绑定历史准入包，取 `package.contract` 的注册与生效配置，以及 `package.bindings` 的原始 `root_stats.file`、41 个生产资产及运行时登记。R6 包中的 421 条 `bindings.files` 只作为绑定元数据消费，不打开其指向的文件；唯一允许的源数据值读取为 root-stats NPZ 和 D6 `probe_inputs.npz`。核对本次允许读取的路径、长度、成员集合和 SHA-256；授权材料给出封闭的逐路径读写清单。 [独立裁决](../../derived/audit/w1_d6_joint_fit_lr_half_v1/execution_review/verified_outcome.json) · [R6 元数据](../../derived/audit/w1_d6_joint_fit_lr_half_v1/r6/attempt_001/r6_admission.json)
2. D6 索引下只消费 `initial.pt`、`probe_inputs.npz`、`payload.json`、`optimizer_roster.json`、`history.jsonl` 和 `probe_index.jsonl`。按登记的角色、长度和摘要验证同一份读取快照，两份 JSONL 整文件验哈希后各解析首行。`initial.pt` 为 `role=initial`、`step=0`，实测长度 15,005,267 bytes，SHA-256 `1a7f11410e042831055d8c6b5500d06b242041eca7cbc76e825156a9af403ee2`。沿用历史端点准入规则，基于绑定的生效配置重建预期 `TrainConfig` 并与 checkpoint 对账，核对 `step`、`val`、可用性标记、D6 身份和原始根统计。model/conditioner 状态摘要须等于首条 `idx=0, step=0` probe 日志的 `endpoint_state_sha256`。CPU 安全加载沿用 D6 的 `weights_only=True` 和限定的 NumPy 类型白名单。两个模块的完整键集、dtype、形状和有限性须在 strict load 前通过。加载与转移后的状态摘要须再次匹配。 [D6 工件索引](../../derived/experiments/omniretarget_first/w1_d6_joint_fit_lr_half_v1/d6/artifacts_index.json) · [端点加载和绑定实现](../../src/ireset/eval/w1_d6_joint_fit.py#L571)
3. 从 `probe_inputs.npz` 取注册顺序下的六个 `[0,60)` 训练窗。按历史输入契约拒绝缺失、额外或重复成员，核对 `x0`+五个条件字段+`wfc`+`valid`+`clip_idx`+`start` 共 **10 个成员/窗**的形状、dtype、字节序和数值域。逐槽复算 `payload.json` 所覆盖的八个模型输入摘要，另核验 `clip_idx` 和全零 `start`。`valid` 六窗均为全 1。使用 `history.jsonl` 第 1 步的 `micros[0]` 重建首个 micro-batch，要求 `step=1`、`micro=1`、`event=2`、`kind=train`。实际 `canonical_slot` 顺序为 **`[1,0,4,3,2,5]`**，其中共同窗是规范槽 0、批位置 1。按历史实现复算有序 `payload_sha256`。设备转移后从实际张量重推批次顺序及逐槽字段摘要，复核整条消费记录。全部逐窗输出按完整 `window_id` 关联。 [批次摘要实现](../../src/ireset/eval/w1_d6_joint_fit_evidence.py#L168) · [首个 micro 的归档账本](../../derived/experiments/omniretarget_first/w1_d6_joint_fit_lr_half_v1/d6/history.jsonl)
4. 参数名册取历史 `optimizer_roster.json` 的冻结顺序。逐项核对实际 `model.*` 和 `conditioner.*` 的名字、形状、dtype 及可求导状态。名册共 **130 个张量、3,739,210 个坐标**。根统计只读 R6 登记的 `file`，资产按闭包快照构造原生产 FK 和箱体静态变换。正式材料同时绑定历史 `launch.implementation.files` 中本次所用的预测、损失、FK、载荷及运行时依赖，以及新增诊断、独立复核代码和外层启动命令。外层 supervisor 的冻结命令/源码身份在授权材料中列明；由该 supervisor 在导入数值 worker 前核验完整 worker 闭包，再在 wait/清理后复哈希，同一文件清单两次须完全一致。 [真实装置构造](../../src/ireset/eval/w1_d6_joint_fit_real.py#L222) · [参数名册](../../derived/experiments/omniretarget_first/w1_d6_joint_fit_lr_half_v1/d6/optimizer_roster.json)

## 3. 整批目标与逐窗梯度

在外层 supervisor 创建的干净、无继承模块状态的独立 Python worker 中使用历史 `DeterministicG1`、`ConditionEncoder`、六窗批形状和 `Trainer.predict`，配置为 `seed=20261008`、`d_model=240`、`n_layers=4`、`n_heads=6`、`ffn_mult=4`、`time_window=31`。从单个已验证输入快照构造六条 `RegisteredWindowDataset` 记录，分别将同一 dataset 传给 Trainer 的 train/val 参数；直接调用 `Trainer` 构造器，不调用 D6 的 `build_trainer` helper。Trainer 会创建两个 DataLoader，但不创建 iterator、不取样、不调用 `__getitem__`；仅对这六条已物化记录调用一次 `default_collate` 得到历史排序批，不调用 `canonical_collates`。构造 Trainer 时设置 `build_optimizer=False`。从已验证的 step0 状态加载两个模块，并设为训练模式。运行时为登记的 `cuda:0` / float32，TF32 与 CPU/CUDA autocast 关闭，matmul precision 为 `highest`，其余身份和标志与 D6 的登记值逐项核对。构造、转移、加载和数值操作入口复核实际运行时；每个已进入操作返回后同步并检查截止。RNG 不变门以 Trainer 构造、模块加载和 DataLoader 创建完成后的快照为起点；`Trainer.__init__` 固有的 Torch/NumPy reseed 与模型随机初始化属于测量前初始化，不计入该门。`condition_only` 给 conditioner 的五键为 `kps_obj`、`kps_clip`、`box_t_clip`、`box_quat`、`contact`。`x0` 是损失目标。 [Trainer 预测](../../src/ireset/train/trainer.py#L319) · [条件字段守卫](../../src/ireset/eval/w1_d6_joint_fit.py#L159)

整批只执行一次预测和一次原生产 `LossAssembler`。其返回的活标量即 `L_batch`，并要求它与 `observer['total']` 为同一 Tensor 对象。取 `observer['term_elements']` 的五项和独立键 `observer['fk_elements']`，保留它们到 `x0_pred` 的计算图。按历史批位置切片，使用下表的固定分母、生产项序与实际损失权重构造六个活标量 `L_i`。逐窗标量按生产左结合顺序明确构造：`((((w_pos·pos + w_rot6d·rot6d + w_joint·joint) + w_geodesic·geodesic_rad2) + w_vel·vel) + w_fk·fk_m2)`。运行前逐项核对有效权重。教师 FK 由同一次生产损失的目标分支生成，使用绑定的教师 `x0` 与有序 15 个 site。 [LossAssembler 归约与 observer](../../src/ireset/train/losses.py#L90)

| 项 | 每窗分母 | 权重 |
| --- | ---: | ---: |
| `pos` | 60 × 3 | 1 |
| `rot6d` | 60 × 6 | 1 |
| `joint` | 60 × 29 | 1 |
| `geodesic_rad2` | 60 | 0.5 |
| `vel` | 59 × 38 | 0.5 |
| `fk_m2` | 60 × 15 个平方点距 | 2 |

`w_contact=0`，实际接触分支计数为 0。保存六项有限非负的 float32 元素、原生产返回的七项标量（六项及 `total`）和六窗求导实际消费的分项与总标量。CPU 从原始元素与冻结权重以 float64 重算每项的整批值和各窗值，逐个与实测值对账。对账与整批对六窗均值均使用 `abs(a−b) ≤ 2e-6 × max(abs(a), abs(b))`。在首个 VJP 前通过同口径的分项与总量检查。

用同一计算图执行七次 `torch.autograd.grad`，先取 `L_batch`，再按规范窗序取 `L_i`。每次调用前冻结有序输入名册 `(x0_pred, model 参数, conditioner 参数)`、其名称/形状/dtype 摘要及进程内对象身份，冻结标量角色、窗键/批槽和 `call_id` 并写入 request 记录。整批入口核对实际传给 autograd 的标量与 `LossAssembler` 返回的标量是同一张量对象；逐窗入口核对其活标量是由登记窗键映射到实际批槽的重建结果。调用返回的元组严格按 request 名册位置解包：第 0 项为 `h`，其余项按冻结参数名册逐项映射为 `g`。在任何排序、复制或序列化前，紧接着生成一条不可拆分 response 记录，绑定 request SHA、同一 `call_id`、`h` 与每个具名 `g` 的形状/dtype/原始字节 SHA。CPU 正控使用解析可手算的共享参数图，逐项核验七次原始 autograd 返回与预期具名梯度；仅交换返回元组中的参数梯度项必须翻转该正控。独立读盘复核逐调用核对 request-response 成对绑定、张量名册顺序和原始数组摘要。首六次保留计算图，末次释放。`h` 必须为 `[6,60,38]`，`g` 逐张量匹配名册形状，二者均为有限、稠密的 float32。逐次移至 CPU 保存，再释放该次返回的显存张量。测量前后核对参数及持久 buffers 相对**调用前快照**逐位不变；Torch CPU/CUDA RNG 状态逐设备相等，且参数 `.grad` 仍为空。模型 construction 的随机初始化与 Trainer constructor reseed 不纳入求导区间的 RNG 不变门。`g_batch` 是重建的首个 micro 的**未缩放、裁剪前总目标梯度**。原 Trainer 在实际累积时将该批损失除以 4。 [Trainer 的 micro 缩放](../../src/ireset/train/trainer.py#L391)

对每个 `h_i`，其余五个批槽必须数值全零，且 `h_i` 全零时该次 `g_i` 也必须全零。逐槽比较时先把两侧提升为 float64，要求 `||h_batch[slot(i)] − h_i[slot(i)]/6||₂ ≤ 2e-4 × ||h_i[slot(i)]||₂/6`。左侧直接以归约后的 float64 范数与 float64 右侧比较，不先舍入回 float32；右侧为零时必须逐位精确相等。逐窗角色和实际批槽在此参与判门。

从原始 float32 值提升到 float64 后计算 `g_mean = sum(g_i)/6`。对每个参数张量、model/conditioner 各自整体及两者联合，核验 `||g_batch − g_mean||₂ ≤ E`，其中 **`E = 1e-3 × sum_i(||g_i||₂)/6`**，均使用该组坐标。该提案容差来自纯合成生产结构复核：固定一组 D6 架构随机初始化，对 9 组 CPU 生成、CUDA 消费的条件/目标输入执行一次预测与七次求导；在原 `2e-4` 下，实际 model 参数加性误差最高约为限值的 `9.95e-4`，会把有效合成控制判为不一致。重定义归一量 `rho = ||g_batch − g_mean||₂ / (sum_i ||g_i||₂/6)` 后，9 组中 model/conditioner/joint 的最大 rho 为 `2.014e-7`；因此预注册 `1e-3` 上界约为该合成浮点底噪的 4,965 倍，且每组须在冻结合成复验中低于该上界的 25% 才可准入。该余量只覆盖固定 float32 求值/归约产生的加性误差，不松动裁决提案线，最终候选还把 `margin` 加到抵消要求的负向一侧。六窗都为零时要求整批也精确为零。所有原始数组保持 float32 NPY；复核转为 float64 C-contiguous。范数平方和及点积使用 NumPy `sum(dtype=float64)` 对每张量 C-order 展平后归约，再按 optimizer roster 顺序左序累加张量值；`g_mean` 按窗 0→5 顺序逐数组 float64 加法后除以 6。正式实现身份包括 NumPy/PyTorch 版本与 CPU/GPU runtime facts。失配记为证据不完整并给出具体窗、参数或组。保存的各项梯度须通过独立读盘复核后才可用于裁决。

**复阅实测（2026-11-21，非 I/C 验收）**：在 RTX 3090 上用随机初始化的实际 D6 `Trainer` 架构、生产 `LossAssembler` 和合成 FK，完成 9 组只读输入的七次 VJP；不加载 D6 checkpoint、不读真实批或资产、不执行更新。九组的最高参数 `rho=2.0132e-7`，最高 `h` 误差为其 `2e-4` 上限的 `5.35e-4`，损失总量对六个逐窗标量均值的最高相对差为 `2.29e-7`。单组及 module/joint 参数门均在 `E=1e-3` 下通过，且均低于 25% 准入线。原 `E=2e-4` 对首组 model/joint 正控为 4.97 倍超限，因此已在任何真实授权前将提案定为上值；正式 I/C 仍须独立重做冻结的九输入组并增加第二模型初始化种子。

## 4. 投影及判据

`g_0` 取共同窗的梯度，`E_all` 是联合参数组的加性容差。先求未归一的点积和范数，通过 `||g_0||₂ > 1e-10` 后再计算以下比值。

```text
own          = ⟨g_0, g_0⟩ / 6
other_i      = ⟨g_0, g_i⟩ / 6                    (i=1…5)
other_ratio  = sum(other_i) / own
batch_ratio  = ⟨g_0, g_batch⟩ / own
margin       = ||g_0||₂ × E_all / own
```

`own` 是共同窗在六窗等权平均中的自身贡献，为该窗梯度平方范数的 `1/6`。`other_ratio` 是五窗投影和相对于 `own` 的比值。对两条曲线 `L_0(θ−t·g_batch)` 和 `L_0(θ−t·g_0)`，其 `t=0` 导数之比为 `batch_ratio/6`。同时报告五个 `other_i`、六窗范数、可定义的余弦、`batch_ratio − (1 + other_ratio)` 及两个模块各自的点积贡献。整批加性门经柯西–施瓦茨不等式给出 `abs(batch_ratio − (1 + other_ratio)) ≤ margin`。该余量只量化整批梯度实测值与逐窗平均之间的一致性。`[1+other_ratio−margin, 1+other_ratio+margin]` 全正或全负时，分别标记沿 `−g_batch` 的共同窗总损失一阶方向为下降或上升，区间含零时方向待定。

证据完整、`||g_0||₂ > 1e-10` 且 **`other_ratio + margin ≤ −0.25`** 时，可建议另拟覆盖 step10 的短前缀对照。其余完整结果为 `attribution_insufficient`。`other_ratio − margin > −0.25` 表示抵消幅度低于提案线。下界 `≤ −0.25` 且上界 `> −0.25` 时，阈值边界未辨。`||g_0||₂ ≤ 1e-10` 时，记录共同窗梯度过小，相关比值、余量及方向字段为 `null`。结论限于 step0 的首个 micro 和原始欧氏梯度，step10 用于界定候选对照长度。 [首个持续差距的归档依据](../verification/omniretarget_w1_d6_window_error_review.md#3-共同窗-d6-与单窗-b)

## 5. 调用上限、证据及独立复核

拟用独立根 `derived/audit/w1_d6_shared_gradient_step0_v1/` 和唯一正式 `attempt_001`。外层以 `mkdir(exist_ok=False)` 独占占用正式尝试；目录已存在时拒绝，不复用或覆盖。先登记授权、输入身份和调用预留，再启动数值进程。下表为一轮完整执行的精确调用计数。授权材料须另包含逐路径 I/O 清单，绑定每个输入/输出文件的用途、字节上限、读取/哈希/成员访问次数和同步回读次数；输入摘要与文件读取分别计量。所有计数由冻结 call matrix 的入口守卫累积，实际矩阵须逐项等于冻结值。异常收口保留实际进入和完成的计数。独立复核阶段为 CPU 落盘值核验。

| 单元 | 次数或上限 |
| --- | ---: |
| D6 / R6 来源元数据文件读取、哈希、解析 | 授权文件清单内每个白名单文件各 1 次；同一已验证字节快照供所有核验复用 |
| D6 索引及六份选定工件读取 / 哈希 | `artifacts_index.json`、`initial.pt`、`probe_inputs.npz`、`payload.json`、`optimizer_roster.json`、`history.jsonl`、`probe_index.jsonl`，各 1 次；NPZ 数组打开复用同一快照字节 |
| 运行实现闭包读取与哈希 | 外层 supervisor 在数值 worker 导入前、wait/清理后各 1 次/文件 |
| 历史输入 NPZ 打开 / 成员读取 | 2 个归档 / `probe_inputs.npz` 60 个成员（10×6）、root stats 2 个成员 |
| `initial.pt` 字节读取、CPU 反序列化 / 模块 strict load | 1 / 1 / 2 |
| 41 个生产资产读取与哈希 | 各 41 次 |
| Python runtime interpreter / 5 个依赖源码 / interpreter binary 身份哈希 | 1 / 5 / 1 |
| pytorch-kinematics distribution metadata / file roster / 两个 FK 源码文件哈希 | 1 / 1 / 2 |
| site 记录 / URDF 解码，root stats 数组构造 | 1 / 1 / 1 |
| G1 spec 构造（joint / scene plain） | 1 / 1 |
| tokenizer / metric decoder / model / conditioner 构造 | 各 1 |
| model / conditioner 设备转移 | 各 1 |
| 生产 FK / FK chain 构造，FK chain 设备转移 | 各 1 |
| MuJoCo plain/scene 编译 / data 构造 / reset / `mj_forward` | 3 / 1 / 1 / 1 |
| `LossAssembler` 构造 | 1 |
| `RegisteredWindowDataset` 构造 / `default_collate` | 1 / 1 |
| train/val DataLoader 构造 / iterator 创建 / 数据迭代 | 2 / 0 / 0 |
| 六窗输入批设备转移 | 1（十键） |
| conditioner / backbone / 生产总损失 | 各 1 |
| 生产状态解码 | 2（预测与目标各 1 次，均为 360 状态） |
| FK 前向 | 1 次预测可微调用 + 1 次教师目标 no-grad 调用（各 360 状态、15 sites） |
| 同时取得预测梯度和参数梯度的 autograd 调用 | 7（整批 1，逐窗 6） |
| 运行时/截止边界检查及 CUDA 同步 | 数值代码实现后、授权前由可执行矩阵固定精确点位与次数；覆盖 setup、strict load、设备转移、conditioner/backbone、loss 内 decode/FK、各 VJP 和终检；缺点/多点均拒绝 |
| 接触分支 / 优化器构造 / 原生训练累积 / 裁剪 / 更新 / checkpoint 写入 | 各 0 |

完整矩阵还列明文件与摘要操作的按路径次数。数值输出固定为 20 个 `.npy`：7 份 `g`、7 份 `h`、6 份逐元素损失；每个文件恰写/哈希 1 次，并由独立复核恰读/哈希 1 次。固定 JSON 工件为 `launch.json`、`inputs.json`、`call_matrix.json`、`vjp_records.json`、`loss_scalars.json`、`counts.json`、`independent_review.json`、`outcome.json` 和外层 `return_observation.json`，每个恰写/读回 1 次；`stdout.log`/`stderr.log` 分别限 256 KiB、每份创建 1 次且只由外层读回 1 次。名册同时计入原子临时文件、文件与目录同步次数；不在清单中的文件读取、写入、重试或日志输出均拒绝。

整轮墙钟硬上限拟为 **300 秒**。外层调用方在首笔真实输入核验、占用预留和解释器启动前取单调起点。240 秒工作截止涵盖计算、输入复哈希和独立复算，余下 60 秒用于收口。每个冻结调用单元进入前及同步返回后均检查对应截止；已进入的 CUDA 单元允许完成，若返回后越过工作截止则记账、标记 `incomplete` 并阻断所有后继数值单元。对组合调用还须守住内部的下一个已计量单元边界：`Trainer.predict` 的 conditioner 返回后、backbone 进入前；生产损失每次 `decode_clip` 返回后及每次 FK 调用返回后、下一次 decode/FK 前；各次 VJP 与 CPU 归约前后。只做组合调用外层检查不能替代这些内部门。所有 CUDA 完成时钟检查都先同步设备。工作截止仍未完成核验时，外层停止并回收仍在工作的进程组。开跑前记录单卡可用显存，结束时报告峰值。worker 在 240 秒工作截止前仅写入 `publication=pending_actual_readback` 的候选及终态数据并同步；外层 supervisor 对该候选、日志及 worker 实际 `wait()`/清理完成后写入并回读 `return_observation.json`，再采样实际最终时钟，核验总墙钟不超过 300 秒。此最后样本只进入当前调用返回的内存 `ExternalReturn` 见证，不再做任何文件/目录 I/O 或成功输出；消费者必须同时持有该见证及盘上 pending 收据，裸文件不能自证已完成。收口后实现闭包重哈希须在 `return_observation` 写入前完成，并与导入前 SHA 完全一致。来源或计数失配、证据不足、超限、梯度加性失配或复核不一致，均以 `incomplete` 收口并记录已测范围和失败位置。进程异常退出且缺终态记录时，保留正式尝试及未结算预留。

工件峰值上限拟为 **128 MiB**，覆盖临时文件、定额日志和封口。七份参数梯度数据为 `7 × 3,739,210 × 4 = 104,697,880 bytes`，七份预测梯度共 `383,040 bytes`，六项逐元素损失共 `131,568 bytes`；数组负载合计 **105,212,488 bytes**。每个 VJP 的参数梯度和预测梯度分别保存为小端 float32 NPY，共 14 文件；另保存六个逐元素损失数组，共 20 文件。每个 NPY 头上限 4 KiB，逐文件写临时文件、同步并原子替换，最多同时存在一个 NPY 临时副本；不得把全部数组先组成一个整包临时归档。JSON、请求/返回记录和两份定额日志合计上限 8 MiB，文件系统分配/目录与封口预留另计 1 MiB。含最大单个 NPY 临时副本的保守峰值为 `105,212,488 + 20×4,096 + 3,739,210×4 + 4,096 + 8,388,608 + 1,048,576 = 129,692,528 bytes < 128 MiB`。输入快照在内存中消费。逐笔写入前计入现有工件、拟写临时文件完整上界及收口预留，核验预算和实际空闲空间。

独立复核者从新鲜读盘的原始值与历史名册重推分项、七次返回的身份与形状、逐窗预测梯度的槽位、输出及参数加性门、五个投影、余量和裁决。复核同时对账来源、状态摘要、实际调用、工件摘要及发布候选。复核独立性覆盖归约和裁决，梯度值由一次生产计算图上的七次 PyTorch 求导取得。

## 6. 合成验收与授权

先在独立合成目录以 CPU 手算夹具及登记 CUDA/float32 运行时下的合成张量夹具验收顺序、生产损失归约、梯度加性、数值边界和收口。CUDA 夹具使用随机合成输入、D6 生产模型/conditioner 架构和生产损失、合成 FK；禁止读取项目真实数据/checkpoint/资产。CPU 微型解析图用于手算投影控制；CUDA 运行用于验证生产 GPU reduction 与 autograd 的浮点余量，不替代真实来源准入。

- 令 `g_0=(6,0)`，其余五窗各为 `(0,1)`。手算得 `own=6`、`other_ratio=0`、`batch_ratio=1`，相对单窗方向的投影比为 `1/6`。
- 将一个其他窗改为 `(-3,0)`，其余四窗仍为 `(0,1)`。手算得 `other_ratio=-0.5`、`batch_ratio=0.5`，达到抵消提案线。改为 `(-12,0)` 时 `batch_ratio=-1`，一阶方向反转。分别验证 `−0.25` 等值与两侧、投影区间含零、共同窗梯度处于范数门及其两侧。
- 交换两个输入槽时，保持窗的多重集，检验首次 micro 的有序载荷绑定。交换两次逐窗 VJP 的完整返回对时，六窗梯度均值相等而槽位门拒绝。另构造仅交换两份 `g_i`、保持所有 `h_i` 和整批梯度不变的夹具：取 `g_0=(1,0)`、`g_1=(0,100)`、`g_2…g_5=(0,-10)`，交换前 `other_ratio=0`，仅交换 `g_0/g_1` 后为 `−0.4`，但六窗参数梯度和不变；测试须确认参数加性门仍过、调用点绑定收据与落盘对账拒绝该变异，不能发布候选。再分别测试仅交换 `h_i`、在原始返回后/序列化前替换一份 `g_i`、交换名册内张量顺序。用六窗均值张量替代整批原标量时，入口同一性门应拒绝。另测参数名册漂移、梯度张量缺失、逐窗标量互换、分项权重写错和落盘值被改写。
- 用受控时钟覆盖首笔预留、中途 CUDA 单元返回、外层退出观察和最后同步回读。让 conditioner 返回越过工作截止，断言 backbone=0；让预测 FK 返回越线，断言教师 decode 和第二次 FK=0；让教师 decode 返回越线，断言目标 FK=0；让已进入的 VJP 返回越线，断言后续 VJP=0 且调用如实计入。验证最后回读超限和退出证据缺失均为 `incomplete`，存储超限在新写入前被拒绝。核验各错误的拒绝点、实际计数和尝试占用。
- 从批准实现闭包的精确调用矩阵生成正控；逐项额外调用、缺少调用和数量少一均须拒绝。导入前或收口时改变任一实现文件字节，均须为 `incomplete`。
- 对 `h` 比较门与参数加性门分别覆盖恰在阈值内、超出阈值一个 float64 可表示邻点及零尺度精确相等。参数 CUDA 正控须使用真实 D6 模型/conditioner 架构和生产 LossAssembler，固定上述 9 组输入，并再以第二个模型初始化 seed 重复；model/conditioner/joint 所有组的 `rho` 均须小于 `2.5e-4`（`E` 的 25%）。另设逐位正确的微型解析反例与严格阈值正/负控制，确认门确会翻转。对 CPU float64 元素重算与原 production float32 标量对账，覆盖全零、单个非零元素、次正规元素和混合尺度。若冻结容差不能稳定接受正控并拒绝负控，须在真实授权前修订方案、重跑合成验收并重新冻结；真实运行后的 fail-closed 结果不可用来调宽容差。

实现验收通过后冻结方案、输入白名单、实现、合成证明、判据、调用矩阵与额度。新的明确授权须绑定该材料并限定一次 step0 checkpoint 加载、一次整批生产前向与损失、七次 VJP、表列 FK 和 **零参数更新**。交付包括一份机读裁决与一份人读报告，记录是否值得起草后续短前缀对照。 [六窗复盘的停止规则](omniretarget_w1_d6_window_error_review_plan.md#5-如何决定下一步)

本 v0.3 仍是方案草案，不足以签发真实执行授权。合成 I/C 已完成；`attempt_006` 固定并验收的是合成路径调用、同步与工件 I/O 矩阵，不涵盖真实来源白名单读取、资产构造、外层 supervisor、真实预算/返回见证。真实执行前必须另生成并冻结这些真实路径的逐项矩阵及来源/实现身份材料，将其与合成证明共同交由新的授权绑定；缺少该材料时真实入口必须拒绝。后续任何短前缀训练是新的训练更新，必须另立方案和独立授权，不属于本诊断建议。

## 10. I/C 合成实现记录（2026-11-21，非真实执行授权）

I/C 纯合成实现已完成，验收细节见[验证记录](../verification/omniretarget_w1_d6_shared_gradient_i_c.md)。权威材料为 `derived/audit/w1_d6_shared_gradient_step0_v1/synthetic/attempt_006/`；18 个 D6 架构 CUDA 用例（9 组输入 × 两个模型初始化 seed `20261008`/`20261122`）全部通过。`h` 槽位最大误差为登记 `2e-4` 界的 `0.002612`；model/conditioner/joint 最大 `rho` 分别为 `2.24779090e-7`、`1.20235993e-7`、`2.24766225e-7`，均低于 `2.5e-4` 合成准入线。CPU 独立复核 18/18 逐例重读并重算；单例工件峰值 `121,088,256` bytes，低于 128 MiB。

每例的精确运行计数、32 个同步点、工件读写/哈希矩阵、独立复核摘要和 16 文件实现闭包见验收 JSON；实现身份前后摘要为 `f65e074e2eea825933e0b6250674e368ae8bd35afdae0f2ea02e3b175e8feda3`。`attempt_001` 的槽位门失败及 `attempt_002`–`attempt_005` 的证据缺项均原样保留；只有 `attempt_006` 是当前权威。实现测试 18 passed；额外 Trainer/D6 相关回归结果见验证记录。

## 11. 真实执行准入材料（2026-11-21；未授权）

已增加 metadata-only 材料生成器、fail-closed supervisor 入口及反例测试。最新材料按独占递增 attempt 保存在 `derived/audit/w1_d6_shared_gradient_step0_v1/real_admission/`；早期尝试均原样保留，具体权威文件、SHA 与验证状态见[准入/监督验证记录](../verification/omniretarget_w1_d6_shared_gradient_admission.md)。该交付绑定既有 D6 `fit_insufficient` 独立裁决、固定 SHA 的历史 R6 metadata admission、D6 artifact index、I/C `attempt_006` 和当前 24 文件实现/方案闭包；只打开四份 JSON 元数据及实现/方案文本。D6 六个目标输入仅从既有 artifact index 消费登记的路径/字节数/SHA；不重新哈希或打开 checkpoint、NPZ、数据集或生产资产。R6 已登记的 41 项资产哈希作为历史元数据引用，不代表本轮重新验证资产。

材料固定候选六窗、未来六份 D6 工件、root-stats 与 41 个生产资产读集合、单次执行上限 300s（240s 工作/60s 收口）、一次 attempt、更新 0，以及未来输出文件数量上限。当前 I/C 的 32 同步点与合成工件矩阵不能证明真实输入、资产构造、外层 `wait()`、进程组回收、同步回读与返回见证；这些在 admission JSON 中明确标为 `proposed_only_not_runtime_verified`。因此 admission state 是 `metadata_only_materials_drafted_worker_missing`、`executable=false`；草案 `authorized=false/template_only=true`，六项 checklist 均 null。supervisor 当前只做拒绝，不读授权、不启动进程，也不生成真实尝试根。**这不是可签发授权的材料包**：必须先实现真实 worker 与外层监督/返回证据链，再冻结完整精确矩阵并独立复核，生成绑定新实现闭包的材料后才可提交新的授权审阅。没有沿用 R6 或 D6/F 的任何授权。

**边界：**这只验证合成目标、梯度归约、槽位/名册绑定、计数和有限存储/截止守卫。真实 archive/checkpoint/资产读取、真实 FK、真实七次 VJP、真实裁决均未运行，参数更新严格为 0。`--real` 入口在任何路径检查前拒绝；本记录不构成真实执行授权。真实阶段仍须新的冻结材料、完整外层 supervisor 与 I/O/runtime 矩阵、明确一次性授权；不得复用已消费的 D6/R6 授权。
