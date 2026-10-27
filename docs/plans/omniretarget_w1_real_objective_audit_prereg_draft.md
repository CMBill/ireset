# W1 真实训练目标审计预注册草案（未注册、不可执行）

日期：2026-10-26，按静态复核意见修订。状态：**draft，not_registered，not_authorized，not_run**。父方案：[W1训练拟合与目标一致性单因素诊断执行方案](omniretarget_w1_training_fit_objective_audit_plan.md) §5。合成审计结果见[验证记录](../verification/omniretarget_w1_training_objective_audit_verification.md)。

> 2026-10-26 实施补充：审计器、CLI 与 **54 项**合成/反例测试已实现。三轮独立复核共 **13 P1 + 5 P2** 已全部修复：首轮实际 VJP 计数、落盘梯度逐元素绑定、FK 点重算、跨 run 调用额度、RNG 恢复、合法负结果进入 F、校准证据重验、负成本、行字段/额外行、非数值数组；第二轮预算撤销的 P 不启动 F 新调用、F 每次重放前实时截止+收口预留、F 封口失败同步撤回磁盘完成声明并更新封口哈希、入口先证明完整链调用余额、F 嵌套 `units`/`unit_artifacts` 容器类型异常只记 gap 不中断；第三轮撤回写盘纳入同一 attempt 成本覆盖（无法证明覆盖保持 publishing）、admission/输入绑定/前置消费者状态作为重放硬门、负结果封口失败保留 mismatch 但撤回 `complete_evidence` 并重绑哈希。合成自测 `synthetic_contract_passed`（98 objective + 14 gradient 行、0失败/0缺口、conditioner/model/production VJP/independent VJP 各 4 次即上限内、contact 0、更新/FD/DDIM/checkpoint 写入 0）；真实 provider、真实 checkpoint/G1 FK、cohort/预算与执行注册仍未运行/未创建。详见[实现与合成自测验证记录](../verification/omniretarget_w1_real_objective_audit_verification.md)。本文件仍是待授权草案，实施完成不勾选 §8。
>
> **2026-10-27 准入材料补充：**第四轮复核 2 P1+1 P2、1 项成本下限回归与 P2 授权严格化均已修复，测试增至 **59 项**；[可审阅准入包与冻结校准契约](omniretarget_w1_real_objective_audit_admission_package.md)已生成（校准契约 SHA `2904fbc4…`、证据 `49371d30…`、准入包 `attempt_004` SHA `1cedbb15…`）。§8 八项仍全部未勾选；真实执行仍为 `not_run`，未创建 cohort/预算/执行注册。

> 本文件定义待授权的范围、登记与验收契约。本轮修改仅为文档工作，不创建执行注册、cohort、预算或运行目录。完成 §8 前，真实数据审计仍为 `not_run`，`factor_selected=false`、`auto_execute=false`、最终交互 `not_ready`。文中的预算与调用数均为提案，不是已获授权或已验证可完成的运行额度。

## 1. 研究问题与可区分假设

### 1.1 问题

在**固定四个 train 窗口、两份 checkpoint、条件输入、t/noise 和生产实现**上，审计以下事实：

1. 生产编码、五项解析损失及其输出空间梯度，是否符合独立重建的冻结公式？
2. 生产路径实际消费的分母、权重和 FK 前小计是否正确；FK 标量聚合与总目标加权和是否可复算？
3. 如有偏离，是否能从绑定原始输入与生产计算图独立重现？如未发现偏离，只交付覆盖范围内的一致性结论，训练失败归因保持 `attribution_insufficient`。

### 1.2 假设与覆盖

- **H_covered**：注册 batch 上已覆盖的编码、归约、权重和输出空间梯度符合冻结公式。该结论只描述实现符合性，不解释目标设计是否适合任务。
- **H_impl**：有效绑定输入上存在可复现的编码、归约、权重或损失到预测张量的梯度路径偏离。独立确认后可登记为后续单因素候选依据；本轮不修正生产目标后继续运行。
- **其他归因**：本轮不比较数据、标签、覆盖或泛化的相对责任；无偏离时仍为 `attribution_insufficient`，不发布“更可能来自数据”或“目标实现已排除”的结论。

| 子项 | 默认覆盖 | 允许结论 |
| --- | --- | --- |
| clip 系编码、根标准化、关节映射 | 独立重编码四窗 | 已覆盖范围内一致或偏离 |
| pos/rot6d/joint/geodesic_rad2/vel | 分项标量、分母、有效权重、输出空间梯度及 FK 前小计 | 已覆盖范围内一致或偏离 |
| fk_m2 | 使用生产侧保存的预测/目标关键点复算聚合与权重 | 仅聚合契约；真实 FK 几何、解码到 FK 的正确性及导数均 `not_audited` |
| total | 五项解析损失与生产 FK 标量的加权和 | 仅总目标算术一致性 |
| contact | W1 有效权重为 0；监督分支不得执行，条件 contact 通道仍保留 | 分支关闭契约 |
| 网络参数梯度、累积、裁剪、AdamW、更新、选模 | 不执行 | `not_audited`；R3 为 `not_run` |

整体目标覆盖固定为 `objective_coverage=partial`。生产配置包含 `w_fk=2.0` 与 `grad_accum=4`；本轮不能发布全目标或完整训练消费链通过。

### 1.3 与旧诊断的区别

本轮检查真实 micro-batch 的目标算子与损失到预测张量的梯度，不重跑旧预测诊断、DDIM 采样质量或旋转融合。旧 r6 的 R 6/W 5/P 3/F 1 尝试数超限与未计费预检继续保留，不能作为本轮授权或预算依据。

## 2. Cohort、登记与预算

### 2.1 身份与固定历史范围

- cohort 提案为 `w1_real_objective_audit_v1`，以研究问题、父 SOURCE/ATTR 身份和两份 checkpoint 身份定义成员；方案哈希、实现 SHA 与环境作为版本绑定，不产生新额度。
- 历史扫描根固定为 `derived/experiments/omniretarget_first/`，运行根必须是其直接子目录。扫描该固定根下所有本 cohort 登记及尝试，不能只读当前 run；父目录、run 名、实现修订或方案哈希变化不重置历史。
- 冻结 `cohort.json` 的身份、阶段额度、调用额度与 `accounting_schema=pre_registered_v1`。旧预测和融合历史单列引用，保留其拒收原因及成本下界；不把旧余额转入本轮。
- 新 cohort 仍须取得明确执行授权。本文没有创建 cohort 文件或分配额度。

### 2.2 总预算、阶段预算与尝试上限提案

全部阶段串行，CPU/GPU 活动均计入父进程单调钟墙时；GPU 相关边界同步采样。文件哈希、加载、重建、生产求导、独立求导、复核、失败和最终登记均计费。

| 计费阶段 | 内容 | cohort 累计上限 | 每次启动保留的收口尾部 | cohort 内 attempt 上限 |
| --- | --- | ---: | ---: | ---: |
| R0 | 事前登记后的身份核验、四窗准入 | 180s | 30s | 2 |
| P | 同一进程内的 R1 生产前向与 R2 梯度取证 | 360s | 60s | 2 |
| F | 独立加载、重放、重推与收口 | 360s | 60s | 2 |
| 合计 | 全部成功、失败、崩溃、撤销、预检尝试 | **900s** | 尾部已包含在各阶段额度中 | 分阶段累计 |

阶段不互借；attempt 上限不重置调用上限。重试只允许在历史可证、剩余额度能够承担完整注册工作及收口时启动；不因负结果重试，不追加窗口或 t。预算是否足够尚未实测，不能用真实资产的未登记试跑来估时。

预算入口先读取既有历史，扣除累计计费与必要尾部，再分配新 attempt；不得把自身刚写入的 running 行作为“零历史”读取。已用额度、剩余额度、截止时间与调用余额写入事前登记。历史未知、额度耗尽或预留不足均停止，不启动实质工作。

### 2.3 事前登记、unit 准入与三方对账

1. 授权后，先核验固定根历史；分配 attempt，写明 scope、预期父资产身份、选择规则、额度、截止时间和实现快照，并写 `running/complete=false` 索引与台账。此后才允许 R0 的真实资产哈希、加载或数值检查。
2. R0 在上述登记下核验候选身份，按 §4.1 冻结完整窗口键，生成并封存 `registration.json`；随后按该注册加载四窗，写 `admission.json`。R0 失败同样保留 attempt、成本和具体原因。
3. P/F 启动前各自写 running 登记，绑定同一已封存注册。内部 unit 须持 active running 父 attempt，scope、cohort、注册哈希一致，输出严格位于父 `units/<unit_id>/`，子截止不晚于父截止；内部入口不得绕过准入。
4. 盘上 attempt、索引、台账按 `(stage, attempt)` 集合相等、每键恰一行；逐项核对状态、scope、报告哈希、注册哈希、`measured_wall_s` 与调用计数。台账 `wall_s=round(measured_wall_s,1)`；数值须有限、非负且排除 bool。索引与台账原始计费值精确相等，不能用累计汇总替代逐行对账。
5. 本 cohort 使用完整新 schema；缺行、重复、不可读、缺必需字段或未知调用成本均为 unknown。running/publishing 残留不可消费，不将缺失成本或次数补成零。
6. 采用已有经审计的 publishing/封口及尾部成本覆盖协议：发布终态内容时仍不可消费；封口后检查截止及计费是否覆盖实际耗时，计入并记录 `seal_tail_reserve_s`。撤销、崩溃与刷新自身也计费。无法证明覆盖时撤回完成声明，保留 `publishing/complete=false`，返回失败；撤回报告失败也不得发布可消费终态或 exit 0。
7. 索引与台账均原位更新，不追加同一 attempt 的第二条终态行。F 复核全部阶段历史，并在自身收口后接受相同消费者检查。

入口在分配前核验既有历史；运行中的核验只可识别本进程持有的、已事前登记且仍 active 的当前 attempt，不把它当作已完成历史或零成本。其他 running/publishing 行仍为 unknown；最终消费者要求当前 attempt 也已合法封口。

`measured_wall_s` 沿用现有保守计费语义，包含已登记的 seal 尾部；另保存实际采样耗时用于覆盖核验。崩溃、SIGKILL 或写盘失败留下未知状态时，后续入口停止，不能另起 run 洗掉该状态。

## 3. 候选固定资产与数据用途

以下为候选，不是本轮已核验事实：

- SOURCE：`derived/experiments/omniretarget_first/20261012_21e5784_w1_a3bc_reviewfix8/w1_reviewfix8/`。
- W1 checkpoint：`derived/experiments/omniretarget_first/20261013_be00c50_w1_de/` 中 deterministic 与 diffusion 的 best step4000。
- 配置：[omniretarget_motion.yaml](../../configs/train/omniretarget_motion.yaml)；同时记录 checkpoint 的原有效配置与本审计配置，逐字段说明差异。
- 实现：当前 HEAD、明确登记的 tracked diff 及 implementation snapshot；包含生产 loss/tokenizer/conditioner/model、取证适配、独立参考、消费者和计费辅助代码。环境绑定 Python、库与 CUDA runtime、设备、lock、数值后端及确定性设置。

注册需核验列举的文件闭包：父注册与清单、选中输入及缓存日程、root_stats、两份 checkpoint、必要 G1/FK 资产、配置和实现。checkpoint 的 model 与 conditioner 均 strict 加载，检查模型类型、step、可用性、有限性及 manifest/cache/root_stats 身份。任一项不可核验即停止。

| 集合/资产 | 本轮用途 |
| --- | --- |
| train24 名单与既有统计来源 | 只核验身份与选窗资格；使用已冻结的 train-only root_stats 还原 tokenizer，不重拟合全 train24 统计 |
| 选中两个 train demo 的四个完整窗口 | 唯一真实数值输入与模型输入；其他窗口不进入目标测量或求导 |
| val31/val6/known | 仅引用既有名单、划分及身份锚；不加载逐帧样本、缓存、标签或预测，不计算指标、不选模 |
| 教师、缓存、checkpoint | 只读已冻结资产；不重生成标签、缓存或检查点 |

root_stats 的来源与 train24 名单哈希必须可对账；本轮不以重新读取全训练集拟合统计来扩大四窗审计范围。

## 4. 固定审计矩阵与执行阶段

### 4.1 数据、批次、噪声和调用矩阵

- **数据键**：从冻结 train24 元数据按 demo 名规范字典序、窗口 start 升序取前两个各具至少两个完整窗口的 demo，各取前两个完整窗口，得到四个唯一键。完整窗口固定 `T=60`；顺序为 demo 后 start。R0 在读取窗口数值前写入 `(demo,start,end,T,cache_key,split,input_sha)`；只按元数据选取，不看真实误差换窗。任一键缺失或窗口不合法即保留 gap，不静默替换。
- **批次**：四窗一次组成 `[4,60,38]` micro-batch，不重排、不补齐、不丢窗。本轮是**累积前 micro 目标**，无 optimizer step，也不执行 `loss/grad_accum`、累积或裁剪。生产配置的 `grad_accum=4` 原样登记，不能将审计解释成正式 4×4 消费链验证。
- **执行单元**：P/F 的 unit 固定为一个模型与上述四窗完整 batch，unit_id 为 deterministic 或 diffusion。单窗明细是证据键，不是可缩小 batch 的执行单元；任一必需窗无效，该 unit 留缺口，不能改成三窗前向或求导。
- **扩散输入提案**：依四窗顺序固定 `t=[0,250,500,999]`，每窗只有一个 t；不是四窗×四 t 的矩阵。固定专用 CPU generator、`noise_seed=20261008`，一次生成 float32 `epsilon=[4,60,38]` 后搬至设备，无额外随机抽样。记录生成顺序、t/epsilon 数组与 SHA；独立按冻结 alpha 公式重建 `x_t`，t=0 仍为第一档加噪。这不声称重现历史训练时抽到的随机流。
- **模型模式**：model 与 conditioner 均为 train 模式、启用梯度；独立 F 使用相同模式、设备和精度。禁止借用会切为 eval/no_grad 的生成入口。固定并保存 CPU/CUDA RNG、autocast/TF32/确定性设置；专用噪声生成器与全局 RNG 分开。
- **默认关闭**：参数 FD、网络参数反向、真实 FK 导数、optimizer/scheduler step、DDIM sample、选模和 checkpoint 写入均为 0。参数 FD 不再作为本轮运行时可选分支；需要时须先修订矩阵与预算。

| 阶段/模型 | 真实 conditioner 前向 | 真实 model 前向 | 生产输出空间 batched VJP | 独立 float64 batched VJP |
| --- | ---: | ---: | ---: | ---: |
| R0 | 0 | 0 | 0 | 0 |
| P：deterministic | 1 | 1 | 1 | 1 |
| P：diffusion | 1 | 1 | 1 | 1 |
| F：deterministic 重放 | 1 | 1 | 1 | 1 |
| F：diffusion 重放 | 1 | 1 | 1 | 1 |
| cohort 累计上限 | **4** | **4** | **4** | **4** |

完整前向定义为 conditioner→model 的一次实际执行；两者仍分别包装计数。求导每次 `autograd.grad`/`backward` 调用均计数，独立参考同样计数；提案总上限为 **8 次 batched VJP**，每次固定提取五项与实际 FK 前小计共六个导数切片。网络参数反向为 0，额外切片重算不另获免费额度。

此表替换初稿笼统的“≤2 次反向”：初稿没有计入独立求导与 F。batched VJP 的算子支持、计数与取证保真须先通过纯合成验证；不能在真实阶段临时退回多次求导。所有失败调用也占用累计上限，单元数和次数从包装器实测，不从表格反填结果。

### 4.2 R0：事前登记后的身份与准入

R0 自身按 §2 先登记再工作。核验身份闭包和 train-only 统计，冻结四窗键与噪声输入，封存注册后加载四窗；核对 shape/dtype、有限性、旋转 6D 有效性、关节限位、缓存键及原始 root/q/wfc 的绑定。目标张量保持 `requires_grad=false`。

登记输出还包括有效配置、数值域与逐字段容差、必测键多重集、数组 schema、调用矩阵和截止时间。任何必需字段尚未冻结则 admission 不通过，不启动 P。R0 封口通过后 P 才可启动。

### 4.3 R1：真实生产前向、编码与标量取证

P 为同一 active attempt，R1→R2 在同一进程内执行并保留原计算图；不从 NPZ 恢复图，也不另做一次未计数前向。

生产接线固定为：冻结窗口→生产 `StateTokenizer`/数据集条件→strict 加载的 conditioner→对应 checkpoint model→原 `LossAssembler`。扩散路径先用固定 x0/t/epsilon 生成 x_t；deterministic 不注入噪声。保存模型实际输出的 `x0_pred`，并核对其与 loss 实际消费的预测张量为同一图内对象。

当前 [LossAssembler](../../src/ireset/train/losses.py) 只返回 total 张量及 detach 后的分项浮点数。因此，未来实现须提供最小取证适配：在分项转为浮点数前捕获原可微张量、生产路径进入 FK 前的实际 loss 中间张量 `L5`、原 total，以及用于每窗归约的生产中间量。观察器不得改权重、输入、返回值或计算图；实现与关闭观察器的合成对拍须逐位一致并绑定源码哈希。

独立路径从原始 root/q/wfc、冻结 lo/hi/mean/std 重建 x0，从保存且被 checkpoint 重放绑定的预测重算五项损失。保留：

- 每模型×四窗×七分项的未舍入标量，以及每模型 batch 的同七项标量；七项为 pos/rot6d/joint/geodesic_rad2/vel/fk_m2/total。
- 每项 shape、dtype、单位、归约分母、原权重、production/independent/abs/rel/within_tolerance。pos/rot6d/joint 分母分别为 F×3、F×6、F×29；vel 为 (F−1)×38；geodesic 为 F；FK 为 F×K 且对坐标先求平方和。batch 分母另乘 B。
- 生产预测/目标 FK 关键点，从这些点独立重算 fk_m2 聚合；这共享了 FK 输出，覆盖只记 `aggregation_only`。
- 实际 `L5` 与其五项加权和、total 与 `L5+w_fk·fk_m2` 对账；每窗均值与 batch 标量对账。原 contact 监督分支的执行次数必须为 0。

### 4.4 R2：原生产图与独立梯度

- 生产侧对 R1 捕获的五项张量和**实际 `L5`**，通过一次 batched VJP 取得相对于原 `x0_pred` 的六组梯度。不得把预测 detach 为新叶子后重建一份损失来充当生产图证据，也不得重组五项来替代实际 L5。
- 独立侧用绑定预测/目标及冻结公式构建 float64 输出空间参考图，一次 batched VJP 重算相同六组梯度。两侧逐字段比较，另核对生产 L5 梯度等于独立五项加权梯度和。五项梯度和不与包含 FK 的 total 梯度比较。
- 梯度数组固定为 `[6,4,60,38]`，切片顺序冻结为 pos/rot6d/joint/geodesic_rad2/vel/L5；保存 dtype、定义域状态、有限性、范数、最大差与逐字段判据。零梯度按独立期望判断，不把“非零”当作普遍通过条件。
- loss 到实际预测的图断开、权重未消费或归约错误，如在有效输入上复现，登记 mismatch。网络内部到参数的导数、真实 FK 导数、正式累积/裁剪与 optimizer 均保持 `not_audited`。
- 结果落盘后释放图；R2 失败后的新进程不能用已保存数组冒充恢复生产图，重新前向须同时满足原 cohort 的全部额度。

### 4.5 F：独立复核与收口

F 自身先登记，核验 R0/P 的封口、注册、文件闭包、调用事实和三方历史；缺前置时写明 gaps 并诚实 incomplete，不裸退或补跑 P。P 交付负结果或部分缺口时，只有输入绑定和原始证据完整的已完成单元可进入表内 F 重放，其余单元逐键留 gap；不得通过 F 补齐 P 未执行的单元。

独立复核器读取 raw 工件，自行重建窗口与分项键多重集、x0、t/epsilon/alpha/x_t、分母、权重、标量、L5、梯度期望和 summary。它重新 strict 加载两份 model/conditioner，在前向前恢复 P 对应单元的冻结 RNG 状态并按注册再生噪声，在表内各执行一次同输入前向及梯度取证，逐位绑定 P 的同代码同环境记录，并用独立公式再次核验。不能只重算 summary、比较哈希或消费 P 自报的 within_tolerance。

F 同时核对每窗→batch 关系、失败/缺口、覆盖状态、调用次数和全部预算域。报告写盘及 seal 成本也包含在 F；最终消费者拒绝未封口、历史未知或完成声明与证据矛盾的报告。仅在 §5 的通过条件及封口同时成立时 exit 0。

### 4.6 R3：真实更新不在本轮范围

本轮 `R3=not_run`，不分配更新预算。真实单步更新如需开展，须另立预注册与执行授权，冻结初始化、AdamW 状态、lr/weight_decay/betas/eps、grad_clip/grad_accum、更新次数和独立重放。不能在本轮“发现梯度问题”后临时开启，也不能用合成更新结果替代。

## 5. 数值域、容差与结论

### 5.1 先合成校准，后打开真实数值

全部阈值必须在真实窗口数值和模型输出打开前，以解析/纯合成样例确定，并保存 `tolerance_contract.json` 与合成校准证据哈希。当前没有运行新增校准，以下未闭合项仍是执行阻塞：

| 比较对象 | 需冻结的规则 | 本稿状态 |
| --- | --- | --- |
| 同代码、同环境、同输入的记录复制 | dtype/shape/逐位一致；bool 与数值分开；计数为非 bool 整数 | 契约已定义，取证实现待合成验证 |
| 独立重编码与标量 | 逐字段 dtype、atol/rtol、单位及归约；1e−9/1e−9（float64）、2e−5/2e−5（float32 对 float64）仅保留为初稿候选值 | 待合成校准，不可直接当已验证阈值 |
| 输出空间梯度 | 每项 atol/rtol、近零尺度下界及范数/方向诊断的适用条件 | 待合成校准；不得直接沿用标量绝对容差 |
| 旋转与关节映射适用域 | 6D 两列范数与独立性门、近 π 的角距门、clamp 边界及已知数值负门的处理 | 待合成冻结精确阈值 |

校验顺序为结构/类型→非空与形状→定义域→双侧有限→容差。浮点比较按冻结的 `abs(a-b) ≤ atol+rtol·abs(reference)`；参考接近零时按预注册尺度规则判断，不能用大绝对容差吞掉小量梯度。bool 不能作为 0/1，计数不得接受小数；独立 float64 与生产 float32 的预期差异是显式转换规则，不是结构失败。

独立旋转公式不复用被审计的 SO(3) log/6D 转换实现。既有近 π、平行/零 6D 等限制逐项保留：落在事前排除的独立参考域外时，保留原键并记 gap；域内可复现偏离记 mismatch。不得看完真实数据后扩大排除区、换窗或放宽阈值。

### 5.2 分类、优先级与完整性

| 观察 | 结论/字段 |
| --- | --- |
| 注册必测项全覆盖，独立核验及封口通过，无失败/缺口/协议偏离 | `real_batch_covered_contract_passed`；`covered_contract_passed=true` |
| 有效绑定输入上独立复现编码、分母、权重、结构或梯度路径偏离 | `real_batch_objective_mismatch`，保留入口与原始证据 |
| 缺失/不可读/坏结构导致不能比较、参考域外、身份不明、未知成本或超额度 | `audit_incomplete`，保留 evidence_gaps/协议原因 |
| 真正未纳入的 FK 几何/导数、参数梯度、累积/裁剪/更新 | 对应覆盖项 `not_audited`，不补成通过 |

`real_batch_objective_contract_passed` 不再作为本轮允许的整体结论。所有结果固定 `objective_coverage=partial`、`full_objective_contract_passed=null`、`attribution=attribution_insufficient`。

失败与缺口分列。若已独立确认有效 mismatch，后续出现缺口仍保留 mismatch 结论，同时置 `complete_evidence=false`、`covered_contract_passed=false`；缺口不能覆盖已证实的偏离。生产与独立路径的分母/权重不同本身可为 mismatch；无法证明输入或字段有效才为 incomplete。

`complete_evidence=true` 要求所有**已注册覆盖项**、调用计数、阶段历史及独立重放完整；它不改变未审计项。缺 R0/P、预算撤销、超调用或 running/publishing 均阻止通过。mismatch/incomplete 返回非零，未知封口不能 exit 0。

## 6. 停止与变更规则

以下事件停止新增生产求值，并按 §5 分类和封口：资产或注册漂移；缺窗/重窗/键不符；退化或非有限输入；有效输入上的生产偏离；截止或调用额度不足；未登记的真实工作；消费者历史未知。已产生的负证据只有在输入有效、身份可证、预算和调用余额仍满足原矩阵时，才可由 F 独立复核；未知历史、无效输入或额度不足时只整理已有证据与诚实收口，不再调用模型或求导。

不在运行中调整窗口、t/noise、权重、数值域、容差、调用定义或结论词汇。需要修订时保留旧记录，按同 cohort 累计已有消耗并重新审阅；方案/代码变化不自动提供新预算。发现目标偏离后仅交付证据与影响范围，另立单因素方案才能干预；本轮不自动续训、改教师、扩数据或接入生产。

## 7. 原始产物、独立验收与合成反例

### 7.1 强制证据契约

| 产物 | 必需内容 |
| --- | --- |
| cohort/登记/台账 | `cohort.json`、事前 `attempt_registration.json`、阶段索引、`costs.json`、注册/实现/环境哈希、额度、截止、状态、调用实测、尾部及撤销事实 |
| `registration.json` / `admission.json` | 身份闭包、有效配置、窗口及 t 顺序、噪声规则、数值域、容差、覆盖表、必测键及准入原因 |
| `window_keys.json` | 四个注册键及其输入/缓存 SHA、split、T、start/end；按多重集核验，不能只比数量 |
| `objective_real_arrays.npz` | 原始 root/q/wfc、lo/hi/mean/std、全部实际条件张量、x0/prediction、实际 t/epsilon/alpha/x_t、生产 FK 两侧关键点及归约中间量 |
| `objective_real_rows.json` | 每模型×每窗×七项及 batch×七项、L5、原始未舍入值、分母、权重与比较事实 |
| `gradient_real_arrays.npz` / `gradient_real_rows.json` | 两侧六组输出梯度、精确形状/dtype、图取证绑定、范数/差值与逐字段判定 |
| `call_trace.json` / `rng_state_facts.json` | 实际 conditioner/model/求导调用、模型模式、输入输出身份、开始/结束及异常、CPU/CUDA 与专用 generator 状态 |
| `tolerance_contract.json` / `independent_verification.json` | 合成校准绑定、独立重推与重放事实、必测键覆盖、失败/缺口和历史核验 |
| `audit_summary.json` | 从 raw 重算的 counts、覆盖范围、完整性、分类、失败/缺口；R3/not_ready 等边界 |

实现前冻结上述数组的**精确键、shape、dtype、单位、坐标系及四元数排列**，实际条件至少覆盖 kps_obj/kps_clip/box_t_clip/box_quat/contact/wfc/valid；缺必需 schema 不得准入。多重集须覆盖两模型×四窗的七项标量、两模型的 batch 标量/L5 与六切片梯度，不得以空集合或删字段维持通过。

所有产物写入独立 attempt，除同 attempt 的受控发布/撤销外不覆盖旧证据。工件缺失、坏容器或必需值非有限须正常落盘失败报告；不能异常中断后留下可消费的完成声明。生产与复核共享原始资产和冻结规范，不共享顶层通过判定、实测派生值或独立梯度参考函数。

### 7.2 实现后、真实执行前的纯合成准入

在 tmp 沙箱使用解析 fixture、随机初始化微型模型或替身，隔离 PROJECT_ROOT；不得读真实 dataset/cache/checkpoint 或生产历史。必须覆盖完整正向链及以下反例：

- 缺/重/换窗口、缺分项/梯度、空数组/空 summary、坏结构；均不能空真通过。
- 修改 root、目标、条件、预测、model/conditioner 身份或 noise/t/x_t，即使同步改写 summary 也不能通过绑定。
- detach 原生产预测路径、改分母/权重、以重新拼装的小计冒充实际 L5；必须得到对应图/消费路径失败。
- 梯度清零、整体缩放、小量误差被吞、bool/数值混用、NaN/Inf、计数小数，以及近 π/退化域的边界处理。
- 删 running 登记、换 run 重置次数、三方成本不等、超预算/调用、封口失败、撤回失败与不可消费状态的传播。

反例预期按冻结语义从 raw 变异重推，不读取工件自报原因。合成验证通过后再冻结实现和阈值；不预报测试数量，也不把已有 33 项合成目标测试冒充本轮取证与计费覆盖。

## 8. 执行授权与准入清单

以下全部满足才允许真实执行；文档修改不勾选这些事项：

> 2026-10-27 状态：可审阅的[准入包与冻结校准契约](omniretarget_w1_real_objective_audit_admission_package.md)已生成；校准契约 SHA `2904fbc4…`、证据 `49371d30…`，准入包 `attempt_004` SHA `1cedbb15…`（P2 授权严格化后），无授权/模板探测均被拒绝且未创建运行目录。下列 8 项仍未勾选，真实执行仍为 `not_run`。

- [ ] 用户明确授权本问题的真实资产加载、四窗前向与输出空间求导；更新、参数 FD 和 DDIM 仍为 0。
- [ ] 用户确认新 cohort 与固定历史范围、旧拒收记录保留、阶段不互借与历史不重置。
- [ ] 用户确认 R0/P/F 的 180/360/360s 提案、各阶段两次累计尝试、收口预留，以及含 F/独立参考在内的 4 次完整前向和 8 次求导上限。
- [ ] 真实工作前的登记/封口/消费者及父 unit 准入已实现并通过沙箱反例。
- [ ] 生产取证适配与独立参考完成合成对拍；batched VJP 支持、调用计数及图保真已验证。
- [ ] 数值域、梯度及标量逐字段容差完成纯合成校准并冻结，当前 pending 项清零。
- [ ] 执行时先在 R0 登记下核验身份和冻结四窗完整键、t/noise、配置、实现快照、环境及数组 schema；R0 封口通过后才启动 P。
- [ ] 独立复核器按 raw 键集、输入、梯度、调用与全阶段历史验收；整体覆盖仍为 partial，最终交互仍 not_ready。

当前上述真实执行事项均未完成确认：**草案修订完成不代表真实预注册或执行验收；真实数据审计、训练及单因素实验继续 `not_run`。**
