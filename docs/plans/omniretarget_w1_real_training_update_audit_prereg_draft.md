# W1 真实网络参数梯度与优化更新链路审计预注册方案

日期：2026-10-29。版本：v0.8。状态：**`ic_implemented_synthetic_complete_reviewfixed / not_registered / not_authorized / not_run`**。

本次交付只起草/修订方案。静态核对基于 HEAD `f7f5c94b291b27127ad8fdc7e4b783c91f97ada7` 的源码、配置及父报告；没有读取真实 checkpoint 张量、NPZ 数值或数据集，没有执行网络、FK、反向或参数更新。下文的预算、调用上限和数值门均为待审阅提案，尚未分配或校准。

v0.2 已按首轮审阅的 1 P1+2 P2 修订有效 float32 求值、更新量可辨性和逐元素状态零值门；v0.3 按第二轮 1 P2 补齐跨路径相消的误差传播和分类，见 §4.3、§6.4 和 §11。同输入公式错误仍为 failure；允许的上游数值差异被相消放大、无法证明跨路径一致时为必测 gap，不能通过。

v0.4 记录 I/C 实施与纯合成验收（仅合成数据/随机初始化/解析 FK，未读真实 checkpoint、NPZ、资产或数据集）：共享 update unit、审计器/CLI/测试、有效 float32 求值契约、容差校准、更新量/局部清零/20% 偏差/1 ULP 相消反例、F 新进程重放与准入包均已完成。合成全链为 **`audit_incomplete`（`complete_evidence=true`、`covered_contract_passed=false`、`failures=[]`、`problems=[]`）**，原因是必测的 `parameter_delta_precision_gap`、`cross_path_sensitivity_gap` 和 2 个 `cross_path_bound_unverifiable`，不是实现错误。校准同时证明了：(i) 当前 CUDA/float32 默认 **foreach** dispatch 与 §4.3.1 求值图逐位一致，CUDA 单 tensor/`foreach=False` 路径不同，契约绑定默认 foreach；(ii) 独立 `x_t` 必须使用生产 **binary32** `alpha_cumprod` 提升到 float64 的有效常数；(iii) 数学零梯度在 float32 autograd 中留下约 1e-9 的相消噪声，1e-30 的极小门不可行，改为经校准的 1e-6 绝对零噪声门。真实 provider、注册、授权和 R0→P→I→F 真实 stage 仍未实施。

v0.5 记录第三轮审阅的 **8 P1+4 P2** 修复与重跑（仍仅合成）：零噪声门只适用于深相消带且拒绝非零擦除；m/v/theta 跨路径保留原始相对门并显式记录零/符号/单侧更新分歧；F 对账保存报告、RNG 与两侧行多重集；worker 仅在完整快照核验合并后结算，并继承父截止/逐单元准入；否决在 terminal 前撤回磁盘声明；撤回纳入 publishing/成本下限协议；完整数值负结果可进入 F 并保留失败；区间逐节点向外 R32 舍入且输入预算覆盖梯度门允许集合；clip 边界与原生 `min(1,clip/(norm+1e-6))` 一致；校准状态比值按 rtol 归一化并取全链+标量最大；工件上界逐类计算。重跑校准 contract `9a964372…`、evidence `85246d75…`、fp32 canonical `6d7a24d0…`；full 合成全链 **`audit_incomplete`（complete_evidence=true、covered=false、failures/problems 空、I/F 各 611 条必测 gap：392 `parameter_delta_precision_gap` + 219 `cross_path_sensitivity_gap` + 0 `cross_path_bound_unverifiable`）**，`training_updates=16`，调用矩阵精确，F 新进程逐位重放；工件 6.895 GiB，单链上界 7.145 GiB、cohort 上界 14.236 GiB≤16 GiB；新模块 **39 passed**，相关回归 **263 passed**（全项目套件未重跑）。真实 provider/授权/注册/真实 R0→P→I→F 仍未实施。

v0.8 记录第六轮审阅的 **2 P1** 修复与重跑（仍仅合成）：①**跨路径更新量改用未舍入的 I 参考**。原 `_cross_path_flags` 与 `max_delta_ratio` 传入 `delta_i_measured`，比较的是两条已舍入的实测增量；§6.2 要求 `abs(delta_measured_P-delta_ref64_I)≤rtol*abs(delta_ref64_I)`。现 close 门用 P 实测 vs I 的未舍入 `delta_ref64`，零/非零存在性另比较 P/I 两个参考；评审二步例（theta `4.3598099728114903e-4`、m `1e-3`、v `1e-7`、gP `-8.899898268282413e-3`、gI `-8.9`）用已舍入 I 实测得 0.994504（错误通过），用未舍入参考得 **1.001830**，现记 `cross_path_sensitivity_gap`。校准 `delta_cross_path` 随之重算：v0.7 曾据已舍入实测把门调至 `1e-3`；用未舍入参考后，原始提案 `5e-4` 下实测 **0.216102**（4×=**0.8644≤1**），因此 v0.8 **恢复 5e-4**（更严格），不再保留 1e-3。`small_unobservable_update` 仍为 `audit_incomplete`（两侧参考同为零/非零，存在性不因实测为零而误报）。②**参考 FK 前的截止单元**。v0.7 的 `i_output_vjp_{u}_{m}` 检查位于 `ReferenceContext.evaluate` 之后，受控时钟在首个 forward 返回后过期时仍会执行 480 次 `reference_mj_forward` 与 3600 次 `reference_mj_jac_site`；现 `run_i_stage` 在 evaluate 前检查 `i_ref_fk_{u}_{m}`，`ReferenceContext` 在目标窗构造、evaluate、FK 与 Jacobian 调用前分别执行同一 `check`（含截止与 `unit_admission`），过期后不再启动新的 FK/Jacobian 单元。重跑校准 contract `271df4e3…`、evidence `aa4bd831…`、fp32 canonical `6d7a24d0…`；full 合成全链 **`audit_incomplete`（complete_evidence=true、covered=false、failures/problems 空、I/F 各 582 条必测 gap：392 `parameter_delta_precision_gap` + 190 `cross_path_sensitivity_gap` + 0 `cross_path_bound_unverifiable`）**，`training_updates=16`，调用矩阵精确，F 新进程逐位重放；工件 6.895 GiB、单链上界 7.145 GiB、cohort 上界 14.236 GiB≤16 GiB；新模块 **54 passed**，相关回归 **263 passed**（全项目套件未重跑）。真实 provider/授权/注册/真实 R0→P→I→F 仍未实施。

v0.7 记录第五轮审阅的 **3 P1** 修复与重跑（仍仅合成）：①校准统计改为**逐门适用性**——某行的 `classification` 为 `audit_incomplete` 仅因更新量不可辨时，其已通过的 m/v 比较仍是有效观测；同 seed 校准链中 `diffusion:u1:cross_attn.v.weight` 仅有 `parameter_delta_precision_gap`，旧 `rtol=1e-3` 下状态归一化误差 **0.289626**（4×=1.1585>1，旧 0.156936 只统计 `passed` 行），按逐门适用性重定 `state_cross_path=2e-3` 后 **0.144813**（4×=0.579）；同一修正也把可辨跨路径更新量在旧 `5e-4` 下的 **0.297752**（4×=1.191>1）暴露出来，重定 `delta_cross_path=1e-3` 后 **0.148876**（4×=0.5955）；`delta_same_input` 0.231385（4×=0.9255）、`theta_cross_path` 0.075190（4×=0.3008）。②cotangent 分组零噪声门落实**两侧范数均在 `ZERO_GRADIENT_NOISE_SCALE=1e-10` 深带**的限制：参考 joint 1e-13、生产 joint 1e-8（范数 8.34e-7）时整张量与旧分组门会放过，现由 `tiny_group_joint` 拒绝；两侧均在深带的合法噪声仍通过。③`run_i_stage` 在 FK 后的输出 VJP 前、独立 clip 前和参考更新前分别检查截止；`explicit_reference_updates`/`diagnostic_parameter_update_events`/`optimizer_formula_checks` 改为在 AdamW 计算实际执行后才计数。重跑校准 contract `3bce1fd5…`、evidence `8c8d983c…`、fp32 canonical `6d7a24d0…`；full 合成全链 **`audit_incomplete`（complete_evidence=true、covered=false、failures/problems 空、I/F 各 577 条必测 gap：392 `parameter_delta_precision_gap` + 185 `cross_path_sensitivity_gap` + 0 `cross_path_bound_unverifiable`）**，`training_updates=16`，调用矩阵精确，F 新进程逐位重放；工件 6.895 GiB、单链上界 7.145 GiB、cohort 上界 14.236 GiB≤16 GiB；新模块 **51 passed**，相关回归 **263 passed**（全项目套件未重跑）。真实 provider/授权/注册/真实 R0→P→I→F 仍未实施。

v0.6 记录第四轮审阅的 **4 P1+1 P2** 修复与重跑（仍仅合成）：①校准状态比值覆盖 `abs(b)>0` 的**全部**坐标并使用门的实际分母 `atol+rtol*abs(b)`（旧 `ZERO_STATE_ABS=1e-6` 掩码隐藏了 1e-17 量级的 m/v，同 seed 校准链实测最大归一化误差 **0.313871**，4× 为 1.255>1）；据此把 `state_cross_path` 提案由 `5e-4` 调到 **`1e-3`**（原始逐元素相对门与零/符号语义不变，仍拒绝 50% 状态差和 0 vs 1e-9），重算最大 **0.156936**、4× **0.628≤1**。②pre-seal 矩阵否决的撤回封口保存执行后的 `counter.snapshot()` 而非 running 行的 pre-F 余额，`prior_call_counts` 不再少报 8 次更新/32 次 backbone 前向；撤回时用 before/final/consumed 三方对账。③`run_p_stage`/`run_i_stage` 的本地截止检查无条件执行（不再依赖可选 guard），主链在 backward/clip/step/VJP 前调用；主 P/I 也传入 `unit_admission` 守卫，撤销/非 running 父登记即停止，I 在 `ReferenceContext` 的 240 次参考 FK 之前也先检查。④输出 cotangent 增加 **p/rot6d/q 三组** independent relative-L2/cosine 门并计入行裁决与行哈希，整张量门单独通过不再掩盖整组清零。⑤F 阶段 `deadline` 已扣一次 `reserve_s`，父 pre-spawn 检查、子进程 timeout 与 worker/`unit_admission` 均不再二次扣除；剩余 300s/预留 180s 时仍有 120s 合法工作窗口。重跑校准 contract `582ea60a…`、evidence `56ae9663…`、fp32 canonical `6d7a24d0…`；full 合成全链 **`audit_incomplete`（complete_evidence=true、covered=false、failures/problems 空、I/F 各 600 条必测 gap：392 `parameter_delta_precision_gap` + 208 `cross_path_sensitivity_gap` + 0 `cross_path_bound_unverifiable`）**，`training_updates=16`，调用矩阵精确，F 新进程逐位重放；工件 6.895 GiB、单链上界 7.145 GiB、cohort 上界 14.236 GiB≤16 GiB；新模块 **46 passed**，相关回归 **263 passed**（全项目套件未重跑）。真实 provider/授权/注册/真实 R0→P→I→F 仍未实施。

## 0. 唯一问题与覆盖边界

[真实目标审计](../verification/omniretarget_w1_real_objective_audit_report.md)已在四窗上核验目标标量和五项解析损失的输出梯度；[真实 FK 审计](../verification/omniretarget_w1_real_fk_audit_report.md)随后核验保存预测的训练解码、FK 几何与 FK 输出梯度。两轮均在各自覆盖范围内通过，尚未验证真实网络参数反向和优化更新。

本方案只回答：**在同一冻结权重、输入、参数前态和优化器前态上，生产的参数梯度→四 micro 累积→全局裁剪→AdamW 更新，是否与独立目标输出梯度、参数 VJP 和显式更新公式一致？**

| 层次 | 本方案覆盖 | 仍未覆盖 |
| --- | --- | --- |
| 参数反向 | 完整 conditioner 与两种 backbone 的 named-parameter roster、总目标参数梯度、None/零/非零语义 | 第二套网络求导实现、参数 FD、逐损失项的参数归因 |
| 更新单元 | micro_batch=4、grad_accum=4、model+conditioner 联合 L2 clip、两个连续 AdamW 更新 | 全数据 loader/epoch 调度、S0 与正式采样等价性、历史续训 |
| 复核 | 相同前态下的独立参考、生产两步连续性、新进程从盘重放 | 选模、DDIM、端到端质量、干预收益和主因比例 |

现有 [Trainer.save](../../src/ireset/train/trainer.py) 没有保存 AdamW 状态、RNG 延续或 loader 游标。本轮只把 step-4000 checkpoint 当作初始权重，每个模型新建 step-0 的优化器状态；诊断更新编号为 1、2，**不称为历史 step-4001/4002 续训**。

一次完整正式链计划执行 **16 次诊断参数更新**，包括参考路径和 F 重放；真实数据更新不能登记为 0。所有副本和状态只作审计证据，不能作为 best/last 或生成入口。即使通过仍为 `objective_coverage=partial`、`full_objective_contract_passed=null`、`attribution=attribution_insufficient`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false`。

## 1. 来源、固定输入和 cohort

### 1.1 不可变父锚

目标父根：`derived/experiments/omniretarget_first/20261027_b45cd7c_w1_real_objective_audit/`。下表路径相对此根。

| 文件 | SHA-256 |
| --- | --- |
| `registration/attempt_001/registration.json` | `ecc11368219a33c95a6038769398fd11789dcdd04430760f54ae3a84ba58a66a` |
| `registration/attempt_001/admission.json` | `61c5b16d883d0fa084632a548bd7428e2fe580540990feed47926546bd32ce95` |
| `production/attempt_001/p_report.json` | `1ea15ddc2da7a0acb5125f1ecc92e1ea7e0cc74589e7a7ad5055b26055b7cdcb` |
| `production/attempt_001/objective_real_arrays.npz` | `4cf20acb13fcdee207e0492a3b07818cc9ed31202409177fa1ed3f13086fd2c8` |
| `production/attempt_001/gradient_real_arrays.npz` | `49b21c6c5d82a19e92f2c0d870b1d7f036986604c009c6195e6f86a035765ea7` |
| `finalize/attempt_001/audit_summary.json` | `0a839071125a156273b3d579a564965cd0edb196b8f900b05417edd0cb00db97` |
| `finalize/attempt_001/independent_verification.json` | `f0bf351bd0bd184f52198ac481293bae9127607ec98864010768430a4dcbd961` |

FK 父根：`derived/experiments/omniretarget_first/20261028_2083249_w1_real_fk_audit/`。

| 文件 | SHA-256 |
| --- | --- |
| `registration/attempt_001/registration.json` | `93a7096fd58139d52351763befc673f9be49c42882fc898ac063f9582a33d6bb` |
| `finalize/attempt_001/fk_audit_final.json` | `c1b8e0e7136b053dbed6d9710993fae3112896d8560571237aefb4c3a35ab674` |
| `finalize/attempt_001/independent_verification.json` | `9432985de7cb6f3dc19ab6b19876d897846bd9f7373df9fc45bed456530bb647` |

R0 同时核验父生产者的盘×索引×台账和消费者状态，不只信任 final 的 passed 字段。父数组、登记、校准、冻结方案和 checkpoint 均保持只读；父保存的旧 prediction/Jacobian 不作为新权重前向的参考值。

### 1.2 checkpoint 与四窗

两 checkpoint 路径相对 `derived/experiments/omniretarget_first/20261013_be00c50_w1_de/`：

| 模型 | checkpoint | SHA-256 |
| --- | --- | --- |
| deterministic | `formal/deterministic/attempt_001/best.pt` | `7b1efd57ceff6510d95c5fd682649c7900297030923904c3a7542f59af7b2eff` |
| diffusion | `formal/diffusion/attempt_001/best.pt` | `1f9c16b124ab2d77877e93cfa39af124aa30a4f5aeb998f05350de26dcd437b1` |

两份登记 step 均为 4000。R0 只核对字节哈希和父元数据；P/I/F 才实际 strict load，核对 payload 的 step/config/root_stats/identity/finite 标记、state_dict 键集及参数值。缺状态不能降为 non-strict load。

| 窗键 | demo | 原长 T | slice（end 不含） |
| --- | --- | ---: | --- |
| W0 | `sub10_largebox_063` | 133 | `[0,60)` |
| W1 | `sub10_largebox_063` | 133 | `[30,90)` |
| W2 | `sub10_largebox_085` | 148 | `[0,60)` |
| W3 | `sub10_largebox_085` | 148 | `[30,90)` |

仅使用父保存的这四个 train 窗，均属 sub10；保留重叠窗口键，不补尾窗，不增加受试者，不声明泛化覆盖。R0 的必需源 schema 为：

| 输入 | 必需 dtype/shape |
| --- | --- |
| `x0` | float32 `[4,60,38]` |
| `kps_obj` / `kps_clip` | float32，各 `[4,60,52,3]` |
| `box_t_clip` / `box_quat` / `contact` | float32，分别 `[4,60,3]` / `[4,60,4]` / `[4,60,4]` |
| `raw_root` / `raw_q` / `raw_wfc` | float64，分别 `[4,60,7]` / `[4,60,29]` / `[4,4,4]` |
| `root_mean/root_std` / `joint_lo/joint_hi` | float64，分别各 `[3]` / 各 `[29]` |
| `alpha_cumprod` | float64 `[1000]`，与配置独立重推和生产有效调度对账 |

batch 的 `wfc` 显式由 `raw_wfc` 按父协议取 float32。编码及损失使用真实构造器生效的 float32 常数，独立参考再将这些常数提升到 float64；另核对原始来源常数。所有必需数组先检查 dtype、精确 shape、非空、有限性与物理/旋转域，再参与比较。

### 1.3 新身份与资源来源

拟定 cohort：`w1_real_training_update_audit_v1`。其身份由参数更新问题、上述父注册/两 checkpoint、四窗、两步矩阵和参考定义共同确定；改 run 名、实现/方案 SHA 或阈值文件不重置身份。

固定历史根为 `derived/experiments/omniretarget_first/`，run 必须是直接子目录。旧预测诊断不可验收、旧融合 S 预算不足及父目标/FK 的既有额度分别保留；本方案不借用、不续写这些 cohort。

## 2. 生产链、参数 roster 与最小工程改动

### 2.1 生效配置与运行时

绑定 [omniretarget_motion.yaml](../../configs/train/omniretarget_motion.yaml)，文件 SHA `0e2cfa96849ee2de14ac84b7343289090c606f78116bb79114ff81b046b521e0`。本轮继承其模型/损失/更新算子，`steps=4000` 仅为父来源；诊断步数另固定为 2，不改原配置文件。

| 项 | 冻结值 |
| --- | --- |
| network | d_model 240，4 layers，6 heads，FFN×4，time_window 31 |
| batch/update | micro_batch 4，grad_accum 4，lr `1e-4`，weight_decay `1e-5`，grad_clip `1.0` |
| loss | pos/rot6d/joint 各 1，geodesic `.5`，FK 2，contact 0，vel `.5` |
| diffusion | 1000，scaled_linear beta `1e-4→0.02`，x0 prediction，clip_sample false |
| AdamW defaults | betas `(0.9,0.999)`，eps `1e-8`，amsgrad/maximize/capturable/differentiable false，foreach/fused None，decoupled weight decay true |
| runtime | CUDA/float32，TF32 off，matmul `highest`，autocast off，deterministic_algorithms false，Torch default dtype float32；CPU 数值线程 1，真实单元顺序执行 |

记录实际 PyTorch optimizer param_groups、默认 dispatch 与库版本/源码身份；不强制改 foreach/fused 或 attention 后端以求一致。conditioner 的接触日程仍输入；`w_contact=0` 只禁止手部监督/FK/SDF 分支，不删除 `contact_proj`。

### 2.2 共用 update unit

当前 [Trainer.run](../../src/ireset/train/trainer.py) 在最后一步总会验证并保存 checkpoint。审计不能通过把 `steps=2` 直接调用完整 run、禁用 save/validate 后又宣称原生 run 全覆盖。

I/C 工程阶段先抽出最小更新单元，由原生 run 和审计共同调用：`zero_grad(set_to_none=True)`→四次既有 `_forward`→`(loss/4).backward()`→有限性检查→联合 global clip→原生 AdamW.step。loader 取样、历史日志、验证和保存仍由原生 run 管理；审计只向该单元提供固定 batch 序列。

新增可选只读 observer，捕获 loss、prediction 输入梯度、叶参数的当前 micro 梯度贡献、累计梯度和更新前后状态。叶 hook 不返回替代梯度、不改 `.grad`，每个 micro 核对实际触发次数。`g_unscaled=4*g_scaled` 只可标为派生量，不伪称另做一次 backward；也不能用两个累计 `.grad` 相减冒充未舍入的当前 micro 梯度。

`observer=None` 与重构前的原生成功/异常路径必须先通过纯合成基线对拍，涵盖 `_forward` 的默认全局 RNG 分支和已有 CPU noise_generator 分支。不能复制一份审计循环代替真正共享生产单元，不能新增默认模式开关改变既有数值。

### 2.3 全参数键集与 None 语义

C 在不加载真实权重的同配置构造器/meta inventory 上冻结 `expected_parameter_roster.json`：顺序为 model 后 conditioner，各自沿原生 `named_parameters()`；每项含名字、dtype、精确 shape、numel、requires_grad、optimizer group/index 和共享存储关系。P/I/F 的实际实例及 strict-load 权重必须匹配，重复对象不得重复入 optimizer。

令两模型的 trainable tensor 数为 `N_det,N_diff`，C 冻结其具体值和 roster SHA 后才能生成准入包；本草案不猜测行数。空 roster、漏 conditioner、错形状、错顺序或重复键均拒绝。

`None` 与数值零不能互换。原生 AdamW 对 None 参数不更新、不衰减、不初始化状态；零梯度 tensor 仍参与 step 和衰减。当前 deterministic 的 `root_in.weight/joint_in.weight` 因零输入可合法为零梯度，不要求每个参数非零。允许 None 的精确静态键集须由 C 的图契约冻结，不能按真实缺失结果事后扩大。每个模型的两步矩阵须实际激活 model 和 conditioner 两组至少一个非零梯度，否则该组记 `gradient_contract_unexercised` 缺口。

## 3. 固定 micro、噪声与 RNG 协议

### 3.1 有序 micro 矩阵

每个模型执行两个连续更新，每步四个 B=4、F=60 micro。数字为窗口 ID；重复有意登记，使确定性模型不退化为四次相同窗口多重集。每步各窗恰消费四次。

| update | micro 0 | micro 1 | micro 2 | micro 3 |
| ---: | --- | --- | --- | --- |
| 1 | `[W0,W0,W1,W2]` | `[W1,W2,W2,W3]` | `[W0,W1,W3,W3]` | `[W0,W1,W2,W3]` |
| 2 | `[W0,W0,W2,W3]` | `[W1,W1,W2,W3]` | `[W0,W1,W2,W2]` | `[W0,W1,W3,W3]` |

单路径为 2 模型×2 步×4 micro=16 micro、64 样本槽；四条执行路径（P、I、F-P、F-I）合计 64 micro、256 样本槽，只有四个不同源窗。F 重放不增加独特样本覆盖。

### 3.2 diffusion 输入

噪声协议名 `w1_real_training_update_noise_v1`，seed `20261008`。新建专用 CPU `torch.Generator`，依次对 update 1/2、micro 0–3 执行：

```python
g = torch.Generator(device="cpu").manual_seed(20261008)
# 在此 generator 上按 update→micro 顺序执行下面两次 draw，不在每个 micro 重置。
t = torch.randint(0, 1000, (4,), generator=g, device="cpu", dtype=torch.int64)
epsilon = torch.randn((4, 60, 38), generator=g, device="cpu", dtype=torch.float32)
```

每个路径生成 8 对，槽位对应 §3.1 的 batch 顺序；同一重复窗口的不同槽不共用 epsilon。R0 在登记后先生成全部值，保存 RNG 前后状态、t/epsilon 和独立 float64 `x_t` 参考；锁定 Torch 版本和数组哈希。t 由协议确定，不沿用父 probe 的 `[0,250,500,999]`，不宣称当前随机矩阵覆盖极端时刻。

P 使用既有 `_forward(batch, noise_generator=g)` 的 CPU generator 路径，真实 draw 与 R0 的 t/epsilon 逐位对账；实际 `model.add_noise` 和输入 `x_t` 必须捕获。I 同样重新从该 seed 生成并核对后使用原生 add_noise，保持网络输入逐位相同。独立 alpha/x_t 公式只用于校验加噪语义，不用提升精度后舍入的结果替换生产输入；v0.4 的实现使用生产 **binary32** `alpha_cumprod` 提升到 float64 的有效常数做独立 `x_t`，原始 float64 schedule 仅用于独立对账，校准观测比值 0.2168（2e-6 提案的 4× 为 0.867）。

F 单独重建 R0 schedule 一次，再为 F-P、F-I 分别重建 draw 流。全链合计 48 个 draw unit（每 unit 一次 randint 和一次 randn），见 §7；生成不得偷偷使用额外全局噪声或预热 forward。

每个模型/微批前后捕获 Python、NumPy、Torch CPU/CUDA 的可恢复 RNG、专用 generator 与 module mode；F 恢复对应状态并核验。当前源码未见 Dropout，这不免除模式/RNG 对账。原生 loader/global-CUDA RNG 的历史消费不在真实本轮覆盖内。

## 4. 参数梯度与独立更新参考

### 4.1 P：生产两步连续执行

P 按 deterministic→diffusion 顺序，为每模型 strict load 一次、创建空 AdamW；state 在首次 step 前应无已初始化 m/v，逻辑 step 为 0。两步前向都实际经过 conditioner、backbone、输出头/sigmoid 和 LossAssembler。

每个 micro 只执行一次生产 backward。保存 live 的六项 loss、L5/total、prediction 的 `d(loss/4)/d prediction`、叶 hook 的 scaled 参数贡献、每个 micro 后 `.grad`；四次后保存 pre-clip/global norm/post-clip、参数和 optimizer pre/post-state。所有值无日志舍入。

step 2 必须逐位接续 step 1 的参数、m/v/step，不重置优化器、不从 checkpoint 重启。任何非有限值、无效旋转域或部分更新异常都保留最早位置和参数不确定性，不修复后继续。

### 4.2 I：共同前态下的独立目标和参数 VJP

I 为每模型另 strict load 同一 checkpoint。step 1 使用原始权重和空状态；step 2 显式恢复 P 保存的 step-2 pre-state。两步输入都独立绑定来源；P 的前态必须与 P 上一步后态逐位连续，且 step-1 输出先经过参考公式对账。

这是逐步**共同前态**的算子对拍，不是两条自由演化训练轨迹。I 的 step-1 后态不用于自己的 step 2；恢复操作单列 `anchor_state_restores`，不冒称它是一次梯度更新。数值负结果下仍可定位 step 2 的局部算子，但不能用复制 P 前态掩盖 step-1 的失败。

I 每个 micro 重新执行真实 conditioner/backbone，并要求 memory/prediction 及 diffusion 输入与 P 同前态逐位相等。然后在 detached float64 prediction 上独立计算：

- pos、rot6d、joint、geodesic、vel 的元素/分母/标量与一次 batched output VJP（固定六片：五个 raw 项及 L5）；
- FK 的 NumPy/SciPy 解码、MuJoCo 点/J35、D_decode/J38、raw FK cotangent；
- 六项 cotangent 按显式键序 `(pos,rot6d,joint,geodesic_rad2,fk_m2,vel)` 与权重 `[1,1,1,.5,2,.5]` 加权求和，再除 4、取有效 float32，送入真实网络的**一次总目标参数 VJP**，包含 conditioner 参数；不得沿五项参考的返回顺序直接拼接后误配 FK/vel 权重。

不额外做六次参数 backward，不宣称逐项参数贡献归因。总目标 VJP 以 `allow_unused` 保留 None 语义，完整形状/名字绑定，不转成零后对账。目标不接参数图；不能从 P 的 prediction `.grad` 或参数 `.grad` 生成参考答案。

FK 使用已审计的独立公式、锁定资产及生效 float32 常数，域门见 §6。每个 I/F-I 进程只缓存一次四窗 target 的 240 个独立 FK 状态；每个 prediction micro 的 240 状态分别调用参考 positions 与 jacobians，不缓存重复预测、不复用父旧 prediction/Jacobian。生产有两次 FK batch/micro，target 在 no_grad 内；独立 FK 分母是 `4×60×15=3600`，不再乘 3。

独立性限于目标输出梯度、FK 与优化器公式；参数 VJP 仍使用相同网络和 PyTorch autograd。参数 FD 或第二求导后端没有在本轮获准，不能声称已独立证明全部网络导数。

### 4.3 累积、全局裁剪与 AdamW

参考按 micro 原顺序以 float32 累积 scaled VJP，另以 float64 从保存贡献重算总和。所有 model+conditioner 非 None 参数一起计算 L2 norm；独立裁剪系数按原生语义：

```text
n = sqrt(sum_over_all_parameters_and_elements(g*g))
c = min(1, 1.0/(n + 1e-6))
g_clip = c*g
```

`1e-6` 是当前 PyTorch clip 的固定分母项，不可省略，触发状态按 `c<1` 判定。保存生产返回 norm、两种独立归约、c 与逐元素 post-clip。若未触发，记录 `clip_not_triggered`，不改 grad_clip 或缩放真实 loss 以制造分支覆盖；触发分支由 C 合成反例验证，真实覆盖表必须注明未触发。

独立 AdamW 不调用 optimizer.step 或其 functional helper。对每个有 grad 参数的本地 step，先保存以下 float64 数学参考：

```text
m_new = beta1*m_old + (1-beta1)*g_clip
v_new = beta2*v_old + (1-beta2)*g_clip*g_clip
mhat = m_new/(1-beta1**step)
vhat = v_new/(1-beta2**step)
theta_new = (1-lr*weight_decay)*theta_old - lr*mhat/(sqrt(vhat)+1e-8)
```

数学参考与**有效 float32 参考**分开保存。不能把上述代数式在 float64 算完再统一 cast，当作原生 AdamW 的逐元素答案；当前 PyTorch 的 m 使用 lerp，第二步相消时两种求值可以显著不同。对 None 参数执行原生 skip 规则；零梯度不等于 skip。step 2 必须包含至少一个可辨的非零 m/v 递推，否则记未覆盖缺口。

#### 4.3.1 有效 float32 运算契约

C 必须生成 `adamw_fp32_contract.json`，绑定 Torch/wheel/设备、默认实际 dispatch、相关库源码 SHA、标量类型、舍入、FMA 和 subnormal/flush-to-zero 规则。该契约在准入包前冻结，R0/P/I/F 核验；不能按真实结果挑选 FMA/非 FMA 或 single/foreach 路径。以下固定候选为当前 CUDA/float32 默认 foreach 路径，不强制修改生产 dispatch；C 若不能证明与默认路径一致，须在真实读取前修订本方案，不能产生可授权的契约。

令 `R32` 为 IEEE binary32 round-to-nearest/ties-to-even，`FMA32(a,b,c)` 为精确乘加后只做一次 `R32`。输入 theta/m/v/g 均为已绑定 float32 值，g 是 post-clip 梯度；配置标量先保留 Python binary64 原值，再在下列节点转换。每个有 grad 参数先将本地整数 step 加一，随后按表中顺序计算：

| 顺序 | 有效运算；赋值处形成规定的舍入边界 |
| --- | --- |
| host 常数 | `bc1=1-beta1**step`、`bc2_sqrt=(1-beta2**step)**0.5`、`h=-lr/bc1`、`d=1-lr*weight_decay`，均先以绑定的 binary64 标量计算 |
| 衰减 | `theta_decay=R32(R32(d)*theta_old)`；该值在最终参数加法前已落为 float32 |
| 一阶状态 | `a=R32(1-beta1)`、`z=R32(g-m_old)`、`m_new=FMA32(a,z,m_old)`；冻结 lerp 的差分与 FMA，不能改成两项加权和后统一 cast |
| 二阶状态 | `v_scaled=R32(R32(beta2)*v_old)`、`g2=R32(g*g)`、`v_new=FMA32(R32(1-beta2),g2,v_scaled)`；不得跨越 v 的乘法落盘边界融合 |
| 分母 | `s=R32(sqrt(v_new))`、`b=R32(s/R32(bc2_sqrt))`、`denom=R32(b+R32(eps))`；三个节点分别舍入 |
| 参数 | `ratio=R32(m_new/denom)`、`theta_new=FMA32(R32(h),ratio,theta_decay)`；冻结 addcdiv 的除法与最终乘加边界 |

这是待 C 证明的具体求值图，尚无校准通过声明；本候选采用渐进下溢。C 用纯合成标量/小张量覆盖相消、零、正常极小值、subnormal、标量转换和两步递推，并在新解释器复验；必须确定唯一行为。未知 FTZ/FMA、影响规定节点结果的 FTZ/近似运算、未覆盖的默认 dispatch 或源码漂移均阻止准入，须先修订方案，不能让模拟器接受多种结果中的任意一种。独立参考使用 CPU 数组算术/IEEE 舍入模拟实现这些节点，不调用被测 optimizer、functional helper 或同一 Torch 更新内核代算答案。

v0.4 I/C 已在当前 CUDA/float32（torch 2.14.0+cu126，RTX 3090）**默认 foreach** dispatch 上逐位验证该图：4 组固定/对抗/次正规/首步用例、两步递推、`torch.optim.adam(w)` 源码 SHA 和新解释器重算全部通过。CUDA 单 tensor/`foreach=False` 的标量除法是倒数乘而非正确舍入除法，与默认 foreach 不同；CPU 单 tensor 的 addcmul/最终加也不同。因此契约只绑定默认 foreach，CPU-only 或强制其他 dispatch 的主机不得准入。未修改表中任何节点。

每个节点保存有效常数、数学未舍入值、规定的 float32 值和误差界。误差界由**参考输入与求值图**逐节点传播：标量转换与基本运算使用相邻 float32 值的舍入区间，FMA 只计一次舍入，区间运算向外取整；subnormal 按冻结规则处理。记输出界为 `B_m/B_v/B_theta`；不能用接近零的最终 m 本身代替相消前操作数尺度，也不能从 P 的实际后态或差异反推误差界。数学参考与有效参考的偏差只用于解释舍入及核验这些界，不直接当作生产数值失败。

#### 4.3.2 两条参考的使用与公式复核

I 使用自己的独立 post-clip 梯度、共同前态及上述有效求值图，得到并应用一次 float32 更新；float64 数学参考只作数组证据，不另应用更新。每个有效更新计入 explicit_reference_updates。

另从 **P 的 post-clip 梯度和前态**独立重算有效 m/v/theta/step，要求落盘后态与这个同输入的 float32 参考**逐位一致**；I 的后态同样核对自己的独立梯度和共同前态。原先的固定 m/v atol 和 theta 8 ULP 不能替代该门。由此隔离“目标/VJP 的路径差异”与“更新器自身错误”。逐元素零值检查见 §6.3；数学相消的正常结果不能因对 float64 最终小值使用相对容差而被误判。P/I 使用不同但已通过梯度门的输入时，状态/更新量的跨路径差异另按 §6.4 判定，不能跳过同输入逐位核验，也不能直接归因为 AdamW 错误。

这仍是每个模型/update 的一次 `optimizer_formula_checks` 只读数组复核，包含数学参考、有效求值及误差/零值/更新量判据，不应用第二次参数更新；不增加 §7 的网络、VJP、AdamW 或参考更新次数。相应数组字节数纳入 C 的 shard/资源上界。

当前 `lr*weight_decay=1e-9`，衰减可能落在 float32 分辨率以下。报告实际可辨作用，不把被舍入掉的衰减声称为已观察到非零效应，也不临时改 lr/wd。通过只证明冻结配置下的实际算子，不证明超参数有效或训练质量改善。

## 5. 正式阶段与独立收口

正式链为 **R0→P→I→F**；这里 I 是独立参考阶段，工程实现另称 I/C。

| 阶段 | 固定工作 | 消费门 |
| --- | --- | --- |
| R0 | 事前登记；授权/校准/历史、父来源/消费者、四窗/schema/常数、静态 roster/配置、资产闭包、运行时与噪声矩阵 | 不构造网络/优化器，不 torch.load，不编译 FK；真实张量读取前必须已登记 |
| P | 两模型各两个连续原生更新；严格加载与实际 roster 核验；捕获完整梯度/状态 | R0 可消费且完整阶段额度存在，factory 前重查身份/截止 |
| I | 从盘重读 P；共同前态的独立输出梯度/参数 VJP/clip/AdamW | P 来源和可消费终态已证明；完整数值负结果不自动阻断定位 |
| F | 新 Python 子进程，fresh checkpoint/后端/RNG；重跑 P 两步、再从相同前态重跑 I；重建键集、有效求值/零值/可辨性掩码及跨路径输入误差传播，比较落盘全量数组并重判所有门 | 逐单元准入；父 unknown、预算撤销或无法绑定时只整理已有证据，不构造模型或新增调用 |

F 的独立性来自新进程、新状态和磁盘重读；数学参考可与 I 共用，不冒称第三套公式。只有一套 F worker，内部顺序执行，不同时保留多套 CUDA 计算图。记录 pid/ppid/python/start_method、逐 unit 准入、实际调用和退出原因。

固定比较目录由 C 根据非空 roster 生成并冻结。令 `N=N_det+N_diff`：

| 行族 | 独立键/行数 |
| --- | --- |
| model_setup | 2 模型，2 行 |
| input_replay | 2 模型×2 step×4 micro，16 行；包括有序槽、t/noise、memory/prediction |
| objective | 上述 16 键×六项+L5+total，128 行 |
| output_cotangent | 16 行；P scaled total vs 独立重推，含 p/r6d/q 分组 |
| micro_parameter | 每模型 2×4×其 roster，共 `8N` 行 |
| accumulated_parameter | 每模型 2×其 roster，共 `2N` 行 |
| clipped_parameter | 同上，`2N` 行 |
| adamw_parameter_state | 同上，`2N` 行；含 theta/m/v/step/None、双参考、误差界、逐元素零值/可辨性掩码、同输入公式与跨路径分类；传播区间作为行字段，不增加行数 |
| update_global | 每模型两步，4 行 |
| step_continuity | 每模型 step1-post→step2-pre，2 行 |
| 合计 | **`14N+168` 行**，C 冻结具体数与完整多重集 |

所有行含状态、源数组引用/dtype/shape/hash、实测/参考、容差来源、失败位置和缺口原因；F 独立生成相同键集，不从 P/I 自报行数缩减。F 重放行只是复核，不加计为新覆盖样本。失败行、域缺口和因前置停止而未测的键仍保留，禁止空集合通过。

## 6. 域与数值门提案

6D 正常域沿用 `||a||>1e-6`、`||b_perp||>1e-6`、`||a×b_perp||>1e-12`；clamp 恰 ±1 的 FK 导数记 domain_gap，不用单边替代。独立 geodesic 沿用 `min_angle_rad=1e-6/min_sin_theta=1e-3` 的父参考适用域。新的 prediction 超出域时保存具体键和 gap，不能用父通过结果豁免，不能零值回退或换噪声/窗口。

### 6.1 比较门与适用条件

以下为拟冻结比较门。C 只用合成数据和随机初始化模块，验证正常适用域的数值比较 `4×observed` 仍在门内；逐位门必须零差异。应得 gap 的精度/跨路径敏感性用例与应拒绝的篡改用例单独验收，不能混进正向误差统计。保存原始证据/hash并在新解释器复跑；失败须在真实读取前修订方案再审阅，不自动抬高阈值。

| 数值比较 | atol | rtol | 附加门 |
| --- | ---: | ---: | --- |
| 独立 alpha/x_t vs 原生有效输入 | `2e-6` | `2e-6` | t/epsilon 逐位；alpha 还须有限、`0<alpha≤1` 且按冻结 schedule 递减 |
| loss/分母归约 | `2e-6` | `2e-5` | 公式、权重、单位和非空性 |
| total 输出 cotangent | `2e-8` | `2e-4` | p/r6d/q 分组 relative-L2/cosine |
| micro/pre/post-clip 参数梯度 | `1e-8` | `2e-4` | 逐参数及 model/conditioner 分组门 |
| global norm/clip 系数 | `1e-7` | `2e-4` | 独立公式无条件 `min(1,clip/(norm+1e-6))`，`coef<1` 即触发；`norm==clip` 也必须缩放 |
| P vs I 的 theta | `2e-7` | `2e-6` | 仅作大底数交叉检查；不能替代以下更新量/公式门 |
| 可辨坐标的同输入更新量 | `0` | `5e-4` | 仅当 `B_delta≤0.25*rtol*abs(delta_ref)` 且确已改变存储值时可辨；更紧写界按 §6.2 记 gap；不加 ULP 或固定绝对项；逐门适用性统计（含仅 precision gap 的行）实测 0.231385（4×=0.9255≤1） |
| P vs I 的可辨更新量 | `0` | `5e-4` | 原始跨路径门，以 I 的**未舍入** `delta_ref64` 为参考（不是 I 的实测增量）；单侧可辨/单侧零增量显式记分歧，失败/零分支分歧按 §6.4 分类，不能转为通过；逐门适用性统计实测 0.216102（4×=0.8644≤1）；评审二步例用已舍入 I 实测得 0.994504（错误通过），用未舍入参考得 1.001830 |
| P/I 有效 m/v 的数值路径比较 | `0` | `2e-3` | 保留原始逐元素相对门（无绝对豁免）；只有两侧精确同为零才等价；零/符号分歧与单侧零更新记录并按 §6.4 分类，v≥0；校准必须按该门自身适用性（`m_ok`/`v_ok` 通过即可，不要求整行 `classification=passed`）覆盖 `abs(reference)>0` 的全部坐标并使用同一 `atol+rtol*abs(reference)` 分母；v0.7 由此发现 0.289626（4×=1.1585>1）并重定为 `2e-3`（0.144813、4×=0.579） |
| P-m/P-v 与 P-gradient 公式对账 | 逐位 | 不适用 | §4.3 有效求值图；同时核对每个坐标的零/非零、符号和状态存在性 |
| P-theta/step 的更新公式对账 | 逐位 | 不适用 | 同输入有效 float32 结果；取消固定 8 ULP 豁免 |

常规 close 为逐元素 `abs(a-b)≤atol+rtol*abs(reference)`；覆盖通过要求全部适用门满足，不能择一通过。跨路径原始门失败时只按 §6.4 区分 failure/gap，不改阈值或把失败改成通过；其他硬门无条件执行。每个梯度 tensor/固定分组在参考 norm≥`1e-10` 时还要求 relative-L2≤`2e-4`、cosine≥`1-1e-6` 且生产非全零；更小时改用**经校准的绝对零噪声门** `max_abs_diff≤1e-6`（`ZERO_GRADIENT_ABS`），cosine 记 null+原因；p/rot6d/q 分组门同样只在参考与生产**两侧范数**均不超过 `ZERO_GRADIENT_NOISE_SCALE=1e-10` 时使用该绝对门，参考 1e-13 而生产范数 8.34e-7 的 joint 组必须拒绝（`tiny_group_joint`）。该门只处理数学零梯度（例如 attention key bias 的 softmax 相消）在 float32 autograd 中留下的约 `1e-12–1e-9` 噪声；1e-30 提案不可行。**零噪声豁免只适用于两侧范数均不超过 `ZERO_GRADIENT_NOISE_SCALE=1e-10` 的深相消带**；范数高于该带的 tensor 是非零比较，必须通过 relative-L2/cosine。即使在同一深带内，参考非零而生产全零（全擦除）也拒绝，并记录 `nonzero_erased` 坐标数；该门不放宽任何非零比较。跨路径 m/v/theta **不再使用绝对 close 豁免**：保留原始相对门，只有两侧精确同为零才等价；零/符号分歧和单侧零更新/单侧可辨增量显式记录并按 §6.4 分类。该梯度规则不能复用为 m/v 的绝对容差或更新量门。**校准按各比较门的适用性分别统计**：一行只要该门自身通过（例如 m/v 的 `m_ok`/`v_ok`；整行因更新量不可辨而 `audit_incomplete` 不影响）就是有效观测，直接 failure 或被该门拒绝的行不计入。分母与门完全一致：`abs(a-b)/(atol+rtol*abs(b))` 覆盖 `abs(b)>0` 或 `a≠b` 的每个坐标，不再用 `ZERO_STATE_ABS` 排除 1e-17 量级的 m/v。v0.6 只用 `classification=passed` 行，漏掉了 precision-gap 行：同 seed 校准链在 `state_cross_path=1e-3` 下的真实最大值为 **0.289626**（4×=1.1585>1，旧记录 0.156936 低估），v0.7 据此把提案重定为 **`2e-3`**，重算 **0.144813**、4×=**0.579≤1**；v0.7 曾据**已舍入的 I 实测增量**把 `delta_cross_path` 调至 `1e-3`（旧统计 0.297752、4×=1.191>1）；v0.8 按 §6.2 改用 I 的**未舍入** `delta_ref64` 后，原始提案 `5e-4` 下实测 **0.216102**（4×=**0.8644≤1**），因此恢复 **`5e-4`**（更严格），不再保留 `1e-3`；`delta_same_input` **0.231385**（4×=0.9255）、`theta_cross_path` **0.075190**（4×=0.3008）。放宽只作用于原始状态/更新量的相对比较，不改变零/符号直接失败、同输入逐位公式门或更新量可辨门。

### 6.2 更新量可辨性与精度缺口

P/I 共同前态已是精确绑定的 float32 值。实测增量用 `float64(theta_after32)-float64(theta_before32)` 计算；不再为这些已知存储值额外累加“前后态各4 ULP”。

对每条路径 `X∈{P,I}`，用其自身 post-clip 梯度和共同前态独立重推有效求值，取**最终参数 FMA 写入之前**的增量：`delta_ref64_X=(theta_decay32-theta_before32)+h32*ratio32_X`。乘数和前态来自绑定输入及独立求值，不能从实测后态反推。`B_delta_X` 只覆盖该路径最终写入的舍入：RN32 取独立有效后态上下相邻有限 float32 间距较大值的一半，以 float64 保存，渐进下溢同样按相邻值区间处理；FTZ 影响结果、无法界定或溢出即 gap。前置节点已形成明确的有效 float32 输入，不将 `B_theta`、§6.4 的输入传播区间或前态 ULP 再累加到此界；界的生成不读实测误差。逐坐标规则为：

1. 每路径 `delta_ref64_X==0` 时单列 `expected_no_change`，该路径实测增量必须为零，同输入公式门仍须通过。零分支不使用相对误差除法，也不声称观察到非零衰减/更新效应；P/I 参考零/非零不同另记跨路径分歧，按 §6.4 分类。
2. 每路径非零参考仅在 `B_delta_X≤0.25*5e-4*abs(delta_ref64_X)` 且有效参考确实改变存储值时，标 `update_delta_observable_X=true`，要求 `abs(delta_measured_X-delta_ref64_X)≤5e-4*abs(delta_ref64_X)`；不给判据再加 `B_delta_X`、ULP 或 `1e-9`。同输入公式门也必须通过。0.25 的可辨裕量把“舒适可辨”与比较门分开：若可辨条件也用完整 5e-4 门，写入舍入误差本身可达门的 100%，`4×observed≤proposal` 的校准规则在数学上不可满足。两侧均可辨时另保留原始跨路径门 `abs(delta_measured_P-delta_ref64_I)≤5e-4*abs(delta_ref64_I)`（I 侧必须用**未舍入参考**，不能用 I 的实测增量；可辨性仍按同输入门的 `0.25*5e-4` 裕量判定）；任一侧可辨而另一侧不可辨/增量为零时显式记单侧分歧，按 §6.4 分类，不当作通过。
3. 每路径其余坐标标 `update_delta_observable_X=false`，记录 `parameter_delta_precision_gap` 及坐标/原始值/界。误差界只判断“能否证明”，不能把不满足更新量门的差异变成通过。任一路径存在该必测 gap，F 最终为 `audit_incomplete`，即使其他层完整；同输入公式等已测失败仍保留，不把精度不足宣称为 AdamW 实现错误。固定矩阵若因此无法完成覆盖，保留该结果，不能更换权重、窗口、lr、dtype 或阈值求通过。

v0.8 纯合成全链（264 个参数 tensor、d_model 240、四窗、两步）在 step 1 出现 **392** 个 `parameter_delta_precision_gap`（first-step `m/√v≈0.1`，更新约 `1e-5`，而 float32 参数写入分辨率对 |theta|~0.1–1 为 `1e-8–6e-8`），另有 **190** 个 `cross_path_sensitivity_gap` 和 **0** 个 `cross_path_bound_unverifiable`。全部同输入公式/直接门通过，`failures=[]`、`problems=[]`；这是冻结矩阵在 lr=`1e-4` 与 float32 写入门下的真实精度/相消限制，不是 AdamW 实现错误，也不能通过换权重/窗口/lr/threshold 求通过。相对 v0.7 的 185 个 sensitivity gap，v0.8 因把可辨跨路径更新量门从 `1e-3` 恢复为原提案 `5e-4`（更严格）并改用未舍入 I 参考而多 5 个；392 个 precision gap 不变（它们来自更新量写界而非状态门）。v0.4 记录的 660/340/2 与 F 合计 1002 条为旧 F 将 I 行 gap 与 I 报告 gap 重复计数的统计；后续按 I 行一次计数，原始旧工件保留不改写。

校准必含审阅标量：前态 `1.0`、参考后态 `0.999997615814209`、被改后态 `0.9999980926513672`，存储更新量少20%。同输入有效公式应拒绝被改后态；即使更新量因可辨性不足进入 gap，也绝不能被整行或最终裁决接受为通过。C 同时提供可辨的大更新正例及正确但不可辨的小更新 gap 正例。

### 6.3 逐元素状态零值与非数值语义

m/v 的零值门在范数、分组聚合和数值容差之前执行。分别用 P 的绑定梯度/前态及 I 的独立梯度/共同前态，从有效 float32 求值图重推**每个坐标**的 `zero_mask/nonzero_mask`、符号、有效值和 underflow 标记；F 从盘独立重建这些掩码。P 的后态须逐位等于 P 同输入参考，I 的后态须逐位等于 I 参考；P/I 之间的零值/符号分歧另记 `cross_path_zero_or_sign_disagreement` 并按 §6.4 分类，不转成零、忽略坐标或用其他大元素的相对误差吞掉。

只有被冻结的 IEEE/FTZ 规则独立导出的实际零值才可解释为下溢；无法证明即 `state_zero_semantics_unverifiable` gap。**本路径同输入参考**是可表示非零值而本路径实测为零，记 `state_nonzero_erased` 失败；本路径参考为零而本路径实测非零同样失败，v 的任何负值失败。这些直接失败不能被跨路径传播区间豁免。None/状态未初始化按 §2.3 单独核对，不能转换为零 tensor。数学相消用有效求值结果判断，不能因 float64 理想式非零就否决合法的 float32 零值。

必测反例为非全零张量的**局部清零**：首步 `g=1e-17` 对应 `v≈1e-37`（正常、非 subnormal 的 float32），清零此坐标并保留其他较大 v 仍必须失败；另测可表示微小 m、合法真零、真实下溢、相消后零值和正负号。删除或协调篡改掩码也须由原始前态/梯度重推拦截。原 `1e-20/1e-30` 状态 atol 与“全张量仍非零”不再是验收依据。

布尔/计数/None、roster、形状、step 和配置精确比较；原生 AdamW step tensor 要求 0 维、有限非布尔数值，值精确等于该参数的非 None 更新次数（连续参与时为1/2），另存 JSON 整数 step。None 不初始化/不递增；拒绝1.9、字符串或广播数组。

F-P 的输入、prediction、梯度、参数和 AdamW 状态与 P **逐位**重放；F-I 相同浮点路径也须逐位重放，独立 float64 公式的跨后端比较才使用对应数值门。deterministic_algorithms=false 下若无法满足，登记 replay mismatch，不临时改成近似门。**F 另须逐字段对账保存的 P/I 报告**（micro loss/parts、gnorm、clip_coef/triggered、RNG before/after digest、prediction/memory/arrays hash、state_presence/step_facts、report 内嵌 rows 与 rows 文件），并将保存行与重放行按 `(key, 全行 canonical hash)` **多重集**比较；重复/缺失/额外/漂移行均拒绝，不能先用 key→row 字典去重。输出 cotangent 行必须携带 `pos/rot6d/joint` 三个分组门及其 problems；F 重放重算这些分组并按 canonical hash 绑定，任何只改分组事实（例如把被清零的 joint 组标为通过）也报 `row_drift`。报告或 RNG 被改写而 NPZ 未变时，F 必须报 `report_micro_drift`/`report_npz_sha` 等 verification problem，而不是只比 NPZ。

全部数组先做容器/键集/dtype/精确 shape/非空/有限性校验；bool 与数值不能混比。新增/缺失/重复参数、改明细、全零替换、NaN/Inf、只保存范数均阻止通过。直接硬门的完整超容差结果是 mismatch；跨路径数值门按 §6.4 分类，不可计算/缺证据是 gap。

### 6.4 跨路径输入误差传播与相消分类

同输入公式逐位正确，不代表两套不同梯度在相消附近会产生相近的 m 或更新量。§6.1 允许的上游梯度数值差异必须与更新器自身错误分开；最终参数写入是否可辨（§6.2）也不能代替这一敏感性检查。

C 在 `adamw_fp32_contract.json` 中冻结传播规则。对 I 的有效 post-clip 梯度 `g_I`，先由 §6.1 已冻结门定义允许输入集合：传播预算必须覆盖**梯度门实际允许的全部输入**，不能再用更紧的“tensor 小范数预算”把合法输入排除。逐坐标预算为：
- deep cancellation band `||g_I||≤ZERO_GRADIENT_NOISE_SCALE=1e-10`：`E_j=ZERO_GRADIENT_ABS=1e-6`（零噪声门）；
- `1e-10<||g_I||≤1e-6`：`E_j=2e-4*||g_I||`（分组 relative-L2 门）；
- `||g_I||>1e-6`：`E_j=1e-8+2e-4*abs(g_I[j])`（逐元素与分组门）。
relative-L2/cosine、非全零、None 和裁剪门仍须单独满足。区间 `G_j=[g_I[j]-E_j,g_I[j]+E_j]∩finite_binary32` 只是这些门允许集合的保守外包，不替代原门，也不证明任何实现误差已获允许。v0.4 的 `min(1e-30+2e-4*max_abs_ref, cell)` 会比梯度门的绝对项更紧，已撤销；撤销状态绝对豁免后，该收紧在同一合成矩阵造成 32 个 `cross_path_bound_unverifiable`（review_fix1 中间运行），v0.5 改为门允许集合后为 0。

从已绑定的共同 theta/m/v/step 前态和 `G_j`，沿 §4.3.1 **唯一有效求值图**传播，生成逐坐标 `K_m/K_v/K_theta/K_delta`。`K_delta` 是可能的最终存储增量集合的外包；它与各路径的 `B_delta_X` 分开。区间必须**逐节点**向外舍入到 binary32，至少覆盖 `z=R32(g-m)`、`m=R32(a*z+m)`、`g2=R32(g*g)`、`v=R32(c*g2+v_scaled)`、`s=sqrtf(v)`、`b=R32(s/bc2)`、`den=R32(b+eps)`、`ratio=R32(m/den)`、`theta_decay=R32(theta*d)`、`theta=R32(h*ratio+theta_decay)`；平方跨零、相消、符号/零值变化、分母和最终写入全部包括，不能只做中心点的一阶线性估计，也不能在 float64 中心值上直接加减半 ULP。传播预算仅由 I 的绑定输入、共同前态、固定阈值和求值图生成；禁止读取 P/I 实测后态、实测差异或按真实结果放宽区间。`B_m/B_v/B_theta` 的数学式舍入界也与这里的上游输入传播区间分开保存。合成端点反例必须对每个实际 binary32 输入（含 `g_I` 与相邻可接受值）重推有效图的 m/v/theta/delta，全部落在对应 `K` 内；当前 16×5 组随机尺度端点检查通过。

按以下优先级裁决，保存原始跨路径 close/零值结果和分类，不能覆盖原始结果：

1. 先保留来源/前置既有的 failure/gap，不以传播替代其核验；可计算的 None/roster、上游目标/参数梯度/累积/clip 等直接门失败，保留最早直接 failure。P/I 任一同输入 m/v/theta/step、零值或自身可辨更新量门失败，同样直接记 failure。传播区间再宽也不得降级这些失败，尤其不能放过只改后态或局部清零。
2. 所有前置门和两侧同输入硬门均通过，但原始跨路径状态/更新量门失败或零值/符号不一致时，检查独立重推的两侧有效结果和实测存储增量是否均落在对应 `K` 内。若在区间内，记必测 `cross_path_sensitivity_gap`：允许的上游数值差异经有效求值被放大，本矩阵无法证明跨路径一致；不能归为 AdamW 公式错误，也不能算通过。
3. 传播规则/原始量缺失、无法构造可信有限区间，或前置及同输入硬门均通过而结果却落在区间外，记必测 `cross_path_bound_unverifiable`，保留矛盾的原始位置；不能用有矛盾的参考界宣称已定位生产错误。原始跨路径门通过时仍须满足其他全部硬门；落在区间内本身从来不是通过条件。

任一上述 gap 使 F 为 `audit_incomplete`；其他独立测得的 failure 继续保留，不因 gap 删除。F 从落盘梯度/共同前态/冻结规则重新生成预算、区间、原始门与分类，不信任保存的掩码或自报 gap。该工作属于现有 `optimizer_formula_checks` 的只读数组复核，不增加更新、网络或 VJP 调用；CPU 时间与新增字段字节纳入原阶段预算和 C 的资源上界。

必测纯合成标量为：共同前态 `m=R32(.001)`、`v=1.0000000116860974e-7`、`theta=0`、step=1，第二步 `g_P=-0.008999999612569809`、`g_I=-0.008999998681247234`（仅差1 ULP，相对差约 `1.03e-7`）。按候选有效求值，`m_P≈5.4948032e-11`、`m_I≈1.4808030e-10`，相对差约62.9%；两侧更新均可辨，且各自公式正确，必须由跨路径原始门失败得到 `cross_path_sensitivity_gap`，不能误报更新器错误或通过。C 另须验证上游门超限、后态单改/局部清零仍为直接 failure，以及预算/区间/分类被协调篡改时被 F 原始量重推拒绝。本段是待实现的验收要求，不是校准通过记录。

## 7. 累计预算、调用和资源上限

### 7.1 时间与尝试

| 阶段 | 同 cohort 累计预算 | 阶段收口预留（已含） | attempt 上限 |
| --- | ---: | ---: | ---: |
| R0 | 240s | 30s | 2 |
| P | 900s | 120s | 2 |
| I | 900s | 120s | 2 |
| F | 1200s | 180s | 2 |
| 合计 | **3240s（54 分钟）** | 不互借 | 每阶段独立 |

预算不是估时。所有 attempt、失败/中断/被取代运行累计；重试不会再获得一份时间预算。每次工作截止为 stage 起点加上「该阶段剩余预算−阶段收口预留」，收口预留包含在阶段总额中，与另计入成本的 seal-tail 不混用。启动器在 uv/import 之前记录单调起点，入口/读盘/strict-load/构造/资产编译/传输/同步/计算/写盘/撤回/封口均计费；额外真实预检或数值 post-check 不能游离账本。

### 7.2 名义矩阵与累计硬上限

每个 P/I attempt 上限为其一列名义值，F attempt 为其一列；同阶段累计调用上限为该列×2，cohort 为全链列×2，均与最多两次 attempt 同时约束。尚有时间不能突破调用额度。R0 前证明完整链额度，重试前证明余下阶段额度，不能启动必然被调用门截断的链。

| 真实计数器 | R0 | P | I | F | 一次全链名义 | cohort 硬上限 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| checkpoint loads | 0 | 2 | 2 | 4 | 8 | 16 |
| model constructions | 0 | 2 | 2 | 4 | 8 | 16 |
| conditioner constructions | 0 | 2 | 2 | 4 | 8 | 16 |
| native optimizer constructions | 0 | 2 | 0 | 2 | 4 | 8 |
| conditioner forwards | 0 | 16 | 16 | 32 | 64 | 128 |
| backbone forwards | 0 | 16 | 16 | 32 | 64 | 128 |
| native add_noise batches | 0 | 8 | 8 | 16 | 32 | 64 |
| CPU t+epsilon draw units | 8 | 8 | 8 | 24 | 48 | 96 |
| production LossAssembler | 0 | 16 | 0 | 16 | 32 | 64 |
| production float32 FK batch | 0 | 32 | 0 | 32 | 64 | 128 |
| production parameter backward | 0 | 16 | 0 | 16 | 32 | 64 |
| independent output batched VJP | 0 | 0 | 16 | 16 | 32 | 64 |
| independent total parameter VJP | 0 | 0 | 16 | 16 | 32 | 64 |
| native global clip | 0 | 4 | 0 | 4 | 8 | 16 |
| independent clip formula | 0 | 0 | 4 | 4 | 8 | 16 |
| native AdamW steps | 0 | 4 | 0 | 4 | 8 | 16 |
| explicit reference updates | 0 | 0 | 4 | 4 | 8 | 16 |
| diagnostic parameter update events | 0 | 4 | 4 | 8 | **16** | **32** |
| anchor_state_restores | 0 | 0 | 2 | 2 | 4 | 8 |
| optimizer_formula_checks | 0 | 0 | 4 | 4 | 8 | 16 |
| reference mj_forward | 0 | 0 | 7920 | 7920 | 15840 | 31680 |
| reference mj_jacSite | 0 | 0 | 57600 | 57600 | 115200 | 230400 |
| scene setup mj_forward（另计） | 0 | 1 | 0 | 1 | 2 | 4 |
| micro execution units | 0 | 16 | 16 | 32 | 64 | 128 |

一套 I/F-I 的参考 forward 为 target cache 240 + 16 micro×(240 positions+240 Jacobian)=7920；jacSite 为 16×240×15=57600。每个 FK batch 固定 240 状态/15 点；不融合参考 helper 或缓存预测以改变矩阵。生产 scene 初始化每个干净 P/F 进程各一次，属于构造成本，contact 分支仍为零。

`diagnostic_parameter_update_events = native_AdamW_steps + explicit_reference_updates`，`training_updates` 为该总数的严格整数派生字段，一次完整链为16、cohort上限32，F 必须重推其等式。源权重加载、共同前态恢复、公式的只读数组复算分别记录，不混入或隐去实际更新；不通过“丢弃副本”把 training_updates 写成零。

本链以下累计计数必须为 JSON 整数 0：`parameter_fd`、`state_fd`、`ddim_calls`、`validation_calls`、`selection_calls`、`production_checkpoint_writes`、`contact_branch_executions`。完整网络/optimizer 状态会保存为 audit artifact，必须标 `diagnostic_only=true, usable_for_generation=false`；不可因不叫 .pt 就隐瞒其内容或作为可部署 checkpoint。

### 7.3 资源与中断

资源提案：单 GPU/单计算 worker，GPU peak≤22 GiB、host peak≤32 GiB，同 cohort 工件≤16 GiB，启动时磁盘 free≥20 GiB。C 按 schema **逐类**给出数组字节上界，不允许用固定“参数副本数”代替：P 每 attempt 为 33 份参数 tensor（theta_init 1 + 每 micro accum/contrib 16 + 8 个 update theta/state 副本）加前向数组/元数据；I 为每 update 17 份 float64 有效/传播/数学参考数组 + 10 份 float32 行/显式更新数组加 cotangent/FK 元素/元数据；F 为一次 P 重放加一次 I 重放；再计入 fixture、注册元数据和每阶段 2 次 attempt。按 full roster（264 tensor、7,567,220 元素）计算单链上界 **7.145 GiB**、cohort 上界 **14.236 GiB≤16 GiB**；v0.8 权威合成实际约 6.895 GiB（精确字节见验证记录）。R0 核验资源余额；这些是限制，尚无真实峰值实测。超限/OOM 诚实 incomplete，不减 micro、切半精度、追加重计算或并行。

每个 unit 前持久化完整调用预留、身份和截止，在 backward/每个 FK 状态/更新/大文件 shard 前后重查工作截止。**阶段 `deadline=t0+remaining_s-reserve_s` 是唯一的工作截止，`reserve_s` 只扣一次**：主 P/I 的本地检查在每个 model/update/micro/pre-backward/clip/step/VJP、参考 target/FK/Jacobian 前无条件执行，另带 `unit_admission`（父登记必须 running、cohort/registration 匹配、未过截止）；F 的 pre-spawn、子进程 timeout、worker guard 与 `unit_admission` 均直接比较 `perf_counter()` 与同一个绝对截止，不再二次扣预留。**F worker 继承父进程绝对 `perf_counter` 截止**，并在读取保存工件、构造 model/optimizer、每个 micro、每次 backward/VJP、clip 和 update 前重查父 running 登记与截止；启动时父截止已过则不 spawn。worker 在 finally 上报本表全部24类计数及零计数的完整快照，必须全覆盖且每项为非负 JSON 整数，不接受部分 delta 或 bool。父进程**仅在完整快照核验并合并成功后** settle；任一失败/无快照（被杀/超时）保持 reservation `consumed=null`、历史 unknown，不以父 counter 差值结算为零。未结算额度不当作零返还。

## 8. 登记、成本封口与磁盘契约

沿用当前已审计的固定根归属和 publishing 语义，I/C 必须重新覆盖本方案的计数/更新异常，不能复制旧 helper 后假设天然成立。

1. 实质读取/计算前写 cohort、active running attempt 的盘×索引×台账；R0 前仅做授权/校准/历史和文件身份检查。来源、scope、注册 SHA、unit 目录必须匹配，子进程截止不晚于父。
2. index/ledger 分半读取，先证归属再证完整：任何存活成员证据加另一半损坏为 unknown；完整外部 cohort 跳过不计数；有标记且半损坏无法排除成员时 unknown；无成员证据的 legacy 按既有规则跳过。
3. 每个 attempt 精确对账有限非负成本、调用增量/累计快照、状态、report/array hashes 和目录枚举；缺失/未登记尝试、调用快照矛盾或 running/publishing 都停止消费。撤回/否决封口必须写入**执行后的累计调用快照**，不得复用 running 行的 pre-stage 余额；索引/台账两侧同时更新，并与 reservation `consumed` 按 `final-before == consumed` 逐类对账，不一致即 `call_balance_delta:*` 否决。
4. 终态先发布 publishing/complete=false。成本含自身持久化和尾部预留 `max(1s,2×观测最大写延迟)`；封口后至多4轮核验/刷新，撤回成本同属本 attempt、以旧成本为下限（`cost_floor`），不得把旧成本改小。未知覆盖保留 publishing，不发布低估的 incomplete 历史。
5. 预算/调用/全链矩阵/消费者否决必须在 F terminal 前撤回磁盘 complete/pass 声明、重写报告并重绑哈希；pre-seal 否决直接以 incomplete 封口，post-seal 否决重新走 publishing→terminal 并保留旧成本下限。撤回不确认时返回 audit_incomplete、保留 publishing 非可消费行。区分证据完整性与数值通过：I 的完整数值 mismatch 仍须以 `complete_evidence=true` 进入 F，F 独立重放并保留已测 failures；只有预算撤销/未知封口/证据缺失阻止 F。不得沿用旧 terminal 标志以成功退出。

不能消除的任意终止/文件系统边界须按现有协议暴露，不新增“固有时钟边界”豁免。父产物与旧授权字节不改；本方案的真实授权与登记另存新目录。

## 9. 工件与最终裁决

正式目录为 `registration/production/independent/finalize/attempt_NNN/`。I/C 与准入材料另存 `derived/audit/w1_real_training_update_audit/`，不作为正式运行证据。

| 工件 | 必需内容 |
| --- | --- |
| registration/admission/expected_keys | 父锚、固定窗、roster、micro/noise、环境/资产闭包、参考公式/容差、预算/调用、授权与实现逐文件哈希；绑定 `adamw_fp32_contract.json` 的完整求值图、跨路径输入误差传播/分类规则与合成证明 |
| parameter_roster/micro_schedule | 全参数名/shape/顺序、graph/None 契约，样本多重集、t/epsilon 与独立 x_t |
| production/independent reports | 原始六项/L5/total、输出 cotangent、完整梯度/clip/state 比较、每行失败/缺口 |
| 按 model/update 切分的 NPZ shards | theta/m/v/step pre/post、micro contribution/累计/pre/post-clip、prediction/条件、参考元素/Jacobian/cotangent、RNG；有效常数/逐节点双参考/舍入界、各路径自身增量与可辨性/零值掩码、跨路径输入预算与 `K_m/K_v/K_theta/K_delta`；每 shard 独立 schema/hash |
| audit_summary/independent_verification | F 从盘重推键集、所有数组/明细/公式、直接硬门与跨路径原始门/分类、状态连续性、预算/调用与消费者状态；分别记录 `same_input_formula_passed/raw_cross_path_passed/cross_path_classification`，不得以 gap 覆盖原始失败 |

不允许 NPZ 合并覆盖同名键而漏掉冲突；JSON 列表/映射、dtype/shape、缺失数组均先验证，异常登记 gap 而不是留下崩溃的 running。F 不能只比范数或信任自报 complete/reasons/counts；对每个 shard 和全部明细分别对账。

| 裁决 | 必须满足 | 含义 |
| --- | --- | --- |
| `real_training_update_covered_contract_passed` | 全 roster/固定矩阵完整；P/I/F 可消费；failures/gaps 空（含精度、零值语义与跨路径敏感性缺口）；两步连续、全部直接硬门/跨路径原始门/有效求值/逐位重放、调用/成本通过 | 只排除该矩阵和有效域上已覆盖算子的偏离；报告实际 clip/衰减分支覆盖 |
| `real_training_update_mismatch` | 证据完整可消费，无必测 gap，存在独立重推的直接 failure | 定位最早直接差异：输入/目标 cotangent/参数 VJP/累积/clip/同输入 AdamW/重放；跨路径原始超差不能独自归因 AdamW |
| `audit_incomplete` | 任一必测 gap（含 `parameter_delta_precision_gap` / `state_zero_semantics_unverifiable` / `cross_path_sensitivity_gap` / `cross_path_bound_unverifiable`）、前置/来源无效、预算/调用不足、未知历史或不可消费封口 | 停止；保留已测失败，不把数值不可辨或相消敏感性误称为实现错误或覆盖通过 |

`complete_evidence` 与 `covered_contract_passed` 分开；完整负结果可为 true/false。`implementation_contract_passed` 只表示审计器的登记/重推完整性。没有 val31 或动作质量测量，不以训练 loss 下降、参数非零或审计通过选择干预因素。

## 10. I/C 交付、合成验收与授权门

### 10.1 工程交付

v0.4 已新增 `src/ireset/eval/w1_real_training_update_audit.py`、`scripts/audit_w1_real_training_update.py`、`tests/test_w1_real_training_update_audit.py`，并最小修改 `trainer.py` 抽出共用 update unit/只读捕获（`predict`、`begin_update`、`accumulate_micro`、`clip_gradients`、`step_optimizer`，以及审计注入的 `loss_fn`/`build_optimizer`）。默认 CLI 仅纯合成；真实模式校验专用授权、冻结准入包/校准、完整调用余额和固定历史根后仍拒绝（I/C 未注册真实 provider）。以下 I/C 组已完成纯合成验收：

| I/C 组 | 必须验证的正向链与反例 |
| --- | --- |
| 共享生产单元 | 重构前/后 observer=None 逐位成功路径及失败/计数/RNG；observer不改数值；真实 run 的验证/保存逻辑回归，但不以真实资产测试冒充纯合成 |
| 累积/图 | loss/4 漏除/重复除、清错梯度、错micro/窗口多重集、conditioner detach/漏参/重复参数、None与零混淆、合法 deterministic 零输入梯度；合成大batch只验证无随机差异的等价用例 |
| clip/AdamW | 未触发与真实触发 clip、漏 `1e-6`、只裁模型、step2重置m/v、错误wd/epsilon/bias correction、负v/百万倍v、广播shape、step1.9/bool/字符串 |
| 有效求值/相消 | 在当前默认 dispatch 上验证 §4.3 每个节点及 FMA/标量转换/FTZ；`m_old=R32(.001),g=R32(-.009)` 的合法第二步相消必须按有效求值通过；float64 代数式一次 cast 不能冒充答案；错舍入节点/错 FMA 需拒绝 |
| 跨路径相消分类 | §6.4 的1 ULP梯度差正例：上游门及两侧同输入公式通过、更新可辨，原始跨路径相对差约62.9%仍应为必测gap；另测参考零/符号分歧、普通稳定通过、上游超限与仅改后态的直接failure。区间缺失/错误/从实测差异反推、协调篡改分类均须由F重推识别，不能拿区间包含代替通过 |
| 更新量可辨性 | 可辨更新正例、正确但不可辨更新的必测 gap 正例；§6.2 更新少20%但旧ULP门全过的反例、只改theta不改m/v、错误掩码/B_delta/零分支；任何失败不得由ULP项补成通过 |
| 状态局部清零 | 非全零张量中只清零 `v≈1e-37` 或可表示微小m仍拒绝；合法真零/下溢/相消零、符号/None与零混淆；同步改掩码也须被原始梯度/前态重推拦截 |
| 独立参考 | 输出图误接、未含FK/错误3600分母、FK/vel权重错配、借用P梯度、错mean/std/关节顺序、6D/geodesic/clamp域；近π负门不放宽 |
| 来源/工件/F | 错checkpoint/roster/共同前态、noise/x_t/RNG错接、缺/重/多行、空/非数值/NaN/Inf数组、bool混数字、自报明细与反例语义篡改；F读取落盘梯度而不是内存对象 |
| 协议 | 启动/import计费、跨run尝试/调用累计、完整外部cohort、半损坏成员、未结算worker、预算撤销后零新调用、慢封口/撤回成本、磁盘声明与返回/消费者一致 |

C 只用合成解析数据、随机初始化 full-architecture 模块与合成 FK；静态 roster 不加载真实权重。真实 G1 测试/真实 checkpoint 前向、真实训练更新不得混入授权前校准。保存合成证据、精确测试范围、保护清单、有效运算/传播分类契约、容差/roster/schema/calibration evidence SHA；固定正例/反例/应有gap的语义，独立复核从原始量重新分类，不信任自报预期原因。普通稳定正例仍须通过全部原始门，跨路径敏感性gap正例单列，不能用其差异放宽阈值。不拿历史全项目计数当作新验收。

**v0.8 I/C 交付事实（纯合成）**：第六轮审阅 2 P1 全部修复并加重反例：①跨路径更新量 close 门改用 P 实测 vs I 的**未舍入** `delta_ref64`（存在性另比较 P/I 两个参考）；评审二步例（theta `4.3598099728114903e-4`、m `1e-3`、v `1e-7`、gP `-8.899898268282413e-3`、gI `-8.9`）用已舍入 I 实测得 0.994504（错误通过），用未舍入参考得 **1.001830**，现记 `cross_path_sensitivity_gap`；校准 `delta_cross_path` 在恢复的原提案 `5e-4` 下实测 **0.216102**（4×=0.8644），v0.7 基于已舍入实测的 `1e-3`（0.148876）撤回。`small_unobservable_update` 仍为 `audit_incomplete`。②`run_i_stage` 在 `ReferenceContext.evaluate` 前检查 `i_ref_fk_{u}_{m}`，`ReferenceContext` 在目标窗/evaluate/FK/Jacobian 前执行同一截止+`unit_admission` 守卫；受控时钟在首个 forward 返回后过期时不再新增 480 次 `reference_mj_forward`/3600 次 `reference_mj_jac_site`，Jacobian 单元不启动。`tests/test_w1_real_training_update_audit.py` **54 passed**，相关回归 **263 passed**（全项目套件未重跑）。校准 contract SHA `271df4e3ee0bf84cc6e92e5c436cac3d5f0ef08cb19addbb787cf3543bc339ca`、evidence `aa4bd83167ea11a7c1cb159e494269b7d843f8233bbbc272558caf9c59e0a9ed`、fp32 canonical `6d7a24d0cd077454120dfbaaad37fcc378b10572e4a33a26df8ca09f7e0003e1`。**最终全链** `derived/audit/w1_real_training_update_audit/synthetic/v0.8_final/`（full d240/4 layers、264 tensor、3864 行、四窗、两步）：四 stage done/sealed，调用矩阵精确（8 loads、64/64 forwards、48 draw units、32/32 VJP、64 production FK、16/16 updates、15840/115200 reference FK、64 micro），`training_updates=16`、零计数 0；结论 **`audit_incomplete`**（complete_evidence=true、covered=false、failures/problems 空、I/F 各 **582** gap：392 `parameter_delta_precision_gap` + 190 `cross_path_sensitivity_gap` + **0** bound-unverifiable）；F 新进程 pid≠父、ppid=父、exit 0、报告/RNG/行多重集复核通过；成本约 161s（R0/P/I/F 各 1/2 次，精确值见验证记录）；工件约 6.895 GiB、单链上界 7.145、cohort 14.236≤16。只读准入包在文档冻结后重新生成（`authorized=false/registered=false/run=not_run/section_10_2_checked=false`）；真实 provider/checkpoint/数据/资产仍未读取。

### 10.2 新的明确授权记录

旧 FK 的 §9.2 授权仅允许保存输入 FK 审计，不能用于本方案。真实执行需用户另外明确授权，记录引用原指令，并严格校验 schema/身份/类型/数值。以下十项当前均未勾选：

- [ ] 确认新问题/cohort、四个 sub10 窗、两固定 checkpoint 与父消费者，不借旧预算/调用。
- [ ] 确认 fresh AdamW；诊断 step1/2 不代表历史续训或质量训练。
- [ ] 确认两个连续生产更新、共同前态参考、完整参数/None roster、固定 micro 与 CPU RNG 协议。
- [ ] 确认真实网络/输出与参数反向、FK 参考、global clip 和 AdamW；一次链16个参数更新，累计硬上限32，不能声明训练更新0。
- [ ] 确认 §7 全部 per-attempt/per-stage/cohort 调用矩阵、未结算即unknown和重试计费。
- [ ] 确认 R0/P/I/F 240/900/900/1200s、各最多两次及收口预留；资源上限22/32/16 GiB、磁盘free≥20 GiB。
- [ ] 确认通过合成验收的共享生产单元、有效float32求值图/逐元素零值/逐门适用性状态校准与`2e-3`相对门/`5e-4`可辨跨路径更新量门（以 I 未舍入参考为准）/严格同输入更新量门/cotangent p·rot6d·joint 分组门（两侧深带）及跨路径输入误差传播；参考 target/FK/Jacobian 与输出 VJP/clip/update 前均检查截止；精度不可辨/相消敏感性为必测gap，同输入公式错误仍为failure；同前态逐位与F新进程重放，不事后改dtype/确定性/阈值。
- [ ] 确认 parameter/state FD、DDIM、验证/选模、生产checkpoint写入、contact分支全部0；审计状态明确diagnostic_only。
- [ ] 确认 publishing/成本只增不减、三方对账、撤回磁盘声明、unknown失败退出和旧证据保护。
- [ ] 确认通过仍partial/not_ready、不自动选因素/续训/改教师/扩数据/接入生产。

授权 JSON 应绑定本文件、实现/环境、准入包、含跨路径传播/分类规则的有效float32求值契约、校准证据、roster/schema 和父来源 SHA；budgets/attempts/caps 取 §7 精确值，计数与清单用严格 JSON 整数/bool，不接受字符串或 0.0 冒充整数。授权之前不得建立真实 cohort 或消耗真实额度。

### 10.3 当前交付与下一步

v0.8 已完成 I/C 实施、第六轮 2 P1 修复、纯合成验收、冻结校准/有效 float32 契约、F 新进程重放和只读准入包重发；共享 update unit 与原生 Trainer 的 observer=None 路径逐位等价。真实 provider、真实 checkpoint/数据/资产读取、真实 R0→P→I→F、授权/注册均未实施；本轮真实新增调用及参数更新为 0。合成全链是完整的负/不完整结果（`audit_incomplete`、582 个必测 gap），不冒充覆盖通过。

下一步复阅 v0.8 的校准/准入包、逐门适用性状态校准与 `2e-3`/`5e-4` 相对门、未舍入 I 参考的跨路径更新量门、有效 float32 dispatch 绑定、1e-6 深带零噪声门（两侧范数）、25% 更新可辨裕量、step-1 精度 gap 语义、cotangent 分组门、参考 FK/Jacobian 截止单元和真实 provider 计划；随后另行实施真实 provider 与 R0→P→I→F 的前置材料。收到本方案新的明确真实授权后，才执行真实 stage。U0 incomplete、UR no_quality_gain、原F not_ready、旧预测r6不可验收与旋转融合不接入全部保留。

## 11. 修订记录

| 版本/审阅项 | 本次文档修订 | I/C 必须给出的证据 |
| --- | --- | --- |
| v0.2 / P1 更新量被ULP吞掉 | 取消更新量的固定绝对项及前后态ULP叠加；同输入有效参数后态逐位核验；独立未舍入增量另设严格相对门，可辨性不足为必测gap | 旧门放过的20%偏差不得通过；大更新正例通过、小更新不可辨正例诚实gap；F重推掩码与界 |
| v0.2 / P2 相消求值未冻结 | 分开数学参考与有效float32求值，列明lerp、FMA、v乘法、分母与最终写入边界；冻结运算/环境契约与基于操作数的舍入界 | 正常第二步相消在当前默认dispatch通过；错误求值图被拒；未知profile不准入 |
| v0.2 / P2 状态部分清零 | 撤销固定状态atol，逐元素零值/符号/有效后态核验先于聚合；由原始量重建掩码 | `g=1e-17` 的正常非零v被局部清零须拒；合法真零/下溢/相消零不误报 |
| v0.3 / P2 跨路径相消误判 | 两侧自身增量/公式与原始跨路径门分开；从I梯度、共同前态及冻结上游误差预算传播有效区间，硬门错误仍failure，无法证明跨路径一致为必测gap，不能转为通过 | 1 ULP梯度差、m/可辨更新约62.9%差异的合法算子为sensitivity gap；稳定正例通过；上游超限/后态篡改仍拒；F独立重推预算/区间/分类 |
| v0.4 / I/C 实施与校准 | 共享 update unit；审计器/CLI/测试；有效 float32 契约在默认 CUDA foreach 上逐位验证；独立 x_t 用生产 binary32 alpha；1e-30 极小门改为经校准的 1e-6 零噪声门；合成全链诚实记录 step-1 精度/相消 gap | 25 tests；calibration evidence/fp32 contract；20%更新/局部清零/1ULP/区间篡改反例；F 新进程逐位重放；准入包 |
| v0.5 / 8 P1 数值门与协议缺口 | 零噪声门限制到 1e-10 深带并拒绝非零擦除；状态跨路径撤销绝对豁免、保留相对门和零/符号/单侧更新分歧；F 对账保存报告/RNG/行多重集；仅完整 worker 快照结算；worker 继承截止+逐单元准入；否决在 terminal 前撤回磁盘声明；撤回纳入 publishing/成本下限；完整数值 mismatch 进入 F；区间逐节点向外 R32 舍入且输入预算覆盖梯度门 | 100×1e-9 全清零/深带全擦除/真实相消三反例；m=0 vs 1e-9、50% 状态差、单侧零增量反例；报告 loss/RNG 篡改与重复行被 F 拒；无快照 reservation 保持 null；过期 F 不 spawn；矩阵否决后磁盘 audit_incomplete；撤回成本≥实测；I failures 完整进入 F；端点包含检查；`norm==clip` 缩放到 0.999999046 |
| v0.6 / 4 P1 校准、撤回与截止、分组门 | 状态校准覆盖全部适用坐标并用门分母，`state_cross_path` 由 5e-4 调至 1e-3；pre-seal 撤回保存执行后累计快照并与 `consumed` 对账；P/I 本地截止无条件执行且主链带 `unit_admission`、I 在 reference context 前检查；输出 cotangent 加 p/rot6d/q 分组门并进入行裁决/F 行绑定；F 的 `reserve_s` 只扣一次 | 同 seed 校准 0.313871→0.156936（4×=0.628）；`prior_call_counts`=16 更新/64 forwards；受控时钟在 backward 前停止且无 backward 计数；joint 组清零被 `group_joint` 拒绝而整张量门通过；剩余 300s/预留 180s 仍 spawn |
| v0.7 / 3 P1 逐门适用性、分组深带与 I 截止单元 | 校准按各比较门自身适用性统计（precision-gap 行的已通过 m/v 仍计入），`state_cross_path` 由 1e-3 调至 2e-3、`delta_cross_path` 由 5e-4 调至 1e-3；p/rot6d/q 分组零噪声门要求参考与生产两侧范数均在 1e-10 深带；I 在输出 VJP/clip/参考更新前分别检查截止，三个更新计数改为实际执行后记录 | `cross_attn.v.weight` 0.289626（4×=1.1585>1）与 delta 0.297752（4×=1.191>1）进入统计并分别重定为 0.144813/0.148876；joint 参考 1e-13/生产 8.34e-7 被 `tiny_group_joint` 拒绝；受控时钟 clip=3、三个计数=3，输出 VJP 前停止时 `independent_output_vjp=0` |
| v0.8 / 2 P1 未舍入更新量参考与参考 FK 截止 | 跨路径更新量 close 门改用 P 实测 vs I 的**未舍入** `delta_ref64`（存在性比较 P/I 参考）；`delta_cross_path` 从基于已舍入实测的 1e-3 恢复为原提案 5e-4；`run_i_stage` 在 `ReferenceContext.evaluate` 前检查，`ReferenceContext` 在目标窗/evaluate/FK/Jacobian 前执行截止+`unit_admission` | 评审二步例已舍入 0.994504（错误通过）→未舍入 1.001830（sensitivity gap）；校准 5e-4 下 0.216102（4×=0.8644）；small-unobservable 仍 incomplete；受控时钟首 forward 后过期不再新增 480/3600 reference FK/jacSite，Jacobian 不启动 |

该表记录方案要求，不能当作实现/测试完成声明。v0.8 的 I/C 证据见 §10.1 与[验证记录](../verification/omniretarget_w1_real_training_update_audit_verification.md)；§7 预算、调用矩阵、尝试次数与16/32次更新额度保持原值，未建立真实 cohort、预算或执行注册。
