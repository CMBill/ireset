# W1 真实参数梯度与更新链路失败只读归因

日期：2026-11-04。承接[一次真实恢复报告](/home/c112/codes/bill/ireset/docs/verification/omniretarget_w1_real_training_update_binding_migration_report.md)。本轮用户授权工作为“进行下一步失败归因”，只分析已保存工件。

**329 项 failure 已逐项分账：156 项属于 attention key bias 的结构零梯度路径，另外 173 项全部集中在 diffusion 第 2 步。** 后一组的首个输出空间超差定位到 micro 0、slot 1 的 W0 第 0/6 帧，共 5 个 rot6d 坐标；这些帧的绝对根旋转接近 π，生产 FK 解码中的 trace-only 矩阵转四元数是首要待验证原因。现有工件缺少分项 cotangent 和 I 微批参数梯度原数组，尚不能完成该函数的独占因果证明。

原始数值结论保持 **`audit_incomplete`、329 failure / 629 gap、`covered_contract_passed=false`**。本报告解释拒绝原因，不重分类旧 failure 为通过；合成接受、F 重放一致和本轮分析核对通过均不代替数值通过。

## 1. 来源、范围和复核

真实运行根固定为 [20261102_2704709_w1_real_training_update_audit](/home/c112/codes/bill/ireset/derived/experiments/omniretarget_first/20261102_2704709_w1_real_training_update_audit)。选中 R0_002/P_002/I_001/F_001，原终检 SHA `ede2254f9212a34a50f1478e85bc98dc199db56dd719efff208d2886ad0d9799`，28 文件实现身份仍为 `0cc0622c86829b3e4bcbdefc1a3c1edb552ec02b31294dfd98bb70fd563205d8`。

使用 P/I 报告与 rows、原始 P/I 四份 NPZ、F 已保存的两份 replay rows、外层结果，以及父目标数组中的 `x0`。为解释 Gate B 覆盖差异，另读取当前冻结的普通合成 attempt_018 的 P 预测数组与 I rows。未使用旧 incomplete P_001 的部分数组。

分析程序只导入标准库和 NumPy；在保存的数组上做差值、范数、掩码统计和旋转特征的代数分析。checkpoint payload 读取、模型构造、forward、VJP/backward、FK、优化器与参数更新均为 **0**。分析墙钟 **12.6469 秒**，单列于归因工件，不回写真实执行台账；cohort 仍为 18 次更新、567.845409012 秒，本次一次真实授权已用完。

独立产物位于 [failure_attribution/attempt_001](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution/attempt_001)：

| 工件 | 内容 |
| --- | --- |
| [analysis.json](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution/attempt_001/analysis.json) | 汇总、34 项交叉核对、18 份来源的 SHA/bytes、分析边界；SHA `652d0025622edfee90573de5f1aa17786558155e5913f8bc4d250ab4ce4bf186` |
| [failure_index.json](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution/attempt_001/failure_index.json) | 全部 329 项原 failure、原 row、JSON pointer、参数步与微批关联；同目录另有 CSV |
| [gap_index.json](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution/attempt_001/gap_index.json) | 全部 629 项 gap、原状态门、同参数步梯度失败关联；同目录另有 CSV |
| [cotangent_localization.json](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution/attempt_001/cotangent_localization.json) | 16 个微批的坐标误差、最差帧、保存的 r6 与绝对/相对旋转特征 |
| [precision_details.json](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution/attempt_001/precision_details.json) | 381 个 precision 行的逐侧不可辨识坐标数及样例 |
| [interval_details.json](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution/attempt_001/interval_details.json) | 25 个 bound 行的既有区间与保存输出越界统计；u1 的 m/v 原分类掩码另有说明 |
| [key_bias_updates.json](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution/attempt_001/key_bias_updates.json) | 32 个 key bias 参数步的保存更新量与 P/I 差异 |
| [analyze.py](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution/attempt_001/analyze.py) | 可复核的独立分析脚本；拒绝覆盖已完成的本次归因目录 |
| [artifact_manifest.json](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/failure_attribution/attempt_001/artifact_manifest.json) | 上述分析产物与脚本的 SHA/bytes |

34 项交叉核对全部通过，`verification_problems=[]`。329/629 列表由逐行事实重建，与 I、F 及外层结果逐项一致；F 已保存 rows 与原 P/I rows 相等。16 个 cotangent 的数组 SHA 和最大门比值与原行一致；381 个 precision 行的保存掩码与冻结可辨识性条件一致。

18 份读取来源中，15 份与已有封口/报告/父锚摘要一致，包括重新流式核验的原始 P/I 四份及合成 P 两份 NPZ。另 3 份是外层结果和 F 两份 replay rows，记录实际摘要并以上述列表/行等值核对，不称它们已获新的独立封口。原根前后仍为 83 文件 / 7,835,408,541 bytes：路径、大小、mtime 及非 NPZ 文件摘要无变化。28 个实现文件前后逐一匹配原身份。本轮未重新执行整个历史验证器，也未重跑项目测试。

## 2. 失败拓扑

| 模型 / 更新步 | micro_param | accumulated | clipped | output_cotangent | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| deterministic / u1 | 18 | 6 | 6 | 0 | 30 |
| deterministic / u2 | 32 | 8 | 5 | 0 | 45 |
| diffusion / u1 | 23 | 7 | 6 | 0 | 36 |
| diffusion / u2 | 129 | 80 | 8 | 1 | 218 |
| 全部 | 202 | 101 | 25 | 1 | 329 |

329 个检查失败对应 **127 个不同的模型×更新步×参数组合**，另有 1 个 cotangent 行；不能当成 329 个独立实现缺陷。

| 归因分组 | micro_param | accumulated | clipped | cotangent | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| self/cross attention 的 `k.bias` | 105 | 29 | 22 | 0 | 156 |
| diffusion u2 的其余参数和输出 | 97 | 72 | 3 | 1 | 173 |

该分组按参数结构区分，不能解释为两组原因互不作用；diffusion u2 的 key bias 也可能受到该微批输出偏差影响。

128 个目标标量行全过，最大门比值为 0.007585；未报告 memory/prediction/noise/x_t 重放失败。528 个 P 自身累积检查、528 个 P 自身裁剪检查、4 个 global norm/coef 检查均过；528 个 AdamW 参数步的同输入公式门均过。**这将已观察到的问题定位到 P/I 梯度差异及其敏感性/可辨识性，不能解释为 AdamW 同输入实现错误，也不能据此宣布跨路径更新通过。**

## 3. attention key bias：结构零梯度与真实舍入残差

[Attn.forward](/home/c112/codes/bill/ireset/src/ireset/model/backbone.py:52) 对每个 key 使用同一个 bias。对于一个 query：

`q·(Wk·ctx_j + b)/sqrt(d) = q·Wk·ctx_j/sqrt(d) + q·b/sqrt(d)`。

右侧 bias 项对所有可见 key 相同，softmax 对整行常数平移不变，因此该参数的精确数学梯度为零；固定 keep-mask 不改变这个结论。两模型、四层、self/cross 的 16 个 key bias 参数都具有此结构。

实际 float32 softmax/backward 和归约存在消差残差。P 原生 backward 与 I 的独立 cotangent→参数 VJP 使用略有差别的 cotangent；在数学零空间附近，残差之间的相对 L2/cosine 差异可很大。

| 检查层级 | key bias failure | 最大逐坐标绝对差 | 逐坐标门失败数 | 触发原因 |
| --- | ---: | ---: | ---: | --- |
| micro_param | 105 | 1.310e-9 | 0 | 104 项 group 门、1 项 tiny 门 |
| accumulated | 29 | 3.522e-9 | 0 | group 门 |
| clipped | 22 | 4.147e-10 | 0 | group 门 |

例如 `deterministic:u1:m0:blocks.0.cross_attn.k.bias`：P/I 范数为 9.1239e-10 / 8.8020e-10，最大绝对差 5.0932e-11，逐坐标门比仅 0.00509，但相对 L2 为 0.23198、cosine 0.97469，故被冻结的 2e-4 相对门拒绝。

唯一 tiny 失败为 `deterministic:u1:m0:blocks.3.self_attn.k.bias`：I 范数 8.6014e-11，P 范数 1.0023e-10，刚越过双侧均须 ≤1e-10 的深噪声门；随后 1e-12 tiny 门拒绝。它不是缺失梯度，也不是单侧全零。

合成 attempt_018 的 128 个 key bias 微批比较全部使用零噪声豁免，I 范数仅 2.2731e-12–8.7618e-12；其 32 个累积行最大也仅 1.9424e-11。真实失败微批 I 范数最高 **9.6668e-8**，累积最高 **2.3812e-7**。因此现有合成噪声标定没有覆盖真实残差尺度。这解释了 Gate B 零 failure 与真实 key bias failure 并存。

**不能仅把这 156 项删掉，或统一上调零噪声阈值。** 已保存的 key bias 更新参考量最高达到 3.3367e-5；P/I 参数写回最大差 **3.37698e-6**，出现在 deterministic u1 的第 0 层 cross key bias。数学零梯度的有限精度残差已进入 AdamW 状态和可见参数更新，不能直接称为“无影响噪声”。后续必须同时解释梯度残差的合理边界与优化器影响。

## 4. diffusion u2：近 π 的绝对根旋转与局部 cotangent 超差

唯一输出行失败为 `cotangent:diffusion:u2:m0`。P/I 预测逐位重放通过，但保存的总 cotangent 有 5 个坐标超过冻结逐坐标门 `2e-8 + 2e-4*abs(I)`。以下索引均从 0 开始，slot 1 是该 micro 中第二个 W0，不能与 slot 0 的 W0 合并。

| slot / 窗口帧 / x0 通道 | P | I | 门比值 |
| --- | ---: | ---: | ---: |
| 1 / 0 / 4 | 3.6637875019e-4 | 3.6424541826e-4 | 22.97634 |
| 1 / 0 / 5 | 7.4823875912e-4 | 7.4804364715e-4 | 1.15037 |
| 1 / 0 / 7 | -9.0401990747e-6 | -9.0749755688e-6 | 1.59416 |
| 1 / 0 / 8 | -8.6469874077e-5 | -8.6251869996e-5 | 5.85240 |
| 1 / 6 / 4 | 4.5332513400e-4 | 4.5347416065e-4 | 1.34628 |

所有超差都在 rot6d 分组。该微批整体相对 L2 为 9.6551e-5，三个分组门也全部通过（最大归一化组门比 0.62850）；这些整体指标不能豁免局部 22.98 倍超差。

第 0 帧占该 micro 总 cotangent 差的平方范数 **99.4280%**，第 6 帧再占 **0.49124%**。对保存的 r6 仅作 float64 代数分析得到：

| 特征 | W0 第 0 帧 | W0 第 6 帧 |
| --- | ---: | ---: |
| `1 + trace(R_root)` | 3.11943e-6 | 2.46973e-5 |
| trace 分支得到的 `abs(qw)` | 8.83095e-4 | 2.48482e-3 |
| 绝对根角距 π | 0.00176619 rad | 0.00496964 rad |
| 相对目标旋转角 | 0.404858 rad | 0.468295 rad |
| 6D 第一列范数 / 正交后第二列范数 | 0.9520 / 1.0204 | 1.0634 / 1.0312 |

这些点的 6D 向量远离退化线，相对目标旋转也不接近 π。**接近 π 的是 FK 根姿态的绝对旋转，不能与 geodesic 损失的相对旋转奇点混淆。**

[LossAssembler.decode_clip](/home/c112/codes/bill/ireset/src/ireset/train/losses.py:69) 的生产 FK 路径为 `rot6d → R → mat_to_quat_torch → quat_apply → FK`；[矩阵转四元数实现](/home/c112/codes/bill/ireset/src/ireset/model/tokens.py:174) 始终使用 `qw=sqrt(1+trace(R))/2`，再除以 `4*qw` 并归一化。此时 `1+trace(R)` 很小，trace 消差和反向传播的条件变差，末端归一化只保证值有限，不保证导数稳定。I 参考使用 float64 直接 R/切空间解码和 MuJoCo 雅可比，绕过该 trace-only quaternion 链。

当前合成 attempt_018 的最小 `abs(qw)` 为 deterministic 0.99827 / diffusion 0.99723，完全没有覆盖上述绝对近 π 输入。这与 key bias 残差尺度共同构成已证实的合成覆盖缺口。

173 项非 key bias 失败的分布是：cotangent 1；micro 0 的 96 个参数比较；micro 2 的 `root_out.weight` 1 项；累积 72；裁剪 3。三个非 key bias 裁剪失败参数为 `root_in.weight`、`blocks.2.cross_attn.k.weight`、`root_out.weight`，门比分别为 1.10001、1.00838、4.50087。

数值链路和集中分布支持“局部输出偏差经参数 VJP、累积和裁剪传播”的解释，但**尚未测得各分项的因果贡献**。尤其 micro 2 的输出 cotangent 已通过（最大门比 0.34445），`root_out.weight` 参数门仍失败（3.28446，相对 L2 仅 3.0941e-7），应单独保留为局部消差/映射敏感性待查项，不能声称修复 micro 0 就已解释全部 173 项。

## 5. 629 项 gap 的关联与独立限制

“同参数步有梯度 failure”仅按模型、更新步、参数对应，不是独立因果证明；一个 AdamW 行可同时产生两种 gap。

| gap | 总数 | 同参数步有梯度 failure | 未检测到同参数步梯度 failure |
| --- | ---: | ---: | ---: |
| parameter_delta_precision_gap | 381 | 125 | 256 |
| cross_path_sensitivity_gap | 223 | 67 | 156 |
| cross_path_bound_unverifiable | 25 | 25 | 0 |
| 条目合计 | 629 | 217 | 412 |

**25 个 bound 行与 25 个 clipped 失败参数步集合完全相同**：22 个 key bias、3 个上一节的非 key bias。原区间包围的是允许梯度误差集合；这些点已未通过梯度输入门，其输出超出允许集合传播区间不能另算作“AdamW 同输入公式失败”。仍保留原 `bound_unverifiable`，不把区间不可证转为通过。

381 个 precision 行都能从保存的 `obs_p/obs_i`、未舍入参考增量和写回界验证存在非零不可辨识坐标。逐侧累计 P/I 不可辨识坐标分别为 1,222,256 / 1,222,249（跨参数步累计）；其中实测更新恰为 0 的坐标只有 351 / 359。**不可辨识不只表示“更新被舍入成零”，也包括更新已改变参数，但写回误差上界仍不足以支持冻结相对精度声明。**

例如 deterministic u1 的 `root_in.bias[8]`：保存的 P `delta_ref=-9.99975699e-5`、`delta_measured=-9.99867916e-5`、半 ULP 写回界 `1.49011612e-8`。冻结可辨识条件要求 `bound <= 0.25 * 5e-4 * abs(delta_ref)`，右侧约 `1.24997e-8`，因此即便该行原始跨路径门全部通过，也必须记 gap。381 行中共有 **176 行原始跨路径门全部通过**，这 176 个 AdamW 行仅有精度可辨识性 gap；不撤销其上游可能已有的梯度 failure。

223 个 sensitivity gap 仍有 156 个未检测到同参数步梯度 failure。它们说明当前已允许的梯度差异及 AdamW/写回敏感性不足以支撑更严格的跨路径声明；并不因为修复某一处梯度超差就自动消失。本轮未重新计算有效优化器图或改写这些分类。

## 6. 最小后续修复与验证顺序（尚未执行）

| 顺序 | 具体工作 | 必须交付的证据 |
| --- | --- | --- |
| 1 | 针对保存的 W0 第 0/6 帧隔离 FK 与 geodesic cotangent；比较现有 trace-only 转换、稳定转换及直接 R 路径 | 分项误差能定量解释保存的总差；确认生产与参考哪一环节偏离，而非只凭近 π 相关性定案 |
| 2 | 若确认 quaternion 转换为原因，优先替换该适配函数的病态分支，保持 wxyz、归一化、dtype/device 和现有入口 | 单位旋转、精确/两侧接近 π、一般旋转、分支交界与真实保存点的值和梯度反例；原正常路径回归 |
| 3 | 为 attention key bias 单独建立数学零梯度及实际 AdamW 影响验证，再选择数值稳定实现或专门的结构零协议 | 不能按参数名直接忽略、删除 roster 或统一放宽门；保留非零梯度擦除、presence、符号/单侧零反例，并检查 checkpoint/严格加载兼容性 |
| 4 | 单独处理 micro 2 `root_out.weight` 和无对应梯度 failure 的 412 个 gap 条目，修正失败行状态展示 | 逐项列出仍不可辨识/敏感的原因；`failure/comparison.ok` 与状态一致；不承诺所有 gap 清零 |

已安装依赖静态提供 `pytorch_kinematics.transforms.matrix_to_quaternion`，源码采用四个候选分支并选择最大分母，可作为顺序 2 的成熟库候选。本轮只读过 API/源码，未运行其数值或 backward；精确 π/单位旋转时未选分支的平方根及梯度仍须专门验证，不能将“已有函数”当作适配已通过。

后续受控验证应另列“保存输入的局部输出梯度/小型 attention 反例”范围和预算，先保持真实模型加载与训练更新为 0。本轮未执行上述新梯度重算，也未实施代码修复。若实现或协议改变，必须产生新版本证据、完成相应回归与 Gate B/M5 绑定；原 M6 授权不能授权新的真实 R0/P/I/F、重试或额外更新。

## 7. 归因边界

已证实的是失败分账、结构零梯度恒等式、失败触发门、保存输出的局部超差、合成输入覆盖缺口，以及 gap 与已失败梯度/写回可辨识性的关系。trace-only quaternion 链是强定位候选，尚缺分项 VJP/替代路径对照；没有保存的 I 微批梯度原数组，无法离线恢复精确失败坐标或重新路由 cotangent。原 P/I/F、commit、返回证明、容差和冻结方案全部保持原字节。

附带发现：202 个已失败的 micro 参数行仍写 `status=synthetic_verified`，而 `failure=micro_parameter`、`comparison.ok=false`。原因是 [I 行构造](/home/c112/codes/bill/ireset/src/ireset/eval/w1_real_training_update_audit.py:3601) 比较失败后没有更新 status。正式 failure 列表和 `numeric_passed=false` 已正确拒绝，故不改变本次失败数；归因程序按 `failure/comparison.ok` 读取，未受展示标签误导。该展示修复列入后续工作，本轮未改冻结实现。

`objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false` 和既有质量结论保持。

## 8. 后续局部分项验证（2026-11-04）

用户随后指令“继续”已完成本报告§6的保存输入局部验证，详见[独立验证记录](/home/c112/codes/bill/ireset/docs/verification/omniretarget_w1_real_training_update_local_gradient_verification.md)。原cotangent逐位复现，5个输出超差主要由trace-only quaternion FK导致；成熟库归一化候选使原5坐标最大门比降至0.00140329。key bias的存在零梯度/None区别和已有AdamW状态影响也已验证。[最小修复方案](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_gradient_numerics_repair_plan.md)已制定，尚未修改生产实现。

该后续轮包含新的局部输出/随机attention梯度和运动学FK计算，独立合计10.3349s、57 autograd/1444状态；不得沿用本报告§1针对只读分析轮的“新梯度/FK为0”表述。训练网络/checkpoint/训练更新仍为0，原工件及329/629结论保持；参数级传播与其余gap未解决。
