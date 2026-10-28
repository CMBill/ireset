# W1 真实训练解码、FK 几何与输出空间导数审计预注册草案

版本：v0.1，2026-10-28。状态：**`draft / implemented_synthetic / not_registered / not_authorized / not_run`**。

本次交付包含方案文档与 I/C 纯合成实现、校准和准入包。下列真实预算、调用额度、阈值和新 cohort 仍是待审阅提案，尚未分配或执行真实 R0→G→D→F；真实数组、锁定 G1 资产、MuJoCo 场景和 checkpoint 均未读取或编译。冻结校准契约与准入包见 §10。

上级证据：[真实批次目标与输出空间梯度审计报告](../verification/omniretarget_w1_real_objective_audit_report.md)。边界来源：[已执行的真实目标预注册](omniretarget_w1_real_objective_audit_prereg_draft.md)、[训练目标合成审计方案](omniretarget_w1_training_fit_objective_audit_plan.md)。旧方案与已封口产物保持原字节。

## 0. 立项依据与唯一问题

上一轮真实 R0→P→F 在四个固定窗口上得到 `real_batch_covered_contract_passed`：98 目标行、14 梯度行完整，五项解析损失、`L5`、FK 聚合算术及其已覆盖的输出空间梯度与参考一致。该结果没有验证真实 FK 几何或 FK 项经过解码的导数，不能据此认定完整目标、训练更新或最终动作质量通过。

本方案只回答：**对已封口的真实 target 和两模型保存预测，生产的“38 维归一化输入→训练解码→15 点 FK→FK 损失输出梯度”是否与独立几何及链式求导一致？**

| 层次 | 本方案覆盖 | 仍未覆盖 |
| --- | --- | --- |
| 解码 | 训练用 clip 系根平移、6D 旋转到姿态、29 关节 clamp/反归一化 | 推理世界系重建、窗口融合、轨迹修复 |
| FK 几何 | 锁定 G1 的 URDF/Torch 与 MJCF/MuJoCo 的 15 点坐标 | 资产对真机的几何真实性、碰撞/接触质量、动力学 |
| 导数 | FK 标量损失对保存预测 38 维输入的完整梯度；独立参考的 35/38 维 Jacobian 固定抽样 FD | 生产 FK 的全部逐输出 Jacobian、网络参数梯度、优化器与训练更新 |
| 归因 | 定位到解码、FK 核、坐标/次序、归约或 FK 梯度的实现一致性 | 端到端失败责任百分比、唯一主因、单因素干预效果 |

即使通过，仍保留 `objective_coverage=partial`、`full_objective_contract_passed=null`、`attribution=attribution_insufficient`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false`。不触发自动续训、教师替换、数据扩张或融合接入。

## 1. 来源、身份与固定输入

### 1.1 父证据绑定

唯一父运行根为 `derived/experiments/omniretarget_first/20261027_b45cd7c_w1_real_objective_audit/`。以下路径均相对此根；哈希引自真实审计报告，R0 重新核对文件和生产者登记。

| 文件 | SHA-256 |
| --- | --- |
| `registration/attempt_001/registration.json` | `ecc11368219a33c95a6038769398fd11789dcdd04430760f54ae3a84ba58a66a` |
| `registration/attempt_001/admission.json` | `61c5b16d883d0fa084632a548bd7428e2fe580540990feed47926546bd32ce95` |
| `production/attempt_001/p_report.json` | `1ea15ddc2da7a0acb5125f1ecc92e1ea7e0cc74589e7a7ad5055b26055b7cdcb` |
| `production/attempt_001/objective_real_arrays.npz` | `4cf20acb13fcdee207e0492a3b07818cc9ed31202409177fa1ed3f13086fd2c8` |
| `production/attempt_001/gradient_real_arrays.npz` | `49b21c6c5d82a19e92f2c0d870b1d7f036986604c009c6195e6f86a035765ea7` |
| `finalize/attempt_001/audit_summary.json` | `0a839071125a156273b3d579a564965cd0edb196b8f900b05417edd0cb00db97` |
| `finalize/attempt_001/independent_verification.json` | `f0bf351bd0bd184f52198ac481293bae9127607ec98864010768430a4dcbd961` |

同时绑定父 `window_keys`、原始目标/梯度行、call trace、索引与台账，核验父 R0/P/F 均可消费，不能仅凭 summary 的通过声明准入。两份 step-4000 checkpoint 的身份仅作为父来源保留，本方案不重新加载 checkpoint，不重新生成预测。

父 cohort `w1_real_objective_audit_v1` 的四类调用均已使用 4/4。本方案不追加该 cohort 的调用，也不转移其剩余时间。

新问题拟用固定 cohort `w1_real_fk_geometry_derivative_audit_v1`。身份由本问题、上述父注册/数组、固定窗口与检查矩阵共同确定；改 run 名、父目录、实现 SHA、方案哈希或阈值文件不重置 cohort。新身份的理由是新增真实解码/FK 几何及 FK 导数问题，不是延长旧模型调用额度。

### 1.2 固定四窗与数组

| 窗键 | demo | 原长 T | start | end（不含） |
| --- | --- | ---: | ---: | ---: |
| W0 | `sub10_largebox_063` | 133 | 0 | 60 |
| W1 | `sub10_largebox_063` | 133 | 30 | 90 |
| W2 | `sub10_largebox_085` | 148 | 0 | 60 |
| W3 | `sub10_largebox_085` | 148 | 30 | 90 |

顺序固定为 W0–W3。两个 demo 均属 `sub10`，不声明跨受试者覆盖。重叠帧保留其窗口键，不去重、不换窗、不追加困难帧。

| 来源键 | 必需类型与形状 | 用途 |
| --- | --- | --- |
| `x0` | float32 `[4,60,38]` | 生产 target；独立几何也从其原位模式提升到 float64 |
| `prediction_deterministic` / `prediction_diffusion` | float32，各 `[4,60,38]` | 两模型保存预测，复制为 leaf 后只求输入梯度 |
| `root_mean/root_std` | float64，各 `[3]` | 根统计来源；核验 std 正且有限 |
| `joint_lo/joint_hi` | float64，各 `[29]` | 关节顺序与范围来源；逐项 lo < hi |
| `x0_independent`、`raw_root/raw_q/raw_wfc` | 父 schema；分别 `[4,60,38]`、`[4,60,7]`、`[4,60,29]`、`[4,4,4]` | 来源和编码关系核验，不替代生产 target |
| `fk_points_pred_<model>` / `fk_points_target_<model>` | float32，各 `[4,60,15,3]` | 新生产路径与父实测点逐位绑定 |
| `term_elements_<model>_fk` | float64 `[4,60,15]` | 父聚合证据重读；不作为新参考真值 |

其余父必需键由绑定的父 schema 核验。所有数值数组先检查 dtype、精确形状、非空与有限性；布尔、对象或字符串数组不进入数值比较。

逻辑输入为 target、det、diff 三组，共 `3×4×60=720` 个状态槽。实际 LossAssembler 对每个模型各算一次 target，因此生产 FK 有四个角色 `target_det/target_diff/pred_det/pred_diff`，共 960 个状态槽；两份 target 都须绑定和相互核验，不把重复计算隐去。

### 1.3 锁定资产、配置和环境

R0 哈希绑定 `g1_29dof_locked.urdf`、`g1_29dof_locked.xml`、`g1_29dof_locked_w_largebox.xml`、`lock_record.json` 及所有引用 mesh/include 闭包。配置沿用父已冻结有效权重：`w_fk=2.0`、`w_contact=0.0`，其余权重与父完全相同。

生产路径固定 CUDA/float32、TF32 关闭、autocast 关闭、matmul precision `highest`、`deterministic_algorithms=false`；独立 NumPy/SciPy/MuJoCo 及 FK 核 float64 诊断固定 CPU/float64，CPU 数值线程固定为 1，实质单元顺序执行。注册 Python、uv.lock、Torch/CUDA、MuJoCo、SciPy、pytorch_kinematics、GPU/驱动和全部实现文件身份。环境不满足冻结条件时停止，不能静默改 dtype、开启确定性或放宽重放要求。

## 2. 几何、坐标与独立参考契约

### 2.1 生产路径必须原样经过训练解码

复用 [LossAssembler](../../src/ireset/train/losses.py) 的既有 observer。两模型各一次，以保存预测 leaf 和父 batch 运行，保留 live `fk_m2`、解码结果、15 点 FK、逐点平方残差；目标保持 no-grad。只对 `fk_m2` 求一次输入 VJP，不反向 `L5` 或 total，不构造学习模型或优化器。

根统计与 lo/hi 按真实构造器先取 float32；独立参考将这些**实际生效的 float32 常数**提升到 float64，避免将常数转换误差算成 FK 错误。原始 float64 统计另作来源对账。

实际训练解码为：

```text
p_clip = x[0:3] * std32 + mean32
R = GramSchmidt_columns(x[3:6], x[6:9])
quat = production_mat_to_quat(R)
q = lo32 + (clamp(x[9:38], -1, 1) + 1) * (hi32 - lo32) / 2
```

同时保存生产 R（由实际输出 quat 重建）与独立 R，比较旋转矩阵而非四元数符号。不得以推理 tokenizer 的世界系解码替代训练 `decode_clip`，也不得仅比较 FK 前的 6D 矩阵而漏掉实际矩阵→四元数路径。

### 2.2 独立几何

独立解码用 NumPy float64 的两列 Gram–Schmidt、SciPy Rotation 的稳定矩阵/四元数转换，显式执行 xyzw→wxyz；不调用生产 Torch 旋转函数。将解码后的 clip 根直接作为 MuJoCo 临时世界系根；两侧都输出 clip 坐标，不向单侧额外施加 `raw_wfc`。

MuJoCo 使用锁定 plain MJCF，按编译后的 joint/site 名称解析地址，不复制 Torch 的 site offset 表来构造参考。核验 `nq=36/nv=35`、29 关节名字/顺序/qpos 地址/限位、15 site 名字/所属 body/局部位姿及资产闭包。共同资产的语义错误仍可能共同存在，本审计只证明所锁资产间的一致性。

15 点顺序冻结为以下名称（均带 `kpt_` 前缀）：

```text
pelvis, l_hip, r_hip, l_knee, r_knee,
l_shoulder, r_shoulder, l_elbow, r_elbow,
l_ankle, r_ankle, l_toe, r_toe, l_wrist, r_wrist
```

另以独立解码的 target/det/diff 三组物理姿态运行 CPU/float64 `G1TorchFK`，仅用于隔离 FK 核差异；此项是固定必测诊断，不把 float64 核结果冒充真实 float32 训练路径。

### 2.3 FK 标量与梯度

对每模型固定 `B=4,F=60,S=15`：

```text
e[b,f,s] = kp_pred[b,f,s] - kp_target[b,f,s]
L_fk = sum_{b,f,s} ||e[b,f,s]||² / 3600
L_fk_weighted = 2 * L_fk
```

分母是 `B×F×S=3600`，不是再乘坐标维数 3。逐窗损失分母为 900，batch 必须等于四窗均值；逐点元素、窗口、batch 和权重都重新计算。

独立参考按以下链式法则构造，不使用生产 autograd/FK 图：

```text
xi = [dp_clip(3), dtheta_left_clip(3), dq(29)]                 # 35D
J35 = J_mujoco_nv @ blockdiag(I3, R.T, I29)                  # [15,3,35]
D_decode = d xi / d x                                      # [35,38]
J38 = J35 @ D_decode                                       # [15,3,38]
g_fk[b,f] = (2/3600) * sum_s J38[b,f,s].T @ e[b,f,s]
g_fk_weighted = 2 * g_fk
```

MuJoCo freejoint 角速度为 body/local 表达，注册切空间为左乘 clip 轴表达，因此角块须乘 `R.T`。物理姿态增量使用 `R_next=Exp(dtheta)@R`，不是四元数存储分量增量。

`D_decode` 由独立解析微分构造：平移块 `diag(std32)`；6D 块使用归一化导数 `(I-uu.T)/||a||`、Gram–Schmidt 与叉积微分，再取 `vee(dR@R.T)`；关节在 clamp 内为 `(hi32-lo32)/2`，光滑饱和区为 0。不得直接调用生产旋转函数或用生产梯度作参考。

每模型一次 production VJP 得到完整 `[4,60,38]` 原始梯度。加权梯度由冻结系数 2 推导，是权重算术核验，不宣称额外执行了一次 weighted backward。逐窗梯度使用 **batch 梯度切片**，不得偷偷改成 1/900 的局部损失梯度。FD 验证独立 J35/J38；生产侧在此只审计 FK 损失梯度收缩，不宣称完整生产 Jacobian 已逐元素验证。

## 3. 数值域、精度与通过门

### 3.1 不可微或病态输入

事前固定 6D 正常域为 `||a||>1e-6`、`||b_perp||>1e-6`，且 `||a×b_perp||>1e-12`。不足、非有限或无法构造有效 SO(3) 时记录具体键、范数和 `domain_gap`，不生成零矩阵占位，不换帧。

关节输入严格处于 `(-1,1)` 或严格位于其外时均有确定的局部导数；饱和区零导数是必测结果。恰为 ±1 时不存在唯一双侧导数，相关必测导数记录 gap。FD 任一步长跨越 clamp 分段或落出冻结旋转域时，该行记录 `stencil_domain_gap`，不改用单边 FD、不缩小步长挑选通过结果；几何和其他可定义坐标仍测量并保存。

近 π 的有效 6D 旋转本身不因此获得豁免。生产 trace-only 矩阵→四元数分支的 floor/条件数作为事实登记；相对稳定参考的几何或梯度偏离正常记 mismatch。既有近 π 精度负门保留，不能靠扩容差、删除点或声明算法拒收来获得通过。

### 3.2 容差提案与冻结校准

下表为待校准的**上界提案**。正式实现先在人工解析小链、非退化旋转、clamp 内外/边界、零残差与非零残差上做纯合成校准；不加载真实预测、G1 资产或 checkpoint。保存输入、版本、全部观测误差和证据哈希，独立重推并复跑。`4× observed` 取已登记正向域用例，不能超过对应提案上界；超过则校准不通过，须在真实读取前修订方案并重新审阅，不能自动抬高阈值。近 π 和不可微等已登记负例另表保留，须维持预期失败/域缺口，不纳入放宽阈值的观测。校准后冻结完整数值契约及哈希，不沿用父目标审计的容差文件代替本项校准。

| 检查 | 绝对容差 | 相对容差 | 附加条件 |
| --- | ---: | ---: | --- |
| float32 解码 p/q | `2e-6` m/rad | `2e-6` | 独立参考使用生效常数 |
| 实际解码旋转矩阵 | `2e-5` | `0` | 另报相对旋转角，不替代矩阵门 |
| CPU/float64 FK 核 vs MuJoCo | `1e-6` m | `1e-6` | 全部 site 坐标逐元素 |
| 生产 float32 FK vs MuJoCo | `5e-6` m | `2e-6` | 不以宏平均覆盖局部失败 |
| FK 标量/逐点元素 | `1e-8` m² | `2e-5` | 元素、分母、窗口和 batch 均检查 |
| J35 vs 双侧 FD | `2e-6` | `2e-5` | 两个固定步长都检查 |
| J38 vs 双侧 FD | `1e-5` | `1e-4` | 两个固定步长都检查 |
| FK 原始/加权输入梯度 | `1e-8` | `1e-4` | 同时满足下述分组门 |
| 父生产 FK 绑定、F 生产重放 | 逐位 | 不适用 | 输入/生效常数/输出/梯度按 dtype+shape+原位模式 |

数值门均先校验类型、形状、非空、有限性，再比较 `abs(prod-ref) <= atol + rtol*abs(ref)`。布尔值不充当数值，计数精确比较。对每模型/窗及 batch 的 p、rot6d、joint 梯度组，参考 L2 范数 ≥ `1e-10` 时还要求相对 L2 误差 ≤ `2e-4`、cosine ≥ `1-1e-6`，且生产组不得为全零；参考组更小时要求生产/参考最大绝对差 ≤ `1e-12`。clamp 饱和坐标的零梯度也按 `1e-12` 检查，不能被普通 atol 吞掉。

通过须所有固定必测元素/组均通过，任何必测域缺口阻止 covered-pass。F 的生产逐位重放失败登记 mismatch，不临时改为数值近似。新参考及 FD 在冻结容差内重现；JSON 明细、计数、域掩码、分母与派生算术另作完整对账。

## 4. 实现顺序与正式阶段

### 4.1 I/C：真实授权前的工程工作

拟新增 `src/ireset/eval/w1_real_fk_audit.py`、`scripts/audit_w1_real_fk.py`、`tests/test_w1_real_fk_audit.py`。优先复用现有成熟 FK/旋转库、observer 和已审计账本原语，只增加最小适配，不改生产数值路径。

I 完成静态契约、独立公式、固定键生成器、schema、调用计数与封口协议；C 完成纯合成正向链、反例、冻结校准。默认 CLI 只能纯合成；真实模式缺授权或准入包时，在读取真实数组、加载资产、创建正式运行根之前拒绝。I/C 的产物另存 `derived/audit/w1_real_fk_audit/`，不能计作正式 G/D/F 证据。

### 4.2 R0→G→D→F

| 阶段 | 固定工作 | 进入下一阶段条件 |
| --- | --- | --- |
| R0 | 事前登记；授权/校准/历史/父消费者状态与哈希；四窗/schema/生效常数/域；资产闭包、实现与环境冻结；不执行 FK 或模型 | R0 可消费、绑定通过；缺域信息如实保留，不能伪造通过 |
| G | 两模型生产 LossAssembler 与 FK 输入梯度；720 状态独立几何、480 预测状态 J35/D_decode/J38；固定 float64 核诊断；父 FK 逐位绑定 | 绑定有效、预算与封口已知，且 D 所需数组存在；纯数值 mismatch 不阻止已注册矩阵继续定位 |
| D | 固定 32 状态的 35/38 维、两步长、双侧 FD；重算解析参考比较 | 全键均登记，缺口/失败分别保留；不得追加单元 |
| F | 新进程、新后端，从磁盘重读父/G/D；独立生成键集、重算 G、重跑全部 D、核对所有数组/明细/门/调用与成本，最后封口 | 按 §8 裁决，不信任保存的 passed/counts/expected_reasons |

R0 不编译运动学后端，nq/nv 和编译后地址/site 位姿等由 G 的第一个已登记单元在计算前核验。G/F 的生产构造器调用 `largebox_static_frame()` 时会首次编译场景并执行一次 `mj_forward`；此初始化单列计数，不说成接触损失分支执行。接触监督/SDF 分支仍必须为 0。

F 可在预算内重放已封口的完整数值负结果；遇到 R0/来源失配、G/D 预算撤销、running/publishing、未知历史或封口不可消费时，仅整理已有证据，**不构造后端、不新增 FK/VJP/FD**。任一 real unit 前再次核验父 attempt 状态、注册/路径/哈希、剩余额度与 `now+reserve < deadline`。

### 4.3 G 与 F 的固定比较行集

| 行族 | 固定键构造 | 行数 |
| --- | --- | ---: |
| decode | 4 生产角色 × 4 窗 × p/R/q | 48 |
| production_fk | 4 角色 × 4 窗 | 16 |
| kernel64_fk | 3 逻辑输入 × 4 窗 | 12 |
| parent_fk_binding | 4 角色 × 4 窗 | 16 |
| fk_scalar | 2 模型 ×（4 窗 + batch） | 10 |
| fk_gradient | 2 模型 × raw/weighted ×（4 窗 + batch） | 20 |
| 合计 | 完整数组逐元素检查，行内保留 p/rot6d/joint 组统计与全部判据 | **122** |

每行保存精确键、状态、数组引用/哈希、dtype/shape、单位、分母、误差与失败元素数、域信息、容差来源和判据。两份 target 的逐位相等是 target 绑定行的必测字段。F 不从 G 自报行数生成 roster；新增/缺失/重复/错窗/错角色、空数组、明细不符都阻止通过。

## 5. 固定有限差分矩阵

FD 只取两模型保存预测，不取 target。每模型四窗各取局部帧 **`[0,29,30,59]`**，即 `2×4×4=32` 个状态。每个窗仍保留独立键；绝对帧为 `start+local_frame`，重叠绝对帧不去重。

固定步长均为 **`[1e-5,1e-6]`**。物理平移轴单位 m，旋转/关节轴单位 rad，归一化轴无量纲；不依据真实误差挑 h。

| FD 族 | 输入增量 | 每行比较 | 行数 | ± 参考姿态求值 |
| --- | --- | --- | ---: | ---: |
| tangent35 | 3 平移 + 3 左乘旋转 + 29 关节基向量 | 15×3 坐标差分 vs J35 对应列 | `32×35×2=2240` | 4480 |
| normalized38 | 38 个归一化坐标基向量，经独立解码 | 15×3 坐标差分 vs J38 对应列 | `32×38×2=2432` | 4864 |
| 合计 | 固定两个 h 都核验 | 不取两个误差中的较小值 | **4672** | **9344** |

MuJoCo 解析 Jacobian 与 FD 的扰动构造分开：FD 通过 NumPy/SciPy 的显式 `Exp(dtheta)@R` 与独立解码构造姿态，不调用生产 retraction/解码 helper。保存正负扰动输入、坐标、差分向量、解析列、h 与域判定；F 重新生成和执行全部 stencil，而非只比较保存的汇总误差。

这是**状态/输入 FD**；未来授权须显式允许该固定矩阵，`parameter_fd=0` 不等于全部 FD 为 0。缺口行仍保留固定键和无法求值原因，调用数可小于名义值，但不能发布为 measured/pass；不得用被跳过的份额扩展其他样本。

## 6. 累计预算、尝试与调用上限提案

### 6.1 时间和尝试

固定历史根为 `derived/experiments/omniretarget_first/`，本 cohort 的运行根必须是其直接子目录；不开放改根参数。

| 阶段 | 同 cohort 累计上限 | 其中收口预留 | 累计尝试上限 |
| --- | ---: | ---: | ---: |
| R0 | 180s | 30s | 2 |
| G | 420s | 60s | 2 |
| D | 600s | 90s | 2 |
| F | 600s | 90s | 2 |
| 合计 | **1800s（30 分钟）** | 已包含在阶段内 | 各自独立 |

这是上限提案，不是运行时间预测；本轮不跑真实 pilot 估时。F 必须在自身 600s 内重放 G+D，不能借用 G/D 剩余时间。入口检查历史累计成本/尝试、剩余工作额度和收口预留；无法承担则不启动，保留未完成事实，不换 cohort。

计时覆盖启动/import、读盘/哈希、资产编译、构造、正反向、CPU/GPU 同步、传输、重放、异常/撤回/索引台账与最终封口。真实预检、额外 post-check 中的数值重算也须事前计费；F 后只做文件哈希/文档维护不追加数值计算。

### 6.2 名义工作量和 cohort 硬上限

下表每个 G/F 只运行一个工作进程；F 在同一新进程执行 G+D 重放。D 中 MuJoCo 后端无额外零位姿预热/抽样。名义值假定全部输入在域内，无失败重试。

| 计数器 | G 一次 | D 一次 | F 一次 | 同 cohort 累计硬上限 |
| --- | ---: | ---: | ---: | ---: |
| LossAssembler 调用 | 2 | 0 | 2 | 8 |
| 生产 float32 FK 批调用 | 4 | 0 | 4 | 16 |
| CPU/float64 FK 核批调用 | 3 | 0 | 3 | 12 |
| production FK 输入 VJP | 2 | 0 | 2 | 8 |
| MuJoCo 参考 `mj_forward` | 1200 | 9344 | 10544 | 42176 |
| `mj_jacSite` | 7200 | 0 | 7200 | 28800 |
| 其中 FD 姿态求值 | 0 | 9344 | 9344 | 37376 |
| scene setup `mj_forward` | 1 | 0 | 1 | 4 |

G 的 1200 次参考 forward = 720 个逻辑状态点坐标 + 480 个预测状态 Jacobian；`480×15=7200` 次 jacSite。每个 FK 批固定 240 个状态、15 点，不能增加 batch 大小绕过批调用计数。累计硬上限按对应阶段最多两次制定：**每阶段自身另受 `2×本阶段名义值` 限制**，不是允许一个 stage 用尽其他 stage 的余量。scene setup 与参考 forward 分开记，MuJoCo 实际 forward 总数为两者之和。

实际调用处包装计数，分别保存 per-attempt delta、stage/cumulative 快照、输入工作量与 call trace；不以函数名或固定常数冒充实际执行次数。批次/FD 状态单元在调用前持久化预留其完整工作上界，中断后未证实未使用的额度不返还。下一个 attempt 的额度从历史单调延续，不因新进程归零。

以下全链计数必须是 JSON 整数 **0**：`checkpoint_loads`、`conditioner_calls`、`model_calls`、`parameter_backward`、`parameter_fd`、`optimizer_steps`、`training_updates`、`ddim_calls`、`checkpoint_writes`、`contact_branch_executions`。正常可执行本方案不需要任何额外网络调用额度。

## 7. 事前登记与封口协议

沿用已审计语义，实现前以 `ae6a0fc` 的归属修复为基线核对，不能只复制旧版本 helper：

1. 每个正式 stage 在实质读取/计算前，登记 cohort、运行身份、授权/校准哈希、父 attempt、输出目录、截止时间、调用预留，并写 running/complete=false 索引及台账。R0 事前门仅可读授权、校准、历史与文件身份信息；读取真实数组必须已登记。
2. 固定根历史先证归属再证完整，index/ledger 分半读取。任何可读成员行或标记已证明归属而另一半损坏即 unknown；完整外部 cohort 跳过且不计成本/次数/调用；有标记且半损坏无法排除成员身份时仍 unknown；无归属证据的 legacy 按已有语义跳过。
3. 盘×索引×台账逐 attempt 对账身份、状态、报告哈希、有限非负成本和精确数值；累计快照与增量对账。遗漏尝试、矛盾、running/publishing、负成本或无法证明剩余额度即 unknown=停止，不由总时限尚有余额豁免。
4. 每个 unit 的目录、scope、注册哈希和截止时间必须属于 active running 父 attempt；子进程截止不晚于父，调用预留不能超出 stage 与 cohort 余额。只有前置消费者状态、绑定与实时截止均通过才构造后端或开始新单元。
5. 终态先通过不可消费的 publishing 发布；登记成本包括自身写盘与保守尾部预留，并执行封口后覆盖核验。成本只增不减；撤回也计入同一 attempt，不退还预算。无法证明覆盖则保持 publishing/complete=false 并返回失败。
6. 预算或封口撤销同步撤回磁盘 summary 的完整/通过声明并重绑哈希；撤回写盘不可证明时仍不可消费。保留已经测得的 mismatch，但未知封口最终为 audit_incomplete，不能沿用旧 terminal 标志成功退出。

实现测试覆盖中断后的账本诚实性与消费者拒绝；所有故障注入使用临时合成目录，不修改父真实登记。本方案不主张消除任意文件系统/终止时序，无法核实的边界必须以未知状态暴露，不能登记为额外通过豁免。

## 8. 独立收口、产物与裁决

拟定正式目录为 `registration/geometry/derivative/finalize/attempt_NNN/`，每个阶段保留报告、索引、台账与调用记录。至少保存：

- `registration.json`、`admission.json`、`expected_keys.json`：父哈希、四窗、122/4672 固定多重集、域阈值、完整 schema、配置/生效常数、资产/环境、授权/校准、预算与累计余额。
- `geometry_rows.json`、`fk_geometry_arrays.npz`：四角色生产解码/FK、三逻辑输入独立和核64坐标、两份 target 绑定、原始点残差、逐点元素及窗口/batch 标量。
- `fk_gradient_arrays.npz`：480 预测状态的 J35、D_decode、J38、生产/独立 raw/weighted 梯度、域掩码、分组证据；保留完整值，不只存范数。
- `fd_rows.json`、`fd_arrays.npz`：4672 键的正负输入/坐标、解析列、双侧差分、失败/缺口及两步长结果。
- `fk_audit_final.json`、`independent_verification.json`：从盘重推的键集、全部门、失败/缺口、调用/成本、消费者状态及明确边界。

F 不能把 summary 作为唯一真值：从绑定源数组重推解码/域、几何/梯度和 FD，逐元素核对落盘数组，再重建每行字段、分母、bool 状态、计数、所有明细及最终门。删除必需数组/行、同步篡改自报通过和 expected_reasons、只保留少数窗或只保存范数均不能通过。

G 与 F 可以复用同一套独立 NumPy/MuJoCo 参考公式；F 的独立性来自新进程、磁盘重读和重算，不冒称第三套数学实现。参考本身由解析小链、独立扰动 FD 和既有后端对拍共同约束，仍受固定资产/采样范围限制。

| 裁决 | 必须满足 | 后续 |
| --- | --- | --- |
| `real_saved_fk_covered_contract_passed` | 可消费终态；完整固定矩阵；failures/gaps 均空；全部逐元素、分组、重放、预算/调用门通过 | 仅排除本矩阵上已覆盖层的实现偏离，不自动启动其他层审计 |
| `real_saved_fk_mismatch` | 可消费终态且证据完整，无必测 gaps；存在独立重推的数值失败 | 提交具体层/键/原始值与参考差异，另行决定后续修复或干预 |
| `audit_incomplete` | 任一必测 gap、前置/来源无效、预算/调用不允许继续、未知历史或不可消费封口 | 停止；保留已测失败，不宣称覆盖通过 |

`complete_evidence` 与 `covered_contract_passed` 分开；完整负结果可为 true/false。最终 `implementation_contract_passed` 仅表示本审计器的实现/登记/重推契约，无论其值如何都不替代 `full_objective_contract_passed=null` 或动作质量结论。任何裁决均维持 U0 incomplete、UR no_quality_gain、原 F not_ready、旧预测 r6 不可验收、旋转融合不接入。

## 9. 实施前验收与授权清单

### 9.1 工程验收

真实授权前应完成以下合成/接口验证，并在新的验证记录中列明具体结果，不引用旧全项目计数代替本轮：

| 验证组 | 必需覆盖 |
| --- | --- |
| 解码/几何 | 列/行 6D 混用、wxyz/xyzw、错误 site/关节顺序、单侧世界系转换、root std/限位替换、target 两副本矛盾；近 π 保留负门 |
| Jacobian/梯度 | 35/36/38 维混淆、左右乘/`R.T` 错误、漏 std/限位尺度、分母误乘 3、FK 权重错接；清零梯度、极小值、FD 步长选择、clamp 内/外/边界、6D 退化 |
| 证据独立性 | 父输入/哈希失配；只存范数；空/错形状/非数值/NaN/Inf；bool 混数字；删/重/额外键、改明细/分母/通过字段、替换 FD 原始量仍自报一致 |
| 准入/调用 | 授权前不读真实值/不建运行根；前置 publishing 不构造后端；实际 FK/VJP/scene 初始化计数；跨 attempt/run 累计与不足时提前拒绝 |
| 账本/封口 | 成员半损坏、完整外部 cohort、无标记成员、损坏标记；未登记尝试、负成本、慢持久化/撤回成本、不可验证封口保持 unknown 且失败退出 |

优先复用 [既有 FK 测试](../../tests/test_fk.py) 中的约定，但其随机状态对拍与非零梯度冒烟不代替本方案真实保存预测的解码→FK 梯度证据。运行会加载真实锁定 G1 的测试或 `validate_fk.py` 属真实资产计算，不能包装成授权前纯合成校准；如需额外执行，须明确登记范围、成本与调用额度。

### 9.2 真实执行必须另外取得授权

当前以下均未勾选；本草案存在不构成授权。授权记录拟采用专用 schema，严格校验 JSON 类型、精确预算/次数、完整键集和每项 true，绑定本方案、准入包、校准及父来源哈希。

- [ ] 确认新问题与 cohort 身份，父运行仅作不可变输入，不借旧调用或预算。
- [ ] 确认固定四窗、三逻辑输入、720/960 槽口径与两份父 target 绑定。
- [ ] 确认仅保存预测 leaf 的 FK 输出梯度；无网络/checkpoint/参数反向或训练更新。
- [ ] 确认允许 32 状态、两个 h、35/38 维的状态 FD，4672 行/9344 姿态求值及 F 全量重跑。
- [ ] 确认冻结校准契约、近 π 负门、clamp/6D 域缺口和逐位重放要求。
- [ ] 确认锁定资产闭包、15 点/29 关节顺序、设备/精度/确定性配置与初始化计数。
- [ ] 确认 R0/G/D/F 180/420/600/600s、各最多两次、阶段不互借以及 §6 全部累计调用/工作量上限。
- [ ] 确认工程反例、固定根归属、三方对账、撤回费用与不可验证封口的 fail-closed 已验收。
- [ ] 确认通过仍 partial/not_ready；失败不自动选择因子、续训、改教师或接入生产。

### 9.3 下一步与当前状态

下一项交付先是 **I/C 的实现、纯合成自测和可审阅冻结准入包**，之后才申请本方案 §9.2 的真实执行授权。准入包须列出源码身份、校准观测/证据哈希、精确数组/键 schema、名义工作量、全阶段调用/成本上限与未覆盖层；不能把本文件的提案数字登记成已执行或已授权事实。

截至 2026-10-28，本方案的 I/C 工程工作已完成（见 §10）：审计器、CLI、39 项测试、纯合成校准与 R0→G→D→F 自测均已完成；冻结校准契约与只读准入包已生成。真实 cohort、预算、注册与正式 R0/G/D/F 均未创建或执行，§9.2 九项仍未勾选。旧真实审计的已覆盖通过结论及全部质量负结果保持不变。

## 10. I/C 实施、纯合成校准与准入包（2026-10-28）

本节记录本轮实际完成的 I/C 工程工作。它是纯合成/接口证据，不是真实 R0→G→D→F 证据，也不构成 §9.2 授权。

### 10.1 实现与测试

| 交付 | 内容 |
| --- | --- |
| `src/ireset/eval/w1_real_fk_audit.py` | 固定 122 行几何/梯度键、32 状态/4672 行 FD 键生成器与 schema；独立 NumPy/SciPy 解码、`D_decode`、`J35`/`J38`、损失与梯度；29 关节/15 site 合成长链（torch 生产替身 + NumPy 参考）；调用计数；cohort/index/ledger/publishing 封口；R0→G→D→F；纯合成校准；授权/准入门 |
| `scripts/audit_w1_real_fk.py` | `--synthetic`（默认）、`--calibrate-synthetic`、`--admission`、`--real`；真实模式在读取数组、加载资产或创建正式运行根之前校验授权、校准、准入包和固定历史根 |
| `tests/test_w1_real_fk_audit.py` | **43 passed（26.37s）**；覆盖列/行 6D 混用、wxyz/xyzw、退化 6D、clamp 边界/外侧、`R.T`/枢轴/尺度反例、分母 3600/900、梯度分组与 clamp 零梯度、父绑定逐位、FD 域缺口、键多重集/行字段/数组篡改、账本负成本/running/publishing/半损坏/外部 cohort/慢持久化/撤回成本、授权与准入严格门、真实入口零写入拒绝、合成全链与资产隔离 |
| `src/ireset/train/losses.py` | 仅增加可选 `fk=`/`static_frame=` 注入（默认仍构造 `G1TorchFK` 与 `largebox_static_frame()`）；用于无真实资产的合成解码/FK。`test_generation.py` + `test_label_quality.py` 回归 **26 passed** |

合成测试证明：`LossAssembler` 的真实训练 `decode_clip` 路径被实际执行；合成 fixture 不调用 `largebox_static_frame()`；`w_contact=0` 时 contact 分支执行数为 0。

### 10.2 冻结校准

纯合成校准（seed `20261028`，64 状态，4 个 FD 状态 × 35/38 维 × 两个 h）写入：

- `derived/audit/w1_real_fk_audit/tolerance_contract.json`，SHA-256 `82209619f85da976565edd2af3af485a38b726bc783fc6e49ccc11cbb75f8023`；
- 证据 SHA-256 `8f46fbe1830d4576f61ff85c8384a78e057ff3f5a900299e5c8c3440e3df0ed9`；
- `mode=synthetic_calibrated`，`real_asset_loaded/real_checkpoint_loaded/real_dataset_loaded=false`，`real_model_calls=0`。

观测到的“相对提案上界比例”最大值如下（均满足 `4× ≤ 1`，因此阈值保持提案值、未被抬高）：

| 检查 | 观测最大比例 | 4× |
| --- | ---: | ---: |
| decode_float32 | 0.09019 | 0.3608 |
| decode_rotation_matrix | 0.008517 | 0.03407 |
| kernel64_fk | 4.29e-10 | 1.72e-9 |
| production_fk | 0.05652 | 0.2261 |
| fk_scalar | 0.002838 | 0.01135 |
| fk_gradient | 0.20024 | 0.8010 |
| j35_fd | 5.47e-5 | 2.19e-4 |
| j38_fd | 2.28e-5 | 9.14e-5 |

已登记边界/负例：近 π 生产矩阵误差与 quat 标量作为事实保留；`|x_j|=1` 无唯一双侧导数（gap）；`|x_j|>1` 导数为 0；退化 6D 为 gap；零残差时损失与梯度为 0。近 π 与 clamp 边界不参与容差放宽。校准器会在加载契约时重算证据与 4× 规则；替换证据、同步重哈希后放宽阈值或改动提案阈值均被拒绝。

### 10.3 纯合成 R0→G→D→F

运行根 `derived/audit/w1_real_fk_audit/synthetic/20261028/`：

- R0/G/D/F 四次尝试均 `done/complete=true`、可消费；结论 **`synthetic_contract_passed`**，`complete_evidence=true`、`covered_contract_passed=true`、`problems=[]`、`gaps=[]`。
- G：122 几何行 + 20 梯度行，0 失败；D：4672 FD 行，0 失败/0 缺口；F 新 provider/后端重读磁盘、重算 G+D、逐位/容差对账与全 stencil 重放通过，独立验证 `problems=[]`。
- 调用与提案矩阵一致：G `loss_assembler=2`、`production_fk_batch=4`、`kernel64_fk_batch=3`、`production_vjp=2`、`mj_forward=1200`、`mj_jac_site=7200`；D `mj_forward=9344`、`fd_pose_eval=9344`；F `mj_forward=10544`、`mj_jac_site=7200`、`fd_pose_eval=9344`。合成模式 `scene_setup_forward=0`（真实模式应为 1）。
- 全链零计数均为 JSON 整数 0：`checkpoint_loads`、`conditioner_calls`、`model_calls`、`parameter_backward`、`parameter_fd`、`optimizer_steps`、`training_updates`、`ddim_calls`、`checkpoint_writes`、`contact_branch_executions`。
- 合成成本 R0 1.1040s / G 1.3501s / D 2.9289s / F 2.9780s = **8.3610s**；仅证明管线可运行，**不是**真实运行时间估计。
- 边界保持：`objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false`。

### 10.4 准入包与仍未覆盖的内容

`--admission` 生成只读准入包，绑定：父运行 7 个文件的既有 SHA-256、固定四窗、122/4672/32 schema 与键哈希、实现文件哈希、环境、冻结校准证据、阶段成本/尝试上限、全链调用上限和 §6 零计数。该包不读取父数组、不编译资产、不加载 checkpoint、不创建 cohort 或正式运行根；其自身状态保持 `authorized=false / registered=false / run=not_run`。

仍未覆盖并须在结论中保留：真实生产 float32/CUDA 前向、真实 MuJoCo 几何/编译后地址、真实父 FK 逐位绑定、真实 FD/F 重放、网络参数梯度、累积/裁剪/AdamW/更新、选模、DDIM 与端到端动作质量。F 的独立性来自新进程/新后端/磁盘重读；参考公式与 G 共用同一套 NumPy/MuJoCo 实现，不是第三套数学证明。

### 10.5 当前状态

- 已完成：I/C 实现、39 项测试、冻结校准、纯合成 R0→G→D→F、只读准入包。
- 未完成/未授权：§9.2 九项、真实 cohort/预算/注册、真实 R0/G/D/F、真实资产闭包编译、任何训练或生产接入。
- 下一交付：用户审阅 §9.2 与准入包后明确授权；R0 先重核父哈希/四窗/生效常数/资产闭包/环境，再按 R0→G→D→F 执行。
