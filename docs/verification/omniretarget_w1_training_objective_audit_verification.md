# W1 训练拟合与目标一致性合成审计验证记录

日期：2026-10-22（A4 `(demo,start,t)` 键级多重集与 A5 独立键级复核补齐后重跑；第五轮 1 P1 + 1 P2 修复保留）。对应方案：[W1训练拟合与目标一致性单因素诊断执行方案](../plans/omniretarget_w1_training_fit_objective_audit_plan.md) A0–A5。**本次交付是纯合成审计实现与运行，不是真实数据审计或单因素训练预注册。**

## 0. 结论与边界

| 项目 | 实际结果 |
| --- | --- |
| 权威命令 | `uv run --locked python scripts/audit_w1_training_objective.py --out derived/audit/w1_training_objective_audit` |
| 退出码 | 0 |
| 生产侧结论 | `synthetic_contract_passed` |
| 独立复核结论 | `synthetic_contract_passed`；`problems=[]` |
| 权威产物 | `derived/audit/w1_training_objective_audit/attempt_002/`（防覆盖入口；根目录与 `attempt_001/` 旧产物已被取代） |
| 审计行 | **A1–A4 67 + A0 静态 22 = 89**（A1 23、A2 14、A3 19、A4 11；A0 22） |
| 失败/缺口 | **0 / 0** |
| 已登记限制 | **5**（见 §8） |
| 真实数据/真实 checkpoint/真实 G1 FK | 0 / 0 / 0（均 `not_run`） |
| 生产模型/训练 checkpoint 前向、真实数据训练更新 | **0 / 0** |
| 单因素选择 / `auto_execute` | `false` / `false` |
| 最终交互 / 生产接入 | `not_ready` / `false` |

`synthetic_contract_passed` 只表示**解析合成适用域**内的编码、标量目标、梯度、更新循环、聚合与选模契约通过；不证明真实数据行为、G1 FK 正确性、训练主因、泛化或物理可行性，也不升级 `not_ready`。A3 的采样/调度器事实使用随机初始化的微型 `DiffusionG1`（`d_model=8/1层`）与替身模块，**不是** W1 训练 checkpoint 或生产模型调用；真实模型调用计数为 0。方案 §5 的真实数据门全部未触发，未创建执行注册/cohort/预算。

## 1. 交付物

| 文件 | 作用 |
| --- | --- |
| [src/ireset/eval/training_objective_audit.py](../../src/ireset/eval/training_objective_audit.py) | 独立合成审计模块：A0 静态契约、A1/A2/A4 对拍、A3 循环/采样/RNG、A5 独立复核；不修改默认 loss、模型或数据入口 |
| [scripts/audit_w1_training_objective.py](../../scripts/audit_w1_training_objective.py) | 默认纯合成入口；显式拒绝 manifest/cache/checkpoint/预注册/预算/demo/step/seed 等真实或研究输入；不调用旧诊断脚本 |
| [tests/test_w1_training_objective_audit.py](../../tests/test_w1_training_objective_audit.py) | 33 项正向链与篡改反例（含 A4 键级 12 类变体、A5 证据攻击矩阵、DDIM 时间表/末步替换）；临时目录，不扫描生产历史 |
| [derived/audit/w1_training_objective_audit/attempt_002/](../../derived/audit/w1_training_objective_audit/attempt_002/) | 权威合成产物：`static_contract.json`、`s0_formal_comparison.json`、`objective_rows.json`、`synthetic_inputs.npz`、`gradient_rows.json`、`gradient_arrays.npz`、`update_parity.json`、`update_arrays.npz`、`sampling_trace.json`、`rng_state_facts.json`、`selection_evidence.json`、`audit_summary.json`、`independent_verification.json` |

权威运行基于已提交的 `8e1402dc403e430b48c5e9665b70e824733107f8`；运行时本轮 A4 键级代码/测试修复尚未提交，随后只同步了文档，未再改动代码或工件。`static_contract.json` 记录运行时的 `git.dirty=true`、tracked diff SHA-256 `c75ba518f7a350246ad2f188b722d7c4e38c2a5c9e45de65b09fed59547e82f2`，并单独登记三个实现文件 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `src/ireset/eval/training_objective_audit.py` | `5f2a58534bf6f7f8e4d17c482e947f449d6b878097c04f83e8ba4d67f745a35b` |
| `scripts/audit_w1_training_objective.py` | `2816a31d1a211b9f46caf51e391cdb61934b33b55540519aace1d47e3306c832` |
| `tests/test_w1_training_objective_audit.py` | `63661617d551356626dddf9d72d501b201837321db6c03137536fc02ec7f7bd5` |

配置文件 `configs/train/omniretarget_motion.yaml` 原字节 SHA-256：`0e2cfa96849ee2de14ac84b7343289090c606f78116bb79114ff81b046b521e0`。

## 2. A0 身份、参数映射与 S0/正式差异

- `static_contract.json` **22/22** 静态检查为 `static_confirmed`；没有 `contract_mismatch`。
- 登记 11 个被审源码文件 SHA-256（loss/tokens/backbone/conditioner/deterministic/trainer/generator/w1_training/windows/w1_s0/train_generation），有效配置、12 项参数→消费入口映射、3 段状态维度/单位/坐标系、7 个损失项/归约/权重、`formal_validation_contract` 与 `TOLERANCES` 一并写入。
- 状态定义：`0:3` 标准化 clip 系根位置（m）、`3:9` clip 系旋转 6D（不裁剪）、`9:38` 线性限位归一化关节（干净侧 `u=2(q-lo)/(hi-lo)-1`，解码 clamp 到 `[-1,1]`）。
- 损失权重与配置逐项一致：`pos=1, rot6d=1, joint=1, geodesic=0.5, fk=2, contact=0, vel=0.5`。审计专用 `W1_SYNTHETIC_WEIGHTS` 显式使用 `w_contact=0`，避免误用 dataclass 默认 `0.5`。
- S0/正式差异表 12 项 `static_confirmed`、1 项 `runtime_not_checked`（实际新运行仍需重新登记数据）。关键差异：S0 六窗整批 24 次样本消费/更新（六唯一窗）；正式 `micro_batch=4 × grad_accum=4` 为 16 次样本消费，`drop_last=True`（110 窗/epoch 丢 2 窗）；S0 每 10 步固定六窗探针、保存 last；正式每 250 步窗口 loss、每 1000 步完整 val31 生成并按 demo 等权宏选 best。
- 已登记差异：正式 `Trainer._check_batch/_forward` 在 loss 前拒绝退化 6D；S0 只检查有限 loss，没有同等退化旋转门（S0 是诊断路径，不承担正式训练验收）。

## 3. A1 编码、解码与标量目标

- NumPy 编码对独立解析公式：最大绝对误差 **0.0**；Torch 编码对同一解析期望在容差内一致（列顺序、clip 变换、根标准化、关节映射）。
- 列隔离：改根位置仅 `0:3` 变化；改单个关节仅对应列（示例 `9+3=12`）；改四元数仅 `3:9`。
- `decode_clip` 与 tokenizer 世界系解码：位置最大误差 **0.0**、关节最大误差 **0.0**、旋转矩阵最大误差 **2.8011e-8**（tokenizer 的旋转解码硬编码 float32，按生产 float32 容差判定）；世界系四元数 `|dot|` 最小 **0.9999999999999998**（符号等价）。
- 逐项损失（float64、独立 NumPy/scipy 公式）：
  - 全目标 `total` = **5.599772600542075e-4**；分项 `pos=4.9549856908e-5`、`rot6d=4.9329040122e-5`、`joint=5.0213358583e-5`、`geodesic_rad2=1.3462212551e-4`、`vel=4.4898224644e-6`、`fk_m2=1.7066451523e-4`（合成 FK）。
  - 分组均值 vs 38 通道整体 MSE：分组和 **1.4909225561e-4**，整体 MSE **5.0021347641e-5**，差 **9.9070907972e-5**；实现 `total` 与分组和差 **2.7105e-20**，证明没有暗中退化成整体 MSE。
  - `vel` 对全部 38 通道共同归约：**4.4898224644e-6**；若按三组分别求 vel 再相加会得到 **1.3597526734e-5**，二者不同，归约口径被钉住。
  - batch 重排/重复使用**独立目标与非零、不同大小的逐样本残差**；排列/重复后的 total/pos/vel 均以独立期望均值为准（副本 total=**6.3221850458e-4**），不再退化为零损失。`pos/rot/joint/adjacent` 四类扰动分别对拍通过。
- 数值域：近零 6D 的 `ok=False`；零 6D 时 `LossAssembler.decode_clip` 会用单位四元数占位但 tokenizer 的 `ok` 明确为 False，正式 Trainer 拒绝（已登记为“仅正式调用方有门”的限制）；饱和 logits 输出严格在 `[-1,1]`；关节边界 `u=±1/±1.5` 按 clamp 后的限位值解码。
- 近 π 负门原样保留：相对角 `π-1e-4` 时独立期望 `1.2336220116`，实现 `geodesic_rad2=0.0616674696`（相对误差 **0.9500**）。这是已知的 `so3_log_torch` 近 π 精度限制，**没有被零旋转/零误差回退成通过**，也不是本轮新增缺陷；行状态为 `synthetic_verified` 并带 `limitation=near_pi_precision_negative_gate_retained`。
- 平行 6D 列：tokenizer 自身的 `ok` 标志没有识别（`decode` 随后在 scipy 因非正行列式抛 `ValueError`）；正式 Trainer 的 `_rotation6d_valid` 与生成器 `_valid_windows` 均拒绝该输入。记录为工具/未使用解码路径的限制，而不是静默零旋转通过。
- 接触分支：`w_contact=0` 时 `SyntheticFK` 运动 FK 调用 2 次（各 15 site），手部 site 调用 **0** 次、SDF 调用 **0** 次，`parts` 无 `contact_m2`；参考接触日程仍能进入 `ConditionEncoder`（memory 有限）。`w_contact>0` 时手部 FK 执行、`contact_m2` 产生（接触标志 **2.618358465**、监督 mask 分支 **5.412933529**）。所有 FK 行标记 `fk_backend=synthetic`，真实 G1 FK 对拍 `not_run`。

## 4. A2 梯度路径与尺度

- 六个分项 `pos/rot6d/joint/geodesic_rad2/vel/fk_m2` 的输出空间梯度均有限且非零；每行记录未加权/加权梯度范数、权重与零值原因。
- `grad(total)` 与逐项加权梯度和：最大绝对误差 **4.3368086899e-19**（float64）。
- 权重未消费：全零权重时 `total=0`、梯度范数 **0.0**；非零权重时对应梯度非零。
- 预测分支未 detach：`gradient_fact` 返回有限非零梯度；反例 `p.detach()` 被登记为 `prediction_not_in_graph`/非零 False，而不是静默通过。
- FK 目标分支：`SyntheticFK` 记录两次调用 `requires_grad=[True, False]`，预测为 True、目标为 False（目标在 `torch.no_grad` 内）。
- 有限差分（float64，远离 clamp/sigmoid 极值与 SO(3) 分支切点）：六项方向导数相对误差最大 **3.7703e-9**；余弦在双侧非零有限时给出数值，零梯度项为 `null` + `prediction_gradient_zero`。
- 生产 float32：梯度有限，对 float64 的最大相对差 **1.710175e-5**；该行明确标记审计使用合成构造器，真实 `LossAssembler` 构造器硬编码 float32 的适用域仍按限制登记。
- 参数空间链式法则：`w` 的参数梯度与 `direction·∂loss/∂pred` 独立重算一致（**0.1574401088**）。

## 5. A3 更新循环、噪声与 RNG

- 共 4 个 S0-vs-正式 Trainer parity run（确定性/扩散 × 无裁剪/裁剪），每个 run 两侧各 **4** 个 micro-batch 实际消费；同一初始替身参数、同一有序 micro-batch、同一 t/epsilon。
- 四个 run 的 `parity_problems=[]`：前向输出 **逐位相等**、分项 loss（pos/rot6d/joint/geodesic/vel/fk/total）**逐项一致**、更新后参数 **逐位相等**、裁剪前梯度 **逐位相等**、裁剪后梯度 **逐位相等**、AdamW `step/exp_avg/exp_avg_sq` **逐位相等**、扩散 t/epsilon/x_t **逐位相等**、参数确实发生变化、模型与 conditioner 两个模块梯度均非零。
- 四个 run 的**完整原始证据**写入 `update_arrays.npz`（micro 目标、两侧前向、分项 loss、裁剪前后梯度、参数、AdamW 状态、扩散 t/noise/x_t）；A5 从这些数组和绑定输入**逐一重放**四组更新，并重算两组选模，不再只检查文件存在或自报字段。
- micro loss 确实除以 `grad_accum`：累加梯度与“逐 micro loss 求和后除 grad_accum”的最大差 **0.0**，梯度范数 **1.3948284388**；若漏除一次，两种路径可区分。等大小 micro（2+2）累加与一次 4 窗大 batch 的最大差 **1.8626451492e-9**（≤1e-6），确认等大小累计正确。样本加权与 micro 等权的最大差 **0.3381143212**，说明实现语义是 micro 均值等权，S0 因六窗等大小而不产生该差异。
- optimizer 成员：2 个参数、2 个 optimizer 参数、重复 **0**；AdamW 单步参考（lr=1e-4、wd=1e-5、betas=0.9/0.999）最大误差 **0.0**；Trainer 无 `lr_scheduler` 调用，固定 lr。
- 扩散：`add_noise` 与独立 alpha 公式在 t=0/1/500/999 的 `x_t` 最大误差 **0.0**；A5 用保存的 micro_x0/t/noise 按同一 alpha 基础独立重建 x_t 并逐元素对账（noise 清零/Inf/t 越界均拒绝）；t=0 噪声比例 **0.0100008296**（非无噪）；`prediction_type=sample`、`clip_sample=False`、`thresholding=False`、`final_alpha_cumprod=1.0`；DDIM 单步（t=600、eta=0）与独立公式最大误差 **0.0**；**完整 DDIM 循环实际执行**：时间表在采样前冻结为 `[800,600,400,200,0]`，`sample_calls=1`、`forward_calls=5`、`scheduler_step_calls=5`、实际时间步逐项相等；预测器依赖 x_t/t，参考轨迹由**独立 eta=0 DDIM 更新公式**逐步构造（不复用被测 `scheduler.step`），末步偏差 **0.0**；采样时间表改成 `[4,3,2,1,0]` 或 t=0 更新返回旧状态均被拒绝。
- 数据消费：S0 六窗整批 **24** 次消费、6 个唯一窗口、每个恰好 4 次；正式 4×4、10 窗数据集 **32** 次消费、8 个 micro-batch、每 epoch 丢 **2** 窗，4 个 epoch 的丢弃集合分别为 `[2,8]/[1,6]/[6,8]/[0,7]`，累计覆盖 0–9 全部窗口。
- RNG：`_evaluation_state` 在故意异常后恢复 Python/NumPy/Torch CPU 状态与模块训练标志；DataLoader 使用独立 generator 时不污染全局流，`generator=None` 时会消耗全局流（两向都记录）。

## 6. A4 聚合、评价与选模

- 宏/微排名故意不同：method1 宏 **0.25**、微 **0.4166666667**；method2 宏 **0.5**、微 **0.1666666667**——宏偏好 method1，微偏好 method2。正式选模只消费注册 val 全集的 **demo 等权宏**，不会把窗口微平均当宏。
- 窗口微平均：6 个合成 val 窗口的独立逐窗 loss 均值与 `Trainer.validate` 最大差 **4.8506383914e-12**。
- 注册键级多重集：`(demo,start,t)` 注册 **6** 键（demoA 两窗、demoB 一窗，每窗 t=0/500），观测多重集必须逐键相等；完整时宏 **0.21**、微 **0.1783333333**，两者不同。删整窗、重复整窗、换 demo、缺 t、重复 t、缺 metric、bool/数值混用、NaN、Inf、缺 `start`、`numerical_valid` 用整数、额外 t 共 **12 类**变体全部 `complete=false`、`macro=null`，并保留 `missing_window/duplicate_window/demo_not_registered/missing_t/duplicate_t/missing_field/bool_metric/nan_metric/inf_metric/unexpected_window` 等具体原因；空注册/空观测与重复注册键也不能空真通过。A5 用**独立实现**从原始键级行重建多重集、宏/微与 12 类变体原因，而不是复用生产侧检查器或信任 row 自报字段。
- 合成物理 FK MPJPE（仅聚合代数，`fk_backend=synthetic`）：demoA 4 窗、demoB 2 窗，宏 **1.1023083019e-3**、微 **7.9140083287e-4**；两者不同且排名可区分。
- 注册 roster：3/3 完整时宏 **0.11**、`complete=true`；缺行、重复 demo、T 不符、status 缺失、`numerical_valid=False`、bool/NaN/Inf 指标、额外 demo、缺指标键共 **10** 类变体全部 `complete=false`、`macro=null`，坏行保留在固定分母内。
- 选模：完整第二次评估宏 0.2 时 `best_formal_step=2`；第二次缺一行时保持 `best_formal_step=1`，不会因分母缩水选出“更优”。
- 表示：同输入同代码的 float32 重复调用逐位相等；新合成产物保留未舍入值（`has_beyond_6_decimals=true`）；训练 `loss_history` 的 core 字段全部按 6 位小数舍入，已登记“不能从舍入日志伪造精确逐位 total 对账”。

## 7. A5 独立复核与篡改反应

`independent_verification.json` 不信任顶层通过字符串，从原始 `synthetic_inputs.npz`/`gradient_arrays.npz`/`update_arrays.npz`/行文件重算：

| 检查 | 结果 |
| --- | --- |
| 独立重算 pos/rot6d/joint/geodesic/vel/fk/total（输入/配置逐字段绑定） | 与生产者行一致；total 重算 **5.5997726005e-4**；root/q/wfc 重编码必须逐位回到 x0；有效损失权重须与配置文件及 W1 契约一致 |
| 从原始输入重算六项梯度 vs 保存梯度 | 每项均要求一致（容差 1e-8），不是只验证保存数组的加权和 |
| update replay：4 个固定 run_id 的前向、分项 loss、裁剪前后梯度、参数、AdamW 状态 | 四个 run 的精确集合/模型类型/grad_clip/micro 数/part_keys/数组形状先核验，再逐 run 重放；只保留一组或重复 run 均拒绝 |
| diffusion noise→x_t | 用保存的 micro_x0/t/noise 独立重建 x_t；noise 清零、Inf 或 t 越界均拒绝 |
| 裁剪身份/形状 | NPZ 的 grad_clip 必须等于固定 run 的 1e9/1e-6（同时与 JSON 一致）；标量参数/梯度/gnorm 必须为 0 维，bias/权重为精确 [38]；with_clip 必须实际触发裁剪（pre>clip、post≈clip）。 |
| 小量梯度/状态 | 裁剪后梯度约 1e-7，使用 `atol=1e-8 + rtol=1e-5` 专用容差；保存梯度/AdamW 状态必须非零，并与 `0.1·grad`、`0.001·grad²` 公式对账；清零 32 个数组被拒绝。 |
| 二阶状态尺度/符号 | `exp_avg_sq` 必须非负；公式与重放比较使用 `1e-30` 级尺度相对容差，而非 1e-8 绝对容差；改 −1e-9 或整体 ×1e6 均被拒绝。 |
| 明细严格类型 | A2 明细对账中 bool 只与 bool 比较；FD `ok=true` 改成整数 `1` 不再等值。 |
| AdamW 步数 | 保存的 step 必须是非布尔、有限数值标量且精确等于 1；1.9 或字符串 "1" 被拒绝。 |
| 有限差分重算 | 从绑定 pred/target、固定方向种子 7 与 eps=1e-6 重算 fd/analytic/rel_error/ok/cosine 并逐项对账，不能只信保存的派生标记。 |
| batch 独立期望/生成关系 | 固定形状 [3,T,38]/[6,T,38] 与非空；perm 必须是 [0,1,2] 的合法排列；重复数组必须逐位等于原数组拼接；未排列/排列/重复均值互相一致且与重新计算值对账。 |
| A2 全量明细 | `gradient_rows.json` 与 `objective_rows.json` 的全部 **14** 行状态/actual 逐字段对账；另独立重算六个梯度的范数、加权范数与 `direction_cosine`，并重算 FD 的 fd/analytic/rel_error/ok/cosine。 |
| summary schema | 空 `audit_summary.json` 或缺少 counts/failures/gaps/scope/boundaries 等字段判 incomplete，不能跳过收口核验。 |
| 完整有效配置 | 重解析配置的 seed/grad_clip/steps/…/loss 全字段与 effective_config 及 A0 检查值对账。 |
| 梯度工件/行语义 | 空 `gradient_arrays.npz`/`{}` 或 `gradient_rows.json={}` 判 incomplete；`grad.<term>` 的 grad_norm/加权范数/有限性/非零性与重算对账；FD `rel_error/ok` 与冻结门对账 |
| A0 静态检查键集 | 22 个检查 ID 的固定集合/无重复/状态 `static_confirmed`，删减或重复均拒绝 |
| summary gaps | `summary.gaps` 与缺失必测行列表、`counts.gaps` 一致，非空缺口阻止通过 |
| selection replay：两组 raw callback 行 | 从 `per_demo` 重算 demo 等权宏/complete，重推 best step：完整 run=2、缺行 run=1；A5 `checks` 含 `independent.selection:{complete,incomplete}_run` |
| 结构/有限性 | 顶层与嵌套 mapping/list、必需键、形状、非空集合先校验；NaN/Inf 在容差判定前登记为失败 |

复核攻击矩阵（33 项测试与独立脚本；均在临时副本/合成故障注入，权威产物未改写）：

| 攻击 | A5 反应 |
| --- | --- |
| `update_parity.json={}` | `structure_empty:update_parity.runs` / `audit_incomplete` |
| `best_formal_step` 2→999 | `selection_row_mismatch:complete_run:2!=999` / `implementation_mismatch` |
| loss total 置 NaN | `nonfinite:objective.loss.total_weighted_sum.total` / `implementation_mismatch` |
| `grad_vel` 一个元素 NaN | `nonfinite:gradient_arrays.grad_vel` / `implementation_mismatch` |
| 全部 `grad_*` ×2 | `gradient_recompute_mismatch` / `implementation_mismatch` |
| `synthetic_inputs.root` +100m | `input_x0_not_bound` / `implementation_mismatch` |
| `effective_config.loss.w_fk=2000` | `static_effective_loss_mismatch` / `implementation_mismatch` |
| 正式 Trainer 的 x_t 换成干净 x0 | `update_replay_mismatch`（前向/分项 loss）与 s0/formal x_t 不等 / `implementation_mismatch` |
| DDIM 循环禁用/未执行 | `sampling_full_loop_not_executed` / `implementation_mismatch` |
| `objective_rows.json=[]` | `structure_not_mapping` + 缺行，正常落盘 / `audit_incomplete` |
| `max_abs_errors=[]` | `sampling_x_t_errors_empty_or_mismatched`，不再 `max([])` 崩溃 / `audit_incomplete` |
| 空 `gradient_arrays.npz` + `gradient_rows.json={}` | 梯度工件结构/必测行缺失 / `audit_incomplete` |
| `grad.vel` 状态改 `runtime_not_checked` | `required_row_not_verified` / `audit_incomplete` |
| `grad.vel.grad_norm=NaN` | `nonfinite` + 梯度行对账失败 / `implementation_mismatch` |
| FD `vel.rel_error=1.0, ok=false` | `finite_difference_row_not_verified` / `implementation_mismatch` |
| 仅保留一组 update run | `update_run_set_mismatch` / `audit_incomplete` |
| 两侧 noise 清零/置 Inf | `noise_to_x_t` / `nonfinite` / `implementation_mismatch` |
| 删除 22 条静态检查至 1 条 | `static_check_set_mismatch` / `audit_incomplete` |
| summary 添加假 gap | `summary_gap_list_mismatch` / `audit_incomplete` |
| 排列后 total +0.1 | `batch_invariance_mismatch:permuted` / `implementation_mismatch` |
| DDIM 固定 t=0、跳过 scheduler.step | `sampling_full_loop_forward_or_step_count` / `implementation_mismatch` |
| with_clip 数组换成 no_clip | `grad_clip_identity` / `implementation_mismatch` |
| 标量参数改成三维广播数组 | `structure_shape` / `audit_incomplete` |
| FD 原始 fd=1.0 或删除 fd/analytic | `finite_difference_recompute_mismatch` / `implementation_mismatch` |
| batch actual 与 independent_mean 同时 +0.1 | `batch_invariance_mismatch` / `implementation_mismatch` |
| `gradient_rows` 的 grad.vel 范数改 123/456 | `gradient_detail_mismatch` / `implementation_mismatch` |
| `audit_summary.json={}` | `structure_empty:audit_summary` + 缺 schema / `audit_incomplete` |
| effective_config seed/grad_clip 改 999 | `static_effective_config_mismatch` / `implementation_mismatch` |
| 采样时间表 [4,3,2,1,0] 或 t=0 更新返回旧状态 | 时间表/参考轨迹不匹配，A3 row `contract_mismatch` |
| with_clip 的 post/grad/AdamW 32 个数组清零 | `saved_grad_zero`/`saved_adamw_state_zero`/`update_replay_mismatch` / `implementation_mismatch` |
| `batch_target_r/batch_pred_r` 置空 | `structure_shape` / `audit_incomplete` |
| 重复批预测替换为 target、行值归零 | `batch_repeat_not_original` + `batch_mean_invariance_mismatch` / `implementation_mismatch` |
| perm 改为 [0,0,0] | `batch_permutation_invalid` / `implementation_mismatch` |
| gradient_rows 的 FD 明细改成 fd=1/analytic=123/rel=1/ok=false | `gradient_detail_mismatch` / `implementation_mismatch` |
| grad.vel direction_cosine 改 9 | `gradient_detail_mismatch` + 重算余弦门 / `implementation_mismatch` |
| AdamW step 改 1.9 或字符串 "1" | `adamw_step_not_exact_one` / `implementation_mismatch` |
| with_clip 的 `exp_avg_sq` 改 −1e-9 | `saved_adamw_state_negative` / `implementation_mismatch` |
| with_clip 的 `exp_avg_sq` 整体 ×1e6 | `adamw_state_formula_mismatch` / `implementation_mismatch` |
| gradient_rows 的 FD `ok` 由 true 改为整数 1 | `gradient_detail_mismatch` / `implementation_mismatch` |
| A4 键级 base 删除一个 `(demo,start,t)` 行 | `selection_window_base_not_complete` + 计数/聚合不一致 / `implementation_mismatch` |
| A4 `missing_window` 变体行替换为完整 base | `selection_window_variant_not_rejected:missing_window` / `implementation_mismatch` |
| A4 键级 row 的宏 +0.1 | `selection_window_aggregate_mismatch:macro` / `implementation_mismatch` |
| 注册键重复一项 | `selection_window_base_not_complete`（重复注册键） / `implementation_mismatch` |
| base 行 loss 改为 bool | `selection_window_base_not_complete`（`bool_metric`） / `implementation_mismatch` |
| 期望原因改成不存在的字符串 | `selection_window_variant_reason_missing:missing_t` / `implementation_mismatch` |
| 键级 `variants` 容器改成列表 | `selection_window_variants_not_mapping` / `audit_incomplete`，复核器不崩溃 |
| 变体 gaps 改为非字符串对象 | `selection_window_row_variant_gaps_mismatch` / `implementation_mismatch`，`set()` 前先做类型校验 |

其余旧攻击（目标 total +0.1、删除 `grad.vel`、写坏 JSON、改 update 参数、改 `grad_vel`、删除 `static_contract.json`）继续被拦截；验证器顶层异常也会由 wrapper 写入 `verifier_exception` 报告，而不是中断。

## 8. 已登记限制（不冒充通过）

1. **近 π 精度负门**：`so3_log_torch` 在 `π-1e-4` 附近把角度恢复为约 0.70 rad；独立期望与实现的相对差约 0.95。保留负门，不以零旋转或放宽容差换通过。
2. **平行 6D 列**：`StateTokenizer.decode` 的 `ok` 标志只检查两列范数，平行列会先被当作有效、随后 scipy 抛错；正式 Trainer 与生成器各自的 `_rotation6d_valid`/`_valid_windows` 拒绝该输入。该 NumPy 解码路径当前不被生产训练/生成消费；作为工具限制登记。**最小修复建议（本轮未实施，避免扩大生产代码改动面）**：若未来消费该路径，把 `ok` 改为与 `_rotation6d_valid` 同式的正交列谓词（`a` 范数、`b - proj_a b` 范数、`a × orth` 范数均设下限），使退化输入返回 `ok=False` 而不是抛错。
3. **零 6D 解码占位**：`LossAssembler.decode_clip` 对零列会得到单位四元数；tokenizer `ok=False` 且正式 Trainer 拒绝。S0 没有同等退化门，仅检查有限 loss（S0 是诊断入口）。**最小修复建议（本轮未实施）**：若 S0 要用于正式训练声明，应在 S0 循环内复用 `_check_batch`/`_rotation6d_valid`；当前 S0 只作为诊断，不参与验收。
4. **float32 审计路径**：float64 审计使用 `object.__new__` 的合成构造器；真实 `LossAssembler.__init__` 的 lo/hi/root stats 为硬编码 float32，不能宣称装配器已支持 float64；真实 G1 FK 对拍 `not_run`。
5. **历史日志舍入**：`loss_history` 按 6 位小数记录，不能用于逐位 total 对账；精确对账使用本次未舍入合成工件。

## 9. 测试与复现

```
uv run --locked pytest tests/test_w1_training_objective_audit.py -q
# 33 passed in 29.77s
```

测试覆盖：完整 A1–A4 正向链与 A5 证据攻击矩阵；解析编码/解码、分组均值与 38 通道 vel、四类扰动、batch 重排/重复、近零/平行/近 π/关节边界/饱和 logits、接触分支静默、FK 合成标记；逐项梯度、权重消费、detach 反例、FK target no_grad、有限差分、float32、参数链式法则；S0/正式四项 parity、裁剪/无裁剪、micro 除法、样本加权区别、optimizer 成员、AdamW 参考、无 scheduler、x_t/t=0/DDIM、两种数据消费、RNG 恢复与独立 generator；宏微排名、物理 FK MPJPE、roster/坏行、选模、`(demo,start,t)` 键级多重集与 12 类变体、确定性与舍入限制；独立复核的目标/梯度/更新数组/键级 raw 行篡改、缺行、坏 JSON、缺工件；CLI 真实输入拒绝与端到端合成运行。

**未重跑全项目套件。** 上一次全量记录仍为 2026-10-22 首轮修复后的 **947 passed + 1 skipped**（旧范围），本记录不冒充新的全量结果。审计模块的 33 项测试与权威运行是本次新增的独立证据（A4 键级与第五轮复核攻击矩阵均在测试内）。

## 10. 边界与下一交付

- 未加载任何真实数据集、缓存、checkpoint、教师或 G1 资产；未对任何 W1 训练 checkpoint/生产模型进行前向或反向，未产生真实数据 optimizer 更新；A3 仅对随机初始化微型 `DiffusionG1` 与最小替身模块做合成前向/更新。
- 未创建执行注册、cohort、预算或正式 run；`derived/audit/` 只是合成审计产物目录，不是研究 attempt。
- 未选择训练拟合因素，未修改默认 loss/模型/数据入口、缓存、教师、checkpoint 或旧 final；旧 U0 `incomplete`、UR `no_quality_gain`、原 W1 F `not_ready`、r6 不可验收与融合候选不接入均保持不变。
- 通过合成审计只表示实现契约在合成适用域成立；真实数据的拟合/泛化/物理质量仍需方案 §5 的正式预注册与用户授权。已起草[真实训练目标审计预注册草案](../plans/omniretarget_w1_real_objective_audit_prereg_draft.md)，状态为 `draft / not_registered / not_run`，不构成授权或执行记录。若没有可区分假设，下一交付应为 `attribution_insufficient`，而不是重跑旧探针或把 `not_ready` 升级。
