# W1 真实训练目标审计实现与合成自测验证记录

日期：2026-10-26（初版实现与合成自测；同日完成三轮独立复核修复：首轮 8 P1 + 2 P2，第二轮 3 P1 + 2 P2，第三轮 2 P1 + 1 P2；2026-10-27 完成第四轮 2 P1 + 1 P2，并在复核后修复 1 项 P2 成本下限回归；同日增补准入包与冻结校准契约，§13；P2 授权严格化 §14）。对应草案：[W1 真实训练目标审计预注册草案](../plans/omniretarget_w1_real_objective_audit_prereg_draft.md)。**本记录的前 12 节是实现与纯合成自测，第 13–14 节是准入材料与 P2 授权严格化；均不是真实数据审计、预注册执行或训练更新。** 草案仍为 `draft / not_registered / not_authorized / not_run`；`factor_selected=false`、`auto_execute=false`、最终交互 `not_ready`、生产接入 `false`。

## 0. 结论与边界

| 项目 | 实际结果 |
| --- | --- |
| 实现状态 | complete（R0/P/F 骨架、注册/准入、生产取证、batched VJP、独立参考、F 重放、账本/封口、CLI） |
| 合成自测 | `synthetic_contract_passed`；`problems=[]`、`gaps=[]`、`complete_evidence=true` |
| 权威合成命令 | `uv run --locked python scripts/audit_w1_real_objective.py --synthetic --out derived/audit/w1_real_objective_audit` |
| 权威合成产物 | `derived/audit/w1_real_objective_audit/attempt_016/`（P2 授权严格化后；`attempt_015` 及更早为历史实现运行，保留不改写） |
| 固定矩阵 | 2 个 train demo × 2 个完整窗口 = 4 窗；deterministic/diffusion 两个模型；`t=[0,250,500,999]`；噪声种子 `20261008` |
| 行/数组 | objective rows **98**、gradient rows **14**；`objective_real_arrays.npz` 与 `gradient_real_arrays.npz` 全键通过 |
| 调用计数 | conditioner **4**、model **4**、production VJP **4**、independent VJP **4**（均达到上限且不超） |
| 接触监督分支 | 执行 **0** 次（`w_contact=0`） |
| 训练更新 / 参数 FD / DDIM / checkpoint 写入 | **0 / 0 / 0 / 0** |
| 真实 dataset/cache/checkpoint/G1 FK | **未加载 / 未加载 / 未加载 / not_run** |
| 新 cohort / 预算 / 执行注册 | **未创建** |

合成自测使用解析四窗、`_SyntheticConditioner`/`_SyntheticModel` 替身和 `SyntheticFK`；真实的 `LossAssembler`（含新增 observer）、`DiffusionG1`/`DDPMScheduler` 噪声路径和 batched VJP 被实际执行。真实 conditioner/backbone/checkpoint 的严格加载与四窗前向/求导仍未运行；因此本记录不构成真实数据行为、G1 FK 几何、参数梯度、累积/裁剪、AdamW 或选模证据。

## 1. 交付物

| 文件 | 作用 |
| --- | --- |
| [src/ireset/eval/w1_real_objective_audit.py](../../src/ireset/eval/w1_real_objective_audit.py) | 审计模块：固定矩阵、窗口键、注册/准入、生产 observer、独立 float64 参考、batched VJP、F 重放、publishing/seal 账本、授权/容差门与合成校准 |
| [scripts/audit_w1_real_objective.py](../../scripts/audit_w1_real_objective.py) | 默认纯合成入口；`--calibrate-synthetic` 写冻结容差契约；`--real` 需授权记录+已校准契约+固定根直接子目录，当前未运行 |
| [tests/test_w1_real_objective_audit.py](../../tests/test_w1_real_objective_audit.py) | **59 项**正向/反例：完整链路、observer 逐位透明、detach 拒绝、噪声公式、近 π 域、键多重集、账本/running/publishing/超限/封口写失败、父 unit 准入、F 重放篡改、CLI 拒绝；首轮复核新增实际 VJP 计数、落盘梯度逐元素绑定、FK 点重算、跨 run 调用额度、RNG 恢复、合法负结果进入 F、校准证据重验、负成本、行字段/额外行、非数值数组；第二轮新增预算撤销 P 不启动 F 调用、F 实时截止、封口超时撤回磁盘声明、完整链调用余额、嵌套容器类型异常；第三轮新增撤回写盘计入同一 attempt 成本、无法证明覆盖保持 publishing、准入/绑定/前置消费者硬门阻断模型构造、负结果封口失败只撤回证据完整标志；第四轮新增慢写盘后终止仍保留覆盖的 publishing、身份异常阻断模型重放、负结果撤回未知封口返回 audit_incomplete、旧预留成本不得被撤回降低；P2 新增授权记录严格类型/精确 schema·预算·次数/阶段集与显式计数零的反例 |
| [src/ireset/train/losses.py](../../src/ireset/train/losses.py) | `LossAssembler.__call__` 新增可选 `observer`：捕获 live 分项张量、FK 前 `L5`、归约中间量、预测/目标 FK 点；`observer=None` 数值路径逐位不变 |

实现文件 SHA-256（合成产物写入时）：

| 文件 | SHA-256 |
| --- | --- |
| `src/ireset/train/losses.py` | `04b74a45ff59c5acbf8b1fda9361b58c9e56d5466990122524a808071ab80820` |
| `src/ireset/eval/w1_real_objective_audit.py` | `ccd96bfc5a90a5761a80f76b54cb5f1bcd574dc7b403973f362eaa232aeafe1e` |
| `scripts/audit_w1_real_objective.py` | `6fa2584845decbc2dcb4244b4a11d60f92437a0e273a780b1a206d89c3498cde` |
| `tests/test_w1_real_objective_audit.py` | `2ae792e71b5cb3674aca652b2eaa81a910671d6bf1a2e68478597b3762543ba4` |

权威产物哈希：

| 产物 | SHA-256 |
| --- | --- |
| `.../attempt_016/finalize/attempt_001/audit_summary.json` | `fc084933eff3a09478b91968162ea83b11c55be049ceaef3cc25f731fdfdafc2` |
| `.../attempt_016/finalize/attempt_001/independent_verification.json` | `74d0e8a78f988fe2da48c3066ecb063a24dca35f9c91b0af09a4307ad2c46500` |
| `.../attempt_016/production/attempt_001/p_report.json` | `578582ab14031df4b2b3761237cfbe3d366411874817b450047005f5f8d01d3f` |
| `.../attempt_016/production/attempt_001/objective_real_arrays.npz` | `932f2f0cc3ef0c7bdaae2bdb34cb5361a17dee468df4699e45c80144a619f0cc` |
| `.../attempt_016/production/attempt_001/gradient_real_arrays.npz` | `73cc59a3d2dfb4bfef26d11977dd8488b29d457eb7390556cd5fdd241f890a88` |

## 2. R0：注册、身份与准入

- 合成 provider 不读取真实 dataset/cache/checkpoint/history；`verify_identity` 返回 synthetic 标记，`parent_anchors_for_reference_only.loaded=false`。
- R0 先写 running/complete=false 索引与台账，再写 `registration.json`、`tolerance_contract.json` 与 `window_keys.json`；之后才加载四窗。
- 注册包含：固定四窗键、`t=[0,250,500,999]`、一次 CPU 噪声（seed `20261008`、sha256）、`scaled_linear` alpha sha256、W1 权重、调用上限、阶段预算、容差契约、数组/梯度 schema、98/14 必测键、覆盖表、实现/环境身份、截止时间与 `effective_config`。
- admission 检查 shape/dtype/有限性、旋转 6D、关节限位、四元数范数、t/noise、生产 x0 对独立重编码的逐元素容差；当前合成 `encoding` 为 `within_tolerance`（max abs diff ≈9.08e-7，float32 对 float64）。
- checkpoint/model/conditioner 在合成路径为替身；真实路径的 strict load、step/usable/finite/manifest/cache/root_stats 校验代码已实现但 **not_run**。

## 3. P：生产取证与独立参考

- 每模型一次 conditioner→model 完整前向：deterministic 为 `model(memory, frames=60)`；diffusion 为 `DDPMScheduler.add_noise`→`model(x_t,t,memory)`。
- `LossAssembler` observer 捕获：
  - live 分项张量 `pos/rot6d/joint/geodesic_rad2/vel/fk_m2/total`；
  - FK 前实际 `L5`；
  - 每项归约元素（`fk` 为 `[4,60,15]` 的逐 site 平方和）；
  - 预测/目标 FK 关键点、预测是否仍为 loss 实际消费对象、接触分支是否执行。
- `observer=None` 与 observer 开启的返回值/分项逐位一致（测试钉死）；detach 预测会被拒绝。
- 每模型一次 **batched VJP** 从原图取 `[6,4,60,38]`：`pos/rot6d/joint/geodesic_rad2/vel/L5`；不 detach 重建、不重拼损失。P 与 F 各模型各一次 production VJP + 一次 independent VJP，cohort 累计 4/4；F 不再额外调用 `independent_vjp`。计数发生在实际调用处，失败调用也占用额度。
- 独立参考用 float64：独立 6D Gram-Schmidt、独立相对旋转角（`theta=2 asin(norm(v))`）、五个分项、`L5`、`total`、FK 聚合（FK 元素由保存的预测/目标点独立重算，行内 `aggregation_only` 仍为 true）。
- P 在首个确认 mismatch 后停止新 unit；已完成 unit 的 artifact、行、梯度、RNG 状态与 `unit_artifacts` 哈希保留。该部分证据封口为 terminal incomplete 时仍可进入 F 复核，不被当作未知历史。
- 行表：每模型×4 窗×(7 项+L5) + batch×(7 项+L5) + L5/total 算术 + 7 项窗均值对账 = 49/模型；梯度 7/模型（6 切片+L5 加权和）。
- 合成结果全部 `within_tolerance`；生产与独立梯度逐切片在冻结容差内。`fk_m2` 只审计聚合，真实 FK 几何/导数仍 `not_audited`。

## 4. F：独立重放与收口

- F 先核验 R0/P `stage_state`、注册/容差 artifact 哈希、identity/implementation 可复现、admission 数组哈希、P call trace、父 unit 准入；缺前置写 gaps，不补跑 P。
- F 从磁盘读取 P report 与 `unit_artifacts` 哈希，只对 report 中已测量的 unit 重放；未测量 unit 以 `p_unit_not_measured` 与缺失行逐键留 gap。P 的负结果封口为 terminal incomplete 时仍进入 F，最终保留 mismatch，不强行改写为 audit_incomplete。
- 从 raw NPZ 重建四窗、x0、noise/alpha/x_t；从保存的两侧 FK 点独立重算平方距离元素，再核对元素、分母与聚合；不使用保存的 `fk_elements` 作为参考。
- 重新 strict 生成 fresh 模型/替身；每个 unit 前恢复 P 保存的 Python/NumPy/Torch CPU/CUDA RNG 状态，执行一次同输入前向+双 VJP。逐位比较 P 的 prediction/x_t/FK 点/分项元素/production gradient，并**逐元素**对账落盘的 production/independent 梯度；不再使用内存 `p["units"]` 或仅比较范数摘要。
- 行表从磁盘数组重推：98/14 键多重集进入 gaps；`denominator`、`weight`、`aggregation_only`、`window`、`status`、`within_tolerance` 与 `abs_error/limit/relative_error` 的算术一致性均被核验；额外行、额外字段和缺失字段均拒绝。
- 合成重放全部 `prediction_bitwise/x_t_bitwise/production_grad_bitwise` 通过；`problems=[]`、`gaps=[]`。RNG 恢复反例（fresh 构造器额外抽样）仍通过；落盘梯度整体反号、FK 点整体偏移 1m、额外行/分母/权重 999999 均被拒绝。
- 篡改反例（预测 +0.1、production/independent gradient 反号、x_t 清零、声明 gradient mismatch、call count 99、字符串数组）均被 F 拒绝或登记为 incomplete；CLI 真实入口在缺少授权/校准契约时拒绝且不创建 run 目录。

## 5. 账本、预算与封口

- `cohort.json` 冻结 `cohort_id`、`accounting_schema=pre_registered_v1`、三阶段预算/尝试上限、总预算与调用上限。
- 每次 attempt 先写 running/complete=false 索引+台账；`finish_attempt` 采用 pending→publishing→seal，终态发布前不可消费；封口预留、封口后截止/成本覆盖检查、撤销失败保持 publishing/complete=false 且返回失败。`SealResult.terminal` 区分“已封口的合法部分证据”与“publishing/running 未知历史”：P/F 的 terminal incomplete 可被下一阶段消费，但 `sealed/complete` 仍为 false。
- `stage_state` 拒绝 running/publishing/非完成/成本不一致/工件哈希不符；`stage_history` 与 `stage_state` 均要求成本为有限**非负**值并与舍入值对账；`upsert_cost` 直接拒绝负值。
- 调用额度按 cohort 累计：索引/台账中的 `call_balance` 是累计快照，`prior_call_counts` 取同一 fixed-root 下各 run 的逐阶段单调最大值，跨 attempt/跨 run 扣除；缺 balance、ledger/index 不一致或盘目录无登记行时返回 unknown 并在 R0 前停止，不会重置为全额。
- 合成链 R0/P/F 均 `done/complete=true`；索引/台账含 measured、sampled、seal tail、budget、call balance。
- 反例：running/publishing 不可消费；三次三方不一致为 unknown；预算 0s 封口为 incomplete；封口写失败不发布 done；负成本（-100s）在生产和消费入口均被拒绝；同一 cohort 第二次完整运行因累计调用额度耗尽在 R0 前被拒绝；父 unit 必须 active running、scope/注册/截止匹配且输出在父 `units/<id>/`。

## 6. 测试与回归

- `uv run --locked pytest tests/test_w1_real_objective_audit.py -q` -> **58 passed（28.81s）**（首轮 42 项 + 第二轮 6 项 + 第三轮 6 项 + 第四轮 4 项参数化/反例）；P2 后为 **59 passed（28.72s）**，见 §14。
- 相关既有回归：`tests/test_w1_training_objective_audit.py tests/test_generation.py tests/test_omniretarget_w1.py` -> **85 passed（64.82s）**；`tests/test_w1_de_numerics.py tests/test_omniretarget_w1_de.py` -> **73 passed（333.75s）**（第三轮记录；第四轮未重跑相关回归）。
- **全项目套件本轮未重跑**；旧 947 passed + 1 skipped 仅作历史范围记录，不冒充本次结果。
- 合成校准命令 `--calibrate-synthetic` 生成 `mode=synthetic_calibrated`、非空 `evidence_sha256`、`observed` 与 98/14 行计数的契约；真实入口要求该模式，默认 `synthetic_default` 契约被拒绝。`tolerance_contract_problems` 现在重算证据哈希、4× 阈值推导关系，并用记录 seed 重跑纯合成校准；删除证据、替换哈希+放宽阈值、同步伪造 observed 三类反例均被拒绝。

## 7. 已知限制与未执行项

1. 真实 provider 未运行：真实 dataset/cache/checkpoint 严格加载、四窗选择、真实 conditioner/backbone 前向、真实 G1 FK、真实 CUDA 数值与 RNG 均 `not_run`。
2. `fk_m2` 现在由保存的两侧 FK 点独立重算元素并核对聚合/分母/权重，但仍只覆盖“用生产 FK 点复算聚合”；真实 FK 几何、解码到 FK 正确性及导数 `not_audited`。
3. 网络参数梯度、累积、裁剪、AdamW、更新、选模、DDIM 采样均 `not_audited`；本轮 `R3=not_run`。
4. 独立参考与 F 共享同一套独立公式实现；F 独立于生产路径并重算 raw 数组，但不是第二个独立公式实现。该边界需在真实预注册复核时决定是否另写参考。
5. RNG 状态恢复与重放绑定已在纯合成路径验证；真实模型构造/前向是否消耗 RNG、CUDA RNG 设备数变化、真实非确定性算子在 `deterministic_algorithms` 关闭时的逐位可复现性仍 `not_run`。F 要求 production gradient 逐位一致；真实 GPU 若无法满足，会登记 mismatch 而非放宽。
6. 校准证据由纯合成重跑复验（同 seed、无真实资产）；这证明契约未被删除/替换/协调放宽，不证明真实 float32/float64 差异落在校准阈值内。
5. 近 π 的 float32 `so3_log` 精度限制仍在：合成夹具避开该分支；真实四窗若落入独立参考域外，会按 `independent_rotation_domain` 记 gap，不能当作通过。
6. float32 标量容差对 `geodesic_rad2/total/L5` 有登记 override（2e-3/1e-3）；这是精度允许，不是结构性放宽。梯度容差仍使用冻结的 `gradient_float32`。
7. 合成 selftest 使用替身模型/合成 FK，不能证明真实模型前向、真实 FK 或真实训练目标在主训练数据上的行为。

## 8. 下一授权门

真实执行前仍须完成草案 §8 的全部事项：用户明确授权真实资产加载/四窗前向/输出空间求导；确认 cohort/历史/预算/attempt 上限；确认登记/封口/消费者/父 unit 准入沙箱反例；确认生产 observer 与独立参考、batched VJP、合成反例；用 `--calibrate-synthetic` 生成并冻结容差契约；再在 R0 登记下核验四项固定资产、四窗完整键、配置、实现快照、环境与 schema。**本记录不勾选这些事项，也不创建 cohort 或预算。**

## 9. 独立复核修复（2026-10-26）

复核在已提交实现上确认 8 P1 + 2 P2，并全部修复；权威合成运行重跑为 `attempt_007`，结论仍 `synthetic_contract_passed`，真实执行仍未启动。

| # | 复核问题 | 修复 | 反例 |
| --- | --- | --- | --- |
| 1 | [P1] F 额外独立 VJP 未计数，实际 production 4/independent 6、报告仍记 8 次 | 删除 F 的直接 `independent_vjp`；复用重放 unit 的 independent 梯度；所有 conditioner/model/production VJP/independent VJP 在实际调用处计数，失败也占用 | `test_f_counts_every_actual_vjp`：F 实测 production/independent 各 2，快照各 2；全链 4/4 |
| 2 | [P1] F 只比较内存 `p["units"]` 与范数摘要，落盘梯度反号仍通过 | F 从 P report/数组/梯度 NPZ 读取；逐元素比较 fresh replay 与落盘 production/independent 梯度（production 逐位） | `test_f_binds_disk_gradients_not_in_memory_units`：两侧梯度反号被拒 |
| 3 | [P1] FK 聚合复用保存的 `fk_elements`，预测点整体偏移 1m 仍通过 | 从保存的两侧 FK 点重算 `((pred-target)^2).sum(-1)`，先核对元素，再用重算元素生成行、分母与聚合 | `test_f_recomputes_fk_from_saved_points`：点偏移被拒；元素/分母/行均重推 |
| 4 | [P1] 重试重置调用额度，两次完整运行均获全额 | `call_balance` 视为累计快照；`prior_call_counts` 跨 attempt/run 取逐阶段单调最大值并扣除；缺 balance/ledger/盘登记时 unknown，R0 前停止 | `test_retry_cannot_reset_call_quota`、`test_prior_call_counts_are_cohort_wide_and_fail_closed` |
| 5 | [P1] F 未恢复 P 的 RNG，只比较哈希；构造器额外抽样即四项 mismatch | 保存可恢复的 Python/NumPy/Torch CPU/CUDA 状态；每个 replay unit 前恢复；重放后逐状态与哈希对账 | `test_f_restores_rng_before_replay`：fresh 构造器额外抽样仍通过 |
| 6 | [P1] 合法负结果以 incomplete 返回，F 不执行 | `SealResult.terminal` 区分合法封口的部分证据与未知历史；P/F 的 terminal incomplete 继续下一阶段；F 只重放已测量 unit 并保留 mismatch | `test_valid_negative_result_can_enter_f`：w_pos=2 负结果进入 F，P 停于 deterministic，F 保留 mismatch |
| 7 | [P1] 校准证据未核验，删除证据/替换哈希/放宽阈值仍通过 | 重算 evidence hash；强制 4× observed 推导关系；用记录 seed 重跑纯合成校准并比对 evidence/阈值；默认契约强制等于冻结默认值 | `test_calibration_evidence_is_reverified`：删除、放宽、协调伪造 observed 均被拒 |
| 8 | [P1] 历史接受负成本，预算被“补回” | `stage_history`/`stage_state`/`upsert_cost` 要求有限非负；负值使历史 unknown 并阻止新 attempt | `test_negative_costs_are_refused_by_both_entries` |
| 9 | [P2] 额外行、分母/权重 999999 不进入最终判据 | 磁盘行多重集进入 gaps；逐行核验 `denominator/weight/aggregation_only/window/scope/model/term/status/within_tolerance` 与完整字段；派生比较字段按行内算术自洽校验 | `test_f_rejects_extra_rows_and_row_field_tampering` |
| 10 | [P2] 字符串数组触发 `np.isfinite` TypeError 中断复核 | `_check_arrays` 先做 shape/dtype/数值类型分派，再检查有限性；非数值数组记 `array_not_numeric` 并继续写 incomplete 报告 | `test_non_numeric_arrays_do_not_abort_verification`、`test_check_arrays_string_dtype_is_a_normal_problem` |

复核修复后的权威合成结果：objective **98** 行、gradient **14** 行，`problems=[]`、`gaps=[]`、`complete_evidence=true`；R0/P/F 均 done/complete；调用 **4/4/4/4**、contact 0、训练更新/参数 FD/DDIM/checkpoint 写入 0；`audit_summary.json` sha `bdfe267c…`、`independent_verification.json` sha `74d0e8a7…`、`p_report.json` sha `3f2dd980…`。**仍未加载真实资产、未运行真实审计、未创建 cohort/预算/执行注册。**

## 10. 第二轮独立复核修复（2026-10-26）

第二轮只读复核在首轮修复后的实现上确认 3 P1 + 2 P2，并全部修复。权威合成运行重跑为 `attempt_009`，结论仍 `synthetic_contract_passed`；真实执行仍未启动。

| # | 复核问题 | 修复 | 反例 |
| --- | --- | --- | --- |
| 1 | [P1] 预算撤销产生的 P terminal incomplete 仍进入 F 重放，两模型前向/求导白花 | `_p_budget_exhausted` 识别 P 封口的 `budget_exceeded_*`/`seal_tail_reserve_*` 及 `unit_deadline`/`AuditCallBudgetExceeded` gap；F 以 `replay=False` 只整理已有证据，不构造模型、不做前向或 VJP | `test_budget_revoked_p_does_not_start_f_model_calls`：P 被 `budget_exceeded_after_registration` 撤销后，F 调用计数停在 P 的 2/2/2/2，不再增加 |
| 2 | [P1] F 只在 unit 准入时比较子/父截止，不检查当前时间；首单元返回后已超时，第二模型仍执行 | 每个 replay unit 前检查 `time.perf_counter() + replay_reserve_s >= deadline`；`replay_reserve_s` 取阶段 `reserve_s`（缺省回落 `SEAL_TAIL_RESERVE_S`）。耗尽即置 `replay_enabled=False`，后续 unit 只记 gap，不新增调用 | `test_f_live_deadline_stops_new_model_calls`：首单元后睡眠越过截止，第二模型未执行，`calls["model"]==1` |
| 3 | [P1] F 封口失败只修改内存 summary；返回 audit_incomplete，但磁盘仍为 `synthetic_contract_passed/complete_evidence=true` | `_withdraw_pass_claim` 从磁盘读取 summary，改写为 audit_incomplete、complete_evidence/covered_contract_passed=false，追加撤回原因；`_report_hashes` 重算 `report_sha256`/`summary_sha256`，经 `_persist_row_entry` 原位更新索引与台账。撤回/重绑失败时行改 `publishing/complete=false`，保持不可消费 | `test_f_seal_timeout_revokes_on_disk_pass_claim`：F 封口注入 0s 预算后，磁盘 summary 与返回 summary 均为 audit_incomplete，`stage_state` 无 report/summary hash mismatch，行保留 `budget_exceeded_*` 原因 |
| 4 | [P2] 入口只拒绝已耗尽的调用额度；历史每类用 1 次、剩 3 次仍启动需要 4 次的完整链，执行 P 两单元及 F 首单元后才抛 `AuditCallBudgetExceeded` | 新增 `FULL_CHAIN_CALLS={每类:4}`；在写 cohort、分配 R0 之前检查 `prior + required <= cap`，不足即 `SystemExit`，不启动任何阶段 | `test_insufficient_call_quota_stops_before_any_stage`：历史 1/1/1/1 时入口拒绝，索引仍只有原 R0 行，P/F 目录不存在，未进入任何阶段或模型构造 |
| 5 | [P2] 只校验 P report 顶层映射；`units` 改为包含模型名的 list 时成员检查通过，随后 `.get` 抛 `AttributeError`，F 未写报告并留下 running | 读取成员前先校验 `units`、`unit_artifacts`、admission、`arrays_sha256`、call trace/counts 的容器类型；异常容器记 gap。缺失 artifact 声明记 gap，只有声明存在且哈希不符才记 problem | `test_f_nested_container_types_are_gaps_not_crashes`（`units` list / `unit_artifacts` list 两参数）：均写 incomplete 报告，不崩溃 |

第二轮权威合成产物：`derived/audit/w1_real_objective_audit/attempt_009/`。objective **98** 行、gradient **14** 行、`problems=[]`、`gaps=[]`、`complete_evidence=true`、调用 **4/4/4/4**、contact 0、训练更新/参数 FD/DDIM/checkpoint 写入 0；`audit_summary.json` sha `098b5067…`、`independent_verification.json` sha `74d0e8a7…`、`p_report.json` sha `64eda19f…`，两份 NPZ 与首轮逐位相同。**仍未加载真实资产、未运行真实审计、未创建 cohort/预算/执行注册；`objective_coverage=partial`、`full_objective_contract_passed=null`、`attribution=attribution_insufficient`、最终交互 `not_ready`、生产接入 false 不变。**

## 11. 第三轮独立复核修复（2026-10-26）

第三轮只读复核在第二轮修复后的实现上确认 2 P1 + 1 P2，并全部修复。权威合成运行重跑为 `attempt_013`，结论仍 `synthetic_contract_passed`；真实执行仍未启动。

| # | 复核问题 | 修复 | 反例 |
| --- | --- | --- | --- |
| 1 | [P1] `_withdraw_pass_claim` 在 `finish_attempt` 之后重写报告、哈希、索引和台账，却复用旧 `measured_wall_s`；注入 1.3s 撤回写盘延迟后 F 实际 1.429s、仅登记 1.113s，`stage_history` 仍接受 | 撤回写盘纳入同一 attempt 的成本覆盖循环：分别计时 summary 重写与 index/ledger 持久化，保留旧成本作为下限；实际耗时超过旧登记时按 `observed + reserve` 原位更新 `measured_wall_s/sampled_wall_s/seal_tail_reserve_s`，并循环核验最后一笔写盘是否被覆盖。剩余预算不足或核验失败时不发布终态，保持 `publishing/complete=false` 并登记 `withdraw_cost_unverifiable` | `test_withdrawal_write_cost_is_charged_to_same_attempt`：1.3s 撤回写盘后 `measured_wall_s >= 实际耗时` 且 `stage_history` 接受；`test_withdrawal_without_affordable_coverage_stays_publishing`：剩余 0.2s 时行保持 publishing，`stage_state` 报 `attempt_final_check_pending` |
| 2 | [P1] admission 未通过或输入绑定异常只加入 gaps/problems，未阻止后续模型构造与重放；完整注册链中把 `admission.passed` 改为 false 后 F 仍新增 conditioner/model/两类 VJP 各两次 | R0/P 的前置 `stage_state`、admission artifact/passed/registration、admission 数组哈希、noise/alpha/x_t 复现、独立 x0 重编码与 x0 容差全部作为重放硬门；任一失败在 model/loss factory 之前置 `replay_enabled=false`，只整理已有证据。前置块因 identity/registration/array/row 问题被跳过时，`fresh is None` 守卫也禁止进入 unit | `test_f_input_gate_blocks_full_chain_model_calls`：admission 篡改后全链调用停在 P 的 2/2/2/2；`test_f_binding_and_consumer_gate_skip_model_construction`（x0 绑定、P publishing 两参数）：`NoModelProvider` 的 factory 未被调用，F 记 `f_replay_input_gate_failed` |
| 3 | [P2] 负结果封口失败仍声明证据完整：扩散产生有效 mismatch 后 F 因超预算封口为 complete=false，但返回结果与磁盘 summary 仍为 `complete_evidence=true` | `_withdraw_pass_claim` 对所有 `complete_evidence=true` 的不可消费封口生效；`preserve_conclusion=true` 保留已测得的 mismatch 结论，仅撤回 `complete_evidence/covered_contract_passed`，追加 `f_evidence_completeness_withdrawn` 与封口原因并重绑报告/摘要哈希。pass 结论仍改写为 `audit_incomplete`；publishing/running 未知历史即使保留 producer mismatch，最终消费结论仍为 `audit_incomplete` | `test_f_negative_result_seal_failure_withdraws_completeness_only`：F 重放检测到 prediction mismatch，F 封口注入 0s 预算后磁盘/返回结论仍为 `synthetic_pipeline_mismatch`，但 `complete_evidence=false`，无 report/summary hash mismatch |

第三轮权威合成产物：`derived/audit/w1_real_objective_audit/attempt_013/`。objective **98** 行、gradient **14** 行、`problems=[]`、`gaps=[]`、`complete_evidence=true`、调用 **4/4/4/4**、contact 0、训练更新/参数 FD/DDIM/checkpoint 写入 0；`audit_summary.json` sha `a13d65bb…`、`independent_verification.json` sha `74d0e8a7…`、`p_report.json` sha `5b84bcf4…`。与 `attempt_009` 相比，两份 NPZ 与 `independent_verification.json` 逐位不变；summary/p_report 的差异仅来自实现身份、注册哈希、时间戳及 unit artifact/RNG facts 哈希。**仍未加载真实资产、未运行真实审计、未创建 cohort/预算/执行注册；`objective_coverage=partial`、`full_objective_contract_passed=null`、`attribution=attribution_insufficient`、最终交互 `not_ready`、生产接入 false 不变。**

## 12. 第四轮独立复核修复（2026-10-27）

第四轮只读复核在第三轮修复后的实现上确认 2 P1 + 1 P2，并全部修复；复核后另确认 1 项 P2 成本下限回归并修复。权威合成运行重跑为 `attempt_015`，结论仍 `synthetic_contract_passed`；真实执行仍未启动。

| # | 复核问题 | 修复 | 反例 |
| --- | --- | --- | --- |
| 1 | [P1] `_withdraw_pass_claim` 只把 summary 重写计入写盘延迟，`_persist_row_entry` 的 index/ledger 写入未计时且直接落终态；两次 1.3s 慢写盘后仍按 1s 预留登记（实际 2.6227s、登记 2.3199s），`stage_history` 接受被低估的终态成本 | `_persist_row_entry` 分别上报 index/ledger 写盘耗时；`persist` 再记录整次调用延迟并入 reserve。所有尚未证明覆盖的行先写 `publishing`，覆盖核验通过后才写终态并再次核验；终止或预算不足只会留下非可消费的 publishing 行 | `test_withdrawal_slow_persist_then_kill_stays_publishing_and_covered`：第二次 1.3s 写盘落盘后终止，磁盘为 `publishing/complete=false`，`measured_wall_s` 覆盖该写盘，`stage_history` 为 None |
| 2 | [P1] F 的 `verify_identity` 抛异常时只追加 gap，后续分支与重放硬门均未包含该异常，完整链中 model/VJP 调用仍从 P 的 2/2/2/2 增至 4/4/4/4 | 异常同时置 `replay_enabled=false`、记录 `f_replay_identity_gate_failed`；model/loss factory 与逐 unit 均不再执行，只整理已有证据 | `test_f_identity_exception_blocks_model_replay`：完整链第 2 次 `verify_identity` 抛 OSError，F 调用计数停在 2/2/2/2，gap 含 `replay_skipped:...:identity_unverifiable` |
| 3 | [P2] 撤回把行改为 `publishing` 后，最终裁决仍读取撤回前的 `seal.terminal`；负结果封口失败且撤回耗尽预算时磁盘为 publishing、历史为 None，返回值却仍为 `synthetic_pipeline_mismatch` 且 `seals.F.terminal=true` | 撤回后按盘上行状态与 `stage_state` 消费者核验重算 `terminal/sealed`（`attempt_not_complete` 之外的问题一律视为非终态）；未知封口返回 `audit_incomplete`，已测 mismatch 保留在证据中 | `test_f_negative_result_withdrawing_past_coverage_is_incomplete`：磁盘 summary 保留 `synthetic_pipeline_mismatch`、`complete_evidence=false`，返回结论 `audit_incomplete`，`seals.F.terminal/sealed=false` |
| 4 | [P2] 复核回归：撤回终态重写直接用 `observed + tail` 覆盖成本，未保留 `base_cost` 下限；旧封口因慢写盘登记 3.9099s，撤回后降为 2.3126s，`stage_history` 接受下降值，等于回退保守计费并增加预算余额 | 终态与刷新统一使用 `max(base_cost, observed + tail)`，预算可负担性也按该值判定；旧登记成本只增不减 | `test_withdrawal_never_lowers_the_registered_cost`：旧登记 3.9099s、实际 elapsed 很小时撤回后 `measured_wall_s` 与 `stage_history` 仍 ≥3.9099s |

第四轮权威合成产物：`derived/audit/w1_real_objective_audit/attempt_015/`。objective **98** 行、gradient **14** 行、`problems=[]`、`gaps=[]`、`complete_evidence=true`、调用 **4/4/4/4**、contact 0、训练更新/参数 FD/DDIM/checkpoint 写入 0；`audit_summary.json` sha `a27c06c0…`、`independent_verification.json` sha `74d0e8a7…`、`p_report.json` sha `b05512d6…`。与 `attempt_014` 相比，两份 NPZ 与 `independent_verification.json` 逐位不变；summary/p_report 的差异仅来自实现身份、注册哈希与时间戳。**仍未加载真实资产、未运行真实审计、未创建 cohort/预算/执行注册；`objective_coverage=partial`、`full_objective_contract_passed=null`、`attribution=attribution_insufficient`、最终交互 `not_ready`、生产接入 false 不变。**

## 13. 准入包与冻结校准契约（2026-10-27）

本轮交付**可审阅的真实审计准入包与冻结校准契约**，对应草案 §8 的第 4–6 项工程材料。**不是授权、不是注册、不是真实运行**：§8 八项仍未勾选，真实 dataset/cache/checkpoint 未加载，真实模型前向/反向、真实 G1 FK 与真实数据更新均为 0，未创建 cohort/预算/执行注册。详见[准入包文档](../plans/omniretarget_w1_real_objective_audit_admission_package.md)。执行记录 `derived/audit/w1_real_objective_admission/20261027T080142Z_ed143bc/`：`report.md` SHA-256 `70a1114de4c7592a5f8adaf859ca313c806da24ba4e73955fe4692f39c20f708`；`integrity.json` SHA-256 `e1998ef8…`（384 个旧证据文件清单、11 个引用资产、attempt_015 五项权威哈希、无真实运行目录）。

### 13.1 冻结合成校准

| 项目 | 结果 |
| --- | --- |
| 命令 | `uv run --locked python scripts/audit_w1_real_objective.py --calibrate-synthetic --tolerance-out derived/audit/w1_real_objective_audit/tolerance_contract.json` |
| 契约文件 | SHA-256 `2904fbc41113ba379cee2e4de4704befcac3ae8bb69c3178bcf141e48fd38eb3`；路径 `derived/audit/w1_real_objective_audit/tolerance_contract.json` |
| 模式/种子 | `synthetic_calibrated` / `synthetic_only=true` / seed `20261026` |
| 证据 | `evidence_sha256=49371d30a0a03679a1352e32db4b9751cd4988d92359bd6fc63324ee8995852d`；`calibration_sha256` 与其相等 |
| 证据规模 | objective **98** 行、gradient **14** 行；conditioner/model/production VJP/independent VJP 各 **2**；`real_dataset_loaded=false`、`real_checkpoint_loaded=false`、`real_model_calls=0` |
| 观测 | encode `9.082717760122705e-7`、loss `4.217589067052885e-9`、gradient `2.7271278616966023e-9`（atol） |
| 推导 | `max(默认, 4×观测)`；三个观测键的 4× 值均低于默认，**无任何键被抬高**；`scale_floor`/`rotation_domain` 与 `term_overrides` 不变 |
| 独立核验 | 记录目录 `verify_calibration.py` 重算规范证据哈希、在模块外重述默认表与 4× 规则、在新解释器重跑校准；`passed=true`、`problems=[]`、`keys_widened_above_default=[]` |
| 反例 | P2 后的 59 项测试中，删除证据、替换哈希并放宽、同步伪造 observed 均被 `tolerance_contract_problems` 拒绝 |

观测相对误差中的 loss `0.977` 是近零分母比值，冻结规则只用绝对项，不能读成 97% 容差。该契约证明阈值在真实数值打开前已冻结且未被删除/替换/协调放宽；不证明真实 float32/float64 差异、CUDA 确定性与 G1 FK 行为落在阈值内。

### 13.2 准入包

| 项目 | 结果 |
| --- | --- |
| 实现 | [src/ireset/eval/w1_real_admission.py](../../src/ireset/eval/w1_real_admission.py)、[scripts/w1_real_objective_admission.py](../../scripts/w1_real_objective_admission.py)、[tests/test_w1_real_admission.py](../../tests/test_w1_real_admission.py) |
| §13 生成时的包（已被 §14 取代） | `derived/audit/w1_real_objective_audit/admission/attempt_003/admission_package.json`，SHA-256 `2cc733d3394497c92e4e407fd6a627ad0d1936496d7bba3a6b09caf7b3f79cfe`；索引 SHA `5a9da0ae…` |
| 生成器绑定 | 包内 `generator.files`：`w1_real_admission.py` `258b60dc…`、`w1_real_objective_admission.py` `299904ba…`、`test_w1_real_admission.py` `bde54657…`；校准契约 SHA 同前 |
| §13 结论 | `passed=true`、`problems=[]`、`authorized=false`、`registered=false`、`run=not_run`、`section_8_checked=false` |
| 候选四窗 | `sub10_largebox_063` T=133、`sub10_largebox_085` T=148，各 start `[0,30]`、end `[60,90]`；按冻结字典序均为 `sub10` |
| 固定身份 | manifest `a350a8b8…`、W1 注册 `5561f685…`、W1 final `7f1a9a59…`、预测注册 `5fcf8b3a…`、预测 final `ce9843fd…`、prereg `071c49d3…`、config `0e2cfa96…`、det ckpt `7b1efd57…`、diff ckpt `1f9c16b1…`、cache identity `90af39f1…`、root_stats `b9d9cc04…`，全部与登记相等 |
| t/noise/alpha | `[0,250,500,999]`；noise seed `20261008` sha `cb09992b…`；alpha sha `9f3be039…` |
| 行集 | objective **98**、gradient **14**；完整键表写入包内 |
| 预算/上限 | R0/P/F 180/360/360s、各 2 次、预留 30/60/60s、合计 900s；调用各 **4**、全链 4、batched VJP 合计 **8** |
| 只读范围 | `metadata_only=true`、`window_values_loaded=false`、`schedules_read=false`、`checkpoint_torch_load=false`、模型/conditioner/VJP 调用 0、未创建 cohort/注册 attempt |
| 审阅注意事项 | ①两 demo 同属 `sub10`，不声明跨受试者覆盖；②`deterministic_algorithms=false`，F 要求逐位重放，失败登记 mismatch 而不放宽 |

`attempt_001`（SHA `7b709112…`，未含审阅注意事项）、`attempt_002`（SHA `6d98a6cc…`，未绑定生成器）与 `attempt_003`（SHA `2cc733d3…`，P2 前）保留为被取代证据；P2 后当前准入包为 `attempt_004`，见 §14。包内资产**文件哈希**已核对；资产闭包、checkpoint step/可用性/有限性与 strict load 明确延期到 R0。

### 13.3 测试与执行门

- `uv run --locked pytest tests/test_w1_real_admission.py -q` -> **8 passed（5.08s）**：固定矩阵/只读范围、默认与放宽契约拒绝、checkpoint 漂移、manifest 漂移、motion 排除顺序、单一合格 demo 的 incomplete、授权模板非授权、包目录不覆盖。
- `uv run --locked pytest tests/test_w1_real_objective_audit.py -q` -> **58 passed（28.31s）**（当时未改被审实现，重跑确认）；P2 后为 **59 passed（28.72s）**，见 §14。
- 无授权真实入口：exit 1，`--real requires --authorization (section 8 is not checked)`，探测目录未创建。
- `authorized=false` 模板：exit 1，`not_authorized` + 11 项 `authorization_checklist:*`，探测目录未创建。

### 13.4 边界

`source=synthetic_only`（校准）＋ `metadata_only`（准入包）；真实资产/checkpoint/模型调用与真实数据更新 0；未创建 cohort/预算/执行注册；§8 八项未勾选；`objective_coverage=partial`、`full_objective_contract_passed=null`、`attribution=attribution_insufficient`、`factor_selected=false`、`auto_execute=false`、最终交互 `not_ready`、生产接入 false。下一交付须由用户明确授权后按 R0→P→F 执行；本记录不勾选 §8。

## 14. P2 授权记录严格化（2026-10-27）

用户复核指出 `authorization_problems` 偏宽：未检查 `schema`，预算/次数经 `float()`/`int()` 转换后比较，且允许 `ddim_calls=None`。本轮已修复并重新取得合成与准入包证据；**仍未授权、未注册、未运行**。执行记录 `derived/audit/w1_real_objective_admission/20261027T113741Z_5d022ca_p2/`（`report.md` SHA-256 `cf40b64d…`）。

| 项目 | 结果 |
| --- | --- |
| 代码身份 | HEAD `5d022cafe4cb0e958e26c72fbbaf8a7afef02438`；P2 轮开始时 clean |
| 修复 | 新增 `AUTHORIZATION_SCHEMA` 精确校验；三个计数要求 JSON 整数 0（拒绝 `0.0`/`False`/`"0"`/`None`）；`budget_s` 要求精确 JSON 数字（非 bool/字符串/null/NaN）；`attempts` 要求精确 JSON 整数；`budgets` 键集恰为 `R0/P/F` |
| 测试 | `tests/test_w1_real_objective_audit.py` **59 passed（28.72s）**；`tests/test_w1_real_admission.py` **8 passed（5.09s）**；新反例覆盖 28 类变异 + 2 个正向用例 |
| 合成重验 | `attempt_016`：`synthetic_contract_passed`、98/14 行、0失败/0缺口、调用 4/4/4/4；两份 NPZ 与 `attempt_015` 逐位相同，`independent_verification.json` 相同 |
| 当前准入包 | `admission/attempt_004/`，SHA-256 `1cedbb159b8bd1892c010e8b6038e384d53452bc435dc94658c6d08a0ee48193`；索引 `ca1e5fd0…`；`passed=true/problems=[]/authorized=false/run=not_run/section_8_checked=false` |
| 实现/生成器绑定 | audit module `ccd96bfc…`、tests `2ae792e7…`；admission module `d5486b3c…`、entry `299904ba…`、admission tests `bde54657…` |
| 模板 | SHA `2e4aabbc…`，`authorized=false`；真实入口仍 exit 1（`not_authorized` + 11 项 checklist），探测目录未创建 |
| 校准契约 | 文件 SHA `2904fbc4…` 与证据 `49371d30…` 未变；新模块下 `tolerance_contract_problems=[]` |
| 保护 | 384 个旧审计文件清单；`attempt_016`/`attempt_015` 五项权威哈希全部相等；11 个引用资产相等；无真实运行目录 |
| 被取代 | `admission/attempt_003` 为 P2 前包，保留为历史证据 |

边界不变：§8 八项未勾选；真实资产/checkpoint/模型前向反向/真实 G1 FK/真实数据更新为 0；未创建 cohort/预算/注册；`objective_coverage=partial`、`full_objective_contract_passed=null`、`attribution=attribution_insufficient`、`factor_selected=false`、`auto_execute=false`、最终交互 `not_ready`、生产接入 false。真实执行仍须用户明确授权后按 R0→P→F 进行。
