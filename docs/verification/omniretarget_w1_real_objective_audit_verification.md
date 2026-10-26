# W1 真实训练目标审计实现与合成自测验证记录

日期：2026-10-26（初版实现与合成自测；同日完成独立复核 8 P1 + 2 P2 修复）。对应草案：[W1 真实训练目标审计预注册草案](../plans/omniretarget_w1_real_objective_audit_prereg_draft.md)。**本次交付是实现与纯合成自测，不是真实数据审计、预注册执行或训练更新。** 草案仍为 `draft / not_registered / not_authorized / not_run`；`factor_selected=false`、`auto_execute=false`、最终交互 `not_ready`、生产接入 `false`。

## 0. 结论与边界

| 项目 | 实际结果 |
| --- | --- |
| 实现状态 | complete（R0/P/F 骨架、注册/准入、生产取证、batched VJP、独立参考、F 重放、账本/封口、CLI） |
| 合成自测 | `synthetic_contract_passed`；`problems=[]`、`gaps=[]`、`complete_evidence=true` |
| 权威合成命令 | `uv run --locked python scripts/audit_w1_real_objective.py --synthetic --out derived/audit/w1_real_objective_audit` |
| 权威合成产物 | `derived/audit/w1_real_objective_audit/attempt_007/`（`attempt_001`–`attempt_006` 与顶层旧目录为更早实现运行，保留不改写） |
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
| [tests/test_w1_real_objective_audit.py](../../tests/test_w1_real_objective_audit.py) | **42 项**正向/反例：完整链路、observer 逐位透明、detach 拒绝、噪声公式、近 π 域、键多重集、账本/running/publishing/超限/封口写失败、父 unit 准入、F 重放篡改、CLI 拒绝；复核新增实际 VJP 计数、落盘梯度逐元素绑定、FK 点重算、跨 run 调用额度、RNG 恢复、合法负结果进入 F、校准证据重验、负成本、行字段/额外行、非数值数组 |
| [src/ireset/train/losses.py](../../src/ireset/train/losses.py) | `LossAssembler.__call__` 新增可选 `observer`：捕获 live 分项张量、FK 前 `L5`、归约中间量、预测/目标 FK 点；`observer=None` 数值路径逐位不变 |

实现文件 SHA-256（合成产物写入时）：

| 文件 | SHA-256 |
| --- | --- |
| `src/ireset/train/losses.py` | `04b74a45ff59c5acbf8b1fda9361b58c9e56d5466990122524a808071ab80820` |
| `src/ireset/eval/w1_real_objective_audit.py` | `d4ba125dda0ee4c2d070277dd29de3ad3b6a79fe4e51685a3a34487dbe308df1` |
| `scripts/audit_w1_real_objective.py` | `6fa2584845decbc2dcb4244b4a11d60f92437a0e273a780b1a206d89c3498cde` |
| `tests/test_w1_real_objective_audit.py` | `e3d5e2d613ba5766580a584ad4c8741d83cc01e8cddefb5e0872750900fd6712` |

权威产物哈希：

| 产物 | SHA-256 |
| --- | --- |
| `.../finalize/attempt_001/audit_summary.json` | `bdfe267c536e6ee1526510ceeb22d15ca1ae8c036a0ae435e4348db67d1a0e2d` |
| `.../finalize/attempt_001/independent_verification.json` | `74d0e8a78f988fe2da48c3066ecb063a24dca35f9c91b0af09a4307ad2c46500` |
| `.../production/attempt_001/p_report.json` | `3f2dd980300bd6a137e25b64ac8884152d3a6dba745c73755d54c3c3d66eae20` |
| `.../production/attempt_001/objective_real_arrays.npz` | `932f2f0cc3ef0c7bdaae2bdb34cb5361a17dee468df4699e45c80144a619f0cc` |
| `.../production/attempt_001/gradient_real_arrays.npz` | `73cc59a3d2dfb4bfef26d11977dd8488b29d457eb7390556cd5fdd241f890a88` |

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

- `uv run --locked pytest tests/test_w1_real_objective_audit.py -q` -> **42 passed（14.62s）**。
- 相关既有回归：`tests/test_w1_training_objective_audit.py tests/test_generation.py` -> **51 passed（33.85s）**；`tests/test_omniretarget_w1.py` -> **34 passed（40.33s）**；`tests/test_w1_de_numerics.py tests/test_omniretarget_w1_de.py` -> **73 passed（326.99s）**。
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
