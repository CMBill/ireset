# W1 真实训练目标审计实现与合成自测验证记录

日期：2026-10-26。对应草案：[W1 真实训练目标审计预注册草案](../plans/omniretarget_w1_real_objective_audit_prereg_draft.md)。**本次交付是实现与纯合成自测，不是真实数据审计、预注册执行或训练更新。** 草案仍为 `draft / not_registered / not_authorized / not_run`；`factor_selected=false`、`auto_execute=false`、最终交互 `not_ready`、生产接入 `false`。

## 0. 结论与边界

| 项目 | 实际结果 |
| --- | --- |
| 实现状态 | complete（R0/P/F 骨架、注册/准入、生产取证、batched VJP、独立参考、F 重放、账本/封口、CLI） |
| 合成自测 | `synthetic_contract_passed`；`problems=[]`、`gaps=[]`、`complete_evidence=true` |
| 权威合成命令 | `uv run --locked python scripts/audit_w1_real_objective.py --synthetic --out derived/audit/w1_real_objective_audit` |
| 权威合成产物 | `derived/audit/w1_real_objective_audit/attempt_003/`（`attempt_001/` 与顶层旧目录为更早实现运行，保留不改写） |
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
| [tests/test_w1_real_objective_audit.py](../../tests/test_w1_real_objective_audit.py) | **30 项**正向/反例：完整链路、observer 逐位透明、detach 拒绝、噪声公式、近 π 域、键多重集、账本/running/publishing/超限/封口写失败、父 unit 准入、F 重放篡改、CLI 拒绝 |
| [src/ireset/train/losses.py](../../src/ireset/train/losses.py) | `LossAssembler.__call__` 新增可选 `observer`：捕获 live 分项张量、FK 前 `L5`、归约中间量、预测/目标 FK 点；`observer=None` 数值路径逐位不变 |

实现文件 SHA-256（合成产物写入时）：

| 文件 | SHA-256 |
| --- | --- |
| `src/ireset/train/losses.py` | `04b74a45ff59c5acbf8b1fda9361b58c9e56d5466990122524a808071ab80820` |
| `src/ireset/eval/w1_real_objective_audit.py` | `0d345a5f67df6241ca82af72f6db6bdd42ac72cfdf486a370c2ba08696d15274` |
| `scripts/audit_w1_real_objective.py` | `6fa2584845decbc2dcb4244b4a11d60f92437a0e273a780b1a206d89c3498cde` |
| `tests/test_w1_real_objective_audit.py` | `15a56a093ca2afdd18a11e2273909488ddeb319999904abfb694fdc99e111062` |

权威产物哈希：

| 产物 | SHA-256 |
| --- | --- |
| `.../finalize/attempt_001/audit_summary.json` | `0ff9007d14a221038542541afba5bfadd03647b2909621ba7a1b9ca192d2ecee` |
| `.../finalize/attempt_001/independent_verification.json` | `5f03c04361b306945e300cf7d16c4bfbe714b7b51081d566c3a508e9b82fa79a` |
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
- 每模型一次 **batched VJP** 从原图取 `[6,4,60,38]`：`pos/rot6d/joint/geodesic_rad2/vel/L5`；不 detach 重建、不重拼损失。
- 独立参考用 float64：独立 6D Gram-Schmidt、独立相对旋转角（`theta=2 asin(norm(v))`）、五个分项、`L5`、`total`、FK 聚合（只用生产保存的 FK 点，`aggregation_only`）。
- 行表：每模型×4 窗×(7 项+L5) + batch×(7 项+L5) + L5/total 算术 + 7 项窗均值对账 = 49/模型；梯度 7/模型（6 切片+L5 加权和）。
- 合成结果全部 `within_tolerance`；生产与独立梯度逐切片在冻结容差内。`fk_m2` 只审计聚合，真实 FK 几何/导数仍 `not_audited`。

## 4. F：独立重放与收口

- F 先核验 R0/P `stage_state`、注册/容差 artifact 哈希、identity/implementation 可复现、admission 数组哈希、P call trace、父 unit 准入；缺前置写 gaps，不补跑 P。
- 从 raw NPZ 重建四窗、x0、noise/alpha/x_t；重新 strict 生成 fresh 模型/替身并执行一次同输入前向+双 VJP；逐位比较 P 的 prediction/x_t/production gradient，独立重算 independent gradient 与全部行表。
- 合成重放全部 `prediction_bitwise/x_t_bitwise/production_grad_bitwise` 通过；`problems=[]`、`gaps=[]`。
- 篡改反例（预测 +0.1、production gradient 清零、x_t 清零、声明 gradient mismatch、call count 99）均被 F 拒绝；CLI 真实入口在缺少授权/校准契约时拒绝且不创建 run 目录。

## 5. 账本、预算与封口

- `cohort.json` 冻结 `cohort_id`、`accounting_schema=pre_registered_v1`、三阶段预算/尝试上限、总预算与调用上限。
- 每次 attempt 先写 running/complete=false 索引+台账；`finish_attempt` 采用 pending→publishing→seal，终态发布前不可消费；封口预留、封口后截止/成本覆盖检查、撤销失败保持 publishing/complete=false 且返回失败。
- `stage_state` 拒绝 running/publishing/非完成/成本不一致/工件哈希不符；`stage_history` 要求盘×索引×台账 attempt 集合相等且原始计费值与舍入值对账。
- 合成链 R0/P/F 均 `done/complete=true`；索引/台账含 measured、sampled、seal tail、budget、call balance。
- 反例：running/publishing 不可消费；三次三方不一致为 unknown；预算 0s 封口为 incomplete；封口写失败不发布 done；父 unit 必须 active running、scope/注册/截止匹配且输出在父 `units/<id>/`。

## 6. 测试与回归

- `uv run --locked pytest tests/test_w1_real_objective_audit.py -q` -> **30 passed**。
- 相关既有回归：`tests/test_w1_training_objective_audit.py tests/test_omniretarget_w1.py tests/test_generation.py` -> **85 passed**；`tests/test_w1_training_objective_audit.py tests/test_w1_de_numerics.py` -> **78 passed**。
- **全项目套件本轮未重跑**；旧 947 passed + 1 skipped 仅作历史范围记录，不冒充本次结果。
- 合成校准命令 `--calibrate-synthetic` 生成 `mode=synthetic_calibrated`、非空 `evidence_sha256`、`observed` 与 98/14 行计数的契约；真实入口要求该模式，默认 `synthetic_default` 契约被拒绝。

## 7. 已知限制与未执行项

1. 真实 provider 未运行：真实 dataset/cache/checkpoint 严格加载、四窗选择、真实 conditioner/backbone 前向、真实 G1 FK、真实 CUDA 数值与 RNG 均 `not_run`。
2. `fk_m2` 只覆盖“用生产 FK 点复算聚合与权重”；真实 FK 几何、解码到 FK 正确性及导数 `not_audited`。
3. 网络参数梯度、累积、裁剪、AdamW、更新、选模、DDIM 采样均 `not_audited`；本轮 `R3=not_run`。
4. 独立参考与 F 共享同一套独立公式实现；F 独立于生产路径并重算 raw 数组，但不是第二个独立公式实现。该边界需在真实预注册复核时决定是否另写参考。
5. 近 π 的 float32 `so3_log` 精度限制仍在：合成夹具避开该分支；真实四窗若落入独立参考域外，会按 `independent_rotation_domain` 记 gap，不能当作通过。
6. float32 标量容差对 `geodesic_rad2/total/L5` 有登记 override（2e-3/1e-3）；这是精度允许，不是结构性放宽。梯度容差仍使用冻结的 `gradient_float32`。
7. 合成 selftest 使用替身模型/合成 FK，不能证明真实模型前向、真实 FK 或真实训练目标在主训练数据上的行为。

## 8. 下一授权门

真实执行前仍须完成草案 §8 的全部事项：用户明确授权真实资产加载/四窗前向/输出空间求导；确认 cohort/历史/预算/attempt 上限；确认登记/封口/消费者/父 unit 准入沙箱反例；确认生产 observer 与独立参考、batched VJP、合成反例；用 `--calibrate-synthetic` 生成并冻结容差契约；再在 R0 登记下核验四项固定资产、四窗完整键、配置、实现快照、环境与 schema。**本记录不勾选这些事项，也不创建 cohort 或预算。**
