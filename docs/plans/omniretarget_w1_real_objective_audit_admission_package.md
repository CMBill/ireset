# W1 真实训练目标审计准入包与冻结校准契约

日期：2026-10-27（P2 授权严格化后更新；同日用户授权并完成真实 R0→P→F）。状态：**准入包已生成、合成校准契约已冻结；真实四窗审计已执行一次并通过（覆盖范围 partial）**。上级草案：[W1 真实训练目标审计预注册草案](omniretarget_w1_real_objective_audit_prereg_draft.md) §8。实现与合成自测见[验证记录](../verification/omniretarget_w1_real_objective_audit_verification.md)；真实结果见[审计报告](../verification/omniretarget_w1_real_objective_audit_report.md)与验证记录 §15。

> 本文件仍是**准入材料**，不是执行授权；实际授权记录为 `derived/audit/w1_real_objective_audit/authorization/attempt_002/authorization.json`（SHA `1071810f…`）。用户授权后只执行了 R0→P→F 一次：R0/P/F 全封口 `done/complete=true`，结论 `real_batch_covered_contract_passed`，`problems=[]`、`gaps=[]`。更新、参数 FD、DDIM、checkpoint 写入、修复与生产接入仍为 0；`objective_coverage=partial`、`full_objective_contract_passed=null`、最终交互 `not_ready`。

## 1. 交付物与执行身份

| 项目 | 值 |
| --- | --- |
| 代码身份 | P2 时期 HEAD `5d022cafe4cb0e958e26c72fbbaf8a7afef02438`；真实执行 HEAD `b45cd7c654c2a2bb23d86fd76050f9ad651a7e67`（clean） |
| 环境 | Python 3.12.13；`uv.lock` `f809284e…`；torch 2.14.0+cu126；CUDA runtime 12.6；RTX 3090 |
| 训练配置 | `configs/train/omniretarget_motion.yaml` `0e2cfa96849ee2de14ac84b7343289090c606f78116bb79114ff81b046b521e0` |
| 冻结校准契约 | `derived/audit/w1_real_objective_audit/tolerance_contract.json`，文件 SHA-256 `2904fbc41113ba379cee2e4de4704befcac3ae8bb69c3178bcf141e48fd38eb3` |
| 校准证据 | `evidence_sha256=49371d30a0a03679a1352e32db4b9751cd4988d92359bd6fc63324ee8995852d` |
| 当前准入包 | `derived/audit/w1_real_objective_audit/admission/attempt_005/admission_package.json`，SHA-256 `1aadb296f1a94cd4bd39c654661712a6c46670fca0211ecf4a9b4452d86e45b8` |
| 包索引 | `.../admission/attempt_005/package_index.json`，SHA-256 `bdedc33c5b13668e840179890978860d3815ef89e700674f19356108e3513192` |
| 执行授权 | `derived/audit/w1_real_objective_audit/authorization/attempt_002/authorization.json`，SHA-256 `1071810f96325acd4a20d77395b7461864a8102f0268b10153c7e52908dcdfcb`；11 项 checklist 全 true、预算 180/360/360s、各 2 次、计数整数 0；`deterministic_algorithms=false` 显式冻结 |
| 真实运行 | `derived/experiments/omniretarget_first/20261027_b45cd7c_w1_real_objective_audit`；R0/P/F attempt_001 全 `done/complete=true`；结论 `real_batch_covered_contract_passed`，`problems=[]`、`gaps=[]`；总成本 7.5015s/900s |
| 执行记录 | `derived/audit/w1_real_objective_admission/20261027T113741Z_5d022ca_p2/`（P2）；`.../20261027T080142Z_ed143bc/` 为首轮准入包记录，保留 |
| 新增代码 | [w1_real_admission.py](../../src/ireset/eval/w1_real_admission.py)、[w1_real_objective_admission.py](../../scripts/w1_real_objective_admission.py)、[test_w1_real_admission.py](../../tests/test_w1_real_admission.py)；P2/执行前另修改 [w1_real_objective_audit.py](../../src/ireset/eval/w1_real_objective_audit.py) 与 [test_w1_real_objective_audit.py](../../tests/test_w1_real_objective_audit.py) |

`attempt_001`（SHA `7b709112…`，未含审阅注意事项）、`attempt_002`（SHA `6d98a6cc…`，未绑定生成器）、`attempt_003`（SHA `2cc733d3…`，P2 前）与 `attempt_004`（SHA `1cedbb15…`，P2 后、执行前）保留为被取代证据；`attempt_005` 为当前准入包。包内 `generator.files` 绑定生成器三个文件的 SHA-256（`d5486b3c…`、`299904ba…`、`bde54657…`）与校准契约 SHA；`implementation` 绑定 losses `04b74a45…`、审计模块 `4a2c2172…`、真实 CLI `6fa25848…`、60 项审计测试 `5646f0bb…`。P2 只收紧授权门（schema、三个计数、预算/次数类型与阶段集），不触审计数值路径：合成 `attempt_016` 仍为 `synthetic_contract_passed`，98/14 行、调用 4/4/4/4，两份 NPZ 与 `attempt_015` 逐位相同。

执行前修复提交 `b45cd7c`：无 `cohort.json` 且不可解析的旧历史目录不再使 cohort 历史变为未知；带本 cohort 标记的损坏台账仍 fail-closed。修复只影响 cohort 历史扫描，不改变目标/梯度/容差/窗口/预算语义。拒绝发生在建立 cohort 之前，无 attempt/计费/真实值/checkpoint/模型调用；修复后重新生成 `attempt_005` 并重发授权 `attempt_002`（范围与预算不变）。

生成命令：

```bash
uv run --locked python scripts/audit_w1_real_objective.py \
  --calibrate-synthetic \
  --tolerance-out derived/audit/w1_real_objective_audit/tolerance_contract.json

uv run --locked python scripts/w1_real_objective_admission.py \
  --tolerance-contract derived/audit/w1_real_objective_audit/tolerance_contract.json
```

## 2. 冻结合成校准契约

### 2.1 证据事实

| 字段 | 值 |
| --- | --- |
| 模式 | `synthetic_calibrated`，`synthetic_only=true` |
| 种子 | `20261026` |
| 证据行数 | objective **98**、gradient **14** |
| 校准调用 | conditioner/model/production VJP/independent VJP 各 **2**（上限 4） |
| 真实标记 | `real_dataset_loaded=false`、`real_checkpoint_loaded=false`、`real_model_calls=0` |
| 编码对拍 | float32 对 float64 `within_tolerance`，max abs **9.082717760122705e-7** |
| 生成时刻 | 2026-10-27T16:01:50+0800（实际时刻只记在证据与执行记录，不改变契约语义） |

### 2.2 阈值推导

规则：`frozen = max(文档默认值, 4 × 合成观测)`；`scale_floor` 与 `rotation_domain` 不由观测推导。观测值为：

| 比较项 | 合成观测 atol | 4× 观测 | 默认 atol/rtol | 冻结 atol/rtol | 是否被观测抬高 |
| --- | ---: | ---: | --- | --- | --- |
| `encode_float32` | 9.0827e-7 | 3.633e-6 | 2e-5 / 2e-5 | **2e-5 / 2e-5** | 否 |
| `loss_float32` | 4.2176e-9 | 1.687e-8 | 2e-5 / 2e-5 | **2e-5 / 2e-5** | 否 |
| `gradient_float32` | 2.7271e-9 | 1.091e-8 | 2e-5 / 2e-4 | **2e-5 / 2e-4** | 否 |

`encode_float64`、`loss_float64`、`gradient_float64`、`rotation_domain` 及 `term_overrides`（`geodesic_rad2` 2e-3、`total`/`L5` 1e-3）保持默认。**没有任何键被抬高**：本次校准确认默认阈值，而不是放宽它们。观测相对误差中 `loss_float32_max=0.977` 是接近零分母的比值，不是 97% 的容差；冻结规则只使用绝对项，不能把它读成相对容差。

### 2.3 独立核验

执行记录中的 [verify_calibration.py](../../derived/audit/w1_real_objective_admission/20261027T080142Z_ed143bc/verify_calibration.py) 不复用模块的推导函数：

1. 以规范 JSON 重算证据哈希，等于 `evidence_sha256` 与 `calibration_sha256`；
2. 在脚本内重述默认表与 4× 规则，逐键重算并核对契约阈值；
3. 在**新解释器**中用记录种子重跑 `synthetic_calibration`，证据与阈值逐值相等，生产者校验器返回 `[]`；
4. 结果 `passed=true`、`problems=[]`、`keys_widened_above_default=[]`。

58 项真实审计测试另覆盖反例：删除证据、替换哈希并放宽阈值、同步伪造 `observed` 后重算哈希，均被 `tolerance_contract_problems` 拒绝。

本契约证明的是“阈值在真实数值打开前已按纯合成证据冻结，且未被删除/替换/协调放宽”；它**不证明**真实 float32/float64 差异、真实 CUDA 确定性或真实 G1 FK 行为落在阈值内。P2 授权严格化后契约文件 SHA 与证据哈希不变；新模块下 `tolerance_contract_problems` 仍返回 `[]`（P2 执行记录 `integrity.json`）。

## 3. 固定审计矩阵

### 3.1 固定资产（文件哈希已在准入包中逐项核对）

| 资产 | 路径 | 登记/当前哈希 | 状态 |
| --- | --- | --- | --- |
| SOURCE 训练清单 | `.../20261012_21e5784_w1_a3bc_reviewfix8/w1_reviewfix8/audit/attempt_001/training_manifest.json` | `a350a8b8…` | 相等 |
| W1 执行注册 | `.../20261013_be00c50_w1_de/freeze/execution_registration.json` | `5561f685…` | 相等 |
| W1 最终裁决 | `.../20261013_be00c50_w1_de/decisions/attempt_001/w1_de_final.json` | `7f1a9a59…` | 相等 |
| 预测诊断注册 | `.../20261021_worktree_w1_prediction_consistency_r6/freeze/execution_registration.json` | `5fcf8b3a…` | 相等 |
| 预测诊断最终 | `.../decisions/attempt_001/prediction_consistency_final.json` | `ce9843fd…` | 相等 |
| 预注册 | `.../w1_reviewfix8/freeze/preregistration.json` | `071c49d3…` | 相等 |
| deterministic checkpoint | `.../formal/deterministic/attempt_001/best.pt` | `7b1efd57…`，step 4000 | 文件相等；step/可用性/有限性待 R0 strict load |
| diffusion checkpoint | `.../formal/diffusion/attempt_001/best.pt` | `1f9c16b1…`，step 4000 | 同上 |
| 缓存身份 | `.../w1_reviewfix8/cache/attempt_001/cache_identity.json` | `90af39f1…` | 相等 |
| train-only root stats | `.../cache/attempt_001/root_stats.npz` | `b9d9cc04…` | 相等 |

资产闭包（预注册资产、checkpoint 内部 step/可用性/有限性、模型/conditioner strict load）**未在准入包阶段核验**，由 R0 在登记下执行；这是 §8 第 7 项的执行内容，不是本包已完成的证据。

### 3.2 四个候选窗口

从冻结 train24 元数据按“demo 名规范字典序、motion 准入、每 demo 前两个完整 T=60 窗口”得到：

| # | demo | start | end | T | 可用整窗 | cache key（前 16） | input SHA（前 16） | derived SHA（前 16） |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | `sub10_largebox_063` | 0 | 60 | 133 | 3 | `29741e7b2df2b4af` | `81d9ad0d9d44e0c7` | `ee7722b9c05ce6f5` |
| 2 | `sub10_largebox_063` | 30 | 90 | 133 | 3 | `29741e7b2df2b4af` | `81d9ad0d9d44e0c7` | `ee7722b9c05ce6f5` |
| 3 | `sub10_largebox_085` | 0 | 60 | 148 | 3 | `60774514ffeafa4e` | `90d1102fd564864a` | `6f1cb597711624bc` |
| 4 | `sub10_largebox_085` | 30 | 90 | 148 | 3 | `60774514ffeafa4e` | `90d1102fd564864a` | `6f1cb597711624bc` |

两条审阅注意事项已写入包内：

- **同一受试者**：按冻结字典序，前两个合格 demo 均属 `sub10`。本审计只声明四窗上的实现符合性，不声明跨受试者覆盖；不得在看到数值后换 demo 或窗口。
- **确定性设置**：`torch.are_deterministic_algorithms_enabled()` 在冻结与执行时均为 `false`，授权记录显式固定该值，未在运行前静默开启。R0 记录实际设置（`tf32=false`、`autocast=false`）；F 要求生产梯度逐位重放，若真实 CUDA 无法复现则登记 mismatch 而不放宽。本次四窗上两模型 prediction/x_t/FK 点/分项/production 梯度均逐位复现，但该事实不外推为其他形状/算子/环境的确定性保证。

窗口键在包内标记为 `candidate_pending_R0_registration`；R0 必须在读取窗口数值前重新推导并封存 `(demo,start,end,T,cache_key,split,input_sha)`。任一键缺失或窗口不合法即保留 gap，不静默替换。

### 3.3 批次、时间步与噪声

| 项目 | 冻结值 |
| --- | --- |
| micro-batch | 四窗一次 `[4,60,38]`，不重排、不补齐、不丢窗 |
| 模型模式 | conditioner 与 model 均 `train()`，启用梯度 |
| 优化步 | **0**；`grad_accum=4` 只登记原配置，不执行累积/裁剪/AdamW |
| t | `[0, 250, 500, 999]`，一窗一个 t |
| 噪声 | CPU 专用 generator，seed `20261008`，一次 float32 `[4,60,38]`，sha `cb09992b…` |
| alpha | `scaled_linear`，beta `1e-4→0.02`，1000 步，sha `9f3be039…` |
| contact | 有效权重 0；监督分支必须执行 0 次，条件通道保留 |

## 4. 审计字段、行集与产物

### 4.1 数组 schema（完整键表见包内 `array_schema`）

| 组 | 键 | shape / dtype |
| --- | --- | --- |
| 原始输入 | `raw_root` / `raw_q` / `raw_wfc` | `[4,60,7]`/`[4,60,29]`/`[4,4,4]`，float64 |
| 条件 | `kps_obj` / `kps_clip` / `box_t_clip` / `box_quat` / `contact` | `[4,60,52,3]`、`[4,60,3]`、`[4,60,4]`、`[4,60,4]`，float32 |
| 编码 | `x0` / `x0_independent` / `joint_lo` / `joint_hi` / `root_mean` / `root_std` | `[4,60,38]` float32/float64，`[29]`、`[3]` float64 |
| 扩散 | `t` / `noise` / `alpha_cumprod` / `x_t_diffusion` | `[4]` int64、`[4,60,38]` float32、`[1000]` float64 |
| 每模型 | `prediction_*` / `term_elements_*_{pos,rot6d,joint,geodesic_rad2,vel,fk}` / `fk_points_pred_*` / `fk_points_target_*` | 预测 `[4,60,38]` float32；元素 float64：pos `[4,60,3]`、rot6d `[4,60,6]`、joint `[4,60,29]`、geodesic_rad2 `[4,60]`、vel `[4,59,38]`、fk `[4,60,15]`；FK 点 `[4,60,15,3]` float32 |

### 4.2 梯度 schema

`production_grad_{model}` 为 float32，`independent_grad_{model}` 为 float64，均为 `[6,4,60,38]`；切片顺序冻结为 `pos / rot6d / joint / geodesic_rad2 / vel / L5`。每模型一次 batched VJP 取六组，production 与 independent 各计一次调用。

### 4.3 必测行集

| 类别 | 数量 | 构成（每模型） |
| --- | ---: | --- |
| objective rows | **98** | 4 窗 ×（7 项 + L5）= 32；batch ×（7 项 + L5）= 8；`L5_arithmetic`/`total_arithmetic` = 2；7 项窗均值对账 = 7；合计 49/模型 |
| gradient rows | **14** | 6 切片 + `L5` 加权和 = 7/模型 |

行键是固定多重集，不是计数：额外行、缺字段、重复键、分母/权重/`aggregation_only`/`window`/`status`/`within_tolerance` 与行内算术不一致均拒绝。完整 98/14 键表见包内 `required_rows`。

### 4.4 产物与覆盖

| 阶段 | 必需产物 |
| --- | --- |
| R0 | `registration.json`、`window_keys.json`、`tolerance_contract.json`、`admission.json`、`r0_report.json` |
| P | `p_report.json`、`objective_real_rows.json`、`objective_real_arrays.npz`、`gradient_real_rows.json`、`gradient_real_arrays.npz`、`call_trace.json`、`rng_state_facts.json` |
| F | `audit_summary.json`、`independent_verification.json` |

覆盖固定为 `objective_coverage=partial`。审计项：clip 编码、根标准化、关节映射、五项解析损失、FK 前 `L5`、`fk_m2` 聚合、`total` 算术、五项+L5 输出空间梯度、contact 分支关闭。未审计项：真实 FK 几何与导数、网络参数梯度、累积/裁剪/AdamW/更新、选模、DDIM 采样。`full_objective_contract_passed=null`、`attribution=attribution_insufficient`。

## 5. Cohort、累计预算、尝试与调用上限

| 阶段 | 内容 | cohort 累计上限 | 收口预留 | attempt 上限 |
| --- | --- | ---: | ---: | ---: |
| R0 | 身份核验、四窗准入 | 180s | 30s | 2 |
| P | 同一进程生产前向 + 梯度取证 | 360s | 60s | 2 |
| F | 独立加载、重放、重推、收口 | 360s | 60s | 2 |
| 合计 | 成功/失败/崩溃/撤销/预检全部计费 | **900s** | 已含 | 分阶段累计 |

| 调用 | P（两模型） | F（两模型） | cohort 上限 |
| --- | ---: | ---: | ---: |
| conditioner 前向 | 2 | 2 | **4** |
| model 前向 | 2 | 2 | **4** |
| 生产 batched VJP | 2 | 2 | **4** |
| 独立 batched VJP | 2 | 2 | **4** |
| 合计 batched VJP | 4 | 4 | **8** |

R0 调用为 0。历史扫描根固定为 `derived/experiments/omniretarget_first/`，运行根必须是其直接子目录；同 cohort 全部 attempt 累计，阶段不互借，方案/实现/run 名变化不重置。已花额度、剩余额度、截止与调用余额写入事前登记；历史未知、额度不足或预留不足即停止，不启动实质工作。旧预测/融合历史只作引用，不转入余额。

2026-10-27 实际执行（每阶段仅 attempt_001）：R0 **3.1723s**/180、P **2.4054s**/360、F **1.9237s**/360，合计 **7.5015s**/900；调用 conditioner/model/production VJP/independent VJP 各 **4/4/4/4**；索引/台账各 3 行、全部 `done/complete=true`，无 running/publishing 残留。无第二次尝试。

## 6. 草案 §8 逐项状态

| # | §8 事项 | 当前状态 | 证据 |
| ---: | --- | --- | --- |
| 1 | 用户授权真实资产加载、四窗前向与输出空间求导；更新/FD/DDIM 仍 0 | **已完成** | 授权记录 `1071810f…`；三个计数器整数 0；运行未产生更新/FD/DDIM |
| 2 | 用户确认新 cohort、固定历史、旧拒收保留、阶段不互借、历史不重置 | **已完成** | `cohort.json` SHA `dc904ab8…`；历史根固定；旧融合/预测记录只作引用 |
| 3 | 用户确认 180/360/360s、每阶段 2 次、收口预留、4 次完整前向与 8 次求导 | **已完成** | 授权预算精确；实际 3.1723/2.4054/1.9237s，各 1 次尝试，调用各 4 |
| 4 | 真实工作前的登记/封口/消费者/父 unit 准入已实现并通过沙箱反例 | **已完成** | `b45cd7c` 后 68 项 audit+admission 测试通过；真实路径 R0/P/F 封口通过 |
| 5 | 生产取证适配与独立参考合成对拍；batched VJP、计数、图保真已验证 | **已完成** | observer 逐位透明、四类调用各 4、落盘梯度逐元素绑定；真实两模型重放逐位 |
| 6 | 数值域、梯度及标量逐字段容差完成纯合成校准并冻结 | **已完成** | 契约 SHA `2904fbc4…`、证据 `49371d30…`、独立核验通过；无阈值被抬高 |
| 7 | R0 登记下核验身份、四窗完整键、t/noise、配置、实现快照、环境与 schema；R0 封口后才启动 P | **已完成** | R0 `done/complete=true`；identity/admission 全过；四窗/包候选逐字段；checkpoint strict load 通过 |
| 8 | 独立复核按 raw 键集、输入、梯度、调用与全阶段历史验收；整体 partial、最终交互 not_ready | **已完成** | F `done/complete=true`；98/14 行无缺/多/重；两模型逐位重放；`problems=[]/gaps=[]` |

**八项均已完成并产生真实证据。** 覆盖范围仍为 `objective_coverage=partial`：真实 FK 几何/导数、参数梯度、累积/裁剪/AdamW/更新、选模和 DDIM 仍未审计；`full_objective_contract_passed=null`、`attribution=attribution_insufficient`、最终交互 `not_ready`。

## 7. 授权门与执行顺序

### 7.1 授权前拒绝证据与授权后结果

| 探测 | 命令 | 结果 |
| --- | --- | --- |
| 无授权 | `--real --out .../w1_real_objective_audit_unauthorized_probe --tolerance-contract ...` | **exit 1**：`--real requires --authorization (section 8 is not checked)`；探测目录未创建 |
| 模板（`authorized=false`） | 同上，附 `--authorization .../authorization_request_template.json` | **exit 1**：`not_authorized` 与 11 项 `authorization_checklist:*`；探测目录未创建 |

2026-10-27 用户授权后，`authorization/attempt_002/authorization.json`（SHA `1071810f…`）通过上述严格门；真实入口 exit 0，R0/P/F 全封口，结论 `real_batch_covered_contract_passed`。第一次入口在建立 cohort 前被无标记旧目录的未知历史拒绝，修复提交 `b45cd7c` 后重发授权；拒绝时无 cohort/attempt/计费/真实值/模型调用。

授权记录模板已保存为 P2 执行记录中的 `authorization_request_template.json`（SHA-256 `2e4aabbc…`；首轮模板 `d11ae079…` 保留为历史），其 `authorized=false`、`template_only=true`，不能作为授权使用。有效授权记录必须满足 `authorization_problems` 的严格契约：

- `schema` 必须精确等于 `w1_real_objective_authorization_v1`；`authorized` 必须为 `true`；`cohort_id` 与 `execution_id` 必须等于 `w1_real_objective_audit_v1`。
- `training_updates`、`parameter_fd`、`ddim_calls` 必须是 **JSON 整数 0**；`0.0`、`false`、`"0"`、`null` 都是不同声明，均被拒绝。
- 11 项 checklist 必须全为 `true`。
- `budgets` 的键集必须恰为 `R0/P/F`；每阶段 `budget_s` 必须是精确 JSON 数字并等于冻结值（`180`/`360` 与 `180.0`/`360.0` 同值；字符串、bool、null、NaN、近似值均拒绝），`attempts` 必须是 JSON 整数且精确等于 2。
- 不接受 `ddim_calls=None`；不接受额外或缺失阶段。

记录本身的 SHA-256 会写入 cohort 与注册。P2 新增的严格性测试覆盖 **28 类变异**（2 schema、14 计数、5 预算秒、5 次数、2 阶段集），并保留“整数秒 `180` 合法”的正向用例。

### 7.2 已执行的顺序（2026-10-27）

1. 用户授权并冻结授权记录；执行方核对字段与 SHA，未修改预算或矩阵。
2. `uv run --locked python scripts/audit_w1_real_objective.py --real --out derived/experiments/omniretarget_first/20261027_b45cd7c_w1_real_objective_audit --authorization derived/audit/w1_real_objective_audit/authorization/attempt_002/authorization.json --tolerance-contract derived/audit/w1_real_objective_audit/tolerance_contract.json --device cuda`。
3. **R0**：先写 running/complete=false 索引与台账，再核验全部固定身份与资产闭包，重新推导并封存四窗键、t/noise、配置、实现快照、环境与数组/梯度 schema；加载四窗并执行准入。R0 封口 `done/complete=true` 后才启动 P。
4. **P**：两模型各一次真实 conditioner→model 前向与生产/独立 batched VJP，保存 raw 数组、行、调用与 RNG 事实；无 mismatch，未触发停止。
5. **F**：独立 strict 加载、恢复 RNG、按注册重放、从 raw 重推行集与梯度、核对调用与全阶段历史、封口；两模型逐位重放成功。
6. 出口：`real_batch_covered_contract_passed`（exit 0）。`objective_coverage=partial`、`attribution=attribution_insufficient`、最终交互 `not_ready`、生产接入 false。

## 8. 边界与下一交付

- 2026-10-27 已按用户授权完成真实 R0→P→F 一次：R0/P/F 全封口 `done/complete=true`，结论 `real_batch_covered_contract_passed`，98/98 objective、14/14 gradient、`problems=[]`、`gaps=[]`。
- 覆盖范围仍是 partial：真实 FK 几何/导数、参数梯度、累积/裁剪/AdamW/更新、选模、DDIM 采样与端到端质量**未审计**；`full_objective_contract_passed=null`、`attribution=attribution_insufficient`、最终交互 `not_ready`、生产接入 false。
- 同一受试者（两个 demo 均属 `sub10`）与 `deterministic_algorithms=false` 的边界保留；执行时未更换窗口、t/noise、权重或容差。本次逐位重放成功，但不构成其他形状/算子/环境的确定性保证。
- 覆盖范围内未观察到目标偏离，因此**不触发**单因素诊断；若下一问题是端到端失败归因，应先另立方案扩展真实 FK/参数梯度/更新/DDIM 覆盖，再决定干预。任何干预仍须新预注册，不自动续训、改教师、扩数据或接入生产。
- 真实报告：[W1 真实批次目标与输出空间梯度审计报告](../verification/omniretarget_w1_real_objective_audit_report.md)；执行记录与只读 post-check 见 `derived/audit/w1_real_objective_audit/authorization/attempt_002/`。
- `factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false` 不变。
