# W1 真实批次目标与输出空间梯度审计报告

日期：2026-10-27。状态：**真实 R0→P→F 已按授权执行一次并全部封口；覆盖范围内通过，目标未观察到偏离；整体覆盖仍为 partial，最终交互仍 not_ready。** 上级草案：[W1 真实训练目标审计预注册草案](../plans/omniretarget_w1_real_objective_audit_prereg_draft.md) §8。准入材料：[准入包与冻结校准契约](../plans/omniretarget_w1_real_objective_audit_admission_package.md)。实现与合成验证：[实现与合成自测验证记录](omniretarget_w1_real_objective_audit_verification.md)。

> 本报告只证明四个注册窗口上的生产目标/输出空间梯度与独立参考一致，且 F 能从落盘 raw 数组逐位重放。它不证明真实 G1 FK 几何与导数、网络参数梯度、累积/裁剪/AdamW/更新、选模、DDIM 采样或端到端交互质量。`full_objective_contract_passed=null`、`objective_coverage=partial`、`attribution=attribution_insufficient`、`factor_selected=false`、`auto_execute=false`、`production_integration=false`、`final_interaction_method=not_ready` 不变。

## 0. 结论

| 项目 | 实际结果 |
| --- | --- |
| 结论 | `real_batch_covered_contract_passed`（exit 0） |
| 证据完整 | `complete_evidence=true`、`covered_contract_passed=true` |
| 失败/缺口 | `problems=[]`、`failures=[]`、`gaps=[]` |
| R0 | identity/admission 全过；四窗键、t/noise/alpha、配置、实现、环境、schema、checkpoint strict load 全部绑定 |
| P | 两模型各一次真实 conditioner→model 前向；各一次 production/independent batched VJP |
| 行集 | objective **98/98**、gradient **14/14**，全部 `within_tolerance`；多重集无缺/多/重 |
| 最大目标误差 | 生产 vs 独立：**1.0332e-9**（`batch|deterministic|L5`）；含算术/对账行 **1.5134e-9**（`batch|diffusion|total_arithmetic`） |
| 最大目标相对误差 | **1.2081e-7**（`batch|deterministic|window_mean_reconciliation|geodesic_rad2`） |
| 最大梯度误差 | **6.4061e-10**（`grad|diffusion|geodesic_rad2`）；最大相对 **1.6713e-7**（`grad|deterministic|vel`） |
| 容差占用 | 目标最大 `abs_error/limit` **7.29e-5**；梯度最大 **2.80e-5**；均远低于 1 |
| F 独立重放 | 两模型 prediction / FK 点 / 分项 / production 梯度**逐位**相等；diffusion 的 `x_t` 亦逐位相等（deterministic 无 `x_t`）；independent 梯度绑定落盘工件 |
| 调用 | conditioner **4/4**、model **4/4**、production VJP **4/4**、independent VJP **4/4** |
| 接触分支 | 执行 **0** 次（`w_contact=0`） |
| 训练更新 / 参数 FD / DDIM / checkpoint 写入 | **0 / 0 / 0 / 0** |
| 阶段成本 | R0 **3.172s**/180、P **2.405s**/360、F **1.924s**/360；合计 **7.501s**/900 |
| 尝试数 | 每阶段 **1/2**；无第二次尝试 |
| 单因素诊断 | **未触发**：覆盖范围内未观察到目标偏离；不做自动续训、改教师、扩数据或切换生产 |

## 1. 授权、身份与冻结

### 1.1 授权记录

用户于 2026-10-27 明确授权本轮真实执行，范围固定为四个注册窗口、两模型前向、输出空间 batched VJP、独立重放；更新、参数 FD、DDIM、修复和生产接入均不在授权内。授权记录 11 项 §8 checklist 全为 true，预算精确为 R0/P/F `180/360/360` 秒、每阶段 2 次，三个计数器为 JSON 整数 0。

| 项目 | 值 |
| --- | --- |
| 授权记录 | `derived/audit/w1_real_objective_audit/authorization/attempt_002/authorization.json` |
| 授权 SHA-256 | `1071810f96325acd4a20d77395b7461864a8102f0268b10153c7e52908dcdfcb` |
| 记录中的准入包 | `admission/attempt_005/admission_package.json`，SHA `1aadb296f1a94cd4bd39c654661712a6c46670fca0211ecf4a9b4452d86e45b8` |
| 包索引 SHA | `bdedc33c5b13668e840179890978860d3815ef89e700674f19356108e3513192` |
| 冻结容差契约 | SHA `2904fbc41113ba379cee2e4de4704befcac3ae8bb69c3178bcf141e48fd38eb3` |
| 代码身份 | HEAD `b45cd7c654c2a2bb23d86fd76050f9ad651a7e67`，clean，tracked diff 为空 |
| 运行根 | `derived/experiments/omniretarget_first/20261027_b45cd7c_w1_real_objective_audit` |

### 1.2 就绪前的一次修复（未消耗真实证据）

第一次真实入口在建立 cohort 之前被拒绝：`call history unknown: refusing to grant a fresh call quota`。原因是 cohort 扫描器在判断一个历史目录是否属于本 cohort 之前就严格解析了它的 `costs.json`；`20261015_9000ac8_w1_attribution` 没有 `cohort.json` 且使用裸 list schema，使整个历史被判为未知。

修复提交 `b45cd7c`：无 `cohort.json` 且无法解析的**无标记**历史目录被跳过（本 cohort 的注册 run 一定在任何 attempt 前写 `cohort.json`）；带本 cohort 标记但台账损坏的目录仍然 fail-closed。该修复只影响 cohort 历史扫描，不触及目标、梯度、容差、窗口、调用矩阵或预算语义。修复后 audit+admission 测试 **68 passed（28.77s）**，准入包重新生成为 `attempt_005`，授权记录随后重发为 `attempt_002`。拒绝时没有写出 `cohort.json`、没有注册 attempt、没有计费、没有打开真实窗口/checkpoint/模型；授权范围与预算未改变。

### 1.3 固定矩阵

| 项目 | 值 |
| --- | --- |
| 模型 | deterministic、diffusion（各 step 4000 的 best checkpoint） |
| 批次 | `[4,60,38]`，四窗一次，不重排、不补齐、不丢窗 |
| t | `[0,250,500,999]` |
| 噪声 | CPU generator，seed `20261008`，一次 float32 `[4,60,38]`，SHA `cb09992b…` |
| alpha | `scaled_linear`，`beta 1e-4→0.02`，1000 步，SHA `9f3be039…` |
| 权重 | `w_pos=1.0`、`w_rot6d=1.0`、`w_joint=1.0`、`w_geodesic=0.5`、`w_fk=2.0`、`w_contact=0.0`、`w_vel=0.5` |
| 接触监督 | 关闭；P 的 `contact_branch_executions=0` |
| 确定性 | `torch.are_deterministic_algorithms_enabled()=false`（冻结时即 false，未在运行前静默开启）；`tf32=false`、`autocast=false`、`matmul_precision=highest` |

四个窗口与准入包逐字段一致：

| # | demo | start | end | T | cache key（前 16） | input SHA（前 16） | derived SHA（前 16） |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | `sub10_largebox_063` | 0 | 60 | 133 | `29741e7b2df2b4af` | `81d9ad0d9d44e0c7` | `ee7722b9c05ce6f5` |
| 2 | `sub10_largebox_063` | 30 | 90 | 133 | `29741e7b2df2b4af` | `81d9ad0d9d44e0c7` | `ee7722b9c05ce6f5` |
| 3 | `sub10_largebox_085` | 0 | 60 | 148 | `60774514ffeafa4e` | `90d1102fd564864a` | `6f1cb597711624bc` |
| 4 | `sub10_largebox_085` | 30 | 90 | 148 | `60774514ffeafa4e` | `90d1102fd564864a` | `6f1cb597711624bc` |

两条审阅限制仍然有效：两个 demo 同属 `sub10`（冻结字典序），本审计**不声明跨受试者覆盖**；窗口、t/noise、权重、容差在看到真实数值后均未更换。

## 2. R0：身份、资产与四窗准入

R0 先写 running/complete=false 索引与台账，再写注册与准入；只有 R0 封口 `done/complete=true` 后 P 才启动。

| 核验项 | 结果 |
| --- | --- |
| 父锚 | W1 注册 `5561f685…`、W1 final `7f1a9a59…`、预测诊断注册 `5fcf8b3a…`、预测终局 `ce9843fd…`、SOURCE manifest `a350a8b8…`；全部相等 |
| 预注册/输入/资产 | 预注册 `071c49d3…`、train24 规范序、55 段输入与 A1 资产闭包按冻结哈希重验；`identity.passed=true`、`problems=[]` |
| 配置 | `configs/train/omniretarget_motion.yaml` `0e2cfa96…` |
| checkpoint 文件 | det `7b1efd57…`、diff `1f9c16b1…`；文件哈希相等 |
| checkpoint 内容 | 两模型均 step 4000、`checkpoint_usable=true`、`parameters_finite=true`、`diagnostic_only=false`；model/conditioner `load_state_dict(strict=True)`；加载后全参数有限；root_stats 与缓存逐位相等；config/assets/manifest identity 通过 |
| 状态摘要 | det model `7c1f53b0…`/conditioner `a8550e4c…`；diff model `937ee203…`/conditioner `94648823…` |
| 环境 | Python 3.12.13、torch 2.14.0+cu126、CUDA runtime 12.6、RTX 3090、mujoco 3.13.0、diffusers 0.40.0、`uv.lock` `f809284e…` |
| 四窗 | 4/4；demo/start/end/T/cache_key/split/input_sha/derived_sha 与授权及准入包逐字段相等 |
| 编码 | 生产 x0 对独立重编码 `within_tolerance`；max abs **5.7067656e-8**（容差 2e-5），bad elements 0 |
| 数值域 | root 四元数范数最小 `0.9999999999999998`；box `0.9999999404`；原始关节限位最大违反 `-0.02618`（负值=未越限） |
| t/noise/alpha | `[0,250,500,999]`；noise/alpha SHA 与冻结值相等且可由种子/调度独立重建 |
| schema | 17 个原始/条件/编码/扩散键与每模型预测/分项/FK 点键全部 shape/dtype/有限性通过；梯度键 float32（production）/float64（independent）通过 |

R0 产物：`registration.json` SHA `ecc11368…`、`window_keys.json` `34d76073…`、`admission.json` `61c5b16d…`、`r0_report.json` `f66c1781…`。R0 阶段成本 **3.1723s**，封口 `done/complete=true`。

> 旧预测/归因终局在 R0 只作引用（`budget_transfer=false`），其 `conclusion` 字段未被本审计消费；R0 记录的 `attribution_conclusion_reference=unknown` 不影响身份或数值判定。

> 运行日志中的 `Unknown attribute "name" in .../collision` 是 URDF 解析器对 collision `name` 属性的非致命警告；FK 后端仍记录为 `production`，未改变审计数值或结论。

## 3. P：真实前向与输出空间梯度

### 3.1 调用与图

- deterministic：conditioner 1 次、model 1 次、production VJP 1 次、independent VJP 1 次；`x_t=None`。
- diffusion：conditioner 1 次、`add_noise` → `x_t`、model 1 次、production VJP 1 次、independent VJP 1 次。
- 两模型 `prediction_problems=[]`、`fk_backend=production`、`contact_branch_executed=false`。
- 独立参考的 `geodesic_domain_ok=true`、`bad_count=0`，无近 π 域缺口。
- 无参数反向、无 FD、无优化器状态；`call_trace.parameter_backward=0`、`parameter_fd=0`、`production_forwards=2`。

### 3.2 目标标量（batch）

| 项 | deterministic 生产 | deterministic 独立 | diffusion 生产 | diffusion 独立 |
| --- | ---: | ---: | ---: | ---: |
| pos | 4.275276e-3 | 4.275276e-3 | 8.505109e-3 | 8.505109e-3 |
| rot6d | 3.248936e-3 | 3.248936e-3 | 4.809657e-3 | 4.809657e-3 |
| joint | 9.864869e-3 | 9.864869e-3 | 9.629244e-3 | 9.629244e-3 |
| geodesic_rad2 | 5.265119e-3 | 5.265118e-3 | 1.187178e-2 | 1.187178e-2 |
| vel | 2.670784e-4 | 2.670784e-4 | 5.246352e-4 | 5.246352e-4 |
| fk_m2 | 3.998824e-3 | 3.998824e-3 | 4.196598e-3 | 4.196598e-3 |
| L5 | 2.015518e-2 | 2.015518e-2 | 2.914222e-2 | 2.914222e-2 |
| total | 2.815283e-2 | 2.815283e-2 | 3.753541e-2 | 3.753541e-2 |

这些数值是标准化残差平方的窗口均值，只用于实现一致性；**不构成动作质量、可跟踪性或最终交互可行性的评价**。

### 3.3 98 行目标证据

| 范围 | 行数 | 全部 within | 最大 abs | 最大相对 | 最大 abs/limit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 窗口 + batch 生产 vs 独立（80 行） | 80 | 80 | 1.0332e-9 | 1.0213e-7 | 2.696e-5 |
| 算术/对账（18 行） | 18 | 18 | 1.5134e-9 | 1.2081e-7 | 7.293e-5 |
| 合计 | **98** | **98** | **1.5134e-9** | **1.2081e-7** | **7.293e-5** |

每模型×每项的窗口+batch最大误差（limit 为该行 `atol + rtol·|independent|`）：

| 模型 | 项 | 行数 | 最大 abs | 最大相对 | limit 范围 |
| --- | --- | ---: | ---: | ---: | --- |
| deterministic | pos | 5 | 1.741e-10 | 4.35e-8 | 2.006e-5–2.010e-5 |
| deterministic | rot6d | 5 | 1.359e-10 | 4.68e-8 | 2.005e-5–2.008e-5 |
| deterministic | joint | 5 | 7.564e-11 | 8.09e-9 | 2.019e-5–2.020e-5 |
| deterministic | geodesic_rad2 | 5 | 5.133e-10 | 9.75e-8 | 2.0004e-3–2.0007e-3 |
| deterministic | vel | 5 | 2.090e-11 | 7.83e-8 | 2.0004e-5–2.0006e-5 |
| deterministic | fk_m2 | 5 | 1.172e-10 | 2.93e-8 | 2.005e-5–2.010e-5 |
| deterministic | total | 5 | 3.364e-10 | 1.19e-8 | 1.0027e-3–1.0029e-3 |
| deterministic | L5 | 5 | 1.033e-9 | 5.13e-8 | 1.0019e-3–1.0022e-3 |
| diffusion | pos | 5 | 3.003e-10 | 5.26e-8 | 2.011e-5–2.021e-5 |
| diffusion | rot6d | 5 | 1.984e-10 | 6.08e-8 | 2.006e-5–2.015e-5 |
| diffusion | joint | 5 | 5.444e-10 | 5.65e-8 | 2.017e-5–2.022e-5 |
| diffusion | geodesic_rad2 | 5 | 7.948e-10 | 6.22e-8 | 2.0007e-3–2.0020e-3 |
| diffusion | vel | 5 | 5.358e-11 | 1.02e-7 | 2.0006e-5–2.0015e-5 |
| diffusion | fk_m2 | 5 | 3.668e-11 | 8.74e-9 | 2.0065e-5–2.0121e-5 |
| diffusion | total | 5 | 9.918e-10 | 2.64e-8 | 1.0030e-3–1.0046e-3 |
| diffusion | L5 | 5 | 6.101e-10 | 2.79e-8 | 1.0022e-3–1.0039e-3 |

阈值来源保持冻结：基础 `loss_float32` 2e-5/2e-5；`geodesic_rad2` 2e-3/1e-4；`total`/`L5` 1e-3/1e-4；算术/对账行使用基础项。**没有任何阈值被抬高**，也没有任何行被删除或以空集合维持通过。

## 4. 输出空间梯度（14 行）

`production_grad_*` 落盘为 float32 `[6,4,60,38]`，`independent_grad_*` 为 float64；切片顺序 `pos/rot6d/joint/geodesic_rad2/vel/L5`。每模型一次 batched VJP 取六组。

| 模型 | 切片 | 生产范数 | 独立范数 | 最大 abs | limit | abs/limit | 最大相对 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic | pos | 4.873555e-3 | 4.873555e-3 | 1.592e-10 | 2.097e-5 | 7.59e-6 | 3.27e-8 |
| deterministic | rot6d | 3.004134e-3 | 3.004134e-3 | 4.554e-11 | 2.060e-5 | 2.21e-6 | 1.52e-8 |
| deterministic | joint | 2.381064e-3 | 2.381064e-3 | 1.681e-11 | 2.048e-5 | 8.21e-7 | 7.06e-9 |
| deterministic | geodesic_rad2 | 9.516384e-3 | 9.516384e-3 | 5.407e-10 | 2.190e-5 | 2.47e-5 | 5.68e-8 |
| deterministic | vel | 2.375483e-4 | 2.375483e-4 | 3.970e-11 | 2.005e-5 | 1.98e-6 | 1.67e-7 |
| deterministic | L5 | 8.740624e-3 | 8.740624e-3 | 2.876e-10 | 2.175e-5 | 1.32e-5 | 3.29e-8 |
| deterministic | L5 加权和对账 | 8.740624e-3 | 8.740624e-3 | 2.876e-10 | 2.175e-5 | 1.32e-5 | 3.29e-8 |
| diffusion | pos | 6.873908e-3 | 6.873908e-3 | 1.747e-10 | 2.137e-5 | 8.17e-6 | 2.54e-8 |
| diffusion | rot6d | 3.655155e-3 | 3.655155e-3 | 5.010e-11 | 2.073e-5 | 2.42e-6 | 1.37e-8 |
| diffusion | joint | 2.352456e-3 | 2.352456e-3 | 1.452e-11 | 2.047e-5 | 7.09e-7 | 6.17e-9 |
| diffusion | geodesic_rad2 | 1.428175e-2 | 1.428175e-2 | 6.406e-10 | 2.286e-5 | 2.80e-5 | 4.49e-8 |
| diffusion | vel | 6.309059e-4 | 6.309059e-4 | 3.999e-11 | 2.013e-5 | 1.99e-6 | 6.34e-8 |
| diffusion | L5 | 1.231266e-2 | 1.231265e-2 | 3.641e-10 | 2.246e-5 | 1.62e-5 | 2.96e-8 |
| diffusion | L5 加权和对账 | 1.231266e-2 | 1.231265e-2 | 3.641e-10 | 2.246e-5 | 1.62e-5 | 2.96e-8 |

最大梯度误差 **6.4061e-10**、最大相对 **1.6713e-7**、最大容差占用 **2.80e-5**。`L5` 与五切片加权和一致到 **3.641e-10**（独立侧）和 **2.876e-10**（deterministic 侧）；`production` 与 `independent` 的 L5 范数差约 **1e-9**。

> 标注说明：`gradient_real_rows.json` 的 `production_dtype` 字段记录的是比较时统一提升后的 `float64`，不是落盘数组的 dtype。实际 `production_grad_*` 为 float32、`independent_grad_*` 为 float64，F 的 `_gradient_schema` 已按此逐键校验；本报告以上表与 NPZ dtype 为准。

## 5. F：独立加载、重放、重推与封口

- F 独立 strict 加载两份 checkpoint（fresh 模型/conditioner），恢复 P 保存的 Python/NumPy/Torch CPU/CUDA RNG 状态，再按注册输入重放。
- 两模型的 `prediction_bitwise=true`、`production_grad_bitwise=true`、`independent_grad_bound_to_artifact=true`、`rng_restored_before_replay=true`；`x_t_bitwise=true` 对 diffusion 为实际重放结果（deterministic 没有 `x_t`，该字段按协议记 true，不作数值声明）。
- 保存的 prediction、`x_t_diffusion`、FK 预测/目标点、五个分项元素、production 梯度与 independent 梯度均与落盘 raw 工件逐位/容差绑定；未使用 P 的内存对象。
- F 从 raw 数组重建 98 行目标与 14 行梯度；行多重集恰等于注册键集，`row_multiset.objective=[]`、`row_multiset.gradient=[]`、`row_errors={objective:null, gradient:null}`。
- `consumer_states={"R0":[],"P":[]}`；R0/P 的索引/台账/磁盘三方一致，无 running/publishing 残留。
- F 的 `independent_verification.json`：`problems=[]`、`gaps=[]`，结论 `real_batch_covered_contract_passed`；`audit_summary.json` 的 counts 为 windows 4、models 2、objective 98/98、gradient 14/14。
- 未执行训练更新、参数 FD、DDIM、checkpoint 写入、修复或生产入口切换；边界表全部保持。

**限制**：F 的“独立参考”与 F 的重推共享同一套参考公式实现；F 证明了生产路径与该参考一致、且工件可逐位重放，但不是第三套独立公式。真实 G1 FK 只审计到 FK 点的 `fk_m2` 聚合，不审计 FK 几何正确性或导数。

## 6. 账本、成本与完整性

| 阶段 | attempt | 状态 | complete | 实测墙钟 | 阶段预算 | 调用余额（conditioner/model/prod VJP/indep VJP） |
| --- | --- | --- | --- | ---: | ---: | --- |
| R0 | attempt_001 | done | true | 3.1723s | 180s | 0/0/0/0 |
| P | attempt_001 | done | true | 2.4054s | 360s | 2/2/2/2 |
| F | attempt_001 | done | true | 1.9237s | 360s | 4/4/4/4 |
| 合计 | — | — | — | **7.5015s** | **900s** | 上限 4/4/4/4 |

- 索引与台账各 3 行，逐阶段一行；`measured_wall_s` 两处相等，`wall_s=round(measured,1)`；无额外/缺失/重复 attempt。
- 调用上限与全链余额在 R0 前已证明；实际未超。接触分支 0；`parameter_backward=0`、`parameter_fd=0`、`ddim_calls=0`、`checkpoint_writes=0`。
- 索引引用的报告/摘要/工件路径（去重 14 个）全部与磁盘相等；P 的 arrays/rows/call_trace/rng_facts 六个哈希、F 的 verification 哈希、R0 的 registration/admission 哈希均通过。
- 只读 post-check（`authorization/attempt_002/postcheck.py`，无模型调用）重算：授权/包/契约绑定、四窗与包候选逐字段、98/14 行多重集、逐行 limit/within/abs_error、L5/total 与窗口均值、梯度范数/最大差/limit、调用余额、阶段成本、checkpoint/实现哈希、索引工件哈希；结果 `passed=true`、`problems=[]`，记录 `postcheck.json`。

## 7. 判断与后续

### 7.1 覆盖范围内的判断

**通过，未观察到目标偏离。** 在四个注册窗口、两个固定 step-4000 checkpoint、`[4,60,38]` 单批、`t=[0,250,500,999]` 和冻结容差下：

1. 生产 loss 的五个解析分项、FK 前 `L5`、FK 聚合 `fk_m2` 与 `total` 算术和独立 float64 参考一致；
2. 五项 + `L5` 的输出空间梯度与独立 batched VJP 一致；
3. 独立参考和 F 重放能在落盘 raw 数组上逐位重建；
4. contact 分支确实未执行，调用/预算/封口/消费者链路完整。

因此“目标实现偏离”在本覆盖范围内**不是**当前证据支持的失败原因。这个结论不能外推为训练目标已全面验证：真实 FK 几何/导数、参数梯度、累积/裁剪/AdamW/更新、选模、DDIM 与端到端动作质量均未审计。

### 7.2 后续单因素诊断

本轮**不触发**单因素诊断：没有覆盖范围内的偏离可作为干预目标。若下一问题是如何解释端到端交互失败，正确顺序是先扩展审计覆盖，而不是直接改目标或调参：

1. 先做真实 G1 FK 几何/导数审计（独立 FK 参考与有限差分/解析对拍），因为当前只审到 FK 聚合；
2. 再做网络参数梯度、累积/裁剪/AdamW 单步更新与选模审计；
3. 最后做 DDIM 采样/时间步路径审计，并把各层证据与端到端失败分类对齐；
4. 任何干预仍需另立单因素预注册方案，保留本 cohort 全部成本与失败记录，不自动续训、改教师、扩数据或接入生产。

### 7.3 边界

`objective_coverage=partial`、`full_objective_contract_passed=null`、`attribution=attribution_insufficient`、`r3=not_run`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false`、`training_updates=0`、`parameter_fd=0`、`ddim_calls=0`、`checkpoint_writes=0`。四窗同属 `sub10`，不声明跨受试者覆盖。`deterministic_algorithms=false` 下本次逐位重放成功，不构成其他形状/算子/环境下的确定性保证。

## 8. 复现与证据

```bash
# 真实执行（已冻结的授权与契约）
uv run --locked python scripts/audit_w1_real_objective.py --real \
  --out derived/experiments/omniretarget_first/20261027_b45cd7c_w1_real_objective_audit \
  --authorization derived/audit/w1_real_objective_audit/authorization/attempt_002/authorization.json \
  --tolerance-contract derived/audit/w1_real_objective_audit/tolerance_contract.json \
  --device cuda

# 只读 post-check（无模型调用）
uv run --locked python \
  derived/audit/w1_real_objective_audit/authorization/attempt_002/postcheck.py
```

| 产物 | SHA-256 |
| --- | --- |
| `registration/attempt_001/registration.json` | `ecc11368219a33c95a6038769398fd11789dcdd04430760f54ae3a84ba58a66a` |
| `registration/attempt_001/admission.json` | `61c5b16d883d0fa084632a548bd7428e2fe580540990feed47926546bd32ce95` |
| `production/attempt_001/p_report.json` | `1ea15ddc2da7a0acb5125f1ecc92e1ea7e0cc74589e7a7ad5055b26055b7cdcb` |
| `production/attempt_001/objective_real_rows.json` | `d0d87de7a7082000e64330df90dd3b4664514192ebc820b3497f319da6eebf84` |
| `production/attempt_001/objective_real_arrays.npz` | `4cf20acb13fcdee207e0492a3b07818cc9ed31202409177fa1ed3f13086fd2c8` |
| `production/attempt_001/gradient_real_rows.json` | `88bbbc59e3b1b7351d8356cf8249823afe4e7e2ba45478cf5b2a588369296d92` |
| `production/attempt_001/gradient_real_arrays.npz` | `49b21c6c5d82a19e92f2c0d870b1d7f036986604c009c6195e6f86a035765ea7` |
| `production/attempt_001/call_trace.json` | `e0cc97b1fba34f53dd3f181bb6ce664ef66a7e271d9ab588b475216a102dbbd4` |
| `production/attempt_001/rng_state_facts.json` | `9735b01f0390d75124161d75e2ee270a74e7d7d70b571bf02c624dacbcaf7a16` |
| `finalize/attempt_001/audit_summary.json` | `0a839071125a156273b3d579a564965cd0edb196b8f900b05417edd0cb00db97` |
| `finalize/attempt_001/independent_verification.json` | `f0bf351bd0bd184f52198ac481293bae9127607ec98864010768430a4dcbd961` |
| `cohort.json` | `dc904ab8e71bd45fe4d2d1248b2dff3ab4af4ebe09818af46666684657abd7d2` |
| `attempts_index.json` | `f0e6c1b37c51fd9fe2193c5d58e1f7ef39488ec194f184a5c5f27cf9e738ea17` |
| `costs.json` | `f09799d3239da48865349c4482595352cd5b09a80bcecccd6646adb42333a93b` |

测试：`uv run --locked pytest tests/test_w1_real_objective_audit.py tests/test_w1_real_admission.py -q` -> **68 passed（28.77s）**。全项目套件本轮未重跑，不能用旧的全量结果冒充本轮验证。
