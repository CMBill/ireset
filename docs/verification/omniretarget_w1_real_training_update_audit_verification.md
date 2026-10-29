# W1 真实网络参数梯度与优化更新链路审计 —— I/C 工程与纯合成验收记录

日期：2026-10-29。方案：[omniretarget_w1_real_training_update_audit_prereg_draft.md](../plans/omniretarget_w1_real_training_update_audit_prereg_draft.md)
（v0.6）。状态：**I/C 实现、第四轮审阅修复与纯合成验收完成；真实 provider/授权/注册/真实 R0→P→I→F 未实施**。

本记录只覆盖合成证据与冻结准入材料。没有读取真实 checkpoint 张量、真实 NPZ 数值、数据集或 G1 资产，
没有执行真实网络前向/反向、真实 FK、真实参数更新或生产 checkpoint 写入。

## 1. 交付文件

| 文件 | 作用 |
| --- | --- |
| `src/ireset/eval/w1_real_training_update_audit.py` | 审计器：roster/行目录、合成 fixture、P/I 捕获、独立参考、有效 float32 AdamW、跨路径传播、R0/P/I/F、预算/封口、校准、准入、真实门 |
| `scripts/audit_w1_real_training_update.py` | CLI：`--synthetic`、`--calibrate-synthetic`、`--admission`、`--real`（门后拒绝） |
| `tests/test_w1_real_training_update_audit.py` | 45 项纯合成反例/接口测试（v0.5 为 39 项，v0.6 新增 6 项第四轮反例） |
| `src/ireset/train/trainer.py` | 抽出共享 `predict` / `begin_update` / `accumulate_micro` / `clip_gradients` / `step_optimizer`，新增只读 observer、审计注入 `loss_fn`/`build_optimizer` 与 `clip_coefficient`；默认路径不变 |

共享单元由原生 `Trainer.run` 与审计共同调用。`observer=None` 与 `observer={}` 的
theta/m/v/step 逐位相等，loss/parts 与全局 RNG 不变；NaN 输入的异常类型/原因一致
（`tests` 覆盖 deterministic/diffusion）。

## 2. 冻结校准与有效 float32 契约

目录：`derived/audit/w1_real_training_update_audit/`。v0.4/v0.5 的旧校准保存在
`history/v0.4_5d9230cf/` 与 `history/v0.5_9a964372/`（后者由 attempt_004 内嵌 contract 原字节重建并核对 SHA），
以下为 v0.6 当前文件。

| 工件 | SHA-256 |
| --- | --- |
| `tolerance_contract.json` | `582ea60a73b2852cc676e9eeee4e5003394f673567171e5efb37f9f86191177d` |
| 校准 evidence（contract 内） | `56ae9663f3d1c103071986a16a4f95c75136e5919a5eb72ccff71efa8b46c0ee` |
| `adamw_fp32_contract.json`（规范化内容哈希） | `6d7a24d0cd077454120dfbaaad37fcc378b10572e4a33a26df8ca09f7e0003e1` |

`tolerance_contract_problems(recompute=True)` 在新进程重跑校准并逐项通过；`4×observed` 全部 ≤ 1。

观测比值（`max_ratio_of_proposal`）：

| 键 | 观测 | 4× |
| --- | ---: | ---: |
| `alpha_x_t` | 0.2168 | 0.8672 |
| `loss_scalar` | 0.0082 | 0.0326 |
| `output_cotangent` | 0.0171 | 0.0683 |
| `parameter_gradient` | 0.0904 | 0.3617 |
| `global_norm_clip` | 0.0007 | 0.0029 |
| `theta_cross_path` | 0.0116 | 0.0464 |
| `state_cross_path` | **0.1569** | **0.6277** |
| `delta_cross_path` | 0.0745 | 0.2980 |
| `delta_same_input` | 0.1471 | 0.5883 |

后四项取全链通过行与标量近例的最大值（`source=synthetic_chain+scalar_near_1e-7`）。

**状态校准的分母修复（v0.6）**：旧 `max_state_ratio` 用 `ZERO_STATE_ABS=1e-6` 掩码排除了
`1e-17` 量级的 m/v，且分母用 `max(|m_p|,|m_i|)`；这与原始门
`abs(a-b)≤atol+rtol*abs(reference)` 不一致。现 `_state_cross_path_ratio` 覆盖每个
`abs(reference)>0` 的坐标、分母为 `atol+rtol*abs(reference)`、`reference==0` 且 `a≠b`
记为无界。同 seed 校准链在旧 `rtol=5e-4` 下的真实最大值为 **0.313871**（4×=**1.255>1**，
旧记录 0.009235 只统计了大状态），因此 v0.6 把 `state_cross_path` 提案调到 **`1e-3`**；
同一全坐标规则重算 **0.156936**、4×=**0.628**。该 2× 放宽只用于原始状态相对比较，
零/符号直接失败、同输入逐位公式门、更新量门和 `m=0 vs 1e-9`/50% 状态差异拒绝均不变。

有效求值契约：

- 当前环境 torch 2.14.0+cu126 / RTX 3090，CUDA/float32 **默认 foreach** dispatch 与
  §4.3.1 节点图逐位一致：4 组固定/对抗/次正规/首步用例、两步递推、源码 SHA 与新进程重算全部通过。
- CUDA 单 tensor/`foreach=False` 的标量除法是倒数乘而非正确舍入除法；CPU 单 tensor 的
  addcmul/最终加也不同。契约只绑定默认 foreach；CPU-only/强制其他 dispatch 不得准入。
- 独立 `x_t` 使用生产 **binary32** `alpha_cumprod` 提升到 float64 的有效常数；原始 float64
  schedule 只作独立对账。
- 零噪声门：`ZERO_GRADIENT_ABS=1e-6` 只适用于两侧范数均 ≤ `ZERO_GRADIENT_NOISE_SCALE=1e-10`
  的深相消带；深带内参考非零而生产全零仍拒绝（`nonzero_erased`）。带外 tensor 必须通过
  relative-L2≤2e-4/cosine≥1-1e-6；100×1e-9 全清零在带外由 group 门拒绝，深带全擦除由
  `zero_erased` 拒绝，~1e-12 的真实相消噪声仍通过。
- 输出 cotangent 除整张量门（atol 2e-8/rtol 2e-4）外，另按 `pos`(0:3)、`rot6d`(3:9)、
  `joint`(9:38) 三组执行独立 relative-L2≤2e-4/cosine≥1-1e-6 且生产非全零；组参考小于
  `1e-10` 时用 1e-6 绝对零噪声门并拒绝整组擦除。组事实与 problems 进入 I 行 canonical hash，
  F 重放重新计算并拒绝任何只改分组事实的行漂移。
- 更新量可辨裕量：`B_delta≤0.25*5e-4*abs(delta_ref)` 才标可辨；0.25 使写入舍入误差不被
  同门归一化，`4×observed≤proposal` 可满足。跨路径 m/v/theta 保留原始相对门，只有精确双零
  才等价；零/符号/单侧零更新显式记录并按 §6.4 分类。
- clip 边界：原生公式无条件 `coef=min(1,clip/(norm+1e-6))`，`coef<1` 即触发；`norm==clip`
  缩放为 `0.999999046`，不再用 `norm>clip` 漏掉边界。Trainer observer 与独立 clip 同式。

语义反例（校准 evidence 内，全部按预期分类）：

| 用例 | 预期 | 结果 |
| --- | --- | --- |
| 1 ULP 梯度差、m/可辨更新约 62.9% 差异 | `cross_path_sensitivity_gap` | 通过（不是 AdamW failure/通过） |
| 稳定同梯度 | `passed` | 通过 |
| 后态改 1 ULP | `direct_failure` | 通过 |
| 更新少 20% | `direct_failure` | 通过 |
| 局部清零 `v≈1e-37` | `direct_failure`（含 `v_nonzero_erased`） | 通过 |
| 正确但不可辨的小更新 | `audit_incomplete` + precision gap | 通过 |
| 从实测差异反推/缩窄区间 | 原始量重推拒绝 | 通过 |

## 3. 权威纯合成全链 R0→P→I→F（v0.6）

运行根：`derived/audit/w1_real_training_update_audit/synthetic/v0.6_final/`。

- 配置/规模：full `omniretarget_motion` 网络（d_model 240、4 layers、6 heads、time_window 31）；
  roster **130 deterministic + 134 diffusion = 264** 个 tensor，行目录 **3864 = 14N+168**。
- 四窗 `sub10_largebox_063/085` 两个 demo 的重叠窗；每模型两个连续更新、各 4 个 B=4 micro；
  16 个诊断 micro/路径；CPU generator seed `20261008`。
- 四 stage 全部 `done`/sealed，索引与台账一一对应；四份 reservation `consumed` 全部结算，
  `prior_call_counts` 等于终态累计快照。
- 调用矩阵精确等于方案 §7.2 全链名义值：checkpoint loads 8、model/conditioner constructions 8/8、
  native optimizer 4、conditioner/backbone forwards 64/64、add_noise 32、draw units 48、
  production loss 32、production FK **64**、parameter backward 32、output/total VJP 32/32、
  native clip 8、independent clip 8、native AdamW 8、explicit updates 8、diagnostic updates 16、
  anchor restores 4、formula checks 8、reference FK 15840、jacSite 115200、micro units 64；
  `scene_setup_mj_forward=0`（合成无真实资产）。
- `training_updates=16`（native 8 + explicit 8）；全部零计数为 JSON 整数 0（FD/state FD/DDIM/
  validation/selection/生产 checkpoint 写入/contact 分支）。
- **结论：`audit_incomplete`**，`complete_evidence=true`，`covered_contract_passed=false`，
  `failures=[]`、`problems=[]`。I 与 F 各 600 条必测 gap：`parameter_delta_precision_gap`
  **392**、`cross_path_sensitivity_gap` **208**、`cross_path_bound_unverifiable` **0**。
  v0.5 的 219 个 sensitivity gap 因本轮状态校准覆盖小坐标并采用 `1e-3` 相对门而减少 11；
  392 个 precision gap 不变。v0.4 的 660/340/2 与 F 合计 1002 条是旧 F 对 I 行 gap 与
  I 报告 gap 的重复计数；旧工件保留不改写。
- 失败/缺口原因：step 1 的 `m/√v≈0.1` 更新约 `1e-5`，而 |theta|~0.1–1 的 float32 写入分辨率为
  `1e-8–6e-8`，大量坐标低于 §6.2 的 5e-4 可辨门；其余为跨路径近相消敏感性。区间输入预算改为
  梯度门允许集合并逐节点向外 R32 舍入后，0 个坐标落在可信区间外。
- F：独立 Python 子进程（pid≠父、ppid=父、exit 0），继承父绝对截止并在读取/构造/micro/backward/
  VJP/clip/update 前重查父 running 登记；重建 schedule、重读磁盘、逐位重放 P/I；保存报告逐字段
  （loss/parts/gnorm/coef/RNG digest/arrays hash/state presence）、两文件 rows 与报告内嵌 rows、
  两侧行多重集/数组 schema/原始量分类、cotangent 三组事实全部复核，`verification_problems=[]`，
  worker problems=[]。
- 成本：R0 **1.1862s**、P **13.6780s**、I **57.6963s**、F **86.6444s**，合计 **159.205s**；
  每阶段 1/2 次、无借额（R0 240/P 900/I 900/F 1200s）。
- 工件：实际 **6.895 GiB**；逐类上界单链 **7.145 GiB**、cohort（每阶段 2 次 attempt）**14.236 GiB
  ≤16 GiB**。
- 关键 SHA：`audit_result.json` `a03b6fc3e74bc57207d5de5198d8200986b1c00dfdc8f9086faccbc14faf7575`、
  `finalize/attempt_001/training_update_final.json` `ffbb939b1d863224945acd44c9d72a0353ca14548b83789a3dec933380ea827b`、
  `registration/attempt_001/registration.json` `c9d3a08e35209a2d4196b40b7412bd7efa459257ddf9af16c7274269395b53b6`；
  R0 registration 绑定方案 v0.6 `f69694c68832d759ecb68526cb5fb9619a79d5ef662aa82fe1cacf781a53ae89`。

## 4. 准入包（v0.6）

路径：`derived/audit/w1_real_training_update_audit/admission/attempt_008/admission_package.json`，
SHA-256 **`dc163b6001ae9eef57153ddd797aee05d464b93384fb4b7cb6a36a77142c0c93`**。

- `passed=true`、`problems=[]`；`status={authorized:false, registered:false, run:not_run,
  section_10_2_checked:false}`。
- 绑定 tolerance contract 文件 SHA `582ea60a…`、evidence `56ae9663…`、fp32 canonical
  `6d7a24d0…`、roster `264`/3864 行、两份父运行 7+3 个文件哈希、两 checkpoint 与 train config。
- `read_scope`：metadata-only；`window_values_loaded=false`、`checkpoint_torch_load=false`、
  `cohort_created=false`、`registered_attempt_created=false`、conditioner/model/VJP/optimizer 调用 0；
  只构造了 CPU 随机初始化 roster（2 个 model constructor）。
- 实现文件哈希与 v0.6 权威运行 R0 registration 完全一致（方案 SHA `f69694c6…`、
  审计模块 `fbe46689…`、测试 `b816ea01…`）；消费侧 `admission_package_problems` 返回空。
- 资源上界 `one_chain_gib=7.145`、`estimated_artifacts_gib=14.236`、`within_cap=true`。
- 授权模板十项 checklist 仍全为 `null`；真实模式在授权/准入/调用余额校验后仍拒绝
  （`real_provider_not_registered_in_ic`）。

## 5. 测试

- 新增模块：`uv run --locked pytest tests/test_w1_real_training_update_audit.py -q` → **46 passed（93.85s）**。
  v0.6 新增 7 项第四轮反例：小状态/门的实际分母校准；撤回写入最终累计快照并与 `consumed`
  对账；P 阶段在 backward 前执行本地截止（慢 forward 后无 backward/step 计数）；主 P/I
  `unit_admission` 守卫拒绝封口/未登记父尝试；I 在 reference context 前检查、过期时不执行 240 次参考 FK；cotangent joint 组清零被分组门拒绝、缺失
  cotangent 记为 `group_shape` 且不崩溃、行哈希捕获分组篡改；F 在 `remaining=300s/reserve=180s`
  时仍 spawn 且 deadline 只扣一次预留。
- 相关回归（v0.6 重跑）：`tests/test_omniretarget_w1.py`、`tests/test_w1_de_numerics.py`、
  `tests/test_w1_training_objective_audit.py`、`tests/test_w1_real_objective_audit.py`、
  `tests/test_w1_real_admission.py`、`tests/test_w1_real_fk_audit.py` 合计 **263 passed（148.51s）**。
- 全项目套件本轮未重跑；不把历史全量计数当作本轮验收。

## 6. 复现命令

```bash
uv run --locked pytest tests/test_w1_real_training_update_audit.py -q
uv run --locked python scripts/audit_w1_real_training_update.py --calibrate-synthetic
uv run --locked python scripts/audit_w1_real_training_update.py --synthetic \
  --tolerance-contract derived/audit/w1_real_training_update_audit/tolerance_contract.json \
  --out derived/audit/w1_real_training_update_audit/synthetic/v0.6_final
uv run --locked python scripts/audit_w1_real_training_update.py --admission
```

## 7. 第四轮审阅修复映射（4 P1 + 1 P2）

| # | 缺口 | 修复与反例 |
| --- | --- | --- |
| P1-1 | 校准仍排除小 m/v | `_state_cross_path_ratio` 覆盖全部 `abs(reference)>0` 坐标、用门分母；旧 5e-4 实测 0.313871（4×=1.255>1），改 1e-3 后 0.156936（4×=0.628）；`m=0 vs 1e-9`、50%、单侧零仍拒绝 |
| P1-2 | F 撤回封口保存执行前余额 | pre-seal 否决把 `counter.snapshot()` 传入撤回，索引/台账保存最终累计；`final-before==consumed` 逐类对账；矩阵否决反例 `prior_call_counts`=16 更新/64 forwards，不再是 8/32 |
| P1-3 | 主 P/I 截止检查为空操作 | `check()` 无条件 `_ensure_deadline`，主链在 backward/clip/step/VJP 前执行；主 P/I 另传 `_stage_unit_guard`（running/cohort/registration/deadline）；慢 forward 反例在 backward 前停止且 `production_parameter_backward=0` |
| P1-4 | 输出 cotangent 缺分组门 | `compare_cotangent_gradient` 增加 `pos/rot6d/joint` independent relative-L2/cosine 与整组擦除检查；行保存 `group_gates`/problems/`group_max_ratio`，F 按 canonical hash 绑定；joint 清零反例整张量通过但分组拒绝 |
| P2-5 | F 重复扣收口预留 | `deadline=t0+remaining-reserve` 为唯一工作截止；pre-spawn、timeout、worker guard、`unit_admission` 都直接比较 `perf_counter()` 与该截止；300s/180s 反例仍 spawn，deadline 约 120s |

v0.5 的第三轮 8 P1+4 P2 修复与 39 项测试保留在历史（`admission/attempt_004`、`synthetic/v0.5_final`），
当前权威为 v0.6。

## 8. 边界与未覆盖

- 真实 provider、真实 checkpoint strict load、真实父数组/条件、四窗真实输入、真实网络参数反向、
  真实 FK/资产闭包、真实 AdamW 更新、真实 R0→P→I→F、真实授权/注册均未实施或未运行。
- 合成全链的 `audit_incomplete` 只说明该矩阵在冻结精度门下不能证明覆盖通过；不说明 AdamW/网络实现
  错误，也不说明真实链会通过或失败。600 条 gap 是精度/相消的必测不完整，不是 failure。
- 选模、DDIM、端到端质量、干预收益、数据 loader/epoch 等价、生产接入仍 `not_audited`；
  `objective_coverage=partial`、`full_objective_contract_passed=null`、
  `attribution=attribution_insufficient`、`factor_selected=false`、`auto_execute=false`、
  `final_interaction_method=not_ready`、`production_integration=false`。
- U0 `incomplete`、UR `no_quality_gain`、原 F `not_ready`、旧预测 r6 不可验收、旋转融合不接入全部保留。
- 下一步：复阅 v0.6/校准/准入包，另行实施真实 provider 并取得方案明确的真实授权；在此之前不得创建
  真实 cohort 或消耗真实调用额度。

## 9. 历史证据

- v0.4：权威运行 `synthetic/final/`、校准 `history/v0.4_5d9230cf/`、准入 `admission/attempt_001/`
  （`audit_incomplete`、1002 条旧统计、SHA `2c7f52ef…`）；其 F 行 gap 重复计数和状态绝对豁免
  已在 v0.5 修复，旧工件不改写。
- v0.5：文档冻结前的 `synthetic/v0.5_final`（方案 SHA `7ca93760…`）移入
  `synthetic/history/v0.5_final_plan_7ca93760/`，对应准入 `admission/attempt_003/`（SHA `a9c8b50e…`）；
  方案 `991d2d38…` 的权威运行为 `synthetic/v0.5_final/` 与 `admission/attempt_004/`（SHA `c7b4fae2…`），
  611 条 gap（392/219/0）、39 tests。
- v0.6 文档冻结前用旧方案字节运行过一次 v0.6 语义链，已移入
  `synthetic/history/v0.6_pre_plan_*/`（计划外中间运行，不作为准入依据）；之后发现缺失
  cotangent 时分组门会 `KeyError`，修复为记录 `group_shape` 并补测试，修复前运行移入 `synthetic/history/v0.6_pre_nonefix_*/`；再加 I reference-context 检查前的运行移入 `synthetic/history/v0.6_pre_refctx_*/`，加检查后、改方案前的运行移入 `synthetic/history/v0.6_pre_plan46_*/`。`admission/attempt_005/`（`f05fc5a0…`）、
  `attempt_006/`（`05875d09…`）与 `attempt_007/`（`accb923f…`）为中间版本，不作准入依据。
  当前权威为方案 `f69694c6…`、`synthetic/v0.6_final/` 与 `admission/attempt_008/`（SHA `dc163b60…`）。
