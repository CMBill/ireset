# W1 真实网络参数梯度与优化更新链路审计 —— I/C 验证与真实执行记录

日期：2026-11-04。冻结方案：[预注册 v0.9](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_real_training_update_audit_prereg_draft.md)、[执行计划 v0.11](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_real_training_update_provider_execution_plan.md)及[alpha 修正补充计划 v0.12](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_real_training_update_alpha_fix_plan.md)。
状态：**绑定迁移 M0–M5 已完成，Gate B 已按最终字节更新为 `calibration/attempt_015` + `synthetic/attempt_015` + `binding_migration_synthetic/attempt_002` + `binding_migration_gate_b/attempt_002` + `admission_real/attempt_015`；M6 未授权、未执行。首次真实结果仍为 R0 完成、P incomplete、I/F 未启动、实际更新 2 次。**

当前 Gate B 权威见 §12（历史 attempt_013）与 §13.4（绑定迁移最终字节 attempt_015/attempt_002）；迁移初稿/修订1历史见 §13/13.1、当前方案修订见 §13.2、M0/M1 见 §13.3；真实执行事实见 §11 与[首次执行报告](/home/c112/codes/bill/ireset/docs/verification/omniretarget_w1_real_training_update_audit_report.md)。§1–10.11 保留原 I/C、合成、准入及未授权时点的历史记录；其中“真实未读/未运行”仅适用于各节原记录时点。旧方案与 attempt_012 工件保留原字节。

## 1. 交付文件

| 文件 | 作用 |
| --- | --- |
| `src/ireset/eval/w1_real_training_update_audit.py` | 审计器：roster/行目录、合成 fixture、P/I 捕获、独立参考、有效 float32 AdamW、跨路径传播、R0/P/I/F、预算/封口、校准、准入、真实门 |
| `scripts/audit_w1_real_training_update.py` | CLI：`--synthetic`、`--calibrate-synthetic`、`--admission`、`--real`（门后拒绝） |
| `tests/test_w1_real_training_update_audit.py` | 55 项纯合成反例/接口测试（v0.8 为 54 项，v0.9 新增 1 项第七轮反例） |
| `src/ireset/train/trainer.py` | 抽出共享 `predict` / `begin_update` / `accumulate_micro` / `clip_gradients` / `step_optimizer`，新增只读 observer、审计注入 `loss_fn`/`build_optimizer` 与 `clip_coefficient`；默认路径不变 |

共享单元由原生 `Trainer.run` 与审计共同调用。`observer=None` 与 `observer={}` 的
theta/m/v/step 逐位相等，loss/parts 与全局 RNG 不变；NaN 输入的异常类型/原因一致
（`tests` 覆盖 deterministic/diffusion）。

## 2. 冻结校准与有效 float32 契约

目录：`derived/audit/w1_real_training_update_audit/`。v0.4–v0.8 的旧校准保存在
`history/v0.4_5d9230cf/`、`history/v0.5_9a964372/`、`history/v0.6_582ea60a/`、
`history/v0.7_3bce1fd5/` 与 `history/v0.8_271df4e3/`，以下为 v0.9 当前文件。

| 工件 | SHA-256 |
| --- | --- |
| `tolerance_contract.json` | `7d899d88955e2ef9660471be83bcde5c2387ddef73302bf20a8c650f7e290f72` |
| 校准 evidence（contract 内） | `aa4bd83167ea11a7c1cb159e494269b7d843f8233bbbc272558caf9c59e0a9ed` |
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
| `theta_cross_path` | 0.0752 | 0.3008 |
| `state_cross_path` | **0.1448** | **0.5793** |
| `delta_cross_path` | **0.2161** | **0.8644** |
| `delta_same_input` | 0.2314 | 0.9255 |

后四项按**各比较门自身的适用性**取全链观测与标量近例的最大值（`source=synthetic_chain+scalar_near_1e-7`）：
只要该门自身通过（例如 m/v 的 `m_ok`/`v_ok`），即使整行因更新量不可辨而 `audit_incomplete` 也计入；
直接 failure 或被该门拒绝的行不计入。

**跨路径更新量的未舍入参考（v0.8）**：§6.2 要求
`abs(delta_measured_P-delta_ref64_I)≤rtol*abs(delta_ref64_I)`。v0.7 的 `_cross_path_flags`
与 `max_delta_ratio` 传入 `delta_i_measured`，比较的是两条已舍入的实测增量。评审二步例
（theta `4.3598099728114903e-4`、m `1e-3`、v `1e-7`、gP `-8.899898268282413e-3`、
gI `-8.9`）用已舍入 I 实测得 0.994504（错误通过），用未舍入参考得 **1.001830**，现记
`cross_path_sensitivity_gap`。现 close 门用 P 实测 vs I 的未舍入 `delta_ref64`，零/非零存在性
另比较 P/I 两个参考；`small_unobservable_update` 仍为 `audit_incomplete`（两侧参考同为非零、
实测均为零，不因存在性误报）。校准 `delta_cross_path` 随之重算：v0.7 曾据已舍入实测把门调至
`1e-3`；改用未舍入参考后，原始提案 `5e-4` 下实测 **0.216102**（4×=**0.8644≤1**），因此 v0.8
**恢复 `5e-4`**（更严格），`1e-3` 撤回。`state_cross_path=2e-3` 与逐门适用性统计不变。

**目标窗解码后的 FK 截止（v0.9）**：v0.8 的 `ReferenceContext.__init__` 只在每个目标窗
`independent_decode` **之前**检查截止；解码返回后到 `synthetic_fk_np` 之间没有守卫，若解码期间
越过 deadline，该窗目标 FK 仍会执行并计入 `reference_mj_forward`（评审复现为一窗 60 个状态），
直到下一窗入口才抛出 `budget_deadline`。现解码返回、目标 FK 调用前**再次执行同一
`reference_target_{key}` 检查**（同一 `check`：截止 + `unit_admission`）；`evaluate` 的
`reference_fk`/`reference_jacobian` 前置检查不变。新增受控时钟反例：第 2 个目标窗解码后过期时，
只有第 1 窗的 FK 执行（`reference_mj_forward` 等于该窗状态数），第 2 窗 FK 计数 0、Jacobian 与
输出 VJP 均为 0。

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
  `joint`(9:38) 三组执行独立 relative-L2≤2e-4/cosine≥1-1e-6 且生产非全零。**分组零噪声门同样要求
  参考与生产两侧范数均在 1e-10 深带**：参考 joint 1e-13、生产 joint 1e-8（范数 8.34e-7）
  由 `tiny_group_joint` 拒绝，两侧均在深带的合法噪声仍通过。组事实与 problems 进入 I 行
  canonical hash，F 重放重新计算并拒绝任何只改分组事实的行漂移。
- 参考 FK 截止单元（v0.8/v0.9）：`run_i_stage` 在 `ReferenceContext.evaluate` 前检查
  `i_ref_fk_{u}_{m}`；`ReferenceContext` 在目标窗构造、**目标窗解码后 FK 前**、evaluate、FK 与
  Jacobian 调用前分别执行同一 `check`（截止 + `unit_admission`）。受控时钟在首个 forward 返回后
  过期时，不再新增 480 次 `reference_mj_forward`/3600 次 `reference_mj_jac_site`；解码期间过期时
  该窗 FK 不启动；FK 若已开始则完成，Jacobian 不启动。输出 VJP/clip/参考更新前的检查与
  “执行后计数”保持不变。
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

## 3. 权威纯合成全链 R0→P→I→F（v0.9）

运行根：`derived/audit/w1_real_training_update_audit/synthetic/v0.9_final/`。

- 配置/规模：full `omniretarget_motion` 网络（d_model 240、4 layers、6 heads、time_window 31）；
  roster **130 deterministic + 134 diffusion = 264** 个 tensor，行目录 **3864 = 14N+168**。
- 四窗 `sub10_largebox_063/085` 两个 demo 的重叠窗；每模型两个连续更新、各 4 个 B=4 micro；
  16 个诊断 micro/路径；CPU generator seed `20261008`。
- 四 stage 全部 `done`/sealed，索引与台账一一对应；四份 reservation `consumed` 全部结算，
  `prior_call_counts` 等于终态累计快照（16 更新/64 backbone forwards）。
- 调用矩阵精确等于方案 §7.2 全链名义值：checkpoint loads 8、model/conditioner constructions 8/8、
  native optimizer 4、conditioner/backbone forwards 64/64、add_noise 32、draw units 48、
  production loss 32、production FK **64**、parameter backward 32、output/total VJP 32/32、
  native clip 8、independent clip 8、native AdamW 8、explicit updates 8、diagnostic updates 16、
  anchor restores 4、formula checks 8、reference FK 15840、jacSite 115200、micro units 64；
  `scene_setup_mj_forward=0`（合成无真实资产）。
- `training_updates=16`（native 8 + explicit 8）；全部零计数为 JSON 整数 0（FD/state FD/DDIM/
  validation/selection/生产 checkpoint 写入/contact 分支）。
- **结论：`audit_incomplete`**，`complete_evidence=true`，`covered_contract_passed=false`，
  `failures=[]`、`problems=[]`。I 与 F 各 582 条必测 gap：`parameter_delta_precision_gap`
  **392**、`cross_path_sensitivity_gap` **190**、`cross_path_bound_unverifiable` **0**。
  v0.8 的 185 个 sensitivity gap 在该轮增加 5 个：可辨跨路径更新量门从基于已舍入实测的
  `1e-3` 恢复为原提案 `5e-4`（更严格）并改用未舍入 I 参考；392 个 precision gap 不变。
  v0.9 只新增解码后、目标 FK 前的截止检查，不触数值路径，因此 gap 组合保持 392/190/0。
  v0.4 的 660/340/2 与 F 合计 1002 条是旧 F 对 I 行 gap 与 I 报告 gap 的重复计数；旧工件保留不改写。
- 失败/缺口原因：step 1 的 `m/√v≈0.1` 更新约 `1e-5`，而 |theta|~0.1–1 的 float32 写入分辨率为
  `1e-8–6e-8`，大量坐标低于 §6.2 的 5e-4 可辨门；其余为跨路径近相消敏感性。区间输入预算改为
  梯度门允许集合并逐节点向外 R32 舍入后，0 个坐标落在可信区间外。
- F：独立 Python 子进程（pid 3138783、ppid 3136077，pid≠ppid；exit 0），继承父绝对截止并在
  读取/构造/micro/backward/VJP/clip/update 前重查父 running 登记；重建 schedule、重读磁盘、
  逐位重放 P/I；保存报告逐字段（loss/parts/gnorm/coef/RNG digest/arrays hash/state presence）、
  两文件 rows 与报告内嵌 rows、两侧行多重集/数组 schema/原始量分类、cotangent 三组事实全部复核，
  `verification_problems=[]`，worker problems=[]。
- 成本：R0 **1.1946s**、P **13.3511s**、I **58.9684s**、F **87.5429s**，合计 **161.0571s**；
  每阶段 1/2 次、无借额（R0 240/P 900/I 900/F 1200s）。
- 工件：实际 **6.895 GiB**（7,404,305,878 bytes）；逐类上界单链 **7.145 GiB**、cohort（每阶段
  2 次 attempt）**14.236 GiB≤16 GiB**。
- 关键 SHA：`audit_result.json` `96b2c2ae50b0b29f6c64cbf869f436e3b284da5363e7a9f70fa4dce995c2ee9e`、
  `finalize/attempt_001/training_update_final.json` `3055f29dda0e2a645f6093e707b3400fbae629bd98037a1259dcabc31d288abb`、
  `registration/attempt_001/registration.json` `43a52e9699760fb3aa4b1ebb39d98904f0ad786aa4fa30df55da3dd973f9b3e4`；
  R0 registration 绑定方案 v0.9 `e2ba8ef9edb7381dbe410f7e0e290a6fbd702323a4416eeacfee4d2d14796242`。

## 4. 准入包（v0.9）

路径：`derived/audit/w1_real_training_update_audit/admission/attempt_012/admission_package.json`，
SHA-256 **`44e229f1052823a92b8376b36ea8d79aba0854db3780264b3699e16319d855be`**。

- `passed=true`、`problems=[]`；`status={authorized:false, registered:false, run:not_run,
  section_10_2_checked:false}`。
- 绑定 tolerance contract 文件 SHA `7d899d88…`、evidence `aa4bd831…`、fp32 canonical
  `6d7a24d0…`、roster `264`/3864 行、两份父运行 7+3 个文件哈希、两 checkpoint 与 train config。
- `read_scope`：metadata-only；`window_values_loaded=false`、`checkpoint_torch_load=false`、
  `cohort_created=false`、`registered_attempt_created=false`、conditioner/model/VJP/optimizer 调用 0；
  只构造了 CPU 随机初始化 roster（2 个 model constructor）。
- 实现文件哈希与 v0.9 权威运行 R0 registration 完全一致（方案 SHA `e2ba8ef9…`、
  审计模块 `7346df72…`、测试 `b273cfd0…`）；消费侧 `admission_package_problems` 返回空。
- 资源上界 `one_chain_gib=7.145`、`estimated_artifacts_gib=14.236`、`within_cap=true`。
- 授权模板十项 checklist 仍全为 `null`；真实模式在授权/准入/调用余额校验后仍拒绝
  （`real_provider_not_registered_in_ic`）。

## 5. 测试

- 新增模块：`uv run --locked pytest tests/test_w1_real_training_update_audit.py -q` → **55 passed（97.10s）**。
  v0.9 新增 1 项第七轮反例：`independent_decode` 第 2 次调用返回后把受控时钟推到 deadline 之后，
  `reference_target_W1` 抛 `budget_deadline`，只有第 1 窗目标 FK 执行（`reference_mj_forward`
  等于该窗状态数）、第 2 窗 FK=0、jacSite=0、output VJP=0；修复前该窗仍执行并计入 60 个状态。
  v0.8 的 3 项第六轮反例（未舍入 I 参考；`i_ref_fk_1_0`；FK 后过期 Jacobian 不启动）保留。
- 相关回归（v0.9 重跑）：`tests/test_omniretarget_w1.py`、`tests/test_w1_de_numerics.py`、
  `tests/test_w1_training_objective_audit.py`、`tests/test_w1_real_objective_audit.py`、
  `tests/test_w1_real_admission.py`、`tests/test_w1_real_fk_audit.py` 合计 **263 passed（150.09s）**。
- 全项目套件本轮未重跑；不把历史全量计数当作本轮验收。

## 6. 复现命令

```bash
uv run --locked pytest tests/test_w1_real_training_update_audit.py -q
uv run --locked python scripts/audit_w1_real_training_update.py --calibrate-synthetic
uv run --locked python scripts/audit_w1_real_training_update.py --synthetic \
  --tolerance-contract derived/audit/w1_real_training_update_audit/tolerance_contract.json \
  --out derived/audit/w1_real_training_update_audit/synthetic/v0.9_final
uv run --locked python scripts/audit_w1_real_training_update.py --admission
```

## 7. 第七轮审阅修复映射（1 P1）

| # | 缺口 | 修复与反例 |
| --- | --- | --- |
| P1-1 | 目标窗解码后、FK 前缺截止检查 | `ReferenceContext.__init__` 在 `independent_decode` 返回后、目标 `synthetic_fk_np` 调用前再次执行同一 `reference_target_{key}` 检查（截止 + `unit_admission`）。受控时钟第 2 窗解码后过期：仅第 1 窗目标 FK 执行（`reference_mj_forward`=该窗状态数），第 2 窗 FK=0、jacSite=0、output VJP=0，于 `reference_target_W1` 抛 `budget_deadline`；修复前该窗仍执行并计入 60 状态 |

v0.8 的第六轮 2 P1 修复与 54 项测试保留在历史（`synthetic/history/v0.8_final_plan_c3909a22/`、
`admission/attempt_010/`），当前权威为 v0.9。

## 8. 边界与未覆盖

- 真实 provider、真实 checkpoint strict load、真实父数组/条件、四窗真实输入、真实网络参数反向、
  真实 FK/资产闭包、真实 AdamW 更新、真实 R0→P→I→F、真实授权/注册均未实施或未运行。
- 合成全链的 `audit_incomplete` 只说明该矩阵在冻结精度门下不能证明覆盖通过；不说明 AdamW/网络实现
  错误，也不说明真实链会通过或失败。582 条 gap 是精度/相消的必测不完整，不是 failure。
- 选模、DDIM、端到端质量、干预收益、数据 loader/epoch 等价、生产接入仍 `not_audited`；
  `objective_coverage=partial`、`full_objective_contract_passed=null`、
  `attribution=attribution_insufficient`、`factor_selected=false`、`auto_execute=false`、
  `final_interaction_method=not_ready`、`production_integration=false`。
- U0 `incomplete`、UR `no_quality_gain`、原 F `not_ready`、旧预测 r6 不可验收、旋转融合不接入全部保留。
- 下一步：复阅 v0.9/校准/准入包，另行实施真实 provider 并取得方案明确的真实授权；在此之前不得创建
  真实 cohort 或消耗真实调用额度。

## 9. 历史证据

- v0.4：权威运行 `synthetic/final/`、校准 `history/v0.4_5d9230cf/`、准入 `admission/attempt_001/`
  （`audit_incomplete`、1002 条旧统计、SHA `2c7f52ef…`）。
- v0.5：`synthetic/v0.5_final`（方案 `991d2d38…`）与 `admission/attempt_004/`（`c7b4fae2…`），
  611 条 gap（392/219/0）、39 tests；文档冻结前的旧运行在 `synthetic/history/v0.5_final_plan_7ca93760/`。
- v0.6：`synthetic/history/v0.6_final_plan_f69694c6/`、`admission/attempt_008/`（`dc163b60…`），
  600 条 gap（392/208/0）、46 tests；校准 `history/v0.6_582ea60a/`。
- v0.7：`synthetic/history/v0.7_final_plan_df2a1a2b`（方案 `df2a1a2b…`，577 条 gap、
  392/185/0、51 tests）与 `admission/attempt_009/`（`27d54a3f…`）；校准 `history/v0.7_3bce1fd5/`。
  v0.8 过程中用 1e-3 delta 门运行过的初步链保留在 `synthetic/history/v0.8_pre_1e-3_6ec0be6b/`
  （校准 `history/v0.8_pre_6ec0be6b/`），用 v0.7 方案字节运行过的 5e-4 初步链保留在
  `synthetic/history/v0.8_pre_plan_df2a1a2b/`，`synthetic/history/v0.8_final_plan_70fad4f8/`
  为改方案字节前的中间权威运行；均不作准入依据。
- v0.8：权威运行 `synthetic/history/v0.8_final_plan_c3909a22/`（方案 `c3909a22…`，582 条 gap、
  392/190/0、54 tests，成本 160.2192s、工件 7,404,305,871 bytes）、准入 `admission/attempt_010/`
  （`d8ccda86…`）与校准 `history/v0.8_271df4e3/`。
- v0.9：两次改方案字节前的中间运行保留在 `synthetic/history/v0.9_final_plan_e3849c3b/`
  （成本 164.4234s）与 `synthetic/history/v0.9_final_plan_9ff3b1f1/`（成本 160.5867s），均为
  582 条 gap，不作准入依据；当前权威为方案 `e2ba8ef9…`、`synthetic/v0.9_final/` 与
  `admission/attempt_012/`（SHA `44e229f1…`）。

## 10. Gate A/B 真实 provider 接入与执行计划实现（2026-10-31）

执行计划：[W1 真实 provider 接入与执行计划](../plans/omniretarget_w1_real_training_update_provider_execution_plan.md)
（v0.11，`gate_a_implemented / gate_b_acceptance_passed / gate_c_not_authorized /
gate_d_e_not_run`）。§10.1–§10.4 记录 v0.6 初次 Gate A/B 实现；§10.6 记录 v0.7 的 5 P1+4 P2
修复与 attempt_007；§10.7 记录 v0.8 的 4 P1+3 P2 修复与 attempt_008；§10.8–§10.9 记录
v0.9 的 4 P1+2 P2 修复、选择字段撤回与 attempt_009/010；§10.10 记录 v0.10 的 1 P1+1 P2
修复与 attempt_011；§10.11 记录 v0.11 的 1 P2 修复与 attempt_012，为当前权威。Gate C
（新的 §10.2 明确授权）、Gate D（真实 R0）与 Gate E（真实 P→I→F）未执行。

### 10.1 Gate A 实现

- 新增 `src/ireset/eval/w1_real_training_update_provider.py`。`RealTrainingProvider.metadata()`
  只读取父锚路径/字节 SHA、NPZ header（键/dtype/shape/order，不读 payload）、父 index/ledger
  终态、checkpoint/config/资产引用文本、静态 roster 与 runtime；`read_scope` 记录
  `npz_values_read=0`、`checkpoint_payload_read=0`、`torch_load_calls=0`、
  `model_constructor_calls=0`、`muJoCo_compile_calls=0`。
- `materialize()` 在已登记 P/I/F 阶段读取四窗数组、常数、R0 噪声与父数组，先做 dtype/shape/
  有限性/旋转域/joint 域/关节限位/box quat/wfc 检查、独立 encode 与 tokenizer 复现、alpha
  重推与 x_t 参考，P 写入 `input_admission.json`；I/F 重新读取并逐字段绑定该工件。失败时
  P/I 在构造任何模型前以 `complete_evidence=false` 收口。
- 真实参考后端 `RealReferenceBackend` 在每次 `positions/jacobians` 入口执行截止与父登记
  守卫，并在每个 `mj_forward/mj_jacSite` 前检查绝对截止、成功返回后分别增加
  `reference_mj_forward/reference_mj_jac_site`；部分失败保留已完成量，不按 batch 名义补足。
  合成后端适配同一接口，`ReferenceContext` 不再直接要求 `SyntheticChain`。
- 真实 `build_trainer` 使用 strict `load_state_dict`（缺键/额外键/dtype/shape/非有限/step/
  config/root_stats/usable 均拒绝），P/F-P 构造一次生产 LossAssembler/FK/箱体静态场景，
  I/F-I 禁止生产 loss 且不构造原生优化器；`Trainer` 只新增可选 `construction_observer`
  计数钩子，默认 `None` 路径数值不变。

### 10.2 续跑、选择与 F 时序

- `stage_state(root, stage, attempt=None)` 支持显式 attempt；默认允许一个已结算失败 attempt
  与唯一可消费 attempt 共存，多个可消费候选、未封口/撤回选择或未知历史均拒绝；
  `verify_history` 要求盘 × index × ledger × reservation 一致、成本有限非负、`final-before
  == consumed`。
- `_cohort_chain` 按 R0→P→I→F 顺序重推每个 attempt 的调用增量；`selected_chain_counts`
  （仅选中链，精确等于 mode 名义列）、`stage_call_counts`（同阶段全部历史）与
  `cohort_call_counts`（全部 attempt）分开保存；选中链 16 次不因 cohort 17 次被否决，反
  之亦然。`--resume --stage` 只追加一个新 attempt，上游复用不重复计费。
- F worker 在 `running/consumed=null` 时以 `candidate_attempts` + 本地 `L_F` 计算
  `provisional_chain_counts` 并检查阶段/cohort 上限；父进程仅在完整 24 类快照核验合并后
  settle，再持久化 `call_balance_before/after`、从盘重读并终检父绑定/选中链/历史上限；
  封口后按通常消费者规则确认四个终态 attempt，通过后才把完整 `selected_attempts/
  selected_chain_counts` 写入报告并重封口；失败时按 publishing→terminal 撤回。
- scene 计数按 mode 校验：synthetic 全 0，real P/F 各 1、全链 2；admission/full-chain/F
  父子检查均按 mode 生成预期。

### 10.3 Gate B 执行

全部命令在最终实现字节上执行；校准/合成命令不设置 `OMP_NUM_THREADS`（与 v0.9 运行环境一致），
仅 pytest 使用 `OMP_NUM_THREADS=1`。

- 新 provider 测试 `tests/test_w1_real_training_update_provider.py`：**20 passed**
  （临时文件、解析 fixture、fake MuJoCo/G1Spec、metadata/schema/stub；不读真实 checkpoint/
  数组/资产）。
- 既有 `tests/test_w1_real_training_update_audit.py`：**55 passed**。
- 相关回归 6 文件（`test_omniretarget_w1.py`、`test_w1_de_numerics.py`、
  `test_w1_training_objective_audit.py`、`test_w1_real_objective_audit.py`、
  `test_w1_real_admission.py`、`test_w1_real_fk_audit.py`）：**263 passed**。
- 最终字节合并运行（上述 2 + 6 文件）：**338 passed（392.20s）**；全项目套件未重跑。
- 校准 `derived/audit/w1_real_training_update_audit/calibration/attempt_005/`：
  `tolerance_contract.json` SHA `7e40a94dc960eae8a055e227ca9273adbc5047c5681027cc29bfc5698665a46d`、
  evidence `f0050d6a8035c63fb1d1d35f4db7faa42b5901e9940bfe719d88c87c77b467f0`、
  fp32 canonical `f281ab1fb625e9ff74adea689afbdd5a69b551542ba278076d630d53dabf9267`、
  `calibration_verification.json` SHA `767c9d66a3db188ebecd398719fd09dd098a4e77ee32064bac2c8f68f15ac882`；
  `recompute_problems=[]`、`four_x_ok=true`。`tolerances/rules/tiny_zero_floors` 与 v0.9
  逐项相同；仅环境（内核版本/CPU 线程）使 fp32 canonical/evidence SHA 与 v0.9 不同，未改阈值。
- 完整合成链 `derived/audit/w1_real_training_update_audit/synthetic/attempt_005/`：
  `audit_result.json` SHA `897021226645c1c240e81bc0b78d03a042c9d077c61ad10c2e066f6f43d5e584`、
  final `013f43a43a03122b6812c74a193a8be06f3df77984b61b2ee7b54b66cefca5e8`；
  四 stage `done/sealed`、264 tensor/3864 行、调用矩阵精确（8 loads、64/64 forwards、
  48 draw units、32/32 VJP、64 production FK、16/16 updates、15840/115200 reference FK、
  64 micro）、`training_updates=16`；结论 **`audit_incomplete`**（`complete_evidence=true`、
  `failures=[]`、`problems=[]`、**582** gap：392 `parameter_delta_precision_gap` +
  190 `cross_path_sensitivity_gap` + 0 bound-unverifiable）。F 新进程 pid 204048/ppid 201457
  （pid≠ppid、ppid=父）、exit 0、`verification_problems=[]`；成本 R0 1.1869/P 13.7560/
  I 57.8924/F 96.3983 = **169.2336s**，各 1/2 次；工件 7,404,741,172 bytes（约 6.896 GiB）。
  `synthetic_acceptance.json` SHA `706c0b4167294032aa3b165f0e7cc30d349403bf6f3e22e525d79efdaf54318e`，
  `passed=true/problems=[]`，绑定重算报告、roster、所有阶段报告与实现/环境。
- 真实 metadata-only admission `derived/audit/w1_real_training_update_audit/admission_real/attempt_005/`
  SHA **`a0d55a92636dec1fc7eda60d2f8e42f87e0bfcb56ae11755bd5886170dde6978`**，
  `passed=true/problems=[]`、`provider_mode=real/scope=metadata_only`、
  `authorized=false/registered=false/run=not_run/section_10_2_checked=false`；
  `read_scope` 的 `npz_values_read/checkpoint_payload_read/torch_load_calls/
  model_constructor_calls/muJoCo_compile_calls/forward/VJP/optimizer` 全 0，
  `asset_closure_verified=true`（37 引用、closure `46cb786a…`）；父锚 10/10、checkpoint 2/2、
  训练配置、264 tensor/3864 行、真实全链 scene=2/16 updates 与校准验证
  （canonical `245cb998…`）/接受报告 SHA 全部绑定；validator `[]`。
- 被取代证据：首次中间运行使用 `OMP_NUM_THREADS=1`，在
  `i_accumulated:diffusion:u1:model.net.root_out.weight` 出现 1 条近阈值 failure
  （585 gap），不作为 Gate B 依据；`synthetic/attempt_001` 与 `attempt_002` 已在记录事实后
  删除以释放磁盘（两者均非当前实现字节），`calibration/attempt_001–004`、
  `synthetic/attempt_003–004` 与 `admission_real/attempt_001–004` 保留为历史；其中
  attempt_003/004 因后续实现字节变化被取代，当前权威为 attempt_005。
  阈值、窗口、lr、dtype、噪声/微批矩阵未调整。

### 10.4 边界

- 未读取真实 checkpoint payload、父 NPZ 数值、G1 mesh 或编译锁定资产；未构造真实模型/优化器、
  未执行真实前向/反向/FK/AdamW；未创建真实 cohort/attempt/reservation；未取得新的 §10.2 授权。
- Gate D/E 未运行；真实 R0→P→I→F 的数值行为、严格加载、input_admission、参考 FK 和
  调用矩阵仍未验证。Gate B 通过只表示本模式的元数据准入与合成回归。
- 合成 `audit_incomplete` 是冻结精度/相消门下的必测 gap，不是 AdamW/网络实现错误；不说明真实
  链会通过或失败。`objective_coverage=partial`、`full_objective_contract_passed=null`、
  `attribution=attribution_insufficient`、`factor_selected=false`、`auto_execute=false`、
  `final_interaction_method=not_ready`、`production_integration=false`；U0 `incomplete`、
  UR `no_quality_gain`、原 F `not_ready`、旧预测 r6 不可验收与旋转融合不接入全部保留。
- v0.9 的 `tolerance_contract.json`、`adamw_fp32_contract.json`、`synthetic/v0.9_final/`、
  `admission/attempt_012/` 与冻结方案字节未改写。全项目套件本轮未重跑。

### 10.5 Gate B 后的下一步

先由用户复阅本记录、执行计划 v0.11、`calibration/attempt_012`、`synthetic/attempt_012` 与
`admission_real/attempt_012`（§10.11）；随后提供新的 §10.2 明确授权记录（10 项 checklist
全 true，并绑定执行计划/校准验证/接受报告/准入包/实现哈希）。未取得授权前不得运行
`--real`/`--resume`，不得创建真实 cohort 或消耗真实调用额度。

### 10.6 v0.7 审阅修复与 Gate B 重新冻结（2026-10-31；已被 §10.7 取代）

独立审阅确认 5 P1 + 4 P2；全部按反例修复后，Gate B 以最终字节重新执行。本节保留
attempt_007 的历史事实；当前权威为 §10.7（v0.8/attempt_008）。attempt_006 与更早 attempt
仅作历史保留。

**修复项（均有新增反例）**

- **P1 物化前守卫**：`materialize` 入口、父数组摘要/解码、tokenizer、input admission、
  参考后端构造，以及 `build_trainer` 构造与 `_load_checkpoint`（SHA 与 payload 读取前）
  执行同一 deadline+guard。隔离反例中 guard 在 `materialize_start` 即抛出，数组读取与
  MuJoCo 编译均未发生；`trainer_construction`/`checkpoint_sha` 同样先于构造/哈希/`torch.load`。
- **P1 源数组实际摘要**：每次 materialize（含续跑复用已封口 R0）先重算父
  `objective_arrays` 与 train config SHA，再解码；`input_admission.source_sha256` 使用实际值。
  文件追加字节后 `_load_arrays` 在 `np.load` 前以 `drifted` 拒绝。
- **P1 实现冻结清单**：`IMPLEMENTATION_FILES` 15→25，补入 `model/tokens.py`、
  `model/deterministic.py`、`kinematics/torch_adapter.py`、
  `kinematics/rotation_adapter.py`、`geometry/distance_adapter.py`、
  `embodied/g1_spec.py`、`paths.py`、`eval/training_objective_audit.py`、`pyproject.toml`、
  `uv.lock`；静态测试断言这些计算依赖在清单中且哈希非空。
- **P1 接受文件闭包**：`synthetic_acceptance.json` 的每个阶段记录 attempt/目录/逐文件
  SHA+bytes，另记录 fixture 与 run 级 `cohort.json/attempts_index.json/costs.json` 摘要；
  消费端检查必需文件、报告路径与逐文件存在性/大小/SHA。删除阶段文件、改写内容、搬移全部
  报告路径或篡改 runtime identity 均被拒；旧 attempt_005 接受报告因无文件清单被拒。
- **P1 运行时身份**：fp32 contract 新增 `runtime_identity`（Python 3.12.13、Torch
  2.14.0+cu126、NumPy 2.5.3、SciPy 1.18.1、CUDA 12.6、RTX 3090、MuJoCo 3.13.0、两个
  AdamW 源码 SHA）；校准验证、合成接受、真实准入、R0、P/I 物化与 F 子进程均做无计算
  身份比较，漂移即退回 Gate B。`fp32_contract_problems` 同时复核该身份。
- **P2 续跑调用量**：`required` 只汇总 resumed 阶段及下游。新增反例：P 首次失败消耗 2 次
  加载，重试 P/I 成功，F 失败已结算 4 次（历史 10），resume F 只需 4 次，10+4=14≤16
  成功；旧 10+8=18 的误拒消除，最终 `checkpoint_loads=14`。
- **P2 续跑起点**：`first_start` 只归 resumed 阶段；记录 `pre_register_attempt` 的 t0 证明
  resume P 时 `t0(P) < t0(I) < t0(F)`，上游耗时不再重复计入下游截止。
- **P2 显式选择唯一性**：两个可消费 attempt 下，默认选择与显式指定任一都被
  `stage_multiple_consumable` 拒绝。
- **P2 完成硬门**：`attempt_not_complete` 不再从 `allowed` 过滤；单个或两个
  `done/complete=false` 均不可消费，显式选择返回 `attempt_not_complete`。

**测试（最终字节）**

- `tests/test_w1_real_training_update_provider.py`：**31 passed**（新增 11 项守卫/摘要/
  文件闭包/身份/续跑/选择反例）。
- `tests/test_w1_real_training_update_audit.py`：**55 passed**。
- 六个相关回归文件：**263 passed**。
- 合并运行（2+6 文件）：**349 passed（378.47s）**；全项目套件未重跑。

**Gate B 最终产物（attempt_007）**

- 校准 `derived/audit/w1_real_training_update_audit/calibration/attempt_007/`：
  `tolerance_contract.json` SHA `b97e3e6be7bf02eec08592af69160b4fb101e7f7f9dd6184cc64c0016c4874e4`、
  evidence `8fc90e39d42e4f94ead51bcaba26418a1992aa2f8bb6af593fba0f115e982d90`（v0.9 冻结
  evidence 为 `aa4bd831…`；v0.7 新增 runtime identity 后 `calibration.fp32_contract_sha256`
  由 `6d7a24d0…` 变为 `47390b08…`，故 evidence 摘要不同；数值门/校准数值相同）、
  fp32 canonical `47390b08c52697106573264b6c0d5439b4bc4be78cf390e0cd0e3824c30088cd`
  （新增 runtime identity 后变化）、`calibration_verification.json` 文件 SHA
  `756531cefbd12b1a89db10a613ae295cf6611701b1fc341ff0d4f166871e6659`、canonical
  `8928add378959544e06cf456aea100bc80babedb126134dbe880b8731898f648`；
  `recompute_problems=[]`、`four_x_ok=true`，最大 4× 比值 `delta_same_input=0.925540`
  （其余 `alpha_x_t=0.867157`、`delta_cross_path=0.864409`、`state_cross_path=0.579252`、
  `parameter_gradient=0.361710`、`theta_cross_path=0.300759`）。阈值/规则/零底与 v0.9
  逐项相同。
- 完整合成链 `derived/audit/w1_real_training_update_audit/synthetic/attempt_007/`：
  `audit_result.json` SHA `9ca4b597f2ce5fb2c0e32e69e6fe169938600530f2f94025a82a55302e484568`、
  final `8b2b6f3e28426ea0d5a47d185314562548fb868bf622745c79b54ae849989a49`、
  registration `c9a83f6dc9e09bc6bd592d78ee34da5c34d8986f3cc415ec935d6a9e52a1977d`；
  四 stage done/sealed、264 tensor/3864 行、调用矩阵精确（8 loads、64/64 forwards、
  48 draw units、32/32 VJP、64 production FK、16/16 updates、15840/115200 reference FK、
  64 micro）、`training_updates=16`；结论 **`audit_incomplete`**（`complete_evidence=true`、
  `failures=[]`、`problems=[]`、**582** gap：392 precision + 190 sensitivity + 0
  bound-unverifiable）。F 新进程 pid 1700838/ppid 1698246（pid≠ppid、ppid=父）、exit
  `ok`、`verification_problems=[]`；成本 R0 1.1800/P 13.7248/I 58.7144/F 96.2849 =
  **169.9041s**，各 1/2 次；工件 7,404,757,239 bytes（约 6.896 GiB）。
  `synthetic_acceptance.json` SHA
  `715f087b11fcab0baf713df03fbb7b911954694439cec6e40562278a4694c08f`、
  `passed=true/problems=[]`、`run_sha256=cec55857195d3bb58fb4f87a90c5ecdb81c681eab406d2c95079850e63a6ba31`；
  阶段文件数 R0 3/P 5/I 5/F 12、fixture 6、run 3，消费校验 `[]`。
- 真实 metadata-only admission
  `derived/audit/w1_real_training_update_audit/admission_real/attempt_007/` SHA
  **`3fa6938801499ad5efa0b80bbf4121c857b30e93649a74198a4e6571e859b25b`**，
  `passed=true/problems=[]`、validator `[]`，绑定 acceptance `715f087b…`、校准 canonical
  `8928add3…`、evidence `8fc90e39…`、fp32 `47390b08…`；实现闭包 **25 文件**；父锚 10/10、
  checkpoint 2/2、资产闭包 `checked=true`/37 引用/problems 空；read scope
  `npz_values_read/checkpoint_payload_read/torch_load_calls/model_constructor_calls/
  muJoCo_compile_calls/forward/VJP/optimizer=0`，`asset_closure_verified=true`、
  `cohort_created=false/registered_attempt_created=false`；真实全链 scene=2/8 loads。
  runtime identity 与校准记录逐项一致。
- 被取代证据：`synthetic/attempt_006` 与 `admission_real/attempt_006` 曾在本轮修复后、
  plan 文件更新前生成；plan 更新改变了实现身份，故 attempt_006 被 attempt_007 取代并保留
  为历史。首次 attempt_007 合成运行因磁盘余量 15 GiB < R0 的 20 GiB 门在 R0 失败
  （`r0_disk_free`，无阶段计算）；该失败目录为本轮临时输出，删除后以最终字节重跑。
  为满足磁盘门，曾将已冻结且被取代的 `synthetic/final`（v0.4 历史权威）临时移至
  `/dev/shm`，Gate B 完成后原样移回；`derived/audit/w1_real_training_update_audit/`
  其余历史 attempt 未删除。完成后磁盘余量约 7.3 GiB。
- 边界不变：未读取真实 checkpoint payload/父 NPZ 数值、未编译真实资产、未构造真实模型/
  优化器、未执行真实前向/反向/FK/AdamW、未创建真实 cohort/授权/注册；Gate C/D/E 仍未运行。
  `objective_coverage=partial`、`full_objective_contract_passed=null`、
  `factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、
  `production_integration=false`。v0.9 方案/校准/合成/准入字节未改写。

### 10.7 v0.8 审阅修复与 Gate B 重新冻结（2026-10-31；已被 §10.8 取代）

独立审阅确认 **4 P1 + 3 P2**；全部按反例修复后，Gate B 以最终字节重新执行。本节保留
attempt_008 的历史事实；当前权威为 §10.8（v0.9/attempt_009）。§10.6（v0.7/attempt_007）
及更早 attempt 仅作历史保留。

**修复项（均有新增反例）**

- **P1 构造分项守卫**：`_production_loss` 在 `G1TorchFK` 之前、`LossAssembler`（真实大箱体
  场景编译）之前分别执行同一 deadline+guard；场景编译完成后立即计入
  `scene_setup_mj_forward`，即使后续单元拒绝也保留已完成的实际计数；构造失败不缓存半成品
  assembler。隔离时钟使 FK 构造返回时已过期：仅 FK 被构造，`LossAssembler` 与
  `scene_setup_mj_forward` 均为 0；成功后再次调用仍只计 1 次真实编译。
- **P1 资产闭包复验**：每次 `materialize`（含续跑复用已封口 R0、跳过 metadata）在父数组/
  config 实际摘要之后、tokenizer/参考后端/模型构造之前执行 `_verify_asset_closure`：先核验
  父 FK registration 文件自身冻结 SHA，再核对四个锁定文件、37 个引用、解析目标集合、
  `from`/`tag`/SHA、`reference_count` 与 `closure_sha256`。隔离反例中改写 mesh 字节或删除
  include 均在真实编译前以 `asset_reference_sha`/`asset_reference_missing` 拒绝。
- **P1 封口摘要消费**：`_attempt_consumable` 新增 `_sealed_evidence_problems`，逐项核验
  `report`/`summary` 的保存 SHA 与 `artifacts` 路径/摘要；complete 但无报告、报告被改写或
  数组工件漂移均不可消费。隔离检查把报告 `complete_evidence` 改为 false 而保留原封口摘要后，
  `stage_state` 返回 `report_drifted`；恢复原字节后重新可消费，篡改工件后返回
  `artifact_drifted`。
- **P1 F 身份门阻断**：F worker 的 `worker_runtime_identity` 与候选预检同列硬门，任一非空
  即在 `from_spec`/fixture/真实读取/构造/重放之前终止；`from_spec` 现在解析并加载
  acceptance、核对 `acceptance_sha256`，使 `materialize` 的身份检查在 worker 内同样生效；
  `finally` 仍按实际计数收口。隔离反例中故意漂移身份后 worker 返回 2、`from_spec` 从未
  被调用、所有计数为 0。
- **P2 线程数身份**：`RUNTIME_IDENTITY_KEYS` 增加 `cpu_threads`（本次校准/全链均为 20）；
  父进程、合成接受、真实准入、R0、F worker 与 P/I 物化均核对该值，单线程环境漂移不再因
  其他字段相同而放行。
- **P2 payload dtype 契约**：`load_state_dict(strict=True)` 前新增逐张量键集、dtype、shape
  与有限性检查；float64 权重（含 `1.0000000001` 被静默舍入为 `1.0` 的形态）、坏形状、
  缺失键与非有限值均在复制前拒绝，`strict=True` 仍保留。隔离的小型 CPU 模块反例覆盖
  `model_dtype/model_shape/model_nonfinite/model_missing`。
- **P2 文档订正**：v0.9 冻结校准 evidence 为 `aa4bd83167ea11a7c1cb159e494269b7d843f8233bbbc272558caf9c59e0a9ed`
  （fp32 canonical `6d7a24d0…`）；attempt_007 为 `8fc90e39…`（fp32 `47390b08…`，v0.7 新增
  runtime identity）；attempt_008 为 `28bb6625…`（fp32 `a9513ca3…`，v0.8 新增
  `cpu_threads`）。三者数值门/校准数值逐项相同，但 `calibration.fp32_contract_sha256`
  变化使 evidence 摘要变化，不能表述为“与 v0.9 相同”。

**测试（最终字节）**

- `tests/test_w1_real_training_update_provider.py`：**36 passed**（新增 5 项：FK→场景截止、
  payload dtype/shape/finite、资产闭包复验、`from_spec` acceptance、worker 身份先于真实构造；
  另有线程数身份断言）。
- `tests/test_w1_real_training_update_audit.py`：**56 passed**（新增 1 项封口摘要/工件反例）。
- 六个相关回归文件：**263 passed**。
- 合并运行（2+6 文件）：**355 passed（379.25s）**；全项目套件未重跑。

**Gate B 最终产物（attempt_008）**

- 校准 `derived/audit/w1_real_training_update_audit/calibration/attempt_008/`：
  `tolerance_contract.json` SHA `6af1d60529d83a0d01b75d24bad1c871718e8b3b29541de3ef55537381d481a5`、
  evidence `28bb66259d12dd224bef34064cef2419efd02aed847175a487abdd403237bfc6`、
  fp32 canonical `a9513ca311d530c8554844a474080aefd1359e300b1f0790c7aa256756f44f3e`；
  `calibration_verification.json` 文件 SHA
  `44c20b42e0d556b76fb5e65685366999de52d065e0c12e355fd3e19593dfdfef`、canonical
  `54091ea62f230a48687baba5224f722009f452342144d31c753660f6c5fb46e4`；
  `recompute_problems=[]`、`four_x_ok=true`，最大 4× 比值 `delta_same_input=0.925540`
  （其余 `alpha_x_t=0.867157`、`delta_cross_path=0.864409`、`state_cross_path=0.579252`、
  `parameter_gradient=0.361710`、`theta_cross_path=0.300759`、`output_cotangent=0.068296`、
  `loss_scalar=0.032639`、`global_norm_clip=0.002895`、`independent_float64=0.0`）。
  阈值/规则/零底与 v0.9 逐项相同；runtime identity 记录 Python 3.12.13、Torch
  2.14.0+cu126、NumPy 2.5.3、SciPy 1.18.1、CUDA 12.6、RTX 3090、MuJoCo 3.13.0、
  `cpu_threads=20`、AdamW 源码 `6f4a96e7…`/`39c5eaa0…`。
- 完整合成链 `derived/audit/w1_real_training_update_audit/synthetic/attempt_008/`：
  `audit_result.json` SHA `1e31b28e7ef194c4bf8a2dde3cdc8080536f6a7fea889f8c7fa365cef51d6627`、
  final `cda2647b2ba0042976acd11640aa23db861fe74c1a86ca190a7c21a36221ceb7`、
  registration `7b3d95a75987475464f1e3da6b9f1982d10499e267dcaeed3f65656eb0ae49a5`；
  四 stage done/sealed、consumer problems 均空、264 tensor/3864 行、调用矩阵精确
  （8 loads、64/64 forwards、48 draw units、32/32 VJP、64 production FK、16/16 updates、
  15840/115200 reference FK、64 micro、0 synthetic scene）、`training_updates=16`；结论
  **`audit_incomplete`**（`complete_evidence=true`、`failures=[]`、`problems=[]`、
  **582** gap：392 precision + 190 sensitivity + 0 bound-unverifiable）。F 新进程
  pid 1832099/ppid 1829291（pid≠ppid、ppid=父）、exit `ok`、`verification_problems=[]`；
  成本 R0 1.1751/P 13.5609/I 58.2882/F 121.0190 = **194.0433s**，各 1/2 次；工件
  7,404,757,557 bytes（约 6.896 GiB）。`synthetic_acceptance.json` SHA
  `f3976183f288900fb74b9eb2ad840a86d5066f5b7b3e8a32e9529521009a8224`、
  `passed=true/problems=[]`、`run_sha256=eec4bad18b50aa6cc4452992f2296fb197c7db56256f7785263f90b3756f26a2`；
  阶段文件数 R0 3/P 5/I 5/F 12、fixture 6、run 3，消费校验 `[]`。
- 真实 metadata-only admission
  `derived/audit/w1_real_training_update_audit/admission_real/attempt_008/admission_package.json`
  SHA **`8a16c0599822538d07deba5001cb074812a196e7b76c4aaeecf34042d28048c0`**（452,490 bytes），
  `passed=true/problems=[]`、validator `[]`，绑定 acceptance `f3976183…`、校准 canonical
  `54091ea6…`、evidence `28bb6625…`、fp32 `a9513ca3…`；实现闭包 **25 文件**且与当前树逐位
  一致（plan `f82c89a8…`、audit `6166f675…`、provider `1e4b79fd…`、provider tests
  `21a6519a…`、audit tests `a1e12ea8…`）；父锚 10/10、checkpoint 2/2、资产闭包
  `checked=true`/37 引用/problems 空；read scope
  `npz_values_read/checkpoint_payload_read/torch_load_calls/model_constructor_calls/
  muJoCo_compile_calls/conditioner_forward/model_forward/vjp_calls/optimizer_steps=0`，
  `asset_closure_verified=true`、`cohort_created=false/registered_attempt_created=false`；
  `runtime_identity.cpu_threads=20` 且与校准记录逐项一致；状态保持
  `authorized=false/registered=false/run=not_run/section_10_2_checked=false`。
- 被取代证据：`synthetic/attempt_007` 与 `admission_real/attempt_007` 在本轮实现/plan 字节
  更新后被 attempt_008 取代并保留为历史。为满足 R0 的 20 GiB 磁盘门，本轮曾将已冻结且被
  取代的 `synthetic/final` 与 `synthetic/attempt_006` 临时移至
  `/dev/shm/w1_quarantine_20261031/`，Gate B 完成后原样移回；首次 attempt_008 因余量
  18.9 GiB < 20 GiB 在 R0 失败（`r0_disk_free`，无阶段计算），该临时输出与本轮 pytest
  临时目录（约 5.2 GiB）已删除后重跑。其余历史 attempt 未删除；恢复后磁盘余量约
  **3.45 GiB**。
- 边界不变：未读取真实 checkpoint payload/父 NPZ 数值、未编译真实资产、未构造真实模型/
  优化器、未执行真实前向/反向/FK/AdamW、未创建真实 cohort/授权/注册；Gate C/D/E 仍未运行。
  `objective_coverage=partial`、`full_objective_contract_passed=null`、
  `factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、
  `production_integration=false`。v0.9 方案/校准/合成/准入字节未改写。

### 10.8 v0.9 复审修复与 Gate B 首次冻结（2026-10-31；已被 §10.9 取代）

独立审阅确认 **4 P1 + 2 P2**；全部按反例修复后，Gate B 以最终字节重新执行。本节保留
attempt_009 的历史事实；当前权威为 §10.9（同一 v0.9 计划，P2-2 选择字段撤回补充后的
attempt_010）。§10.7（v0.8/attempt_008）、§10.6（v0.7/attempt_007）及更早 attempt 仅作
历史保留。

**修复项（均有新增反例）**

- **P1 封口必需文件与报告引用闭包**：`_sealed_evidence_problems` 新增
  `_required_stage_files` 与 `_report_reference_problems`。R0 要求
  `registration.json`/`admission.json`/`call_reservations.json`（real 另加
  `noise_schedule.npz`），P 要求 `p_report.json`/`p_rows.json`/两个模型数组/
  `call_reservations.json`（real 另加 `input_admission.json`），I 要求对应五件，F 要求
  final/`worker_spec.json`/`worker/worker_result.json`/`call_reservations.json`；同时核验
  报告中的 registration/noise、模型 `arrays_path`/`arrays_sha256`、P `input_admission`
  路径与 SHA。隔离反例：改写 R0 `registration.json`/`noise_schedule.npz`、P
  `input_admission.json` 或模型数组后，`stage_state` 返回
  `artifact_drifted`/`*_reference_drifted`，不再进入 `r0_noise_from`/I 构造。
- **P1 Trainer 内部构造分项守卫**：`construction_observer` 在 model、conditioner、
  optimizer 完成后分别执行同一 deadline+guard。受控时钟令 conditioner 后过期时，
  optimizer 不再构造，已完成 model/conditioner 计数各为 1、optimizer 为 0。
- **P1 参考后端第二次编译守卫**：`G1Spec()` 的 `__post_init__` 编译与随后
  `load_plain()` 编译之间新增 `reference_backend_recompile` 检查，第二次编译后再检查
  `reference_backend_attach` 才分配 `MjData`。受控反例中第一次编译返回时过期，第二次
  编译不再启动；正常路径仍为单数据实例。
- **P1 F 来源门阻断**：噪声来源 SHA、R0 回放一致性、P `input_admission` SHA 与 `passed`
  收集完成后，在 `run_p_stage`/真实物化/构造/重放前以 `f_source_precheck_failed` 终止；
  `finally` 保留实际计数（反例中仅 8 个 `cpu_draw_units`，其余 0），三项错误
  `schedule_source_sha`/`p_input_admission_sha`/`p_input_admission_not_passed` 均上报。
- **P2 场景完成即计数**：`largebox_static_frame` 新增可选 `observer`，在真实
  `mj_forward` 成功返回后立即调用；provider 传入计数回调并把静态帧显式交给
  `LossAssembler`。模拟 assembler 后续设备张量 OOM 时 `scene_setup_mj_forward=1`、
  半成品不缓存，且不会二次调用场景缓存。
- **P2 最终消费核验计费**：新增 `_charge_final_verification`；F 封口后的 P/I/F 消费核验
  （重新哈希大工件）实际耗时超过账面成本时，以 `cost_floor` 重封口；越界则转
  incomplete 并返回 `post_seal_verification_budget_exceeded`，外层撤回完整选择和完成声明。
  本次全链核验墙钟 **17.4570s** 已重封口计入 F 成本，账面 `measured_wall_s=165.8239s`；
  受控 100s/102.8s 反例不再保留 complete/selection。

**测试（最终字节）**

- `tests/test_w1_real_training_update_provider.py`：**39 passed**（新增第二次编译守卫、
  Trainer 分项守卫、场景 forward 计数；扩展 F 来源反例）。
- `tests/test_w1_real_training_update_audit.py`：**59 passed**（新增封口必需文件/报告引用、
  R0 noise、P input_admission 与最终核验计费反例）。
- 六个相关回归文件：**263 passed**；另跑 `tests/test_geometry.py` **8 passed**、
  `tests/test_omniretarget_w1_review.py` **40 passed**。
- 合并运行（2+6 文件）：**361 passed（386.36s）**；全项目套件未重跑。

**Gate B 最终产物（attempt_009）**

- 校准 `derived/audit/w1_real_training_update_audit/calibration/attempt_009/`：
  `tolerance_contract.json` SHA
  `47d529736be79740f62492fbce53704d7164bb356d243b7fdf8b9d813f99f229`、evidence
  `28bb66259d12dd224bef34064cef2419efd02aed847175a487abdd403237bfc6`、fp32 canonical
  `a9513ca311d530c8554844a474080aefd1359e300b1f0790c7aa256756f44f3e`；
  `calibration_verification.json` 文件 SHA
  `a1b8f9ca926c94927b8a553f631d6c535ccdd5d38d2fbd867cbcb38ee2ef0090`、canonical
  `0d3bcf4be163454ace81eeed65fc6fbc8b1a9fc00cab5317108723f033fbb842`；
  `recompute_problems=[]`、`four_x_ok=true`，最大 4× 比值 `delta_same_input=0.925540`
  （`alpha_x_t=0.867157`、`delta_cross_path=0.864409`、`state_cross_path=0.579252`），
  阈值/规则/零底与 v0.9 逐项相同；`cpu_threads=20`。
- 完整合成链 `derived/audit/w1_real_training_update_audit/synthetic/attempt_009/`：
  `audit_result.json` SHA `4046b1bb9a8440ec57932a4c3e6f1b9aa17de04bb5f634b67fbc0c91897dfed7`、
  final `9129eab81673ddb3ef8def98652ec091f351a47d9852a57f49c3af6916d9d030`、
  registration `c5fc0afef591fecb80a4341515e3975ae125f8f428ab9fe4cb3949f7f466e527`；
  四 stage done/sealed、consumer problems 均空、264 tensor/3864 行、调用矩阵精确
  （8 loads、64/64 forwards、48 draw units、32/32 VJP、64 production FK、16/16 updates、
  15840/115200 reference FK、64 micro、0 synthetic scene）、`training_updates=16`；结论
  **`audit_incomplete`**（`complete_evidence=true`、`failures=[]`、`problems=[]`、
  **582** gap：392 precision + 190 sensitivity + 0 bound-unverifiable）。F 新进程
  pid 3505207/ppid 3501398（pid≠ppid、ppid=父）、exit `ok`、`verification_problems=[]`；
  成本 R0 1.1774/P 13.2543/I 57.8275/F 165.8239 = **238.0830s**，各 1/2 次；工件
  7,404,777,964 bytes（约 6.896 GiB）。`synthetic_acceptance.json` SHA
  `45393f5c36b8100cfd875c11f456b33d07822397e519edc678ea6b2a5b6a8dd9`、
  `passed=true/problems=[]`、`run_sha256=93af1d3a731d2afed16c5a0992bdbd676696e049137785c4f19df833e2999824`；
  阶段文件数 R0 3/P 5/I 5/F 12、fixture 6、run 3，消费校验 `[]`。
- 真实 metadata-only admission
  `derived/audit/w1_real_training_update_audit/admission_real/attempt_009/admission_package.json`
  SHA **`3e063f66f6a0a82230ecce1d7707c67d5d5ee86cab3919de20d8767e587db4f0`**（452,490 bytes），
  `passed=true/problems=[]`、validator `[]`，绑定 acceptance `45393f5c…`、校准 canonical
  `0d3bcf4b…`、evidence `28bb6625…`、fp32 `a9513ca3…`；实现闭包 **25 文件**且与当前树
  逐位一致（plan `29451f6b…`、audit `084a3d35…`、provider `e088c1e3…`、
  distance_adapter `bd1981d4…`、provider tests `9eb2cd98…`、audit tests `7696d76a…`）；
  父锚 10/10、checkpoint 2/2、资产闭包 `checked=true`/37 引用/problems 空；read scope
  `npz_values_read/checkpoint_payload_read/torch_load_calls/model_constructor_calls/
  muJoCo_compile_calls/conditioner_forward/model_forward/vjp_calls/optimizer_steps=0`，
  `asset_closure_verified=true`、`cohort_created=false/registered_attempt_created=false`；
  `runtime_identity.cpu_threads=20` 且与校准记录逐项一致；状态保持
  `authorized=false/registered=false/run=not_run/section_10_2_checked=false`。
- 被取代证据：`synthetic/attempt_008` 与 `admission_real/attempt_008` 在本轮实现/plan
  字节更新后被 attempt_009 取代并保留为历史。为运行全链测试与 Gate B，本轮曾将已冻结且
  被取代的 `synthetic/final` 与 `synthetic/attempt_006` 临时移至 `/dev/shm`，完成后原样
  移回；未删除任何历史 attempt。当前余盘约 **150.96 GiB**，满足 R0 的 20 GiB 门。
- 边界不变：未读取真实 checkpoint payload/父 NPZ 数值、未编译真实资产、未构造真实模型/
  优化器、未执行真实前向/反向/FK/AdamW、未创建真实 cohort/授权/注册；Gate C/D/E 仍未运行。
  `objective_coverage=partial`、`full_objective_contract_passed=null`、
  `factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、
  `production_integration=false`。v0.9 方案/校准/合成/准入字节未改写。

### 10.9 v0.9 P2-2 选择字段撤回补充与最终 Gate B 冻结（2026-10-31；已被 §10.10 取代）

§10.8 的 P2-2 已使越界核验把 row 转为 incomplete 并使返回值 `selected_attempts=None`；
补充审阅发现磁盘 final summary 仍保留 `selected_attempts`/`selected_chain_counts`/
`selected_chain_training_updates`。现 `_amend_final_report` 在撤回完成声明时同步把三项
置 null，matrix-veto 反例新增对应断言。该补充只修改审计器的一处撤回字段与测试；其余
4 P1+2 P2 修复、容差、调用矩阵与零底不变。本节保留 attempt_010 的历史事实；当前权威为
§10.10（v0.10/attempt_011）。

**测试（最终字节）**

- provider **39 passed**、审计 **59 passed**、六个相关文件 **263 passed**；合并
  **361 passed（386.67s）**；`tests/test_geometry.py` **8 passed**、
  `tests/test_omniretarget_w1_review.py` **40 passed**。

**Gate B 最终产物（attempt_010）**

- 校准 `derived/audit/w1_real_training_update_audit/calibration/attempt_010/`：
  `tolerance_contract.json` SHA
  `c6b8ab1e48913ec8584da11485a4c29c7511c4967acdf84d1ee7b29a2f4dfd6b`、evidence
  `28bb66259d12dd224bef34064cef2419efd02aed847175a487abdd403237bfc6`、fp32 canonical
  `a9513ca311d530c8554844a474080aefd1359e300b1f0790c7aa256756f44f3e`；
  `calibration_verification.json` 文件 SHA
  `0615eaa95a9826f271eca2161bf84bc55f4438ce9f8420119c93b7c2752ccdaa`、canonical
  `afeb494b63964a94824234ed4c5af7679b9cf189129d4af19ea429e2d7573e54`；
  `recompute_problems=[]`、`four_x_ok=true`，最大 4× `delta_same_input=0.925540`，阈值/
  规则/零底与 v0.9 逐项相同；`cpu_threads=20`。
- 完整合成链 `derived/audit/w1_real_training_update_audit/synthetic/attempt_010/`：
  `audit_result.json` SHA `fd4634ccf7b8780f2d8b91c05136e949859d969c0a3907a70fbcd08374a3a127`、
  final `a14905df0cb028c0a4f12244fcc66f642203ec7b45a24cee5edf81babceebdcc`、
  registration `0bd4124c657c8133627ce0f06af8deafaeb7fd6b01ecd802ba87f65ee8c85f92`；
  四 stage done/sealed、consumer problems 均空、264 tensor/3864 行、调用矩阵精确
  （8 loads、64/64 forwards、48 draw units、32/32 VJP、64 production FK、16/16 updates、
  15840/115200 reference FK、64 micro、0 synthetic scene）、`training_updates=16`；结论
  **`audit_incomplete`**（`complete_evidence=true`、`failures=[]`、`problems=[]`、
  **582** gap：392 precision + 190 sensitivity + 0 bound-unverifiable）。F 新进程
  pid 3562581/ppid 3558707、exit `ok`、`verification_problems=[]`；成本
  R0 1.1858/P 13.6011/I 58.7056/F 164.6997 = **238.1923s**；最终消费核验墙钟
  **17.3142s** 已由 cost-floor 重封口计入 F，账面 `measured_wall_s=164.6997s`；工件
  7,404,777,952 bytes。`synthetic_acceptance.json` SHA
  `c070b8df94d131e3df9620609e5413f4c54da56617ea4918b3b7b6d9cbfd1fab`、
  `passed=true/problems=[]`、`run_sha256=a30c233b85a1d0303cdf1bb5b7bf439a79b3bcc8d357c55b700e1bd0a54a87ee`；
  阶段文件数 R0 3/P 5/I 5/F 12、fixture 6、run 3，消费校验 `[]`。
- 真实 metadata-only admission
  `derived/audit/w1_real_training_update_audit/admission_real/attempt_010/admission_package.json`
  SHA **`d4d9fe5a923246584f3121ba1da022a478033cd55a131eb8ec38e73840344968`**（452,490 bytes），
  `passed=true/problems=[]`、validator `[]`，绑定 acceptance `c070b8df…`、校准 canonical
  `afeb494b…`、evidence `28bb6625…`、fp32 `a9513ca3…`；实现闭包 **25 文件**且与当前树
  逐位一致（plan `48b81fb2…`、audit `1d3fe858…`、provider `e088c1e3…`、
  distance_adapter `bd1981d4…`、provider tests `9eb2cd98…`、audit tests `bc11a068…`）；
  父锚 10/10、checkpoint 2/2、资产闭包 `checked=true`/37 引用/problems 空；read scope
  `npz_values_read/checkpoint_payload_read/torch_load_calls/model_constructor_calls/
  muJoCo_compile_calls/conditioner_forward/model_forward/vjp_calls/optimizer_steps=0`，
  `asset_closure_verified=true`、`cohort_created=false/registered_attempt_created=false`；
  `runtime_identity.cpu_threads=20` 且与校准记录逐项一致；状态保持
  `authorized=false/registered=false/run=not_run/section_10_2_checked=false`。
- 被取代证据：`synthetic/attempt_009` 与 `admission_real/attempt_009` 在撤回字段补充后
  被 attempt_010 取代并保留为历史；更早 attempt 未删除。当前余盘约 **146.08 GiB**，
  满足 R0 的 20 GiB 门。
- 边界不变：未读取真实 checkpoint payload/父 NPZ 数值、未编译真实资产、未构造真实模型/
  优化器、未执行真实前向/反向/FK/AdamW、未创建真实 cohort/授权/注册；Gate C/D/E 仍未运行。
  `objective_coverage=partial`、`full_objective_contract_passed=null`、
  `factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、
  `production_integration=false`。v0.9 方案/校准/合成/准入字节未改写。

### 10.10 v0.10 场景内部守卫与阶段消费计费修复（2026-10-31；已被 §10.11 取代）

独立审阅确认 **1 P1 + 1 P2**；全部按反例修复后，Gate B 以最终字节重新执行。本节保留
attempt_011 的历史事实；当前权威为 §10.11（v0.11/attempt_012），§10.8–§10.9
（v0.9/attempt_009/010）及更早 attempt 仅作历史保留。

**修复项（均有新增反例）**

- **P1 场景构造内部守卫**：`largebox_static_frame` 新增 `check` 回调，在 `G1Spec()`
  （含 plain 编译）、`load_scene()` 第二次编译、`MjData` 分配和 `mj_forward` 前分别执行
  同一 deadline+guard；provider 传入 `production_scene_plain_spec/compile/data/forward`
  包装。受控反例中首次 plain 编译返回即过期：`scene_compile` 检查立即失败，
  `load_scene`/`MjData`/`forward` 均未启动、`scene_setup_mj_forward` 不增加；另一反例在
  `scene_forward` 前失败时 forward 未执行。
- **P2 阶段间消费核验计费**：新增 `_charge_stage_consumption`，在 R0/P/I 封口后的
  `stage_state` 消费核验完成后，以实际耗时 `cost_floor` 重封口；若超过该阶段预算，撤回
  完成声明与报告 complete 字段并返回 `post_consumption_verification_budget_exceeded`，
  调用方在进入下游阶段前停止。P 预算 900s、封口 90s、核验 1000s 的受控反例中 row 转
  incomplete、报告 `complete_evidence=false`，不再启动 I。
- **测试**：provider **40 passed**（新增场景 helper 内部守卫反例）、审计 **60 passed**
  （新增阶段消费核验计费反例）、六个相关文件 **263 passed**；合并 **363 passed
  （383.70s）**；另 `test_geometry` **8 passed**、`test_omniretarget_w1_review`
  **40 passed**。

**Gate B 最终产物（attempt_011）**

- 校准 `derived/audit/w1_real_training_update_audit/calibration/attempt_011/`：
  `tolerance_contract.json` SHA
  `657c381330f473543f4561787ed00a654f5167ff394a4a5587b82d04c740dc8c`、evidence
  `28bb66259d12dd224bef34064cef2419efd02aed847175a487abdd403237bfc6`、fp32 canonical
  `a9513ca311d530c8554844a474080aefd1359e300b1f0790c7aa256756f44f3e`；
  `calibration_verification.json` 文件 SHA
  `83ae286ad50250ca08984003a1912de896b8837f3f8ef77c6fc3337ace63cda6`、canonical
  `d882d2931270982ad1c0ccfd11d2ea88c9eef30ff24ce3dc9e41e7dd64a3ae91`；
  `recompute_problems=[]`、`four_x_ok=true`，最大 4× `delta_same_input=0.925540`，阈值/
  规则/零底与 v0.9 逐项相同；`cpu_threads=20`。
- 完整合成链 `derived/audit/w1_real_training_update_audit/synthetic/attempt_011/`：
  `audit_result.json` SHA `18f79c48d02154a4517fb3e4e6d772b21af361f307e6a12efd5985a0f2e504d0`、
  final `e08307be88a90f5aa6856c71ed14b63ca0bb66b90c93df9a30c59a7e80587a48`、
  registration `f191f6279c78f54ca81eb992532a3d7d4b1d7931ec982d5a8d76397c64ba2cac`；
  四 stage done/sealed、consumer problems 均空、264 tensor/3864 行、调用矩阵精确
  （8 loads、64/64 forwards、48 draw units、32/32 VJP、64 production FK、16/16 updates、
  15840/115200 reference FK、64 micro、0 synthetic scene）、`training_updates=16`；结论
  **`audit_incomplete`**（`complete_evidence=true`、`failures=[]`、`problems=[]`、
  **582** gap：392 precision + 190 sensitivity + 0 bound-unverifiable）。F 新进程
  pid 3819064/ppid 3815196、exit `ok`、`verification_problems=[]`；成本
  R0 1.1763/P 17.7981/I 71.3107/F 164.7421 = **255.0271s**；阶段消费核验墙钟
  R0 0.0196/P 4.4059/I 12.8267/F 17.3649s，P/I/F 均按 `cost_floor` 重封口计入对应成本；
  工件 7,404,795,448 bytes。`synthetic_acceptance.json` SHA
  `f379c3a8be1d5a81349092037585230c2cb65b3d4ef48a421a2709acf95ecf29`、
  `passed=true/problems=[]`、`run_sha256=832c7adebc29437c4327f985e1f46d5e06cc8f20cf232930fc71f23509162547`；
  阶段文件数 R0 3/P 5/I 5/F 12、fixture 6、run 3，消费校验 `[]`。
- 真实 metadata-only admission
  `derived/audit/w1_real_training_update_audit/admission_real/attempt_011/admission_package.json`
  SHA **`3a30b8d61fdd2fa9989a2eb28ec40e14b8bec6a8f0e264d1adc9564453047b05`**（452,490 bytes），
  `passed=true/problems=[]`、validator `[]`，绑定 acceptance `f379c3a8…`、校准 canonical
  `d882d293…`、evidence `28bb6625…`、fp32 `a9513ca3…`；实现闭包 **25 文件**且与当前树
  逐位一致（plan `5f82ebde…`、audit `66c02a4a…`、provider `65a5549e…`、
  distance_adapter `efcc80b4…`、provider tests `e3999870…`、audit tests `5d7ac3e1…`）；
  父锚 10/10、checkpoint 2/2、资产闭包 `checked=true`/37 引用/problems 空；read scope
  `npz_values_read/checkpoint_payload_read/torch_load_calls/model_constructor_calls/
  muJoCo_compile_calls/conditioner_forward/model_forward/vjp_calls/optimizer_steps=0`，
  `asset_closure_verified=true`、`cohort_created=false/registered_attempt_created=false`；
  `runtime_identity.cpu_threads=20` 且与校准记录逐项一致；状态保持
  `authorized=false/registered=false/run=not_run/section_10_2_checked=false`。
- 被取代证据：`synthetic/attempt_010` 与 `admission_real/attempt_010` 在场景内部守卫与
  阶段消费计费修复后被 attempt_011 取代并保留为历史；更早 attempt 未删除。当前余盘约
  **171.11 GiB**，满足 R0 的 20 GiB 门。
- 边界不变：未读取真实 checkpoint payload/父 NPZ 数值、未编译真实资产、未构造真实模型/
  优化器、未执行真实前向/反向/FK/AdamW、未创建真实 cohort/授权/注册；Gate C/D/E 仍未运行。
  `objective_coverage=partial`、`full_objective_contract_passed=null`、
  `factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、
  `production_integration=false`。v0.9 方案/校准/合成/准入字节未改写。

### 10.11 v0.11 阶段间噪声解码计费修复（2026-11-02）

独立审阅确认 **1 P2**；按反例修复后，Gate B 以最终字节重新执行（`calibration/attempt_012`、
`synthetic/attempt_012`、`admission_real/attempt_012`）。本节为当前权威；§10.10
（v0.10/attempt_011）及更早 attempt 仅作历史保留。

**修复项（均有新增反例）**

- **P2 噪声解码移入已登记阶段**：原 `r0_noise_from` 在 R0 消费核验之后、P 的 t0 与
  running/reservation 登记之前直接解码 `noise_schedule.npz`。受控反例中 R0 已记账
  90s、解码耗时 1000s 后 P 才以 1090s 为起点登记，1000s 完全未入账。现 R0 只向后续
  阶段传递封口工件中的路径与摘要：`frozen_noise_source` 从 sealed `artifacts` 读取
  路径与摘要，不在阶段间重新哈希或解码；P/I 各自在 `run_p_stage`/`run_i_stage` 内、
  `materialize` 前调用 `_decode_r0_noise`，先按封口摘要核验文件，再由阶段同一 `check`
  （deadline + unit admission）在解码前、解码后分别守卫，解码耗时进入该阶段
  `finish_attempt` 的 `measured_wall_s` 与 `_charge_stage_consumption`。P 解码越界时
  以实际耗时封口 incomplete，I 不再启动；F 继续在其 worker 进程内解码并由 F 阶段计费。
  阶段间不传递解码后的噪声对象。
- **测试**：审计 **64 passed**（新增 P/I 解码先于 `materialize` 且同一 guard 在
  `r0_noise_decoded` 拒绝后 `materialize` 未执行；摘要漂移在解码前拒绝；封口摘要只取自
  sealed `artifacts`、不重新哈希文件），provider **40 passed**、六个相关文件
  **263 passed**，合并 **367 passed（393.93s）**；另 `test_geometry` **8 passed**、
  `test_omniretarget_w1_review` **40 passed**。

**Gate B 最终产物（attempt_012）**

- 校准 `derived/audit/w1_real_training_update_audit/calibration/attempt_012/`：
  `tolerance_contract.json` SHA
  `f77824e5609ad120702132e58872984d84ac48e7eb9088152d9d6ff08c8707d4`、evidence
  `28bb66259d12dd224bef34064cef2419efd02aed847175a487abdd403237bfc6`、fp32 canonical
  `a9513ca311d530c8554844a474080aefd1359e300b1f0790c7aa256756f44f3e`；
  `calibration_verification.json` 文件 SHA
  `49a359ec3d82daed686eacec5da81a1ce522148686ce14979afc9a0f52bfbd90`、canonical
  `d2ea697783ae475507c1d16d63fde46fd0b25b3d48a3190cd2cf10ec7ebcf7a7`；
  `recompute_problems=[]`、`four_x_ok=true`，最大 4× `delta_same_input=0.925540`，阈值/
  规则/零底与 v0.9 逐项相同；`cpu_threads=20`。
- 完整合成链 `derived/audit/w1_real_training_update_audit/synthetic/attempt_012/`：
  `audit_result.json` SHA `3ded65b1d18ee71660bc2940110d7d39d026aa591f6e57ac39e64bd11ed7e73d`、
  final `1fb8075c8ceb58139e88a3fb80fc866bb47c725065bcbe35335355af64786515`、
  registration `358f7d0c205354599a2e25656e013199bad92cfb01cd3a919f542ce05c50d3bf`；
  四 stage done/sealed、consumer problems 均空、264 tensor/3864 行、调用矩阵精确
  （8 loads、64/64 forwards、48 draw units、32/32 VJP、64 production FK、16/16 updates、
  15840/115200 reference FK、64 micro、0 synthetic scene）、`training_updates=16`；结论
  **`audit_incomplete`**（`complete_evidence=true`、`failures=[]`、`problems=[]`、
  **582** gap：392 precision + 190 sensitivity + 0 bound-unverifiable）。F 新进程
  pid 3959223/ppid 3955521、exit `ok`、`verification_problems=[]`；成本
  R0 1.1719/P 17.3034/I 67.7256/F 158.0184 = **244.2193s**；阶段消费核验墙钟
  R0 0.0067/P 4.2455/I 12.4193/F 16.7836s，P/I/F 均按 `cost_floor` 重封口计入对应成本；
  工件 7,404,795,428 bytes。P/I 的 NPZ 与 rows 与 attempt_011 逐位相同，证明合成数值
  路径未因本修复变化。`synthetic_acceptance.json` SHA
  `58dc92455eaf8f6fa266a833a7b63462d42ad0e554efd9aa1bc1c1d0368d7a4a`、
  `passed=true/problems=[]`、`run_sha256=cd890107e3e30b93bfc7599f42c6d8dc05d1bb46580e6a86b52f1012cccda3f6`；
  阶段文件数 R0 3/P 5/I 5/F 12、fixture 6、run 3，消费校验 `[]`。
- 真实 metadata-only admission
  `derived/audit/w1_real_training_update_audit/admission_real/attempt_012/admission_package.json`
  SHA **`8d4ec09e6f51015f3985b0539da9d7e9b962042141f67ddc53660edc012292ab`**（452,490 bytes），
  `passed=true/problems=[]`、validator `[]`，绑定 acceptance `58dc9245…`、校准 canonical
  `d2ea6977…`、evidence `28bb6625…`、fp32 `a9513ca3…`；实现闭包 **25 文件**且与当前树
  逐位一致（plan `e66cd8a9…`、audit `91f228e2…`、provider `65a5549e…`、
  distance_adapter `efcc80b4…`、provider tests `e3999870…`、audit tests `5bda56a0…`）；
  父锚 10/10、checkpoint 2/2、资产闭包 `checked=true`/37 引用/problems 空；read scope
  全零；`runtime_identity.cpu_threads=20` 且与校准记录逐项一致；状态保持
  `authorized=false/registered=false/run=not_run/section_10_2_checked=false`。
- 被取代证据：`synthetic/attempt_011` 与 `admission_real/attempt_011` 在噪声解码计费修复
  后被 attempt_012 取代并保留为历史；更早 attempt 未删除。当前余盘约 **164.18 GiB**，
  满足 R0 的 20 GiB 门。
- 边界不变：未读取真实 checkpoint payload/父 NPZ 数值、未编译真实资产、未构造真实模型/
  优化器、未执行真实前向/反向/FK/AdamW、未创建真实 cohort/授权/注册；Gate C/D/E 仍未运行。
  `objective_coverage=partial`、`full_objective_contract_passed=null`、
  `factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、
  `production_integration=false`。v0.9 方案/校准/合成/准入字节未改写。

## 11. 用户明确授权后的首次真实执行（2026-11-02）

用户确认“同意草案十项，授权按固定预算执行一次真实 R0→P→I→F（3240 秒、16 次诊断更新），不自动重试。”。新的 §10.2 [正式授权](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/authorization/attempt_012/authorization.json) SHA **`199bfb30f11502aaf57acf38f1c49525ac58e511f870f230f223cb12b8525d78`**，十项全 true，绑定 v0.11 / attempt_012 和授权草案 `66d1261d…`；`authorization_problems=[]`。未改动冻结预注册、执行计划、25 文件实现清单或 Gate B 工件。

首次运行根为 `derived/experiments/omniretarget_first/20261102_2704709_w1_real_training_update_audit`。启动时钟在 uv/import 前记录并传入原 CLI，首次入口仅一次，无 `--resume` 或自动重试。Gate D 的 R0 `attempt_001` done/complete=true，metadata admission passed/problems=[]，R0 的 checkpoint payload/数组数值/模型/编译为 0，CPU draw=8；R0 registration `6abab029…`。

Gate E 的 P `attempt_001` 在登记后完成四窗物化、input_admission passed（`b1dbefcf…`）、两份 checkpoint strict load、两套模型/conditioner/optimizer 构造及 deterministic 的两次原生更新；随后在 diffusion trainer 返回前触发 **`production diffusion alpha schedule differs`**。扩散首次 forward 未启动，P 封为 incomplete/complete=false；I/F 均未创建 attempt，未重试。结论 **`audit_incomplete`、exit 3、complete_evidence=false、covered_contract_passed=false**；全链 16 次未完成，实际 training_updates=2。

两份 reservation 均已结算，index/ledger 的 status/complete/测量成本/调用余额一致；逐类核对 `after-before=consumed`，未超预留/caps，没有本地 running/publishing 或 consumed=null。登记成本 R0 **58.599905234s**、P **8.502827990s**，合计 **67.102733224/3240s**（含保守 seal-tail）；外层实测 66.205312315s，单列不替换账本。

实际累计：checkpoint/model/conditioner/native optimizer 构造各2，conditioner/backbone/loss assembler/parameter backward/micro 各8，生产 FK batch16，clip/native AdamW/diagnostic update各2，scene setup1，CPU draw8（全部归 R0），native add_noise/独立 VJP/显式参考更新/参考 mj_forward+jacSite 均0。当前完整选择与 selected-chain 计数未确认；两次更新未经过 I/F，不能声明参数链路通过。

P 的 p_report/p_rows/p_diffusion 缺失，已写出的 p_deterministic 部分 NPZ 未列入 P 封口 artifacts，保留且不可消费。P 提前返回没有生成 audit_result.json 或 F final，未事后补造。异常与返回事实见[已哈希输出](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/authorization/attempt_012/execution_stdout.log)（`24c3c31a…`）及[退出记录](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/authorization/attempt_012/execution_exit.json)；[只读执行观察](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/authorization/attempt_012/execution_observation.json) SHA **`54ca9d2ad155fdac77c1d4d60acd3037cfb8ec289af3d31f492ebd72ab1394e4`** 标明并非正式 stage report，保存原运行10文件/共466,350,106 bytes 的摘要与完整调用/成本事实。

执行后只读 JSON/源码/文件字节摘要，没有重载真实 checkpoint/NPZ 值、重建模型或追加数值探针；未重跑 pytest、校准或合成链。25 个实现文件和6份主要冻结材料摘要不变。下一步先纯合成定位 alpha 求值语义并重新冻结受影响 Gate B；真实恢复必须处理旧 cohort 的实现/授权绑定，保留已消耗的调用、attempt 与成本，不能更名重获额度。本次一次运行已结束，不自动续跑。

`objective_coverage=partial/full_objective_contract_passed=null/factor_selected=false/auto_execute=false/final_interaction_method=not_ready/production_integration=false` 与旧质量结论保持。详见[首次真实执行报告](/home/c112/codes/bill/ireset/docs/verification/omniretarget_w1_real_training_update_audit_report.md)。

## 12. alpha 有效常数修正与 Gate B attempt_013（2026-11-03）

本轮依据[补充计划 v0.12](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_real_training_update_alpha_fix_plan.md)，SHA **`9b49e673d975fdd1fd61ea80e7926618d328750af6054b312827ca97fac646a0`**，仅修复既有预注册 §3.2 的有效常数实现，并重新冻结 Gate B。原预注册 v0.9、执行计划 v0.11、attempt_012 校准/合成/准入、首次真实根及授权目录保留原字节；未真实重试或续跑。

### 12.1 复现与实现

固定原配置、Diffusers 0.40.0、Torch 2.14.0+cu126 的纯合成 scheduler 探针确认：原生 alpha 与“float64 完整计算后末尾转 float32”有 **767/1000** 个位置不同，最大 **11 ULP**、绝对差 `1.7881393432617188e-7`。首个不同坐标 t=2：原生 `0.9996920824050903`，旧参考 `0.9996920228004456`。独立 CPU float32 `linspace → square → 1-betas → cumprod` 与原生结果逐位一致。

诊断位于 `derived/audit/w1_real_training_update_audit/alpha_schedule_fix/attempt_001/`：`alpha_schedule_diagnosis.json`、`alpha_schedule_arrays.npz`、`preserved_files.json`。探针仅构造 scheduler，checkpoint payload、父数组数值、模型/优化器、forward/update 均为0；它不是对真实失败 attempt 的补算或补造报告。

- `effective_alpha_reference(config)` 从配置独立重建生产有效 binary32 常数后提升到 float64；`alpha_cumprod_reference` 的父 raw float64 来源门保持。fixture、P/I/F 的独立 x_t 和真实 input admission 统一使用新 helper，生产 scheduler 与生产输入不改。
- `verify_production_alpha` 在真实/合成 provider 以及 fixture 的 diffusion 首次 forward 前核对 float32 dtype、shape 和严格逐位值；1 ULP、旧末尾舍入结果、错 schedule、dtype、shape、NaN 均拒绝。真实构造保留 deadline/unit guard。
- input admission 分别冻结 raw/effective alpha 摘要和 `torch_cpu_float32_scaled_linear_v1` 协议；I/F 将有效摘要与协议也列为消费硬门。
- runtime identity 增加 Diffusers 版本和 DDPM/DDIM 两个实际导入源码 SHA；各 metadata 消费端仍仅核对版本/字节，不执行数值探针。`cpu_threads=20`。
- 实现闭包 **26 文件**：两份审计源码、两份测试发生本轮改动，新增 v0.12 补充计划；旧 v0.11 与 v0.9 仍在清单中，`EXECUTION_PLAN` 指向补充计划，`PREREG_DRAFT` 保持原路径。audit `006e942b…`、provider `d08ba01d…`、audit tests `980bc098…`、provider tests `bf934a6b…`。

### 12.2 回归与数值校准

修复前两个新增合成用例按预期失败，分别复现原生 scheduler 被误拒、input admission 的有效 alpha 摘要错误；保存 `baseline_alpha_regression.log`。修复后定向 **16 passed（7.44s）**；最终字节合并执行 provider **45** + audit **72** + 六个既有相关文件 **263** = **380 passed（373.02s）**，较旧版新增13个用例。六文件为 `test_omniretarget_w1`、`test_w1_de_numerics`、`test_w1_training_objective_audit`、`test_w1_real_objective_audit`、`test_w1_real_admission`、`test_w1_real_fk_audit`。全项目套件未重跑。命令和日志摘要见同目录 `regression_summary.json`。

校准目录：`derived/audit/w1_real_training_update_audit/calibration/attempt_013/`。

| 项目 | SHA-256 |
| --- | --- |
| tolerance contract 文件 | `d8515d2f51f966da7324e8c0f385d9b12d034fcbe79fc160d82b6939bd65b649` |
| calibration evidence | `b848cbb4e54553dd19daf69a471076ddfc3bd2c6030c55608413d746d0c096b2` |
| fp32 canonical | `574d9a0a2115e3619aa260f3eef4ef83c8ff85e377e2f23eebc078434a0d441e` |
| adamw_fp32_contract.json 文件 | `2c4ddba84d43a2264eca94b603b400af8be764de97352e3cf933e569822c44b2` |
| calibration_verification.json 文件 | `ac66dd8b15fa60885b8f6488316f3aed469584b78318197d5eb833d06426bd63` |
| verification canonical | `49ffc21ba57283c1614b9070dbd07560a32065fc71b5878625b6c1519771cbeb` |

实际重算 `recompute_problems=[]`、`four_x_ok=true`。阈值、规则、零底与 attempt_012 逐项相同；仅 `alpha_x_t` 观测变化：比值 `0.21678926013164576 → 0.04576622897692079`，4× `0.867157040526583 → 0.18306491590768317`。其余门的观测逐项相同，最大4×仍为 `delta_same_input=0.9255398370845472`，`delta_cross_path=0.8644089322616761`、`state_cross_path=0.5792524288194545`。fp32 数值探针、节点图、host constants 与旧版相同，canonical 变化仅来自新增 runtime identity；没有放宽阈值。

### 12.3 完整合成链与新旧对比

目录：`derived/audit/w1_real_training_update_audit/synthetic/attempt_013/`。

| 工件 | SHA-256 |
| --- | --- |
| audit_result.json | `10dc7f364a090d58e870ee52087998dcd47a7bf4650f98cee719b752170ce4a8` |
| final | `0ba1babef620b4a726dabb40cfbb0f109727cdeff22c862a4a38f1729349427d` |
| R0 registration | `34d9a769f2c537708e51109f7c1b6cdaf5ff2a49d29e1ed89d71233309facf39` |
| synthetic_acceptance.json | `8f655f41184234a61536f5bfb058609538d7d5381fff05b7b3775576838686d5` |

四阶段均 `attempt_001`、done/sealed、消费 problems 为空；**264 tensor/3864 行**齐全。选中链/阶段/cohort 调用精确相等，`training_updates=16`，零计数保持。结论 **`audit_incomplete` / exit 3**：`complete_evidence=true`、`covered_contract_passed=false`、`failures=[]`、`problems=[]`，**582** gap（392 `parameter_delta_precision_gap` + 190 `cross_path_sensitivity_gap` + 0 bound-unverifiable）。F 独立进程 pid **1749699** / ppid **1746039**、exit0/`ok`，`verification_problems=[]`。

| 阶段 | 已封口成本（秒） | 消费核验墙钟（秒） |
| --- | ---: | ---: |
| R0 | 1.190880829 | 0.007949325 |
| P | 17.504695744 | 4.262634910 |
| I | 68.774587591 | 12.406381461 |
| F | 160.340700071 | 16.772381485 |
| 合计 | **247.810864235** | — |

P/I/F 消费核验已按 `cost_floor` 重封口入账。工件 **7,404,799,802 bytes**。acceptance `passed=true/problems=[]`、消费 validator `[]`；阶段文件数 R0/P/I/F = **3/5/5/12**，fixture6、run3，`run_sha256=f827d86c3c5ea78eb03645e1f2bd36863d2b3b08212a93f5814b788911f45135`。

[逐项比较证据](/home/c112/codes/bill/ireset/derived/audit/w1_real_training_update_audit/alpha_schedule_fix/attempt_001/gate_b_comparison.json) SHA **`7893d74b0417c8c1d2a527e75d3cadca3ada013ec8bf544c47a05a05dc77516b`**，`passed=true`：

- P/I 及 F-P/F-I 两模型共 **8 个 NPZ 与 attempt_012 字节完全相同**，覆盖原生 x_t、prediction、梯度、参数及优化器数组；四份 rows 的内容也全部相同。gap 集合、行级数值判定、调用与覆盖边界不变。
- 去除运行路径差异后，P 和 F-P 报告各仅8个 `x_t_max_ratio` 变化，I/F-I 报告无其他变化。fixture 的8份独立 x_t 数组变化，逐份按新有效常数重推相等；raw alpha、t/epsilon、windows、两份合成权重和 roster 不变。
- 阈值/规则/零底逐项相同；其余校准数值证据相同。**62 个受保护文件全部保持原字节**，包括旧方案、Gate B attempt_012、首次真实10文件、原授权目录和生产 generator。保护清单 SHA `03eba72eb291cfb6b3c2b8108752ad8fb3cb68b99633a6218734c3a705a84d53`。

### 12.4 真实 metadata-only admission 与后续边界

CLI 输出新包为 `derived/audit/w1_real_training_update_audit/admission_real/attempt_013/attempt_001/admission_package.json`（`--out` 指定 attempt_013 容器，写包器按包 attempt 追加）。SHA **`53de1b1e03d2928803a6140b10342d642e87f95d85fd52dd0d863f174e4073b2`**、**455,723 bytes**，`passed=true/problems=[]`、validator `[]`。父锚10/10、checkpoint2/2、资产37引用及26个实现文件均匹配，runtime identity 与校准/接受一致。

scope 为 `metadata_only`：记录13次文件字节摘要和36键 NPZ 头部；checkpoint payload/NPZ values/torch.load/模型构造/MuJoCo 编译/forward/VJP/optimizer steps 均为0，`cohort_created=false`、`registered_attempt_created=false`，资产闭包 `asset_closure_verified=true`。新包状态 `authorized=false/registered=false/run=not_run/section_10_2_checked=false`。

本轮未读取真实 checkpoint payload 或父 NPZ 数值、未编译真实资产、未执行真实 forward/backward/FK/更新，未创建新真实授权/cohort/注册；未提交。首次真实历史仍为 **2 次更新、67.10273322396097 秒**，原 R0/P attempt、reservation、未封口部分数组与账本全部保留。Gate B 的16次是合成更新，不增加真实额度。

旧授权绑定旧实现且“一次执行、不自动重试”已执行完毕；现有 resume 仍要求旧 cohort 的实现及授权身份不变。本轮未修改此硬门。后续先制定保留原消耗的身份/授权绑定迁移方案，完成复审并取得针对新绑定的明确授权；不得更名重置消耗，不直接用旧授权运行新实现。

`objective_coverage=partial/full_objective_contract_passed=null/factor_selected=false/auto_execute=false/final_interaction_method=not_ready/production_integration=false` 与旧质量结论保持。

## 13. 既有 cohort 绑定迁移方案制定（2026-11-03，仅文档）

本节保留[绑定迁移与有限恢复方案 v0.13](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_real_training_update_binding_migration_plan.md)初稿制定时点：初稿SHA **`721a585707e9c904e6e85ff3db7961b3023410d395851ed26c6582f3f66545ad`**，不是当前方案文件摘要；当前修订见§13.1。初稿状态为 `plan_only/not_implemented/real_resume_not_authorized`，当轮仅制定方案、核对JSON/源码/文件字节摘要并同步文档入口；没有新增迁移执行记录。

方案决定沿用原根和cohort身份，追加第二代绑定，按新R0 attempt_002→P attempt_002→I attempt_001→F attempt_001恢复。旧marker、两条终态行及原attempt工件保留；未来index/ledger可追加新行，旧行canonical内容不变。新执行序号解决R0_001→P_001后追加R0_002的累计前态排序问题，按代次唯一选链解决两个done R0共存，完整历史仍参与预算和调用核验。F预检/父结算/封口终检继承同一身份与余额契约。

原真实根仍10文件/466,350,106 bytes、R0 done与P incomplete各一次、两份reservation已结算、I/F未创建；实际消耗仍为2次更新和67.10273322396097秒。原3240秒上限余 **3172.897266776039秒**；R0余181.40009476602427秒、P余891.4971720100148秒、I/F余900/1200秒，R0/P各只剩原第二次attempt。计划将迁移复核与发布计入新R0；完整恢复最多新增16次、累计最多18次，而非重领16次或3240秒额度。

v0.13列出原始摘要锚点、单向身份依赖、schema与事件提交、未知/中断处理、24类调用矩阵、读取/计费守卫、反例、实施M0–M5及另行明确授权后的单次M6。当前26文件实现闭包、`EXECUTION_PLAN`与v0.12冻结材料不变；实施后必须重新回归并冻结新校准/普通及迁移合成接受/metadata准入，attempt_013不能直接替代新实现证据。

文档核验通过：`git diff --check` 无问题；24类矩阵与原观察/nominal/caps逐项一致，阶段/总余量按原账本重算一致，18处新增或修改的本地链接有效。26个关键文件（原3份冻结方案、真实根10文件、原授权目录9文件、当前Gate B的4份关键材料）与本轮事前SHA/bytes一致，26文件实现闭包也与attempt_013候选清单一致，mismatches均为空。改动仅为新方案及AGENTS/README/本验证记录。

本轮仅做文档验证，不重跑pytest、校准或合成数值链；**380 passed仍是§12既有结果**。未读取真实checkpoint payload/父NPZ数值、未构造真实模型/优化器或编译资产，未新增真实授权/cohort/注册，未运行真实`--real`/`--resume`。所有partial/null/false/not_ready/不接入及旧质量结论保持。

### 13.1 修订1：2 P1+2 P2方案修复（2026-11-03，仅文档）

本节保留[v0.13 修订1方案](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_real_training_update_binding_migration_plan.md)制定时点：修订1文件SHA **`28dc72745d5b85ef56a136a4d4541d33350e5c74c2b6ba4ed112ab79a60de405`**、45,389 bytes是历史摘要，不代表当前文件；当前修订见§13.2。该时点状态为`plan_only/not_implemented/real_resume_not_authorized`。

| 审阅问题 | 本轮方案修复 |
| --- | --- |
| P1 消费核验与不可变commit循环 | §6.2定义内部准备态核验；全量核验/选择、cost_floor及有界重封口稳定后冻结行摘要，再写唯一finish/commit。公开消费要求commit、stage_closed、terminal及无撤销，执行中仅允许已验证持锁上下文；成功闭合前完成对应尾部检查。提交后异常追加revoke/成本负债，不改已提交行；撤销写入失败仍因缺闭合链拒绝 |
| P1 登记前成本负债不可发现 | §6.1把唯一权威日志固定为原根launch_journal，uv/import及实质预检前持久化claim；即使原根仍只有旧v1账本，所有历史/准入/预算读取器也扫描。换授权目录或invocation不能隐藏负债，按invocation/stage/seq关联已结算成本去重；未结清即unknown |
| P2 旧v1缺首个前驱头 | §4.1精确定义legacy_head_payload、canonical字节及seq=3专用承接。独立从两条旧行、两份reservation和原文件摘要重算头为`b499a6ce7e5ef8fac06821cc063f27bd109374d9673791551fe92ad56302a424`；seq≥4必须接前一commit，不为旧seq=1/2补造提交 |
| P2 未结算小计冒充总更新数 | §11将调用/时间known分别报告；缺完整结算时training_updates/cohort_training_updates/cohort_call_counts及selected字段为null，单列已证明下界与unsettled_attempts。旧2+新P4+新I4而F快照缺失时下界10、实际总数未知；F部分执行但完整结算6次时精确总数16，仍不宣称链完成 |

§8补入对应的未来合成反例，M0/M1/M5、F身份传递和新版授权绑定同步上述要求。**这些是后续实施与测试要求，本轮未实现迁移代码或执行这些反例。**

文档核验通过：`git diff --check`及新方案空白检查无问题，本地链接/源码行号有效；24类矩阵与原观察、attempt_013 nominal及原caps逐项一致，四阶段与总预算按原账本重算一致，legacy_head独立重算一致。26个受保护关键文件的SHA/bytes与本轮事前基线全部相同；当前26文件实现闭包与attempt_013候选清单相同。原真实根仍10文件、466,350,106 bytes，无launch_journal或迁移事件；旧消耗仍为2次更新/67.10273322396097秒，剩余3172.897266776039秒。

改动仅限本方案及AGENTS/README/本验证记录；未运行pytest、校准或数值链，**380 passed仍是§12历史结果**。未解码真实checkpoint/父NPZ、未构造模型/优化器或编译资产，未新增真实授权/cohort/注册，未运行真实`--real`/`--resume`。当前实现/证据权威仍v0.12/attempt_013，后续按M0–M5准备，再按M6要求取得绑定最终字节的新明确授权；partial/null/false/not_ready/不接入及旧质量结论保持。

### 13.2 修订2：成功收据写后耗时与失败闭合（2026-11-03，仅文档）

[v0.13 修订2方案](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_real_training_update_binding_migration_plan.md)当前文件SHA **`1b27a41462c4d40fb3e0f54b240c3ac6669e4871062191d470dc734721142b17`**，53,647 bytes。状态仍为`plan_only/not_implemented/real_resume_not_authorized`。

本轮修复复审的1项P1：原流程只有成功收据写前检查，不能证明其自身write/fsync耗时被尾部计费覆盖。§6.2将stage_closed/terminal改为候选，写入、文件/目录fsync及回读返回后，实际耗时必须同时不超过已计费floor、严格小于阶段remaining_s，身份/摘要也一致，才允许持锁上下文放行；R0慢写后不得提前启动P。

§6.3要求先在内存中否决并阻断下游，再追加不可覆盖revoke/failure及已知未覆盖成本下界；同一阶段的重复时间观测取最大覆盖区间，不重复收费，不改写已提交行/finish/commit。撤销记录无法写入也不能继续。公开闭合还须有外层实际wait/exec返回来源及覆盖证明，不能凭done候选、锁释放或无revoke推断成功；证明缺失/失败即成本unknown，公开selected/complete按有效状态撤回。

新增return_observation仅为未来接口，本轮未创建该记录。方案明确执行窗口包含候选收据发布、fsync、回读和清理，截止于外层实际返回；执行后观察存档仅复制既有返回事实、单列墙钟。该边界及证明门必须绑定到新spec/授权，不能借存档开展项目核验、模型/资产读取或更新。已有v1历史不补造证明、不重新计费。

§8补充“尾部1秒、stage_closed实际5秒”应阻断P且未覆盖下界至少4秒、独立阶段预算越界、terminal慢写、最后清理延迟、撤销写入失败、缺/假返回来源及正常audit_incomplete/exit3闭合的未来反例；M1/M2、F传递、接受闭包与授权要求同步。**这些是实施验收要求，本轮未实现或执行这些反例，未运行pytest/校准/数值链，380 passed仍为§12既有结果。**

文档检查通过：`git diff --check`、新方案空白、本地链接及源码行号核验无问题；24类矩阵与原观察/attempt_013 nominal/原caps逐项相同，四阶段预算及总余量重算一致，legacy_head仍为`b499a6ce…`。26个关键文件SHA/bytes与事前保护清单一致，当前26文件实现闭包与attempt_013候选清单一致。真实根仍10文件/466,350,106 bytes，无launch_journal或迁移/返回观察记录；原消耗2次更新/67.10273322396097秒、剩余3172.897266776039秒保持。

改动仅限方案、AGENTS、docs/README与本验证记录。未解码真实checkpoint/父NPZ、未构造模型/优化器或编译资产，未新增真实授权/cohort/注册，未运行真实`--real`/`--resume`；当前实现证据仍v0.12/attempt_013，partial/null/false/not_ready/不接入与旧质量结论保持。

### 13.3 M0/M1 开始实现（2026-11-03，未真实续跑）

按 v0.13 修订2 开始实现，详细结果见[绑定迁移实现验证记录](omniretarget_w1_real_training_update_binding_migration_verification.md)。M0 已在原根之外生成 history snapshot（`28a542e6…`）和 protected manifest（`a033e856…`），独立重算 legacy head 为 `b499a6ce…`；原真实根仍 10 文件/466,350,106 bytes，未写入 launch journal/binding/attempt。

M1 已实现：规范 JSON/重复键拒绝、legacy head 与 source copy 校验、执行序号/前驱头重推、generation 显式选链、准备态内部核验、一次性 finish/commit、提交后禁改写、固定 launch journal 扫描、stage_closed/terminal 写后实际耗时门、外层 return observation、v2 授权/迁移准入/草案 CLI。普通 resume 遇到 v2 授权明确拒绝，不会自动升级旧身份。

本轮新增 9 项迁移合成反例（含 seq/legacy head/缺返回观察/慢写 revoke/授权额度重置）；`test_w1_real_training_update_audit.py`+迁移测试 **81 passed（116.10s）**、provider **45 passed（149.10s）**。全项目套件未重跑。M3 新 Gate B、M4 校准/普通+迁移双合成接受、M5 最终迁移准入与新授权草案、M6 真实 R0_002→P_002→I_001→F_001 仍未执行；本轮不创建真实授权/注册/返回观察，不改变原 2 次更新/67.10273322396097 秒与 3172.897266776039 秒剩余。`objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false` 及旧质量结论保持。

### 13.4 绑定迁移 M3–M5 完成与最终字节 Gate B（2026-11-04，未真实续跑）

按[绑定迁移方案 v0.13 修订2](/home/c112/codes/bill/ireset/docs/plans/omniretarget_w1_real_training_update_binding_migration_plan.md)完成 M3–M5，详细事实见[绑定迁移实现验证记录 §6](omniretarget_w1_real_training_update_binding_migration_verification.md)。

- **M3**：`IMPLEMENTATION_FILES` 28 项（原 26 + 本方案 + 迁移测试），`EXECUTION_PLAN` 指向本方案，`PREREG_DRAFT` 保持 v0.9。最终字节定向测试 **141 passed（359.87s）**、M3 回归集 **380 passed（469.08s）**；全项目套件未重跑。
- **M4**：校准 `calibration/attempt_015` 通过（contract `634f2d0f…`、evidence `b848cbb4…`、fp32 `574d9a0a…`、verification `4cef73aa…`，`recompute_problems=0/four_x_ok=true`，阈值/规则/零底与 attempt_013 逐项相同）。普通合成 `synthetic/attempt_015` 为 `audit_incomplete`/16 updates/**582 gap**，acceptance `9eb4ce4c…` passed。迁移合成 `binding_migration_synthetic/attempt_002` 为旧失败前缀 + 新链 **cohort 18**（selected 16），acceptance `65ced169…` passed。Gate B v2 `binding_migration_gate_b/attempt_002/synthetic_acceptance.json` SHA **`e937c27adf1cf6cddba69f123b781ef19290a25e9bb83d4bbcc07414daf38c66`**，passed/problems 空；消费端重算通过，绑定两侧 child acceptance、候选收据、commit、外层 return observation 与独立重算 legacy head。两条链的 P/I/F **各 12/12 文件与 attempt_013 逐字节相同**（8 NPZ + 4 rows），未放宽数值门。
- **M5**：在最终文档树上重新生成只读材料——real metadata admission `admission_real/attempt_016/attempt_001/admission_package.json`（passed/metadata_only）、history snapshot `binding_migration/attempt_003/history_snapshot.json`（legacy head `b499a6ce…`、cost 67.10273322396097s、1 个未封口旧文件）、protected manifest v2（94 entries、drift 0）、binding spec `binding_migration_spec/attempt_003/binding_spec.json`、migration admission `binding_migration_admission/attempt_002/migration_admission.json`（passed）、v2 授权草案 `binding_migration_authorization/attempt_002/authorization_draft.json`（`authorized=false/template_only=true`、10 项 checklist 均 null）与独立复核 `m5_verification.json`（passed/problems 空）。精确 SHA 与绑定值见复核 JSON，不再写入文档以免 git 工作区身份再次漂移。原真实根仍 **10 文件 / 466,350,106 bytes**，无 launch journal、无新注册/授权/数值更新；`scan_launch_journals` known/0 records，`verify_history` 无问题。
- **边界**：未读取真实 checkpoint payload/父 NPZ 数值/资产编译，未构造真实模型/优化器，未执行真实 forward/backward/FK/AdamW，未创建真实授权/cohort/注册或真实 R0_002→P_002→I_001→F_001。旧 attempt_013/012 与首次真实记录保留原字节；`calibration/attempt_014` 等中间尝试保留但被最终字节 attempt_015/attempt_002 取代。M6 仍需绑定最终字节的新明确授权；`partial/null/false/not_ready/不接入` 与旧质量结论保持。
