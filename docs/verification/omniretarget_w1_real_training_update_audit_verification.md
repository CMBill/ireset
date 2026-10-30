# W1 真实网络参数梯度与优化更新链路审计 —— I/C 工程与纯合成验收记录

日期：2026-10-29。方案：[omniretarget_w1_real_training_update_audit_prereg_draft.md](../plans/omniretarget_w1_real_training_update_audit_prereg_draft.md)
（v0.8）。状态：**I/C 实现、第六轮审阅修复与纯合成验收完成；真实 provider/授权/注册/真实 R0→P→I→F 未实施**。

本记录只覆盖合成证据与冻结准入材料。没有读取真实 checkpoint 张量、真实 NPZ 数值、数据集或 G1 资产，
没有执行真实网络前向/反向、真实 FK、真实参数更新或生产 checkpoint 写入。

## 1. 交付文件

| 文件 | 作用 |
| --- | --- |
| `src/ireset/eval/w1_real_training_update_audit.py` | 审计器：roster/行目录、合成 fixture、P/I 捕获、独立参考、有效 float32 AdamW、跨路径传播、R0/P/I/F、预算/封口、校准、准入、真实门 |
| `scripts/audit_w1_real_training_update.py` | CLI：`--synthetic`、`--calibrate-synthetic`、`--admission`、`--real`（门后拒绝） |
| `tests/test_w1_real_training_update_audit.py` | 54 项纯合成反例/接口测试（v0.7 为 51 项，v0.8 新增 3 项第六轮反例） |
| `src/ireset/train/trainer.py` | 抽出共享 `predict` / `begin_update` / `accumulate_micro` / `clip_gradients` / `step_optimizer`，新增只读 observer、审计注入 `loss_fn`/`build_optimizer` 与 `clip_coefficient`；默认路径不变 |

共享单元由原生 `Trainer.run` 与审计共同调用。`observer=None` 与 `observer={}` 的
theta/m/v/step 逐位相等，loss/parts 与全局 RNG 不变；NaN 输入的异常类型/原因一致
（`tests` 覆盖 deterministic/diffusion）。

## 2. 冻结校准与有效 float32 契约

目录：`derived/audit/w1_real_training_update_audit/`。v0.4–v0.7 的旧校准保存在
`history/v0.4_5d9230cf/`、`history/v0.5_9a964372/`、`history/v0.6_582ea60a/` 与
`history/v0.7_3bce1fd5/`，以下为 v0.8 当前文件。

| 工件 | SHA-256 |
| --- | --- |
| `tolerance_contract.json` | `271df4e3ee0bf84cc6e92e5c436cac3d5f0ef08cb19addbb787cf3543bc339ca` |
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
- 参考 FK 截止单元（v0.8）：`run_i_stage` 在 `ReferenceContext.evaluate` 前检查
  `i_ref_fk_{u}_{m}`；`ReferenceContext` 在目标窗构造、evaluate、FK 与 Jacobian 调用前分别执行
  同一 `check`（截止 + `unit_admission`）。受控时钟在首个 forward 返回后过期时，不再新增
  480 次 `reference_mj_forward`/3600 次 `reference_mj_jac_site`；FK 若已开始则完成，Jacobian
  不启动。输出 VJP/clip/参考更新前的检查与“执行后计数”保持不变。
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

## 3. 权威纯合成全链 R0→P→I→F（v0.8）

运行根：`derived/audit/w1_real_training_update_audit/synthetic/v0.8_final/`。

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
  v0.7 的 185 个 sensitivity gap 在 v0.8 增加 5 个：可辨跨路径更新量门从基于已舍入实测的
  `1e-3` 恢复为原提案 `5e-4`（更严格）并改用未舍入 I 参考；392 个 precision gap 不变。
  v0.4 的 660/340/2 与 F 合计 1002 条是旧 F 对 I 行 gap 与 I 报告 gap 的重复计数；旧工件保留不改写。
- 失败/缺口原因：step 1 的 `m/√v≈0.1` 更新约 `1e-5`，而 |theta|~0.1–1 的 float32 写入分辨率为
  `1e-8–6e-8`，大量坐标低于 §6.2 的 5e-4 可辨门；其余为跨路径近相消敏感性。区间输入预算改为
  梯度门允许集合并逐节点向外 R32 舍入后，0 个坐标落在可信区间外。
- F：独立 Python 子进程（pid 1902814、ppid 1900096，pid≠ppid；exit 0），继承父绝对截止并在
  读取/构造/micro/backward/VJP/clip/update 前重查父 running 登记；重建 schedule、重读磁盘、
  逐位重放 P/I；保存报告逐字段（loss/parts/gnorm/coef/RNG digest/arrays hash/state presence）、
  两文件 rows 与报告内嵌 rows、两侧行多重集/数组 schema/原始量分类、cotangent 三组事实全部复核，
  `verification_problems=[]`，worker problems=[]。
- 成本：R0 **1.1789s**、P **13.7790s**、I **58.2535s**、F **87.0078s**，合计 **160.2192s**；
  每阶段 1/2 次、无借额（R0 240/P 900/I 900/F 1200s）。
- 工件：实际 **6.895 GiB**（7,404,305,871 bytes）；逐类上界单链 **7.145 GiB**、cohort（每阶段
  2 次 attempt）**14.236 GiB≤16 GiB**。
- 关键 SHA：`audit_result.json` `49b3791bd313edea0d15fc2ab44b20015d48a2b543785a27f3fc98de7f0a878c`、
  `finalize/attempt_001/training_update_final.json` `08dea9a2d1e46f6b5403c98ea2044ce9fb95fcee0568e971368e48c69117293c`、
  `registration/attempt_001/registration.json` `97405a3c68c7b0200c8408bfaa22006953d545255a6a3cd916e57ff7a18c8f4c`；
  R0 registration 绑定方案 v0.8 `c3909a22ad77d913b3f2d71651b06fa0cee0de62292b8d698bbf12ac9d1789f4`。

## 4. 准入包（v0.8）

路径：`derived/audit/w1_real_training_update_audit/admission/attempt_010/admission_package.json`，
SHA-256 **`d8ccda86c3f8ce742ef8c7878dc8b1b3248c5b5afc026a3caed1e53767b09803`**。

- `passed=true`、`problems=[]`；`status={authorized:false, registered:false, run:not_run,
  section_10_2_checked:false}`。
- 绑定 tolerance contract 文件 SHA `271df4e3…`、evidence `aa4bd831…`、fp32 canonical
  `6d7a24d0…`、roster `264`/3864 行、两份父运行 7+3 个文件哈希、两 checkpoint 与 train config。
- `read_scope`：metadata-only；`window_values_loaded=false`、`checkpoint_torch_load=false`、
  `cohort_created=false`、`registered_attempt_created=false`、conditioner/model/VJP/optimizer 调用 0；
  只构造了 CPU 随机初始化 roster（2 个 model constructor）。
- 实现文件哈希与 v0.8 权威运行 R0 registration 完全一致（方案 SHA `c3909a22…`、
  审计模块 `8957c70d…`、测试 `1e3da1c4…`）；消费侧 `admission_package_problems` 返回空。
- 资源上界 `one_chain_gib=7.145`、`estimated_artifacts_gib=14.236`、`within_cap=true`。
- 授权模板十项 checklist 仍全为 `null`；真实模式在授权/准入/调用余额校验后仍拒绝
  （`real_provider_not_registered_in_ic`）。

## 5. 测试

- 新增模块：`uv run --locked pytest tests/test_w1_real_training_update_audit.py -q` → **54 passed（97.38s）**。
  v0.8 新增 3 项第六轮反例：跨路径更新量用未舍入 I 参考（评审二步例：已舍入 0.994504 错误通过、
  未舍入 1.001830 为 sensitivity gap，`small_unobservable_update` 仍 incomplete）；I 在参考 FK 前
  检查 `i_ref_fk_1_0` 且过期时 `reference_mj_forward` 仅含构造期 240、jacSite=0、output VJP=0；
  `ReferenceContext` 在 FK 后过期时 Jacobian 不启动，reference FK=480、jacSite=0。
- 相关回归（v0.8 重跑）：`tests/test_omniretarget_w1.py`、`tests/test_w1_de_numerics.py`、
  `tests/test_w1_training_objective_audit.py`、`tests/test_w1_real_objective_audit.py`、
  `tests/test_w1_real_admission.py`、`tests/test_w1_real_fk_audit.py` 合计 **263 passed（152.44s）**。
- 全项目套件本轮未重跑；不把历史全量计数当作本轮验收。

## 6. 复现命令

```bash
uv run --locked pytest tests/test_w1_real_training_update_audit.py -q
uv run --locked python scripts/audit_w1_real_training_update.py --calibrate-synthetic
uv run --locked python scripts/audit_w1_real_training_update.py --synthetic \
  --tolerance-contract derived/audit/w1_real_training_update_audit/tolerance_contract.json \
  --out derived/audit/w1_real_training_update_audit/synthetic/v0.8_final
uv run --locked python scripts/audit_w1_real_training_update.py --admission
```

## 7. 第六轮审阅修复映射（2 P1）

| # | 缺口 | 修复与反例 |
| --- | --- | --- |
| P1-1 | 跨路径更新量仍使用已舍入的 I 增量 | `_cross_path_flags`/`max_delta_ratio` 的 close 门改用 P 实测 vs I 未舍入 `delta_ref64`；存在性比较 P/I 参考。评审二步例已舍入 0.994504（错误 passed）→未舍入 1.001830（`cross_path_sensitivity_gap`）；校准 `delta_cross_path` 在恢复的 5e-4 下 0.216102（4×=0.8644），v0.7 的 1e-3 撤回；`small_unobservable_update` 仍 incomplete |
| P1-2 | I 在参考 FK 前缺截止检查 | `run_i_stage` 在 `ReferenceContext.evaluate` 前检查 `i_ref_fk_{u}_{m}`；`ReferenceContext` 在目标窗/evaluate/FK/Jacobian 前执行同一截止+`unit_admission` 守卫。受控时钟首 forward 后过期不再新增 480/3600 reference FK/jacSite，Jacobian 不启动；before-evaluate 检查时 reference FK 仅 240、output VJP=0 |

v0.7 的第五轮 3 P1 修复与 51 项测试保留在历史（`synthetic/history/v0.7_final_plan_df2a1a2b/`、
`admission/attempt_009/`），当前权威为 v0.8。

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
- 下一步：复阅 v0.8/校准/准入包，另行实施真实 provider 并取得方案明确的真实授权；在此之前不得创建
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
- 当前权威为方案 `c3909a22…`、`synthetic/v0.8_final/` 与 `admission/attempt_010/`（SHA `d8ccda86…`）。
