# W1 真实网络参数梯度与优化更新链路审计 —— I/C 工程与纯合成验收记录

日期：2026-10-29。方案：[omniretarget_w1_real_training_update_audit_prereg_draft.md](../plans/omniretarget_w1_real_training_update_audit_prereg_draft.md)
（v0.5）。状态：**I/C 实现、第三轮审阅修复与纯合成验收完成；真实 provider/授权/注册/真实 R0→P→I→F 未实施**。

本记录只覆盖合成证据与冻结准入材料。没有读取真实 checkpoint 张量、真实 NPZ 数值、数据集或 G1 资产，
没有执行真实网络前向/反向、真实 FK、真实参数更新或生产 checkpoint 写入。

## 1. 交付文件

| 文件 | 作用 |
| --- | --- |
| `src/ireset/eval/w1_real_training_update_audit.py` | 审计器：roster/行目录、合成 fixture、P/I 捕获、独立参考、有效 float32 AdamW、跨路径传播、R0/P/I/F、预算/封口、校准、准入、真实门 |
| `scripts/audit_w1_real_training_update.py` | CLI：`--synthetic`、`--calibrate-synthetic`、`--admission`、`--real`（门后拒绝） |
| `tests/test_w1_real_training_update_audit.py` | 39 项纯合成反例/接口测试（v0.4 为 25 项，v0.5 新增 14 项审阅反例） |
| `src/ireset/train/trainer.py` | 抽出共享 `predict` / `begin_update` / `accumulate_micro` / `clip_gradients` / `step_optimizer`，新增只读 observer、审计注入 `loss_fn`/`build_optimizer` 与 `clip_coefficient`；默认路径不变 |

共享单元由原生 `Trainer.run` 与审计共同调用。`observer=None` 与 `observer={}` 的
theta/m/v/step 逐位相等，loss/parts 与全局 RNG 不变；NaN 输入的异常类型/原因一致
（`tests` 覆盖 deterministic/diffusion）。

## 2. 冻结校准与有效 float32 契约

目录：`derived/audit/w1_real_training_update_audit/`。v0.4 的旧校准保存在
`history/v0.4_5d9230cf/`，以下为 v0.5 当前文件。

| 工件 | SHA-256 |
| --- | --- |
| `tolerance_contract.json` | `9a9643728efecd32394a25df8b58923ef3bc9ff50d91e4a53423e554f422dc05` |
| 校准 evidence（contract 内） | `85246d758590a921947bc64e9f862c08d17b0af7410489a435e4bb79318cef6a` |
| `adamw_fp32_contract.json`（规范化内容哈希） | `6d7a24d0cd077454120dfbaaad37fcc378b10572e4a33a26df8ca09f7e0003e1` |

`tolerance_contract_problems(recompute=True)` 在新进程重跑校准并逐项通过；`4×observed` 全部 ≤ 1。

观测比值（`max_ratio_of_proposal`）：

| 键 | 观测 |
| --- | ---: |
| `alpha_x_t` | 0.2168 |
| `loss_scalar` | 0.0082 |
| `output_cotangent` | 0.0171 |
| `parameter_gradient` | 0.0904 |
| `global_norm_clip` | 0.0007 |
| `theta_cross_path` | 0.0116 |
| `state_cross_path` | 0.0400 |
| `delta_cross_path` | 0.0745 |
| `delta_same_input` | 0.1471 |

后四项取全链通过行与标量近例的最大值（`source=synthetic_chain+scalar_near_1e-7`），不再用标量覆盖
全链观测；`max_state_ratio` 按 `rtol=5e-4` 归一化，不再把未除 rtol 的 2e-5 当作 0.04 的比值。

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

## 3. 权威纯合成全链 R0→P→I→F（v0.5）

运行根：`derived/audit/w1_real_training_update_audit/synthetic/v0.5_final/`。

- 配置/规模：full `omniretarget_motion` 网络（d_model 240、4 layers、6 heads、time_window 31）；
  roster **130 deterministic + 134 diffusion = 264** 个 tensor，行目录 **3864 = 14N+168**。
- 四窗 `sub10_largebox_063/085` 两个 demo 的重叠窗；每模型两个连续更新、各 4 个 B=4 micro；
  16 个诊断 micro/路径；CPU generator seed `20261008`。
- 四 stage 全部 `done`/sealed，索引与台账一一对应；无未结算预留。
- 调用矩阵精确等于方案 §7.2 全链名义值：checkpoint loads 8、model/conditioner constructions 8/8、
  native optimizer 4、conditioner/backbone forwards 64/64、add_noise 32、draw units 48、
  production loss 32、production FK **64**、parameter backward 32、output/total VJP 32/32、
  native clip 8、independent clip 8、native AdamW 8、explicit updates 8、diagnostic updates 16、
  anchor restores 4、formula checks 8、reference FK 15840、jacSite 115200、micro units 64；
  `scene_setup_mj_forward=0`（合成无真实资产）。
- `training_updates=16`（native 8 + explicit 8）；全部零计数为 JSON 整数 0（FD/state FD/DDIM/
  validation/selection/生产 checkpoint 写入/contact 分支）。
- **结论：`audit_incomplete`**，`complete_evidence=true`，`covered_contract_passed=false`，
  `failures=[]`、`problems=[]`。I 与 F 各 611 条必测 gap：`parameter_delta_precision_gap`
  **392**、`cross_path_sensitivity_gap` **219**、`cross_path_bound_unverifiable` **0**。
  v0.4 的 660/340/2 与 F 合计 1002 条是旧 F 对 I 行 gap 与 I 报告 gap 的重复计数；旧工件保留不改写。
- 失败/缺口原因：step 1 的 `m/√v≈0.1` 更新约 `1e-5`，而 |theta|~0.1–1 的 float32 写入分辨率为
  `1e-8–6e-8`，大量坐标低于 §6.2 的 5e-4 可辨门；其余为跨路径近相消敏感性。区间输入预算改为
  梯度门允许集合并逐节点向外 R32 舍入后，0 个坐标落在可信区间外。
- F：独立 Python 子进程（pid≠父、ppid=父、exit 0），继承父绝对截止并在读取/构造/micro/backward/
  VJP/clip/update 前重查父 running 登记；重建 schedule、重读磁盘、逐位重放 P/I；保存报告逐字段
  （loss/parts/gnorm/coef/RNG digest/arrays hash/state presence）、两文件 rows 与报告内嵌 rows、
  两侧行多重集/数组 schema/原始量分类全部复核，`verification_problems=[]`，worker problems=[]。
- 成本：R0 **1.1894s**、P **13.6238s**、I **58.6908s**、F **86.5483s**，合计 **160.0523s**；
  每阶段 1/2 次、无借额（R0 240/P 900/I 900/F 1200s）。
- 工件：实际 **6.895 GiB**；逐类上界单链 **7.145 GiB**、cohort（每阶段 2 次 attempt）**14.236 GiB
  ≤16 GiB**。
- 关键 SHA：`audit_result.json` `082060bcd99af0769c6b6a7bb0c38efe4070ca843efb8c510ead70b00714c643`、
  `finalize/attempt_001/training_update_final.json` `9380f2bcca0d5d9d56a6ff4328ceaecfaf45b5ae901fd0ea05fa586f476f214a`、
  `registration/attempt_001/registration.json` `a9a7c90a08558ad45f4099be88e328d8986137b332f31e8b64c05cc2e460404e`；
  R0 registration 绑定方案 v0.5 `991d2d382e45d23429d464adb0be012210033888d4e87a1ec4019e494ac45c1f`。

## 4. 准入包（v0.5）

路径：`derived/audit/w1_real_training_update_audit/admission/attempt_004/admission_package.json`，
SHA-256 **`c7b4fae260b4ec1779e1e74e3286063519955d68e5c92b8d2bb55b3054132105`**。

- `passed=true`、`problems=[]`；`status={authorized:false, registered:false, run:not_run,
  section_10_2_checked:false}`。
- 绑定 tolerance contract 文件 SHA `9a964372…`、evidence `85246d75…`、fp32 canonical
  `6d7a24d0…`、roster `264`/3864 行、两份父运行 7+3 个文件哈希、两 checkpoint 与 train config。
- `read_scope`：metadata-only；`window_values_loaded=false`、`checkpoint_torch_load=false`、
  `cohort_created=false`、`registered_attempt_created=false`、conditioner/model/VJP/optimizer 调用 0；
  只构造了 CPU 随机初始化 roster（2 个 model constructor）。
- 实现文件哈希与 v0.5 权威运行 R0 registration 完全一致（方案 SHA `991d2d38…`）；
  消费侧 `admission_package_problems` 返回空。
- 资源上界 `one_chain_gib=7.145`、`estimated_artifacts_gib=14.236`、`within_cap=true`。
- 授权模板十项 checklist 仍全为 `null`；真实模式在授权/准入/调用余额校验后仍拒绝
  （`real_provider_not_registered_in_ic`）。

## 5. 测试

- 新增模块：`uv run --locked pytest tests/test_w1_real_training_update_audit.py -q` → **39 passed（90.58s）**。
  v0.5 新增 14 项反例：100×1e-9 全清零/深带全擦除/真实相消正例；m=0 vs 1e-9、50% 状态差、
  单侧零增量；区间实际 binary32 端点包含；clip `norm==clip` 边界；校准状态比值归一化+最大值；
  工件逐类上界；无快照不结算；worker 过期截止不执行单元；过期 F 不 spawn；F 报告 loss/RNG 篡改
  与重复行；I 完整数值 mismatch 进入 F；撤回成本下限；矩阵否决撤回磁盘声明。
- 相关回归（v0.5 重跑）：`tests/test_omniretarget_w1.py`、`tests/test_w1_de_numerics.py`、
  `tests/test_w1_training_objective_audit.py`、`tests/test_w1_real_objective_audit.py`、
  `tests/test_w1_real_admission.py`、`tests/test_w1_real_fk_audit.py` 合计 **263 passed（149.82s）**。
- 全项目套件本轮未重跑；不把历史全量计数当作本轮验收。

## 6. 复现命令

```bash
uv run --locked pytest tests/test_w1_real_training_update_audit.py -q
uv run --locked python scripts/audit_w1_real_training_update.py --calibrate-synthetic
uv run --locked python scripts/audit_w1_real_training_update.py --synthetic \
  --tolerance-contract derived/audit/w1_real_training_update_audit/tolerance_contract.json \
  --out derived/audit/w1_real_training_update_audit/synthetic/v0.5_final
uv run --locked python scripts/audit_w1_real_training_update.py --admission
```

## 7. 第三轮审阅修复映射（8 P1 + 4 P2）

| # | 缺口 | 修复与反例 |
| --- | --- | --- |
| P1-1 | 零噪声门放过非零梯度被清零 | 深带 `1e-10` 才可绝对等价，全擦除/带外走 relative/cosine；100×1e-9 全清零、深带全擦除拒绝，~1e-12 相消噪声通过 |
| P1-2 | 状态绝对豁免掩盖跨路径分歧 | 撤销 `ZERO_STATE_ABS` close 豁免，保留原始相对门；零/符号/单侧零增量显式记录；m=0 vs 1e-9、50% 差异拒绝，精确双零通过 |
| P1-3 | F 未对账保存报告/RNG/行多重集 | `_report_binding_problems` 比较报告字段/RNG/arrays hash/rows 文件；行按 canonical hash 多重集比较；改 loss/RNG、追加重复行被拒 |
| P1-4 | worker 无快照仍按父差值结算 | `_settle_f_calls` 仅接受覆盖全部计数种类的非负 JSON 整型快照；无/部分/bool/负数保持 `consumed=null` |
| P1-5 | worker 未继承截止/逐单元准入 | spec 传绝对 deadline/父登记；guard 在读取/factory/micro/backward/VJP/clip/update 前执行；过期 F 不 spawn，过期 worker 不执行单元 |
| P1-6 | 否决后磁盘仍声明完成 | pre/post-seal 否决重写报告、重绑哈希、以 incomplete 重封口；矩阵否决后磁盘 `complete_evidence=false`、`incomplete_reason=pre_seal_veto` |
| P1-7 | 撤回持久化成本可低估 | 撤回复用 publishing→terminal，`cost_floor=旧成本`，写后覆盖核验；慢写盘反例 ledger 成本≥实测 |
| P1-8 | 完整数值负结果无法进入 F | I/P 用 `complete_evidence` 封口、`numeric_passed` 单列；完整 mismatch 进入 F，失败保留；预算/未知封口仍阻断 |
| P2-9 | 区间遗漏规定舍入节点 | z/m/g²/v/sqrt/b/den/ratio/theta 逐节点向外 R32 舍入；输入预算覆盖梯度门；端点包含检查通过，0 bound-unverifiable |
| P2-10 | clip 边界与生产不一致 | 无条件 `min(1,clip/(norm+1e-6))`，`coef<1` 触发；Trainer 与独立公式同式，`norm==clip` 缩放 |
| P2-11 | 校准误差口径/最大值错误 | 状态比值按 rtol 归一化；全链+标量取最大并保留 sources；`delta_same_input` 0.147，4× 全过 |
| P2-12 | 工件上界被合成运行突破 | P/I/F/float64 传播/重放/重试逐类计算；单链 7.145 GiB ≥实际 6.895，cohort 14.236 GiB≤16 |

## 8. 边界与未覆盖

- 真实 provider、真实 checkpoint strict load、真实父数组/条件、四窗真实输入、真实网络参数反向、
  真实 FK/资产闭包、真实 AdamW 更新、真实 R0→P→I→F、真实授权/注册均未实施或未运行。
- 合成全链的 `audit_incomplete` 只说明该矩阵在冻结精度门下不能证明覆盖通过；不说明 AdamW/网络实现
  错误，也不说明真实链会通过或失败。611 条 gap 是精度/相消的必测不完整，不是 failure。
- 选模、DDIM、端到端质量、干预收益、数据 loader/epoch 等价、生产接入仍 `not_audited`；
  `objective_coverage=partial`、`full_objective_contract_passed=null`、
  `attribution=attribution_insufficient`、`factor_selected=false`、`auto_execute=false`、
  `final_interaction_method=not_ready`、`production_integration=false`。
- U0 `incomplete`、UR `no_quality_gain`、原 F `not_ready`、旧预测 r6 不可验收、旋转融合不接入全部保留。
- 下一步：复阅 v0.5/校准/准入包，另行实施真实 provider 并取得方案明确的真实授权；在此之前不得创建
  真实 cohort 或消耗真实调用额度。

## 9. 历史证据

v0.4 的权威运行 `synthetic/final/`、校准 `history/v0.4_5d9230cf/` 和 `admission/attempt_001/`
保留为历史证据（`audit_incomplete`、1002 条旧统计、admission SHA `2c7f52ef…`）；其 F 行 gap
重复计数和状态绝对豁免已在 v0.5 修复，旧工件不改写。v0.5 文档冻结前的 `synthetic/v0.5_final`
（方案 SHA `7ca93760…`）已移入 `synthetic/history/v0.5_final_plan_7ca93760/`，对应准入包
`admission/attempt_003/`（SHA `a9c8b50e…`）；当前权威为方案 `991d2d38…`、`synthetic/v0.5_final/`
和 `admission/attempt_004/`。
