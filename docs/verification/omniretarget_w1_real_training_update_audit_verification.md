# W1 真实网络参数梯度与优化更新链路审计 —— I/C 工程与纯合成验收记录

日期：2026-10-29。方案：[omniretarget_w1_real_training_update_audit_prereg_draft.md](../plans/omniretarget_w1_real_training_update_audit_prereg_draft.md)
（v0.4）。状态：**I/C 实现与纯合成验收完成；真实 provider/授权/注册/真实 R0→P→I→F 未实施**。

本记录只覆盖合成证据与冻结准入材料。没有读取真实 checkpoint 张量、真实 NPZ 数值、数据集或 G1 资产，
没有执行真实网络前向/反向、真实 FK、真实参数更新或生产 checkpoint 写入。

## 1. 交付文件

| 文件 | 作用 |
| --- | --- |
| `src/ireset/eval/w1_real_training_update_audit.py` | 审计器：roster/行目录、合成 fixture、P/I 捕获、独立参考、有效 float32 AdamW、跨路径传播、R0/P/I/F、预算/封口、校准、准入、真实门 |
| `scripts/audit_w1_real_training_update.py` | CLI：`--synthetic`、`--calibrate-synthetic`、`--admission`、`--real`（门后拒绝） |
| `tests/test_w1_real_training_update_audit.py` | 25 项纯合成反例/接口测试 |
| `src/ireset/train/trainer.py` | 抽出共享 `predict` / `begin_update` / `accumulate_micro` / `clip_gradients` / `step_optimizer`，新增只读 observer 与审计注入 `loss_fn`/`build_optimizer`；默认路径不变 |

共享单元由原生 `Trainer.run` 与审计共同调用。`observer=None` 与 `observer={}` 的
theta/m/v/step 逐位相等，loss/parts 与全局 RNG 不变；NaN 输入的异常类型/原因一致
（`tests` 覆盖 deterministic/diffusion）。

## 2. 冻结校准与有效 float32 契约

目录：`derived/audit/w1_real_training_update_audit/`。

| 工件 | SHA-256 |
| --- | --- |
| `tolerance_contract.json` | `5d9230cfd28ef5f48d7748765e15dbbf8648a1d1603af2ffd368526527dfc642` |
| 校准 evidence（contract 内） | `fc39c93d400e3299ca64f778fb9a2912c940b672e789a460a0edd8cc8711ec3c` |
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
| `theta_cross_path` | 7.3e-5 |
| `state_cross_path` | 2.0e-5 |
| `delta_cross_path` | 3.8e-4 |
| `delta_same_input` | 4.7e-5 |

有效求值契约：

- 当前环境 torch 2.14.0+cu126 / RTX 3090，CUDA/float32 **默认 foreach** dispatch 与
  §4.3.1 节点图逐位一致：4 组固定/对抗/次正规/首步用例、两步递推、源码 SHA 与新进程重算全部通过。
- CUDA 单 tensor/`foreach=False` 的标量除法是倒数乘而非正确舍入除法；CPU 单 tensor 的
  addcmul/最终加也不同。契约只绑定默认 foreach；CPU-only/强制其他 dispatch 不得准入。
- 独立 `x_t` 使用生产 **binary32** `alpha_cumprod` 提升到 float64 的有效常数；原始 float64
  schedule 只作独立对账。此前用 float64 alpha 时 `alpha_x_t` 观测为 0.745（4× 越界），改用有效
  常数后为 0.2168，2e-6 提案无需放宽。
- 数学零梯度（attention key bias 等）在 float32 autograd 中留下约 1e-9 相消噪声；方案原
  `1e-30` 极小门不可行，校准为 `ZERO_GRADIENT_ABS=1e-6`、`ZERO_STATE_ABS=1e-6` 的绝对零噪声/
  零状态等价门。该门不放宽非零比较。

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

## 3. 权威纯合成全链 R0→P→I→F

运行根：`derived/audit/w1_real_training_update_audit/synthetic/final/`。

- 配置/规模：full `omniretarget_motion` 网络（d_model 240、4 layers、6 heads、time_window 31）；
  roster **130 deterministic + 134 diffusion = 264** 个 tensor，行目录 **3864 = 14N+168**。
- 四窗 `sub10_largebox_063/085` 两个 demo 的重叠窗；每模型两个连续更新、各 4 个 B=4 micro；
  16 个诊断 micro/路径；CPU generator seed `20261008`。
- 四 stage 全部 `done`/sealed，索引与台账一一对应。
- 调用矩阵精确等于方案 §7.2 全链名义值：checkpoint loads 8、model/conditioner constructions 8/8、
  native optimizer 4、conditioner/backbone forwards 64/64、add_noise 32、draw units 48、
  production loss 32、production FK **64**（P 32 + F 32；每 micro pred+target 两批）、parameter backward 32、
  output/total VJP 32/32、native clip 8、independent clip 8、native AdamW 8、explicit updates 8、
  diagnostic updates 16、anchor restores 4、formula checks 8、reference FK 15840、jacSite 115200、
  micro units 64；`scene_setup_mj_forward=0`（合成无真实资产）。
- `training_updates=16`（native 8 + explicit 8）；全部零计数为 JSON 整数 0（FD/state FD/DDIM/
  validation/selection/生产 checkpoint 写入/contact 分支）。
- **结论：`audit_incomplete`**，`complete_evidence=true`，`covered_contract_passed=false`，
  `failures=[]`、`problems=[]`。I 与 F 各自重推合计 1002 条 gap 记录：`parameter_delta_precision_gap`
  **660**、`cross_path_sensitivity_gap` **340**、`cross_path_bound_unverifiable` **2**。去重后同一逻辑
  坐标由 I 与 F-I 各记录一次；均为必测不完整，不是直接 failure。
- 失败/缺口原因：step 1 的 `m/√v≈0.1` 更新约 `1e-5`，而 |theta|~0.1–1 的 float32 写入分辨率为
  `1e-8–6e-8`，大量坐标低于 §6.2 的 5e-4 可辨门；另有跨路径近相消坐标和 2 个区间无法外包的坐标。
  这与此前“固定权重/窗口/lr/阈值不换”的预注册要求一致。
- F：独立 Python 子进程（pid≠父、ppid=父、exit 0），重建 schedule、重读磁盘、逐位重放 P 的两步更新，
  再从相同前态重放 I；行多重集/数组 schema/原始量分类复核通过，`verification_problems=[]`。
- 成本：R0 1.17s、P 13.71s、I 56.65s、F 74.93s，合计 **146.461s**；每阶段 1/2 次、无借额。
- 合成 fixture 预测最大相对旋转 0.674 rad，decode/geodesic/FK 域通过。

## 4. 准入包

路径：`derived/audit/w1_real_training_update_audit/admission/attempt_001/admission_package.json`，
SHA-256 **`2c7f52efa6e22d62c013644f188c40aa12fc5fbd9d4d9436663302c67d4b4838`**。

- `passed=true`、`problems=[]`；`status={authorized:false, registered:false, run:not_run,
  section_10_2_checked:false}`。
- 绑定 tolerance contract 文件 SHA `5d9230cf…`、evidence `fc39c93d…`、fp32 canonical
  `6d7a24d0…`、roster `264`/3864 行、两份父运行 7+3 个文件哈希、两 checkpoint 与 train config。
- `read_scope`：metadata-only；`window_values_loaded=false`、`checkpoint_torch_load=false`、
  `cohort_created=false`、`registered_attempt_created=false`、conditioner/model/VJP/optimizer 调用 0；
  只构造了 CPU 随机初始化 roster（2 个 model constructor）。
- 实现文件哈希与权威运行 R0 registration 完全一致；消费侧 `admission_package_problems` 返回空。
- 授权模板十项 checklist 仍全为 `null`；真实模式在授权/准入/调用余额校验后仍拒绝
  （`real_provider_not_registered_in_ic`）。

## 5. 测试

- 新增模块：`uv run --locked pytest tests/test_w1_real_training_update_audit.py -q` → **25 passed**。
  覆盖共享单元逐位/RNG/异常、roster/行公式、噪声矩阵、1 ULP/20%/局部清零/小更新 gap/区间篡改、
  CUDA fp32 契约重算与篡改、P/I 调用矩阵与行分类、F 新进程逐位重放、数组/行篡改检测、
  stage 状态/预留、call counter、授权记录和撤回。
- 相关回归：`tests/test_omniretarget_w1.py`、`tests/test_w1_de_numerics.py`、
  `tests/test_w1_training_objective_audit.py`、`tests/test_w1_real_objective_audit.py`、
  `tests/test_w1_real_admission.py` 合计 **186 passed**；`tests/test_w1_real_fk_audit.py` **77 passed**。
- 全项目套件本轮未重跑；不把历史全量计数当作本轮验收。

## 6. 复现命令

```bash
uv run --locked pytest tests/test_w1_real_training_update_audit.py -q
uv run --locked python scripts/audit_w1_real_training_update.py --calibrate-synthetic
uv run --locked python scripts/audit_w1_real_training_update.py --synthetic \
  --tolerance-contract derived/audit/w1_real_training_update_audit/tolerance_contract.json \
  --out derived/audit/w1_real_training_update_audit/synthetic/final
uv run --locked python scripts/audit_w1_real_training_update.py --admission
```

## 7. 边界与未覆盖

- 真实 provider、真实 checkpoint strict load、真实父数组/条件、四窗真实输入、真实网络参数反向、
  真实 FK/资产闭包、真实 AdamW 更新、真实 R0→P→I→F、真实授权/注册均未实施或未运行。
- 合成全链的 `audit_incomplete` 只说明该矩阵在冻结精度门下不能证明覆盖通过；不说明 AdamW/网络实现
  错误，也不说明真实链会通过或失败。
- 选模、DDIM、端到端质量、干预收益、数据 loader/epoch 等价、生产接入仍 `not_audited`；
  `objective_coverage=partial`、`full_objective_contract_passed=null`、
  `attribution=attribution_insufficient`、`factor_selected=false`、`auto_execute=false`、
  `final_interaction_method=not_ready`、`production_integration=false`。
- U0 `incomplete`、UR `no_quality_gain`、原 F `not_ready`、旧预测 r6 不可验收、旋转融合不接入全部保留。
- 下一步：复阅 v0.4/校准/准入包，另行实施真实 provider 并取得方案明确的真实授权；在此之前不得创建
  真实 cohort 或消耗真实调用额度。
