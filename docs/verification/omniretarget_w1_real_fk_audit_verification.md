# W1 真实训练解码、FK 几何与输出空间导数审计 —— I/C 实现与纯合成验证记录

日期：2026-10-28。状态：**`implemented_synthetic / calibrated / admission_ready / not_authorized / not_run`**。

本记录只覆盖方案 §4.1 的 I/C 工程工作、纯合成校准和只读准入包。真实 R0→G→D→F、真实父数组、锁定 G1/MuJoCo 编译、checkpoint、cohort、预算与注册均未执行。所有通过结论均为**纯合成契约通过**，不替代真实证据，也不改变 `objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false`。

## 1. 实现身份

| 文件 | SHA-256 |
| --- | --- |
| `src/ireset/eval/w1_real_fk_audit.py` | `42741604c0d59558c1ab999686fa724bca0f641f2647e92f3b704de33179c9f4` |
| `scripts/audit_w1_real_fk.py` | `80ededcb5edef05e798e8ac89b3d5c89343f0654996e4df63010069b8d451673` |
| `tests/test_w1_real_fk_audit.py` | `7bd0fc5ecd4a526c6291eb96e37946b792787feacda406c6ba2316c1fc257d7d` |
| `src/ireset/train/losses.py`（仅新增可选注入） | `33f68d0bf1b9b5c31940b887bdf5413afebe192ee53fd0229782232abcf8f661` |
| 预注册草案（最终字节，准入包绑定） | `aa676d25efe961466f6208c21e1e699c1e35fcfa58e4823ae937bfb5d8b9cae8` |

`LossAssembler` 新增 `fk=` 与 `static_frame=` 两个可选参数：默认值仍构造 `G1TorchFK` 并调用 `largebox_static_frame()`，生产数值路径不变；合成路径注入 29 关节合成 FK 与单位静态框，避免加载真实资产。真实 `RealMujocoReference` 在 G 首次构造时校验 nq/nv、29 关节名称/顺序/qpos 地址/hinge 类型/限位、15 site 所属 body 与局部位姿，并在真实 G 写入 `reference_facts.json`；该路径未在 I/C 执行。`tests/test_generation.py` 与 `tests/test_label_quality.py` 回归 **26 passed**；与父真实目标审计测试合并 **109 passed（52.58s；43 new + 66 parent）**。

## 2. 测试

`uv run --locked pytest tests/test_w1_real_fk_audit.py -q` → **43 passed（26.37s）**。

覆盖与方案 §9.1 的对应：

| 验证组 | 已覆盖 |
| --- | --- |
| 解码/几何 | 列/行 6D 混用、wxyz/xyzw、退化 6D、`|x_j|=1` 边界 gap、`|x_j|>1` 零导数、近 π 正域外负例、目标两副本逐位 |
| Jacobian/梯度 | 35/38 维 FD、漏 `R.T`、错关节枢轴、错关节尺度、分母 3600/900、窗口均值=batch、清零梯度、clamp 零梯度、分组 L2/cosine 门 |
| 证据独立性 | 键缺失/额外/重复、行字段/分母/状态/bool 篡改、数组非逐位/超容差/NaN/缺键、父绑定逐位、F 改状态+数组后拒绝 |
| 准入/调用 | 真实入口无授权零写入拒绝、实验根守卫、实现/草案哈希绑定与 R0 漂移门、G/D 调用精确计数、累计 cap、尝试上限、R0 零 FK/model 调用 |
| 账本/封口 | 负/NaN/Inf 成本拒绝、running/publishing 不可消费、半损坏历史 unknown、外部 cohort/无标记 legacy 归属、可消费 done 封口、慢持久化不得发布成功、撤回改写/计费/不可验证时保持 publishing |

真实资产相关测试仍按方案 §9.1 保留为真实计算，不包装成 I/C 校准。

## 3. 冻结校准

命令：

```bash
uv run --locked python scripts/audit_w1_real_fk.py --calibrate-synthetic \
  --out derived/audit/w1_real_fk_audit/tolerance_contract.json
```

| 项 | 值 |
| --- | --- |
| 契约文件 | `derived/audit/w1_real_fk_audit/tolerance_contract.json` |
| 契约 SHA-256 | `82209619f85da976565edd2af3af485a38b726bc783fc6e49ccc11cbb75f8023` |
| 证据 SHA-256 | `8f46fbe1830d4576f61ff85c8384a78e057ff3f5a900299e5c8c3440e3df0ed9` |
| 模式 | `synthetic_calibrated` |
| seed / 样本 | `20261028` / 64 状态；4 个 FD 状态 × 35/38 维 × 2 个 h |
| 真实读取 | `real_asset_loaded=false`、`real_checkpoint_loaded=false`、`real_dataset_loaded=false`、`real_model_calls=0` |

观测最大比例与 4× 规则：

| 检查 | max ratio | 4× | 提案上界 |
| --- | ---: | ---: | ---: |
| decode_float32 | 0.090188 | 0.36075 | atol 2e-6 / rtol 2e-6 |
| decode_rotation_matrix | 0.008517 | 0.03407 | atol 2e-5 / rtol 0 |
| kernel64_fk | 4.29e-10 | 1.72e-9 | atol 1e-6 / rtol 1e-6 |
| production_fk | 0.056518 | 0.22607 | atol 5e-6 / rtol 2e-6 |
| fk_scalar | 0.002838 | 0.011352 | atol 1e-8 / rtol 2e-5 |
| fk_gradient | 0.200238 | 0.80095 | atol 1e-8 / rtol 1e-4 |
| j35_fd | 5.47e-5 | 2.19e-4 | atol 2e-6 / rtol 2e-5 |
| j38_fd | 2.28e-5 | 9.14e-5 | atol 1e-5 / rtol 1e-4 |

全部 `4× observed ≤ proposal`，阈值保持提案值，未抬高。校准器在加载契约时重算证据与门槛；替换证据、同步重哈希后放宽阈值、改动提案阈值、删除负例均被拒绝（测试覆盖）。

已登记边界/负例：近 π 生产矩阵误差与 quat 标量作为事实保留（不参与容差）；`|x_j|=1` 无唯一双侧导数（gap）；`|x_j|>1` 导数精确为 0；退化 6D 为 gap；零残差损失/梯度为 0。

## 4. 纯合成 R0→G→D→F

命令：

```bash
uv run --locked python scripts/audit_w1_real_fk.py --synthetic \
  --out derived/audit/w1_real_fk_audit/synthetic/20261028
```

运行根：`derived/audit/w1_real_fk_audit/synthetic/20261028/`（直接位于 `derived/audit/`，不是研究实验根）。合成 fixture 在 R0 前由 CLI 生成一次（等价于真实父产物已在盘），不计入 G/D/F 调用计数；真实模式只哈希父文件。`synthetic/20261028_pre_withdrawal/` 是加入撤回成本测试前的历史运行，仅作对照。

| 阶段 | 结果 | 成本（合成） |
| --- | --- | ---: |
| R0 | 绑定/窗口/键数/schema/零调用检查通过 | 1.1040s |
| G | 122 几何行 + 20 梯度行，0 失败/0 缺口 | 1.3501s |
| D | 4672 FD 行（32 状态 × 两个 h），0 失败/0 缺口 | 2.9289s |
| F | 新 provider/后端、磁盘重读、重算 G+D、逐位/容差对账、全 stencil 重放；`problems=[]` | 2.9780s |

最终 `finalize/attempt_001/fk_audit_final.json`：`conclusion=synthetic_contract_passed`、`complete_evidence=true`、`covered_contract_passed=true`、`failures=[]`、`gaps=[]`、`verification_problems=[]`。SHA-256 `c4732be8d7ccaf8fc32bfc11ad66adcf4c25dffd32bdb43af579a4f7418d8a5c`；独立验证 SHA-256 `08933979a2808d0672ed8c67f14533313819bffc19588ab3eaffc58b0a0e0cfd`。

G 计数：`loss_assembler=2`、`production_fk_batch=4`、`kernel64_fk_batch=3`、`production_vjp=2`、`mj_forward=1200`、`mj_jac_site=7200`、`fd_pose_eval=0`、`scene_setup_forward=0`（合成无 MuJoCo 编译）。D：`mj_forward=9344`、`fd_pose_eval=9344`。F：`mj_forward=10544`、`mj_jac_site=7200`、`fd_pose_eval=9344`。全链零计数均为 JSON 整数 0。

合成成本总计 **8.3610s**，仅证明管线可运行，不是真实运行时间估计。

## 5. 准入包

命令：

```bash
uv run --locked python scripts/audit_w1_real_fk.py --admission \
  --out derived/audit/w1_real_fk_audit/admission \
  --tolerance-contract derived/audit/w1_real_fk_audit/tolerance_contract.json
```

| 项 | 值 |
| --- | --- |
| 包路径 | `derived/audit/w1_real_fk_audit/admission/attempt_006/admission_package.json` |
| 包 SHA-256 | `5cfb7aa381c9dd70a8c262f6b2e04bd571ccb4ef4293f1d6ae078e75bdb19b5f` |
| `passed` | true（`problems=[]`） |
| 状态 | `authorized=false / registered=false / run=not_run / section_9_2_checked=false` |
| 父运行绑定 | 7/7 文件哈希匹配（注册/准入/p_report/两份数组/终局/独立复核） |
| 固定窗口 | 4/4 与父注册 `window_keys` 一致 |
| 键数 | geometry 122 / fd 4672 / states 32 |
| 全链调用上限 | `loss_assembler=4`、`production_fk_batch=8`、`kernel64_fk_batch=6`、`production_vjp=4`、`mj_forward=21088`、`mj_jac_site=14400`、`fd_pose_eval=18688`、`scene_setup_forward=2` |
| 读取范围 | 仅元数据/文件哈希；未加载窗口数组、未编译资产、未 `torch.load` checkpoint、未创建 cohort 或正式运行根 |

包内 `implementation.files` 与 `tolerance.evidence_sha256` 均与当前文件/校准一致；改动任一被绑定文件会使准入包和真实入口拒绝。`attempt_001`–`attempt_005`（`1b27ca15…`/`1c579b2a…`/`7cfb259a…`/`3c0b9f44…`/`81188587…`）是逐步加入撤回成本、慢持久化、R0 实现漂移门、真实参考事实校验和最终草案之前的历史包；`attempt_006` 为当前权威。

## 6. 仍未执行与未覆盖

- 未勾选方案 §9.2 九项；未创建真实 cohort/预算/注册；未执行真实 R0/G/D/F。
- 未读取父 `objective_real_arrays.npz` 的数值数组（准入包只哈希文件）、未编译锁定 MJCF/URDF、未加载两份 checkpoint。
- 未审计：真实 float32/CUDA 生产前向、真实 MuJoCo 编译后地址/site 位姿、真实父 FK 逐位绑定、真实 FD、网络参数梯度、累积/裁剪/AdamW/更新、选模、DDIM、端到端动作质量。
- F 的独立性来自新 provider/后端与磁盘重读；参考公式与 G 共用同一套 NumPy/MuJoCo 实现，不是第三套数学证明。
- I/C 的 `scene_setup_forward=0` 是合成事实；真实 G/F 每个 stage 应各为 1，由 R0/G 首个已登记单元核验。
- 真实入口在无授权、无校准、无准入包或运行根不是固定历史根直接子目录时拒绝，且在读取数组/加载资产/创建运行根之前退出（测试覆盖）。

## 7. 复现命令

```bash
uv run --locked pytest tests/test_w1_real_fk_audit.py -q
uv run --locked python scripts/audit_w1_real_fk.py --calibrate-synthetic \
  --out derived/audit/w1_real_fk_audit/tolerance_contract.json
uv run --locked python scripts/audit_w1_real_fk.py --synthetic \
  --out derived/audit/w1_real_fk_audit/synthetic/20261028
uv run --locked python scripts/audit_w1_real_fk.py --admission \
  --out derived/audit/w1_real_fk_audit/admission \
  --tolerance-contract derived/audit/w1_real_fk_audit/tolerance_contract.json
```

真实执行入口保持不可达：

```bash
# 无授权/校准/准入包时拒绝；本仓库状态未勾选 §9.2
uv run --locked python scripts/audit_w1_real_fk.py --real \
  --out derived/experiments/omniretarget_first/<new_run> \
  --authorization <authorization.json> \
  --tolerance-contract derived/audit/w1_real_fk_audit/tolerance_contract.json \
  --admission-package derived/audit/w1_real_fk_audit/admission/attempt_006/admission_package.json
```

## 8. 边界

`objective_coverage=partial`、`full_objective_contract_passed=null`、`attribution=attribution_insufficient`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false`。旧 U0 `incomplete`、UR `no_quality_gain`、原 F `not_ready`、预测 r6 不可验收、旋转融合不接入均保持不变。真实通过/失败只能由取得授权后的 R0→G→D→F 产生，不能由本记录外推。
