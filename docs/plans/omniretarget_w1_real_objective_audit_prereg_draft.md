# W1 真实训练目标审计预注册草案（未注册、不可执行）

日期：2026-10-26。状态：**draft，not_registered，not_authorized，not_run**。父方案：[W1训练拟合与目标一致性单因素诊断执行方案](omniretarget_w1_training_fit_objective_audit_plan.md) §5。合成审计结果见[验证记录](../verification/omniretarget_w1_training_objective_audit_verification.md)。

> 本文件只定义“如果用户授权真实数据审计，应事前冻结什么”。它不创建执行注册、cohort、预算或运行目录，不加载任何真实 dataset/cache/checkpoint，不进行模型前向/反向或训练更新，也不选择训练单因素。未完成 §8 授权清单前，`real_data_audit=not_run`、`factor_selected=false`、`auto_execute=false`、最终交互仍 `not_ready`。

## 1. 研究问题与可区分假设

### 1.1 问题

在**固定真实 train 窗口、固定 checkpoint、固定 t/noise、固定评估代码**上：

1. 生产 W1 编码、分项损失、总目标与独立 float64 重算是否在冻结容差内一致？
2. 若不一致，差异位于哪个消费入口：clip 系编码/根标准化、分组均值与 38 通道 `vel` 归约、测地/FK 目标、梯度路径，还是优化器/裁剪/累积？
3. 若目标与梯度在固定真实 batch 上一致，是否仍存在可由“训练目标实现”解释的失败证据？若没有，应交付 `attribution_insufficient`，不得把目标审计通过当作训练主因或泛化结论。

### 1.2 假设

- **H_obj**：在注册的真实 batch/checkpoint 上，生产目标的编码、分项与总损失和独立重算一致（冻结容差内）。若成立，目标实现不是该 batch 失败的可复现解释。
- **H_impl**：存在可复现的消费入口差异（例如分组/归约、FK 目标、梯度 detach、样本权重、micro 平均），且修复该差异会改变目标/梯度事实。只有 H_impl 被独立重放确认，才可提出单因素候选。
- **H_data/other**：目标与梯度一致，失败更可能来自数据/标签/覆盖/泛化，而不是目标实现；本轮不宣称该方向已证实，只登记为下一轮可区分问题。

### 1.3 与旧诊断的区别

本草案不重跑旧预测诊断 R/W/P/F，不测量跨窗预测一致性、DDIM 采样质量或旋转融合；它只审计**训练目标的真实 batch 前向与梯度**。旧 r6 的 cohort 尝试数超限与未计费预检仍保留，不能作为本草案的授权或预算依据。

## 2. Cohort、历史与预算边界

| 项目 | 冻结要求 |
| --- | --- |
| 新 cohort 身份 | `w1_real_objective_audit_v1`；注册时必须写入父 SOURCE/ATTR 哈希、方案哈希、实现 SHA、环境三元 |
| 历史 | 保留旧 R 6/W 5/P 3/F 1 尝试数及成本下界；本 cohort 不重置、不洗掉、不借用旧融合 S 预算 |
| 预算 | 注册时显式分配；草案建议 **≤900s** 墙钟、0 optimizer updates、0 生产训练 step；超时即 `audit_incomplete`，不续跑 |
| 数据边界 | 只读冻结 train24/val31/val6 与缓存；不生成新标签、不扩数据、不改教师、不重采样 |
| 模型边界 | 只读固定 W1 checkpoint；不续训、不选 best、不切换生产入口 |
| 结论边界 | 仅允许 `real_batch_objective_contract_passed` / `real_batch_objective_mismatch` / `audit_incomplete`；不升级 `not_ready` |

## 3. 候选固定资产（注册时必须重新核验）

以下是**候选**，不是本轮已核验事实：

- 冻结 SOURCE 候选：`derived/experiments/omniretarget_first/20261012_21e5784_w1_a3bc_reviewfix8/w1_reviewfix8/`（train24/val31/val6 与缓存来源）。
- W1 checkpoint 候选：`derived/experiments/omniretarget_first/20261013_be00c50_w1_de/` 中 deterministic 与 diffusion 的 best step4000；注册时以当时路径与 SHA 为准。
- 配置候选：`configs/train/omniretarget_motion.yaml`；注册时记录原字节 SHA 与有效配置。
- 实现候选：当前 HEAD + 明确登记的 tracked diff；注册时若工作树不干净，必须保存 implementation snapshot，不能只写 commit。

注册时必须逐项核验：数据/缓存的逐文件 SHA、tokenizer lo/hi/root stats、checkpoint 配置/模型类型、seed、`micro_batch`/`grad_accum`、损失权重、FK backend、环境版本。任一项不可核验即 `audit_incomplete`，不得开始前向。

## 4. 固定真实审计矩阵

### 4.1 窗口与时间步

- **R1 数据键**：从 train24 按冻结规范序取 **2 个 demo × 2 个完整窗口 = 4 个唯一窗口**；注册时必须列出 `(demo, start, end, T, cache_key, split)`，不得只看数量。训练集与验证集不得混用；同 demo 的其他窗口不进入本轮。
- **R2 批次**：`micro_batch=4`、`grad_accum=1`，即一个真实 micro-batch 恰好包含上述 4 窗；不重排、不丢窗、不补窗。若真实数据不足 4 窗，注册失败而不是缩分母。
- **R3 t/noise**：deterministic 模型不注入扩散噪声；diffusion 模型使用注册 seed 生成的固定 `t` 与 `epsilon`，注册时逐值写出 `t`、noise SHA，并明确每个 t 只前向一次。不得在结果出来后改 t。
- **R4 前向/反向上限**：总计 ≤4 次完整前向、≤2 次反向；0 次 optimizer step、0 次 scheduler step、0 次 checkpoint 写入。超出即停止并登记 `audit_incomplete`。

### 4.2 R0：身份与准入（无模型调用）

1. 重哈希 SOURCE、缓存、checkpoint、配置、实现文件与环境 lock。
2. 重建 tokenizer/root stats，核对 lo/hi/mean/std 与缓存逐位相等。
3. 核对 4 个窗口的 key、shape、dtype、有限性、旋转 6D 正交性、关节限位；坏窗保留具体原因，不静默替换。
4. 写出 `registration.json` 与 `admission.json`；任一失败即停止，不做前向。

### 4.3 R1：真实编码与目标重算

对每个已准入窗口/批次：

- 生产路径：`StateTokenizer` 编码 → `LossAssembler` 分项与总损失；记录 `pos/rot6d/joint/geodesic_rad2/vel/fk_m2/total` 及归约分母。
- 独立路径：从保存的原始 `root/q/wfc/box_quat/contact` 用独立 NumPy/Torch float64 公式重算编码、分组均值、38 通道 `vel`、测地项与加权和；FK 项只重算聚合与权重，真实 G1 FK 正确性标记 `not_audited`，不得冒充已通过。
- 容差：注册时从 dtype 与操作数冻结；未冻结前不得看结果调线。编码/损失建议 `atol=1e-9, rtol=1e-9`（float64），生产 float32 对 float64 另设 `atol=2e-5, rtol=2e-5`；最终值以注册为准。
- 输出：每窗每个分项的 `production/independent/abs/rel/within_tolerance`，以及形状、dtype、分母、权重。

### 4.4 R2：真实梯度重算

- 只对 `pos/rot6d/joint/geodesic_rad2/vel` 五个解析项做输出空间独立梯度重算与加权和；`fk_m2` 若涉及真实 G1 FK，明确标记 `fk_gradient=not_audited`，不把未审计项算入“全部通过”。
- 校验：预测分支未 detach、目标分支 `no_grad`、权重实际消费、非有限/零梯度显式登记。
- 可选：对一个小参数子集做中心差分，但必须事前固定参数、方向、eps 与预算；不得在真实模型上运行未登记的全参数 FD。

### 4.5 R3：单步更新重放（默认不执行）

真实 optimizer step 属于“真实数据训练更新”，本草案默认**不包含**。只有用户另行授权、并把它写成独立单因素预注册后，才可：

- 固定初始 AdamW 状态、`lr/weight_decay/betas/eps`、`grad_clip`、`grad_accum`；
- 只做 1 个 optimizer step，记录裁剪前后梯度、参数、`step/exp_avg/exp_avg_sq`；
- 与独立重放逐步对账；
- 0 步续训、0 次选模、0 次生产入口切换。

未获授权时，R3 状态为 `not_run`，不得用合成审计的更新结果替代真实更新。

## 5. 结论与失败分类

| 观察 | 允许结论 | 禁止表述 |
| --- | --- | --- |
| 编码/分项/总目标/梯度均在容差内 | `real_batch_objective_contract_passed` | “训练失败主因不是目标实现”只能作为该 batch 的限定结论 |
| 某个消费入口可复现不一致 | `real_batch_objective_mismatch` + 入口名 | 不能直接宣称“这就是训练失败主因”；须另立单因素验证 |
| 缺数据/缓存/checkpoint/证据、非有限、超预算 | `audit_incomplete` | 不能补跑未注册窗口或缩分母 |
| FK 梯度未审计 | 该子项 `not_audited` | 不能写“全部梯度通过” |

若目标审计通过，而真实训练失败仍存在，下一交付应为 `attribution_insufficient`（目标实现方向）或另立数据/覆盖/泛化预注册；不得自动开始单因素网格、续训、改教师或切换生产入口。

## 6. 停止条件

出现以下任一情况立即停止并写 `audit_incomplete`：

- 任一注册资产哈希不符、缓存键不可重建、checkpoint 配置不匹配；
- 窗口缺失、重复、T/start/t 不符、旋转退化、关节越限、NaN/Inf；
- 前向/反向/时间预算超限；
- 独立重算与生产路径结构不一致（shape/dtype/分母/权重）；
- 任何未登记的真实模型调用、optimizer step 或 checkpoint 写入；
- 结果出来后试图调整容差、窗口、t/noise、权重或结论词汇。

## 7. 计划产物（注册后）

`registration.json`、`admission.json`、`window_keys.json`、`objective_real_rows.json`、`objective_real_arrays.npz`、`gradient_real_rows.json`、`audit_summary.json`、`independent_verification.json`。所有产物写入独立新 attempt 目录；不覆盖旧证据。

## 8. 授权清单（全部为 true 才可执行）

- [ ] 用户明确授权加载真实 dataset/cache/checkpoint 与运行模型前向/反向；
- [ ] 用户确认本 cohort 身份与“不借用旧预算/不重置历史”；
- [ ] 用户确认墙钟预算、前向/反向上限、0 optimizer step（或另行授权 R3）；
- [ ] 注册时冻结实现 SHA、tracked diff / implementation snapshot、环境三元；
- [ ] 注册时冻结 4 个窗口键、t/noise、容差、输出字段与停止条件；
- [ ] 独立复核者与生产者分离，复核器从 raw 工件重算，不信任顶层通过字符串。

在以上事项完成前：**本草案不是可执行预注册；真实数据审计、训练和单因素实验均 `not_run`；最终交互 `not_ready`。**
