# 教师重生成鲁棒性修复方案

日期:2026-09-15。前序:`docs/archive/verification/teacher_regen_verification.md` §5"后续"。状态:**已实现(F1+诊断增强);F3 经 P1 检查点取消;F2/F4 机理评估不实现;先导重测 2/6 未达标——失败机理与本文 §2 假设不符,见 [验证记录](../verification/teacher_regen_robustness_verification.md) §2 修订与 §5 后续建议**。

## 1. 背景与目标

路径 B(教师重生成)先导实验(诊断 6 段):**2/6 保留**(接触覆盖 90-100%、任务质量 0.072-0.076、手部认证区间首次成形);四段被拒,集中在两类失败:

1. **转移检查失败(3/4 段)**:sub12_largebox_001 (t=11)、sub16_largebox_000 (t=22)、sub7_largebox_000 (t=24)——均在"拉入时刻"(approach corridor 接近阶段或抓握建立初期)。
2. **教师自带穿透起始帧(1/4 段)**:sub3_largebox_000 (t=1)——旧教师轨迹手部已穿箱 1.4mm,深穿透线性化边缘不稳定。

**目标**:提高先导 6 段接受率从 2/6(33.3%)至 **≥4/6(66.7%)**,为规模化(24 段标定集 → 312 originals)准备;保持已成功 2 段的输出质量与严格验证通过。

## 2. 失败模式详细分析

### 2.1 转移检查失败(类别 #1)

**症状**:
- `status="rejected_frame"`,`skip_reason="state_constraints"`,`t_fail` 在抓握建立初期(11-24 帧,占序列 7-20%)。
- 端点状态可通过 `frame_valid` 检查,但插值中间态触发碰撞行违规。

**根因假设**:
1. **走廊上界收缩不足**:当前实现 `upper = max(gap, prev_dist - 0.15*dt)`。步长平方 dip 余量(最大 6mm,于 `solver_band_margin`)仅作用于 solver 内部 band 行,转移探测的端点/插值使用原始 `band_margin_m=0.0`(validator 级)——**走廊上界本身未同步收紧**。拉入帧的 QP 解可合法贴走廊(比如 8mm),但插值态下探至 8 - 0.3×motion² ≈ 7mm 左右,探测的 validator 级 -2mm 门拒绝。
2. **过渡帧运动量过大**:初次抓握建立时手部从悬停(~30mm)向箱接近,单帧笛卡尔步长可达 15-30mm(实测),motion² 项显著;线性插值的中间态在这种大步下可能穿透更多。
3. **探测与求解器不对称**:SCP 求解器每次迭代重建几何与雅可比,视野动态;转移探测固定采用前后两端的碰撞对集合——端点激活的碰撞对可能在中间态失效或变化,pair 扫描盲区。

**实测证据**(先导日志):
- sub12 t=11 之前顺利,到 t=11 报 `transition_interpolated`。
- sub16/sub7 同模式;t_fail 帧自身 `frame_valid` 可通过(端点无问题),仅插值拒绝。

### 2.2 教师穿透起始(类别 #2)

**症状**:
- sub3_largebox_000,t=1(第二帧,索引从 0 起)立即失败。
- 旧教师该时刻手部几何距箱 ≈ -1.4mm(已进入 box hull),系统性使用深穿透状态作为热启动。

**根因**:
1. **深穿透线性化退化**:距离雅可比在 dist < −1mm 时方向与梯度可能不稳,线性化预测偏差增大,QP 步长方向不可靠。
2. **首帧无 z_prev**:t=0 可接受(或旧教师刚好未穿),t=1 从穿透热启动、又无 multi-start 备选(z_prev=None,pipeline 的 multi-start 仅在 `z_prev is not None` 时生效)。
3. **走廊机制未覆盖首帧**:`prev_hand_dist` 在 t=0 后才有值,首帧走廊上界可能退化为 hard gap(8mm),与实际 −1.4mm 差距过大。

**需要验证**:
- sub3 的 t=0 是否通过?若 t=0 就失败,则为起始状态整体不可达;若 t=0 通过而 t=1 失败,才是线性化/热启动问题。
- 穿透程度:−1.4mm 是 hover 悬浮还是真 penetration?检查 validator 的 box hull SDF 符号(负=内部)。

### 2.3 其他潜在因素

- **solver 预算**:已设 40+40(P0.1 L0),先导计时 82-630ms/帧、迭代常满 40——不是预算不足,而是收敛困难。
- **merit 惩罚**:regeneration 用 `mu *= 30`,防止跟踪目标压倒约束违规;但过高 mu 也可能使 line-search 过保守。
- **hand_speed 硬行**(已禁用):regen_hand_speed_mps=0.0,非活跃因素。

## 3. 修复方案

### 3.1 F1:走廊上界同步收紧(转移检查对策,优先)

**动机**:插值 dip 余量已在 solver band 行生效,但走廊上界(active upper bound)本身未减去该余量,导致端点解可合法贴走廊、插值探测拒绝。

**实现**:
1. 在 `scp.py::_collision_rows` 的 regen_active 上界计算中,**减去插值 dip 余量**:
   ```python
   margin = solver_band_margin(ctx, cfg)  # 调用已有函数
   upper = cfg.regen_hand_contact_gap_m if prev is None else max(
       cfg.regen_hand_contact_gap_m,
       prev - cfg.regen_closing_rate_mps * self.dt - margin)
   ```
2. 验证:相同 `ctx.frame_motion_m` 在 solver 行与走廊上界用同一 margin 公式。

**风险**:
- 过紧的走廊可能使 QP 在大步帧不可行(motion=0.03 → margin≈2.7mm,走廊从 8mm 减为 5.3mm,单帧可达性边际)。
- 若不可行率上升,fallback 到 slack 恢复或 multi-start。

**验收**:
- sub12/sub16/sub7 三段的 t_fail 帧位置后移或通过。
- 诊断:插值前后 hand_dist、corridor upper、实际 dist、margin 数值对齐。

### 3.2 F2:拉入帧 backtrack 沿走廊方向收缩(转移对策,备选)

**动机**:F1 仅收紧求解目标,不改 SCP 接受规则。若 QP 输出步长过大(比如从 20mm 一步到 5mm),插值中间态仍可能越界;当前 backtrack(`_transition_backtrack`)沿 tangent 全维收缩,可能不是走廊最优方向。

**实现**:
1. 在 `pipeline.py::_transition_backtrack` 中,检测失败类型是否为 regen corridor violation(hand_dist 在中间态超上界)。
2. 若是,沿**手部接近方向**(Cartesian,而非全身 35 维 tangent)做定向 backtrack:保持其他自由度不变,仅收缩手到箱的接近量。

**复杂度**:
- 需要分解 tangent step `xi` 到"仅手部分量",实现复杂。
- 当前 backtrack 已沿原切向量试 0.75/0.5/0.25 三步,通用性好;定向版收益不确定。

**决策**:
- 先验证 F1(上界收紧)效果;若 F1 不足再实现 F2。
- F2 列为**可选**,不阻塞先导扩展。

### 3.3 F3:首帧穿透预修复(教师起始对策)

**动机**:旧教师 t=1 已穿透,线性化从内部向外爬不稳定;首帧无 z_prev 使 multi-start 不可用。

**实现**:
1. **检测**:pipeline 开始前,对 `prediction[0]` 运行一次 `frame_valid` 检查;若首帧就违规,标记 `needs_initial_repair`。
2. **预修复**:从参考姿态初值(估计根位置/yaw、G1 零位关节)+ 当前箱姿做一次纯几何修复(只强制非穿透/限位,无锚点/速度行,目标仅 mesh);接受结果作为 t=0 热启动。
3. **Fallback**:若预修复失败,**显式拒绝**整段(起始帧不可达,不应进入逐帧循环后再失败)。

**配置**:
- 新增 `RepairConfig.repair_initial_frame: bool = True`(regen_mode 下默认开启)。
- 首帧修复独立计时,不计入逐帧平均。

**风险**:
- 预修复可能改变 t=0 位姿到与参考人体错位(比如根平移 5cm),后续锚点建立可能不匹配。
- 需验证:sub3 的 t=0 本身是否已穿透;若 t=0 合法、仅 t=1 穿,则预修复目标应是 t=1 而非 t=0。

**澄清点**:
- 若 t=0 合法,t=1 失败是"从合法到穿透"的过渡问题 → F1/F2 可能已覆盖,不需独立预修复。
- 若 t=0 本就穿透 → 预修复必要。
- **先检查 sub3 t=0 状态**,再决定 F3 具体实现(是修 t=0 还是加强 t=0→t=1 过渡)。

### 3.4 F4:Deep-penetration 裁剪到浅层(教师起始对策,备选)

**动机**:若无法避免从穿透热启动(比如旧教师整段都穿),在距离/雅可比计算前**裁剪 dist 到 −2mm**,避免深穿透的数值不稳。

**实现**:
1. 在 `FrameEvaluator.geom_distance` 包装层,返回前 `dist = max(dist, -0.002)`(仅用于修复,validator 保持真实值)。
2. 雅可比方向保持不变(仍指向真实最近点),仅距离量裁剪。

**Trade-off**:
- 裁剪使约束行"看不到"真实穿透深度,QP 可能低估违规。
- 但深穿透下的线性化本就不可信,裁剪可能反而稳定。

**决策**:
- 列为**实验性可选**(需 A/B 对比);仅当 F3 不足时启用。
- 标注为"非物理修正",产物需标记。

## 4. 实施优先级

1. **P0(必做)**:
   - F1:走廊上界同步收紧(代码行数 <10,风险低,直接针对 3/4 失败)。
   - 诊断工具增强:regenerate_teachers.py 添加 `--verbose` 输出每帧 hand_dist、corridor_upper、插值探测细节。
2. **P1(先导扩展前完成)**:
   - F3 前置检查:验证 sub3 的 t=0 状态(合法 vs 穿透)。
   - 根据检查结果实现 F3(预修复 vs 过渡增强)。
3. **P2(可选,视 P0/P1 效果)**:
   - F2:定向 backtrack(仅当 F1 后仍有转移失败)。
   - F4:深穿透裁剪(仅当 F3 后仍有起始失败)。

## 5. 验收标准

### 5.1 先导 6 段(诊断集)

- **接受率**:≥4/6(66.7%),最低 3/6(50%)可接受,需解释剩余失败。
- **已成功 2 段保持**:sub10/sub8 在修复后仍通过 strict 验证,质量指标(接触覆盖、任务 MPJPE、认证区间)不退化。
- **失败分类**:剩余被拒段的失败类别明确(转移 vs 起始 vs 新类别),有诊断日志。

### 5.2 代码质量

- 修复逻辑**不改变** strict validation 的任何容差数值(−2mm 穿透门、0.01m 锚漂门、0.05m/s 准静态门等)。
- 走廊/margin 计算有单元测试:给定 frame_motion,验证 solver_band_margin 与 corridor 上界对齐。
- git commit 分离:F1/F3 各自独立提交,可单独回滚。

### 5.3 性能

- 单帧耗时中位数 ≤ 先导基线(82-630ms,median ~200ms);若 F3 预修复使首帧超 1s,需单列说明。
- 迭代数分布:successful 段的平均 SCP 迭代 ≤ 35(满 40 说明收敛困难,需后续优化但不阻塞先导)。

## 6. 扩展路径

先导 6 段达标后:

1. **标定集 24 段**(train-only,含 6 诊断段):
   - 接受率目标 ≥18/24(75%),严格验证通过 ≥15/24(62.5%)。
   - 物理接触覆盖(contact_recall)≥50% 的筛选比例 ≥70%。
2. **全量 312 originals**:
   - 预期接受率 60-70%(考虑长尾困难样本)。
   - 保留段数目标 ≥180(58%),足够训练集规模。
3. **M3 重训**:
   - 使用重生成教师(干净标签)替换原 OmniRetarget 输出。
   - 训练至饱和(validation loss 收敛,预估 10-15 epoch / ~3h per model)。
   - 重跑 M4 闭环(P0-fix 口径,strict 模式),验证生成质量提升与接受率达标。

## 7. 技术债务与后续

- **走廊机制泛化**:当前硬编码 hand-box,若未来扩展到 foot-ground 主动接触需重构。
- **multi-start 增强**:当前仅 z_prev,可扩展到"参考驱动初值"作为第三起点(对首帧和无 z_prev 场景)。
- **SCP 收敛诊断**:先导的 40 迭代满额(82-630ms)说明收敛慢;后续可分析目标/约束冲突,调整权重或 merit。
- **contact_recall vs interval_recall 差距**:物理接触 90-100%,但认证区间仅 5-9%——认证要求准静态进入,机器人手速抖动 2× 参考;改进方向是求解器稳态化(降低 per-frame chatter),而非放宽门槛。

## 8. 关联文档

- 前序实验:`docs/archive/verification/teacher_regen_verification.md`(先导 2/6 结果与发现)
- 路径 B 设计:`docs/plans/p0_relaxed_validation_plan.md` §7
- 实现方案:`docs/plans/interaction_retarget_impl_plan.md` M3 教师重生成任务
- 修复器设计:plan §8.3/8.4/8.5,`src/ireset/repair/{scp,pipeline}.py`
- Slack 修复与 dip 余量:teacher_regen_verification.md §2.1/2.2

---

**修订记录**:
- 2026-09-15 初稿:基于先导 6 段 2/6 结果与四段失败分析
- 2026-09-15 实现轮:F1(走廊上界减 `solver_band_margin`,与 solver band 行同一公式)与 `--verbose` 诊断已实现并重测;**§2 根因假设被插桩证据修订**(四段失败均非走廊-余量不同步:sub12=教师滑步+零运动累积偏离、sub16=区间激活前自碰撞弦线、sub3=脚部端点+插值(t=0 合法,F3 取消)、sub7=教师手部贯穿箱体俯冲);F2/F4 触发条件/有效性不成立,未实现;先导保持 2/6,已保留段零退化;后续按验证记录 §5 的新设计项推进
