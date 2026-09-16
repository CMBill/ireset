# 教师重生成鲁棒性修复验证记录

日期:2026-09-15。方案:[teacher_regen_robustness_plan.md](../plans/teacher_regen_robustness_plan.md)。状态:**P0 已实现(F1 + 诊断增强);P1 检查点完成(F3 前提证伪,按方案自身决策树取消);F2/F4 机理评估后不实现;先导重测 2/6(与基线持平、已保留段零退化),≥4/6 目标未达成——插桩证据表明四段失败机理与方案 §2 假设不符**,修订记录见 §2/§5。strict 容差数值零改动。

## 1. 实现内容

| 项 | 状态 | 内容 |
|---|---|---|
| F1 走廊上界同步收紧 | **已实现** | `scp.py::_collision_rows`:regen 主动上界 = `max(gap, prev − rate·dt − solver_band_margin(ctx,cfg))`,与 solver band 行用**同一** margin 公式(走廊上行解贴界、插值态下探 ~margin,端点须多收一个 margin)。单测:`test_regen_corridor_and_band_share_one_margin_formula`(四种 motion 量级下走廊上界与 band 地板逐值对齐)+ 更新 `test_regen_active_hand_rows_ignore_geometric_band` 期望值 |
| 诊断增强(P0 第二项) | **已实现** | `RepairPipeline.verbose`/`frame_log`:每次端点检查记录 viol、双手 prev/cand 距离、走廊 dist/upper(`solver.last_corridor`,由 `_collision_rows` 权威记录,无公式重复)、band margin、motion、n_sub;每次插值探测记录 viol、中态手距、violation rows(含几何体对名,`solver.last_row_pairs`);`regenerate_teachers.py --verbose` 逐帧摘要 + 全部失败行打印 + 逐段 `frame_log_<demo>.json`。非 verbose 路径零行为变化 |
| F3 首帧穿透预修复(P1) | **取消** | P1 前置检查(§2.3):sub3 t=0 **合法**(手距箱 31.9/68.3mm)且 t=0 被成功修复至 −1.39mm;按方案 §3.3 决策树"t=0 合法 → 不需独立预修复"。方案前提"旧教师 t=1 手部已穿箱 1.4mm"与实测不符(见 §2.3) |
| F2 定向 backtrack(P2) | 不实现 | 触发条件不满足:sub16 失败在抓取区间激活**前**(t=22<24),无走廊行可言;sub7 的失败类别是端点 `state_constraints` 而非 `transition_interpolated`,`_transition_backtrack` 本就不触发 |
| F4 深穿透裁剪(P2) | 不实现 | 只影响求解器内部线性化;端点/探测仍按真实距离判定(validator 口径),不改变可行性判定;且 sub3 实际失败行是**脚部**(§2.3),sub7 的贯穿是教师数据属性,裁剪只掩盖不解决 |

## 2. 失败机理修订(插桩证据,覆盖方案 §2 假设)

先导四段失败逐帧插桩(`scratch` 诊断脚本,机制已固化为 `--verbose`)结论:

| 段 | t_fail | 方案 §2 假设 | 实测机理 |
|---|---|---|---|
| sub12_largebox_001 | t=11 `state_constraints` | 走廊上界收缩不足 | **教师带滑步行走 + 参考抓取区间自 t=0 激活**。t=0 修复把手压到 −1.39mm(接受地板 −1.5mm 之上 ~0.1mm 刀锋);t=2 起教师以 ~112mm/帧站点运动走远(箱心不动),修复器每帧 tight QP 不可行→slack 接受"零运动"候选(其 motion<50mm→n_sub=1→无探测,侥幸通过);累积偏离后任何追赶步都是 ~120mm 大步,插值弦线穿箱(中态 −27mm)被探测正确拒绝;t=11 箱开始漂移(0.7mm/帧),贴箱手距越过 −1.5mm 地板,小步修正与追踪步全部失败 |
| sub16_largebox_000 | t=22 `state_constraints` | 走廊上界收缩不足 | **抓取区间(t=24 起)激活前的大步抓取,自碰撞弦线**。参考人体手已到箱(跟踪目标在箱面),候选步 130-150mm/帧;终点激活集合(5cm)看不到膝-手对,弦线中态 `right_knee_link~right_rubber_hand_link` −6.4..−10.9mm(< −1.8mm 门)——探测与求解器的 pair 集不对称(方案 §2.1 假设 3 方向正确,但败在**自碰撞对**而非走廊) |
| sub3_largebox_000 | t=1 `state_constraints` | 教师起始穿透(t=1 手已穿 1.4mm) | **t=0 合法且修复成功**;t=1 败因是**脚部**:教师该帧自身脚-地端点 −1.65..−2.16mm(< −1.5mm 接受带),通过脚部的候选再败于插值中态 `left_ankle_roll_link~ground` −2.16mm(< −2mm 门,motion=50.2mm→n_sub=2)。手部非失败行 |
| sub7_largebox_000 | t=24 `state_constraints` | 拉入时刻转移检查 | **教师手部 1-2 帧内贯穿箱体**(55.7mm→−23mm,原始 51.8mm/帧俯冲),走廊(5mm/帧闭合)与速度限结构上无法复现;候选被跟踪目标(weight 25)拖入穿透(−22.9mm < −1.5mm 地板),线搜索/信赖域收缩后停滞。非插值探测问题 |

**共性**:旧教师自身在大步长帧违反严格约束模型(与 M1 结论"教师非可行真值"一致的具体化)。修复器在这些帧必须**合法偏离**教师,而当前逐帧接受机制(对 z_pred 的 stationarity 检查 + task 0.40m 门)与走廊的持续闭合要求,在累积偏离增大后产生退化 QP(预测误差 ≫ 信赖域)。F1 的"走廊-余量不同步"机制在四段中均非主因。

## 3. 先导重测(v3 = v2 + F1,strict,全长度,--verbose)

`derived/data/teachers_regenerated/pilot_diagnosis6_v3_f1/`:

| 段 | v2 | v3(F1) | v2→v3 变化 |
|---|---|---|---|
| sub10_largebox_000 | keep | **keep** | 接触覆盖 1.00/0.905、认证 0.091/0.041、task p95 0.0756——与 v2 一致(认证 R 0.05→0.041 为拉入轨迹微移的整数帧计数差,非退化;keep 判据余量巨大) |
| sub8_largebox_000 | keep | **keep** | 接触 0.963/0.988、认证 0.05/0.00、task p95 0.0723——一致 |
| sub12 | 拒 t=11 | 拒 t=11 | 同帧同因(`state_constraints`) |
| sub16 | 拒 t=22 | 拒 t=22 | 同帧同因 |
| sub3 | 拒 t=1 | 拒 t=1 | 同帧同因 |
| sub7 | 拒 t=24 | 拒 t=24 | 同帧同因 |

- **接受率 2/6,与 F1 前持平**——与 §2 机理修订一致:F1 修的机制不在失败路径上。
- 计时:v3 各段 ms/f 为 v2 的 1.027-1.038 倍;本次运行 `--verbose` 开启(逐帧记录+打印),非 F1 计算量(margin 复用既有函数,O(1)/线性化)。§5.3 基线(82-630ms,f)内。
- 测试 102 → **103 项全过**(新增走廊-band margin 对齐测试;走廊测试期望值按 F1 更新)。

## 4. 方案 §5 验收对照

| 条款 | 结果 |
|---|---|
| 5.1 先导 ≥4/6(最低 3/6) | **未达成(2/6)**;剩余失败类别已明确并有逐帧诊断日志(§2) |
| 5.1 已成功 2 段保持 | **达成**:sub10/sub8 keep,质量指标不退化 |
| 5.2 不改 strict 容差 | **达成**(零容差改动;accept 地板/带值全部原样) |
| 5.2 走廊/margin 单测 | **达成**(对齐测试锁定同一公式) |
| 5.2 F1/F3 分离提交 | F1 独立提交;**F3 无代码**(P1 检查点取消,见 §1) |
| 5.3 单帧耗时 ≤ 基线 | **达成**(增量 ≤3.8%,verbose 开销) |
| 5.3 successful 段平均迭代 ≤35 | 未达标(迭代仍常满 40+40)——方案已注明不阻塞,留 §7 技术债 |

## 5. 后续建议(按证据修订,替代方案 §4 P2)

方案 §4 的 P2(F2/F4)经机理评估不实现(§1)。要达到 ≥4/6,需针对 §2 机理的新设计,**均未实现、需另行立项**:

1. **探测引导的 pair 激活**(对 sub16/sub12 大步弦线盲区):转移探测/终检发现的违规 pair 加入该帧后续线性化的 solver 行,使 QP 能"看见"弦线扫掠。方向在方案 §2.1 假设 3 内。
2. **发散感知跟踪**(对 sub12):z_prev 与 z_pred 站点距离超过阈值(如 0.3m)时,降低 prediction/hand 跟踪权重或显式进入"保持+缓慢跟随"模式,避免每帧不可行→slack 零运动→大步追赶的振荡。
3. **regen 输入的预注册速度筛查**(对 sub7/sub12):现有 20/312 速度跳变排除逻辑仅按教师 qpos 速度;sub7 手部贯穿(sub12 滑步)在**站点/几何**层面跳变,应把"逐帧站点运动超限"纳入 regen 输入筛查并在报告中显式排除,而不是让修复器在不可复现帧上失败。
4. **t=0 走廊下界/软拉入**(对 sub12 诱因):区间首帧硬 gap 允许跟踪目标把手续压到 −1.5mm 地板;给首帧行一个下界(如 ≥0mm)或把 t=0 上界分帧收紧,避免刀锋起始。
5. 手部速度硬行、求解器稳态化(方案 §7 原有条目)保持不变。

## 6. 关联

- 方案:`docs/plans/teacher_regen_robustness_plan.md`(修订记录已加)
- 前序:`docs/verification/teacher_regen_verification.md`(先导 2/6、slack s≥0、dip 余量、手部门重标)
- 产物:`derived/data/teachers_regenerated/pilot_diagnosis6_v3_f1/`(含逐段 `frame_log_*.json`)
- 诊断脚本:机制已并入 `--verbose`;探索期脚本未入库
