# M2-M3 收尾方案实施验证记录(阶段 0-3)

日期:2026-09-16。方案:[m2_m3_completion_plan.md](../plans/m2_m3_completion_plan.md)。状态:**阶段 0/1/2 已实现并测试,阶段 3 重测已完成;阶段门未达标(2/6 < 4/6),按方案 §7 门禁停止扩展,不进入阶段 4 的 24/312 段规模化**。strict 容差数值零改动;任务门保持 0.40 工作值。

## 1. 阶段 0:统计缺陷与证据固化

三项已核实的统计缺陷全部修复(提交 `82d117a`):

| 缺陷 | 修复 | 验证 |
|---|---|---|
| `scp.py` 三条返回分支硬编码迭代数 `scp_max_iters`(40) | `FrameResult` 记录 `it_done`(真实迭代)与新增 `qp_calls`(全部 QP 求解数),所有分支如实填写 | sub7 前 26 帧冒烟:迭代 20/34 真实值;诊断 dict 新增 `qp_calls` 历史 |
| 早退分支漏记 `slack_used`(restoration 后早退误报未用) | `used_slack` 进入全部返回分支 | 同上冒烟:restoration 后早退 `slack_used=True` |
| `ms_per_frame` 分母恒为整段帧数(提前拒绝段耗时被摊薄) | 报告 `ms_per_processed_frame`(实际处理帧,pipeline `timings_s` 口径)+ `ms_per_frame_p95` + `ms_per_input_frame`(旧口径保留并注明);另记录 `screening_s`、`iterations_sum`、`qp_calls_sum` | sub7 前 26 帧:25/26 帧处理、165.1ms/处理帧(旧口径 ~159ms/26 帧摊薄) |

尝试级诊断(方案 §3 第 3-4 项):`frame_log` 新增 `skip_check`(预测帧跳过检查的 ok/原因)与 `attempt`(每次求解:t、来源 prediction/prev_accepted/slack_fallback、回溯类型、accepted、status、真实迭代、QP 调用、slack、结果 frame_valid)记录;`solve_frame` 的成功返回、预算耗尽、信赖域坍缩分支均被覆盖。逐段 JSON 与打印保持既有 `--verbose` 入口。

历史文档归因更正(`teacher_regen_robustness_verification.md`,保留原数字与版本):

1. **sub7"教师手部贯穿箱体"归因修正**:原始教师右手箱距 t=23/24 为 +55.708/+52.571mm(未穿透);约 −23mm 来自求解候选,该 pair 未进入当时的冻结候选集合。
2. **sub3 深度修正**:t=0 合法(双手 +31.9/+68.3mm);t=1 双脚原始箱距 −0.574/−0.622mm(浅穿透,非早期记录的"手部 −1.4mm 深穿透");候选另有脚部端点/插值违规及右手约 −20.8mm 穿透。
3. **性能措辞修正**:v3 相对 v2 的 1.027-1.038 倍增量不能全部归因 verbose(统计口径缺陷当时未修);sub10 认证覆盖变化按"保留状态不变"与"质量指标变化"分开表述(4/74→3/74)。
4. 已保留段零退化与整体 keep 状态分开报告(方案 §2.2)。

## 2. 阶段 1:动态碰撞对激活(提交 `d2e1fae`)

原实现候选对集合在 `solve_frame` 首次线性化时按 5cm 门扫描并冻结,QP/merit/线搜索/成功返回全部消费冻结集合;端点/探测新扫的 pair 只能拒绝、不能回流。

- `FrameSolver._scan_pairs(full)`:全量注册表(1107 对,实测 ~1.4ms)或 partial(快族 robot_box/robot_ground ~100 对 ~0.1ms + 携带 pair + 仍在门内 self 对)。
- `note_discovered_pairs`/`clear_discovered_pairs`:检查发现违规 pair 回流,下一线性化无论当前距离都进候选集;帧边界清除(帧内跨尝试保留)。
- `solve_frame`:每次接受步调度 partial refresh;**成功返回前全量重扫并复核**,新扫描暴露违规则继续迭代,不再返回未检查状态。
- 行预算截断重写:已发现 pair 优先于最近排序保留,溢出记 `last_truncated` 上报,不静默丢行。
- `pipeline._carry_back_discovered_pairs`:端点与探测(含 verbose=False)违规行回流。
- 测试 4 项新增(回流、预算、成功前重扫、穿箱端点拒绝+多起点收敛),109 项全过。

## 3. 阶段 2:插值走廊语义(提交 `1a8e330`)

走廊上界是帧末(完整 dt 闭合)目标;插值探测此前原样继承,把合法接近在中间态误拒(方案 §2.2 实测:30mm→24.178mm 端点过、27mm 中点沿用帧末上界被拒)。

- `_collision_rows(..., probe_mode)`:探测模式下 regen 行保留防穿透下界 `band_lo`(与验证器中间态同口径)、上界放开(INF);`constraint_rows` 透传。
- 探测调用改传 `probe_mode=True`;端点与求解器路径不变(帧末走廊照常约束)。
- 测试:探测行集保留 regen 行且上界 INF;30mm→24mm 合法接近(区间 t=1 起,走廊 24.5mm)端点+中点全过;真实中间穿透拒绝用例保持。

## 4. 阶段 3:六段 strict 重测(v4,阶段 1+2 组合)

`derived/data/teachers_regenerated/pilot_diagnosis6_v4_stage12/`(git `1a8e330` 起,全长度,strict,`--verbose`):

| 段 | v3 t_fail | v4 t_fail | v4 状态 |
|---|---|---|---|
| sub10_largebox_000 | keep | keep | **接受/保留**,task p95 0.0756(与 v3 一致);接触 1.00/0.905→0.97/0.838;认证区间 1/1→**0/0** |
| sub8_largebox_000 | keep | keep | **接受/保留**,task p95 0.0723(一致);接触 0.963/0.988→0.963/0.975;认证 0.05/0.00(不变) |
| sub12_largebox_001 | t=11 | **t=2** | 拒;教师 120mm/帧滑步下弦线对 `left_rubber_hand~largebox` 中态 −27mm,回流后三起点仍不可行 |
| sub16_largebox_000 | t=22 | t=22 | 拒;自碰撞对 `right_knee~right_rubber_hand` 中态 −4.0..−5.3mm,回流后端点可达 −5.3mm 仍超 −1.8mm 门 |
| sub3_largebox_000 | t=1 | **t=11** | 拒;脚部问题已消(v3 的 t=1 脚部失败被阶段 1/2 修复),现败于右手 regen 行 −2.62mm(< −1.5mm 地板),三起点不可行 |
| sub7_largebox_000 | t=24 | **t=34** | 拒;右手 regen 行端点 8.78mm 超上界 8.0mm,三起点 infeasible(34 帧内 pair 回流持续参与) |

**接受率 2/6,阶段门(≥4/6)未达成。**

- **修复有效的直接证据**:sub3 失败从 t=1 推进到 t=11(前 10 帧全部通过,含 v3 败因脚部);sub7 从 t=24 推进到 t=34(拉入完成、抓握保持 10 帧);sub12 从拖到 t=11 变为 t=2 诚实快速拒绝。
- **失败阶段一致化**:四段失败全部收敛到 `state_constraints`(端点),无一为 `transition_interpolated`——插值走廊误拒类别已消除(阶段 2 目标),剩余失败是求解器在大步/贴边帧的可行性问题(阶段 4 条件项范畴)。
- **质量代价必须如实报告**(方案 §6"任何下降明确报告"):sub10 接触覆盖 R 0.905→0.838(74 帧中 5 帧失去物理接触)、认证区间 3+2/74+33 帧→0;sub8 接触 R 0.988→0.975、认证不变。task p95 两段均不变。认证归零的方向与方案 §7"几何接近但认证覆盖低"一致:无中间态走廊后拉入更快、箱体系速度抖动增大,准静态进入门(0.05 m/s)更难触发。**该退化使 v4 保留段的可用性弱于 v3**(认证区间是筛选报告项,keep 判据用的是物理接触覆盖)。
- 计时(新口径,处理帧):sub10 543.8ms/proc-f、sub8 82.6、sub16 88.2、sub3 176.0、sub7 330.3、sub12 534.3(3 帧内三次尝试耗尽)。早期拒绝段不再摊薄。

**标准模式回归检查**:`eval_repair.py --teacher-only --validation-mode strict`(P1 六段口径,git `1a8e330`):**4/6 接受**(sub10/sub16/sub3/sub8 过,sub12/sub7 拒),与 slack 修复后基线(`regen_impact_6seg_strict.json`,git a332001,4/6,同批段)一致——共享修复逻辑未引入标准模式回归。

## 5. 阶段门判定与后续

方案 §6/§7 阶段门:3/6 仅作中间进展,扩展目标 ≥4/6;**2/6 时不进入 24/312 段规模化,继续阶段 4 按剩余证据分流**。本轮不实现阶段 4 条件项(属"条件性工作,不默认全部实现",且 sub12 类跟踪策略与 sub3/sub7 类 regen 行边界需要单独设计对照),留待下一轮:

1. sub12:累积跟踪偏离的量化与策略对照(方案 §7 行 2)。
2. sub16:中间态自碰撞弦线——端点回流已达 −5.3mm 但中态仍超门,需探测点约束/插值状态增量约束(方案 §7 行 1,须雅可比对拍)。
3. sub3/sub7:regen 行贴边(−2.62mm / 8.78>8.0)的三起点不可行——与 §5(鲁棒性验证)第 4 项 t=0 走廊下界/软拉入同族,需单独消融。
4. sub10 认证覆盖退化的稳态化目标(方案 §7 行 5),在 24 段标定前须恢复至 v3 水平或给出可用性结论。

## 6. 提交与产物

| 提交 | 内容 |
|---|---|
| `82d117a` | 阶段 0 统计修复 + 尝试级日志 |
| `d2e1fae` | 阶段 1 动态碰撞对激活 |
| `1a8e330` | 阶段 2 插值走廊语义 |
| 本提交 | 文档 + 方案勾选 + AGENTS 摘要 |

产物:`pilot_diagnosis6_v4_stage12/`(含逐段 frame_log);标准模式:`/tmp/std_mode_6seg_v4.json`(临时,汇总数字已录本文件)。测试:**109 项全过**。
