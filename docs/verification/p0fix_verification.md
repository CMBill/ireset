# P0 审阅修复(F1–F4)验证记录

> **订正指针(2026-09-24)**:本文的 strict 6/6、relaxed 22/24、任务门 0.113 等数字为 P0-fix 版本的历史结果;`skip_reason`/最后失败归因、运行级成本与安静/verbose 成本口径受 2026-09-23 审阅 R5/R6/R8 影响,已由[实现审阅修复验证记录](../archive/verification/m2_m3_stage_gate_review_fix_verification.md)替代。relaxed 产物仍不计入 §10.1,阈值未改。

日期:2026-09-14。依据 [p0_fix_plan.md](../archive/plans/p0_fix_plan.md) 实施;本文记录四项修复的实现、回归测试与全部重测结果(阶段 A–I)。旧 P0 数字(strict 2/6、relaxed 4/6=66.7%、24 段 17/24=70.8%、P0.3 闭环 12.5–25%)全部降级为 **修复前实现的结果**,保留于 [p0_verification.md](../archive/verification/p0_verification.md),不得与本文数字混用。relaxed 产物仍不计入 §10.1 达标。

## 1. 修复内容与代码位置

### F1 手部准静态推断改用箱体系速度(P1)

- `repair/validation.py::_infer_robot_schedule`:手部进入门速度源由世界系 `hand_pts` 差分改为**箱体系 `hand_local` 差分**([validation.py:373-386](../../src/ireset/repair/validation.py))。旧实现中手随箱平移(搬运的定义)使世界系手速恒超 0.02 m/s 门,**搬运相抓握区间被构造性清零**,手部检查在全部 P0 运行中形同虚设;`_check_hands` 的锚漂与相对速度检查本身就在箱体系上,推断与检查自此同系。
- `ContactSchedule.stats` 新增(`embodied/contact_sites.py`):推断的手部区间数/帧数随 schedule 输出,报告可辨识"零区间"与"检查通过"的区别。
- 脚部保持世界系(地面是世界固定的),不变性只要求手部满足。
- 两处 yaml 中"教师手不锚定、无约束力"注释改写为 F1 后语义:门可生效,教师数据因悬浮(gap p95 33mm > enter 10mm)仍少有区间——数据本性(Path B),非永久无约束。

### F2 转移检查使用完整前后场景(P2)

- `repair/scp.py::FrameContext` 新增 `box_pose_prev`(`pipeline._frame_ctx` 传 `prediction[t-1, 36:43]`,t=0 为 None)。
- `pipeline.frame_valid`:前场景 = `z_prev + box_pose_prev`,后场景 = 候选 + 当前箱姿;探测点经终检 `_interpolate` 得到**插值箱姿**并以 `replace(ctx, box_pose=mid)` 检查(旧实现把插值箱姿算出来后丢掉,继续用当前帧箱姿);细分数量沿用终检 `_frame_motion_m` 与 `check_interpolated_states` 开关;**中间态只检查碰撞族行**(0:29 限位与速度/锚点/信赖域行不施加于中间态,与终检探测范围一致);每个探测点重扫碰撞对(端点候选集看不到中途入距的对),扫描后恢复端点集供求解器使用。
- `validation._frame_motion_m` 改为**箱体角点扫掠**(8 角点最大位移,涵盖箱心平移):纯旋转箱扫此前运动量≈0、完全不细分,终检同样存在此盲区,现两侧一致。
- 终检健全性说明:最终验证器本就对 43 维全场景插值(含箱体运动),F2 修复的是顺序预检与终检的一致性及重试效率,不是最终门漏洞;旋转盲区(上一条)是终检真实盲区,已修。

### F3 结构化失败定位(P2)

- `CheckResult` 末尾新增 `first_failure_frame: int | None`(兼容现有位置构造);各检查记录**首次越界帧**:速度族归属差分后帧,插值碰撞归属区间右端帧,脚漂移逐帧相对区间起始锚,手锚漂移逐帧相对 `iv.anchor`(不再定位到区间起点);`location` 字符串保留最差帧供人读。
- `pipeline._first_failure_frame`:对全部失败检查的结构化帧号取**最小值**(旧实现按检查字典插入顺序返回第一个带 `t=` 的失败项,顺序决定重试起点);失败但无定位 → 返回 None → **保守全段重试(start=0)**(旧实现退化为末 5 帧重试,对中段漂移违例必然失败)。
- `failed_checks`(name → location + first_failure_frame)写入重试诊断;`repair_trajectory` 外包一层计时,`total_s` 覆盖两次尝试与全部终检。

### F4 修复后闭环评测与可追溯产物(P1)

- 新增 `scripts/closed_loop_metrics.py`(纯函数,无 MuJoCo 依赖):`paired_task_mpjpe`(共同接受集配对,n=0 未定义,指标缺失计 `n_metric_missing` 并使条件未通过)、`hand_f1_report`(接受集原始值 + 全集成功加权(拒绝记 0),双空未定义不填 1;**既无定义样本又无拒绝样本时加权值为 None**而非 0.0)、`summarize_method`(P0.3 三条件:接受率≥基线、配对 MPJPE≤基线+2cm、成功加权手 F1≥0.95×基线;任一缺失/未定义→该条件与总判定不为 True)、`metadata_block/metadata_compatible`(代码版本/模式/门槛/输入/种子/模型校验值精确匹配才可复用基线)。
- `eval_generation.py` 重写:裸输出指标全部归入 `raw_metrics` 并明确命名 **teacher_fit**(机器人 FK vs 教师 FK 拟合误差);接受轨迹计算 `task_mpjpe_box_m/world_m`(**G1 对应关键点 vs 参考人体关键点**,箱体系为主口径、世界系与修复器帧门公式完全一致作对拍)、重推断手部 F1、修复幅度(根平移/旋转、关节 RMS/max)、slack 帧数/占比、端到端及分位计时;拒绝样本保留计时、**不产生位置误差**;新增裸教师基线(无噪声教师作修复输入,每 demo 修复)与 `--teacher-baseline-json` 元数据校验复用;产物默认名带模式与门槛(`generation_eval_{mode}_tol{gate}.json`),含 git 版本、输入清单、筛查交集、模型校验值元数据。
- `eval_repair.py`:计时补 p50/max 与 `total_s`;slack 帧数/占比(来自 `slack_flags`);默认产物名带模式门槛;写入元数据块。
- **slack 独立记录**(F4.7):`pipeline._run_sequential` 以 `res.slack_used` 逐帧记录 `slack_frames`——多起点 `repaired_prev` 与回溯 `repaired_bt` 路径名不再丢失 slack 事实(旧行为:路径名覆盖导致少计)。

## 2. 回归测试(阶段 A–D)

测试 84 → **99 项全过**(`uv run --locked pytest tests/ -q`)。新增 15 项:

| 测试 | 覆盖 |
|---|---|
| `test_hand_grasp_survives_common_translation` | 审阅探针复现:0.3 m/s 共同平移下抓握区间不再消失,区间/箱体系锚/统计与静态一致(F1) |
| `test_hand_grasp_survives_common_rotation` | 绕箱心刚体共同旋转,`hand_local` 不变,推断与静态一致(F1) |
| `test_slow_box_frame_drift_rejected_by_anchor_gate` | 0.015 m/s 面内滑动(<推断门与相对速度门)累计锚漂 15mm>10mm 必拒;首越界帧独立重算对拍、与最差帧/区间起点可区分(F1+F3) |
| `test_foot_slide_reports_first_exceedance_frame` | 慢滑动脚漂移首越界帧解析对拍(F3) |
| `test_transition_check_catches_box_sweep` | 箱平移穿过机器人:端点净空、中途碰撞被顺序预检拒绝(F2) |
| `test_transition_check_catches_box_rotation_sweep` | 箱原地 pitch 180°:端点各 ~8mm 净空、120° 探测点侵入 33mm;角点扫掠 0.73m 必须触发细分(F2 旋转运动量) |
| `test_transition_check_common_motion_no_false_positive` | 机器人+箱共同平移 0.055m/帧,插值点相对构型不变,无虚假碰撞(F2) |
| `test_first_failure_frame_is_min_over_failed_checks` | 字典插入顺序无关取最小;通过项/无定位项不贡献(F3) |
| `test_unlocated_final_failure_retries_full_segment` | 无定位失败→全段重试(start=0);有定位→帧−窗口(F3) |
| `test_slack_flag_recorded_independent_of_path_name` | 多起点 slack 解在 `repaired_prev` 路径名下 slack_frames 仍计数(F4.7) |
| `test_paired_comparison_uses_common_accepted_set` | 方法 {A,B} vs 教师 {A,C}:n=1;无交集未定义(F4.4) |
| `test_rejected_samples_stay_in_denominator_without_position_error` | 拒绝样本进分母、无位置误差;共同接受样本指标缺失→条件未通过(F4) |
| `test_hand_f1_dual_empty_undefined_and_success_weighted_zero` | 双空不填 1;拒绝/未定义加权记 0(F4.5) |
| `test_conditions_require_complete_evidence` | 三条件需完整证据;任一缺失/未定义→不判通过(F4.6) |
| `test_metadata_reuse_requires_exact_match` | 元数据五要素精确匹配才可复用(F4.3) |

脚部 pivot/slide 既有慢速用例、成对模式不变式等原有测试全部保持通过。

## 3. 阶段 E–G 重测(教师侧)

所有新产物以 `p0fix_` 前缀 + 模式/门槛命名,旧产物未覆盖。

### E:旧接受轨迹复检(`p0fix_recheck_accepted.json`)

- `p02_calibration_traj`(17)+ `p0_trajectories_relaxed`(4)共 21 段已保存接受轨迹在新终检(F1 激活 + 角点细分)下 **21/21 仍全部通过**——无错误接受需撤回。
- **21/21 段手部推断区间为零**:F1 修复消除了坐标系障碍后,教师手部悬浮(距箱 1–3cm,grasp-gap p95 33mm)仍过不了 10mm 进入阈。手部检查在教师数据上依旧空转,确证悬浮为**数据本性**(Path B 教师重生成的核心论据),语义修复无法解决。

### F:6 段诊断集(`p0fix_repair_6seg_{strict,relaxed}_tol0.400.json`)

| 模式 | 修复前(P0) | 修复后 | 说明 |
|---|---|---|---|
| strict | 2/6 = 33.3% | **6/6 = 100%** | 独立 strict 验证器复检 6 条轨迹全部真实通过(worst 见 JSON) |
| relaxed | 4/6 = 66.7% | **6/6 = 100%** | 同上 |

- 跃升主因是 **F3**:P0 的 strict 失败均为终检脚部漂移违例(当时无 `t=` 定位)触发退化的末 5 帧重试,前缀违例原样保留、重试必然失败;结构化首帧 + 全段/定点重试使违例区被真正重解。F2 的探测范围收敛与角点细分也参与(更早发现、更准重试)。
- **手部认证首次真实生效**:sub7_largebox_000 在 strict 下形成 left_hand 区间并通过手部检查(worst 0.637)——几何真正贴近时 F1 门可建立锚。
- 计时如实:sub7 168–219 ms/f(p95 >1s,多起点+回溯主导),sub10 21.7 ms/f;slack 0–19 帧/段。计时不因接受率上升而被隐藏。

### G:24 段标定集(`p0fix_repair_24seg_relaxed_tol{0.400,0.101}.json`)

| 任务门 | 接受率 | 备注 |
|---|---|---|
| 0.40(工作值) | **22/24 = 91.7%** | 修复前 17/24=70.8%;仅 sub10_003、sub8_001 帧级拒绝(与修复前相同的两段深失败) |
| 0.101(固定旧门) | 16/24 = 66.7% | 6 段在旧门下翻拒——固定旧门在新接受集上的影响已量化 |

- ms/f 均值 112.3、均值 p95 276.9;slack 占比均值 0.080、最大 0.333(逐段见 JSON)。
- **任务门重标定**(满足登记集覆盖:同登记 24 段、接受 22≥17):22 段/2,599 帧 p95=0.0927 + 0.02 → **`mpjpe_tol_m = 0.113`**,写入 `configs/eval/task_quality_gates.yaml`(0.101 降级为 superseded 固定对照,标定历史保留于 yaml 与本文)。选择偏置声明同前(标定于 relaxed 接受集,低估全分布尾部)。

## 4. 阶段 H:8 val 闭环(P0.3 口径重跑)

`generation_eval_relaxed_tol0.400.json`(主对照)与 `generation_eval_relaxed_tol0.101.json`(固定旧门诊断变体);8 个 val original(sub10_largebox_002/022/024/060/088/093/094/100,与筛查排除集零交集),seed 7,DDIM-25,120 帧;元数据记录 git rev a332001(**运行时代码为该提交之上的未提交工作树**,以 `p0fix_` 产物前缀区分)、模型校验值与输入清单。接受轨迹 NPZ 存于 `p0fix_gen_traj_relaxed_tol0.400/`。

### 门 0.40(主对照)

| 方法 | 接受率 | 配对 n | 配对任务 MPJPE 差(对教师基线) | 成功加权手 F1 | 接受率≥基线 | 配对≤+2cm | F1≥0.95× | 全条件 |
|---|---|---|---|---|---|---|---|---|
| 裸教师基线 | 3/8 = 37.5% | — | —(任务 MPJPE 均值 0.077m) | 0.0 | — | — | — | — |
| teacher_noise_0.02 | 3/8 = 37.5% | 2 | **+0.0036m ✓** | 0.0 | ✓ | ✓ | ✓(0=0) | **全部满足** |
| teacher_noise_0.05 | 1/8 = 12.5% | 1 | +0.0367m ✗ | 0.0 | ✗ | ✗ | ✓ | 否 |
| 扩散 DDIM-25 | 1/8 = 12.5% | 1 | **+0.0042m ✓** | 0.0 | ✗ | ✓ | ✓(0=0) | 否 |
| 确定性 | 2/8 = 25% | 1 | +0.0780m ✗ | 0.006 | ✗ | ✗ | ✓ | 否 |

- **配对任务 MPJPE 用修复后接受轨迹、共同成功集、机器人关键点 vs 参考人体关键点(箱体系)**——与 P0.3 报告的口径(裸输出、对教师 FK、8.2/11.0-11.1cm)已彻底分离。例:扩散唯一共同接受段 sub10_100,裸输出教师拟合 0.082m vs 任务口径 0.112m,两类数值不可互换。
- **接受率条件是当前约束**:扩散/确定性(12.5%/25%)低于裸教师(37.5%),关系与 P0.3 一致;而噪声 σ0.02 输入**三项条件全部满足**——协议本身可达,约束在生成质量(4k 步未饱和 + 教师悬浮数据),不在修复机制。
- 扩散被接受样本的任务质量与教师基线差 4.2mm;确定性差 7.8cm(其接受段 sub10_002/088 修复幅度更大)。
- 手 F1:全方法(含教师)≈0——参考侧声称抓握、机器人侧区间为零(悬浮),F1=0 为有定义的召回失败,0≥0.95×0 判真;该指标在教师重生成前无区分度,如实报告。
- 计时/slack 如实:接受段 61–523 ms/f、slack 11–63 帧(噪声/生成输入修复量大;sub10_094 教师段 1034 ms/f 后拒绝);端到端与分位计时见 JSON 逐行。

### 门 0.101(固定旧门,诊断变体)

裸教师 2/8=25%;**其余全部方法 0/8**,配对 n=0、条件未定义——旧门对噪声/生成输入结构性过紧(P0.2 已实测的选择偏置在新接受集上进一步放大:24 段教师侧 0.101 也只有 16/24)。0.101 仅作固定对照保留;前向值为重标的 0.113,工作值 0.40。

### 与修复前 P0.3 数字的关系

裸教师 25%→37.5%(F3 重试修复);扩散/确定性接受率不变(12.5%/25%——模型与输入未变,符合预期);噪声基线 12.5%→37.5%(σ0.02)/12.5%(σ0.05)。

## 5. 结论与遗留

**已确立:**

1. 四项修复全部落地并有回归测试锁定;测试 84→**99 项全过**;**未改任何容差数值**——接受率跃升(strict 33.3%→100%、relaxed 70.8%→91.7%、教师闭环 25%→37.5%)全部来自语义正确性:F3 重试定位是 P0 期的主要阻塞(无定位失败触发必然失败的末 5 帧重试);F1 使手部认证在几何贴近时真实生效(sub7 strict 实证);F2 关闭了顺序预检与终检的场景不一致及**终检自身的纯旋转细分盲区**(角点扫掠,两侧一致);F4 使闭环结论建立在配对任务口径上。
2. 旧产物零撤回:21/21 已保存接受轨迹新终检仍过;旧报告数字降级为修复前实现结果并加版本声明。
3. 教师手部悬浮是数据本性:全部教师侧运行手部推断区间为零(21 复检段 + 重跑接受段,唯一例外 sub7)——**Path B(教师重生成)必要性进一步增强**,语义修复已到上限。
4. 任务门重标 0.101→**0.113**(22 段/2,599 帧,p95 0.0927+0.02;选择偏置声明保留;0.101 作 superseded 固定对照)。
5. 闭环判定(门 0.40):扩散/确定性接受率低于裸教师基线,条件不满足;σ0.02 噪声输入全条件满足,证明协议可达、约束在生成质量。**relaxed 产物一律不计入 §10.1 达标。**

**遗留与后续:**

- 生成质量约束:扩散被接受样本质量达标(+4.2mm),但接受率低 → M3 训练延长(4k 步未饱和)+ 教师重生成后重训。
- M2 收尾基线再次变化(扰动集与计时需在 F1–F4 语义下重测;本轮已给出教师侧 6/24 段新基线与计时/slack 全量记录)。
- strict 模式闭环未跑(P0.3 协议基于 relaxed;strict 闭环留待教师重生成后统一评估)。
- 运行时代码未提交,产物 git rev 记录为 a332001+工作树,以 `p0fix_` 前缀与文档区分;提交后新运行自动携带新 rev。
