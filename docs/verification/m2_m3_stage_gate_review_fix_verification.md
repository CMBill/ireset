# M2-M3 实现审阅修复验证记录(R1-R9、历史重算、统一版本六段重测)

日期:2026-09-24。方案:[m2_m3_stage_gate_review_fix_plan.md](../plans/m2_m3_stage_gate_review_fix_plan.md)。上游:[m2_m3_stage_gate_fix_verification.md](m2_m3_stage_gate_fix_verification.md)(第十轮 5/6、标准模式 6/6 与 2 项覆盖偏差)。

> **2026-09-26 订正(优先于本记录中的下列表述;证据与订正过程见 [m2_m3_label_quality_train24_verification.md](m2_m3_label_quality_train24_verification.md) §2):**
> 1. §7 表"已保留段质量"行与本页状态段的"认证交集:sub10 左 3/右 15、sub8 左 29/右 41,全部 ≥ v3"**错误**:sub10 左的 3 是**认证总数**(`certified_frames=3`),其**认证与参考区间交集为 0 帧**(`certified_in_ref_frames=0`,3 帧认证区间落在参考抓握窗口之外;v3 为 3)。来源:`derived/data/teachers_regenerated/quality_tables/six_segment_comparison_reviewfix_20260924.json`(nq/nv 行)及 `segment_quality_report.py` 对 `reviewfix_20260923_nq_strict_task040/sub10_largebox_000.npz` 的独立复算。因此统一版本相对 v3 的质量退化是 **3 项**(sub10 左交集 3→0、sub10 右覆盖 −5.4pp、sub8 右覆盖 −1.25pp),不是"仅两项偏差";"认证交集全达标"作废。
> 2. §6 的"2 项偏差**不可能**由稳定软目标恢复"**撤回**:结论限定为"已测试的 0/0.1/1.0 三种权重未恢复几何接近覆盖"(三权重下 `frames_within_enter` 逐帧相同是事实证据;"不可能"与普遍无关性是未被证明的推断)。
> 3. §4.1/§4.3 的 sub7 拒绝证据 `joint=12.00rad/s` 为舍入显示;实际值 12.000000401362618 rad/s(超限 4.0136e-7 rad/s,帧级违反度 ~3.3e-8,严格终检仍拒绝),结构化未舍入证据由标签分级方案阶段 A 补齐(同上验证记录 §3)。

状态:**R1-R9 全部实现并分项提交,`uv run --locked pytest tests/ -q` 165 passed(本轮新增 31 项,其中 R1/R2/R3 反例各自能识别原缺陷);D 对照确认诊断类修复零行为差异(逐段状态/计数一致、5/5 保留 NPZ 逐字节相同);统一版本 Nq/Nv 六段 5/6 keep(sub12 恢复接受、sub7 转为终检速度拒绝),Nq 与 Nv 除耗时外逐项一致,Ns 标准模式 6/6 与旧标准逐段核对(3 段 QP 迭代数因探测行纠正而变、质量指标同值);历史 R4 受影响字段已逐帧重算(27 行对照),旧 JSON/NPZ 未被覆盖;S01/S10 已在统一版本重跑——两个权重都不改变几何接近覆盖、且分别破坏 sub10 左与 sub8 右认证,保留权重 0。数量门达标(≥4/6、sub10/sub8 保留、keep 全部过独立 strict 与几何筛选);“每侧几何接近覆盖不低于 v3”仍有 2 项偏差且与第十轮同值,按方案 §8 不自动转 train24。** strict/keep/任务工作门 0.40 与全部容差零改动;未运行 train24/312 重生成与重训。

## 1. 提交清单(可审阅分项)

| 提交 | 内容 | 反例/回归 |
|---|---|---|
| `c991f6b` | R5/R7/R8:失败证据、违反行描述符、尝试事件定稿 | `test_repair_contracts.py` 12 项 + `tests/test_cost_contracts.py` 4 项(R6) |
| `12c5741` | R4/R9:逐帧 MPJPE 序列与实例有效配置溯源 | `tests/test_report_contracts.py` 6 项 |
| `a9f9857` | `os` 导入补齐 + regenerate CLI 报告冒烟回归 | CLI 冒烟 1 项(实跑) |
| `c8d266f` | **R1**:探测行只用命中查询 | 截断配对无行/零距离行存活/装配行 FD 3 项 |
| `b9fc080` | **R2**:接触记录法向符号 | 脚球-地面四态梯度/倾斜箱面接触记录 2 项 |
| `a22e775` | **R3**:探测缓存全依赖键 | 回流即时生效/身份幂等/依赖覆盖 3 项 |
| `9658b02` | R5 补全:`rejected_final` 拒绝块带结构化失败帧 | `rejection_evidence` 2 路径 1 项 |

`uv run --locked pytest tests/ -q`:**165 passed**(第十轮到本轮 134→165;`tests/test_repair_contracts.py` 25→45、`test_cost_contracts.py` 0→4、`test_report_contracts.py` 0→7)。分项提交时每项均单独跑过该项反例与整套测试。

## 2. R1-R9:缺陷、修复与反例

| 项 | 缺陷(审阅) | 修复 | 反例识别(修复前实测) |
|---|---|---|---|
| R1 | `mj_geomDistance` 未命中时精确返回 `distmax` 且 `fromto` 未写入;旧判据 `dist > dmax` 使远处配对以 1.2 mm 假距离、零长度线段进入 QP | `scp.py::_probe_rows` 判据改为返回距离 `not dist < dmax` | `AssertionError: a truncated query became a probe row`(0.19 m 配对在 α=1/3、2/3 上 dist=0.0012=floor+margin) |
| R2 | `contact.frame[0:3]` 是 geom1→geom2 法向,旧实现按 geom2→geom1 取向,`ga==geom1` 时分离方向整体反号 | `constraints.py::geom_separation_direction` 改为 `-n if ga == c.geom1 else n` | `('shallow penetration', 25) assert np.float64(-1.0) == 1.0`;倾斜箱面 `dot(u, -s·seg) = -1.0` |
| R3 | 探测缓存只有 z 键,回流新增配对或箱姿变化(z 相同)复用陈旧行,刚发现的违规不进下一次 QP | 新增 `_probe_cache_key`(z/前帧/箱姿/前后箱姿/监视身份/候选身份/协议/预算/下界,逐字节快照)、新增身份即时置空、attempt 起始置空、值-only 不冒充雅可比缓存 | 同一 z 改变箱姿后 `sig(rows) == sig(r0)`(旧实现分支未失败);回流后新装配缺某配对;`solve_frame` 未清缓存 |
| R4 | `mpjpe_growth` 读评测器末状态,增长/分位对照全错(sub10 报 −0.2242 m) | `segment_quality_report.py` 改用 `task_mpjpe_series`(逐帧 FK) | 逐帧重算 sub10 growth −0.0025(旧 −0.2242)、q1 0.4179→0.0697(§5 表) |
| R5 | 失败证据绑定首尝试而非最后执行尝试;`rejected_final` 只给自由文本行 | `pipeline.py` 重写尝试归因(`_finalize_attempt`/`last_chk`/最后尝试 + `check_id` 绑定);`regenerate_teachers.py::rejection_evidence` 带 `validator_failed_checks` 结构化帧号 | 修复前 `reason='infeasible'` 与最后尝试的 `task_quality` 不符;`rejected_final` 缺 `_pass_index`/`cost`/`adopted` 键 |
| R6 | 运行级成本只汇总最后一轮(sub12 上报 65 尝试/1815 QP/148 访问,真实 130/3630/296) | `pipeline.py::PassRecord`/`cost_view` schema 2:全部轮次求和 + 每访问分位 + 唯一帧并集;`cost_legacy_final_pass` 原样保留旧口径 | 历史 verbose 报告 `cost.trajectory_pass=1` 且 `suffix_retries=1`(65/1815/148);D 运行同段给出 130/3630/296、唯一 148、T 148 |
| R7 | 违反行描述符跳过 96 条锚点/速度行,真实探测违反漏报、锚点误标为探测 | `scp.py::row_descriptors`/`_violating` 索引全长度描述符数组(前置+端点+后置+探测),数量不符显式抛错 | 修复前对 9 条真实探测违反抽取结果为空 |
| R8 | 尝试事件在检查/回溯前写入,`frame_valid`/`backtrack` 全为 null | 尝试事件在检查与回溯完成后定稿,`check_id` 稳定唯一、同一 `FrameCheck` 不重复记数 | D 运行:48 帧 repaired(31 prediction + 17 prev_accepted)、状态 solved 48/infeasible 10/review_failed 7、无 null 字段 |
| R9 | 有效配置未记录(权重 0.1 实验被记成权重 0/regen_mode false) | `experiment_provenance.py` 有效记录(asdict 往返 + 规范 JSON sha256)+ 脏树真实补丁/未跟踪代码副本 + 运行中漂移检查 | D/Nq 报告 `effective.argv`/`effective_config_sha256`/`is_default_*` 齐全,`screening` 与 `task_mpjpe_tol_m` 实例化记录 |

补充(R1 的伴随测试与网格事实):零距离真实接触同样塌缩 `fromto`,因此判据必须取“返回距离”而非“线段长度”,`test_probe_rows_keep_zero_distance_contact` 固定该边界;装配行雅可比按 `3e-7` 步长 FD 全对拍,因网格-平面查询在“最深顶点切换”处有折点,步长取大时斜率误差(O(step))超出行精度。箱体 `largebox` 为 mesh geom,其 geom 系与体坐标间存在固定重定心旋转(AGENTS 已记录),倾斜接触测试的方向期望改由 MuJoCo 自身的 `fromto`/接触法向给出。

## 3. D 对照:诊断类修复的行为一致性(§7.2)

D 运行代码状态 = `a9f9857`(R4-R9 已修,R1-R3 未修),完整六段 regen/strict/0.40/weight=0,产物 `derived/data/teachers_regenerated/reviewfix_20260923_d_strict_task040/`(含 `baseline_freeze.json`:代码 rev、六段输入 sha256、历史产物 sha256)。

- **逐段状态一致**:六段 accepted/status/t_fail/skip_reason/final_stage/slack/fallbacks 与冻结最终版逐项相同;5/5 保留 NPZ 与 `fix_20260922_final_regen_quiet` **逐字节相同**(sub10 `1118bd56…`、sub16 `52d8770e…`、sub3 `6bfdb73c…`、sub7 `ac190b3a…`、sub8 `7e5e0a32…`)。
- **R6 证据**:sub12(两轮)总计 attempts 130、qp 3630、visits 296、unique 148、T 148,`cost_legacy_final_pass` 精确等于旧口径 65/1815/148。
- **R5/R8 证据**:全部尝试事件带 `check_id`、无 null `frame_valid`;拒绝绑定最后执行尝试;sub12 拒绝块现含 `validator_failed_checks.velocities.first_failure_frame=3`。
- **R7 证据**:探测违反行带 `pair@alpha=值`,锚点违反归属 `anchor` 族,行数量与值数组不符即抛错(探测行截断另计)。
- **R4/R9 证据**:`effective`/`freeze`/`end` 三块齐全,`end.config_unchanged=true`、`code_changed_during_run=false`;脏树补丁落盘(`dirty_diff_20260924_115135.patch`,内容为当轮文档改动)。

结论:D 与冻结最终版在状态、计数口径(补齐后)与输出内容上一致 ⇒ 诊断/统计/质量/溯源修复不改变求解行为,可安全叠加 R1-R3。

## 4. 统一版本六段重测(§7.2 Nq/Nv/Ns)

命令(两轮完全相同,仅 `--verbose`/`--mode` 不同;未传 `--task-gate-yaml`,实例配置 `task_mpjpe_tol_m=0.4`、`regen_hand_stab_weight=0.0`、`min_hand_recall=0.5`、`min_ref_frames=10`):

```
uv run --locked python scripts/regenerate_teachers.py --set diagnosis6 \
  --validation-mode strict --mode regen [--verbose] \
  --output-dir derived/data/teachers_regenerated/reviewfix_20260923_n{q,v}_strict_task040 \
  --report .../regen_report_diagnosis6_strict.json --note "review fix N{q,v}: ..."
```

### 4.1 数量与拒绝

| 段 | D(R4-R9) | Nq(R1-R9) | Nq keep | Nq 拒绝证据 |
|---|---|---|---|---|
| sub10 | accepted | accepted | ✔ | — |
| sub12 | rejected_final(速度 @t=3) | **accepted** | ✔ | —(旧拒绝为回归证据,方案明确允许变为通过) |
| sub16 | accepted | accepted | ✔ | — |
| sub3 | accepted | accepted | ✔ | — |
| sub7 | accepted | **rejected_final** | — | `velocities` joint=12.00rad/s pos=0.87m/s ang=2.86rad/s,**first_failure_frame=101**(后缀重试自 96 起、两轮同因可复现) |
| sub8 | accepted | accepted | ✔ | — |

Nq/Nv:**5/6 keep**(≥4/6),sub10/sub8 保留,5 个 keep 全部通过独立 strict 终检与几何/接触筛选(`hand_screen=pass`)。与旧 5/6 相比集合变化:sub12 进入、sub7 退出,二者都有本轮实际状态证据(§4.3)。

### 4.2 Nq vs Nv:除耗时外逐项相同

- 5 个保留段 NPZ sha256 完全相同(sub10 `2dc4831a…`、sub12 `70d77b35…`、sub16 `d888cfb8…`、sub3 `566d5987…`、sub8 `d9ead22e…`)。
- 逐段 accepted/status/t_fail/skip_reason/final_stage/slack/fallbacks/task_mpjpe/hand 相同;`cost` 全部计数(attempts/qp/iterations/slack_attempts/backtrack_probes/backtracks_adopted/visits/unique/T/complete)相同。
- **事件级**:433 条 attempt 事件与 1329 条 check 事件逐字段比对无差异(仅 `first_attempt_solve_s` 68.761 vs 68.298 为计时)。
- 耗时独立报告:Nq 各段 mean/p95 见 §4.4;Nv 仅多出 `frame_log_<demo>.json`。

### 4.3 恢复/新增拒绝的机理证据

- **sub12 恢复**:旧拒因为速度终检 @t=3(`joint=12.00rad/s`)。R1 去除 1.2 mm 伪距离行、R3 使回流即时生效后,probe 行只约束真实接近配对,该段单轮即可接受(attempts 73、qp 1927、visits 148、slack 44)。
- **sub7 新拒绝**:单轮完成后速度终检在 t=101 失败(`joint=12.00` 达上限、根 0.87 m/s、角 2.86 rad/s),后缀重试自 96 起、两轮同因同帧,属**可复现的真实失败**而非证据缺失;根因是该段在修正后仍存在瞬时快步/摆动,留作下一轮设计输入(不得改阈值)。

### 4.4 成本

| 段 | Nq wall s | attempts | qp | visits | unique | mean ms/visit | p95 ms/visit |
|---|---|---|---|---|---|---|---|
| sub10 | 39.8 | 63 | 1233 | 120 | 120 | 319.9 | 780.9 |
| sub12 | 54.4 | 73 | 1927 | 148 | 148 | 359.3 | 1351.4 |
| sub16 | 12.4 | 18 | 493 | 89 | 89 | 131.0 | 645.6 |
| sub3 | 50.0 | 81 | 2014 | 118 | 118 | 415.3 | 798.4 |
| sub7 | 87.4 | 119 | 2667 | 292 | 146 | 295.8 | 1123.3 |
| sub8 | 34.9 | 79 | 1416 | 129 | 129 | 262.6 | 725.1 |

六段合计 278.9 s(Nv 276.3 s),低于历史 370 s 参考值——该参考含重试墙钟且受旧代码/机器负载影响,**不作为性能承诺**;sub7 的 visits 292 > unique 146 正是后缀重试的两轮合计(R6 语义)。

### 4.5 共享回归 Ns(标准模式,旧 `fix_20260922_final_standard_strict` 逐段核对)

- **6/6 accepted**,与旧标准模式一致;accepted/status/t_fail/skip_reason/final_stage/slack_frames/hand_screen 及 `task_mpjpe` p50/p95/max **逐段同值**。
- 旧 schema 与本轮重叠计数一致:attempts、slack_attempts、visits、unique、T 全部相同;**3 段 QP 迭代数不同**(sub12 449→453、sub7 238→228、sub8 168→167)——原因是标准模式同样启用探测行(与 regen 开关无关),R1/R3 纠正后的行集合改变了迭代路径。
- 唯一质量差异:sub12 左接触召回 0.9776→0.9701(67 帧中 1 帧分类变化),其余指标同值。
- 结论:标准模式共享修复无撤回,差异有具体原因与处理结论(探测行纠正,非回归)。

## 5. 历史证据重算(R4)与完整六段对照表(§6)

- **R4 重算**:对全部 18 个运行(含 S01/S10)按当前实现重算逐段质量,写 `quality_tables/quality_table_reviewfix_20260924_all_runs.json`(旧表只读保留);`mpjpe_growth`/`q1_mean`/`q5_mean` 新旧对照写 `quality_tables/growth_recompute_reviewfix_20260924.json`,命中旧表 27 行,例如:v3/sub10 −0.2242→−0.0025(q1 0.4179→0.0697)、v3/sub8 −0.7667→−0.0122、stab01/sub12 −0.5574→−0.0122。**旧“误差随时间显著下降”的解读作废**(q1 实为末帧误差被复制到全段),新口径下增长量级为 10⁻²~10⁻³ m 且 sub16 为正。
- **完整六段对照表**:`quality_tables/six_segment_comparison_reviewfix_20260924.json`,以冻结六段为行来源,逐运行(v3、final_quiet、final_standard、D、Nq、Nv、Ns、S01、S10)填写 accepted/status/keep/拒绝原因与结构化帧号/成本/逐侧质量;无产物的段保留行并写明确缺失原因(标准模式“standard mode writes no NPZ”、无保留 NPZ 段“no kept NPZ (quality recomputed from stored artifacts only)”),不填零、不缩分母。
- 未恢复证据缺口:历史运行(D 之前)无 schema-2 成本视图、无尝试/检查事件序列、无 per-visit 分位(只有最后一轮摘要或干脆缺失),对照表中标为缺失,不按相似性补齐;v3 等 F0 之前的报告拒绝行只有 `status`/`t_fail`,无结构化拒绝证据(表中 `reject_evidence_missing` 标注);D 运行的拒绝块早于 R5 报告补全,其结构化帧号位于 `suffix_retry` 子块(表中以 `validator_failed_checks_pre_R5_report` 指出来源)。R5/R7/R8 相关的旧字段一律以本轮新运行补证。

## 6. 条件项 S01/S10(统一版本稳定性对照)

执行理由:数量门通过但“每侧几何接近覆盖不低于 v3”仍存在 2 项偏差(§7),按方案 §7.2 的例外条款,协议修订需要**当前求解版本**上的必要稳定性对照,故两种权重各覆盖完整六段重跑(命令同 Nq,加 `--stab-weight 0.1` / `1.0`;未拼接变体、未逐 demo 调参)。

| 运行 | wall s | keep | 权重 | sub10 L/R 认证 | sub8 L/R 认证 | 几何接近覆盖 sub10 R / sub8 R |
|---|---|---|---|---|---|---|
| Nq | 278.9 | 5/6 | 0(默认) | 3 / 15 | 29 / 41 | 63/74 = 0.8514 / 78/80 = 0.9750 |
| S01 | 258.2 | 5/6 | 0.1 | **0** / 4 | 4 / **0** | 63/74 = 0.8514 / 78/80 = 0.9750 |
| S10 | 253.6 | 5/6 | 1.0 | 3 / 13 | 4 / **0** | 63/74 = 0.8514 / 78/80 = 0.9750 |

- **权重不改变几何接近覆盖**:三种权重下每侧 `frames_within_enter` 逐帧相同(sub10 L 33/33、sub10 R 63/74、sub8 L 77/80、sub8 R 78/80),参考缺口 p95 也一致(sub10 R 61.2 mm),即 2 项偏差**不可能**由稳定软目标恢复,须由标签用途限制/协议修订处理。
- **0.1 破坏认证底线**:sub10 左认证 3→0(低于 v3 的 ≥3)、sub8 左 29→4;与第十轮 stab01 同一现象。
- **1.0 以认证换深度**:sub10 左维持 3、sub3 左 3→7,但 sub8 右 41→0、sub8 左 29→4。
- sub7 在两种权重下同样终止于 `velocities @t=101`(与 Nq 同帧同因),该拒绝与稳定目标无关;S01/S10 的保留段 NPZ 除 sub16 外均与 Nq 不同(sub16 `d888cfb8…` 三权重全同——该段左/右都无参考抓握区间,稳定项不发力),故**新结论只以本轮消融为准**,第十轮 stab 变体仅作历史对照。
- 门禁判定:保留 `regen_hand_stab_weight=0.0`(配置默认值,实例已记录);稳定性消融的负结果不足以支撑协议修订,偏差继续按 §7 作为待审阅项。

## 7. 门判定(§8)

| 门 | 结果 | 证据 |
|---|---|---|
| 数值 | 通过 | R1 截断距离不再成行(反例 + 装配行距离/线段复算);R2 符号与 FD;R3 缓存全依赖与回流即时生效 |
| 证据 | 通过 | R5/R7/R8 反例 + D 运行事件级证据(同 pass/frame/attempt/check) |
| 统计/溯源 | 通过 | R6 由事件逐项复算全部轮次(130/3630/296 与 legacy 65/1815/148 并存);R9 实例有效配置 + 运行中漂移检查 |
| 质量工具 | 通过 | R4 逐帧实现一致,历史受影响字段已重算(27 行),缺失项显式标注 |
| 数量 | 通过 | Nq/Nv 5/6 keep、sub10/sub8 保留、全部 keep 过独立 strict 与几何筛选;与旧 5/6 逐段比较 |
| 已保留段质量 | **未通过(2 项)** | 认证交集:sub10 左 3/右 15、sub8 左 29/右 41,全部 ≥ v3(3/3/4/0);**每侧几何接近覆盖** sub10 右 0.8514 < v3 0.9054(−5.4 pp)、sub8 右 0.975 < v3 0.9875(−1.25 pp),与第十轮同值同因;S01/S10 证明它与稳定权重无关(§6) |
| 标签适用性 | 已产出 | 逐侧参考帧/认证/最长区间/参考重叠、箱体系 site 速度、认证段锚漂、接触缺口与缺失参考帧齐备;零认证侧标“仅几何接近” |
| 共享回归与拒绝 | 通过 | Ns 对旧标准 6/6 逐段核对(迭代数差异已归因);sub12 恢复、sub7 新拒绝均有实际状态证据 |

**结论**:正确性门通过;数量门通过;已保留段质量有 2 项偏差(与第十轮相同、非本轮引入),按方案 §8 **保留新结果、不自动转 train24**,质量协议修订仍需正确指标 + 必要稳定性对照 + 明确标签用途限制,且不得修改 keep/strict 阈值。

## 8. 与方案的偏差、未执行项

- 未运行 train24/312 重生成与重训(方案未要求;阶段门质量未全过)。
- 未新增/修改任何容差、门槛、筛选阈值与探测采样协议;任务工作门仍 0.40。
- 未更新资源登记与 manifest:本轮未产生新的正式数据版本或资源变化。
- 历史运行的成本/事件缺口无法补齐(见 §5),对照表以显式缺失标注。
- 旧验证记录(m2_m3_stage_gate_fix_verification.md、m2_m3_completion_verification.md、p0*_verification.md)保留原数值与适用范围;本记录为其订正指针:其中涉及 `mpjpe_growth`/q1/q5 的对照、`rejected_final` 归因文本、运行级成本总量与标准模式迭代数的表述已由本轮结论替代。

## 9. 后续

1. 质量协议修订材料:以上述正确指标(sub10 右/ sub8 右几何接近覆盖偏差)与 S01/S10 稳定性对照为依据,明确“几何接近”标签的用途限制;keep/strict 阈值不动。
2. sub7 新拒绝机理(速度终检 @t=101、两轮同因)作为下一轮最小设计输入;sub12 恢复后须复核其认证/覆盖未退化。
3. 六段质量全过之后再按原方案跑 `calibration24_train_v1`(≥18/24 + 认证/动作覆盖结论)→ 312 → 重训。
