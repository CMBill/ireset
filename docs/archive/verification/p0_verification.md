# P0 验证记录:分层干预建立教师基准

> **订正指针(2026-09-24)**:本文中 `skip_reason`/`rejected_final` 归因与逐段成本/计时的表述受后续 F0 与 2026-09-23 审阅 R5/R6/R8 影响,已由[实现审阅修复验证记录](m2_m3_stage_gate_review_fix_verification.md)的 schema-2 成本与尝试/检查事件替代;本文数值保留为 P0 版本的历史证据。

> **适用版本声明(2026-09-14)**:本文全部数字为 **P0-fix(F1–F4)修复前实现**的结果——手部推断速度用世界系使搬运相抓握区间构造性清零、终检失败定位依赖字符串解析且无定位失败退化为末 5 帧重试、转移检查冻结箱姿且纯旋转不细分。修复后重测见 [p0fix_verification.md](../../verification/p0fix_verification.md):strict 6/6、relaxed 22/24 = 91.7%、任务门重标 0.113。本文保留为历史证据,两套数字不得混用;§4 的 8.2cm/11.0-11.1cm 已归类为裸输出教师拟合误差,不再用作闭环任务质量依据。

日期:2026-09-13。方案:[p0_relaxed_validation_plan.md](../../plans/p0_relaxed_validation_plan.md)(修订版)。所有实验命令与产物见文中路径;探针级一次性脚本未入库,其结论已由入库的 `scripts/diagnose_repair.py` 系统化复现。

**摘要**:P0.0 失败分类学工具与登记集落地;P0.1 经 L0(非语义)→ L1(语义解耦)→ L2(成对放宽)阶梯,relaxed 模式教师整段接受率 **4/6 = 66.7% ≥ 50% 验收线**;语义修复单独使 **strict 模式从 0% 升至 2/6 = 33.3%**。阶梯过程中发现并修复 3 个存量 bug(跨段网格状态泄漏、逐分量根速度行 vs 终检范数、无条件插值转移检查严于终检)。测试 84 项全过(新增 6 项)。

## 1. P0.0 失败分类学与登记集

- **筛查**(`scripts/screen_teacher_velocity.py` → `derived/data/screening/velocity_screen.json`):312 个 original 中 **20 个(6%)含速度跳变**(根 ≤5.29 m/s、关节 ≤22.29 rad/s,first-spike 中位帧 60,mod-60∈{0,1,59} 占 30%——弱窗口边界聚类)。预注册排除,记录原因;标定集 24 个 original(每受试者前 4)无污染。
- **行级诊断工具** `scripts/diagnose_repair.py`:对每个被拒段输出 t_fail、skip_reason、求解器 status、失败帧预测点违反行分类(9 类)、pred-z_prev 漂移、slack 占比;`--geometry-census` 直接量化参考支撑/抓取区间内的教师几何(无修复)。
- **基线分类学(严格模式,干预前)**与方案 §1.3 对表一致:foot_speed 31 行(4 段)、foot_contact_band 8(4 段)、hand_anchor 15(2 段)、hand_contact_band 4、sub10 为漂移终态速度行。0/6 接受。
- **几何普查(6 段,pooled)**:支撑期脚底间隙 p50=0.6mm / p90=14.8mm / p95=19.1mm / max=40.2mm(548 帧);抓取期手-箱间隙 p50=1.0mm / p90=24.5mm / p95=33.1mm(863 帧);抓取期箱内锚漂移 p90=181.9mm(悬停包含区间放大)。

## 2. P0.1 分层干预与结果

### 2.1 L0(非语义)

- SCP 预算 15+15 → 40+40(`RepairConfig`/`qp_default.yaml`);多起点重启(求解失败后从 z_prev 再解,`pipeline.py`,frame_status 新增 `repaired_prev`)。
- 测量(6 段):sub10 t_fail 59→45、slack 54/59→11/59;其余段不动。结论:预算救不了几何违反,预期内。

### 2.2 L1(语义解耦,§10.1 容差数值不变)

实施中按失败分类学逐步补全,最终形态五项:

1. **逐对几何接触带分类**(`scp.py::_collision_rows`):robot_box/robot_ground 对距离 ≤ 带上限 → 接触带行,更远 → 非接触间隙行;与锚/参考调度无关。`[2mm, 带上限]` 两分类均满足,边界翻转无害。冻结行仍只由锁定锚产生——即"触地但未锚定"状态,覆盖俯仰支撑脚的抬起侧球与悬停参考抓取。(中间形态"参考驱动脚部带"被探针否定:悬停手被带上限强制拉近,t_fail 反而前移;`_hand/_foot_hold_overlap` 与 `FrameContext.ref_foot_hold` 已随最终形态移除。)
2. **准静态锚点建立门**(`RepairConfig.foot_anchor_establish_speed_mps=0.05`):滑动中的脚可触地(带行)但不锁世界锚。
3. **锚点中途释放**(`ContactStateMachine.release` + `pipeline.py`):预测脚速 ≥ 门 → 释放脚锚;预测手漂出 `hand_anchor_tol_m` → 释放手锚。释放由预测侧驱动(修复态被钉住,永远不会自己触发)。参考区间本身不变,只消失冻结需求。
4. **转移插值检查对齐终检**(`pipeline.py::frame_valid`):探测点按终检的自适应细分公式(motion/5cm,≤3 细分层)而非无条件 1/3-1/2-2/3——原实现严于独立终检,拒绝终检会接受的状态。
5. **推断速度门与检查门对齐**(`ValidationConfig`):`foot_entry_speed_mps` 0.3→0.02(=foot_speed_mps),新增 `hand_entry_speed_mps=0.02`(=hand_relative_speed_mps),`__post_init__` 强制 entry ≤ 检查门(与距离不变式同类)。语义:支撑/抓取 := 准静态;快速接近/离开/滑动段是过渡,不认证。慢滑动(如 0.009 m/s 漂 13mm)仍被漂移门捕获(测试覆盖)。

测量(6 段,严格模式,L1 后):帧级分类学清零(foot_speed 31→0、foot_anchor→0、hand_anchor 15→0);4/6 段跑完全部 120 帧,2/6 段终检拒绝。

### 2.3 阶梯中发现并修复的 3 个存量 bug

1. **跨段网格状态泄漏**(`pipeline.py::_mesh_assets`):`_ground_pts` 只在首段计算并缓存,共享管线实例的后续段沿用第一段的箱轨迹地面点——影响 M2/M3 所有多样本运行(网格目标权重 0.1,软目标,门语义不受影响)。已改为逐条件重算(表面采样仍缓存)。
2. **根速度行逐分量 vs 终检范数**(`scp.py`):QP/帧检查的根线/角速度行是逐分量盒约束,终检是范数——对角步长(0.057,0.057,0.057) 逐分量全过、范数超 65%(sub7 实测 1.657×)。已改为范数行(评估侧)+ 当前方向线性化范数行(QP 侧,低于半限时不激活)。回归测试:对角步必须违反、同幅轴向步必须合法。
3. **终检定位缺失使后缀重试失效**(`validation.py`):`_check_feet/_check_hands` 的 CheckResult 无 location → `_first_failure_frame` 返回 None → 后缀重试窗口退化为最后 5 帧。已加最差帧定位(速度/漂移/间隙/锚/相对速度各自标注 t)。

### 2.4 L2(成对放宽,普查定值)

- `config_io.py`:`load_mode_configs(mode)` 同时驱动 `RepairConfig` + `ValidationConfig`(第一稿只接验收器已被证明是空操作);`check_pair_invariants` 加载时强制"修复带 ≤ 验收门"六对不变式(含穿透幅度),违反即抛错。
- 放宽值(6 段普查):`foot_contact_gap_m`/`foot_ground_gap_m` 0.005→**0.017**(支撑悬浮 p90 14.8mm+2mm);`contact_band_gap_m`/`hand_gap_m` 0.01→**0.035**(抓取间隙 p95 33.1mm);`hand_anchor_tol_m`/`hand_anchor_error_m` → **0.19/0.20**(锚漂 p90 181.9mm,yaml 注释显式记录"教师手不锚定,该值无约束力");`hand_relative_speed_mps` → 0.15(同上,无约束力)。**不放宽**:穿透 ±2mm、关节/根限位、速度界、脚支撑速度/漂移门(准静态语义保持 §10.1 原值)。
- 产物 JSON 内嵌 `mode` 字段,文件名带 `_relaxed`。

### 2.5 P0.1 验收测量

| 配置 | 接受率 | 明细 |
|---|---|---|
| 基线(干预前,严格) | 0/6 | 全部 rejected_frame |
| L0 后(严格) | 0/6 | t_fail 后移(sub10 59→45) |
| L1 后(严格) | 0/6 | 分类学清零;2 段终检拒、其余 rejected_final/frame |
| **L1 后 + 范数速度行 + 推断门对齐(严格)** | **2/6 = 33.3%** | sub3、sub8 通过**全严格**管线+独立终检 |
| **+ L2(relaxed)** | **4/6 = 66.7%** | +sub10、sub16;sub12、sub7 终检拒 |

正式产物:`derived/data/validation/p0_repair_relaxed.json`(含逐检查 final_checks 明细与 slack 占比)、接受轨迹 `derived/data/validation/p0_trajectories_relaxed/*.npz`(4 段)。接受段 slack 占比 ≤ 6/118(sub3),以 skip 为主——基准不是靠 slack 硬撑的。strict 对照:`derived/data/validation/diagnose_strict_final.json`。

**达标判定:66.7% ≥ 50%,P0.1 验收通过**;同时记录:strict 模式 33.3% 证明语义修复本身有效,距离 §10.1 的 90% 仍远(教师悬浮/手漂的尾部由 L2 覆盖,本质仍是路径 B 的论据)。

## 3. P0.2 任务门槛标定

- **标定集运行**:24 个 original(每受试者前 4,筛查后无排除)在 relaxed 模式下整段修复,**17/24 = 70.8% 接受**,轨迹存 `derived/data/validation/p02_calibration_traj/`(运行记录 `p02_calibration_run.json`)。24 段集 70.8% 与 6 段诊断集 66.7% 一致,接受率不是小样本偶然。
- **标定**(`scripts/calibrate_task_gates.py`,17 段 / 1,999 帧):修复诱导关键点 MPJPE p50=0.0656 / p90=0.0766 / **p95=0.0807** → **`mpjpe_tol_m = 0.101`**(p95+0.02),锁定于 `configs/eval/task_quality_gates.yaml`(注释含标定日期、样本数与选择偏置声明)。M2 工作值 0.40 收紧 4 倍;标定脚本同时落盘逐段分布 `derived/data/validation/task_gate_calibration.json`。
- **带门槛重跑**:6 段验收集接受率 **66.7% 不变(0pp 下降 < 10pp 验收线)**,产物 `p0_repair_relaxed_gated.json`。解读:接受段的修复诱导误差本就在 ~10cm 内,0.40 的富余从未被合法使用;收紧后修复自由度约束的是拒绝路径中的漂移终态。
- P0.2 验收:门槛锁定 ✓、标定样本 17 段跨全部 6 受试者 ✓(≥24 original 输入中接受者)、重跑下降 0pp ✓。

## 4. P0.3 生成-修复闭环

设置:8 个 val original × {扩散 DDIM-25, 确定性, 教师噪声 σ=0.02/0.05},relaxed 模式,120 帧,M3 修复后重训的两模型(`derived/models/{diffusion,deterministic}`)。跑了两个变体:

### 4.1 带标定门槛(0.101)——全部拒绝,归因清楚

`generation_eval_relaxed.json`:4 类输入 0/8 全部 `rejected_frame`。单段归因探针(val sub10_002):裸教师在 0.101 与 0.40 下**都接受**;σ=0.02 噪声在 0.101 下 t=29 `task_quality` 拒、在 0.40 下**接受**。结论:门槛在"干净教师接受段"上标定(p95 0.081 + 0.02),对输入误差更高的噪声/生成输出(~9-11cm 基线)结构性过紧——输入自身误差占掉预算,修复只剩 ~1cm 自由度。这正是锁值注释预登记的选择偏置,如实记录;门槛用于教师筛选场景,生成-修复对照场景保留 0.40 工作值。

### 4.2 无门槛(0.40)——闭环机制打通

`generation_eval_relaxed_ungated.json`(行内含 t_fail/skip_reason/修复幅度):

| 输入 | 整段接受率 | 修复耗时 ms/帧 | 备注 |
|---|---|---|---|
| 裸教师(补测,同 8 段) | **2/8 = 25%** | — | sub10_002/100 过,其余 t_fail 4-94 |
| 教师噪声 σ=0.02 | 1/8 = 12.5% | — | |
| 教师噪声 σ=0.05 | 1/8 = 12.5% | — | |
| 扩散 DDIM-25 | 1/8 = 12.5% | 370-888 | 接受段=sub10_100(MPJPE 8.2cm) |
| 确定性 | 2/8 = 25% | 191-568 | 接受段=sub10_002/088(MPJPE 11.0/11.1cm) |

- **临时验收条件核对**:确定性 25% = 裸教师 25% ✓、≥ 噪声基线 12.5% ✓;扩散 12.5% **低于**裸教师 25%、与噪声基线持平——部分满足,如实记录。共同成功样本配对(§10.1 公式:MPJPE ≤ 基线 + 2cm):扩散接受段 8.2cm ≤ ~0.10 ✓(n=1);确定性 11.0-11.1cm 超 0.10 约 1.1cm(n=2)。样本数过小,只作机制验证不作结论。
- 接受段修复幅度:根平移均值 3-18mm、关节 max 0.27-0.54 rad(分开以 m/rad 报告,§10.1 修复行)。
- val 8 段裸教师 25% vs 训练侧 24 段集 70.8%:接受率对演示难度强依赖(val 段更难,与 M3 教师质量量化一致),不是模式切换问题。
- 保真度(8 段宏平均,与 M3 记录一致):扩散 MPJPE 9.0cm / 确定性 10.9cm;裸输出终检仍以 penetration 拒(可行性由修复器负责,管线设计不变)。
- **技术债标注**:本节全部数字为"临时放宽容差 + 语义修正 + 0.40 任务门槛"模式,**不计入 §10.1 达标**;严格约束验证推迟至 M2 收尾/教师重生成。

## 5. 结论与后续

- P0 三项验收:P0.1 ✓(66.7% ≥ 50%,24 段集 70.8% 佐证)、P0.2 ✓(标定锁定 + 零下降)、P0.3 机制 ✓/条件部分(确定性过、扩散低于裸教师持平噪声基线,n≤2 不作结论)。
- 最重要的副产物:**语义修复使 strict 模式 0%→33.3%**(2/6 全严格管线通过),且全部修复不触碰任何 §10.1 容差数值——这些是应当保留的正资产(路径 C 固化),与临时放宽解耦。
- 教师数据本性(P0 后强证据):支撑期悬浮 p95 19mm、抓取锚漂 p90 18cm、val 段裸教师仅 25% 接受——语义修复解不了,**路径 B(教师重生成)仍是 §10.1 ≥90% 的必经之路**。
- 遗留:relaxed 手部锚定值(0.19/0.20)无约束力的事实记录;`task_mpjpe_tol_m` 双场景取值(教师筛选 0.101 / 生成对照 0.40)需在教师重生成后统一重标定;M2 扰动集与计时基线已被本轮改变,需重测。

## 5. 测试与工程备注

- 测试 84 项全过(`uv run --locked pytest tests/ -q`),本轮新增 6 项:逐对几何带分类(含 kind 感知假距离)、范数速度行(对角违反/轴向合法)、脚带参考→几何分类演变、准静态建立门、锚点释放(脚+手)、config_io 双模式与不变式违反报错。两个既有测试按新语义更新(快速轴旋/快速滑移为"过渡不认证",慢速违例由漂移门捕获)。
- `eval_repair.py` 新增 `--validation-mode/--output/--save-dir/--teacher-only/--task-gate-yaml`;`eval_generation.py` 新增 `--validation-mode/--output/--task-gate-yaml`;`pipeline.py` 接受轨迹的终检明细写入 diagnostics(`final_checks`)。
- 工程事实:教师 NPZ 四元数在前布局在筛查脚本中再次踩坑(平移在 `[4:7]`,首次误用 `[0:3]` 得出 25% 假尖峰率,已修正为 6%);`_collision_rows` 的 max_active_pairs 截断按距离排序,均匀假距离会因稳定排序只保留先列出的 self 对(测试需按 kind 区分假距离)。
