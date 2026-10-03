# train24fix3 收敛修复与转段执行验证记录

日期:2026-10-02 起,2026-10-03 增补续跑指针与阶段 D 完成记录。对象:[train24fix3 收敛修复与教师数据转段执行计划](../plans/train24fix3_convergence_recovery_plan.md)。状态:**阶段 A/B/C 完成(N1–N3 全未过六段开发门,winner=None);阶段 D(norm_cut S1/S2)于 2026-10-03 预算调整续跑(8h/10h,提交 `33c8c2b`)后全部完成——S1/S2 六段均 0/6 keep,winner=None,阶段 D 按 §7.3 结束(失败分类:四段信赖域塌缩+两段锚漂移终检败),阶段 E/F/G 未进入**。S1/S2 的机理层证据(五帧 5/5、尖峰 10/10 拒、启用态核验、cf 分类 41+72/57+56)成立但均非整段结论。实验根目录:`derived/data/teachers_regenerated/train24fix3_20261002_1147424/`。

起点:代码 `e74a5f3`(方案提交 `dcf9d33`)。阶段 A 按方案 §4.1 任务 1 在任何算法修改前执行。

## 0.1 续跑指针(2026-10-03 写;**当日已按此执行完毕,本节保留为入口文档**)

**一句话现状(2026-10-03 续跑完成后)**:预算调整已登记(S1 8h/S2 10h,用户授权,串行自动,提交 `33c8c2b`)并全部执行——S1 六段 0/6、S2 全套 0/6,**winner=None,阶段 D 按 §7.3 结束**(失败分类与最小问题规格见 §4);阶段 E/F/G 不进入。以下"精确续跑步骤"是当时预注册并已被实际执行验证的路径(命令原样可用;驱动重入限制属实——S1 走 CLI 直跑、S2 走分步函数+手写 summary,辅助脚本 `norm/s1_assemble.py`/`norm/s2_continue.py` 随产物保留)。

**精确续跑步骤**(§12 规则:同冻结配置/profile、调整预算单独登记、续跑保持完整分母;已执行,勾选):

1. **预算调整登记**(需用户授权后写入本节末尾的登记表):S1 六段按实测节奏(首段 51 min 未完)估计 ≥5h;S2 全套(五帧 ~5 min + 113 反事实 ~14 min + 六段 ≥5h + 质量门 ~4 min)估计 ≥6h。合计上限建议登记为 16h,可分两次挂后台。
2. **S1 六段续跑**——注意**不能直接重跑 `--stage norm`**(驱动对已存在产物 refuse-overwrite,S1 的 cf/ 产物已在):用同一冻结 profile 直接跑 CLI,输出目录沿用 `norm/S1/sixseg/rq/`(该目录只有 dirty_diff patch,报告从未写过,无覆盖冲突):
   ```bash
   OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
   uv run --locked python scripts/regenerate_teachers.py \
     --set diagnosis6 --mode regen --validation-mode strict --profile S1 \
     --assert-task-gate 0.40 \
     --output-dir derived/data/teachers_regenerated/train24fix3_20261002_1147424/norm/S1/sixseg/rq \
     --report   derived/data/teachers_regenerated/train24fix3_20261002_1147424/norm/S1/sixseg/rq/regen_report_diagnosis6_strict.json \
     --note "train24fix3 stage-D S1 sixseg continuation (budget-adjusted, frozen profile S1)"
   ```
   完成后跑质量门并核对 R9 干净(`end.code_changed_during_run == False`;**运行期间不得写仓库任何文件**):
   ```bash
   env PYTHONPATH=$PWD/src:$PWD/scripts uv run --locked python scripts/train24fix3_quality_gate.py \
     --run derived/data/teachers_regenerated/train24fix3_20261002_1147424/norm/S1/sixseg/rq \
     --out  derived/data/teachers_regenerated/train24fix3_20261002_1147424/norm/S1/quality_gate.json
   ```
3. **S2(0.049)全套**——`norm/S2/` 目录为空,驱动阶段可直接重入;但 `--stage norm` 会先重跑 S1 的五帧与 cf(撞 refuse-overwrite)。**续跑路径**:直接 `--stage norm` 前先把 `norm/S1/` 移到 `norm/S1_done_20261002/`(保留全部证据),或在 Python 里 `from train24fix3_convergence import _profile_frozen_frames, _norm_counterfactuals, _run_sixseg, Gate; from ireset.repair.profiles import get_profile` 分步执行 S2 并手写 `norm/norm_summary.json`(字段结构照抄 `stage_numeric` 的 summary)。
4. **裁决规则**(方案 §7.3,预注册):S1/S2 任一通过全部六段门(基础 6/6 keep + R9 干净 + 质量门 H/B/C/T 全过 + 尖峰全拒 + 启用态核验)→ winner → 阶段 E(正式六段:2 quiet + 1 verbose + ns + 完整重放,§8.2);两者皆未过 → 阶段 D 结束,交付失败分类(数值未收敛/其他行阻塞/认证/漂移/任务退化/证据不足)与新最小问题规格,**不得因局部成功降低六段门进 train24**。

**产物索引**(实验根 `train24fix3_20261002_1147424/`):

| 目录 | 内容 | 状态 |
|---|---|---|
| `baseline/` | fb/cf 严格重放(新目录)、n0_identity、H/B/C/T 哈希 | 完成,PASS |
| `trace/` | 五帧全迭代 trace(JSON+NPZ,每帧 3 尝试 70–94 QP) | 完成,PASS |
| `numeric/{N1,N2,N3}/` | 每 profile:冻结帧+六段 rq+profile_summary | 完成,winner=None |
| `numeric/numeric_summary.json` | 阶段 C 裁决 | 完成 |
| `norm/norm_run2_driver_20261002.log` 等 | 两次驱动运行日志保全副本 | 完成 |
| `norm/S1/cf/` | 113 个 v3 产物+cf_index | 完成 |
| `norm/S1/sixseg/rq/` | 续跑报告+新旧两次 dirty_diff patch | 完成,**0/6 keep** |
| `norm/S1/{profile_summary,quality_gate}.json` | S1 组装裁决(含 continuation 溯源块) | 完成,selected=false |
| `norm/S2/` | 五帧+113 cf(v3 产物)+六段 rq+profile_summary+quality_gate | 完成,**0/6 keep** |
| `norm/stage_status_budget_stop.json` | 预算中止时间线+续跑范围 | 完成(历史状态,已被续跑超越) |
| `norm/norm_summary.json` | 阶段 D 裁决 | 完成,**winner=None** |

**本轮新增工程事实**(新上下文必读,AGENTS 事实链之外的本轮增量):

- 驱动/工具的哈希比较必须用全长摘要:`closed_loop_metrics.sha256` 是 16 位截断变体,对 F 的 64 位清单必然假漂移(本轮 baseline 首跑即踩,修复 `1147424`;驱动内统一 `sha256_full`)。
- `_qp_row_names` 的 pre 块是**条件布局**:`z_prev=None` 时无 joint_vel/root 行(64 行而非 99),静态描述符错位 29 行——必须按 ctx 条件构造;C2 块行号锚点取自真实装配(`c2_tail_end`),不沿用名义常数。
- `check_rows` 返回的 names 是**碰撞块名**;全行描述符要经 `_check_descriptors`(消费方:`_row_dump`、trace、测试)。
- norm_cut 在 N3 档求解设置(polish/1e-9/20000)下 C2 激活帧成本 ~10× 于 N3(切面重解 × 20000 迭代 QP);冻结帧墙钟 27–59s vs N3 的 3–13s。若预算仍是约束,可考虑的**非算法**缓解只有先跑六段挂后台过夜,不允许改求解设置来提速(那是新候选)。
- `HAND_SITES` 按标签索引、C2 行布局按 site 索引;唯一映射常量 `scp._C2_SITE_LABELS = ("palm","thumb","pinky")`。
- 质量门工具 `train24fix3_quality_gate.py` **无跨运行缓存**(曾按 demo 名缓存导致参照被候选值污染、H 比较假通过——已移除并复验);参照 demo 与运行 demo 集不重叠时记 N/A 不记失败(T 与六段集只共享部分 demo)。
- v3 产物格式 `train24fix3_norm_products_v1`(自哈希+有效配置+注册表哈希+integrity,双视图索引);加载器 `load_norm_products`/`load_norm_index` 在驱动内,篡改(NPZ/格式/索引六变体)由 `tests/test_train24fix3_norm.py` 钉死。
- 驱动 `train24fix3_convergence.py` 六阶段均读阶段摘要拒跳前置门;所有产物写入 refuse-overwrite——**中断后不能整阶段重跑**,续走分步 CLI/函数(见上)。
- OSQP 1.1.3 的 `res.info` 字段名是 `iter/prim_res/dual_res/obj_val/status_polish`(无旧版 `pri_res/dua_res`);solve_qp 的 trace 已按此记录。

**2026-10-03 续跑新增事实**(接上,本轮增量):

- **单点负载观测外推预算可差 3–5×**:同冻结配置同输入,S1 六段 Oct 2 晚间"首段 51 min 未完"而 Oct 3 上午全程 91 min(5432.6s)——机器负载让 ≥5h 估计完全失真;§12 的 INCOMPLETE 处理正确,但续跑预算估计应以安静时段实测为准。
- **预算中止后的阶段重入路径已完整验证**:CLI 直跑六段(rq 目录内新旧 dirty_diff patch 并存无冲突、报告只在结束时落盘)+ 分步函数执行 S2 + 手写 summary(`norm/s1_assemble.py`/`norm/s2_continue.py` 随产物保留)可无损恢复阶段;S1 组装时五冻结帧确定性重跑与保全日志墙钟逐帧吻合(±0.6s),cf 裁决从冻结产物复原不需重算。
- **`load_norm_index(pdir)` 与 `load_norm_products(pdir, stem)` 目录约定不一致**:前者期望父目录(内部拼 `cf/cf_index.json`),后者期望 cf 目录本身(拼 `cf_{stem}.json`)——同族加载器调用时必须区分。
- **S2 锚漂移族是放宽 C2 界引入的新失败模式**:0.049 下 sub16/sub3 整段求解走通(sub3 处理完全部 118 帧、双侧 C2 携带 176/180 帧)但独立终检败于 `hand_grasp` 右手锚点(sub16 锚 t=40;sub3 锚漂自 t=14 累积至 t=85 败)——修复目标从"求解收敛"移至"锚维护与速度界的联合权衡"。

**实现落点清单**(全部已提交、测试覆盖):

- `src/ireset/repair/profiles.py`:`OsqpSettings`/`C2Profile`/`load_profiles`(闭合字段/继承展开/非法值拒绝)/`profiles_yaml_sha256`;注册表 `configs/repair/train24fix3_profiles.yaml`(N0–N3/S1–S2)。
- `src/ireset/repair/scp.py`:`RepairConfig.osqp_*`(默认=N0)、`_frame_qp_settings`(profile 仅 C2 激活帧)、`solve_qp(settings=None)`+trace、`_fixed_post_rows`(axis/norm_cut 双模式真实行)、`_post_row_descriptors`(模式感知缓存)、`build_qp`(C2 盒行/切面行/`_c2_lin` 线性化/共享 site 松弛组/mu 罚权)、`_c2_cut_refine`(切面循环+预算+账目)、`_qp_row_names`、`solve_frame`(trace 插桩+切面接线+逐迭代清缓存)、`FrameResult.diagnostics["c2_cuts"]`。
- `src/ireset/repair/pipeline.py`:`RepairPipeline(..., c2_profile=...)`。
- `scripts/regenerate_teachers.py`:`--profile`(regen-only 守卫;解析值+注册表哈希入 effective 记录与报告)。
- `scripts/train24fix2_audit.py`:`configs_from_record` v2 回填(v2 携带非 N0 新字段即拒)、`fresh_audit_pipe/full_candidate_audit` 的 `c2_profile` 参数。
- `scripts/train24fix3_quality_gate.py`:H/B/C/T 逐侧不退化门+sub16 保护。
- `scripts/train24fix3_convergence.py`:baseline/trace/numeric/norm 阶段 + v3 产物读写。
- 测试:`tests/test_train24fix3_profiles.py`(9)/`test_train24fix3_numeric.py`(7)/`test_train24fix3_norm.py`(11)。

**预算调整登记表**(授权后填写;2026-10-03 已登记):

| 日期 | 项 | 原预算 | 调整为 | 依据 | 授权 |
|---|---|---|---|---|---|
| 2026-10-03 | S1 六段续跑 | §12 norm 总 2h(已耗尽) | **8h** | 首段实测 51min 未完,~10× 成本观察 | 用户(2026-10-03 会话) |
| 2026-10-03 | S2 全套 | 同上 | **10h** | 同上 + S1 冻结帧/cf 实测 ~19min | 用户(同上) |

执行安排(同次授权):**串行自动**——S1 六段完成并组装裁决后,若未过全部门,按 §7.3 预注册规则直接启动 S2 全套,全部完成后一次汇报;若 S1 过门则 S2 记 not_executed。合计上限 18h。本登记在续跑启动前提交(运行期间仓库零写入,R9);线程环境与 `_run_sixseg` 一致(OMP/OPENBLAS/MKL/NUMEXPR/VECLIB=8)。

**执行结果(2026-10-03)**:S1 六段 10:23–11:54 实际墙钟 5432.6s(≈1.5h,远未触顶 8h;Oct 2 的 ≥5h 估计为环境性偏慢);S2 全套 12:08–14:53 实际 ≈2.8h(六段 9611.2s + 五帧 ~4min + cf ~14min + 门 ~4min,远未触顶 10h)。两次运行 R9 均干净(`code_changed_during_run=False`)。

## 0.2 续行前置阶段 Q:审阅修复 Q1–Q4 实施(2026-10-03,方案修订版 §4.3/§4.4;§11.1 提交单位 1)

方案经审阅更新后按 §4.3 只修证据链(不动求解目标、切面生成、步接受、启用态或锚状态;算法变化留给 §7.4 的 R 候选)。本节登记**代码与测试提交**;S1/S2 证据重生成与严格重放属于 §11.1 单位 3,本提交未运行任何候选/六段/train24。

**四项修复与落点**(行号以提交时为准,函数名为准):

- **Q1(审计与求解同一 profile)**:`_profile_frozen_frames` 的 `full_candidate_audit` 调用现显式传入冻结 profile(原调用漏传,norm 采纳候选被 axis 解释器审判)且**保留审计数组**——采纳候选连同完整上下文/前缀/箱体序列冻结为 recovery 产物(`ff/` 目录+双视图索引);结果分列 `n_adopted`/`n_full_audit_pass`/`n_adopted_audit_fail`/`n_evidence_incomplete`/`n_not_adopted`,不合并为单一数字。
- **Q2(norm 完整加载/严格重放+配置·完整性接线)**:新 schema `train24fix3_recovery_audit_v1`(下称 recovery):冻结身份(demo/t/pass/attempt)、完整 `FrameContext` 数组+meta、候选/前缀/箱体序列、**current 与 plus_c2_norm 两分支审计**(current 用冻结 `repair_no_c2` 配置重算,分类取自本次两分支,不再借用旧 current 汇总)、来源全长哈希、完整有效配置(三键)、展开 profile、integrity。加载器 `load_recovery_products` 核验:格式/NPZ 自哈希/**有效配置完整可往返(新格式缺字段即拒)**/profile 可重建/来源活性(全长哈希新鲜复算)/`verify_integrity` 新鲜清单。`replay_recovery`:索引驱动双 kind(ff/cf)、两分支重算+重分类、ff 重跑完整升级链、`deep_diff`+`arrays_diff` 全量比对、距离四视图对来源 NPZ **新鲜查询**交叉核验(16 位哈希域同源比较)、`n==expected` 入判据、输出只写独立目录。`configs_from_record` 改为**格式分派**:仅 v2 允许按登记 N0 回填,其余格式(含 `train24fix3_norm_products_v1`)缺任何 osqp 字段即拒。`_integrity_files` 补入 `scripts/train24fix3_convergence.py`、`scripts/train24fix3_quality_gate.py`、`configs/repair/train24fix3_profiles.yaml`(`src/ireset/repair/profiles.py` 原已由 glob 覆盖)。**目录约定统一**:`load_norm_index` 与 `load_norm_products` 均接受产物目录本身(旧父目录+内部拼 `cf/` 的约定废除——历史 S1/S2 产物在新约定下冒烟可读)。
- **Q3(每次动态 QP 与矩阵一一对应)**:`build_qp` 在 trace 开启时冻结本次装配元数据(`_qp_assembly_meta`:实际行列数、全行描述符(基础行+slack 下侧复制行+slack 非负行)、列身份、切面身份(side/site/k)、行→slack 映射、z_prev 条件布局;维度与命名不符即 raise);`solve_qp` 增 `qp_identity` 关键字,**每次实际求解旁**记录单调 `qp_id`/branch(tight/restoration)/outer_iter/cut_round/实际 `rows_n`/`cols_n`/`n_s`/`n_cuts`/slack 映射,行名列数组逐 QP 入 NPZ 侧;`_c2_cut_refine` 每次重装配重解携带自身身份(cut_round=1..8)。**删除** solve_frame 里"切面循环后只补记最后一条 QP、`rows_n` 用初始 A 行数、attempt 级单一行名缓存"的旧做法;切面账目与最终 slack 向量改入迭代记录(`c2_cut_stats`/`restoration_slack`)。trace 关闭时控制流逐位不变(N2/N3 纯度测试钉死)。
- **Q4(数组键单前缀)**:`_norm_counterfactuals` 写出直接复用审计返回键(恰一层 `rows__<label>__`);历史 v1 双前缀键由 `decode_norm_v1_arrays` **显式分派解码**并返回布局标记(`v1_double_prefix`),混合布局/无双前缀键均为显式错误,禁止静默改名后宣称跨版本数组相同;`_write_norm_products` 加 v1 格式守卫。

**测试**:`tests/test_train24fix3_recovery.py` 15 项——Q1 合法范数位移 axis 拒/带 profile 审计过+数组保留、驱动端到端(spy 断言 profile 到位、产物/索引/计数分列);Q2 configs 回填仅 v2、integrity 清单覆盖三新文件、加载器六类篡改(删完整性/删 osqp 字段/删 profile/来源哈希错/删来源/NPZ 改写)与 doctored live scp 摘要全拒、严格重放双分支重算 n==expected 二跑稳定不污染产物目录、分类翻转/行数组篡改+自哈希重写/删距离来源各自显式败、ff 升级链重放+term 篡改败、索引坏视图拒绝;Q4 单前缀+双前缀显式解码+混合布局拒;Q3 切面重解元数据逐条与矩阵 shape/行名/slack 映射全等(强制多轮切面场景)、solve_frame 全 QP 身份完整+计数三处相等+z_prev 条件布局+attempt 级行名缓存已除、N2/N3 trace 关闭候选逐位不变。全量 **288 → 303 passed + 1 skipped**。

**兼容性与边界**:历史 S1/S2 v1 产物按 v1 路径可读(双前缀解码冒烟通过);旧 v2 产物走原 N0 回填不变(既有 reviewfix 测试全过);`load_norm_index` 约定变更使旧实验根内归档辅助脚本(`norm/s1_assemble.py` 等,ignored 产物)如重跑需按新约定传参——它们是一次性恢复脚本,保留为证据不重跑。**本提交未运行**:S1/S2 证据重生成(单位 3)、六段/train24 正式驱动与子任务状态机(单位 2)、R 候选;Q 修复对历史 N/S 行为的影响面=零(默认/trace 关闭路径逐位不变),阶段 D 的 0/6 裁决与失败分类不被本提交覆盖。

## 0.3 续行前置阶段 Q:执行基建(2026-10-03,§11.1 提交单位 2)

按方案 §8.2/§9/§12 补齐执行基建;测试 303→**~~315~~ 316 passed + 1 skipped**(新增 `tests/test_train24fix3_formal.py` 13 项,见下;**2026-10-03 订正**:原记 315 系计数笔误——303+13=316,单位 3a 时全量收集 317 项实证)。本提交只落代码与测试,**未运行任何正式六段/train24**——阶段 D 结束时 winner=None,正式阶段入口就绪且其拒绝路径被测试覆盖(这正是当前真实状态)。

- **完整门(§8.1 每项 verdict)**:`train24fix3_quality_gate.py` 重写为全家族门——覆盖(报告 demo 非空/无重复/与冻结名单集合相等,删失败 demo 不能缩分母)、NPZ 一致性(kept NPZ 集==accepted 集==metrics 集,T 一致)、全部保留段独立 strict 复验、逐侧 core/complete/task(原有)、**demo 级双手重叠**(`both_hands_metrics`,认证/监督/最长同监督段三计数不降)与**可用窗口**(TeacherWindowDataset 自身 60 帧/步长 30 的索引式 `(T-60)//30+1`,不另定义更宽松"可用")、sub16 C 保护(原有)、**机理族**(尖峰全拒+启用态合规取自 cf 摘要;四个冻结失败侧 sub10:r/sub3:r/sub8:l/sub8:r 对 H 的丢失/增益重分类 GATED 帧归零——复用认证边缘轮同款 `classify_side`,`freeze_segment` 增量 `with_series=True` 返回已算好的逐帧 dist/speed/max_component/supervision/cert 序列,默认 False 历史返回形状逐位不变)、**完整性族**(R9 干净/profile 与运行自身记录一致/行状态与 accepted 一致/argv 在册)、**标签族**(graded_labels_v1 清单:协议、行集==运行 demo 集、每保留段 label NPZ 16 位同源哈希+形状+布尔+supervision==ids≥0+strict_recheck;torch 级空监督零损失/零梯度契约由 `tests/test_label_quality.py` 钉死)。`full=True`(正式 E/F 模式)使**依赖证据的家族成为强制**:缺 cf 摘要/标签清单/profile/名单=显式 False,缺 gate 不是通过。CLI 增 `--full/--expected-set/--profile/--cf-summary/--labels`。
- **子任务状态机(§12)**:驱动新增 `SubtaskRegistry`(`train24fix3_subtasks_v1` 机读索引)——每子任务登记 expected、身份(profile+注册表哈希+rev+工作树状态)、输出哈希、起止、实际处理数与状态;`verified()` 仅返回**完整+同身份+输出哈希仍验**的尝试(resume 只复用此);半写文件留为失败证据,重跑写新 `attempt_NNN` 目录(拒绝覆盖/移动);`--resume` 语义经崩溃续跑测试验证(崩溃→已完成 rq1/rq2 复用、rv 重跑 attempt_002、汇总从不完整文件生成 PASS 被拒)。
- **正式阶段驱动**:`stage_sixseg` 落地——selection 交叉核验(norm/numeric 摘要 winner 与 `--profile` 与 winner 自身 profile_summary.selected 三方一致,错误选定/无 selection/winner=None 均 SystemExit)、rq1/rq2/rv(quiet×2+verbose)+ns(标准模式,**无 --profile**)四子任务、三跑一致(状态/计数/字段)+保留 NPZ 字节一致、ns 对照 **B/ns 参照报告**同计数且零 C2 证据、标签导出子任务、完整门子任务(full 模式)、选择证据双跑重放;全部通过才写 sixseg_summary。`stage_train24` 落地——sixseg PASS 前置、冻结 profile 锁定(不匹配即 SystemExit)、三跑、≥18/24 数量门+外来 demo 拒绝、六受试者均有可用窗口、T 13 段保留、标签+完整门。`_run_sixseg` 扩展(verbose/mode/set_name;标准模式按工具自身命名 `standard_report_*` 且不传 profile);CLI `--profile` 实际传入执行函数。
- **测试**(13 项):伪造指标源的快速契约测试(每家族 verdict 在册且非 None、full 模式缺证据三家族显式 False、删行/重复行/kept 集不一致/strict 失败/GATED 翻转/R9 脏/profile 错/行不连贯/协议篡改/supervision-ids 不一致各自翻假)+ 真实数据双冒烟(C 运行:1 kept、H 比较按 kept 失败、mechanism 真实值;**B 运行:四冻结失败侧 13<15/13<18/27<29/37<41 逐值复现且 ok=False——单位 2 验收冒烟**)+ 注册机单测(复用/身份变更/输出篡改/失败留痕)+ 阶段全流程与崩溃续跑 + 拒绝路径(无 selection/winner=None/profile 不匹配/selected=False/缺 sixseg)。
- **兼容性说明**:`freeze_segment` 的 `with_series` 为加法变更(默认关闭),`train24fix2_cert_trace.py` 文件哈希因此变化——历史产物对其的完整性漂移按方案 §4.1 冻结基线处理(预期差异,不改写旧清单);既有 26 项 v2/reviewfix 测试全过。实验根的 evidence/failure-prefix/recovery 目录随单位 3–4 的实际执行轮建立,驱动入口已就绪。

## 0.4 续行阶段入口:evidence / failure-prefix(2026-10-03,§11.1 提交单位 3a;方案 §4.4 项 6/§5.3/§13)

方案 §13 要求的两个续行阶段入口在单位 2 尚未落地(当时只补齐 sixseg/train24),本提交补齐——**只落代码与测试,正式运行在提交后的冻结工作树上执行**(提交单位 3b 记录结果)。

- **`stage_evidence`(§4.4 项 6,Q 预算 4h)**:历史来源核验(`_historical_norm_evidence`:旧根 norm_summary winner=None、双 S 六段报告 0/6+R9 干净+git_rev 在册、**live S1/S2 profile 与历史 resolved 逐字段相等且注册表哈希未漂移**——漂移即"重跑不是原冻结 profile"、旧 cf 目录 227 文件全 64 位摘要登记)+ 续行 baseline 登记(`_baseline_registration`:H/B/C/T 参照哈希+F 249 产物活性+未跟踪文件内容哈希副本)+ §12 子任务机上的 8 个子任务(ff_S1/cf_S1/双 replay×2 profile,expected 5/113/118,attempt 目录、resume 只复用完整同身份可验尝试)+ 双跑重放稳定性与产物目录不污染 + **逐项对照**——历史 ff 行(stem/t/采纳/来源/status/term)与旧 v1 cf 逐 stem 分类(S1 41/72、S2 57/56)比对;差异为**记录性 verdict**(per-stem diff 列表+分布偏移,`per_item_identical=False`),不回写不强判——机理门(尖峰全拒/启用态)与 n==expected 仍为硬门。预算控制:`--budget-min`,未启动子任务记 `not_run_budget`、段终 INCOMPLETE(半跑不判过)。
- **`stage_failure_prefix`(§5.3,P 预算 6h)**:evidence 全过才启动(缺/不过/INCOMPLETE 各自显式拒绝);绑定清单从**真实旧报告**装配并核验恰为 10 solve(S1×6 t_fail 60/14/36/63/23/32 + S2×4 60/14/23/33,全部 `state_constraints`/单 pass)+ 2 anchor(S2 sub16/sub3,`final_validation` 两 pass)。**solve 绑定**(`_bind_solve_failure`):按 `regenerate_teachers --profile <P>` 同构配置整段重跑+只读取证快照(冻结 ctx 逐字段位级核对),身份对拍 13 项(t_fail/skip_reason/final_stage/last_attempt_id/last_check_id/status×2/n_passes/attempts/qp_calls/iterations/C2 frames_on 双侧)——**任一不符停止绑定不写产物**;通过后用 trace 开启重放失败帧升级链(与段内失败 status/term/iterations/qp_calls 全等才收),产物=前缀+完整上下文+每 QP 行/列身份(紧凑序列化:全 QP 元数据与行/列名、仅末次尝试末 QP 存全矩阵——§5.2 冻结数值用例)+身份块+成本。**anchor 绑定**(`_bind_anchor_failure`):双 `final_fail` 快照(逐 pass 走通轨迹+各自终检)、逐帧轻量事实(C2 启用侧、ctx 手锚是否非空——regen 模式必须恒空)、身份对拍(final_stage/t_fail/suffix_retry_from/violated_rows/双阶段 failed_checks 位置/逐 pass 计数/C2 frames_on);每 pass 轨迹经 `to_teacher_layout` 往返写 NPZ 后 `freeze_segment(with_series=True)` 全侧分析——参考区间/C2 启用帧/认证区间与锚建立帧/**逐区间累计漂移曲线**(本轮新增)/参考速度最大帧/区间起点差异三列并排。摘要可覆写(`_write_stage_summary`:resume 语义),产物目录仍拒绝覆盖。
- **`freeze_segment` 增量**:`with_series=True` 时每认证区间输出累计锚漂曲线 `anchor_drift_curves`(键 `a-b`,与冻结的区间最大值同源一致——真实数据 B/sub8 双侧 10 区间对拍 `max(curve)==anchor_drift_certified_max_m` 逐位)。
- **测试 12 项**(`tests/test_train24fix3_continuation.py`):历史来源真实数据双测(norm 摘要 winner=None+registry/resolved 冻结+cf 目录 113 npz/227 文件登记;`_cf_hist_map` 实测 41/72 与 57/56)、紧凑 QP 序列化(全 QP 行/列身份、仅末 QP 全矩阵)、evidence 全流程+resume(8 子任务全过/机理 verdict 从阶段摘要重发/重放不污染)、**cf 差异定位不强制**(翻转 2 项→`per_item_identical=False`+diff 列表+分布 43/70,S2 不受扰,硬门仍过)、预算 INCOMPLETE(8 子任务全记 not_run)、子任务失败留痕续跑、failure-prefix 三种拒绝路径、真实 spec 计数(10+2 与 t_fail 序列逐值)、全流程+复用(anchor attempt 输出含双 pass 场景 NPZ)、身份不符绑定失败留痕(11 项仍成)、漂移曲线一致性。全量 **316 → 328 passed + 1 skipped**。
- **实现级冒烟(提交前,/tmp 一次性,非正式运行)**:真实 S1/sub12 solve 绑定全链 803s——身份 13 项逐值全配(t_fail=14/attempt [0,14,2]/check `p0:t14:c39`/25 attempts/5729 QP/649 iters/C2 14/14)、升级链重放与段内失败全等、NPZ 748 QP 各带行身份+仅末 QP 全矩阵+前缀 [14,36]。正式 failure-prefix 运行在实验根重做该绑定(冻结 rev 下)。

## 0. 实施提交与测试

| 提交 | 内容 |
|---|---|
| `dcf9d33` | 方案文档(预注册) |
| `a7954e2` | 驱动 baseline 阶段(F 复核/参照登记/N0 冻结) |
| `1147424` | 驱动修复:哈希统一全长摘要 + C 参照期望订正 |
| `a53b6f7` | 提交单位 1:OSQP 设置配置化、逐帧 profile 门控、只读 trace、N0-N3/S1-S2 注册表、质量门工具、驱动 trace/numeric 阶段 |
| `d8e1f25` | 提交单位 2+3(合并,装配改动交织):N2 行预条件 + N3 mu 恢复罚权 |
| `3a23e92` | 阶段 D:norm_cut 范数分支实现(真实范数行/切面装配/共享松弛/v3 产物/norm 驱动阶段) |
| `afffdf4` | 驱动修复:norm 产物目录创建 |
| `84a7a7d` | 阶段 A–D 验证记录+AGENTS 同步(本文件的 §0–§5 与 AGENTS 顶部条目) |
| `2d5c934` | 记录增补 §0.1 续跑指针(新上下文入口、精确续跑命令、产物索引、登记表占位) |
| `33c8c2b` | 预算调整登记(S1 8h/S2 10h,用户授权 2026-10-03,串行自动)——续跑启动前提交保 R9 |
| `6b305c8` | 阶段 D 续跑完成记录+AGENTS 同步:§0.1 执行状态、§4 完整裁决与失败分类、§0/产物索引/预算表更新 |
| `e68ea27` | 续行前置 Q1–Q4 证据修复(§0.2):profile 审计穿透+数组保留、recovery schema/加载器/双分支严格重放+目录约定统一、configs 回填仅 v2+integrity 补三文件、逐 QP 元数据一一对一+删补记、数组键单前缀+双前缀显式解码;测试 288→303+1s |
| `5bea3be` | 提交单位 2 执行基建(§0.3):§8.1 全家族完整门(full 模式缺证据即败)、§12 子任务状态机、正式 stage_sixseg/stage_train24+`--profile` 交叉核验、freeze_segment with_series 增量;测试 303→316+1s(新增 `tests/test_train24fix3_formal.py` 13 项,含 B 运行四失败侧逐值冒烟;表内原记 315 系笔误已订正) |
| (本提交) | 提交单位 3a 续行阶段入口(§0.4):stage_evidence(历史来源核验+续行 baseline 登记+8 子任务+逐项对照不强制)+ stage_failure_prefix(10 solve 绑定+2 anchor 锚漂,身份不符停止绑定)+紧凑 QP 序列化+锚漂曲线增量;测试 316→328+1s;正式运行在提交后冻结树执行 |

测试 262 → 288 passed + 1 skipped(阶段 A–C 完成时;`uv run --locked pytest tests/ -q` @ `84a7a7d`)→ **303 passed + 1 skipped**(Q1–Q4 后)→ **~~315~~ 316 passed + 1 skipped**(提交单位 2 后;订正见 §0.3)→ **328 passed + 1 skipped**(提交单位 3a 后)。288 档构成:`tests/test_train24fix3_profiles.py` 9 项(注册表矩阵/未知字段·非法值·非法继承拒绝、N0 默认=历史硬编码、逐帧门控仅 C2 激活帧、solve_qp 设置穿透 spy、trace 开/关纯度逐位、v2 记录 N0 回填/非 N0 携带拒绝、effective 记录携带 profile)+ `tests/test_train24fix3_numeric.py` 7 项(N0/N1 挂载=无 profile 逐位一致、N2 行变换 ξ 列纯 /B 且松弛列两表述同 −1、±INF 哨兵有限值比较、松弛罚权单位 10000·B 仅 C2 族、高精度同解与 s_m=B·s_norm、真实接受行 N2 下逐位不变、N3 mu 罚权仅 C2 松弛且缺 mu 显式错、tight N3≡N2)+ `tests/test_train24fix3_norm.py` 11 项(见 §4)。303 档新增 `tests/test_train24fix3_recovery.py` 15 项(见 §0.2)。316 档新增 `tests/test_train24fix3_formal.py` 13 项(见 §0.3)。328 档新增 `tests/test_train24fix3_continuation.py` 12 项(见 §0.4)。

首轮 baseline(`train24fix3_20261002_a7954e2/`)因驱动两处登记缺陷判 FAIL(fb/cf 重放本身已 ALL MATCH、integrity 干净):(1) 用 `closed_loop_metrics.sha256` 16 位截断变体比 F 清单 64 位摘要,必然假漂移——AGENTS 已记录的"跨工具哈希必须同源"事实的直接复演;(2) C 参照期望误写 6 NPZ(1/6 keep 轮仅 sub16 保留 1 个)。修复(`1147424`)后重跑为权威 baseline,首轮目录保留为失败尝试证据。

## 1. 阶段 A:baseline(PASS)

`baseline/`,运行时干净树(仅未跟踪 `.zcodeignore`,内容哈希+副本按方案 §12 登记入 `n0_identity.json`;git/env/assets/有效配置/integrity 清单同册)。

- **F 复核**(在算法修改前):fb 严格重放 **5/5 ALL MATCH**(integrity 干净)、cf 严格重放 **113/113 ALL MATCH 且 n==expected**(四视图距离交叉核验全 ok);重放输出写新目录 `baseline/{fb,cf}_replay/`,未触碰 reviewfix3 历史 `diag_replay/`。
- **249 产物清单活性**:全哈希一致(drifted=[])。
- **H/B/C/T 登记**:逐文件全长哈希入 `baseline_summary.json`;计数 H 5/B 6/C 1/T 13 全对。
- **出口条件满足**:基线身份/重放成立、参照完整、profile 可还原且未知字段被拒。

## 2. 阶段 B:五帧完整 trace(PASS)

`trace/`,运行 `a53b6f7`(trace 钩子默认关闭;N0 行为不变由纯度测试钉死)。每帧完整升级链(skip 拒→prediction→prev_accepted→slack)3 次尝试、70–94 个 QP 的逐迭代记录(状态/目标分项/分族违反/回溯/终止)与 QP 数组 NPZ 落盘。

**五帧失败身份逐项复现**(t_fail/状态 `infeasible`/term `no_merit_step_trust_min`/迭代数/QP 数与冻结 solver_fields 全等):sub10@44 26/26、sub12@12 27/27、sub3@14 20/20、sub7@32 31/31、sub8@20 27/27。

**已证实的收敛形态**(§2.2 解释 1/2 的证据;支持"内层求解不足",不支持"结构性联合不可行"):

- 最后尝试的全部恢复 QP 均终止于 `maximum iterations reached`(2500 耗尽),**prim_res ~1e-6~1e-5 而 dual_res 57–272**——原始侧近乎可行、对偶侧远未收敛,ADMM 对偶进度极慢(典型病态/对偶退化签名);
- 恢复松弛大量微负(42–76 个/解,均在 prim 残差内),返回解未达约束集;
- 外层线搜索仅能接受 ~0.1% 量级的 merit 下降,违反全部集中于 post 块(C2 行族),trust 在连续拒绝后塌缩至 `trust_radius_min`(0.00087890625)终止——µm 级 C2 残差未达 1e-6 接受门前耗尽路径。

## 3. 阶段 C:N1→N2→N3 数值候选(完成,winner=None)

`numeric/`,每 profile 五冻结帧升级链 + 完整 diagnosis6 quiet(单进程串行、线程 8、`--assert-task-gate 0.40`、三次运行 `code_changed_during_run=False`)+ §8 质量门。**三个 profile 的基础门(6/6 keep)全部未过,质量门随之未过**:

| profile | 五冻结帧单帧升级 | 六段 keep | 六段失败帧(t_fail) |
|---|---|---|---|
| N1(eps 1e-9/20000/polish) | **4/5 采纳**(sub3 拒) | **0/6** | sub3@9、sub10@32、sub12@13、sub16@30、sub7@35、sub8@21 |
| N2(N1+行预条件,代价保持) | 3/5 采纳(sub10/sub3 拒) | **1/6**(仅 sub16) | sub3@9、sub12@13、sub10@44、sub7@41、sub8@25 |
| N3(N2+C2 罚权=mu) | **4/5 采纳**(sub3 拒) | **0/6** | sub12@11、sub3@23、sub10@33、sub16@41、sub8@22、sub7@53 |

**裁决与机理**(按 §6.2 逐条):

1. **数值候选在冻结帧层面恢复了合法解**(N1/N3 4/5、N2 3/5 单帧升级链采纳;N3 下多帧由 prediction/prev_accepted 直接通过)——µm 级 C2 越界处的信赖域塌缩确属数值/权衡问题,与阶段 B 的"内层求解不足"证据一致。五帧通过数仅用于解释排序(§6.2 条 3),不构成整段结论。
2. **整段全部失败于新的前缀**:更高精度的内层改变首个 C2 激活帧的求解步,状态链自早期帧起漂移,各段在**更早或不同帧**失败(如 sub3 由 14→9/23、sub16 由 keep→30/41 拒)。"冻结帧成功≠整段成功"与方案 §1 的反向警示一致;不存在可用整段门抵消的单帧改善。
3. **N2 与 N0 同为 1/6**(sub16,443.8s):行预条件在 OSQP 内部 scaling=10 之下无显著行为差异,属方案 §6.1 预先声明的允许结果,未追加 epsilon/罚权组合。
4. 三次六段运行 R9 干净(运行期仓库零写入)、任务门断言 0.40 生效、报告内嵌 profile 解析值与注册表哈希。

**出口**:按 §6.2"N1–N3 全未过时进入阶段 D"。未做任何预算外候选搜索(无新增 epsilon/罚权/信赖域/外层预算组合)。

## 4. 阶段 D:范数分支 S1/S2(实施+预算调整续跑全部完成;**两者均未过六段开发门,winner=None,阶段按 §7.3 结束**)

实现提交 `3a23e92`(norm_cut 全量)+ 驱动修复 `afffdf4`(`_write_norm_products` 建目录;首版在首条产物即 FileNotFoundError、无部分写入污染)。测试 277→**288 passed + 1 skipped**(`tests/test_train24fix3_norm.py` 11 项:解析分配自由[范数 1.3mm≤1.5mm 过而逐轴 1.3>0.866mm 拒]、尖峰/界外拒、S1/S2 边界蕴含 0.05 进入门、盒行/切面行独立空间与预条件界对拍、共享 site 松弛单 mu 罚权、切面预算耗尽记 nonclosed、切面逐线性化清除、描述符/证据模式感知、v3 产物与索引篡改拒绝、axis 路径零变化)。续跑:预算调整登记 `33c8c2b`(S1 8h/S2 10h,串行自动,2026-10-03 用户授权),执行细节见 §0.1。

**S1(0.045)完整结果**:

- **五冻结帧 5/5 全部采纳**(sub10@44/sub12@12/sub3@14/sub7@32/sub8@20;墙钟 27–59s/帧,2026-10-03 确定性重跑与 2026-10-02 保全日志逐帧吻合 ±0.6s)——含全部数值候选都无法恢复的 sub3;范数形态在机理层面完全恢复五个失败帧,是本轮所有候选中第一个 5/5。
- **113 反事实全部完成**(v3 产物+双视图索引):分类 **41 PASS + 72 C2_ONLY**(axis 模式为 13+100——范数界多接纳 28 个历史候选);**十个尖峰候选 10/10 被真实范数行拒绝**(全 C2_ONLY);启用态核验通过(全部激活侧在 norm_cut 行下审计)。
- **六段(2026-10-03 续跑完成):0/6 keep**——t_fail 分别为 sub10@60、sub12@14、sub16@36、sub3@63、sub7@23、sub8@32,全部拒于 `state_constraints`;R9 干净、配置无漂移;墙钟 5432.6s(91 min)。质量门因 0 保留 NPZ 全败(H/B/C/T 参照段缺失记 FAIL)。`norm/S1/profile_summary.json` 组装落盘(continuation 溯源块:预算登记/墙钟口径/五帧重跑交叉核对/cf 冻结产物复原)。
- **失败签名与 axis/N 系同源**:全部段最后尝试终止于 `no_merit_step_trust_min`(trust 塌缩 0.00087890625)、最后 QP `maximum iterations reached`、违反行即 `hand_norm[side:site]` 范数行、归一化违反 4.9e-5~1.5e-4(µm 级骑界);多数段败于 C2 启用帧附近或携带期内(sub12/sub3 双侧 t=1 启用、sub7 在首个启用帧即败)。"冻结帧恢复"没有转化为整段成功,与方案 §1 反向警示一致。

**S2(0.049)完整结果(全套)**:

- **五冻结帧 5/5 全部采纳**(墙钟 27.6–59.5s/帧)。
- **113 反事实**:分类 **57 PASS + 56 C2_ONLY**(较 S1 再多接纳 16 个历史候选);**尖峰 10/10 仍被真实范数行拒绝**;启用态核验通过。
- **六段:0/6 keep,R9 干净**,墙钟 9611.2s(sub3 单段 5854.7s、处理完全部 118 帧)。质量门因 0 保留 NPZ 全败。**失败分两族**:
  - **信赖域塌缩族(4 段)**:sub10@60、sub12@14、sub7@23、sub8@33——与 S1 同签名(范数行 µm 级骑界、trust 塌缩、QP 迭代耗尽;sub8 的 t_fail 由 32→33,其余同 S1)。
  - **锚漂移族(2 段,S2 特有)**:sub16 与 sub3 **整段求解走通**(sub16 处理完全部 89 帧、sub3 全部 118 帧)但独立终检败于 `hand_grasp` 右手锚点——sub16 锚 t=40 失败;sub3 锚漂自 t=14 起累积至 t=85 失败(双侧 C2 携带 176/180 帧)。放宽速度界允许的每帧手部运动累积成锚漂超出抓握认证容差。
- `norm/S2/profile_summary.json` + `norm/norm_summary.json` 落盘(**winner=None、not_executed=[]、comparisons 含 S1/S2 全量**)。

**阶段 D 裁决(§7.3 预注册规则)**:S1/S2 均未通过全部六段开发门(基础 6/6 keep 与质量门全败)——**本轮算法候选比较结束,winner=None**。失败分类与最小问题规格:

1. **数值未收敛(信赖域塌缩)×4 段**(sub10/sub12/sub7/sub8):C2 激活帧上速度行 µm 级骑界 → 外层找不到 merit 下降步 → trust 塌缩终止。该签名在 N0–N3(逐轴+内层精度/预条件/罚权)与 S1/S2(范数形态+界值放宽)下全部复现——**再放宽界值或再提高内层精度都不解决**;冻结帧重放可解而整段新前缀不可解,问题在链级(到达状态与行族的联合权衡),最小规格指向恢复方向/merit 结构或外层接受策略的改变。
2. **锚漂移退化 ×2 段**(sub16/sub3):整段可解但右手抓握锚漂超认证容差——最小规格为手锚跟踪/锚行强度与 C2 速度界的联合权衡。
3. **证据边界**:质量门在 0 保留下不可计算;五帧 5/5、尖峰 10/10、cf 分类是机理证据而非整段结论;无"其他行阻塞/任务退化"类(未达终检段不可测)。

下一策略变化(启用语义、时间段优化、跟踪权重或替代机制)须**下一份预注册方案**;不得因局部成功降低六段门进入 train24。

## 5. 阶段 E/F/G

未进入(阶段 D 已于 2026-10-03 完成,但 winner=None——§7.3 阻止转段)。六段/train24 正式验收与生产交接未执行,§8/§9 的门不因本轮单帧/反事实改善而宣称任何通过;下一策略变化须新的预注册方案(见 §4 失败分类与最小问题规格)。
