# train24fix3 收敛修复与转段执行验证记录

日期:2026-10-02 起,2026-10-03 增补续跑指针。对象:[train24fix3 收敛修复与教师数据转段执行计划](../plans/train24fix3_convergence_recovery_plan.md)。状态:**阶段 A/B/C 完成(N1–N3 全未过六段开发门,winner=None);阶段 D(norm_cut S1/S2)实现完成、全量 288 passed + 1 skipped;S1 机理证据完成(五冻结帧 5/5 采纳、113 反事实、尖峰 10/10 全拒),S1 六段按 §12 预算中止记 INCOMPLETE(非门裁决),S2 未执行——等待预算调整登记后续跑**。实验根目录:`derived/data/teachers_regenerated/train24fix3_20261002_1147424/`。

起点:代码 `e74a5f3`(方案提交 `dcf9d33`)。阶段 A 按方案 §4.1 任务 1 在任何算法修改前执行。

## 0.1 续跑指针(新上下文入口——2026-10-03 写,以此为准继续工作)

**一句话现状**:算法代码全部就绪且测试全绿;S1 的机理层证据(五帧/113 反事实/尖峰)已拿到且是本轮最强结果;唯一未完成的是 **S1 的六段比较(预算中止)与 S2 的全部比较**,完成后才能出阶段 D 的 winner 裁决。下一步动作是**登记预算调整并续跑六段**,不需要任何新的算法实现。

**精确续跑步骤**(§12 规则:同冻结配置/profile、调整预算单独登记、续跑保持完整分母):

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
| `norm/S1/cf/` | 113 个 v3 产物+cf_index | 完成 |
| `norm/S1/sixseg/rq/` | 仅 dirty_diff patch(六段 51 min 中止) | **INCOMPLETE,待续跑** |
| `norm/stage_status_budget_stop.json` | 预算中止时间线+续跑范围 | 完成 |
| `norm/norm_summary.json` | 阶段 D 裁决 | **不存在(阶段未完成)** |

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

**实现落点清单**(全部已提交、测试覆盖):

- `src/ireset/repair/profiles.py`:`OsqpSettings`/`C2Profile`/`load_profiles`(闭合字段/继承展开/非法值拒绝)/`profiles_yaml_sha256`;注册表 `configs/repair/train24fix3_profiles.yaml`(N0–N3/S1–S2)。
- `src/ireset/repair/scp.py`:`RepairConfig.osqp_*`(默认=N0)、`_frame_qp_settings`(profile 仅 C2 激活帧)、`solve_qp(settings=None)`+trace、`_fixed_post_rows`(axis/norm_cut 双模式真实行)、`_post_row_descriptors`(模式感知缓存)、`build_qp`(C2 盒行/切面行/`_c2_lin` 线性化/共享 site 松弛组/mu 罚权)、`_c2_cut_refine`(切面循环+预算+账目)、`_qp_row_names`、`solve_frame`(trace 插桩+切面接线+逐迭代清缓存)、`FrameResult.diagnostics["c2_cuts"]`。
- `src/ireset/repair/pipeline.py`:`RepairPipeline(..., c2_profile=...)`。
- `scripts/regenerate_teachers.py`:`--profile`(regen-only 守卫;解析值+注册表哈希入 effective 记录与报告)。
- `scripts/train24fix2_audit.py`:`configs_from_record` v2 回填(v2 携带非 N0 新字段即拒)、`fresh_audit_pipe/full_candidate_audit` 的 `c2_profile` 参数。
- `scripts/train24fix3_quality_gate.py`:H/B/C/T 逐侧不退化门+sub16 保护。
- `scripts/train24fix3_convergence.py`:baseline/trace/numeric/norm 阶段 + v3 产物读写。
- 测试:`tests/test_train24fix3_profiles.py`(9)/`test_train24fix3_numeric.py`(7)/`test_train24fix3_norm.py`(11)。

**预算调整登记表**(授权后填写):

| 日期 | 项 | 原预算 | 调整为 | 依据 | 授权 |
|---|---|---|---|---|---|
| (待登记) | S1 六段续跑 | §12 norm 总 2h(已耗尽) | (建议 ≤8h) | 首段实测 51min 未完,~10× 成本观察 | (用户) |
| (待登记) | S2 全套 | 同上 | (建议 ≤8h) | 同上 + S1 冻结帧/cf 实测 ~19min | (用户) |

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

测试 262 → 288 passed + 1 skipped(最终;`uv run --locked pytest tests/ -q` @ `84a7a7d`):`tests/test_train24fix3_profiles.py` 9 项(注册表矩阵/未知字段·非法值·非法继承拒绝、N0 默认=历史硬编码、逐帧门控仅 C2 激活帧、solve_qp 设置穿透 spy、trace 开/关纯度逐位、v2 记录 N0 回填/非 N0 携带拒绝、effective 记录携带 profile)+ `tests/test_train24fix3_numeric.py` 7 项(N0/N1 挂载=无 profile 逐位一致、N2 行变换 ξ 列纯 /B 且松弛列两表述同 −1、±INF 哨兵有限值比较、松弛罚权单位 10000·B 仅 C2 族、高精度同解与 s_m=B·s_norm、真实接受行 N2 下逐位不变、N3 mu 罚权仅 C2 松弛且缺 mu 显式错、tight N3≡N2)+ `tests/test_train24fix3_norm.py` 11 项(见 §4)。

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

## 4. 阶段 D:范数分支 S1/S2(实施完成;S1 证据完成、六段预算中止 INCOMPLETE;S2 未执行)

实现提交 `3a23e92`(norm_cut 全量)+ 驱动修复 `afffdf4`(`_write_norm_products` 建目录;首版在首条产物即 FileNotFoundError、无部分写入污染)。测试 277→**288 passed + 1 skipped**(`tests/test_train24fix3_norm.py` 11 项:解析分配自由[范数 1.3mm≤1.5mm 过而逐轴 1.3>0.866mm 拒]、尖峰/界外拒、S1/S2 边界蕴含 0.05 进入门、盒行/切面行独立空间与预条件界对拍、共享 site 松弛单 mu 罚权、切面预算耗尽记 nonclosed、切面逐线性化清除、描述符/证据模式感知、v3 产物与索引篡改拒绝、axis 路径零变化)。

**S1(0.045)已完成证据**:

- **五冻结帧 5/5 全部采纳**(sub10@44/sub12@12/sub3@14/sub7@32/sub8@20;墙钟 27–59s/帧)——含全部数值候选都无法恢复的 sub3;范数形态在机理层面完全恢复五个失败帧,是本轮所有候选中第一个 5/5。
- **113 反事实全部完成**(v3 产物+双视图索引):分类 **41 PASS + 72 C2_ONLY**(axis 模式为 13+100——范数界多接纳 28 个历史候选);**十个尖峰候选 10/10 被真实范数行拒绝**(全 C2_ONLY);启用态核验通过(全部激活侧在 norm_cut 行下审计)。
- **六段 INCOMPLETE(预算中止,非门裁决)**:按 §12"总预算 2 小时"于六段首段(sub10)进行 51 分钟、无一段完成时停止(工作进程全程 100% CPU,非挂死);无 keep/质量结论,不以部分结果判门。成本观察:norm_cut 在 N3 档求解设置下 C2 激活帧成本 ~10× 于 N3(切面重解 × 20000 迭代 QP),冻结帧墙钟 27–59s vs N3 的 3–13s——全成本已入账。
- **S2(0.049)未执行**(预算线到达)。状态文件 `norm/stage_status_budget_stop.json` 登记完整时间线与续跑范围。

**续跑范围(§12:续跑同冻结配置、调整预算单独登记)**:S1 六段完整重跑(按实测节奏估计 ≥5h)+ 质量门;S2 五帧+113 反事实+六段+门。在六段完成前,阶段 D 无 winner 裁决、阶段 E/F 不进入;"五帧 5/5 + 尖峰 10/10 + 分类 41/72"是机理证据,不是整段结论。

## 5. 阶段 E/F/G

未进入(依赖 D 六段完成与门)。六段/train24 正式验收与生产交接未执行,§8/§9 的门不因本轮单帧/反事实改善而宣称任何通过。
