# train24fix2 实现审阅修复验证记录

日期:2026-10-01。对象:[train24fix2 实现审阅修复方案](../plans/train24fix2_review_fix_plan.md)(R1-R5)。

> **2026-10-02 订正指针:本文 §0-§8 的取证产物(`train24fix2_reviewfix_20261001/diag/`)经第二轮审阅(§9,缺陷 D1-D5)后被 `train24fix2_reviewfix2_20261002/` 取代**——其中 D3 为证据字段错位(完整场景距离双侧同值且退一帧),§6 的全部结论已在修正工具下按 §9.2 复验成立;§7 阶段 C(sixseg)不受影响。审阅基点 `0500b4a`(独立复跑 225 passed + 1 skipped);实施提交 `2aef500`(方案)→ `a0958df`(订正块)→ `a0bf9d5`(R2)→ `fa341e6`(R4)→ `bab948a`(R5)→ `73814d6`(R1+R3)→ `048523d`(skip 键形修复)→ `2158eb2`(对照脚本);全量测试 **248 passed + 1 skipped**。产物目录:`derived/data/teachers_regenerated/train24fix2_reviewfix_20261001/`(`source_manifest.json` + `diag/` + `sixseg/`);冻结参照只读未动。

## 0. 来源冻结与订正入口(方案 §7.A)

- `source_manifest.json`:rq/rv/ns 三目录 + 旧 `train24fix2_20260927_diag/` 全部 33 个文件 SHA-256 + H/B/T 索引嵌入;本目录为本轮唯一写入目标。
- 原验证记录顶部 §0 订正块:R1 上下文缺陷作废 §3.2 归因(待重验)、撤回 §1.4"无任何异常查询失效"、§3.2/§3.4"联合不可行已证实"降级为"数值/搜索未收敛";rq 1/6 与暂停裁决不变。

## 1. R5:缓存陷阱可证伪性(先行)

- 实证:原测试场景(n_sub=3,内点 α=1/3、2/3)下,移除"每点重扫"的前修复 `_probe_audit` 变体仍 `pass=False`——违规在首个内点就可见,全新求解器首次 `check_rows` 本来就会扫描,端点缓存从未建立,"每点重扫"未被考验。
- 补强后测试:先在候选端点 `check_rows` 建立缓存(断言无 robot_box 候选、`_frame_pairs` 非 None),箱体仅在 α=1/3 中间态扫掠机器人;当前实现 `pass=False` 且报告中间 pair/alpha;前修复变体在相同端点缓存下 α=1/3、2/3 全 `viol=0`(漏检)——测试对旧缺陷失败,可证伪性成立。

## 2. R2:查询异常显式拒绝

`classify_distance_query`(hit/truncated/nan/pos_inf/neg_inf/out_of_range)+ `C2DistanceQueryError`;`_collect_prev_scene_dist` 逐 geom 套用契约,任一 geom 异常即抛出(t/side/geom 名/distmax/value_kind/raw_repr/stage);`_run_sequential` 在采集点捕获并以 `status=rejected_frame`、`final_stage=c2_distance_query_failed`、`reason=c2_distance_query_anomaly` 拒绝,`_pass_record`/cost 全保留;`query_miss` 分列 `side_partial_miss`/`side_all_miss`;`_c2_diag` 增 `query_errors`(报告行透出);`regenerate_teachers` 仅 accepted 写 NPZ(拒绝段无 keep 产物)。E 工具 `prev_scene_distances` 同契约,异常即中止取证裁决。

测试 10 项:NaN/±Inf/超界四变体三帧场景从 **accepted 复现→显式拒绝**(审阅复现的反向钉死)、单 geom 异常毒化整侧、异常抛出变体、正常截断/真实负距命中不变、末帧异常、suffix-retry 遍异常、标准模式/参数 0 零查询零拒绝。

## 3. R1:完整冻结上下文与单帧升级等价

`train24fix2_audit.freeze_context` 按 dataclass 字段全量深拷贝(数组独立副本、稀疏 mesh 保留),`context_diff` 按字段位级核对(新字段自动覆盖,手写子集即 R1 缺陷);`context_arrays/meta/scalar_leaves/from` JSON+NPZ 往返位级一致。`escalate_frame` 复现生产帧升级(V1 skip 检查+失败回流→prediction→prev_accepted→slack,经 `pipe._solve_attempt`,冻结上下文,不从候选重推参考/锚点)。

测试:冻结上下文携带人体参考(机器人 FK 替换被字段核对识破)、后续 FK 不可污染、往返位级、全字段独立副本;升级等价三类受控场景对拍(求解采纳候选**逐位相等**、skip/t=0、三尝试全败+回流实证 `_extra_pairs` 非空+生产 attempt 链一致)。

## 4. R3:完整候选审计与分类

`full_candidate_audit` 五项检查(输入与 V1[实际检查候选明确]、端点全行[新全量扫描+描述符+单一归一化违反和]、全部插值点[每点清缓存重扫、通过点亦记录、probe_mode/band_margin 0]、任务[冻结人体参考下未舍入 MPJPE+0.40 门]、前缀独立终检[新 validator、exclude box_ground、结论仅覆盖前缀]);接受式为全部有限界行**归一化违反总和 ≤1e-6**(反例钉死"各族独立相与"误读);§4.2 分类表(EVIDENCE_INCOMPLETE/V1_UNUSABLE/PASS/C2_ONLY/CURRENT_ONLY/BOTH);产物 `write_products` 版本化 JSON+NPZ、拒绝覆盖;`--replay` 从冻结文件复算同候选双分支并与保存裁决比对。反事实"叠加 C2"分支调用**真实部署装配**(cfg 0.02598+ctx 启用态),当前链基准分支显式 `regen_hand_speed_mps=0.0`;历史 Nq 候选为移植帧,来源 NPZ 哈希+帧号为身份,不伪造 attempt/check ID。

## 5. R4:终止事实与结论强度

`FrameResult.term_reason/iter_budget` 于各出口实录(converged/budget_exhausted/no_merit_step_trust_min/restoration_qp_failed_trust_min);`status` 保持历史词表;attempt 记录与 rejected_frame 诊断携带。分类与文档按 §5 强度表表述:预算耗尽≠联合不可行。

## 6. 阶段 B 取证重验(正式运行,rev `048523d`)

### 6.1 E 阶段补齐(`diag/`,81 断言全过 + 113 反事实,308 s)

- §1/§2.4 冻结证据在 R2 查询契约下**零回归**(80/81 断言行 + 汇总,与原轮一致)。
- 完整审计分类:**尖峰 10/10 C2_ONLY、历史认证 13 PASS/90 C2_ONLY**——与旧分类一致,现每项均带端点全行/插值/严格速度/任务/前缀终检完整证据。
- 冻结重放 `--replay`:113 条记录**全部 matches_saved**(分支逐项:viol_sum/task/inter/prefix)。

### 6.2 五段失败绑定(`fail_bind2_*.json`,身份四项全配)

| 段 | t_fail | C2-on 违反(总和) | C2-off | 任务 MPJPE(正确人体参考) | 插值/速度/任务/前缀 | 关 C2 重解 |
|---|---|---|---|---|---|---|
| sub10 | 44 | hand_speed×2(4.0e-3) | 0 违反 | 0.068866 | 全过/全过/过/过 | prediction 采纳 |
| sub12 | 12 | hand_speed×3(1.8e-2) | 0 | **0.114154** | 全过 | prev_accepted 采纳 |
| sub3 | 14 | hand_speed×5(3.5e-2) | 0 | **0.131966** | 全过 | prev_accepted 采纳 |
| sub7 | 32 | hand_speed×3(6.9e-3) | 0 | 0.169982 | 全过 | prev_accepted 采纳 |
| sub8 | 20 | hand_speed×4(3.4e-3) | 0 | 0.091423 | 全过 | **skip 采纳** |

- **R1 §2.3 验收值逐位命中**:sub12@12=0.114154、sub3@14=0.131966(审阅重放值);旧缺陷版本为 0。
- 五段分类均为 **C2_ONLY**(完整判据:C2-off 分支端点+插值+速度+任务+前缀全过,C2-on 仅 hand_speed 行违反);关 C2 重解候选独立审计 `resolve_c2_off` 分支全过。
- 求解终止事实(R4):五段最后尝试 `status=infeasible` + `term_reason=no_merit_step_trust_min`(信赖域收缩至最小值以下,非预算耗尽)——**按 §5 强度表,这支持"固定流程下未找到可接受解",不构成结构性联合不可行证明**;sub8@20 的 C2-off 重解直接 skip 通过,同帧 C2-on 三级失败,差距是确定性的。
- 冻结重放:五段 on/off 双分支 matches_saved 全真。

### 6.3 结论边界

候选 C2_ONLY、同帧重解成功、前缀通过、整段接受是四个不同结论;本轮证据到前两项为止。六段 1/6 与暂停转段裁决不变。

## 7. 阶段 C:六段有限合法路径对照(rev `2158eb2`)

单进程串行、线程环境 8、`uv run --locked`、`--assert-task-gate 0.40`(未传任务门 YAML),三次运行 `code_changed_during_run=False`、`config_unchanged=True`、`git_rev=2158eb2`;rq/rv 段计 112.6/112.8 s、ns 27.5 s。

### 7.1 rq 对照冻结 `train24fix2_20260927_sixseg/rq`

**逐段完全一致**:accepted/t_fail(44/12/14/32/20)/final_stage(state_constraints)六段全同;九项可比算法计数(attempts/qp_calls_total/iterations_total/slack_attempts/backtracks_adopted/backtrack_probes/checks_total/processed_frame_visits/unique_processed_frames)**六段逐项相等**;C2 frames_on 逐段逐侧相等(sub10 r:16、sub12 l/r:12/12、sub16 r:15、sub3 l/r:14/14、sub7 l/r:10/9、sub8 l/r:4/5);V1 checks/normalized/budget_failures 逐段相等(sub12 budget_failures=3 与旧一致)。**keep 1/6**(sub16)。

### 7.2 rq 与 rv 确定性

五段拒绝 t_fail 逐段相同、计数逐段相同;sub16 NPZ SHA-256 前缀 `73119676…` 与 rv **字节一致**,且与冻结轮 sub16 NPZ **字节一致**——有限合法查询下本轮全部生产行为零变化。

### 7.3 ns(标准模式)三方对照

新 ns vs 旧 C2 轮 ns vs B 前置链 ns:六段 accepted 全 True、九项算法计数**三方逐项相等**;V1 全零操作(normalized=0/budget_failures=0/output_modified=0);n_kept 0/0(标准模式 keep 筛选,与旧一致)。**标准模式零新增 C2 查询**(ns 行无 c2 块)。

### 7.4 query_miss 分列核对(方案 §3.2/§7.C)

sub16 新记录 78 条 = **45 side_all_miss(与冻结轮计数全等)+ 33 side_partial_miss**(0.98-1.28 m 远离手的近端 geom 真实命中、同手远端 geom 截断——部分/整侧分列为方案 §3.2 明确要求的记录补充分列,行开启行为不变:frames_on 与旧逐侧相等);其余五段 query_miss=0。**全部段 query_errors=0**;已知 sub16 远离帧未被新错误分支拒绝(accepted 保持)。

## 8. 验收清单核对(方案 §8)与阶段裁决

- [x] **R1**:冻结/恢复逐字段位级一致(含稀疏 mesh);on/off 仅差 C2 两字段;升级路径三类受控场景对拍(采纳候选逐位相等、attempt 链一致、回流实证);五段来源绑定身份四项全配(t_fail/last_attempt_id/last_check_id/status),sub12@12、sub3@14 任务 MPJPE **0.114154/0.131966 与审阅重放值逐位一致**(旧缺陷版本为 0)。
- [x] **R2**:NaN/±Inf/超界/异常五变体全部显式拒绝且成本保留;正常截断/真实负距命中/禁用态行为不变;query_miss 分列部分/整侧;E 工具同契约。
- [x] **R3**:五段+113 个 E 候选全部完成规定审计;每个分类均带端点全行/插值/严格速度/任务/前缀终检与可恢复 JSON+NPZ;`--replay` 复算 113+5 记录 **ALL MATCH**(不依赖旧汇总值)。
- [x] **R4**:term_reason/iter_budget 全出口实录;五段最后尝试均为 `infeasible`+`no_merit_step_trust_min`(信赖域塌缩,非预算耗尽);验证记录/AGENTS/结构化产物按 §5 强度表表述;原暂停裁决保留。
- [x] **R5**:缓存陷阱测试在前修复变体上失败(端点缓存建立后 α=1/3、2/3 全 viol=0);新增反例 23 项;全量 **248 passed + 1 skipped**。
- [x] **六段**:有限合法路径状态/候选/计数/轨迹零变化;rq≡rv;ns 计数不变;三次运行零漂移;旧资源/参照哈希不变(source_manifest 33 项冻结只读)。

**阶段裁决(方案 §8 末条):审阅修复与证据重验完成;六段仍为 rq 1/6,原 C2 修复目标(认证边缘碎裂恢复)未达成,继续暂停转段。**下一步须依据订正后的证据(五段 C2_ONLY 全证据链 + 信赖域塌缩终止事实 + sub8 关 C2 即 skip 通过的确定性差距)另立数值收敛或策略变化方案并预注册;本轮不选择替代机制、不继续搜索变体、不重跑 train24、不重标阈值。

## 9. 第二轮审阅修复(reviewfix2,2026-10-02)

对上轮取证工具的第二轮审阅发现五项缺陷(D1-D5),审阅基点 `143aebb`(248 passed + 1 skipped),实施提交 `28ab568`(全量 **257 passed + 1 skipped**,新增 9 项反例 `tests/test_train24fix2_review_fix2.py`)。产物目录 `derived/data/teachers_regenerated/train24fix2_reviewfix2_20261002/`(`diag/` 产物 + `diag_replay/` 严格重放摘要 + `source_manifest.json`);**上轮 `train24fix2_reviewfix_20261001/diag/` 的 fail_bind2_* 与 cert_trace_cf_* 产物自此被取代**(D3 证据字段错位 + D2/D4/D5 弱重放契约),其 `sixseg/` 阶段 C 对照不受本轮缺陷影响、结论保留。

### 9.1 五项缺陷与修复

- **D1 绑定身份未对照原报告(循环自比)**:`bind()` 原实现比较本次重跑与本次快照,`rq_dir` 未用。现 `identity_vs_report` 逐项对照 `--rq-dir` 冻结原报告行的 `t_fail`/`rejection.last_attempt_id`/`last_check_id`/`last_attempt.status` 四项,任何不匹配即 `IDENTITY_MISMATCH` **停止绑定**(不写审计,`identity_failed_items` 命名失败项);运行内部一致性检查保留为第二道。反例:四项各自被篡改(t_fail=999 等)逐项翻假;受控场景下伪造行→停止、真实行→进入完整审计。
- **D2 重放仅比五项汇总值**:`deep_diff`/`arrays_diff` 全记录逐叶+数组逐位比对(行描述符、每个插值点明细、严格速度超限、前缀最坏值、重解候选审计、分类、结论强度);CLI 判据含 `classification_matches` 与完整性;fail_bind 重放**重算单帧升级(含 QP)与重解候选审计**;E 重放同理。反例:篡改描述符/前缀最坏值/插值点值/重解 term_reason/分类/行数组(哈希同步修好)各自翻 MISMATCH。
- **D3 113 个 E 产物的完整场景距离字段全部错位**:旧实现把所选单侧的 `[t-1]` 写入双侧(序列 `[t]` 本已是前一完整场景)。现 `prev_scene_flags` 按 `[side]["scene"][t]` 冻结双侧,记录 `prev_scene_dist_both_m` 与 `prev_scene_source`(NPZ 路径+哈希);重放对哈希核验过的源 NPZ **新鲜查询逐侧交叉核验**。修正值:sub10@44 l=`0.0012341966700727114`、r=`0.0012351612733721542`(审阅期望值逐位命中;旧字段双侧同为 `0.0017623104519771862`=[43])。行门控只读 `c2_hand_enabled`(`hand_speed_rows_active`),故旧分类不受影响,证据字段重生成后分布不变。
- **D4 冻结重放用当前配置、加载无完整性校验**:产物内嵌 `effective`(repair/repair_no_c2/validation 三配置)+`integrity`(16 项代码/配置/资产 SHA-256)+`npz_sha256`(NPZ 先写、哈希入 JSON);`load_products` 拒篡改 NPZ 与 pre-v2 产物;重放 `configs_from_record` 只从冻结值重建配置(现 YAML 漂移不改重放计算),`verify_integrity` 报告代码/配置/资产漂移并判不过。反例:任务门 0.20 漂移下重放不变(冻结生效);改产物内冻结门限/速度界→MISMATCH;改保存数组(哈希修好)→数组比对翻假。
- **D5 二次重放吞入自身摘要**:候选发现改 `fail_bind2_index`/`cert_trace_cf_index` 生成时预期清单(缺失/空清单显式拒绝,不再 glob);重放输出一律写 `<diag>_replay/` 独立目录。反例:产物目录放置无格式摘要/诱饵文件被忽略;二次重放稳定 ALL MATCH;产物目录零重放文件。

### 9.2 正式重生成与严格重放(rev `28ab568`,干净树)

- **fail_bind 五段:对照原报告身份四项全配**(`id_vs_report=True` ×5;t_fail 44/12/14/32/20、分类 C2_ONLY、`infeasible/no_merit_step_trust_min`、关 C2 重解全部采纳)——上轮阶段 B 结论在修正后的身份校验下成立。
- **cert_trace 全量**:302 s、81 断言全过、113 反事实重生成,分类分布不变(**13 PASS + 100 C2_ONLY**;尖峰 10/10 C2_ONLY)。
- **严格重放(独立目录)**:fail_bind 5 记录、cf 113 记录**双双 ALL MATCH**——每条含完整记录逐叶、数组逐位、重解升级重算、冻结配置、完整性清单、分类一致;cf 侧 113 次 prev 场景距离新鲜查询交叉核验全过;二次执行重放仍 ALL MATCH(索引驱动,摘要不再回灌)。
- `source_manifest.json`:249 项产物哈希 + 冻结输入活性复核(4 个目录标签与上轮逐哈希一致、HBT 内容冻结的 provenance 文件与 24 项索引条目 NPZ/教师输入截断哈希活性重验)。

### 9.3 裁决

第二轮五项缺陷全部修复且可证伪;上轮全部取证结论(五段身份、C2_ONLY 分类、任务 MPJPE 逐位值、六段零变化)在修正工具下**复验成立**,唯证据字段按 D3 订正。**原裁决不变:审阅修复与证据重验完成;六段仍 rq 1/6,原 C2 修复目标未达成,继续暂停转段。**

### 9.4 新增工程事实

- 仓库工具存在两种 SHA-256 变体:`closed_loop_metrics.sha256` 为 16 位截断(provenance 索引内嵌),`experiment_provenance`/产物清单为完整 64 位——跨工具哈希比对必须同源,否则出现"哈希不符"假阳性(本轮 source_manifest 复核即踩中)。
- 完整记录级确定性实证:113 候选双分支审计 + 五段单帧升级(含 QP 重解)在同一提交状态下全记录逐叶、数组逐位可复现——此前仅五项汇总值被比对,正是 D2 类缺陷的藏身处。
