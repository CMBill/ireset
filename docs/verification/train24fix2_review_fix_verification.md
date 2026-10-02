# train24fix2 实现审阅修复验证记录

日期:2026-10-01。对象:[train24fix2 实现审阅修复方案](../plans/train24fix2_review_fix_plan.md)(R1-R5)。审阅基点 `0500b4a`(独立复跑 225 passed + 1 skipped);实施提交 `2aef500`(方案)→ `a0958df`(订正块)→ `a0bf9d5`(R2)→ `fa341e6`(R4)→ `bab948a`(R5)→ `73814d6`(R1+R3)→ `048523d`(skip 键形修复)→ `2158eb2`(对照脚本);全量测试 **248 passed + 1 skipped**。产物目录:`derived/data/teachers_regenerated/train24fix2_reviewfix_20261001/`(`source_manifest.json` + `diag/` + `sixseg/`);冻结参照只读未动。

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
