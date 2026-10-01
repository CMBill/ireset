# train24 失败类别最小修复验证记录

日期:2026-09-27 起。对象:[train24 失败类别最小修复方案](../plans/train24_failure_fix_plan.md)(修订版)。本记录随各阶段实施递增;未完成的节标注状态。取证产物目录:`derived/data/teachers_regenerated/train24fix_20260927_diag/`(完整结论 JSON:`D_CONCLUSIONS.json`,含 2026-09-27 审阅二轮订正块)。

> **2026-09-27 审阅二轮订正(优先于下文首轮记录)**:四项缺陷(P1 截断距离可绑定走廊、P2a 失败快照混尝试、P2b 离线插值未逐点重扫、P2c verbose 日志 adopted 冻结为 None)已修复并分项提交(`66343e1`/`dbe587a`,全量 **213 passed + 1 skipped**)。**订正一(D-F 分类)**:以修复前代码(68ba62b worktree+P2a 外科补丁)经修正工具重推导,sub7_029 由 foot_speed **改判 band_floor_only**(脚速违反属于第一次尝试的候选;最后执行尝试的候选仅违反手带行 1.66e-5 m)——修正后 band_floor_only **×3**(sub3_001/sub8_002/sub7_029)、mixed_joint_limit(sub12_004)、deep(sub16_012)、foot_speed(sub10_075),F1 启用证据增强并经身份绑定+逐点重扫+前缀终检三重前置严格复现。**订正二(D-R 计数)**:原 r2_decide=107/91 被日志 adopted 冻结污染,作废;r2_any(skip 检查口径,107)与零 QP 反例仍为有效 R2 触发证据;修正工具在当前代码上 r2_decide/r2_any/r1 全 0(R2 完全闭合)。**订正三(六段重跑)**:首轮六段(P1 缺陷在场)归档为 `train24fix_20260927_sixseg_v1_superseded/`;修复后重跑逐项复现同一判决(见 §5),结论现为干净证据下的权威版本。

## 1. D 阶段:只读取证(已完成;分类经审阅二轮订正)

提交:`<D>`。改动:pipeline 新增**默认关闭**的只读快照钩子(`RepairPipeline.snapshot`/`_snap`,采纳/失败/终检失败事件;186 项测试全过,无行为改动)+ 取证脚本 `scripts/train24fix_diagnostics.py`(dv/df/dr/dq)。重放均以独立 evaluator/solver 复算,不触正式运行缓存。

### 1.1 D-V:速度微超限采纳路径绑定

三段重放与冻结证据**逐位一致**(`replay_matches_frozen_evidence=true`,含后缀重试 pass)。绑定结果:

| 段 | 帧/关节 | 采纳路径(pass 0 与 pass 1 相同) | V1 预算需求 |
|---|---|---|---|
| sub7_000 | 101 / 26 | **repaired(紧 QP,20 次,slack=False)** | 1.34e-8 rad |
| sub12_077 | 203 / 16 | repaired_slack(slack=True,29 次) | 3.52e-9 rad |
| sub8_048 | 12 / 26 | **repaired(紧 QP,38 次,slack=False)** | 1.03e-10 rad |

结论:三处全部来自**求解采纳候选**(紧 QP 与 slack 两路径都观测到;紧 QP 同样留 eps 残差——修正了"仅 slack 骑界"的预设);无一来自 skip 采纳;全部远小于 V1 预算 4e-7 rad。V1 须覆盖全部采纳入口,证据不支持排除任何入口。

### 1.2 D-F:六段全约束复算与反事实判定

在冻结失败候选上仅替换 §2.4 指定格(regen 端点 accept 带行下界 −1.5→−1.8 mm),其余行与全部探测界不动:

| 段 | t | 分类(订正后) | 依据 |
|---|---|---|---|
| sub3_001 | 111 | **band_floor_only** | 仅带行 µm 级越界(7.35e-6/3.56e-6 m);反事实通过端点+探测+速度+任务门;前缀终检通过 |
| sub8_002 | 62 | **band_floor_only** | 仅 ankle 行 2.5e-7 m 越界;反事实全过 |
| sub7_029 | 18 | **band_floor_only**(由 foot_speed 改判) | 最后执行尝试候选仅违反手带行 1.66e-5 m;脚速违反属第一次尝试候选(快照身份订正后绑定最后尝试) |
| sub12_004 | 0 | mixed_joint_limit | `joint_limit[j16]` 骑界 2.76e-7 rad + 四条带行 µm 级;非指定格的行仍挡(反事实不过) |
| sub16_012 | 40 | deep_penetration | 手带行 −3.5~−4.0 mm,超终检门 |
| sub10_075 | 69 | foot_speed | sole2:z 越界 9.4e-4 m/frame(实质性,非 eps 类) |

**F1 启用条件成立且经修正工具在修复前代码上严格重验**(band_floor_only ×3,身份绑定+逐点重扫+前缀终检全部通过)。残留类别(脚速 1 段、关节限位骑界 1 段、深穿透 1 段)本轮不设专属策略(方案 §8),如实入账。帧级反事实通过≠整段恢复,以六段回归与 train24 重跑裁决。注:独立 evaluator 复算的部分末候选行值与冻结报告在 µm 级有差(进程内求解器状态),分类不受影响。另:在**当前修复版本**上 df 重放,6 段中 5 段直接 accepted(仅 sub16_012 深穿透仍拒)——F 类失败在组合修复下整体消失,band_floor_only 归因仅存于修复前状态。

### 1.3 D-R:走廊/全入口审计与零 QP 反例

- **R2 触发**:ref_hold 侧手距 5.7~11.8 cm(超出 ~5 cm 候选准入)时主动行不存在——sub10_001 决定性检查 107 帧、sub10_003 91 帧(任一入口 107/43)。
- **零 QP 反例(正式保存)**:sub10_001 帧 30/31/32,ref_hold 含双侧手、手距 9.4~12.6 cm,新求解器返回 accepted=True、**qp_calls=0**、状态逐位不变(可行+任务过+`_stationary` 提前返回)——"调用 solve_frame"确实不等于走廊参与。
- **R1 未独立触发**:无任何已采纳帧违反已激活走廊的上界。R2 行就位后,既有检查链(skip 检查、求解可行性、success_review、pipeline 采纳)即构成走廊契约;R1 落为契约测试(方案 §4.3"满足则复用")。

### 1.4 D-Q:质量基线冻结

18 段(六段 rq 5 NPZ + train24 13 keep)按标签协议 v1 复算并入 `dq_quality_baseline.json`(含输入哈希)。方案 §5 已知下限逐项复现:sub10 交集 0/15、近接 33/63(参考 33/74);sub8 交集 29/41、近接 77/78(参考 80/80)。

### 1.5 启用结论

V1 ✅、F1 ✅(band_floor_only×2)、R2 ✅;R1 为契约验证(经 R2 行,不另建距离门)。

## 2. V1:有界候选规范化(已实施,提交 `8a3ac9a`)

`normalize_joint_velocity` + `strict_joint_speeds`(与终检逐位同式 `abs(diff)/(1/fps)`):已过关节逐位不动;越限关节取 `q_prev ± dt·limit` 边界、同式复算仍超则 `np.nextafter` 内收(≤8 次);总预算 `dt·limit·1e-6 = 4e-7 rad`,超预算**不部分投影**(skip 转正常求解、求解候选按既有失败/重启流程、回溯候选跳过该 alpha)。覆盖 skip/prediction/prev_accepted/slack/回溯/后缀重试全部采纳入口,规范化先于完整接受检查(检查的状态=采纳的状态);修改 `res.z` 即失效 `frame_check`。`_frame_valid` 新增独立严格关节速度门(反例=审阅浮点例 0.8+0.4→12.000000000000004,naive clip 严格不过)。`velocity_projection` 计数(检查/修正/预算失败/最大幅度/输出被修正帧数)入诊断与报告行。11 项测试(浮点反例、sub7 冻结值 12.000000401362618→严格通过、预算边界/超限/多关节全或无、负方向、严格门、skip 端到端、skip 拒绝转求解、求解候选不可用、缓存失效、回溯 alpha)。全量 196 passed。

## 3. F1:regen 端点带行验收界分离(已实施,提交 `b97f63e`;D-F 证据启用:band_floor_only×2)

`_collision_rows` 增加 `mode` 参数(accept 角色显式传递,不再凭上界/margin 猜):**唯一改动格** = regen+端点+accept+有限上界普通接触带与主动行,下界 −1.5→−1.8 mm(`cfg_noncontact`)。QP 驱动(−2+动态 m(ctx):0.02/0.1/0.16 m 运动 → −1.5/+1/+4 mm)、显式 margin/probe(−2)、标准模式 accept(−2)、探测态主动行有效界(−1.8,来自上界无穷的 accept 覆盖)、独立 probe 块(−1.8)全部不动;独立终检 −2 mm 不动。7 项测试覆盖全矩阵终装配界 + −1.6 mm 新接受(旧界算术 0.05 归一化违反复现旧拒)、−1.9 mm 仍拒、限位/脚速反例仍拒、终检不变。

## 4. R2/R1:主动行覆盖与走廊接受契约(已实施,提交见 git log)

**R2**:`_registry_nearest_hand`——ref_hold 侧的最近手 geom/箱 pair 从注册表全量查询(1 m 起步、10 m 升级),绕过 ~5 cm 激活门;仅 `dist < distmax` 的真实命中可用(截断哨兵不作距离);无命中显式失败:合成常违反行 `regen_query_failed[side]:regen`(值=查询界,非测量距离)+ `query_failed` 走廊记录,候选被拒且不进回流(无 pair 身份)。候选路径命中时保持 `source="candidate"` 且同 pair 仅一行(去重)。**R1**:R2 行就位后由既有检查链承担(skip 检查、`solve_frame` 可行性/提前返回、`success_review`、pipeline 采纳全部经 `check_rows` 看到主动行与动态上界),不另建距离门——与 D-R 结论一致(R1 未独立触发)。6 项测试:注册表远距选行+停滞悬空诚实违反、查询失败显式拒绝且回流安全、近距去重、标准模式无主动行、**零 QP 反例新形态**(停滞悬空不再 0 QP 原样接受;合法闭合期 0 QP 通过是正确行为)、采纳检查全路径走廊契约(30→24 mm 合法接近通过、6 mm 其余合法时 skip 允许)。全量 209 passed + 1 skipped。

## 5. 六段回归(已执行两轮;首轮被 P1 缺陷污染归档为 v1_superseded,修复后重跑为权威版本——两轮判决逐项一致)

运行:`train24fix_20260927_sixseg/{rq,rv,ns}`(修复后版本 `dbe587a`,单进程串行、线程环境 8、`--assert-task-gate 0.40`、运行期仓库零写入,三次运行 `code_changed_during_run=False`)。首轮(`767a19d`+`9d77135`,P1 截断距离缺陷在场)结果归档于 `train24fix_20260927_sixseg_v1_superseded/`;重跑逐段比较:sub8 的 NPZ 哈希与首轮相同、其余段哈希变化但**状态/计数/召回/质量逐项一致**,且 rq V1 计数(checks/normalized/budget/out)六段完全相同——P1 未在六段集上触发,经重跑证实而非假设。以下数字为修复后权威版本(两轮同值):

### 5.1 regen quiet/verbose(rq/rv)

- **rq 6/6 keep**(旧 Nq 为 5/6):sub7_largebox_000 由 `rejected_final`(终检速度 @t=101)恢复为 accepted,召回 1.0/1.0(旧为拒绝无产出);其余五段全部保持 accepted+pass。基础门 ≥4/6、sub10/sub8 保留、保留段 strict:通过。
- **rq 与 rv 逐段一致**:六段保留 NPZ sha256 逐对比对**全部相同**;状态/召回/attempt 数一致。安静/_verbose 决定性保持。
- 状态链漂移如预期(F1 验收界+R2 主动行改变 regen 链):attempt 数 63→67/73→71/18→17/81→91/119→99/79→76;task p95 与召回整体持平或改善(sub3 l/r 召回 0.989/0.967→1.0/1.0)。
- **V1 计数(实测)**:checks 105-244/段;normalized 仅 sub12 2 次(最大 9.2e-10 rad);budget_failures 0-67(中间候选被拒后由既有升级路径接手);**全部六段 output_frames_modified=0**——最终采纳链全部严格合法,sub7 的恢复来自组合修复改变状态链(旧超限点不复现),V1 严格门保证不再有采纳态超限。
- R2 主动行在悬空参考抓取段真正参与(sub3/sub7/sub12 attempt 数上升与走廊拉入一致),无 `regen_query_failed` 事件。

### 5.2 标准模式(ns)

- 6/6 accepted,与冻结 Ns 报告**逐段全部计数一致**(status/qp_calls_total/iterations_total/attempts/slack_attempts/backtrack_probes/processed_frame_visits/slack_frames)。
- **V1 实测零操作**(非先验声明):checks 97-180/段,normalized/budget_failures/output_frames_modified 全 0——上轮"Ns 应字节不变"的预期以计数+零操作实测落实(冻结 Ns 无轨迹 NPZ,D-Q 已注明仅报告字段可比)。

### 5.3 D-Q 质量回归门:**未过(4 侧)**

按 D-Q 冻结基线逐侧比较(交集帧数/最长连续交集/近接帧数,`[新, 基线, 过?]`):

| 段:侧 | 交集 | 最长交集 | 近接 |
|---|---|---|---|
| sub10:l | 0, 0 ✓ | 0, 0 ✓ | 33, 33 ✓ |
| **sub10:r** | **13, 15 ✗** | **7, 11 ✗** | 67, 63 ✓ |
| sub12:l / r | 0,0 ✓ / 3,3 ✓ | 0,0 ✓ / 3,3 ✓ | 134,133 ✓ / 134,133 ✓ |
| sub16:l / r | 0,0 ✓ / 0,0 ✓ | 0,0 ✓ / 0,0 ✓ | 0,0 ✓ / 13,13 ✓ |
| sub3:l | 5, 3 ✓ | 5, 3 ✓ | 89, 88 ✓ |
| **sub3:r** | **13, 18 ✗** | **5, 11 ✗** | 90, 87 ✓ |
| **sub8:l** | **27, 29 ✗** | 7, 7 ✓ | 77, 77 ✓ |
| **sub8:r** | **37, 41 ✗** | 13, 13 ✓ | 79, 78 ✓ |

近接帧数十侧全部 ≥ 基线;退化全部在**认证∩参考交集**。区间级证据:sub10:r 旧 [41,52]+[57,61] → 新 [35,38]+[41,44]+[45,52](单帧缺口把 11 帧连续段裂成 7);sub3:r 前段抓取区间 [22,26]/[35,38] 不再认证、后段移位为 [62,66]/[71,76]/[77,81];sub8 两处边缘抖动([62,68]→[64,68];r 侧 [64,67] 小区间消失)。**机理**:R2 走廊拉入+F1 验收界改变手部接近动态,验证器滞回/准静态进入认证在区间边缘碎裂或移位——几何近接(keep 判据与标签协议 proximity 类)全面持平或改善,退化限于认证交集。

### 5.4 裁决

按方案 §5("任何侧下降均阻止自动进入 train24,不能用另一侧或新增段抵消";"如发生…质量指标恶化,暂停自动转段并保留差异结论,不以基础数量门掩盖"):**暂停自动转段,不冻结新基线、不进入 train24 重跑**。基础数量门(6/6,含 sub7 恢复)与决定性(rq/rv 字节一致、ns 计数一致+零操作实测)如实记录,不用于掩盖质量门未过。

## 6. train24 重跑(未执行:§5 质量回归门未过,按预注册规则暂停)

方案 §6 的重跑条件未满足;不按六段结果追补策略、不重标阈值、不混合变体。失败分类交付(供下一轮另立方案):认证交集退化 4 侧,机理为走廊拉入下的进入认证边缘碎裂/移位,近接无退化;sub16_012 深穿透、脚速类(sub10_075)、关节限位骑界(sub12_004)为 D-F(订正后)残留类别,本轮未触及。

## 7. 审阅二轮修复(2026-09-27,提交 `66343e1`/`dbe587a`)

| 编号 | 缺陷与复现 | 修复 | 反例测试 |
|---|---|---|---|
| P1 | 回流远距手部 pair 进入 `scored` 后,截断返回(真距 ~100 mm 记 60 mm)被选为 regen 行 best,走廊上界 94.5 mm 下 frame_valid 由拒翻转为过 | 候选路径要求 `dist < activation+0.01` 真实命中,截断落入注册表升级查询 | `_place_box` 构造真距 100 mm + 回流 pair:判定仍拒、corridor dist=真实距离 |
| P2a | 失败快照 `last_res=res` 为第一次尝试对象,与 last_chk/last_rec(最后尝试)不同源 | 新增 `last_exec_res` 三尝试点各自更新,快照带 `last_attempt_id` | 三次失败尝试候选各异:快照候选=第三次状态,身份字段与 last_rec 一致 |
| P2c | `_finalize_attempt` 以 spread 拷贝入 verbose 日志,`adopted` 冻结 None(sub10 日志 67/67 全 None),求解采纳帧决定性检查不可定位 | 改引用入日志,采纳裁决同步 | 求解采纳帧日志中恰一条 `adopted=True` |
| P2b | `_probe_audit` 沿用端点 pair 缓存,中间态进入范围的 pair 无行(两端远、中间穿机器人:离线 pass=True,正式 transition_interpolated) | 每个插值点重置 `_frame_pairs` 重扫;band_floor_only 加前缀终检前置 | 两端远+中点穿体:审计 fail 且行含碰撞对 |
| 追加 | solve 尝试 check 记录在 id 赋值前序列化,按 id 关联落空 | `_note_check` 先于 `_log_check`(纯记录顺序) | dr 决定性检查按采纳尝试 source 匹配回退;r2_decide 修正后为 0 |

修复后全量 **213 passed + 1 skipped**。F1 启用证据、D-R 结论、六段判决均按上文订正块重验;六段权威判决不变(基础门过、D-Q 质量门 4 侧未过、暂停转段)。
