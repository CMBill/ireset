# train24 失败类别最小修复验证记录

日期:2026-09-27 起。对象:[train24 失败类别最小修复方案](../plans/train24_failure_fix_plan.md)(修订版)。本记录随各阶段实施递增;未完成的节标注状态。取证产物目录:`derived/data/teachers_regenerated/train24fix_20260927_diag/`(完整结论 JSON:`D_CONCLUSIONS.json`)。

## 1. D 阶段:只读取证(已完成)

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

| 段 | t | 分类 | 依据 |
|---|---|---|---|
| sub3_001 | 111 | **band_floor_only** | 仅带行 µm 级越界(7.35e-6/3.56e-6 m);反事实通过端点+探测+速度+任务门;前缀终检通过 |
| sub8_002 | 62 | **band_floor_only** | 仅 ankle 行 2.5e-7 m 越界;反事实全过 |
| sub12_004 | 0 | mixed_joint_limit | `joint_limit[j16]` 骑界 2.76e-7 rad + 四条带行 µm 级;非指定格的行仍挡(反事实不过) |
| sub16_012 | 40 | deep_penetration | 手带行 −3.5~−4.0 mm,超终检门 |
| sub7_029 | 18 | foot_speed | 支撑脚速度 post 行 2e-8~1.1e-7 m/frame 越界与带行 3.1e-6 m 并存;反事实仍被脚速行挡 |
| sub10_075 | 69 | foot_speed | sole2:z 越界 9.4e-4 m/frame(实质性,非 eps 类) |

**F1 启用条件成立**(2 个真实 band_floor_only 反例)。残留类别(脚速 2 段、关节限位骑界 1 段、深穿透 1 段)本轮不设专属策略(方案 §8),如实入账。帧级反事实通过≠整段恢复,以六段回归与 train24 重跑裁决。注:独立 evaluator 复算的部分末候选行值与冻结报告在 µm 级有差(进程内求解器状态),分类不受影响。

### 1.3 D-R:走廊/全入口审计与零 QP 反例

- **R2 触发**:ref_hold 侧手距 5.7~11.8 cm(超出 ~5 cm 候选准入)时主动行不存在——sub10_001 决定性检查 107 帧、sub10_003 91 帧(任一入口 107/43)。
- **零 QP 反例(正式保存)**:sub10_001 帧 30/31/32,ref_hold 含双侧手、手距 9.4~12.6 cm,新求解器返回 accepted=True、**qp_calls=0**、状态逐位不变(可行+任务过+`_stationary` 提前返回)——"调用 solve_frame"确实不等于走廊参与。
- **R1 未独立触发**:无任何已采纳帧违反已激活走廊的上界。R2 行就位后,既有检查链(skip 检查、求解可行性、success_review、pipeline 采纳)即构成走廊契约;R1 落为契约测试(方案 §4.3"满足则复用")。

### 1.4 D-Q:质量基线冻结

18 段(六段 rq 5 NPZ + train24 13 keep)按标签协议 v1 复算并入 `dq_quality_baseline.json`(含输入哈希)。方案 §5 已知下限逐项复现:sub10 交集 0/15、近接 33/63(参考 33/74);sub8 交集 29/41、近接 77/78(参考 80/80)。

### 1.5 启用结论

V1 ✅、F1 ✅(band_floor_only×2)、R2 ✅;R1 为契约验证(经 R2 行,不另建距离门)。

## 2. V1:有界候选规范化(状态:待实施)

## 3. F1:regen 端点带行验收界分离(状态:待实施)

## 4. R2/R1:主动行覆盖与走廊接受契约(状态:待实施)

## 5. 六段回归与重新冻结(状态:待实施)

## 6. train24 重跑与预注册裁决(状态:待实施)
