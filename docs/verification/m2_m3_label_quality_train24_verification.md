# M2-M3 标签分级与 train24 受限验证记录

日期:2026-09-26。方案:[m2_m3_label_quality_train24_plan.md](../plans/m2_m3_label_quality_train24_plan.md)。上游:[m2_m3_stage_gate_review_fix_verification.md](m2_m3_stage_gate_review_fix_verification.md)(R1-R9 统一版本,本记录顶部有其 2026-09-26 订正指针)。

状态:**阶段 A(证据订正与质量协议)、阶段 B(分级标签与消费接口)全部落地;六段回归三次运行(regen quiet/verbose、standard)与冻结基线逐段一致(NPZ 字节级相同);受限转段门通过后执行 train24 完整运行——数量门 13/24 < 18/24 未达标,按预注册规则停止规模扩展、不进入 312 与重训。** 全量测试 `uv run --locked pytest tests/ -q` **186 passed**(165 旧 + 21 新,本方案不预设总数、实测记录)。strict/keep/任务工作门 0.40 与全部容差零改动;未重标任务门、未实施 sub7 边界修复、未训练模型。

## 1. 提交清单(可审阅分项)

| 提交 | 内容 | 测试 |
|---|---|---|
| `ee86467` | 阶段 A:CheckResult.evidence 结构化速度证据(未舍入值/上限/超限量/差分帧/关节号,裁决语义逐位不变)、pipeline 两轮贯通、`--assert-task-gate`;`segment_quality_report` 逐帧统一重写(四类认证分列、严格 `<`、同区间越阈、t0 无差分速度、脱离统计);`ireset.data.labels` + `configs/labels/graded_labels_v1.yaml` 协议冻结;review-fix 验证记录订正指针 | `tests/test_label_evidence.py` 12 项 |
| `21da483` | 阶段 B:`export_label_manifest.py`(清单+掩码 NPZ,strict 复检单列,`--out` 独立目录)、`TeacherWindowDataset(label_manifest=…)` 入口(哈希/T/fps/协议/分组/掩码核验,拒绝回退)、`LossAssembler` 分级接触监督(空掩码零损失零梯度) | `tests/test_label_quality.py` 9 项 |
| `06c0390` | 方案文档收录 | — |
| `cd38265` | `train24_label_analysis.py`(§6 分析) | — |

运行产物(全部独立目录,旧实验只读):`derived/data/teachers_regenerated/labelq_20260926_sub7_replay/`、`labelq_20260926_sixseg/{rq,rv,ns,labels_rq}/`、`labelq_20260926_train24/`、`labelq_20260926_nq_labelcheck/`(阶段 A 冒烟,从冻结 Nq 产物复算)。

## 2. 阶段 A:基线订正(方案 §2.1 三项退化,正式证据)

统一版本 Nq/Nv 相对 v3 的质量退化确认为 **3 项**(此前记录为 2 项、"认证交集全达标"错误):

| 指标 | v3 | 统一版本 Nq/Nv | 证据 |
|---|---|---|---|
| sub10 左手认证∩参考交集 | 3 帧 | **0 帧**(认证总数仍 3,区间 [49,52) 在参考窗口 [55,88) 之外) | `labelq_20260926_nq_labelcheck/labels/sub10_largebox_000.npz` meta(复算)+ `six_segment_comparison_reviewfix_20260924.json` |
| sub10 右手几何接近覆盖 | 67/74 | 63/74(−5.4pp) | 同上 |
| sub8 右手几何接近覆盖 | 79/80 | 78/80(−1.25pp) | 同上 |

- 订正指针已写入 [m2_m3_stage_gate_review_fix_verification.md](m2_m3_stage_gate_review_fix_verification.md) 顶部(原数值与历史适用版本保留,不删除原始记录);"2 项偏差不可能由稳定软目标恢复"撤回为"已测试的 0/0.1/1.0 未恢复几何接近覆盖"。
- **sub7 两轮正式重放证据**(方案 §2.1 打印核查的正式化):`labelq_20260926_sub7_replay/`,两轮终检(首轮+后缀重试自帧 96 起)结构化证据完全一致——帧 101、关节索引 26(29 维零基)、值 **12.000000401362618 rad/s**、上限 12.0、超限 **4.0136261780787663e-07 rad/s**、差分帧 [100,101];帧级违反度 ~3.3e-8 低于帧级接受容差 1e-6 但严格终检拒绝,支持"帧级容差与严格终检边界不一致"定位。拒绝保留,边界对齐另列(本轮不修)。

## 3. 阶段 A 工具:统一报告与标签协议

- **报告从逐帧序列生成**(方案 §2.2):认证总数、认证∩参考交集、交集最长连续区间、双手同时认证四类分列,禁止互相替代;几何接近计数统一用 keep 判据的严格 `distance < hand_enter_m`(旧报告 `<=`,与 keep 判据不一致);+2mm/+5mm 计数仅诊断;越阈次数只计同一参考区间内相邻原始帧;新增最长连续脱离/最大脱离/proximity_recall;各侧距离分位数与三 site 箱体系速度保留,**t0 无差分速度记 NaN 不入统计**(旧实现占位 0 会污染含 t0 的区间);零参考帧比例未定义(None)、零认证明确记 0。
- **标签协议 v1 冻结**(`configs/labels/graded_labels_v1.yaml` + 代码常量 `graded_labels_v1`):用途四类(动作/FK、几何接近标记、局部认证监督、拒绝)、公式与比较符(`<` 与调度包含)、已知 3 项偏差、旧质量门(认证交集≥v3、覆盖≥v3)**继续标未通过**——本轮修订的是允许开展何种数据验证及标签用途,不追认旧门通过。
- `six_segment_quality_table.summarize` 同步携带交集区间、脱离、双手与未舍入证据字段。

## 4. 阶段 B:分级标签与消费接口

- **导出**(`export_label_manifest.py`):每个 demo 一行(状态、分组、输入/输出/资产/配置/协议哈希、帧数、fps、失败证据、标签资格、成本);保留轨迹附 `labels/<demo>.npz`(`hand_proximity_mask[T,2]`、`hand_contact_supervision_mask[T,2]`、`hand_supervision_interval_id[T,2]`,布尔掩码、无区间 -1、交集按连续片段编号并保留认证区间边界、半开 `[start,end)`);掩码由独立验证器对存储产物**复算**(同质量表证据链),保留段 strict 复检结果单列;拒绝段保留行并写明确缺失原因;`--out` 使旧运行目录只读。
- **数据集**(`TeacherWindowDataset(label_manifest=…)`):仅读清单明确列出且分组与锁定 split 一致的保留轨迹;哈希(输出 NPZ、标签 NPZ)、T、fps、协议版本、分组声明、掩码-ID 一致性任一不符即拒绝加载;轨迹需要关节限位裁剪(=修改标签)显式拒绝;不回退原教师文件;窗口 60/30 不变,附带三个分级标签键;`batch["contact"]` 条件通道仍来自参考人体(掩码仅损失用);旧入口(无清单)行为不变。
- **损失**:`hand_contact_supervision_mask` 在场时接触损失按有效帧归一化;空掩码零损失、零接触梯度(与 w_contact=0 梯度逐位一致的测试锁定);动作/FK/速度项与 conditioner 不受掩码影响(逐位相等测试锁定);AABB 代理定义与权重未动。
- **冒烟**(方案 §3.3):`test_graded_manifest_to_loss_smoke` 清单→窗口→条件编码→损失前向/反向,无训练步、无检查点更新。
- 关键执行语义已由测试锁定:**sub10 左手认证总数 3、其接触监督为 0**(六段与 train24 标签导出均如此);未认证帧不能通过旧输入条件回流为接触监督。

## 5. 阶段 C-1:六段回归与受限转段门

三次运行(单进程串行、`OMP_NUM_THREADS=8`、`uv run --locked`、未传任务门 YAML、`--assert-task-gate 0.40`、冻结 rev `06c0390`):

| 运行 | 基线 | 结果 |
|---|---|---|
| rq(regen quiet) | Nq | **逐段一致**:accepted/status/t_fail/skip_reason/final_stage/slack/fallbacks、全部成本计数(attempts/qp/iterations/visits/unique)、task_mpjpe、hand 全部相同;5 个保留 NPZ **sha256 逐字节相同**;5/6 keep、sub7 拒绝同因(velocities @101,带结构化证据) |
| rv(regen verbose) | Nv | 同上逐段一致(含 NPZ 字节级相同);quiet 与 verbose 求解行为等价由两者各自与冻结版一致传递保证 |
| ns(standard) | Ns | 6/6 accepted 逐段一致(标准模式不写 NPZ);逐段状态/计数/task/hand 相同 |

耗时单列(不构成承诺):rq 六段合计 ~276s、rv ~273s、ns ~28s,与历史量级一致。

**受限转段门(方案 §4.1 步骤 3)**:数量门 5/6 ≥ 4/6 ✔;sub10/sub8 保留 ✔;5 个保留轨迹 strict 复检全部通过(`labels_rq` 清单 `all_kept_strict_recheck_passed=true`)✔;掩码与报告契约成立(§4 测试+导出复算)✔;sub10 左手接触监督 = 0 ✔。旧认证/覆盖回归门**继续未通过**(协议 yaml 记录,不追认)。→ 启动 train24。

## 6. 阶段 C-2:train24 完整运行与预注册裁决

(TODO 待 clean 重跑完成后填写)

## 7. 与方案的偏差、未执行项

(TODO)

## 8. 后续

(TODO)
