# M2-M3 收尾阶段 0-3 阶段门未达标原因分析(GLM)

日期:2026-09-17。性质:**分析文档**——仅复核既有 v4 产物与逐段 `frame_log`,未运行新实验、未修改代码与容差。分析对象:[m2_m3_completion_plan.md](../plans/m2_m3_completion_plan.md) 阶段 0-3 的实施结果(验证记录:[m2_m3_completion_verification.md](m2_m3_completion_verification.md)):阶段门要求 ≥4/6 保留,实测 **2/6**,按方案 §7 门禁停止扩展、不进入 24/312 段规模化。

证据基线:`derived/data/teachers_regenerated/pilot_diagnosis6_v4_stage12/`(git `1a8e330`,strict,全长度,`--verbose`),逐段 `frame_log_*.json` 与 `regen_report_diagnosis6_strict.json`;代码引用以该提交为准。

## 1. 结论

阶段 0-3 的修复本身有效且无回归(失败帧后移、失败类别收敛、标准模式 4/6 零回归),但阶段门要求把 4 个失败段中至少 2 个**整段**翻绿,而 v4 剩余四个失败没有一个是阶段 1/2 所针对的"约束集合错误/误拒"类别——它们全部属于方案 §7 明确划给阶段 4 的条件项范畴,其中一段(sub12)是输入不可行、修复器无法拯救。2/6 是这个结构性现实的诚实结果,不是阶段 0-3 的实现失败;方案 §1 预先写明"不得预设修复后一定达到某个接受率"。

## 2. 逐段直接原因(frame_log 实证)

| 段 | v3 t_fail | v4 t_fail | v4 直接死因 | 方案 §7 归类 |
|---|---|---|---|---|
| sub10_largebox_000 | keep | keep | —(保留;质量退化见 §4) | 行 5 |
| sub8_largebox_000 | keep | keep | —(保留,task p95 0.0723 不变) | — |
| sub12_largebox_001 | t=11 | t=2 | **输入不可行**:参考左手单帧位移 ≈0.12m(30fps 下 ≈3.6 m/s),弦线中态穿箱 −26.8..−27.2mm;多次回溯后三起点全部不可行 | 行 4(预注册筛查) |
| sub16_largebox_000 | t=22 | t=22 | **中态+端点双违反**:`right_knee_link~right_rubber_hand` 回流进线性化后,端点仍 −5.3/−6.4mm(门 −1.8mm),中态探测 −4.0..−10.0mm;失败发生在抓取区间外(t=22 走廊行为空) | 行 1(探测点约束) |
| sub3_largebox_000 | t=1 | t=11 | **右手 regen 行贴地板**:上一接受帧 −1.00mm,地板 −1.5mm,内向余量仅 0.5mm;候选 −2.62mm,三起点(prediction/prev_accepted/slack_fallback)15/17/16 次迭代全部 infeasible | 行 3(贴边消融) |
| sub7_largebox_000 | t=24 | t=34 | **右手 regen 行超天花板**:上一帧 5.22mm,候选 8.78mm > 上界 8.0mm(超 0.78mm);三起点 20/25/24 次迭代全部 infeasible | 行 3(t=0 走廊下界同族) |

sub3 的原 t=1 脚部死因、sub7 的 t=24 漏激活死因均已被阶段 1/2 消解(sub3 前 10 帧全过、sub7 拉入完成并保持 10 帧)——失败帧后移本身是修复有效的直接证据。

## 3. 结构性原因:有效修复为何未转化为计数

### 3.1 门算术与串联瓶颈

整段接受 = 所有帧全过。sub3/sub7 各有多重串联瓶颈,阶段 1/2 只消除了各自诊断出的那一类(冻结 pair 集合、中态走廊误拒),消掉后暴露的是**下一个不同机理**的瓶颈。"失败帧后移 + `transition_interpolated` 归零"与"整段计数停在 2/6"并不矛盾——这恰是修复有效但瓶颈串联的证据。此外,diagnosis6 的四个失败段正是先导 2/6 中的全部失败段,是按难度筛出的集合;"两类机制修复能翻绿其中 ≥2 段"是对最难样本的乐观假设。

### 3.2 失败类别与修复类别错配

四个失败段中:sub12 是参考输入单帧跳变(任何端点合法的解,弦线中态都穿箱);sub16 需要把探测点作为约束写进 QP(方案 §4"限制"当时已预言"刷新端点 pair 不能保证弦线安全,sub16 类中间态问题可能需要阶段 4 的探测点约束");sub3/sub7 是 regen 行边界可行性。这些在方案 §7 中均为"条件性工作,不默认全部实现",阶段 0-3 按设计未触碰。

### 3.3 regen 走廊的结构性贴边

- 走廊上界 `max(8mm, prev − 0.15m/s·dt − margin)`(`src/ireset/repair/scp.py:455-457`):手一旦进入 8mm 内,上界固定钳制在 `regen_hand_contact_gap_m = 0.008`(`src/ireset/repair/context.py:105`,取 8mm 是为小于 hand_enter 10mm 以通过认证),**不再随 prev 收紧**;0.15 m/s 闭合率在 30fps 下即 5mm/帧。
- SCP 的解按构造骑坐在约束边界上(regen 模式统一加 0.5mm 内部 band 余量即为此设计,`scp.py:465-467`),手部参考跟踪目标(weight 25,`context.py:124`)与硬走廊行、自碰撞、速度、锚点行在难帧上直接冲突;连 slack 回退都失败,即该线性化下真不可行。
- sub3 仅差 1.12mm、sub7 仅差 0.78mm——毫米级贴边,不是大幅违反,但 strict 门下一样拒绝。
- 注意:"三起点不可行"是求解器在该帧线性化与信赖域内的结论,不构成轨迹级不可行的证明(更大邻域内可能存在可行解),这也是 §7 行 3 要求单独消融而非直接判死的原因。

### 3.4 对照证据排除回归

同批段标准模式 4/6(sub10/sub16/sub3/sub8 过,sub12/sub7 拒),与 slack 修复后基线(`regen_impact_6seg_strict.json`,git `a332001`)一致:共享修复逻辑零回归。regen 模式相对标准模式多拒的 sub16/sub3 增量来自路径 B 专属行与目标(主动手部接触的可行性代价),而非阶段 1/2 引入的回归。sub16 的 t=22 走廊行为空,失败在接近段的大幅手部摆动弦线上;标准模式同段通过,说明该摆动与 regen 专属目标的解路径有关(未单独归因到具体目标项)。

## 4. 阶段 2 的权衡代价(保留段质量)

去掉中态走廊上界消除了误拒,但该上界此前隐性充当"缓入阻尼":sub10 拉入变快、箱体系速度抖动增大,0.05 m/s 准静态进入门更难触发——接触 R 0.905→0.838(74 帧中 5 帧失去物理接触)、认证区间 3+2/74+33 帧 → 0。sub8 接触 R 0.988→0.975、认证不变,两段 task p95 均不变。误拒消除与认证覆盖是同一改动的两面,即 §7 行 5"稳态化目标"存在的原因;即使计数达标,v4 保留段可用性也弱于 v3,24 段标定前必须恢复或给出可用性结论。

## 5. 对门本身的判断与后续

4/6 门隐含的假设是"修好漏激活+走廊语义能翻绿 ≥2 段",v4 证伪了该假设——四段的剩余瓶颈都不属于这两类。按 §6/§7 的预设计,2/6 触发的正是"继续阶段 4、不进入规模化"的止损分支。sub12 若经原始站点/几何证据确认走预注册筛查(排除仍进原始分母),该六段在筛查后口径下的理论上限为 5/6,4/6 门依然可达。可操作项与验证记录 §5 一致:

1. sub12:输入站点速度预注册筛查(§7 行 4;不得用修复候选的违规替代输入证据);
2. sub16:探测点 pair + 插值比例反馈进 QP,雅可比 FD 对拍(§7 行 1;仅端点加 pair 不足,已被 v4 实证);
3. sub3/sub7:regen 行贴边消融——走廊进入 8mm 后上界是否应随 prev 收紧(非 `max` 钳制)、软拉入,与 §5 第 4 项 t=0 走廊下界同族(§7 行 3);
4. sub10:认证覆盖稳态化(§7 行 5)。

## 6. 证据与复现

- 逐段失败详情:`derived/data/teachers_regenerated/pilot_diagnosis6_v4_stage12/frame_log_{sub3,sub7,sub12,sub16}_*.json` 中 `t` 为 t_fail 的 `kind=check/probe` 条目(含 rows 的 lo/hi、hand_*_prev/cand、corridor、motion、carried)与 `kind=attempt` 条目(三起点 status/iterations/slack)。
- 汇总口径:`regen_report_diagnosis6_strict.json`(`n_kept=2`、逐段 t_fail 与 `ms_per_processed_frame`:sub10 543.8、sub8 82.6、sub16 88.2、sub3 176.0、sub7 330.3、sub12 534.3 ms/处理帧;sub12 的 534ms/处理帧为 3 帧内三起点耗尽,早期拒绝不再被整段长度摊薄——阶段 0 统计修复的直接收益)。
- 示例提取(sub3 t=11 右手 regen 行):

```bash
python3 -c "
import json
d=json.load(open('derived/data/teachers_regenerated/pilot_diagnosis6_v4_stage12/frame_log_sub3_largebox_000.json'))
print([e for e in d if e.get('t')==11 and e.get('kind') in ('check','attempt')])"
```

- 弦线中态违反基于离散探测采样(与终检同口径),不宣称连续时间零碰撞;本文档的全部数值来自上述单次 v4 运行。
