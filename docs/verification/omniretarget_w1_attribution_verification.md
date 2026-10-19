# W1 失败归因诊断验证记录

日期：2026-10-15。对应[执行方案](../plans/omniretarget_w1_attribution_plan.md)。本轮已执行完毕，诊断证据完整，最终交互方法仍为 `not_ready`。

已确认两类旋转融合机制：三窗重叠时遗漏中间正权重贡献，以及逐帧最短弧 SLERP 在相对旋转跨 π 时切换分支。严重穿透也在融合前预测窗中存在；教师主要瓶颈的主张仍不获支持，学习精度、条件信息与数据覆盖之间的责任尚未分离。本轮没有训练、修改生产融合器或重写原 W1/F。

## 1. 身份与只读前置

前置为 [W1-D/E 正式链](../../derived/experiments/omniretarget_first/20261013_be00c50_w1_de/decisions/attempt_001/w1_de_final.json)，第七轮修复提交 `9000ac849ea2a1b51663bc0952f0c9c90f07f228` 已完成全量 **801 passed、1 skipped**。原 F 的 `motion_baseline=complete`、`final_interaction_method=not_ready` 保留，原 U0 `incomplete`、UR `no_quality_gain` 保留。

本轮运行根为 `derived/experiments/omniretarget_first/20261015_9000ac8_w1_attribution/`。登记时 HEAD 为 `9000ac8`、tracked diff 为空；方案、诊断实现和当时21项测试为未提交新增文件，其内容哈希已登记。实际执行身份应由登记的字节说明，不能把 HEAD 单独称作实现提交。运行后只增添保存窗口的只读分析、验证记录与一项旋转分支合成反例；冻结方案和诊断入口字节未变。

| 锚点 | SHA-256 |
| --- | --- |
| 原 W1 执行注册 | `5561f68562189b7ba45f7193c280739e6dde5325b4df2ada82e8a9cf69ba5f7e` |
| 原 F 终局 | `7f1a9a595cf7964e99a929c83d3bcaac4aa3482a96e515228921b7b67a576737` |
| 本轮冻结方案 | `4bf026c71311f6d169bcc9d8cb2c9383c601d2f0a93d22febf60a74482542be4` |
| 诊断入口 scripts/diagnose_w1_attribution.py | `ca75d3bea24e1c4f2d2772b1615ce643fc5bd9173d0bbaf42632af5706d993e3` |
| 本轮 freeze/registration.json | `499257b0a983b8af1c94a3d819d573d9be5470cb97dcdf4c02a84ac03d4bab52` |
| diagnosis/attempt_001/attribution.json | `72186392cce075ce5893c68f82acaa41fd3efd45b534970696f698f5563c07ac` |
| decisions/attribution_interpretation.json | `0c3ae4ae00fb3b88684b3de73a907c8e31c9563561ab52fc0bcf9f1236678607` |
| decisions/rotation_branch_addendum.json | `6715a06c549377f2b401da425a42cf11d190df3837fe5258663598adb3b7fd69` |
| decisions/diagnostic_closure.json | `93cac087a881e1a05c3f38d3542208e7690aa69afaaa30a626999960136c9064` |

登记冻结 **523 项**文件，含原155产物、checkpoint、输入/派生源箱、资产、缓存、统计和实际执行代码；入口、逐段运行及收口重哈希均一致。原六个 stage_state 无问题，四预算域逐位不变：r0r1 37.536342120030895s、D 8657.971344897s、E 524.651844115986s、detail_f 570.8396364780492s。

两模型使用原选模文件指定的 best/step4000，deterministic SHA `7b1efd57ceff6510d95c5fd682649c7900297030923904c3a7542f59af7b2eff`，diffusion SHA `1f9c16b124ab2d77877e93cfa39af124aa30a4f5aeb998f05350de26dcd437b1`。DDIM25、seed7、原源/目标箱、尺度、条件和验收参数保持冻结。

## 2. 固定集合、执行与完整性

train6 按冻结 train24 的每受试者字典序首段选取：sub10_063、sub12_047、sub16_000、sub3_001、sub7_003、sub8_005（均为 largebox）。val6 复用注册六段：sub10_100、sub12_008、sub16_052、sub3_018、sub7_032、sub8_001。已知异常案例 sub16_largebox_048 单列，不混入 val6 均值；它由既有 W1 报告公开的末窗异常选入，只承担机制定位。

| 轨迹变体 | train6 | val6 | 已知案例 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| Teacher | 6 | 6 | 1 | 13 |
| Teacher 编码→原生产解码/融合/世界提升往返 | 6 | 6 | 1 | 13 |
| Deterministic raw | 6 | 6 | 1 | 13 |
| Diffusion raw | 6 | 6 | 1 | 13 |
| Deterministic 全贡献旋转对照 | 6 | 6 | 1 | 13 |
| Diffusion 全贡献旋转对照 | 6 | 6 | 1 | 13 |
| 合计 | 36 | 36 | 6 | **78** |

正式执行顺序为登记→13个独立子进程→整段测量/汇总→保存窗口复算→旋转分支分析→诊断收口。生产模型完整推理26次；旋转对照复用同一次推理的实际解码窗，没有重新采样或按质量重试。独立验收保持自适应细分+SLERP穿透口径、逐sole脚速和共享参考分母。窗口级穿透只检查原关键帧，不替代整段自适应验收。

主报告 `complete=true/status=done/evidence_gaps=[]`，78条均完成保存与测量。14条 val6+已知案例模型轨迹与原 W1 raw 的 root/joints **逐位相等**。后续从保存窗口独立重建生产和对照轨迹26组，全部 root/joints 最大绝对差为0；对照位置与关节逐位不变。

条件一致性26/26通过：真实 TeacherWindowDataset 与生成器的五类条件，在52个独立已有窗口×两模型的104次对账中逐键 dtype/shape/torch.equal 相等。推理新增的非整stride末窗明确登记 `inference_only_starts`，不称作训练时已出现的窗口。

## 3. 分组质量与预测残差

教师拟合为各演示对应 FK 误差的宏平均；穿透列为组内最坏整段自适应最大值。train6 与 val6 为固定诊断样本，不外推成总体统计，也不与原 val31 宏混用。

| 集合 | 方法 | 教师拟合 FK 均值 m | raw通过 | 最坏穿透 mm |
| --- | --- | ---: | ---: | ---: |
| train6 | Teacher | 0 | 1/6 | 35.245 |
| train6 | Deterministic | 0.057790 | 0/6 | 153.019 |
| train6 | Diffusion | 0.059154 | 0/6 | 134.637 |
| val6 | Teacher | 0 | 1/6 | 42.958 |
| val6 | Deterministic | 0.115399 | 0/6 | 216.613 |
| val6 | Diffusion | 0.099867 | 0/6 | 194.885 |

保留全部旋转贡献的对照仍是两模型 train6/val6 **各0/6**。train6 Diffusion 的组内最坏穿透反而升至138.114mm；已知案例 Deterministic 穿透190.197→209.055mm，Diffusion 182.185→178.473mm。已知案例两模型教师拟合也分别从0.166208→0.171468m、0.151537→0.157836m，均恶化。因此，本轮对照没有获得生产替换资格。

按同一全局时间轴比较各有效预测窗与融合轨迹，结果如下。表内计数是**关键帧**超过2mm的几何诊断；“所有窗都坏”指该帧每个正权重贡献窗都穿透，“仅融合后坏”指所有贡献窗均通过而融合结果失败，其余为混合情形。

| 集合 | 模型 | 融合坏帧 | 其中所有窗都坏 | 仅融合后坏 | 首次重叠前已有穿透的段 |
| --- | --- | ---: | ---: | ---: | ---: |
| train6 | Deterministic | 975 | 946 | 0 | 6/6 |
| train6 | Diffusion | 859 | 815 | 0 | 6/6 |
| val6 | Deterministic | 786 | 734 | 1 | 6/6 |
| val6 | Diffusion | 721 | 638 | 0 | 5/6 |

26条模型轨迹均存在已穿透的解码窗。即使在训练演示，完整生成仍有约5.8–5.9cm的对应FK误差。这支持显著的预测阶段残差，不能把全部坏结果归于拼接或教师既有缺陷。保存的逐帧碰撞族（robot-ground/robot-box/self）、最坏geom对和新增/共现帧是描述性证据；同帧或同族共现不等于相同因果，固定箱地冲突仍单列而不挪箱体。

## 4. 两类已确认的旋转融合机制

### 4.1 三窗重叠时遗漏有效贡献

生产 [fuse_windows](../../src/ireset/model/stitch.py) 的位置/关节累加全部窗权重，但旋转只读取 `entries[0]` 与 `entries[-1]`。末窗不整stride时会出现三个重叠窗，中间窗正权重被丢弃；新尾窗即使权重为0，也会替换旋转的第二个槽位。13段中11段出现三窗覆盖。

已知案例起点为 `[0,30,60,90,108]`，零基frame108的三个窗分别为start60权重0.3793103448、start90权重0.6206896552、start108权重0。原融合遗漏start90的有效贡献，实际使用start60和零权重尾窗。

同一预测窗，仅改为按窗顺序累计正权重SLERP后，frame108单步根旋转差如下：

| 模型 | 原融合 | 全贡献对照 |
| --- | ---: | ---: |
| Deterministic | 29.715058° | 1.470818° |
| Diffusion | 99.651625° | 1.681849° |

改动仅发生在三窗覆盖帧，两贡献窗部分在注册容差内一致；根平移与关节逐位不变。合成反例、相同真值切窗和零权重对照都通过。这足以定位局部贡献遗漏，不能推出整段已修复。

### 4.2 两窗最短弧跨分支造成时间不连续

已知案例 Diffusion 的全局最坏跳变在frame107，为 **158.324227°**，此时只有start60/start90两个窗，保留全部贡献的对照仍保留该跳变。全部解码窗的最大内部单步为15.041364°；在frame106→107，两个实际输入各自仅变化4.137532°/8.979722°，各自相邻四元数内积0.999348/0.996931，表示连续。

两个窗之间的四元数内积却从−0.0714138001变为+0.0391571894，经过相对π附近。逐帧独立选择的最短弧SLERP因此切换分支，产生远大于各输入局部变化的融合跳变。此处是缺少跨时间路径一致性/拒收契约，不能称作SciPy库计算错误；四元数自身的符号连续也不足以解决相对π的最短弧分支。

Deterministic在frame114也观察到同类跨分支：实际所选start60/start108内积−0.014756→+0.067161，融合单步176.543893°，各输入该步2.473019°/9.116881°；另有三窗遗漏与模型窗自身不连续共同存在。Deterministic窗内全局最坏仍达126.518316°，全贡献对照的整段最坏112.683872°；不能把它简化为单一融合原因。

新增合成反例用两个各自单步<1°的连续旋转窗跨相对π，真实生产融合出现>150°跳变，全贡献对照也相同。该反例区分了本节机制与三窗遗漏。

## 5. 表示往返的负向精度门

Teacher经过真实float32编码、生产解码/融合及世界提升后，根位置最大误差1.20e-7m、关节最大误差5.06e-7rad。旋转注册容差为0.001°，以下三段**未通过**，没有放宽门槛或把它们改记通过：

| demo | 最大旋转误差 ° | 最大逐帧FK均值误差 mm | 最大手部FK误差 mm |
| --- | ---: | ---: | ---: |
| sub16_largebox_000 | 0.002036221 | 0.009171 | 0.019333 |
| sub7_largebox_003 | 0.001259727 | 0.007310 | 0.006600 |
| sub16_largebox_052 | 0.024260369 | 0.114090 | 0.123863 |

其余10/13通过全部注册往返容差。此处“诊断证据complete”指每个固定单元有产物和测量，并不表示往返精度门全部通过。最坏手部误差约0.124mm，远小于模型厘米级FK残差；它是需要保留的精度缺口，不能据此归因百毫米级穿透。各模型归一化6D旋转与实际解码四元数的偏差另存 `decoded_rotation_error_deg`，不与模型相对教师的预测误差混用。

## 6. 预算、复算与验证

独立实验墙钟预算3600s、收口预留60s。核心诊断报告记录394.665884s，诊断台账含写盘尾部记396.724929s；登记、保存窗分析、分支检查和诊断收口均另入本轮 costs.json。**总计416.249327s，约6.94分钟**，没有占用原W1的四个预算域；原155产物和源台账均未改写。

已执行的入口命令如下（输出已存在，入口拒绝重复登记/运行，不应直接覆盖重跑）：

```bash
uv run --locked python scripts/diagnose_w1_attribution.py register --run-root derived/experiments/omniretarget_first/20261015_9000ac8_w1_attribution
uv run --locked python scripts/diagnose_w1_attribution.py run --run-root derived/experiments/omniretarget_first/20261015_9000ac8_w1_attribution
```

仅对已有窗口执行的 `decisions/analyze_saved_windows.py`、`inspect_rotation_branches.py` 和 `close_diagnosis.py` 作为本运行的分析工件保留，输出绑定各自脚本哈希，没有新的模型调用或第二候选。收口重新核对523源锚、78产品、26窗口重建、14生产重放、六stage_state和四预算域。

新增 [tests/test_w1_attribution.py](../../tests/test_w1_attribution.py) **22项**，覆盖三窗遗漏/双窗一致/真值保持、π分支、条件篡改、源漂移、输出路径、缺测量/非有限/缺行/重复行、真实验收和往返接口、预算耗尽与未知历史。执行：

```bash
uv run --locked pytest -q tests/test_w1_attribution.py tests/test_generation.py tests/test_w1_de_raw.py tests/test_w1_de_numerics.py tests/test_w1_de_closure.py
```

结果 **159 passed（31.83s）**；一个既有的未注册slow标记警告。用户已完成的全量801 passed+1 skipped为本轮只读基线，本轮未再跑全项目套件，也不将新增项数相加冒充新的全量结果。未改生产源码、配置、依赖或冻结条件代码，缓存没有重键。新增方案/验证记录、docs索引与AGENTS的91个本地路径/标题链接检查通过，tracked diff及新增文件空白检查通过。

## 7. 产物、结论与后续

| 产物 | 用途 |
| --- | --- |
| [主诊断报告](../../derived/experiments/omniretarget_first/20261015_9000ac8_w1_attribution/diagnosis/attempt_001/attribution.json) | 13段/78轨迹、完整指标/条件/重放/窗口事实和分组汇总 |
| [配对指标CSV](../../derived/experiments/omniretarget_first/20261015_9000ac8_w1_attribution/diagnosis/attempt_001/paired_metrics.csv) | 78行完整比较，不丢精度失败或质量坏行 |
| [配对诊断图](../../derived/experiments/omniretarget_first/20261015_9000ac8_w1_attribution/diagnosis/attempt_001/paired_diagnosis.png) | 穿透、教师拟合与已知案例边界旋转对照 |
| [保存窗口独立复算](../../derived/experiments/omniretarget_first/20261015_9000ac8_w1_attribution/decisions/attribution_interpretation.json) | 26组重建、逐组关键帧归因和往返精度 |
| [分支机制补证](../../derived/experiments/omniretarget_first/20261015_9000ac8_w1_attribution/decisions/rotation_branch_addendum.json) | 相邻窗局部连续、相对内积跨零与最坏跳变 |
| [诊断收口](../../derived/experiments/omniretarget_first/20261015_9000ac8_w1_attribution/decisions/diagnostic_closure.json) | 汇总证据哈希、只读锚/预算复验和归因边界 |

标签确有缺陷，但三种方法的质量差距、融合前失败和训练演示预测残差不支持“标签是主要瓶颈”的结论；6个train配对案例满足父方案的例数下限只是必要条件，不是因果充分证据。条件张量一致也只证明生产/消费无此类错位，不能证明其信息充分或数据覆盖充分。

下一项建议是独立规定并验证**完整贡献与时间路径一致的旋转融合/拒收契约**，同时保留模型自身跳变与碰撞的拒绝；先用固定checkpoint和本轮保存窗做有限对照，再按冻结集合重测。局部融合问题修正后，仍须用受控实验分离学习精度、条件信息与覆盖范围。当前全贡献SLERP仅为诊断，不直接采用；不自动追加训练、扩数据或重做教师。原W1/F的完整动作基线及最终交互 `not_ready` 均保留。
