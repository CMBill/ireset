# W1 失败归因与生成链定位执行方案

日期：2026-10-15。版本：v0.1。执行前契约；结果另写验证记录，登记后不改本文件字节。

用户在 W1 第七轮修复全量回归完成（801 passed、1 skipped，提交 `9000ac8`）后授权执行失败归因诊断。本轮落实[父方案 §9](omniretarget_first_training_plan.md#9-何时处理或更换教师)和[W1-D/E §7.3](omniretarget_w1_de_plan.md#73-路线判断的证据要求)。

## 1. 前置与范围

只读前置为 `derived/experiments/omniretarget_first/20261013_be00c50_w1_de/`。其 F 为 motion_baseline=complete、final_interaction_method=not_ready、归因不足；Teacher raw 2/31、模型各 0/31。U0=incomplete、UR=no_quality_gain 保留。

消费 evaluation/checkpoint_selection.json 指定的两份 best/step4000：

| 模型 | checkpoint SHA-256 |
| --- | --- |
| deterministic | `7b1efd57ceff6510d95c5fd682649c7900297030923904c3a7542f59af7b2eff` |
| diffusion | `1f9c16b124ab2d77877e93cfa39af124aa30a4f5aeb998f05350de26dcd437b1` |

本轮仅推理、测量和受控对照；训练步数为零。保持模型、输入、源/目标箱体、缩放、缓存、统计、DDIM25、seed7 和独立验收参数不变。候选融合仅作归因对照，不替换正式生成器、不重选 checkpoint，不回写 W1/F。

## 2. 冻结集合与分母

从冻结 train24 按受试者字符串排序分组，各组取 demo 字典序首个，得到 train6：

- sub10_largebox_063
- sub12_largebox_047
- sub16_largebox_000
- sub3_largebox_001
- sub7_largebox_003
- sub8_largebox_005

选择依据为身份，覆盖六受试者；完整 train24 和选择算法写入登记，不根据诊断质量替换样本。

复用注册 val6：sub10_largebox_100、sub12_largebox_008、sub16_largebox_052、sub3_largebox_018、sub7_largebox_032、sub8_largebox_001。另列既有已公开异常案例 sub16_largebox_048（末窗起点108），注明其为事后已知案例，只用于机制定位，不混入 val6 宏平均。

合计13段；每段 Teacher、生产解码往返对照、两个模型 raw、两个模型融合对照，共78条完整轨迹测量。两模型共26次完整推理；数值重试仅沿用生产侧 seed+attempt 的最多三次规则，质量不触发重试。逐行缺失或测量异常保留，完整分母不变。

## 3. 执行顺序

### R：先登记，再读新诊断质量

新运行目录必须在 derived/ 下且为空，显式拒绝复用。独占写入 freeze/registration.json，包含本方案哈希、实现身份（HEAD、tracked diff、未跟踪实现哈希）、原执行注册/F/原raw报告/选模文件/两checkpoint/manifest/prereg/cache/root_stats 的实测哈希，以及源文件和已冻结资产锚。

准入复用 W1 的输入/资产/缓存消费侧核验、六个 stage_state 及四预算域核验；数值通过标记不替代文件重哈希。原155产物哈希登记为只读前后锚。每个单元开始和收口重核其消费文件。漂移即停止并登记 incomplete。

### A：条件与表示对照

1. 由真实 TeacherWindowDataset 加载当前 split，逐个已有训练/验证窗对账五类条件张量与生成器条件切片（dtype/shape/torch.equal）。未在训练规则中出现的末窗单独记录，不能伪称已训练窗。
2. 将完整教师机器人状态编码为 float32 x0，由固定输出的诊断模型注入原 G1Generator，经过原 `_sample_windows` 解码、原 fuse_windows 与原 clip→world 提升。该 oracle 仅用于检查表示损失，不是可部署方法。
3. 记录往返根位置最大误差、SO(3)角误差、关节最大误差、FK误差及质量差异。参考容差：根/关节 1e-5 m/rad、旋转 1e-3 度；不以放宽容差处理失败。

### B：Teacher/模型配对与拼接前后定位

对13段生成生产 raw，捕获本次实际归一化预测、解码窗口、条件窗口和 seed/retry 元数据；重算预测6D旋转与解码四元数的SO(3)偏差以定位表示解码误差。val7 与旧 raw 产物重放对账；数组误差容差1e-5，另报是否逐位相等。

每个模型记录：

- 整段 teacher-fit（根位置、根旋转、关节、对应FK）及手部FK误差；与人体参考的任务误差分列。
- Teacher与模型逐帧穿透、同帧失败交集/新增失败、手部近接与认证、脚部速度/漂移、首次失败和碰撞对定位。
- 每个解码窗在原世界系、原时间轴的FK误差/关键帧穿透/手间隙；整段按相同全局帧对齐，区分所有有效窗都失败、所有有效窗都通过但融合后失败、混合情形。此窗级统计仅是关键帧几何诊断，不冒称完整独立验收。
- 真实窗口起点、逐帧重叠数、正权重贡献数、当前四元数融合丢弃的权重。训练集和val6分别汇总；已知案例单列。

### C：固定预测窗口的单因素融合对照

代码检查已知假设：末窗非整stride时可能存在三个重叠窗；现 fuse_windows 的位置/关节使用全部权重，而旋转只读取 entries[0]/entries[-1]。当新窗权重为0时，中间正权重窗可能被丢弃，引入边界突跳。

保持同一组实际预测窗口及位置/关节结果，只有旋转改为按窗口顺序累计正权重SLERP（SciPy Slerp，alpha=新权重/累计权重；忽略零权重）。这是确定性的机制对照，不宣称独一的SO(3)平均解。两贡献窗时须与原SLERP在数值容差内一致；同一真值切窗时须保持真值。

报告改动涉及的帧、原/对照 seam 角跳变、全局角速度、独立验收、穿透和任务指标。只在三个及以上贡献的帧产生差异是对照成立条件；位置和关节须逐位不变。改善一个边界不推出全部质量失败已解释。

## 4. 测量与归因边界

整段复用 W1 的 w1_de_frame_series、w1_de_row_metrics、IndependentValidator 和 val6曲线写入/核验口径：穿透最大值来自自适应细分+SLERP，关键帧比例另列；脚速逐sole最坏、首帧占位排除；双手分母来自共享人体参考。固定箱—地面冲突单列，保持原 excluded_from_strict 约定，不移动箱体。

碰撞对事实按 robot-ground、robot-box、self 分类，保存最坏帧/geom名，避免仅凭总穿透推断手部问题。同帧穿透交集只是共现；不同碰撞对不称同族证据。

至少三个不同 train 演示的同族证据是主张标签主瓶颈的必要条件，还须比较模型预测误差与教师缺陷量级并核查条件/实现；本轮不预设任意 teacher-fit 门槛，也不以“超过三例”自动得出主归因。训练损失与完整生成误差量纲不同，不能直接比较。train6/val6均为诊断样本，不能声称总体泛化统计或测试集效果。

若固定窗口对照证明局部融合缺陷，登记明确机制与作用范围；若严重穿透在单窗已存在，单列为预测侧残余；若模型只复现教师同族缺陷且误差较小，可建议教师处理。证据不足继续登记 insufficient。任何后续修复、扩数据、续训或教师处理仍须单独实验。

## 5. 预算、停止与交付

本轮独立总墙钟上限3600s，包含登记、输入重核、所有子进程、推理、检查、汇总和终态登记，预留60s收口。每段子进程超时=min(600s,剩余可用预算)，顺序执行；父进程事前写 running 成本/索引，异常、超时、部分结果如实入账。未登记或 running 残行=未知，不自动重试；新运行不继承旧W1预算，也不得覆盖本轮失败证据。

产物：freeze/registration.json；diagnosis/attempt_NNN 内逐demo结果、78条整段指标、逐帧NPZ、预测窗口及对照、汇总JSON/CSV/图；costs.json与attempts_index。报告包括完整性、预算、原锚点前后是否相等、配对证据、机制对照、剩余解释和下一实验建议。

验证：针对三重重叠丢贡献的合成反例、两窗/零权重/同真值对照、条件篡改、缺行/缺字段/坏值、只读源漂移和预算/异常出口运行必要测试，再正式执行。实现不修改生产生成器与冻结条件代码，受影响的既有生成/原raw/收口测试须通过。

完成后新增 [失败归因验证记录](../verification/omniretarget_w1_attribution_verification.md)，更新索引与 AGENTS；本方案冻结后不填实际结果。动作基线complete、最终交互not_ready和U0/UR结论仍由原F及各自报告解释。
