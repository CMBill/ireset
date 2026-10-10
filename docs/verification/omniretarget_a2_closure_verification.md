# A2 收口与有限恢复验证记录（C0/C1/D0/G0/UR/C2）

日期：2026-10-10。方案：[A2 收口与有限恢复实现方案](../plans/omniretarget_a2_closure_plan.md)（提交 `e487eb3`）。
实现提交：C0/C1+D0 `a39d763`、NaN 对照修复 `420afbf`、UR `360556d`、对照修复 `880bff8`、
**审阅 13 项缺口修复 `efece4a`（§11）**。
运行根：`derived/experiments/omniretarget_first/20261010_a39d763_a2_closure/`。

## 0. 结论摘要

| 阶段 | 结果 | 关键证据 |
| --- | --- | --- |
| C0 基线复验 | PASS | 方案 §1.1 五锚点逐位一致；新鲜比较与权威报告逐位一致（排除 timestamp/comparison_attempt/cost_s，差异 []）；known 缺口恰 sub12 三行，无新增 |
| C1 U0 停线登记 | registration complete（exit 0） | `baseline_verdict=incomplete`、`training=not_ready`、`stop_u0_expansion` 三状态分离落盘 |
| D0 sub12 诊断 | `supports_step03_candidate` | 控制对照逐位相等；两次零基失败帧 171、帧内第 5 次求解、关键数组逐位一致；Phase I rho=0.2486138173835237 两次逐位相同 |
| G0 预注册 | 完成（读任何恢复标签质量之前） | `recovery/preregistration.json` sha256 `f4676813b20c2784…`，唯一差异 step_size 0.2→0.3 |
| UR 六段重生成 | 6/6 COMPLETE（n_ok=6，含 sub12） | 总墙钟 915.0s；单段最长 sub12 306.95s ≤30min |
| UR 独立比较 | **`no_quality_gain`（complete_evidence=true）** | g1/g2 PASS，g3/g4/g5 FAIL；注册/配置对账腿全过 |
| C2 收口 | registration complete | 原 U0 裁决 incomplete 与 UR 新实验 no_quality_gain 并存，训练 not_ready |

**最终状态：原 A2 的 U0 保持历史 `incomplete`（sub12 缺产物，不再扩量）；新实验 UR（唯一差异
step_size 0.3）六段证据完整但质量门不满足，裁决 `no_quality_gain`。** 按父方案 §3.4，该裁决
使 W1 弱监督诊断路径获得登记资格，但不自动启动；train24/A3 生产仍未就绪。后续进入 W1 的
A3/B/C 执行方案另立预注册文档。

**独立审阅 13 项缺口修复轮（2026-10-10，提交 `efece4a`）后按修正协议重跑**：裁决与度量逐位
不变；权威证据更替为 `baseline/snapshots/attempt_001`（C0，sha `b1fb3e6a…`）、
`diagnosis/attempt_007`（D0 复跑，prefix/canonical 补齐，rho 三组独立逐位一致）、
`recovery/cmp/attempt_002`（UR 强腿复跑）、`decisions/attempt_006`（C2 终局，证据链核验全过，
sha `5ca7960c…`）；测试 484→**506 passed + 1 skipped**。详见 §11。

## 1. 基线锚点核对（§1.1）

五个方案锚点在执行开始时重哈希，全部与方案表逐位一致：

| 锚点 | SHA-256（实测=方案） |
| --- | --- |
| 权威 teacher_selection.json（reviewfix7） | `bc03d085535c462c70864fabf265d399b3a18838d36587138d3f2c67d2f3c683` |
| A1 frozen_lists.json | `395094a30d69158ad08bf5b576f07c541c052187d5bf242f1cc7fde05e353d9d` |
| A1 assets.json | `2f72c9ed0f1832b0c1d80e7d1f2e82382785c56ae83556a63a46ec6789c7ad79` |
| A1 pairing_source_boxes.npz | `39fce035ab63569bd69a4ddf61e1e6cd4fb70d4d1b2a12abf7051bd55ce3d09d` |
| 上游 uv.lock | `0fe6c3679e32925371a95b87a1953c72baeaab7571e91a19ef2f53887c4073b0` |

## 2. C0：基线复验（baseline/）

- **新鲜比较**：经既有比较入口（`--u0-dir` 模式）生成 `baseline/cmp/attempt_004/teacher_selection.json`（前三轮 attempt 为 NaN 假阳性拦截与 UR-only 键修复前的过程证据，保留）。与权威报告逐位对照（仅排除 `timestamp`/`comparison_attempt`/`cost_s`）：**差异 []**。度量、g1–g5、evidence_gaps、admission（输入三方对账/资产 120 项/环境三元/产物链/绑定/数据路径）全部一致。
- **known/unexpected 缺口分列**：known 恰 `sub12:{U0_invalid,binding_not_verified}` + `admission:sub12:binding_recheck_failed`；unexpected=[]。
- **消费映射**：五段 attempt_001、sub12 missing；历史失败 attempt_001/002 均保留。
- **身份**：运行期间 repo 身份稳定（start==end）；上游 checkout bccd4d7 干净过门。
- 落盘：`baseline/baseline_evidence.json`（sha `6a6ce157f5ad1489…` 最新轮）、`baseline/evidence_files.json`（sub12 教师显式 `missing:true`，含消费 PT/发布 NPZ/raw+converted/两份失败 manifest/A1 冻结闭包/行为代码全哈希）。

**过程缺陷（已修复并钉死）**：首轮 C0 被假拦截——无效行 sentinel（缺产物 demo 的 `fps=NaN`）在两份报告中同值，`NaN != NaN` 把 `metrics.U0.sub12.fps` 误报为差异。修复为双 float 值相等或双方 NaN 判同（提交 `420afbf`），反例（双 NaN 匹配 / NaN vs 30.0 不匹配）入 `tests/test_omniretarget_a2_closure.py`。

## 3. C1：停线登记（decisions/）

`decisions/attempt_002/a2_closure.json`（sha `43eb62a5bfa62d1d…`）：`baseline_verdict=incomplete`、
`disposition=stop_u0_expansion`（质量依据 g3/g5 五段全败、g4 1/3；成本依据 sub12 两次默认运行
143.95/144.65s 确定性 infeasible）、`training=not_ready`（三条未满足门逐条列出）、
`registration_status.complete=true`。exit 0 语义为登记完成，报告接口显式声明其不是来源比较通过
（§10.1 反例：分母缩水/行删除/裁决-证据矛盾/锚点篡改均拒绝正常登记，17 项测试钉死）。

## 4. D0：sub12 有限诊断（diagnosis/）

最终轮 `attempt_003/004`（`attempt_004/diagnosis_run.json` sha `7bcd253a75263dfd…`；首两轮
attempt_001/002 因 §5.4 配置快照缺陷作废保留为证据）：

- **控制对照（§5.3 步 1）**：sub16 带包装默认生成 61.9s/89 帧/623 次求解，qpos/human/fps 与
  已登记 raw（哈希先核验）**逐位相等**——包装等价成立，sub12 归因被允许。
- **两次 sub12（零基、原默认初始化）**：均失败于 `RuntimeError: CVXPY solve failed: infeasible`；
  **实际失败帧 171**（包装捕获的 `frame_idx`，与历史进度条 171/347 一致）、**帧内第 5 次求解**、
  init_t=False、step_size=0.2、脚保持双侧 True、原约束 66 条；两次墙钟 161.1/158.6s；关键数组
  （q_locked/q_a_n_last/q_t_last/target_laplacian/obj_pts_local）**逐位一致**。
  【订正（审阅缺口 8，2026-10-10 复核轮）：本段原文"前缀 171 帧存 `prefix_debug.npz`"不实——
  原 attempts 003/004 的两份快照目录**均无** prefix_debug.npz；`_capture_failure` 当时沿异常
  traceback 找 `retarget_motion` 帧，而该帧是异常的**调用者**、尚未退栈，不在 traceback 里。
  修复后经调用栈（f_back 链）捕获并于 attempt_006/007 复跑验证：**prefix 171 帧实际落盘**
  （`qpos [171,43]`、fps 30、diagnostic_only），failing_frame_idx=171 与原判逐位一致。】
- **Phase I（唯一检查）**：66 条原约束恰含 **1 条**显式 SOC（t=0.2 与实例 step_size 核对一致）；
  保留 65 条约束最小化 `||dqa||₂`：status=OPTIMAL、**rho=0.2486138173835237**（两次逐位相同）、
  max_violation=5.55e-17 ≤1e-6、dqa 36 维（上游自身变量尺度，不换算为项目 35 维）。
  分类：`0.2+1e-6 < rho ≤ 0.3-1e-6` 且两次一致 → **supports_step03_candidate**。
- 原问题 canonical 数据（P/A/q/b 稀疏矩阵 + 变量表）与完整 stdout/stderr 入册。
  【订正（审阅缺口 5，2026-10-10 复核轮）：该句中"q"不实——Clarabel 锥数据的目标向量键是
  `c`，首轮序列化器读 `q` 使目标向量缺失，且 `n_var/n_eq/n_ineq` 键不存在时维度记为
  −1/−2（原证据实测 `arrays:["b"]`、`n_variables:-1`，保留为缺陷证据）。修复后 attempt_006/007
  快照 `arrays:["b","c"]`、798 变量/949 行。】
- 诊断墙钟 387.7s ≤ 60min 预算。

**过程缺陷（已修复）**：首两轮子驱动用 tyro 旧 kwarg `argv=` 转储"生效配置"——tyro 1.x 已改名
`args=`，旧 kwarg 落入 `**deprecated_kwargs` 被**静默忽略**，转储的是默认配置（诊断结论不受影响：
失败帧/数组/Phase I 均来自实际运行路径，控制对照逐位相等是更强证据）。修复后重跑（attempts
003/004）配置快照为真实生效配置（smplh/sub12/step 0.2/largebox）。

## 5. G0 + UR：预注册与独立恢复实验（recovery/）

- **预注册（§6.2）**：`preregistration.json` sha256 `f4676813b20c2784…`，在读取任何恢复标签
  质量之前以独占创建写入；唯一差异 `retargeter.step_size: 0.2→0.3`；父证据=权威报告
  （bc03d085…）+ 最终诊断（7bcd253a…）；期望分母=冻结六段。
  *登记过程说明：第一次注册（sha `ac1d7a39…`）引用了带 §4 配置快照缺陷的首轮诊断文件；因当时
  UR 零产物、零消费，删除后以修复轮诊断重新注册——反追认规则针对"产物先于注册"，本次未违反，
  全程留此说明。*
- **生成（§7.1-§7.2）**：`generation/attempt_001`（manifest sha `2facb16b6ec6c3be…`）六段全部
  重生成，**n_ok=6**（sub12 306.95s 首次整段成功；总 915.0s ≤3h）。入口前置门：预注册身份/
  单差异/父哈希；**真实 tyro 配置烟测**（diff 恰 `{retargeter.step_size:[0.2,0.3]}`、规范化
  配置 sha `2952f455a332b056…` 跨 demo 稳定）；同根单一注册守卫；行 argv 携带覆盖；
  manifest 记录完整 default/recovery 解析配置。
- **比较（§7.3）**：`cmp/attempt_001/teacher_selection.json`（sha `695f88c3c90fb05d…`）P/UR
  模式（与 `--u0-dir` 互斥）。admission 全过（含注册对账腿：预注册哈希=现行文件、完整配置重
  diff 恰一字段、行 argv 覆盖、比较时新鲜烟测 step 0.3+规范化哈希一致）。**裁决
  `no_quality_gain`，complete_evidence=true**：
  - g1 身份可用 PASS（6/6）；g2 任务 MPJPE PASS（6/6 在界 0.0669–0.0756m）；
  - g3 穿透/脚 FAIL：UR 最大穿透 19.2–56.0mm vs P 1.5–25.5mm（sub12 56.0mm 为全场最差）；
  - g4 双手 FAIL：改善段数 1/3（仅 sub7 gap p95 −44%；coverage 增益 0 段）；
  - g5 退化 FAIL：六段全部含关键帧穿透违规比例上升族；sub10 手漂移、sub12 左手漂移、
    sub16/7/8 新增 validator 硬失败。
- 报告标签 UR（metrics 键/gaps/`ur_identity`），U0 模式报告与冻结基线逐位不变。

**过程缺陷（已修复并钉死）**：UR 改动最初在 U0 模式 admission 中无条件写入
`source_registration: null`——C2 汇入轮的新鲜比较因此与权威报告出现
`admission.source_registration` 差异、registration 被正确阻断。修复为仅 UR 模式携带该键
（§10.3"旧 CLI 无新参数时与基线一致"），回归断言入 `tests/test_omniretarget_recovery_source.py`。

## 6. C2：收口（decisions/attempt_004；**权威已更替为复核轮 attempt_006，见 §11**）

`--diagnosis` + `--recovery-report` 哈希钉住汇入，`a2_closure.json`（sha `8b346fa569ee04c0…`）：
`baseline_verdict=incomplete`（原报告路径/哈希不变）、`diagnosis=supports_step03_candidate`、
`recovery=no_quality_gain`、`training=not_ready`、`registration_status.complete=true`。
【复核轮订正：本轮汇入只记录了当时的文件哈希、未核验证据链（审阅缺口 6），且其引用的
`baseline/baseline_evidence.json` 已被同日更晚的收口调用覆盖（缺口 1）。该记录保留为历史证据，
权威 C2 以 §11 的 attempt_006（sha `5ca7960c…`，证据链核验通过）为准。】
按 §7.4 行 2 与 §12：当前候选停止，不追加候选；W1（weak_motion_only、接触损失 0、每模型
≤4k）的 A3/B/C 执行方案为下一阶段的独立预注册文档，本轮不实现、不启动任何训练。

## 7. 成本（§11 预算对照）

| 项 | 实测 | 预算 |
| --- | --- | --- |
| C0/C1/C2 三次收口入口（各含新鲜比较） | ~5min+16.9s+~4min | 合计 ≤30min |
| D0（对照+两次 sub12+两次 Phase I） | 387.7s | ≤60min（单次生成 ≤15min：实测 61.9–161.1s） |
| UR 六段生成 | 915.0s | ≤3h（单段 ≤30min：实测最长 306.95s） |
| UR 比较 | 16.2s | 预留内 |
| Phase I 每次求解 | 0.03s | ≤60s |
| GPU 训练 | 0（未创建） | 0 |
| 新产物 | <2GiB | 起步上限 2GiB |

## 8. 测试

- 新增：`tests/test_omniretarget_a2_closure.py` 17 项（C0/C1 §10.1 全反例+CLI 四例）、
  `tests/test_omniretarget_sub12_diagnosis.py` 16 项（§10.2 分类表全行/聚合/对照/超时进程组/
  上游环境真实锥问题自测）、`tests/test_omniretarget_recovery_source.py` 14 项（§10.3 注册门/
  混入/标签/回归）、**复核轮新增 `tests/test_omniretarget_a2_closure_review2.py` 22 项**
  （快照不可覆盖/失败行删除/身份内容钉住/D0 零子进程封锁+控制失败零 sub12/canonical·列映射·
  锥形状篡改/裁决记录核验/汇入链 8 反例+正例/重试上限与预算门/规范基线暴露额外旗标），并按
  §11 的语义强化更新三份既有测试的夹具（canonical 文件+形状字段/argv 烟测替身/快照路径）。
- 既有测试适配两处：`generate_one` 新增可选 `extra_argv` kwarg，review/review4 的 fake 签名
  加 `**kw`（行为断言不变）。
- 全量（复核轮实测）：**506 passed + 1 skipped**（`uv run --locked pytest tests/ -q`，1049s；
  首轮收口时为 484 passed + 1 skipped）。

## 9. 未运行/超出本轮范围

- 自研修复、T4 教师处理、train24/val31 生产、增强、W1 缓存/数据集/损失/训练入口：未实施
  （方案 §2.2 排除项）。
- sub12 之外演示的恢复性调整、其他半径/solver 参数候选：未运行（候选数=1，停止条件生效）。
- UR 各段只运行一次（无工程异常，重试未触发）。

## 10. 工程事实（本轮新增）

1. **tyro 1.x 把 `argv=` 改名 `args=`，旧 kwarg 落入 `**deprecated_kwargs` 被静默忽略**——
   解析结果仍是默认配置且无任何告警。"argv 里出现了 flag 字符串"不构成配置生效证据；生效
   配置必须经真实解析烟测+规范化哈希跨进程对账。该缺陷同时被生成入口（烟测）与 C2 汇入
   （注册对账腿）两道门覆盖。
2. **NaN 哨兵相等语义**：两份逐位一致的报告中无效行 sentinel 同为 NaN，`NaN != NaN` 使逐位
   对照器把同值报成差异——比较器对 float 必须显式定义 NaN==NaN 判同，否则每个缺产物行都成
   假差异。
3. **"新键默认写入"破坏基线逐位性**：给共享 admission 字典无条件加新键（即使值为 None）会
   使 U0 模式新鲜比较偏离冻结基线；模式扩展必须条件性写入，"与基线逐位一致"本身需要回归
   断言钉死。
4. **诊断快照的"零基帧"必须来自包装捕获**（frame_idx=171、帧内第 5 次求解），历史进度条
   数字只作旁证；两次运行关键数组逐位一致是确定性证据，Phase I 的 rho 在同一问题数据上
   逐位可复现（0.2486138173835237）。
5. **上游 dqa 是 36 维自身变量**（q_a_init_idx=-7 含浮动基），rho 不换算为项目 35 维变量——
   步长半径假设的检验必须在原问题的变量与尺度上进行。
6. **（复核轮）异常 traceback 只含被调链**：在 `wrapped_ssi` 的 except 块里，
   `retarget_motion` 是**调用者**且尚未退栈，永远不在当前异常的 traceback 里——要读它的
   局部（`retargeted_motions`/`i`）必须走调用栈（`f_back` 链）。首轮 prefix 因此从未落盘，
   且该缺陷不改变失败帧判定（frame_idx 来自 ssi 行参数，恒记录）。
7. **（复核轮）配置烟测的默认侧必须是规范基线命令**：从"记录 argv 去掉步长旗标"派生默认
   侧，会把额外旗标同时放在 diff 两侧——实测向记录 argv 加
   `--retargeter.penetration-tolerance 0.1`（默认 0.001）仍得单字段 diff。默认侧必须解析
   规范 U0 模板（`_canonical_base_argv`），额外旗标只出现在记录侧才可被逮住。
8. **（复核轮）固定路径快照=父哈希悬空**：C0 证据写固定路径时，同根后续收口调用把它覆盖
   （首轮 D0 父哈希 83287703 与盘上 6a6ce157 失配即此形态）；证据快照必须按 attempt 隔离且
   永不覆盖，下游（D0/G0/C2）才能按哈希回指。repo 身份同理：status 行列表只说"改了哪个
   文件"，tracked diff 哈希才钉住"改成了什么内容"。

## 11. 独立审阅 13 项缺口修复轮（2026-10-10，提交 `efece4a`）

独立审阅确认 UR 的 `no_quality_gain` 已独立复核、停止生产/W1 另立预注册的方向成立，但收口
实现存在 13 项缺口（6 P1+7 P2）；全部修复后按修正协议重跑权威链（度量与裁决逐位不变）。
**裁决不变：原 U0 `incomplete` 保留；UR `no_quality_gain`（complete_evidence=true）；训练
`not_ready`；W1 A3/B/C 另立预注册。**

**修复清单（编号=审阅项）**：①C0 快照按 `baseline/snapshots/attempt_NNN` 不可覆盖；②UR 配置
对账绑定实际生成命令（逐消费行真实 tyro 解析记录 argv，默认侧=规范 U0 模板；canonical sha
与记录对账——工程事实 7 的活体反例在修复验证时当场复现并逮住）；③`--u0-dir` 拒收 UR 目录
（index 注册标记或 manifest source_record 任一即拒）；④D0 完整前置准入（C0 passed/身份稳定/
资产/实际 data-path 输入 vs 冻结/环境 vs 生成记录五腿）+控制对照先行确认；⑤两次一致性扩证据
（canonical c/b 向量+q_a_indices 列映射+快照文件哈希+锥维度形状）+序列化器读 c、维度取自
矩阵+G0/C2 经 `verify_diagnosis_record`（裁决须有事实背书+父基线哈希可解析）；⑥C2 汇入核验
证据链（prereg 绑定/注册父诊断在盘完整/合并诊断=注册件或复核一致/报告结构不变量+裁决须
complete+admission 过+分母满）；⑦C1 继承 identity_stable/checkout_passed；⑧prefix 经调用栈
捕获（工程事实 6）；⑨消费检查验 manifest 行保留（`failure_record_dropped`/`cost_basis_empty`）；
⑩repo 身份含 tracked diff 哈希；⑪生成跨调用重试上限（两次失败后零调用）+UR 累计 3h 预算门
+costs.json 累计；⑫注册核验失败显式进 evidence_gaps；⑬training 未满足原因随合并恢复结果
条件化。

**修正协议重跑（全部 exit 0/预期裁决）**：

- **C0/C1**：`baseline/snapshots/attempt_001/baseline_evidence.json`（sha `b1fb3e6a…`，干净树
  rev `efece4a`、tracked diff=空哈希、运行期身份稳定）+ 新鲜比较 `baseline/cmp/attempt_005`
  与权威报告逐位一致（差异 []）；消费检查含行级保留核验（attempt_001 scope 六段/attempt_002
  scope sub12 行均在）；决策 `decisions/attempt_005`。旧固定路径文件与 decisions 001–004
  保留为过程证据。
- **D0 复跑（attempts 005/006/007，控制先行）**：前置准入五腿全过；sub16 控制对照逐位相等；
  两次 sub12 失败帧 171/帧内第 5 次逐位一致，**rho=0.2486138173835237 第三组独立复现逐位
  相同**（原两轮+审阅者构造复算+本轮两次）；**新证据补齐**：prefix 171 帧落盘
  （`prefix_debug.npz qpos[171,43]`）、canonical `arrays:["b","c"]`、798 变量/949 行、
  canonical 向量/q_a_indices/文件哈希/锥形状六项两两一致；裁决仍
  `supports_step03_candidate`。诊断墙钟 421.8s（两轮合计 ≤60min 预算内）。
- **UR 比较强腿复跑（`recovery/cmp/attempt_002`）**：注册对账腿全过（六段逐行 argv 烟测
  matches_registration 全 True、单差异/身份/canonical sha 一致），裁决 `no_quality_gain`
  （complete_evidence=true，g1/g2 PASS、g3/g4/g5 FAIL，度量与 attempt_001 逐位一致）。
- **C2 终局（`decisions/attempt_006`，sha `5ca7960c…`）**：证据链核验全过——prereg
  `f4676813` 绑定报告、注册父诊断（原 attempt_004，sha `7bcd253a`）在盘完整、合并诊断为
  复核件且帧/序号/rho 一致（`reconfirmation_agrees=true`）、UR 报告结构不变量+no_quality_gain
  须 complete+admission 过+分母满全部成立；training 未满足门落在 local_preferred 缺失+W1
  A3/B/C 前置（不再含已被 UR 满足的"无完整裁决"门）。
- **成本**：C0/C1+C2 各 16.7/16.3s（含新鲜比较），累计入册 `costs.json`（三次调用）。

**遗留说明**：原轮 attempts 001–004 的 D0 证据缺 prefix 与 canonical c 向量（缺陷已订正登记，
结论不受影响——失败帧/数组/Phase I 与 rho 均独立复现）；UR 生成证据（generation/attempt_001，
n_ok=6）不受本轮影响，未重跑（无新失败、重试上限与预算门为事前规则不追溯）。
