# W1 A3/B/C 实施与运行验证记录

方案：[omniretarget_w1_a3bc_plan.md](../plans/omniretarget_w1_a3bc_plan.md)（v0.1，2026-10-11）。
本记录登记接口实现、反例测试与正式运行 W0→A3→缓存/统计→S0→就绪判断的实际结果。
运行根：`derived/experiments/omniretarget_first/20261011_b6eef5c_w1_a3bc/`。
实施提交 `b6eef5c`（全接口+34 项反例）、`949d31f`（A3 出口判定修复）、`669d37e`（运行期三处簿记/配置修复）。

**结论（§10）：`registration_complete=true`、`a3_complete=true`、`b_complete=true`、`c_complete=true`、
`ready_for_w1_d=true`（`decisions/attempt_001/w1_readiness.json`，无未满足项）。**
就绪是工程入口判断，不是可行性声明：原 U0 `incomplete`、UR `no_quality_gain`、训练目标的
最终可行性结论全部不变；W1-D/E（每模型 ≤4,000 步、两模型累计 ≤12h GPU、val31 完整宏平均选模）
本轮**未运行**，须后续明确任务安排。

## 0. 提交与代码状态时间线

| 事件 | 树状态 | 产物 |
| --- | --- | --- |
| 实现提交 | `b6eef5c` 干净树 | 全部接口+反例（全量 597 passed+1 skipped 实测） |
| W0 注册 | `b6eef5c` 干净树（注册内记录 dirty=False） | `freeze/preregistration.json` |
| A3 attempt_001 | `b6eef5c` | 数据/门全对但出口布尔计算错（见 §2.1），保留为证据 |
| `949d31f` | 出口判定修复+回归测试 | — |
| A3 attempt_002（权威） | `949d31f` | `audit/attempt_002/training_manifest.json`，**PASS** |
| cache attempt_001/002 | `949d31f` | 崩于 precompute 残留 `train_rows` 引用（NameError），保留为证据 |
| cache attempt_003（权威） | `949d31f`+未提交修复（=后来 `669d37e` 的内容） | 键控日程 55 份+root_stats+身份 |
| S0 首启 | `949d31f` | 崩于 yaml 折叠串（§2.3），无产物 |
| S0 attempt_001（两模型，权威） | 同 cache attempt_003 树状态 | 两模型全部门通过 |
| `669d37e` | 运行期修复提交（precompute/yaml/成本时钟） | costs.json 两行负值已按 s0_summary 更正并留 note |
| 就绪判断 | `669d37e` 前后（只读） | `decisions/attempt_001/w1_readiness.json` |

W0 的注册以 `b6eef5c` 干净树落盘；A3 权威 attempt 与 cache/S0 的权威 attempt 在含未提交修复的树上
运行（修复内容=后来的 `669d37e`），各 attempt 自身的 manifest/identity 记录了当时实测的 git rev。
这与"正式运行前记录最终方案字节及实现提交"（§2.2）的偏差如实登记：三处修复均为簿记/配置错误，
不触及审查规则、名单、模型配置语义或数据协议；修复后全量复跑见 §5。

## 1. W0 运行预注册（§2）

入口 `scripts/w1_preregister.py`，wall 2.6s（预算 2h 含 A3）。

- **9 项方案锚点全部实测重哈希通过**（表 `w1_training.W1_PLAN_ANCHORS` 与方案 §2.1 逐位一致；
  C2 终局 `decisions/attempt_010` sha `34a392f9…`、UR 比较 `faa2df0c…`、UR 注册 `f4676813…`、
  A1 三件、split 三件）。
- **入口证据重推**（不只读顶层标记）：C2 登记完整+基线 `incomplete`/恢复 `no_quality_gain`/
  训练 `not_ready` 三状态不变；UR 报告经 `reverify_verdict_record`（u_side="UR"）重推无问题、
  admission 通过、证据完整、缺行空、且与 UR 注册文件哈希绑定。
- **train24 复算**：从锁定 train250 按受试者 `SHA256("20261008|teacher_pilot|"+demo)` 前 4，
  规范序 SHA-256 `8fbfd008…`（与方案 §3.1 声明逐位一致），pilot6 ⊆ train24，六受试者各 4 段。
- **输入**：55 段（train24+val31）NPZ/PT 实哈希入册；val31 腿全部等于 A1 冻结值（train18 为新记录）；
  train24∩val31=∅。
- **资产**：A1 资产闭包 120 项重哈希全过。
- **环境实测**：Python 3.12.13 / torch 2.14.0+cu126 / mujoco 3.13.0 / 主 uv.lock 哈希；
  发布数据生产环境未携带字段记 `not_published`（§1.2）。
- 独占性：注册文件已存在时只核验一致后复用——测试钉死（漂移输入 → exit 2、文件不重写）。

## 2. A3 名单与数据审计（§3）

入口 `scripts/audit_omniretarget_training.py`。权威 `audit/attempt_002`（wall 74.2s，预算 2h）。

### 2.1 出口判定缺陷（attempt_001 保留为证据）

55 段全过（数据/门值全部正确）但 exit FAIL：`all(gates.values())` 把证据列表
`missing_rows: []` 当 falsy 布尔门。修复抽 `exit_ok`/`BOOL_EXIT_GATES` 单一判定源
（`949d31f`），`w1_readiness` 同步复用；新增回归测试（非空 missing 经布尔门反射、预算停翻假）。

### 2.2 基础准入与配对（55/55）

- 结构准入全过：形状 `[T,43]`/`[T,52,3]`、fps=30、T 与 PT 一致、全有限、四元数范数在界
  （登记式归一化不回写源数据）、published↔project 布局往返（位置/关节逐位、四元数符号等价）、
  29 关节零越界（A2 期"发布教师 1,826 文件零越界"在 train18 上复证）。
- §4.3 配对门全过：人体相似变换残差 max **1.10e-7 m**（门 1e-5）、箱姿态映射/常数偏移残差同量级、
  拟合 scale 与 1.32/身高相对差 ≤8e-8 量级（逐行入册）。
- **源箱复算与 A1 冻结逐位相等 37/37**（pilot6+val31 全部命中 `pairing_source_boxes.npz`；
  train18 为新配对证据）。源/目标分离事实：偏移 pure-z **55/55**，范数 5.8–16.0 cm
  （与 A1 的 6.05–15.96 cm 一致）——两字段分离保存，未互相赋值。

### 2.3 质量登记（55/55 complete）

冻结验收器协议（自适应穿透+SLERP、逐 site 最坏脚指标、逐帧任务误差、认证区间）：

| 指标 | min–max（55 段） |
| --- | --- |
| 穿透最大值 | 1.5–43.1 mm（48/55 段 strict `penetration` 失败、5/55 `velocities` 失败） |
| 任务 MPJPE p95 | 0.062–0.145 m |

**按 §3.2/§3.3 语义：strict 质量失败是可登记的弱监督缺陷，全部保留在
`weak_motion_only` 准入内（`admission.motion` 与 `quality` 两字段分离），无逐帧屏蔽、无换段。**

### 2.4 出口（§3.4 全过）

55 段均有最终行（分母不缩）；train24 标签质量登记完整（val 亦完整）；独立帧 10,032、
完整窗口 255（窗口数不作独立演示数）；六受试者 train 可用窗口 12/32/13/19/17/17；
T<60 短片段零。

## 3. B 数据协议、缓存与统计（§4）

入口 `precompute_m3_cache.py --training-manifest … --out-dir …`。权威 `cache/attempt_003`（wall 5.7s）。

- **键控日程缓存**：55 份（4 workers spawn，输入序）；缓存键覆盖输入 NPZ/PT 实哈希、
  场景映射版本、源几何、资产哈希、协议、条件代码身份（`condition_code_sha256`，
  覆盖 teacher/pairing/contact_sites/w1_training 四文件）——文件名相同不等于可复用（测试钉死）。
- **root_stats 仅 train24**：`fit_w1_root_stats` 内部过滤 `split=="train"`，4,351 帧/24 文件、
  每帧一次；val 行内容漂移不动 train 统计（测试钉死）。std 下界处理沿用已验证实现。
- **身份闭合**：`cache_identity.json` 绑定注册+清单双哈希、代码/资产哈希、统计定义与文件名单摘要；
  消费侧（数据集/训练/S0）启动时全部重验，缺文件/漂移不回退（测试钉死互斥、协议错、输入漂移、
  派生漂移、缓存缺失五反例）。
- 就绪判断的 b 腿**真实构建了 train/val 两个数据集**（全部启动核验腿实际执行）：
  train 窗口 110（24 段、4,351 帧）、val 窗口 145（31 段、5,681 帧）、首批 `x0=(4,60,38)` 实取。

## 4. C：S0 数值烟测与小样本拟合（§6）

入口 `scripts/w1_s0.py`。权威 `s0/{deterministic,diffusion}/attempt_001`（各 wall 761.9s/765.0s，
合计 1526.9s ≤ 1h GPU 预算；峰值显存 5.47/5.47 GiB）。

小集合：pilot6∩train24 六段（`sub10_097/sub12_082/sub16_000/sub3_027/sub7_022/sub8_035`）
各自首个完整 60 帧窗；正式 DataLoader 先以全 train24 准入窗口（110 窗）通过 micro-batch=4 实取门。

| 门 | deterministic | diffusion |
| --- | --- | --- |
| ≥100 真实 optimizer steps | 1000（micro-batches 4000=steps×grad_accum，计数区分） | 1000（同） |
| 全有限 | true | true |
| 接触分支静默（w_contact=0） | true（无 contact part、无梯度贡献） | true |
| 固定 probe ≥40 次 | 101（step 0 起每 10 步） | 101 |
| 趋势（未加权 pos+rot6d+joint，首/末 20 均值） | **0.4168 → 0.0134**（首 probe 1.182 → 末 0.0117） | **0.4641 → 0.0126** |
| checkpoint 重载 probe 复现 | true（≤1e-5） | true |

趋势门只判拟合趋势，非泛化/搬运质量/显著性证据（§6.2）。扩散 probe 用专用生成器固定 t/noise，
训练 RNG 不受 probe/验证影响（全局流隔离测试钉死，见 §5 工程事实 2）。

## 5. 测试与工程事实

- 新增 `tests/test_omniretarget_w1.py` 34 项（合成世界全离线，不依赖 derived/ 或 resources/）：
  W0 证据篡改四反例+独占复用与漂移拒；准入反例族（形状/fps/有限性/退化四元数/关节越限/T）+
  合法归一化不回写+往返符号等价；源/目标分离与**仅改教师机器人态时条件逐位不变**；
  train24 规范序与哈希；注册核验逐腿反例；清单互斥/漂移/缺缓存不回退；val 不动 train 统计；
  缓存键覆盖身份；零接触分支（FK 调用计数=2+动作梯度保留）与 w_contact>0 回归；
  无 40 批截断+样本加权+固定噪声+RNG 隔离；末帧覆盖；raw 零 RepairPipeline+分母保持+
  外来 checkpoint 拒；S0 步数/累积计数+趋势+预算停+micro-batch 门；A3 出口判定回归。
- 全量：564→**598 passed + 1 skipped**（`uv run --locked pytest tests/ -q`；接口轮 597，
  出口判定回归 +1）。
- 工程事实（本轮新增）：
  1. **证据列表不是布尔门**：`all(gates.values())` 把空 `missing_rows` 判为假——真实 55 段全过轮
     exit FAIL。布尔门与证据字段必须分离判定（`exit_ok`/`BOOL_EXIT_GATES` 单一源）。
  2. **DataLoader 迭代器创建会抽全局 RNG**：`generator=None` 时 base_seed 取自全局流——
     "验证不重播种全局"不够，val loader 必须持独立生成器；隔离经 `torch.rand` 流对照实测钉死。
  3. **墙钟与单调钟不可混用**：`time.perf_counter() - time.time()` 产生 -1.8e9 的"负成本"——
     A2 第六轮"成本非负有限"错误类在成本记账侧复发；修复+按产物 summary 更正并留 note。
  4. YAML 相邻引号串拼接（Python 习惯）不是合法多行标量——配置在首启才暴露，
     `--help` 烟测不覆盖 yaml 解析；正式配置装载须有解析冒烟。
  5. 发布标签在 train18 上的已知缺陷形态与 pilot6/val31 一致（穿透 1.5–43.1mm、
     48/55 段 strict 穿透失败）——`weak_motion_only` 语义下登记不剔除。

## 5R. 审阅修复轮（2026-10-12，提交 `f2099f4`；上位条目的就绪判断与 A3/cache/S0 权威产物自此被本轮取代）

【本节权威产物又被下方 §5R2 第二轮审阅修复轮取代——缓存键因代码身份扩展全部 re-key、
S0/readiness 以新链重建；本节 12 项修复本身继续有效】

独立复核确认两模型 1000 步记录、拟合趋势与 checkpoint 末次 probe 成立、60 项相关测试全过，
但 **`ready_for_w1_d=true` 不予验收**：另发现 12 项缺口（10 P1+2 P2，源几何问题已影响条件数据）
已全部修复。**P1-1 源箱几何未应用相似变换尺度**：日程/质量推断按无尺度锁定 hull 测距——
修复为 body 系缩放（`dist(p, sK)=s·dist(p/s, K)`，非 `s·dist(p,K)`），尺度
（0.68–0.7995）自 `derive_source_box` 贯穿接触推断/审计质量重放/缓存/数据集，derived npz
增加 `similarity_scale`，缓存键升 v2（旧日程文件零复用）；**真实 55 段对拍：47/55 段、
527 个手部帧标记变化，与审阅者独立复算逐位一致**（序列计数依口径而异：对称差 198、
边界移动+消失 94；审阅者 89 在两种口径下未复现，方向与规模吻合，如实登记）。
其余：P1-2 readiness check_c 重算全部 S0 门（步数/微批/曲线/接触分支自 loss_history 重算/
checkpoint 身份绑定当前 manifest+缓存+统计），verdict 字符串不再被信任；P1-3 注册核验
必需集合（九锚点集合相等+inputs 精确=train24∪val31，空块=缺项=多项均拒）；P1-4 数据集
行级对账（全局唯一/行集合=注册选择/split 按注册成员）；P1-5 质量完整性核验实际测量字段
（零参考帧 None 合法）；P1-6 缓存内容哈希登记+消费侧复验（`verify_cache_contents`：
root_stats 文件哈希+逐日程 sha256+键重推+统计身份对账，数据集/S0/评估/就绪统一准入）；
P1-7 S0 预算单调钟+历史成本扣除+未知历史=停止；P1-8 raw 评估复用统一准入（preregistration
腿+缓存内容+checkpoint 四重身份）；P1-9 正式 best 规则执行（`formal_eval` 回调：val31 宏平均
teacher-fit FK MPJPE 选 best，窗口损失不再选择，缺证据无 best；回调接线首次实际执行在 W1-D）；
P1-10 W1 raw 报告与训练 out 的 derived 守卫+不覆盖既有产物（非空目录自动 attempt_NNN）；
P2-11 S0 小集整批等权（batch=小集全量 6，24 样本/步，消除 4+2 碎批的 12≠16）；P2-12 成本
台账逐模型各自计时+attempt+累计预算，缓存入口入册，S0 仅认 `s0_*` 台账。

**新权威链** `derived/experiments/omniretarget_first/20261012_f2099f4_w1_a3bc_reviewfix/
w1_reviewfix/`（干净树 f2099f4；旧链 `20261011_b6eef5c_w1_a3bc` 保留为被取代证据——其
S0 证据按未缩放日程构建）：W0 注册 2.4s（55 输入/120 资产/九锚点全配）；A3 attempt_001
PASS（55/55 准入、源箱=A1 冻结 37/37 逐位、尺度入册、质量完整性按新谓词全过、strict 穿透
失败 48/55 登记不剔除，墙钟 66.8s）；cache attempt_001（55 键控日程逐文件 sha256+root_stats
文件哈希入册，4351 帧/24 文件，消费侧复验 passed，5.4s）；S0 双模型新预算（historical 0.0）
各 1000 步全过——确定性 1295.6s 趋势 0.394→0.0135、扩散 1305.9s 趋势 0.429→0.0113、
重载 ≤1e-5、峰值 8.16GiB、合计 2601.5s≤3600s（整批后单模型成本约为旧轮 2×：660→1296/
870→1306s，如实登记）；readiness decisions/attempt_001 五门全 true（check_c 为重算门：
`recomputed_gates` 全 true+checkpoint 身份对当前链闭合）。测试 598→**616 passed + 1 skipped**
（新增 18 项反例 `tests/test_omniretarget_w1_review.py`：缩放 hull 对拍直接缩放网格真值/
掠过点日程翻转/尺度缺失拒载/空锚点空输入/val 改 train/删行重复/质量字段剥离/统计+日程篡改/
键版本陈旧/预算耗尽与未知历史/篡改缓存与 PT 漂移评估拒绝/宏平均恶化与缺证据无 best/
守卫与 attempt 隔离/doctored 摘要四变体/整批与逐模型台账）。

工程事实（本轮新增）：
1. **相似变换的几何缩放不可省**：位姿映射应用尺度后，几何尺寸必须同乘——body 系
   `dist(p, sK)=s·dist(p/s, K)`，错误形态 `s·dist(p, K)`（只缩结果不缩查询点）在自造
   反例前不易察觉；对拍"直接缩放网格+trimesh 真值"是唯一可靠锁定（本实现第一版即犯此错，
   被真值测试逮住后修正）。
2. **"必需集合"必须按集合判定**：遍历"现有条目"的核验对删除腿免疫——空 anchors/inputs
   块照常"passed"；集合相等（缺/多都报）+逐项哈希两层缺一不可。
3. **文件名不能证明内容**：键控缓存若只按文件名寻址，改写内容（统计均值+100m、接触标记
   0→60）照常消费；生产侧逐文件 sha256+消费侧重哈希+键从当前代码/资产重推才闭合。
4. **verdict 字符串不是门**：摘要可被改写（steps=0/probes 删空/checkpoint 缺失/外来身份仍
   verdict=passed）；就绪判定必须从原始事实（曲线/loss_history/checkpoint 文件）重算并
   与记录 verdict 对账。
5. **碎批等权记账**：batch 4 对 6 样本 drop_last=False 产生 4+2 微批，等权平均=实际 12
   样本当 16 用且权重失真——整批（batch=集合全量）或按样本数加权二选一，并登记
   `small_set_batch_size`/`samples_per_optimizer_step`。
6. **共享预算需要台账而非记忆**：重跑同一 run 根时，历史成本只能来自逐阶段/逐 attempt 的
   台账行；有 attempt 无台账行=未知历史=停止条件（否则预算上限只是纸面约束）。

## 5R2. 第二轮审阅修复轮（2026-10-11，提交 `a0dbfa1`；§5R 的就绪判断与 A3/cache/S0 权威产物自此被本轮取代——12 项修复本身继续有效）

【本节权威产物又被下方 §5R3 第三轮审阅修复轮取代——re-key 因 w1_training.py 纳入
S0_REQUIRED_LOSS_KEYS 常量再次发生、readiness C 门按更强谓词重算；8 项修复本身继续有效】

独立复核确认源几何修正、训练统计与两模型 S0 结果成立（140 项相关测试、55 份缩放日程、
47 段/527 标记、4,351 帧统计逐位一致；checkpoint 末次 probe 差值 0；"89"复算成立=demo×手侧
整段二值序列变化数），但另发现 **8 项缺口（5 P1+3 P2）**，全部修复：

- **P1-1 C 门训练证据深度重算**：`_recompute_s0_verdict` 新增 `loss_history_present`
  （删 loss_history.json 不再被 all([]) 空真接受）/`loss_history_contiguous`（步号恰为
  1..N 且行数=optimizer_steps）/`all_finite` 扩展为每行**全部**损失字段 isfinite（原只查
  gnorm，NaN total 通过）/`probe_errors_finite`+`probe_schedule_valid`（首探 step 0、严格
  增、后续步号=注册 cadence 整倍数、末探覆盖最终状态）；小集从 ⊆pilot6 收紧为**集合相等**
  +`small_set_batch_size` 对账（缩为一个 demo 不再过）。
- **P1-2 checkpoint 实际重建加载重放**：S0 每次尝试新增 `probe_replay.pt`（固定 probe 的
  输入 batch+专用 t/noise+记录的末次误差）；`check_c` 的 `_verify_checkpoint_deep` 从
  checkpoint config 重建 TrainConfig→`build_s0_modules` 重建模型→strict `load_state_dict`
  （清空/异构 state 被拒）→全参数 isfinite（NaN 权重被拒）→内嵌 root_stats 与缓存
  root_stats **逐位相等**（+100m 被拒）→CPU 重放固定 probe 与记录误差对账（容差
  1e-3+1e-3·|rec|）。`reload_reproducible` 标记不再是唯一证据。
- **P1-3 正式选模分母固定注册 val31**：`make_formal_eval` 的 expected=preregistration
  `selection.val31` 全集（原按 `admission.motion` 过滤——一段标签被拒即缩小分母，
  n_ok=1/n_expected=1 判 complete 仍可选 best）；缺行/未准入/不可载标签各留 gap 行
  （`row_missing`/`label_not_admitted`/`label_unloadable`），宏只在**全集** ok 时存在。
- **P1-4 质量完整性对照参考帧核验测量存在性**：`quality_registration_complete` 以
  `ref_grasp_frames`/`ref_support_frames`（并与 `reference_intervals` 复算对账，`sup`=
  ground/object）为存在性契约——参考帧>0 而 hand/foot 侧为 None = **缺测量**（审阅者
  sub10 左手 46 帧反例），参考帧=0 才允许 None；`n_ref_frames` 与参考帧数互查；新增必需
  字段 `penetration_violation_ratio`（有限）/`fps`（有限）/`both.denominator_frames`>0 时
  `valid_coverage`+`gap_series_p95_m` 有限（清空 both 被拒）。真实 manifest 55/55 全过。
- **P1-5 S0 成本盘×索引×台账三方对账**：`_upsert_cost` 按 (stage, attempt) 原位登记
  （事前 `running` 行+全退出口更新）；崩溃路径补写 incomplete 摘要+索引行+`crashed` 台账行
  后重抛；`historical_s0_cost_s` 要求每模型 盘上 attempt 目录=索引=台账行 且无重复行、无
  `running` 残留、墙钟有限≥0——崩溃尝试留下无索引 attempt 的洗零路径关闭（=未知=停止）。
- **P2-6 budget_stopped 裁决**：`run_s0_model` 门新增 `completed_within_budget`，预算停机
  裁决=incomplete（原趋势过关仍 passed）；readiness 重算同拒。
- **P2-7 缓存预算执行**：`run_w1` 读 `CACHE_BUDGET_S`(7200s)——历史 `w1_cache` 台账扣减、
  未知/耗尽即拒；作业分块（块=workers）块间查单调截止；超限=INCOMPLETE 尝试
  （`complete:false`+`missing_demos`+exit 2，不造统计）；`verify_cache_contents` 消费侧
  拒 `complete≠true` 的缓存。
- **P2-8 缓存代码身份纳入距离实现**：`CONDITION_CODE_FILES` 增加
  `src/ireset/geometry/distance_adapter.py`（本轮源几何根因所在文件；原清单漏列——仅回退
  缩放查询不改变 condition_code_sha256，旧键日程仍可复用）。全部缓存键随之 re-key。

**新权威链** `derived/experiments/omniretarget_first/20261011_a0dbfa1_w1_a3bc_reviewfix3/
w1_reviewfix3/`（干净树 a0dbfa1；§5R 链与 `20261011_a0dbfa1_..._reviewfix2` 误参轮
【A3 --out 误传 $RUN/audit 产生 audit/audit 嵌套致 readiness a3_no_attempts；其
W0/A3/cache/S0 数值本身有效，保留为调用契约证据】均保留）：W0 2.5s；A3 attempt_001 PASS
（55/55 准入、质量完整性按新谓词全过，66.9s）；cache attempt_001（complete:true、预算块
52.5s/7200s、55 键控日程——与 §5R 轮文件名**零重叠**=re-key 实证，53.5s，较 §5R 5.4s 慢
系分块并行的 spawn 冷启动成本，预算内如实登记）；S0 双模型各 1000 步 PASS——确定性
1295.0s 趋势 0.394223→0.013504、扩散 1263.9s 趋势 0.428729→0.011308（与 §5R 轮**逐位一致**
=代码身份/预算/台账改动不触训练数值）、probe 101、`probe_replay.pt` 双双在册、S0 合计
2558.9s≤3600s；readiness `decisions/attempt_001` 五门全 true——C 门为重算门且
`identity_problems` 空、checkpoint 重放误差 0.0115/0.0087 与记录末探匹配。台账 6 行齐
（w0 2.5/a3 66.9/cache 53.5 complete:true/s0_det 1295.0 done/s0_diff 1263.9 done/
readiness 12.7）。测试 616→**621 passed + 1 skipped**（审阅文件 18→23 项：loss 历史
删除/NaN/断号、probe 脱 cadence、小集缩水双向、budget 裁决（受控单调钟）、崩溃洗零关闭、
缓存预算耗尽 exit 2+消费侧拒 incomplete、代码身份覆盖 distance_adapter、formal 分母
gap、质量存在性九反例+合法零参考侧）。

工程事实（本轮新增）：
1. **摘要计数不是训练证据**：loss_history 删除后 all([]) 空真、NaN 只查 gnorm、小集 ⊆ 判定
   ——三类都让"门重算"退化为"摘要复读"；证据链必须落到逐行（连续步号/全字段 isfinite/
   probe cadence）与集合相等。
2. **torch.load 只证明容器可读**：清空 state/NaN 权重/+100m 内嵌统计都在"身份标签+步数
   不变"下通过；唯一闭合=生产侧落重放工件（固定输入+t/noise+记录误差），消费侧重建模块
   strict 加载+逐位统计对账+实际重放。
3. **分母必须锚定注册名单而非幸存标签**：任何按 admission 过滤 expected 的评估都在用
   准入结果改写验证集；缺标签保留 gap 行、宏只在全集上存在，才不会被"少评估=更好"反向激励。
4. **存在性契约要有独立账本**：测量为 None 是否合法不能由测量自身报告（n_ref_frames 可被
   一并改写）；参考帧数需第二表示（ref_*_frames）并与第三表示（intervals）复算对账。
5. **共享预算的"已花费"来自三方对账**：盘上 attempt 目录、attempts 索引、台账行按
   (model, attempt) 集合相等——崩溃在分配后登记前的窗口靠事前 running 行+全退出口 upsert
   关闭；重复行/running 残留=未知=停止。
6. **代码身份清单必须覆盖行为的真实依赖闭包**：清单列了调用方（contact_sites）却漏了
   被调方（distance_adapter），几何回退不触发 re-key；清单按 import 闭包而非按"谁定义协议"
   圈定。

## 5R3. 第三轮审阅修复轮（2026-10-12，提交 `027cb3e`；§5R2 的就绪判断与 A3/cache/S0 权威产物自此被本轮取代——8 项修复本身继续有效）

独立复跑确认 §5R2 权威链数值成立（57 项 W1 测试+60 项回归、4,351 帧统计、重放输入、
扩散 t/noise 逐位一致，双模型重放误差 ≈0.01150036/0.00872141，S0 成本 2558.9s），但另发现
**8 项缺口（4 P1+4 P2）**——全部是上轮修复引入或未闭合的核验谓词缺口，全部修复：

- **P1-r3-1 重放有限性前置**（`w1_readiness._verify_checkpoint_deep`）：原实现先比容差
  再看结果——probe_replay.pt 的 x0 改一个 NaN 元素，实际重放误差变 NaN 而
  `abs(NaN−rec)>tol` 恒 False，check_c 仍 passed。现重放**输入**（batch 全张量+probe_t+
  probe_noise）、**输出**（四误差字段）与工件**记录误差**全部 isfinite 前置，容差比较仅在
  双侧有限后执行。
- **P1-r3-2 趋势证据与重放工件对账**（同函数）：原实现只把重放结果与工件自己的
  last_probe 比对——仅改 summary.probes 为 100→50 下降曲线（checkpoint/工件不动、日程/
  有限/趋势全绿）仍过门。现 summary 末探必须与工件记录的 last_probe **步号+四误差字段
  逐位相等**（同一次运行的两种序列化，精确相等是判据而非容差）。
- **P1-r3-3 必需损失字段显式化**（`w1.S0_REQUIRED_LOSS_KEYS`）：原 all_finite 只查
  **现存**字段有限+gnorm 存在——真实 1000 行缩为 {step,gnorm} 仍算完整证据。现按 W1 配置
  （LossAssembler 恒 emit：pos/rot6d/joint/geodesic_rad2/vel/fk_m2/total；w_contact=0 下
  contact 非法）锁定必需集合，`loss_fields_complete` 门先查存在再查有限。
- **P1-r3-4 参考计数缺键即缺证**（`audit_omniretarget_training.quality_registration_complete`）：
  原 `counts.get(side, <复算值>)` 把删除的键洗成复算值、`need=get(side,0)` 把缺键当零参考
  ——真实 sub10 行删 `ref_grasp_frames.l`+`hand.l=None`（左手实有 46 帧参考）仍过；both
  分母 3→0 并删双手测量同样被接受。现左右键**必须齐全**且等于区间复算值（缺键=
  `ref_frames_key_missing`+对账不符双报），both 分母从 reference_intervals 复算**左右抓握
  帧交集**（sub10=3），篡改 0/40 与复算不符即拒、交集>0 才允许有限测量、交集=0 只允许 None。
- **P2-r3-5 probe 日程全量比对**：原谓词只证步号严格递增+cadence 整倍数——删 step=10 单点
  后剩余仍满足。现步号列表必须**恰等于** `[0]+range(every, steps+1, every)` 完整日程
  （以已完成步数为界）。
- **P2-r3-6 缓存全阶段截止**（`precompute_m3_cache.run_w1`）：原截止只在块间检查——最后
  一块（及后续 root_stats 工作）超预算仍 complete=true/consumed_s=1.2>1.0/exit 0。现块返回、
  统计计算、最终登记三处查 deadline，任一超限走同一 incomplete 写入（complete:false+
  `cache_budget_exhausted`+consumed_s 入册+exit 2，missing_demos 为空时如实登记"仅预算
  超限"），不伪造统计。
- **P2-r3-7 S0 预算判定独立于步数**（`w1_s0.run_s0_model`）：原 `budget_stopped =
  越界 and step<cfg.steps`——真实完成 400 步+41 probe 后 checkpoint 阶段越界仍 passed。
  现判定移至保存+重载检查**之后**且无条件于步数：保存、重放、最终登记都在预算内。
- **P2-r3-8 缓存崩溃登记+三方历史**（`precompute_m3_cache`）：原 `_historical_cache_cost_s`
  只数索引条数——worker 在 attempt 分配后抛异常留下无索引无台账的 attempt_001，历史读 0.0，
  下次照常启动 attempt_002。现分配后**事前 running 台账行**（`_upsert_cost` 按 (stage,
  attempt) 原位 upsert）+全退出口更新+崩溃写最小 identity/索引行/`crashed` 行后重抛；
  历史改**盘×索引×台账三方**按 attempt 集合相等（盘上未登记尝试=未知=停止，未 keyed/
  重复/running 残留行=未知）。

**新权威链** `derived/experiments/omniretarget_first/20261012_027cb3e_w1_a3bc_reviewfix4/
w1_reviewfix4/`（干净树 027cb3e；§5R2 链保留为被取代证据）：W0 2.5s；A3 attempt_001 PASS
（55/55 准入、质量完整性按 r3-4 新谓词全过，73.1s）；cache attempt_001（complete:true、
consumed 53.8s/7200s、55 键控日程——与 §5R2 轮文件名**零重叠**：w1_training.py 是
CONDITION_CODE_FILES 成员，新增 S0_REQUIRED_LOSS_KEYS 常量即重键全部日程，re-key 实证）；
S0 双模型各 1000 步 PASS——确定性 1352.8s 趋势 0.394223→0.013504、扩散 1362.4s 趋势
0.428729→0.011308（与 §5R/§5R2 轮**逐位一致**=谓词/预算/台账改动不触训练数值）、probe
101=完整日程 0..1000、probe_replay.pt 在册、S0 合计 2715.2s≤3600s；readiness
`decisions/attempt_001` 五门全 true——C 门重算含 `probe_schedule_valid`（全量日程）/
`loss_fields_complete` 双新门、`identity_problems` 空、checkpoint 重放 0.011500352062284946/
0.008721405640244484 与 summary/artifact 末探**逐位相等**（对账门实证）。台账 6 行齐且
每 attempt 恰一行（w0 2.5/a3 73.1/cache 54.9 done complete:true/s0_det 1352.8 done/
s0_diff 1362.4 done/readiness 13.8）。测试 621→**628 passed + 1 skipped**（审阅文件
23→30 项：NaN 重放输入+NaN 记录误差、伪造下降曲线保留真 checkpoint、{step,gnorm} 与
{step,total,gnorm} 缩行、删 step-2 单点日程、最后一块 1.2s 超限 incomplete+无统计伪造+
消费侧拒、完成目标步数后尾部越界（受控单调钟+保存钩子）、缓存崩溃登记+盘上未登记尝试
拒绝启动；既有质量测试扩展缺键/both 分母篡改 0 与 40/合法零参考侧保留）。真实 manifest
55/55 过 r3-4 新谓词，审阅者 sub10 双反例（46 帧缺键+both 3→0）精确命中。

工程事实（本轮新增）：
1. **NaN 比较 laundering**：`abs(NaN−x)>tol` 恒 False——一切容差判定必须以双侧 isfinite
   为前置条件；重放的**输入**有限性与输出同等必要（NaN 输入必产 NaN 输出，但输入检查
   独立定位污染源）。
2. **同一测量的两种序列化必须对账**：summary（JSON）与工件（torch.save）来自同一次运行
   的同一 Python float，精确相等是判据；不做该对账时"趋势证据"与"重放证据"可以各自为真
   而互不相干（伪造曲线+真 checkpoint 的组合恰恰如此）。
3. **"必需字段"必须是显式配置派生清单**：只查现存字段有限=把"测了什么"交给篡改者决定；
   存在性（loss_fields_complete）与有限性（all_finite）分离成两道门。
4. **证据字典的 `.get(key, default)` 是删除 laundering**：缺键落到复算值或零都把"缺证据"
   变成"合法默认"；键的存在性本身是证据（缺键单独报因），值只与第三表示对账。
5. **预算约束必须覆盖全部阶段边界并独立于任务完成**：块间检查漏最后一块与统计阶段；
   `step==target` 不是预算豁免——保存/重放/登记都在预算内，超限即 incomplete。
6. **崩溃窗口靠事前登记关闭**：attempt 分配到退出登记之间的崩溃只能靠"先写 running 行"
   兜底；历史核验按盘×索引×台账三方集合相等，"未知"必须是停止条件而非零。

## 6. 就绪判断与后续边界（§10）

【本节首段就绪指针已被 §5R3 第三轮审阅修复轮取代——权威就绪记录为
`20261012_027cb3e_w1_a3bc_reviewfix4/w1_reviewfix4/decisions/attempt_001/w1_readiness.json`
（五门全 true，check_c 为重算门+checkpoint 重建重放门+summary/工件末探对账门）；本节
其余边界表述继续有效】
`decisions/attempt_001/w1_readiness.json`：五门全 true、unmet=[]。就绪由事实重算
（A3 行复算、缓存身份对账、双数据集真实构建、S0 门读取）；exit 0=记录完成。

**未运行/边界**：W1-D/E 正式 4k 训练、val31 完整生成比较、val6 详细分析、W1-F 均为
`not_run`（契约在方案 §5.2/§10 固定：每模型 ≤4,000 optimizer steps、两模型累计 ≤12h GPU、
完整 val31 宏平均选模、缺证据不得 best）；裸输出诊断全拒也须交付 0/N 与失败分类，
不得为通过率启用修复或改教师。裸输出不合格时只交付动作基线及失败证据
（父方案 §3.6 语义不变）。
