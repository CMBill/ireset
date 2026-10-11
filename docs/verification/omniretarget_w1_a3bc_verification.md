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

## 6. 就绪判断与后续边界（§10）

`decisions/attempt_001/w1_readiness.json`：五门全 true、unmet=[]。就绪由事实重算
（A3 行复算、缓存身份对账、双数据集真实构建、S0 门读取）；exit 0=记录完成。

**未运行/边界**：W1-D/E 正式 4k 训练、val31 完整生成比较、val6 详细分析、W1-F 均为
`not_run`（契约在方案 §5.2/§10 固定：每模型 ≤4,000 optimizer steps、两模型累计 ≤12h GPU、
完整 val31 宏平均选模、缺证据不得 best）；裸输出诊断全拒也须交付 0/N 与失败分类，
不得为通过率启用修复或改教师。裸输出不合格时只交付动作基线及失败证据
（父方案 §3.6 语义不变）。
