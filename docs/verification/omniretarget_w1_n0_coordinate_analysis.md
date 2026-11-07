# W1 N0 坐标级数值缺口分析记录

日期：2026-11-07。对应方案：[数值缺口有限收口与受控拟合诊断方案 v0.4](../plans/omniretarget_w1_numerics_closure_fit_diagnostic_plan.md) §4。
**状态：N0 已按冻结格式 `n0_coordinate_flags_v1` 执行一次并完整收口；N1/I/R/D1/D6/F 均未实施。** 本轮只读消费 T6 已保存证据，未修改任何旧报告、旧契约、旧授权或旧工件；未运行模型/FK/梯度/更新/训练；未恢复 T6 一次性授权。

## 1. 输入与来源

运行根 `derived/experiments/omniretarget_first/20261102_2704709_w1_real_training_update_audit/`，只读消费（完整 SHA 见 `coordinate_index.json.sources`，此处为前16位）：

| 来源 | SHA-256（前16） |
| --- | --- |
| registration/attempt_003/registration.json | `87961838cabd69dc` |
| production/attempt_003/p_deterministic.npz | `44bb2b195fbf4d67` |
| production/attempt_003/p_diffusion.npz | `8d02383d4be33305` |
| independent/attempt_002/i_rows.json | `07603003f75896d4` |
| independent/attempt_002/i_report.json | `4b79b7cdbaca1e1c` |
| independent/attempt_002/i_deterministic.npz | `bd5e0d1653f342db` |
| independent/attempt_002/i_diffusion.npz | `21359c18c52ed9b1` |

另只读对照 `derived/audit/w1_real_training_update_audit/gradient_numerics_retest/real_analysis/attempt_001/analysis.json`（SHA-256 `d03f2d78c6933b8ada37766674367de903024b88f7019464e6ff8472f4d6cd3a`；旧629项对照，不改写其分类）。

## 2. 冻结格式与实现

- 实现：[src/ireset/eval/w1_n0_coordinate_flags.py](../../src/ireset/eval/w1_n0_coordinate_flags.py)（格式常量与位定义在执行前冻结于模块 docstring），入口 [scripts/analyze_w1_n0_coordinate_flags.py](../../scripts/analyze_w1_n0_coordinate_flags.py)，反例测试 `tests/test_w1_n0_coordinate_flags.py` **9 passed**。
- 行宇宙：`i_rows.json` 中全部 528 条 `adamw:` 行，规范序 (model, update, parameter)；行多重集对 registration roster（n_det=130、n_diff=134，×2 updates）核验。
- `coordinate_flags.npy`：一维 `uint16`，按行序 C-order 展开，共 **15,134,440** 坐标（≤16,000,000 入口预验上限；文件 30,269,008 bytes ≤ 32,000,000+1024 上界）。
- 16 个位定义：b0/b1 `DELTA_UNOBSERVABLE_P/I`（ref≠0 且不可辨识：写界 > 0.25·rtol_same·|ref| 或 theta 未位级变化）、b2/b3 `DELTA_WRITE_ERASED_P/I`（ref≠0 且实测增量为0）、b4/b5 `DELTA_SAME_INPUT_GATE_FAIL_P/I`（可辨识但同输入相对门失败）、b6 `DELTA_ZERO_INVENTED`、b7–b10 theta/m/v/delta 跨路径门失败、b11–b13 m/v/theta 零值或符号不一致、b14 `INTERVAL_OUTSIDE`（任一保存量越出 K_* 传播区间）、b15 `SOURCE_ANOMALY`（重推与保存字节不一致/非有限/形状漂移/绑定失配/分类不可重放）。
- 语义权威：门公式、`obs`/`ok` 判据、`bound_delta`=更新后参考 theta 半ULP、K_* 区间语义、atol/rtol 均取自已审计实现 `src/ireset/eval/w1_real_training_update_audit.py` 并与 `registration.tolerances` 逐项对账（atol≠0 即拒绝启动）。

## 3. 方法与预算

- 纯 CPU、只读、逐行流式处理：每行从 i-NPZ 读 25 个后缀数组 + 3 个状态绑定数组，从 p-NPZ 读 `theta_before/after` 做逐位绑定；不加载 checkpoint、不构造模型、不执行 forward/backward/FK/AdamW。
- 每行重推：obs/ok 可辨识掩码与同输入门（逐位对账保存 uint8）、半ULP写界（float64 精确相等）、`i_theta/i_m/i_v` 与 `m_i/v_i/theta_i` 绑定、p-NPZ theta 绑定、逐坐标 16 位标记；行级从标记重放 `classification`/`gaps` 并与保存值对账。
- 预算按方案 §8：墙钟上限 900s、内部预留 60s、工作截止 840s；t0 于 CLI 入口采样，逐行检查工作截止。实测 **44.12s**，未触发截止，`coverage_complete=true`。
- 容量：flags 30,269,008 B、index 291,496 B（≤16MiB）、gap_rows.csv 151,328 B（≤8MiB）、extrema 3,033,180 B / 6,210 条（≤16,896 条、≤64MiB）、summary 1,186 B；合计约 33.7MB ≪ 493,374,464 B 保守上界。磁盘预检空闲 27GiB ≥ 1GiB+收口余量。

## 4. 结果

### 4.1 事件重放（与 T6 报告逐项一致）

| 事件 | 本分析（由标记重推） | T6 real_analysis |
| --- | ---: | ---: |
| 仅 precision 行 | 350 | 350 |
| 仅 sensitivity 行 | 189 | 189 |
| 两类兼有行 | 147 | 147 |
| 带 gap 行合计 | 392 | 392 |
| 528 行分类重放一致 | 528/528 | — |
| 行级 SOURCE_ANOMALY | 0 行 | — |

### 4.2 坐标位分布（15,134,440 坐标）

| 位 | 坐标数 | 说明 |
| --- | ---: | --- |
| DELTA_UNOBSERVABLE_I | 1,214,770 | 更新量低于写入分辨率，I 侧不可辨识 |
| DELTA_UNOBSERVABLE_P | 1,214,768 | 同上，P 侧 |
| V_CROSS_PATH_GATE_FAIL | 1,499 | 均在保存 K_v 区间内 |
| M_CROSS_PATH_GATE_FAIL | 1,346 | 均在保存 K_m 区间内 |
| DELTA_CROSS_PATH_GATE_FAIL | 844 | 均在保存 K_delta 区间内（obs 掩码内） |
| DELTA_WRITE_ERASED_P / _I | 203 / 201 | “完全未写入”极少；绝大多数精度坐标是“写入了但低于可辨识阈值” |
| THETA_CROSS_PATH_GATE_FAIL | 82 | 均在保存 K_theta 区间内 |
| M_ZERO_OR_SIGN_DISAGREE | 2 | — |
| 同输入公式门失败 / zero-invented / theta·v 零符号不一致 / INTERVAL_OUTSIDE / SOURCE_ANOMALY | **全部 0** | — |

比例极值（行级，定义见 gap_rows.csv 表头）：max_same_delta_ratio 2000.0（=1/rtol_same，完全擦除坐标的解析上界）、max_delta_xp 2725.5、max_m 3916.8、max_v 1145.5、max_theta 10.36。

### 4.3 优先覆盖行：diffusion u2 `model.net.root_out.weight`

`adamw:diffusion:u2:model.net.root_out.weight`（2,160 坐标）：gap=both，precision 353 坐标（bits 0x0003，全部不可辨识而非擦除）、sensitivity 1 坐标（m 门比 1.0861）。行级 max_theta_ratio 0.013891167966513261、max_state_ratio 1.0861218843 与 T6 执行报告记录**逐位一致**。注意：本分析 `max_same_delta_ratio`（36.05）按冻结定义覆盖全部坐标（含不可辨识坐标），与 T6 行字段仅在 obs 掩码内取值（0.2324）口径不同，两者不矛盾。

### 4.4 结论（按方案 §4 标签）

- 350 行 → `precision_limited`：精度缺口坐标全部为“更新量低于 float32 写入分辨率/不可辨识”，其中仅 404 个坐标（P+I 合计）是写入被完全擦除，其余为“变化了但不足以通过相对增量门”。
- 189 行 → `cross_path_sensitive`：跨路径门失败的坐标全部落在已保存的 float64 传播区间 K_* 内，即与 float32 相消敏感一致；无任何区间外差异。
- **0 行 `implementation_issue_candidate`、0 行 `evidence_incomplete`。**
- N1 建议输入：N0 支持形成“有限数值收口说明”——同输入有效 float32 公式逐位一致（obs/ok/半ULP/公式重放全过）、剩余缺口可由保存的舍入/敏感区间解释；严格跨路径门仍未通过（189 行），长期训练/动作质量仍未覆盖。

## 5. 边界（不因本记录改变）

- 本分析是对**已保存证据的内部一致性重放**：它确认保存量自洽且缺口可由精度/相消解释，**不**证明 539 项旧 gap 消除、不构成对旧 mandatory gap 的通过豁免、不宣称实现无错或训练质量已验证。
- T6 结论保持：`audit_incomplete`、`covered_contract_passed=false`、cohort 34 次更新/1347.195507s、一次性授权已消费；`objective_coverage=partial`、`full_objective_contract_passed=null`、`factor_selected=false`、`auto_execute=false`、`final_interaction_method=not_ready`、`production_integration=false`。
- 旧 629 项分类未改写；本记录仅在 gap_rows.csv 中按 (key,kind) 只读附注旧链 state（本次命中为空，因 539 项均属新链 kind 集合的 persisted/new 记录格式差异，未作推断）。
- 未宣称 CPU 算术等价于生产 CUDA 路径；未执行 §5 N1（待本记录审阅后另行安排）。

## 6. 交付物

| 文件 | 说明 |
| --- | --- |
| [coordinate_flags.npy](../..//derived/audit/w1_numerics_n0_n1/n0/attempt_001/coordinate_flags.npy) | 15,134,440 × uint16 标记，SHA `e62a0d8cd27b186c…`（完整值见 summary.json） |
| [coordinate_index.json](../..//derived/audit/w1_numerics_n0_n1/n0/attempt_001/coordinate_index.json) | 位定义、行 offset/shape、原始 NPZ 键模板、来源 SHA、消费容差 |
| [gap_rows.csv](../..//derived/audit/w1_numerics_n0_n1/n0/attempt_001/gap_rows.csv) | 全部 528 行逐行：gap 类别、坐标计数、标签、分类重放、比例极值、旧链附注 |
| [gap_coordinates_extrema.csv](../..//derived/audit/w1_numerics_n0_n1/n0/attempt_001/gap_coordinates_extrema.csv) | 6,210 条极值（每行每类≤16，按门比/绝对误差预定排序+坐标序决胜） |
| [summary.json](../..//derived/audit/w1_numerics_n0_n1/n0/attempt_001/summary.json) | 事件重放、标签计数、预算事实、产物 SHA、coverage_complete=true |

## 7. 修订记录（2026-11-07 独立审阅后）

独立审阅未通过 attempt_001 的"已完整收口"验收（3 P1+5 P2）。全部修复后按修正协议重跑，权威改为 `attempt_002`（[attempt_001](../..//derived/audit/w1_numerics_n0_n1/n0/attempt_001/summary.json) 保留为被取代的 v1 证据，不再消费）。修订对照见方案 §14。

**v2 关键变化与 attempt_002 结果：**

- 格式 `n0_coordinate_flags_v2`（uint32）：新增 b16 `DELTA_PRESENCE_MISMATCH`、b17 `DELTA_ONE_SIDED_OBSERVABLE`。**attempt_002 标出 ONE_SIDED=4 个坐标**，恰为审阅核实的 4 行单侧可辨识（`deterministic u1 blocks.1.ffn.0.weight`、`deterministic u2 blocks.0.ffn.2.weight`、`diffusion u1 blocks.2.cross_attn.out.weight`、`diffusion u2 blocks.1.ffn.2.weight` 各 1 坐标）；PRESENCE=0。
- 消费保存的 `same_input_p/i` 公式门标量并新增 `delta_measured == theta_after − theta_before`（float64 精确）核对；**分侧重构写界**：`bound_i == half_ulp32(theta_i)`，`bound_p` 经 `delta_ref_p` 反解 ratio（精确复核 delta_ref）重构 P 侧公式参考 theta 后逐位核对——528 行全部通过。
- 键集按 `registration.roster.models[*].parameters` 精确集合核验；来源 SHA 自动对照 `i_report/p_report.models.*.arrays_sha256` 封口锚（实测一致）；旧 629 项按 `i_adamw_gap:*:*:*:kind` 键逐键对照，**424 行获得 old_state/new_only 附注**（attempt_001 因键格式不匹配全空）。
- t0 前移至模块导入前；summary 写盘后最终墙钟封口（`measured_wall_s_final=54.251s`，硬截止 900s 内）。
- 位分布（v2 重放，与 v1 一致部分不变）：UNOBSERVABLE P/I 1,214,768/1,214,770、V/M/DELTA/THETA 跨路径门失败 1499/1346/844/82、写擦除 203/201、m 零/符号 2、ONE_SIDED 4、同输入门失败/zero-invented/区间外/来源异常全部 0；分类重放 528/528、0 行 SOURCE_ANOMALY；`coverage_complete=true`。
- 标签分布：203 `precision_limited` + 42 `cross_path_sensitive` + 147 `cross_path_sensitive|precision_limited` + 136 `no_gap`。**结论不变：全部 392 行 gap 均为 precision_limited 或 cross_path_sensitive，无实现问题候选**；测试 13 项反例全过（含 roster 替换、封口锚漂移、写界篡改、same_input 篡改、zero-invention 篡改、attempt 独占等新反例）。
